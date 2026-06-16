"""Unit tests for database models and flow operations."""

from datetime import datetime, timedelta, timezone
import pytest


@pytest.fixture()
def ctx(app):
    with app.app_context():
        yield


def _now():
    return datetime.now(timezone.utc)


def _flow(src="10.0.0.1", dst="8.8.8.8", proto="TCP", bytes_=500, minutes_ago=0):
    from app import Flow
    return Flow(
        timestamp=_now() - timedelta(minutes=minutes_ago),
        src_ip=src,
        dst_ip=dst,
        proto=proto,
        bytes=bytes_,
    )


class TestFlowModel:
    def test_flow_create_and_retrieve(self, app, ctx):
        from app import db, Flow
        with app.app_context():
            f = _flow()
            db.session.add(f)
            db.session.commit()
            assert Flow.query.count() == 1
            assert Flow.query.first().src_ip == "10.0.0.1"

    def test_flow_defaults(self, app, ctx):
        from app import db, Flow
        with app.app_context():
            f = _flow(bytes_=0)
            db.session.add(f)
            db.session.commit()
            assert f.bytes == 0
            assert f.extra is None

    def test_flow_extra_json(self, app, ctx):
        from app import db, Flow
        with app.app_context():
            f = _flow()
            f.extra = {"service": "https", "dst_port": 443}
            db.session.add(f)
            db.session.commit()
            loaded = Flow.query.get(f.id)
            assert loaded.extra["service"] == "https"


class TestPredictionModel:
    def test_prediction_linked_to_flow(self, app, ctx):
        from app import db, Flow, Prediction
        with app.app_context():
            f = _flow()
            db.session.add(f)
            db.session.flush()
            p = Prediction(flow_id=f.id, score=0.95, label="anomaly")
            db.session.add(p)
            db.session.commit()
            assert p.flow_id == f.id
            assert f.predictions.count() == 1

    def test_prediction_cascade_delete(self, app, ctx):
        from app import db, Flow, Prediction
        with app.app_context():
            f = _flow()
            db.session.add(f)
            db.session.flush()
            db.session.add(Prediction(flow_id=f.id, score=0.5, label="normal"))
            db.session.commit()
            db.session.delete(f)
            db.session.commit()
            assert Prediction.query.count() == 0


class TestBlockedIPModel:
    def test_blocked_ip_created(self, app, ctx):
        from app import db, BlockedIP
        with app.app_context():
            b = BlockedIP(ip="1.2.3.4", active=True, created_at=_now())
            db.session.add(b)
            db.session.commit()
            assert BlockedIP.query.filter_by(ip="1.2.3.4").count() == 1


class TestFlowHelpers:
    def test_flow_to_dict_structure(self, app, ctx):
        from app import db, flow_to_dict
        with app.app_context():
            f = _flow()
            db.session.add(f)
            db.session.commit()
            d = flow_to_dict(f)
            assert {"id", "ts", "src_ip", "dst_ip", "proto", "bytes", "score", "label"} <= d.keys()
            assert d["src_ip"] == "10.0.0.1"
            assert d["label"] == "normal"

    def test_get_recent_flows_ordering(self, app, ctx):
        from app import db, get_recent_flows
        with app.app_context():
            for m in [10, 5, 1]:
                db.session.add(_flow(minutes_ago=m))
            db.session.commit()
            flows = get_recent_flows()
            timestamps = [f.timestamp for f in flows]
            assert timestamps == sorted(timestamps)

    def test_is_anomalous_flow_with_high_score(self, app, ctx):
        from app import db, Prediction, is_anomalous_flow
        with app.app_context():
            f = _flow()
            db.session.add(f)
            db.session.flush()
            db.session.add(Prediction(flow_id=f.id, score=0.95, label="anomaly"))
            db.session.commit()
            assert is_anomalous_flow(f) is True

    def test_is_anomalous_flow_no_prediction(self, app, ctx):
        from app import db, is_anomalous_flow
        with app.app_context():
            f = _flow()
            db.session.add(f)
            db.session.commit()
            assert is_anomalous_flow(f) is False

    def test_enforce_retention_purges_old(self, app, ctx):
        from app import db, enforce_flow_retention
        with app.app_context():
            # Old flow (35 min ago — beyond 30-min retention)
            db.session.add(_flow(minutes_ago=35))
            # Recent flow
            db.session.add(_flow(minutes_ago=1))
            db.session.commit()
            purged = enforce_flow_retention()
            assert purged == 1
