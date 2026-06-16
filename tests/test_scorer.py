"""Unit tests for the heuristic scorer and MetaFeatureBuilder."""

from datetime import datetime, timezone
from unittest.mock import MagicMock
import pytest


def _flow(src="10.0.0.1", dst="8.8.8.8", proto="TCP", bytes_=1000, extra=None):
    f = MagicMock()
    f.src_ip = src
    f.dst_ip = dst
    f.proto = proto
    f.bytes = bytes_
    f.timestamp = datetime.now(timezone.utc)
    f.extra = extra or {}
    return f


class TestFlowScorer:
    @pytest.fixture()
    def scorer(self):
        from scoring import FlowScorer
        return FlowScorer()

    def test_bytes_score_large_bytes_approaches_one(self, scorer):
        score = scorer._bytes_score(100_000)
        assert score >= 0.9

    def test_bytes_score_zero_is_zero(self, scorer):
        assert scorer._bytes_score(0) == 0.0

    def test_bytes_score_negative_is_zero(self, scorer):
        assert scorer._bytes_score(-100) == 0.0

    def test_proto_bonus_icmp(self, scorer):
        assert scorer._proto_bonus("ICMP") == pytest.approx(0.08)

    def test_proto_bonus_tcp_is_zero(self, scorer):
        assert scorer._proto_bonus("TCP") == 0.0

    def test_direction_bonus_outbound_private_to_public(self, scorer):
        f = _flow(src="192.168.1.1", dst="8.8.8.8")
        assert scorer._direction_bonus(f) == pytest.approx(0.07)

    def test_direction_bonus_both_private_is_zero(self, scorer):
        f = _flow(src="192.168.1.1", dst="10.0.0.1")
        assert scorer._direction_bonus(f) == 0.0

    def test_predict_score_clamped_between_0_and_1(self, scorer, app):
        from app import db, Flow
        with app.app_context():
            flow = Flow(
                timestamp=datetime.now(timezone.utc),
                src_ip="192.168.1.1",
                dst_ip="8.8.8.8",
                proto="ICMP",
                bytes=500_000,
            )
            db.session.add(flow)
            db.session.commit()
            score, label = scorer.predict(db.session, flow)
        assert 0.0 <= score <= 1.0
        assert label in {"normal", "watch", "anomaly"}

    def test_predict_label_normal_for_tiny_flow(self, scorer, app):
        from app import db, Flow
        with app.app_context():
            flow = Flow(
                timestamp=datetime.now(timezone.utc),
                src_ip="192.168.0.2",
                dst_ip="192.168.0.3",
                proto="TCP",
                bytes=100,
            )
            db.session.add(flow)
            db.session.commit()
            score, label = scorer.predict(db.session, flow)
        assert label in {"normal", "watch"}

    def test_stable_jitter_deterministic(self, scorer, app):
        from app import db, Flow
        with app.app_context():
            flow = Flow(
                timestamp=datetime.now(timezone.utc),
                src_ip="10.0.0.1",
                dst_ip="10.0.0.2",
                proto="UDP",
                bytes=1000,
            )
            db.session.add(flow)
            db.session.commit()
            j1 = scorer._stable_jitter(flow)
            j2 = scorer._stable_jitter(flow)
        assert j1 == j2
        assert 0.0 <= j1 <= 0.05


class TestMetaFeatureBuilder:
    @pytest.fixture()
    def builder(self):
        from model_runner import MetaFeatureBuilder
        return MetaFeatureBuilder()

    def test_ip_to_int_valid_ipv4(self, builder):
        val = builder._ip_to_int("1.2.3.4")
        assert val == (1 * 256**3 + 2 * 256**2 + 3 * 256 + 4)

    def test_ip_to_int_empty_returns_zero(self, builder):
        assert builder._ip_to_int("") == 0

    def test_direction_tag_outbound(self, builder):
        assert builder._direction_tag("192.168.1.1", "8.8.8.8") == "outbound"

    def test_direction_tag_inbound(self, builder):
        assert builder._direction_tag("8.8.8.8", "192.168.1.1") == "inbound"

    def test_direction_tag_internal(self, builder):
        assert builder._direction_tag("10.0.0.1", "10.0.0.2") == "internal"

    def test_estimate_packets_scales_with_bytes(self, builder):
        pkts = builder._estimate_packets(9000)
        assert pkts >= 1.0

    def test_estimate_packets_zero_bytes_is_zero(self, builder):
        assert builder._estimate_packets(0) == 0.0

    def test_safe_float_valid(self, builder):
        assert builder._safe_float("3.14") == pytest.approx(3.14)

    def test_safe_float_none_returns_none(self, builder):
        assert builder._safe_float(None) is None

    def test_safe_int_hex_string(self, builder):
        assert builder._safe_int("0x1bb") == 443
