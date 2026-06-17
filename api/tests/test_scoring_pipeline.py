"""Unit + integration tests for the full scoring pipeline.

Pipeline:
  POST /ingest
  → enqueue_flow_scoring  (task_queue.py)
  → _run_batch → tasks.score_flow_batch
  → DetectionEngine.predict_many
  → MetaEnsembleModel.score_many
  → _insert_predictions → Prediction rows
  → GET /flows / GET /anomalous_flows returns scores
"""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import joblib
import numpy as np
import pytest


# ── shared helpers ─────────────────────────────────────────────────────────

def _ns_flow(**kwargs):
    """Minimal SimpleNamespace that satisfies MetaFeatureBuilder._build_row."""
    base = dict(
        timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
        src_ip="192.168.1.5",
        dst_ip="8.8.8.8",
        proto="TCP",
        bytes=1000,
        extra={},
    )
    base.update(kwargs)
    return SimpleNamespace(**base)


class _FakeEstimator:
    """Minimal sklearn-compatible binary/multiclass classifier for test artifacts."""

    def __init__(self, n_classes: int = 6, high_class: int = 1, prob: float = 0.9):
        self.classes_ = np.arange(n_classes, dtype="int64")
        self.n_features_in_ = 46  # len(MetaFeatureBuilder.FEATURE_COLUMNS)
        self._high_class = high_class
        self._prob = prob

    def predict_proba(self, X):
        n = X.shape[0]
        probs = np.zeros((n, len(self.classes_)), dtype="float32")
        rest_prob = (1.0 - self._prob) / max(1, len(self.classes_) - 1)
        probs[:] = rest_prob
        probs[:, self._high_class] = self._prob
        return probs


class _SmallFakeEstimator:
    """Three-feature binary estimator — used to test _match_shape truncation path."""
    classes_ = np.array([0, 1], dtype="int64")
    n_features_in_ = 3

    def predict_proba(self, X):
        n = X.shape[0]
        return np.column_stack([np.full(n, 0.4), np.full(n, 0.6)])


class _BinaryEst08:
    """Binary estimator that predicts 0.8 on class 1."""
    classes_ = np.array([0, 1], dtype="int64")
    n_features_in_ = 46

    def predict_proba(self, X):
        n = X.shape[0]
        return np.column_stack([np.full(n, 0.2), np.full(n, 0.8)])


class _BinaryEst04:
    """Binary estimator that predicts 0.4 on class 1."""
    classes_ = np.array([0, 1], dtype="int64")
    n_features_in_ = 46

    def predict_proba(self, X):
        n = X.shape[0]
        return np.column_stack([np.full(n, 0.6), np.full(n, 0.4)])


class _BrokenEstimator:
    """Simulates an xgboost model that fails with AttributeError on both predict paths."""
    classes_ = np.array([0, 1], dtype="int64")
    n_features_in_ = 46

    def predict_proba(self, X):
        raise AttributeError("'XGBClassifier' object has no attribute 'use_label_encoder'")

    def predict(self, X):
        raise AttributeError("'XGBClassifier' object has no attribute 'use_label_encoder'")


@pytest.fixture()
def fake_model_path(tmp_path):
    """Joblib artifact {"extra_trees": _FakeEstimator()} at a temp path."""
    path = tmp_path / "meta_model_test.joblib"
    joblib.dump({"extra_trees": _FakeEstimator()}, path)
    return path


# ═══════════════════════════════════════════════════════════════════════════
# 1. MetaFeatureBuilder — edge cases beyond test_scoring_and_features.py
# ═══════════════════════════════════════════════════════════════════════════

class TestMetaFeatureBuilderEdgeCases:
    def test_ip_to_int_invalid_string_returns_non_negative(self):
        from model_runner import MetaFeatureBuilder
        result = MetaFeatureBuilder()._ip_to_int("not-an-ip")
        assert isinstance(result, int) and result >= 0

    def test_safe_int_hex_string(self):
        from model_runner import MetaFeatureBuilder
        assert MetaFeatureBuilder._safe_int("0x1A") == 26

    def test_safe_int_none_returns_none(self):
        from model_runner import MetaFeatureBuilder
        assert MetaFeatureBuilder._safe_int(None) is None

    def test_safe_float_string_number(self):
        from model_runner import MetaFeatureBuilder
        assert MetaFeatureBuilder._safe_float("3.14") == pytest.approx(3.14)

    def test_safe_float_none_returns_none(self):
        from model_runner import MetaFeatureBuilder
        assert MetaFeatureBuilder._safe_float(None) is None

    def test_direction_tag_outbound(self):
        from model_runner import MetaFeatureBuilder
        assert MetaFeatureBuilder()._direction_tag("192.168.1.1", "8.8.8.8") == "outbound"

    def test_direction_tag_inbound(self):
        from model_runner import MetaFeatureBuilder
        assert MetaFeatureBuilder()._direction_tag("8.8.8.8", "10.0.0.1") == "inbound"

    def test_direction_tag_internal(self):
        from model_runner import MetaFeatureBuilder
        assert MetaFeatureBuilder()._direction_tag("192.168.1.1", "10.0.0.1") == "internal"

    def test_estimate_duration_zero_bytes_clamps_to_min(self):
        from model_runner import MetaFeatureBuilder
        assert MetaFeatureBuilder()._estimate_duration(0) == 0.01

    def test_build_uses_explicit_src_dst_bytes_from_extra(self):
        from model_runner import MetaFeatureBuilder
        row = MetaFeatureBuilder().build(_ns_flow(extra={"src_bytes": 100.0, "dst_bytes": 200.0})).iloc[0]
        assert row["src_bytes"] == 100.0
        assert row["dst_bytes"] == 200.0

    def test_build_batch_multiple_rows_proto(self):
        from model_runner import MetaFeatureBuilder
        frame = MetaFeatureBuilder().build_batch([_ns_flow(), _ns_flow(proto="UDP")])
        assert frame.shape[0] == 2
        assert frame.iloc[1]["proto"] == 17.0


# ═══════════════════════════════════════════════════════════════════════════
# 2. MetaEnsembleModel._match_shape
# ═══════════════════════════════════════════════════════════════════════════

class TestMatchShape:
    def test_pads_with_zeros_when_too_few_features(self):
        from model_runner import MetaEnsembleModel
        arr = np.ones((3, 4), dtype="float32")
        padded = MetaEnsembleModel._match_shape(arr, 6)
        assert padded.shape == (3, 6)
        assert padded[:, 4:].sum() == 0.0

    def test_truncates_when_too_many_features(self):
        from model_runner import MetaEnsembleModel
        arr = np.ones((3, 10), dtype="float32")
        trimmed = MetaEnsembleModel._match_shape(arr, 6)
        assert trimmed.shape == (3, 6)

    def test_passthrough_when_shape_matches(self):
        from model_runner import MetaEnsembleModel
        arr = np.eye(4, dtype="float32")
        result = MetaEnsembleModel._match_shape(arr, 4)
        assert result is arr


# ═══════════════════════════════════════════════════════════════════════════
# 3. MetaEnsembleModel — loading and scoring
# ═══════════════════════════════════════════════════════════════════════════

class TestMetaEnsembleModel:
    def test_raises_file_not_found_for_missing_artifact(self):
        from model_runner import MetaEnsembleModel
        with pytest.raises(FileNotFoundError):
            MetaEnsembleModel("/does/not/exist.joblib")

    def test_raises_value_error_for_non_dict_artifact(self, tmp_path):
        from model_runner import MetaEnsembleModel
        bad = tmp_path / "bad.joblib"
        joblib.dump(["not", "a", "dict"], bad)
        with pytest.raises(ValueError, match="Unexpected artifact format"):
            MetaEnsembleModel(bad)

    def test_raises_value_error_for_empty_dict_artifact(self, tmp_path):
        from model_runner import MetaEnsembleModel
        bad = tmp_path / "empty.joblib"
        joblib.dump({}, bad)
        with pytest.raises(ValueError, match="did not contain a supported estimator"):
            MetaEnsembleModel(bad)

    def test_score_many_empty_returns_empty_list(self, fake_model_path):
        from model_runner import MetaEnsembleModel
        model = MetaEnsembleModel(fake_model_path)
        assert model.score_many([]) == []

    def test_score_many_returns_correct_length(self, fake_model_path):
        from model_runner import MetaEnsembleModel
        model = MetaEnsembleModel(fake_model_path)
        results = model.score_many([_ns_flow(), _ns_flow()])
        assert len(results) == 2

    def test_score_many_all_scores_in_unit_interval(self, fake_model_path):
        from model_runner import MetaEnsembleModel
        model = MetaEnsembleModel(fake_model_path)
        for score, label, *_ in model.score_many([_ns_flow()] * 5):
            assert 0.0 <= score <= 1.0
            assert isinstance(label, str)

    def test_score_many_high_prob_maps_to_attack_label(self, fake_model_path):
        from model_runner import MetaEnsembleModel
        # _FakeEstimator: 0.9 on class 1 ("attack")
        model = MetaEnsembleModel(fake_model_path)
        score, label, *_ = model.score_many([_ns_flow()])[0]
        assert score == pytest.approx(0.9, abs=0.01)
        assert label == "attack"

    def test_label_for_probability_above_anomaly_threshold(self, fake_model_path):
        from model_runner import MetaEnsembleModel
        model = MetaEnsembleModel(fake_model_path)
        model.anomaly_threshold = 0.5
        assert model._label_for_probability(0.6) == "anomaly"

    def test_label_for_probability_in_watch_range(self, fake_model_path):
        from model_runner import MetaEnsembleModel
        model = MetaEnsembleModel(fake_model_path)
        model.watch_threshold = 0.3
        model.anomaly_threshold = 0.7
        assert model._label_for_probability(0.5) == "watch"

    def test_label_for_probability_below_watch_is_normal(self, fake_model_path):
        from model_runner import MetaEnsembleModel
        model = MetaEnsembleModel(fake_model_path)
        assert model._label_for_probability(0.1) == "normal"

    def test_score_single_flow_delegates_to_score_many(self, fake_model_path):
        from model_runner import MetaEnsembleModel
        model = MetaEnsembleModel(fake_model_path)
        result = model.score(_ns_flow())
        # Returns (score, label) or (score, label, attack_label)
        assert len(result) >= 2
        assert 0.0 <= result[0] <= 1.0

    def test_feature_count_mismatch_is_handled_gracefully(self, tmp_path):
        from model_runner import MetaEnsembleModel

        path = tmp_path / "small.joblib"
        joblib.dump({"extra_trees": _SmallFakeEstimator()}, path)
        model = MetaEnsembleModel(path)
        results = model.score_many([_ns_flow()])
        assert len(results) == 1

    def test_broken_estimator_is_skipped_and_working_one_scores(self, tmp_path):
        from model_runner import MetaEnsembleModel
        path = tmp_path / "broken_xgb.joblib"
        joblib.dump({"xgboost": _BrokenEstimator(), "extra_trees": _FakeEstimator()}, path)
        model = MetaEnsembleModel(path)
        # xgboost fails both predict_proba and predict → skipped; extra_trees runs alone
        results = model.score_many([_ns_flow()])
        assert len(results) == 1
        score, *_ = results[0]
        assert 0.0 < score <= 1.0

    def test_all_estimators_broken_raises_runtime_error(self, tmp_path):
        from model_runner import MetaEnsembleModel
        path = tmp_path / "all_broken.joblib"
        joblib.dump({"extra_trees": _BrokenEstimator()}, path)
        model = MetaEnsembleModel(path)
        with pytest.raises(RuntimeError, match="no usable estimators"):
            model.score_many([_ns_flow()])

    def test_dual_model_bundle_averages_probabilities(self, tmp_path):
        from model_runner import MetaEnsembleModel

        # xgb returns 0.8 on class 1; extra_trees returns 0.4 on class 1 → average 0.6
        path = tmp_path / "dual.joblib"
        joblib.dump({"xgboost": _BinaryEst08(), "extra_trees": _BinaryEst04()}, path)
        model = MetaEnsembleModel(path)
        score, *_ = model.score_many([_ns_flow()])[0]
        # Top class 1 avg probability ≈ 0.6
        assert score == pytest.approx(0.6, abs=0.01)


# ═══════════════════════════════════════════════════════════════════════════
# 4. DetectionEngine
# ═══════════════════════════════════════════════════════════════════════════

class TestDetectionEngine:
    def test_no_model_predict_many_returns_neutral_tuples(self):
        from model_runner import DetectionEngine
        engine = DetectionEngine()
        engine.model = None
        results = engine.predict_many(None, [_ns_flow(), _ns_flow()])
        assert results == [(0.0, "normal"), (0.0, "normal")]

    def test_no_model_predict_returns_neutral(self):
        from model_runner import DetectionEngine
        engine = DetectionEngine()
        engine.model = None
        assert engine.predict(None, _ns_flow()) == (0.0, "normal")

    def test_predict_many_empty_input_returns_empty(self):
        from model_runner import DetectionEngine
        engine = DetectionEngine()
        assert engine.predict_many(None, []) == []

    def test_predict_many_uses_loaded_model(self, fake_model_path):
        from model_runner import DetectionEngine, MetaEnsembleModel
        engine = DetectionEngine()
        engine.model = MetaEnsembleModel(fake_model_path)
        results = engine.predict_many(None, [_ns_flow()])
        assert len(results) == 1
        score, *_ = results[0]
        assert 0.0 <= score <= 1.0

    def test_reload_if_stale_triggers_on_mtime_change(self, fake_model_path, monkeypatch):
        from model_runner import DetectionEngine
        monkeypatch.setenv("ADNS_META_MODEL_PATH", str(fake_model_path))
        engine = DetectionEngine()
        assert engine.model is not None
        # Fake stale mtime so reload is triggered
        engine._artifact_mtime = engine._artifact_mtime - 10.0
        engine.reload_if_stale()
        # After reload the mtime is refreshed to current
        assert engine._artifact_mtime > 0.0

    def test_reload_if_stale_skips_when_file_missing(self, monkeypatch):
        from model_runner import DetectionEngine
        monkeypatch.setenv("ADNS_META_MODEL_PATH", "/no/such/path.joblib")
        engine = DetectionEngine()
        engine._artifact_mtime = 999.0
        # Should not raise; mtime left unchanged because file not found
        engine.reload_if_stale()
        assert engine._artifact_mtime == 999.0

    def test_load_model_gracefully_falls_back_to_none_on_missing_file(self, monkeypatch):
        from model_runner import DetectionEngine
        monkeypatch.setenv("ADNS_META_MODEL_PATH", "/nonexistent/model.joblib")
        engine = DetectionEngine()
        assert engine.model is None


# ═══════════════════════════════════════════════════════════════════════════
# 5. tasks._insert_predictions
# ═══════════════════════════════════════════════════════════════════════════

class TestInsertPredictions:
    def test_empty_records_returns_zero(self, flask_app):
        with flask_app.app_context():
            from tasks import _insert_predictions
            assert _insert_predictions([]) == 0

    def test_inserts_single_prediction_row(self, flask_app):
        with flask_app.app_context():
            import app as app_module
            from tasks import _insert_predictions

            flow = app_module.Flow(
                timestamp=datetime.now(timezone.utc),
                src_ip="10.0.0.1", dst_ip="1.1.1.1",
                proto="TCP", bytes=500,
            )
            app_module.db.session.add(flow)
            app_module.db.session.commit()

            inserted = _insert_predictions([{
                "flow_id": flow.id,
                "score": 0.85,
                "label": "anomaly",
                "created_at": datetime.now(timezone.utc),
            }])
            app_module.db.session.commit()

            assert inserted == 1
            pred = app_module.Prediction.query.filter_by(flow_id=flow.id).first()
            assert pred is not None
            assert pred.score == pytest.approx(0.85, abs=1e-4)
            assert pred.label == "anomaly"

    def test_duplicate_insert_is_idempotent(self, flask_app):
        with flask_app.app_context():
            import app as app_module
            from tasks import _insert_predictions

            flow = app_module.Flow(
                timestamp=datetime.now(timezone.utc),
                src_ip="10.0.0.2", dst_ip="2.2.2.2",
                proto="UDP", bytes=100,
            )
            app_module.db.session.add(flow)
            app_module.db.session.commit()

            record = {
                "flow_id": flow.id,
                "score": 0.5,
                "label": "watch",
                "created_at": datetime.now(timezone.utc),
            }
            _insert_predictions([record])
            app_module.db.session.commit()
            # Second insert of same flow_id must not raise or duplicate
            _insert_predictions([record])
            app_module.db.session.commit()

            assert app_module.Prediction.query.filter_by(flow_id=flow.id).count() == 1


# ═══════════════════════════════════════════════════════════════════════════
# 6. tasks.score_flow_batch
# ═══════════════════════════════════════════════════════════════════════════

class TestScoreFlowBatch:
    def test_empty_list_returns_zero(self):
        from tasks import score_flow_batch
        assert score_flow_batch([]) == 0

    def test_writes_prediction_with_mocked_detector(self, flask_app, monkeypatch):
        import app as app_module
        import tasks

        with flask_app.app_context():
            flow = app_module.Flow(
                timestamp=datetime.now(timezone.utc),
                src_ip="192.168.0.5", dst_ip="8.8.4.4",
                proto="TCP", bytes=2048,
            )
            app_module.db.session.add(flow)
            app_module.db.session.commit()
            flow_id = flow.id

        monkeypatch.setattr(tasks, "RDNS_ENABLED", False)
        monkeypatch.setattr(
            tasks.detector, "predict_many",
            lambda _sess, flows: [(0.77, "anomaly", "ddos")] * len(flows),
        )
        monkeypatch.setattr(tasks.detector, "reload_if_stale", lambda: None)

        result = tasks.score_flow_batch([flow_id])
        assert result == 1

        with flask_app.app_context():
            pred = app_module.Prediction.query.filter_by(flow_id=flow_id).first()
            assert pred is not None
            assert pred.score == pytest.approx(0.77, abs=1e-4)
            assert pred.label == "anomaly"

    def test_nonexistent_flow_ids_produce_zero_scored(self, flask_app, monkeypatch):
        import tasks

        monkeypatch.setattr(tasks, "RDNS_ENABLED", False)
        monkeypatch.setattr(
            tasks.detector, "predict_many",
            lambda _sess, flows: [(0.5, "normal", None)] * len(flows),
        )
        monkeypatch.setattr(tasks.detector, "reload_if_stale", lambda: None)

        assert tasks.score_flow_batch([999_999, 888_888]) == 0

    def test_attack_type_written_to_flow_extra(self, flask_app, monkeypatch):
        import app as app_module
        import tasks

        with flask_app.app_context():
            flow = app_module.Flow(
                timestamp=datetime.now(timezone.utc),
                src_ip="192.168.0.10", dst_ip="77.88.8.8",
                proto="UDP", bytes=512, extra={},
            )
            app_module.db.session.add(flow)
            app_module.db.session.commit()
            flow_id = flow.id

        monkeypatch.setattr(tasks, "RDNS_ENABLED", False)
        monkeypatch.setattr(
            tasks.detector, "predict_many",
            lambda _sess, flows: [(0.92, "ddos", "ddos")] * len(flows),
        )
        monkeypatch.setattr(tasks.detector, "reload_if_stale", lambda: None)

        tasks.score_flow_batch([flow_id])

        with flask_app.app_context():
            refreshed = app_module.db.session.get(app_module.Flow, flow_id)
            assert (refreshed.extra or {}).get("attack_type") == "ddos"

    def test_normal_prediction_removes_attack_type_from_extra(self, flask_app, monkeypatch):
        import app as app_module
        import tasks

        with flask_app.app_context():
            flow = app_module.Flow(
                timestamp=datetime.now(timezone.utc),
                src_ip="172.16.0.1", dst_ip="4.4.4.4",
                proto="TCP", bytes=100, extra={"attack_type": "old"},
            )
            app_module.db.session.add(flow)
            app_module.db.session.commit()
            flow_id = flow.id

        monkeypatch.setattr(tasks, "RDNS_ENABLED", False)
        monkeypatch.setattr(
            tasks.detector, "predict_many",
            lambda _sess, flows: [(0.05, "normal", None)] * len(flows),
        )
        monkeypatch.setattr(tasks.detector, "reload_if_stale", lambda: None)

        tasks.score_flow_batch([flow_id])

        with flask_app.app_context():
            refreshed = app_module.db.session.get(app_module.Flow, flow_id)
            assert "attack_type" not in (refreshed.extra or {})

    def test_repeated_scoring_does_not_duplicate_prediction(self, flask_app, monkeypatch):
        import app as app_module
        import tasks

        with flask_app.app_context():
            flow = app_module.Flow(
                timestamp=datetime.now(timezone.utc),
                src_ip="10.10.0.1", dst_ip="1.2.3.4",
                proto="TCP", bytes=300,
            )
            app_module.db.session.add(flow)
            app_module.db.session.commit()
            flow_id = flow.id

        monkeypatch.setattr(tasks, "RDNS_ENABLED", False)
        monkeypatch.setattr(
            tasks.detector, "predict_many",
            lambda _sess, flows: [(0.6, "watch", None)] * len(flows),
        )
        monkeypatch.setattr(tasks.detector, "reload_if_stale", lambda: None)

        tasks.score_flow_batch([flow_id])
        tasks.score_flow_batch([flow_id])  # second call should not create duplicate

        with flask_app.app_context():
            count = app_module.Prediction.query.filter_by(flow_id=flow_id).count()
            assert count == 1


# ═══════════════════════════════════════════════════════════════════════════
# 7. task_queue.enqueue_flow_scoring
# ═══════════════════════════════════════════════════════════════════════════

class TestEnqueueFlowScoring:
    def test_empty_list_returns_zero(self, monkeypatch):
        import task_queue
        fake_executor = MagicMock()
        monkeypatch.setattr(task_queue, "_executor", fake_executor)
        assert task_queue.enqueue_flow_scoring([]) == 0
        fake_executor.submit.assert_not_called()

    def test_returns_count_of_valid_ids(self, monkeypatch):
        import task_queue
        submitted = []
        fake_executor = MagicMock()
        fake_executor.submit.side_effect = lambda fn, chunk: submitted.append(chunk)
        monkeypatch.setattr(task_queue, "_executor", fake_executor)

        count = task_queue.enqueue_flow_scoring([1, 2, 3])
        assert count == 3
        assert sum(len(c) for c in submitted) == 3

    def test_chunking_creates_correct_number_of_batches(self, monkeypatch):
        import task_queue
        submitted = []
        fake_executor = MagicMock()
        fake_executor.submit.side_effect = lambda fn, chunk: submitted.append(chunk)
        monkeypatch.setattr(task_queue, "_executor", fake_executor)
        monkeypatch.setenv("ADNS_SCORING_BATCH_SIZE", "2")

        task_queue.enqueue_flow_scoring([1, 2, 3, 4, 5])
        # 5 IDs / batch_size 2 → 3 chunks
        assert len(submitted) == 3
        all_ids = sorted(id_ for chunk in submitted for id_ in chunk)
        assert all_ids == [1, 2, 3, 4, 5]

    def test_falsy_ids_are_filtered_out(self, monkeypatch):
        import task_queue
        submitted = []
        fake_executor = MagicMock()
        fake_executor.submit.side_effect = lambda fn, chunk: submitted.append(chunk)
        monkeypatch.setattr(task_queue, "_executor", fake_executor)

        # 0 and None are falsy — only [1, 2] should be submitted
        count = task_queue.enqueue_flow_scoring([1, 0, None, 2])
        assert count == 2
        all_ids = sorted(id_ for chunk in submitted for id_ in chunk)
        assert all_ids == [1, 2]


# ═══════════════════════════════════════════════════════════════════════════
# 8. API scoring integration — /flows and /anomalous_flows
# ═══════════════════════════════════════════════════════════════════════════

class TestApiScoringIntegration:
    def _insert_flow(self, flask_app, src_ip="10.55.0.1"):
        """Insert a flow directly into the DB with a current timestamp so it passes the time filter."""
        import app as app_module
        with flask_app.app_context():
            flow = app_module.Flow(
                timestamp=datetime.now(timezone.utc),
                src_ip=src_ip,
                dst_ip="8.8.8.8",
                proto="TCP",
                bytes=1234,
            )
            app_module.db.session.add(flow)
            app_module.db.session.commit()
            return flow.id

    def test_flows_score_field_updated_after_direct_scoring(self, client, flask_app, monkeypatch):
        import tasks

        flow_id = self._insert_flow(flask_app, src_ip="10.55.0.1")

        monkeypatch.setattr(tasks, "RDNS_ENABLED", False)
        monkeypatch.setattr(
            tasks.detector, "predict_many",
            lambda _sess, flows: [(0.88, "anomaly", None)] * len(flows),
        )
        monkeypatch.setattr(tasks.detector, "reload_if_stale", lambda: None)
        tasks.score_flow_batch([flow_id])

        flows = client.get("/flows").get_json()
        match = next((f for f in flows if f.get("id") == flow_id), None)
        assert match is not None, "Ingested flow not returned by /flows"
        assert match["score"] == pytest.approx(0.88, abs=1e-3)
        assert match["label"] == "anomaly"

    def test_anomalous_flows_excludes_normal_scored_flows(self, client, flask_app, monkeypatch):
        import tasks

        fid_bad = self._insert_flow(flask_app, src_ip="10.55.1.1")
        fid_ok = self._insert_flow(flask_app, src_ip="10.55.1.2")

        monkeypatch.setattr(tasks, "RDNS_ENABLED", False)
        monkeypatch.setattr(tasks.detector, "reload_if_stale", lambda: None)

        def fake_predict(_sess, flows):
            return [
                (0.95, "ddos", "ddos") if f.id == fid_bad else (0.05, "normal", None)
                for f in flows
            ]

        monkeypatch.setattr(tasks.detector, "predict_many", fake_predict)
        tasks.score_flow_batch([fid_bad, fid_ok])

        anomalous = client.get("/anomalous_flows").get_json()
        ids_returned = [f["id"] for f in anomalous]
        assert fid_bad in ids_returned
        assert fid_ok not in ids_returned

    def test_anomalous_flows_empty_when_no_flows_in_db(self, client):
        result = client.get("/anomalous_flows").get_json()
        assert result == []

    def test_flows_returns_zero_score_for_unscored_flow(self, client, flask_app):
        flow_id = self._insert_flow(flask_app, src_ip="10.55.2.1")
        flows = client.get("/flows").get_json()
        match = next((f for f in flows if f.get("id") == flow_id), None)
        assert match is not None, "Unscored flow not returned by /flows"
        # No Prediction row written, so score falls back to 0.0
        assert match["score"] == pytest.approx(0.0, abs=1e-6)
