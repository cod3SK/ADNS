"""Tests for absent-model loud-failure behaviour.

Verifies that a missing nfstream_model.joblib is never a silent failure:
- NfstreamDetectionEngine logs at ERROR level (not INFO/WARNING)
- is_model_loaded is False and model_error carries the reason
- /capture/autostart returns 503 with a clear message (not 200 + silent no-scoring)
- /model_status reflects the hard-error state

Unit tests create fresh NfstreamDetectionEngine instances with a nonexistent path.
Integration tests monkeypatch the module-level scorer so the real on-disk model
doesn't mask the absent-model path.
"""
from __future__ import annotations

import logging

import pytest

import app as app_module
from model_runner import NfstreamDetectionEngine


# ── Fixture ───────────────────────────────────────────────────────────────────

@pytest.fixture
def no_model_scorer(tmp_path, monkeypatch):
    """Replace app._simulation_scorer with an engine that has no model loaded."""
    engine = NfstreamDetectionEngine(model_path=tmp_path / "no_model.joblib")
    monkeypatch.setattr(app_module, "_simulation_scorer", engine)
    return engine


# ── Unit: NfstreamDetectionEngine absent-model path ──────────────────────────

def test_absent_model_is_not_loaded(tmp_path):
    """is_model_loaded must be False when joblib is missing."""
    engine = NfstreamDetectionEngine(model_path=tmp_path / "no_model.joblib")
    assert not engine.is_model_loaded


def test_absent_model_error_is_not_none(tmp_path):
    """model_error must be a non-empty string describing what is missing."""
    engine = NfstreamDetectionEngine(model_path=tmp_path / "no_model.joblib")
    assert engine.model_error is not None
    assert len(engine.model_error) > 0


def test_absent_model_logs_at_error_level(tmp_path, caplog):
    """Missing model must be logged at ERROR level, not INFO or WARNING."""
    with caplog.at_level(logging.DEBUG, logger="model_runner"):
        NfstreamDetectionEngine(model_path=tmp_path / "no_model.joblib")

    error_records = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert error_records, (
        "Expected at least one ERROR log from model_runner when model is absent; "
        f"got levels: {[r.levelno for r in caplog.records]}"
    )


def test_absent_model_error_message_mentions_path(tmp_path, caplog):
    """The error log must name the missing file so the operator knows what to fix."""
    absent = tmp_path / "no_model.joblib"
    with caplog.at_level(logging.ERROR, logger="model_runner"):
        NfstreamDetectionEngine(model_path=absent)

    combined = " ".join(r.getMessage() for r in caplog.records)
    assert str(absent) in combined or "no_model.joblib" in combined, (
        f"Expected missing path in error message, got: {combined!r}"
    )


# ── Integration: /capture/autostart blocks when model absent ─────────────────

def test_capture_autostart_returns_503_without_model(client, no_model_scorer):
    """/capture/autostart must return 503 when NFStream model is not loaded.

    Live capture producing zero detections is worse than refusing to start.
    """
    resp = client.post("/capture/autostart")
    assert resp.status_code == 503, (
        f"Expected 503 when model absent, got {resp.status_code}: {resp.get_json()}"
    )


def test_capture_autostart_error_body_mentions_model(client, no_model_scorer):
    """The 503 response must name the model so the user knows the fix."""
    resp = client.post("/capture/autostart")
    error = (resp.get_json() or {}).get("error", "")
    assert "model" in error.lower(), f"503 body should mention 'model'; got: {error!r}"


def test_capture_autostart_error_body_mentions_train_script(client, no_model_scorer):
    """The 503 response must hint at the fix (train_nfstream.py)."""
    resp = client.post("/capture/autostart")
    error = (resp.get_json() or {}).get("error", "")
    assert "train_nfstream" in error, (
        f"503 body should mention train_nfstream.py; got: {error!r}"
    )


# ── Integration: /model_status reflects absent model ─────────────────────────

def test_model_status_absent_returns_200(client, no_model_scorer):
    """/model_status must respond (not crash) when model is absent."""
    assert client.get("/model_status").status_code == 200


def test_model_status_absent_reports_absent(client, no_model_scorer):
    """/model_status must report meta_model_status='absent' when model not loaded."""
    body = client.get("/model_status").get_json()
    assert body.get("meta_model_status") == "absent", (
        f"Expected 'absent', got: {body.get('meta_model_status')!r}"
    )


def test_model_status_absent_has_zero_active_estimators(client, no_model_scorer):
    body = client.get("/model_status").get_json()
    assert body.get("active_estimators") == 0
