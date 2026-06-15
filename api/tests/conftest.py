"""Shared pytest fixtures for the ADNS API.

The suite runs the Flask app against a throwaway SQLite database in heuristic
scoring mode (no ML artifacts, no Redis, no reverse-DNS), so it is fast and has
no external dependencies. Environment is configured *before* importing ``app``
because ``app.py`` reads config and calls ``init_db()`` at import time.
"""

import os
import sys
import tempfile
from pathlib import Path

import pytest

API_DIR = Path(__file__).resolve().parent.parent
if str(API_DIR) not in sys.path:
    sys.path.insert(0, str(API_DIR))

# A temp SQLite file shared across the test session.
_DB_FD, _DB_PATH = tempfile.mkstemp(suffix=".db", prefix="adns_test_")
os.close(_DB_FD)

os.environ["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{_DB_PATH}"
# Point model artifacts at paths that do not exist so the DetectionEngine falls
# back to the dependency-light heuristic FlowScorer (no xgboost/sklearn needed).
os.environ["ADNS_META_MODEL_PATH"] = str(API_DIR / "does_not_exist_meta.joblib")
os.environ["ADNS_MODEL_PATH"] = str(API_DIR / "does_not_exist_flow.joblib")
os.environ["ADNS_RDNS_ENABLED"] = "false"
os.environ["ADNS_NSENTER_HOST"] = "false"
os.environ.pop("ADNS_ADMIN_TOKEN", None)

import app as app_module  # noqa: E402  (import after env setup is intentional)


@pytest.fixture(autouse=True)
def no_background_scoring(monkeypatch):
    """Disable the thread-pool scorer for every test.

    Without this, background threads writing predictions to the same SQLite
    file can race with the clean_tables fixture, causing spurious FK errors
    and making test isolation unreliable.
    """
    monkeypatch.setattr(app_module, "enqueue_flow_scoring", lambda flow_ids: 0)


@pytest.fixture(scope="session")
def flask_app():
    app_module.app.config.update(TESTING=True)
    return app_module.app


@pytest.fixture()
def client(flask_app):
    return flask_app.test_client()


@pytest.fixture(autouse=True)
def clean_tables(flask_app):
    """Reset all tables between tests so cases stay isolated."""
    yield
    with flask_app.app_context():
        app_module.db.session.rollback()
        app_module.Prediction.query.delete()
        app_module.Flow.query.delete()
        app_module.BlockedIP.query.delete()
        app_module.db.session.commit()


@pytest.fixture()
def admin_token(monkeypatch):
    """Arm the network-response endpoints with a known token for a test."""
    token = "test-admin-token"
    monkeypatch.setattr(app_module, "ADMIN_TOKEN", token)
    return token


def pytest_sessionfinish(session, exitstatus):
    try:
        os.unlink(_DB_PATH)
    except OSError:
        pass
