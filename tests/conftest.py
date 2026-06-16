"""Shared pytest fixtures for the ADNS test suite."""

import os
import sys

# Must be set before app.py is imported so SQLAlchemy binds to an in-memory DB.
os.environ.setdefault("SQLALCHEMY_DATABASE_URI", "sqlite:///:memory:")
os.environ.setdefault("ADNS_RDNS_ENABLED", "false")
os.environ.setdefault("ADNS_FLOW_RETENTION_MINUTES", "30")
os.environ.setdefault("ADNS_NSENTER_HOST", "false")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "api"))

import pytest


@pytest.fixture(scope="session")
def flask_app():
    from app import app, db
    app.config["TESTING"] = True
    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///:memory:"
    app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {"connect_args": {"check_same_thread": False}}
    with app.app_context():
        db.create_all()
        yield app


@pytest.fixture()
def app(flask_app):
    return flask_app


@pytest.fixture()
def client(app):
    return app.test_client()


@pytest.fixture(autouse=True)
def clean_db(app):
    """Wipe all rows between tests; schema is kept (session-scoped)."""
    from app import db, Flow, Prediction, BlockedIP
    with app.app_context():
        yield
        db.session.rollback()
        Prediction.query.delete()
        Flow.query.delete()
        BlockedIP.query.delete()
        db.session.commit()
