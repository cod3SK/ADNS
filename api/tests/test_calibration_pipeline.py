"""Tests for the deployment-time calibration pipeline.

Covers:
  - Protocol whitelist filter (whitelist.py)
  - Statistical outlier drop (whitelist.py)
  - Pipeline state machine transitions (pipeline.py)
  - API routes: status, start validation, revert, first-run check
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# Ensure api/ is on path so imports resolve.
API_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(API_DIR))

from calibration.whitelist import SAFE_PORTS, apply_outlier_drop, apply_whitelist

# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_df(**kwargs) -> pd.DataFrame:
    """Build a minimal DataFrame for filter tests."""
    n = kwargs.pop("n", 5)
    base = {
        "src_port":    [12345] * n,
        "dst_port":    [443]   * n,   # HTTPS — safe
        "total_bytes": [1000]  * n,
        "src_bytes":   [800]   * n,
        "dst_bytes":   [200]   * n,
        "proto":       [6]     * n,
    }
    base.update(kwargs)
    return pd.DataFrame(base)


# ── Whitelist tests ────────────────────────────────────────────────────────────

class TestApplyWhitelist:
    def test_keeps_safe_dst_port(self):
        df = _make_df(dst_port=[443, 53, 80, 5353, 67])
        kept, n_dropped = apply_whitelist(df)
        assert len(kept) == 5
        assert n_dropped == 0

    def test_drops_unknown_dst_port(self):
        # Port 9999 is not in SAFE_PORTS
        df = _make_df(n=3, dst_port=[9999, 9999, 9999])
        kept, n_dropped = apply_whitelist(df)
        assert len(kept) == 0
        assert n_dropped == 3

    def test_keeps_via_src_port(self):
        # dst_port unknown, but src_port=443 saves it
        df = _make_df(n=2, dst_port=[9999, 9999], src_port=[443, 53])
        kept, n_dropped = apply_whitelist(df)
        assert len(kept) == 2
        assert n_dropped == 0

    def test_empty_df_returns_empty(self):
        df = pd.DataFrame(columns=["src_port", "dst_port", "total_bytes"])
        kept, n_dropped = apply_whitelist(df)
        assert kept.empty
        assert n_dropped == 0

    def test_no_port_columns_keeps_all(self):
        df = pd.DataFrame({"proto": [6, 17], "total_bytes": [100, 200]})
        kept, n_dropped = apply_whitelist(df)
        assert len(kept) == 2
        assert n_dropped == 0

    def test_mixed_safe_and_unsafe(self):
        df = _make_df(n=4, dst_port=[443, 9999, 53, 31337])
        kept, n_dropped = apply_whitelist(df)
        assert len(kept) == 2   # 443 and 53
        assert n_dropped == 2

    def test_mdns_port_5353_kept(self):
        assert 5353 in SAFE_PORTS
        df = _make_df(n=3, dst_port=[5353, 5353, 5353])
        kept, n_dropped = apply_whitelist(df)
        assert len(kept) == 3

    def test_dhcp_ports_kept(self):
        assert 67 in SAFE_PORTS
        assert 68 in SAFE_PORTS
        df = _make_df(n=2, dst_port=[67, 68])
        kept, n_dropped = apply_whitelist(df)
        assert len(kept) == 2

    def test_ssdp_port_kept(self):
        assert 1900 in SAFE_PORTS
        df = _make_df(n=1, dst_port=[1900])
        kept, _ = apply_whitelist(df)
        assert len(kept) == 1


# ── Outlier drop tests ─────────────────────────────────────────────────────────

class TestApplyOutlierDrop:
    def test_no_outliers_keeps_all(self):
        df = _make_df(n=5, total_bytes=[100, 110, 90, 105, 95])
        kept, n_dropped, warn = apply_outlier_drop(df)
        assert len(kept) == 5
        assert n_dropped == 0
        assert warn is False

    def test_drops_large_outlier(self):
        # One huge value is clearly > median + 3*std
        bytes_vals = [100, 110, 90, 105, 95, 1_000_000]
        df = _make_df(n=6, total_bytes=bytes_vals)
        kept, n_dropped, warn = apply_outlier_drop(df)
        assert n_dropped == 1
        assert len(kept) == 5

    def test_warn_fires_above_10_pct(self):
        # Make > 10% of rows outliers
        normal = [100] * 8
        outliers = [10_000_000, 20_000_000]   # 2/10 = 20% > 10%
        df = _make_df(n=10, total_bytes=normal + outliers)
        _, n_dropped, warn = apply_outlier_drop(df)
        assert n_dropped == 2
        assert warn is True

    def test_warn_not_fires_below_10_pct(self):
        normal = [100] * 100
        df = _make_df(n=100, total_bytes=normal)
        _, n_dropped, warn = apply_outlier_drop(df)
        assert n_dropped == 0
        assert warn is False

    def test_empty_df_no_crash(self):
        df = pd.DataFrame(columns=["total_bytes"])
        kept, n_dropped, warn = apply_outlier_drop(df)
        assert kept.empty
        assert n_dropped == 0
        assert warn is False

    def test_missing_column_keeps_all(self):
        df = pd.DataFrame({"proto": [6, 17], "duration": [1.0, 2.0]})
        kept, n_dropped, warn = apply_outlier_drop(df)
        assert len(kept) == 2
        assert n_dropped == 0


# ── Pipeline state tests ───────────────────────────────────────────────────────

class TestPipelineState:
    def setup_method(self):
        """Reset pipeline state before each test."""
        from calibration import pipeline as p
        p._STATE.update({
            "stage": "idle", "progress": 0, "message": "",
            "error": None, "result": None, "cancel_requested": False,
        })

    def test_get_state_returns_copy(self):
        from calibration.pipeline import get_state
        state = get_state()
        state["stage"] = "hacked"
        from calibration.pipeline import _STATE
        assert _STATE["stage"] == "idle"   # original unchanged

    def test_reset_from_failed(self):
        from calibration import pipeline as p
        p._STATE["stage"] = "failed"
        p._STATE["error"] = "something"
        p.reset_pipeline()
        assert p._STATE["stage"] == "idle"
        assert p._STATE["error"] is None

    def test_reset_from_done(self):
        from calibration import pipeline as p
        p._STATE["stage"] = "done"
        p.reset_pipeline()
        assert p._STATE["stage"] == "idle"

    def test_cancel_sets_flag(self):
        from calibration import pipeline as p
        p._STATE["stage"] = "capturing"
        p.cancel_pipeline()
        assert p._STATE["cancel_requested"] is True

    def test_start_raises_if_already_running(self):
        from calibration import pipeline as p
        p._STATE["stage"] = "capturing"
        with pytest.raises(RuntimeError, match="already running"):
            p.start_pipeline("eth0", capture_duration=60)


# ── API route tests ────────────────────────────────────────────────────────────

class TestCalibrationRoutes:
    @pytest.fixture
    def client(self):
        """Flask test client with calibration blueprint registered."""
        import app as app_mod
        app_mod.app.config["TESTING"] = True
        app_mod.app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///:memory:"
        with app_mod.app.app_context():
            app_mod.db.create_all()
        with app_mod.app.test_client() as c:
            yield c

    def test_status_returns_json(self, client):
        from calibration import pipeline as p
        p._STATE["stage"] = "idle"
        r = client.get("/calibration/status")
        assert r.status_code == 200
        data = r.get_json()
        assert "stage" in data
        assert "progress" in data

    def test_start_requires_interface(self, client):
        r = client.post("/calibration/start", json={})
        assert r.status_code == 400
        assert "interface" in r.get_json()["error"]

    def test_start_validates_duration(self, client):
        r = client.post("/calibration/start", json={"interface": "eth0", "capture_duration": 10})
        assert r.status_code == 400

    def test_start_409_when_running(self, client):
        from calibration import pipeline as p
        p._STATE["stage"] = "capturing"
        r = client.post("/calibration/start", json={"interface": "eth0", "capture_duration": 120})
        assert r.status_code == 409
        p._STATE["stage"] = "idle"

    def test_cancel_returns_ok(self, client):
        r = client.post("/calibration/cancel")
        assert r.status_code == 200
        assert r.get_json()["status"] == "cancel_requested"

    def test_reset_returns_ok(self, client):
        r = client.post("/calibration/reset")
        assert r.status_code == 200
        assert r.get_json()["status"] == "reset"

    def test_first_run_check_returns_flags(self, client):
        r = client.get("/calibration/first_run_check")
        assert r.status_code == 200
        data = r.get_json()
        assert "calibrated_model_exists" in data
        assert "suggest_calibration" in data

    def test_revert_404_when_no_backup(self, client, tmp_path):
        # Point revert at a non-existent backup
        from calibration import pipeline as p
        original = p.MODEL_PATH
        p.MODEL_PATH = tmp_path / "model.joblib"
        try:
            r = client.post("/calibration/revert")
            assert r.status_code == 404
        finally:
            p.MODEL_PATH = original
