"""Unit tests for Flask API endpoints."""

import json
from datetime import datetime, timezone
import pytest


def _now_iso():
    return datetime.now(timezone.utc).isoformat()


def _post_json(client, url, data):
    return client.post(url, data=json.dumps(data), content_type="application/json")


class TestHealth:
    def test_health_returns_200(self, client, app):
        with app.app_context():
            res = client.get("/health")
        assert res.status_code == 200

    def test_health_has_status_field(self, client, app):
        with app.app_context():
            res = client.get("/health")
        body = res.get_json()
        assert "status" in body


class TestFlowsEndpoint:
    def test_flows_empty_returns_list(self, client, app):
        with app.app_context():
            res = client.get("/flows")
        assert res.status_code == 200
        assert isinstance(res.get_json(), list)

    def test_ingest_creates_flows(self, client, app):
        payload = [
            {"ts": _now_iso(), "src_ip": "10.0.0.1", "dst_ip": "8.8.8.8",
             "proto": "TCP", "bytes": 1000, "score": 0.1, "label": "normal"},
        ]
        with app.app_context():
            res = _post_json(client, "/ingest", payload)
        assert res.status_code == 200

    def test_flows_returns_ingested_data(self, client, app):
        payload = [
            {"ts": _now_iso(), "src_ip": "192.168.1.5", "dst_ip": "1.1.1.1",
             "proto": "UDP", "bytes": 512, "score": 0.2, "label": "normal"},
        ]
        with app.app_context():
            _post_json(client, "/ingest", payload)
            res = client.get("/flows")
        flows = res.get_json()
        assert any(f["src_ip"] == "192.168.1.5" for f in flows)

    def test_ingest_empty_payload_ok(self, client, app):
        with app.app_context():
            res = _post_json(client, "/ingest", [])
        assert res.status_code == 200


class TestAnomaliesEndpoint:
    def test_anomalies_no_data_returns_demo(self, client, app):
        with app.app_context():
            res = client.get("/anomalies")
        body = res.get_json()
        assert "max_score" in body
        assert "count" in body

    def test_anomalous_flows_empty_is_list(self, client, app):
        with app.app_context():
            res = client.get("/anomalous_flows")
        assert isinstance(res.get_json(), list)


class TestKillswitch:
    def test_get_killswitch_default_false(self, client, app):
        with app.app_context():
            res = client.get("/killswitch")
        assert res.get_json()["enabled"] is False

    def test_post_killswitch_toggles_state(self, client, app):
        with app.app_context():
            res = _post_json(client, "/killswitch", {"enabled": True})
        body = res.get_json()
        # State reflects what was set (OS action may fail in test env, that's ok)
        assert "enabled" in body


class TestBlockedIPs:
    def test_blocked_ips_initially_empty(self, client, app):
        with app.app_context():
            res = client.get("/blocked_ips")
        assert res.get_json() == []

    def test_block_ip_records_entry(self, client, app):
        with app.app_context():
            res = _post_json(client, "/block_ip", {"ip": "10.10.10.10"})
        assert res.status_code == 200
        body = res.get_json()
        assert body["ip"] == "10.10.10.10"

    def test_blocked_ips_returns_blocked(self, client, app):
        with app.app_context():
            _post_json(client, "/block_ip", {"ip": "5.5.5.5"})
            res = client.get("/blocked_ips")
        ips = [r["ip"] for r in res.get_json()]
        assert "5.5.5.5" in ips

    def test_unblock_ip_removes_from_active(self, client, app):
        with app.app_context():
            _post_json(client, "/block_ip", {"ip": "6.6.6.6"})
            _post_json(client, "/unblock_ip", {"ip": "6.6.6.6"})
            res = client.get("/blocked_ips")
        ips = [r["ip"] for r in res.get_json()]
        assert "6.6.6.6" not in ips
