"""Endpoint tests for network-response actions.

Block/unblock always update the DB (no token required); iptables is only
attempted when a valid admin token is present (so it works in environments
where iptables is available). The killswitch POST remains fully token-gated.
"""

import app as app_module


def test_block_ip_works_without_token(client):
    resp = client.post("/block_ip", json={"ip": "1.2.3.4"})
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["status"] == "blocked"
    assert body["os_action"] == "not_configured"

    blocked = client.get("/blocked_ips").get_json()
    assert any(row["ip"] == "1.2.3.4" for row in blocked)


def test_unblock_ip_works_without_token(client, monkeypatch):
    monkeypatch.setattr(app_module, "block_ip_os", lambda ip, allow=False: (True, ""))
    client.post("/block_ip", json={"ip": "2.3.4.5"})
    resp = client.post("/unblock_ip", json={"ip": "2.3.4.5"})
    assert resp.status_code == 200
    assert resp.get_json()["status"] == "unblocked"


def test_killswitch_post_disabled_without_token(client):
    resp = client.post("/killswitch", json={"enabled": True})
    assert resp.status_code == 403


def test_killswitch_get_is_open(client):
    resp = client.get("/killswitch")
    assert resp.status_code == 200
    assert resp.get_json() == {"enabled": False}


def test_block_ip_wrong_token_skips_os_action(client, admin_token):
    resp = client.post(
        "/block_ip",
        json={"ip": "3.4.5.6"},
        headers={"Authorization": "Bearer wrong-token"},
    )
    assert resp.status_code == 200
    assert resp.get_json()["os_action"] == "not_configured"
    # IP still recorded in DB
    blocked = client.get("/blocked_ips").get_json()
    assert any(row["ip"] == "3.4.5.6" for row in blocked)


def test_block_ip_valid_token_triggers_os_action(client, admin_token, monkeypatch):
    monkeypatch.setattr(app_module, "block_ip_os", lambda ip, allow=False: (True, "blocked"))
    resp = client.post(
        "/block_ip",
        json={"ip": "1.2.3.4"},
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["status"] == "blocked"
    assert body["os_action"] == "ok"

    blocked = client.get("/blocked_ips").get_json()
    assert any(row["ip"] == "1.2.3.4" for row in blocked)


def test_block_ip_accepts_x_admin_token_header(client, admin_token, monkeypatch):
    monkeypatch.setattr(app_module, "block_ip_os", lambda ip, allow=False: (True, "blocked"))
    resp = client.post(
        "/block_ip",
        json={"ip": "5.6.7.8"},
        headers={"X-Admin-Token": admin_token},
    )
    assert resp.status_code == 200
    assert resp.get_json()["os_action"] == "ok"


def test_ingest_drops_blocked_ip(client):
    client.post("/block_ip", json={"ip": "9.9.9.9"})
    resp = client.post(
        "/ingest",
        json={"src_ip": "9.9.9.9", "dst_ip": "10.0.0.1", "proto": "6", "bytes": 100},
    )
    body = resp.get_json()
    assert body["blocked"] == 1
    assert body["ingested"] == 0
