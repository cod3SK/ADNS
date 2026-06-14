"""Auth-gating tests for the network-response endpoints.

These endpoints shell out to iptables, so they must fail closed when no admin
token is configured and reject mismatched tokens when one is.
"""

import app as app_module


def test_block_ip_disabled_without_token(client):
    resp = client.post("/block_ip", json={"ip": "1.2.3.4"})
    assert resp.status_code == 403
    assert resp.get_json()["error"] == "endpoint disabled"


def test_unblock_ip_disabled_without_token(client):
    resp = client.post("/unblock_ip", json={"ip": "1.2.3.4"})
    assert resp.status_code == 403


def test_killswitch_post_disabled_without_token(client):
    resp = client.post("/killswitch", json={"enabled": True})
    assert resp.status_code == 403


def test_killswitch_get_is_open(client):
    # Reading state is harmless and must work without a token.
    resp = client.get("/killswitch")
    assert resp.status_code == 200
    assert resp.get_json() == {"enabled": False}


def test_block_ip_rejects_wrong_token(client, admin_token):
    resp = client.post(
        "/block_ip",
        json={"ip": "1.2.3.4"},
        headers={"Authorization": "Bearer wrong-token"},
    )
    assert resp.status_code == 401


def test_block_ip_accepts_valid_token(client, admin_token, monkeypatch):
    # Stub the OS-level iptables call so the test stays platform-independent.
    monkeypatch.setattr(app_module, "block_ip_os", lambda ip, allow=False: (True, "blocked"))
    resp = client.post(
        "/block_ip",
        json={"ip": "1.2.3.4"},
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["status"] == "blocked"
    assert body["ip"] == "1.2.3.4"

    # The IP should now be recorded as blocked.
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


def test_ingest_drops_blocked_ip(client, admin_token, monkeypatch):
    monkeypatch.setattr(app_module, "block_ip_os", lambda ip, allow=False: (True, "blocked"))
    client.post(
        "/block_ip",
        json={"ip": "9.9.9.9"},
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    resp = client.post(
        "/ingest",
        json={"src_ip": "9.9.9.9", "dst_ip": "10.0.0.1", "proto": "6", "bytes": 100},
    )
    body = resp.get_json()
    assert body["blocked"] == 1
    assert body["ingested"] == 0
