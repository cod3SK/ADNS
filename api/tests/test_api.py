"""Endpoint-level tests for the ADNS Flask API."""

import app as app_module


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.get_json() == {"status": "ok"}


def test_ingest_persists_flow_and_appears_in_flows(client):
    payload = {
        "ts": "2026-01-01T00:00:00Z",
        "src_ip": "192.168.1.10",
        "dst_ip": "8.8.8.8",
        "proto": "6",
        "bytes": 1500,
        "src_port": 51000,
        "dst_port": 443,
        "service": "https",
    }
    resp = client.post("/ingest", json=payload)
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["ingested"] == 1

    flows = client.get("/flows").get_json()
    assert any(f["src_ip"] == "192.168.1.10" and f["dst_ip"] == "8.8.8.8" for f in flows)
    # proto "6" must be normalized to a human-readable name.
    match = next(f for f in flows if f["src_ip"] == "192.168.1.10")
    assert match["proto"] == "TCP"


def test_ingest_accepts_list_payload(client):
    batch = [
        {"src_ip": "10.0.0.1", "dst_ip": "10.0.0.2", "proto": "17", "bytes": 200},
        {"src_ip": "10.0.0.3", "dst_ip": "10.0.0.4", "proto": "6", "bytes": 9000},
    ]
    resp = client.post("/ingest", json=batch)
    assert resp.status_code == 200
    assert resp.get_json()["ingested"] == 2


def test_ingest_rejects_non_object_payload(client):
    resp = client.post("/ingest", json="not-a-flow")
    assert resp.status_code == 400


def test_flows_falls_back_to_demo_when_empty(client):
    # No flows ingested in this isolated test -> demo data is returned.
    flows = client.get("/flows").get_json()
    assert isinstance(flows, list) and len(flows) >= 1
    assert {"src_ip", "dst_ip", "proto", "label"} <= set(flows[0].keys())


def test_simulate_generates_and_scores(client):
    resp = client.post("/simulate", json={"type": "ddos", "count": 8})
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["status"] == "ok"
    assert body["type"] == "ddos"
    assert body["generated"] == 8
    assert 0.0 <= body["max_score"] <= 1.0


def test_simulate_unknown_type_returns_400(client):
    resp = client.post("/simulate", json={"type": "definitely_not_real"})
    assert resp.status_code == 400


def test_simulate_clamps_count(client):
    # count is clamped to the [5, 250] range.
    resp = client.post("/simulate", json={"type": "scanning", "count": 9999})
    assert resp.status_code == 200
    assert resp.get_json()["generated"] == 250


def test_anomalies_shape(client):
    client.post("/simulate", json={"type": "ddos", "count": 10})
    stats = client.get("/anomalies").get_json()
    assert {"window", "count", "max_score", "pct_anomalous"} <= set(stats.keys())
