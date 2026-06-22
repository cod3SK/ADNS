# ADNS — System Architecture

This document is the authoritative reference for how data moves through ADNS.
Individual design decisions live in the numbered ADRs; this file describes the
result.

---

## Pipeline overview

```
┌─────────────────────────────────────────────────────────────┐
│  Network interface (Wi-Fi / Ethernet)                        │
└──────────────────────────┬──────────────────────────────────┘
                           │ raw packets
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  _NfstreamCaptureAgent  (api/app.py)                         │
│                                                              │
│  NFStreamer(source=interface,                                 │
│             idle_timeout=120,                                │
│             active_timeout=1800,                             │
│             n_meters=1,                                      │
│             statistical_analysis=True)                       │
│                                                              │
│  Flows expire naturally via idle/active timeouts.            │
│  Each expired flow → _nf_to_flow() → Flow ORM object         │
│    with flow.extra = flow_to_extra(nf_flow)                  │
│    (stores all 21 FEATURE_COLUMNS in flow.extra)             │
│                                                              │
│  Calls /ingest_batch internally every BATCH_SIZE flows.      │
└──────────────────────────┬──────────────────────────────────┘
                           │ Flow ORM objects
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  api/app.py  —  Flask  POST /ingest_batch                    │
│                                                              │
│  1. Skip flows whose src_ip / dst_ip is in blocked set.      │
│  2. db.session.flush() → assigns DB IDs.                    │
│  3. db.session.commit() → flows are durable.                │
│  4. enforce_batch_flow_retention()                           │
│       • delete flows older than ADNS_BATCH_FLOW_RETENTION    │
│         _MINUTES (65 min)                                    │
│  5. enqueue_flow_scoring(flow_ids) → thread pool             │
└──────────────────────────┬──────────────────────────────────┘
                           │ list of Flow.id integers
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  api/task_queue.py  —  ThreadPoolExecutor                    │
│                                                              │
│  Flow IDs chunked: ADNS_SCORING_BATCH_SIZE per chunk (100).  │
│  Workers: ADNS_SCORER_WORKERS (default 2).                   │
│  Each Future calls: tasks.score_flow_batch(chunk_of_ids)     │
└──────────────────────────┬──────────────────────────────────┘
                           │ background thread
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  api/tasks.py  —  score_flow_batch(flow_ids)                 │
│                                                              │
│  Runs inside an explicit Flask app context.                  │
│  1. Fetch flows from DB in chunks of ADNS_SCORING_FETCH      │
│     _CHUNK (256).                                            │
│  2. nfstream_detector.score_many(flows)                      │
│       → reads flow.extra, calls score_matrix()               │
│  3. Upsert Prediction rows (ON CONFLICT DO NOTHING).         │
│  4. db.session.commit()                                      │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  api/model_runner.py  —  NfstreamDetectionEngine             │
│  api/serving_nfstream.py  —  NfstreamScorer                  │
│                                                              │
│  score_many(flows):                                          │
│    For each flow: extra_to_feature_vector(flow.extra)        │
│      → None if _extractor != "nfstream" (scores 0.0)         │
│      → 21-column numpy row if contract features present      │
│    np.vstack(batch_X) → score_matrix(X)                      │
│                                                              │
│  score_matrix(X):                                            │
│    validate_matrix(X, FEATURE_COLUMNS)  ← raises SchemaError│
│    XGBoost.predict_proba(X) averaged with                    │
│    ExtraTrees.predict_proba(X)                               │
│    → list of (score: float, label: str) tuples               │
│                                                              │
│  Absent model: is_model_loaded=False, model_error set,       │
│  ERROR log emitted; /capture/autostart returns HTTP 503.     │
└─────────────────────────────────────────────────────────────┘

               ▲  dashboard reads via polling
               │
┌─────────────────────────────────────────────────────────────┐
│  frontend/adns-frontend/src/App.jsx  —  React dashboard      │
│                                                              │
│  Live data: setInterval(fetchLatest, 2000) — every 2 s       │
│    GET /api/flows           → last MAX_FLOWS=400 flows        │
│    GET /api/anomalies       → aggregate stats                 │
│    GET /api/anomalous_flows → flows where label≠normal       │
│    GET /api/blocked_ips     → active blocked-IP records       │
│                                                              │
│  Capture status: setInterval(fetchCaptureStatus, 3000)       │
│    GET /api/capture_status →                                 │
│      { version, interface, extractor: "nfstream",            │
│        nfstream: { running, flows_captured, batches_         │
│        processed, last_batch, uptime_seconds, last_error } } │
│                                                              │
│  Model health: setInterval(fetchModelStatus, 10000)          │
│    GET /api/model_status →                                   │
│      { meta_model_status, active_estimators,                 │
│        total_estimators, estimators: {xgboost, extra_trees} }│
│                                                              │
│  Batch Analysis tab: setInterval(fetchBatchSummary, 15000)   │
│    GET /api/batch_summary?window=10m|15m|1h                  │
└─────────────────────────────────────────────────────────────┘
```

---

## Simulation path

`POST /simulate` is synchronous and does **not** use the capture agent or the
background thread pool. It generates synthetic `Flow` objects in memory, scores
them immediately via `_simulation_scorer.score_many()`, inserts Flow + Prediction
rows, and returns.

> **Current limitation:** simulated flows carry a simplified `extra` dict without
> the `_extractor: nfstream` marker. `extra_to_feature_vector()` returns `None`
> for these flows, so `score_many()` returns `(0.0, "normal")`. Real
> live-captured flows score correctly. See `ml/model_card.md` for details.

---

## Legacy ingest path (`/ingest`)

`POST /ingest` accepts flow JSON from external sources (e.g. the legacy
`agent/capture.py` tshark agent). Flows ingested this way have no NFStream
contract features in `extra`, so they score 0.0. This path is preserved for
compatibility but is not the detection path for the desktop app.

---

## Key environment variables

| Variable | Default | Stage |
|---|---|---|
| `ADNS_SCORER_WORKERS` | 2 | Thread pool |
| `ADNS_SCORING_BATCH_SIZE` | 100 | Thread pool |
| `ADNS_SCORING_FETCH_CHUNK` | 256 | Scoring |
| `ADNS_BATCH_FLOW_RETENTION_MINUTES` | 65 | Ingest (batch) |
| `ADNS_FLOW_RETENTION_MINUTES` | 30 | Ingest (live) |
| `ADNS_FLOW_RETENTION_MAX_ROWS` | 5,000 | Ingest (live) |
| `ADNS_ADMIN_TOKEN` | *(unset)* | block_ip / unblock_ip |
| `SQLALCHEMY_DATABASE_URI` | `sqlite:///./adns.db` | API |

---

## Feature contract

Single source of truth: `ml/adns_flows/schema.py:FEATURE_COLUMNS` (21 columns).

Extraction: `ml/adns_flows/extract_nfstream.py:flows_to_dataframe_nfstream()` (corpus)
and `api/serving_nfstream.py:flow_to_extra()` (live) call the same logic.

`validate_matrix(data, columns)` raises `SchemaError` on any column name or order
mismatch. Called before every `model.predict()`.

---

## Related documents

- [ADR 0002](0002-async-scoring-redis-rq.md) — async scoring; ThreadPoolExecutor
- [ADR 0003](0003-three-tier-detection-cascade.md) — detection engine (superseded; see amendment)
- [ADR 0004](0004-postgres-persistence-and-retention.md) — persistence and retention
- [ADR 0005](0005-feature-synthesis-for-sparse-telemetry.md) — feature contract (superseded; see amendment)
- [ADR 0006](0006-attack-simulation-subsystem.md) — simulation subsystem
- [ADR 0010](0010-windows-desktop-packaging.md) — desktop packaging
- [`../ml/model_card.md`](../ml/model_card.md) — model metrics and limitations
