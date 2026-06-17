# ADNS — System Architecture

This document is the authoritative reference for how data moves through ADNS:
every component, every call, every batch size, and every timing bound.
Individual design decisions that shaped these choices live in the numbered ADRs;
this file describes the result.

---

## Pipeline overview

```
                          ┌─────────────────────────────────────────────┐
                          │  Network interface (eth0 / INTERFACE)        │
                          └──────────┬──────────────────────────────────┘
                                     │ raw packets
                          ┌──────────┴──────────────────────────────────┐
                          │                                              │
                          ▼  per-packet (real-time)                     ▼  ring-buffer pcaps
┌──────────────────────────────────┐          ┌──────────────────────────────────────────────┐
│  agent/capture.py                │          │  agent/batch_capture.py                      │
│                                  │          │                                              │
│  tshark line-buffered; each line │          │  tshark -b duration:N -b files:M writes      │
│  = one packet. 20 fields parsed. │          │  rotating pcaps. When a file is complete     │
│  Normalized into flow dicts,     │          │  (not the newest by mtime), runs two passes: │
│  buffered.                       │          │    Pass 1: -z conv,tcp/udp → real duration,  │
│                                  │          │      directional bytes + packet counts        │
│  Flush triggers:                 │          │    Pass 2: -T fields → app-layer dissection  │
│    • BATCH_SIZE packets (50)     │          │  Merges on bidirectional 4-tuple. Deletes     │
│    • POST_INTERVAL elapsed (2 s) │          │  processed pcap. Polls every 2 s.            │
│                                  │          │                                              │
│  On flush: POST to /ingest.      │          │  On merge: POST to /ingest_batch.            │
│  On failure: retry after         │          │  Env: INTERFACE, BATCH_WINDOW_SECONDS (15),  │
│    RETRY_DELAY (3.0 s).          │          │    BATCH_DIR, BATCH_API_URL, TSHARK_BIN      │
└─────────────┬────────────────────┘          └──────────────────────┬───────────────────────┘
              │ JSON array, up to BATCH_SIZE flows                   │ JSON array of merged flows
              │ POST /ingest  (local HTTP)                           │ POST /ingest_batch
              ▼                                                       ▼
┌───────────────────────────────────────────────────────────────────────┐
│  api/app.py  —  Flask  POST /ingest  /  POST /ingest_batch            │
│                                                                       │
│  /ingest:                                                             │
│  1. Read blocked-IP set from DB (one SELECT at request start).        │
│  2. For each flow record:                                             │
│       • skip if src_ip or dst_ip is in blocked set                    │
│       • build Flow ORM object, source='live'                          │
│  3. db.session.flush() → assigns DB IDs                              │
│  4. db.session.commit() → flows are durable                          │
│  5. enforce_flow_retention() (live flows only):                       │
│       • delete flows older than ADNS_FLOW_RETENTION_MINUTES (30 min) │
│       • trim to ADNS_FLOW_RETENTION_MAX_ROWS (5 000) if exceeded      │
│  6. enqueue_flow_scoring(flow_ids) → see thread pool below            │
│                                                                       │
│  /ingest_batch (HTTP endpoint + _BatchCaptureAgent._ingest):          │
│  1. Writes flows with source='batch'.                                 │
│  2. enforce_batch_flow_retention() (batch flows only, 65-min window). │
│  3. enqueue_flow_scoring(flow_ids) → same scoring path.              │
│                                                                       │
│  Response: {"status":"ok","ingested":N,...}                           │
└───────────────────────────────┬───────────────────────────────────────┘
                                │ list of Flow.id integers
                                ▼
┌───────────────────────────────────────────────────────────────────────┐
│  api/task_queue.py  —  ThreadPoolExecutor                             │
│                                                                       │
│  Flow IDs are chunked: ADNS_SCORING_BATCH_SIZE per chunk (default     │
│  100).  A batch of 50 flows from one POST /ingest produces one        │
│  submitted Future.                                                    │
│  Worker threads: ADNS_SCORER_WORKERS  (default 2)                    │
│                                                                       │
│  Each Future calls: tasks.score_flow_batch(chunk_of_ids)              │
└───────────────────────────────┬───────────────────────────────────────┘
                                │ background thread → calls score_flow_batch()
                                ▼
┌───────────────────────────────────────────────────────────────────────┐
│  api/tasks.py  —  score_flow_batch(flow_ids)                          │
│                                                                       │
│  Runs inside an explicit Flask app context (safe for the worker       │
│  process and the inline fallback path).                               │
│                                                                       │
│  1. detector.reload_if_stale()                                        │
│       Compares artifact mtimes; reloads DetectionEngine if changed.   │
│  2. Fetch flows from DB in chunks of ADNS_SCORING_FETCH_CHUNK (256). │
│  3. Per chunk — rDNS enrichment (if ADNS_RDNS_ENABLED=true):          │
│       For each flow: resolver.lookup(peer_ip)                         │
│         • In-process LRU cache: size ADNS_RDNS_CACHE_SIZE (500),      │
│           TTL ADNS_RDNS_CACHE_TTL (900 s).                            │
│         • Per-lookup network timeout: ADNS_RDNS_TIMEOUT_MS (500 ms). │
│         • Writes rdns_exists / rdns_hash into flow.extra (not saved). │
│  4. detector.predict_many(session, flows)  →  list of predictions     │
│  5. Upsert Prediction rows (ON CONFLICT DO NOTHING on flow_id).       │
│  6. db.session.commit()  →  predictions are durable and queryable.   │
└───────────────────────────────┬───────────────────────────────────────┘
                                │ predictions written to DB
                                ▼
┌───────────────────────────────────────────────────────────────────────┐
│  api/model_runner.py  —  DetectionEngine                              │
│                                                                       │
│  Single model (MetaEnsembleModel):                                    │
│    ExtraTrees + XGBoost averaged                                      │
│    artifact: model_artifacts/meta_model_combined.joblib               │
│    anomaly threshold: ADNS_META_ANOMALY_THRESHOLD (0.82)              │
│    watch   threshold: ADNS_META_WATCH_THRESHOLD   (0.60)              │
│    Returns (0.0, "normal") when artifact is absent (dev/test).        │
│                                                                       │
│  predict_many() processes the entire chunk as a single DataFrame      │
│  inference call (meta/ml) or sequential predict() calls (heuristic). │
└───────────────────────────────────────────────────────────────────────┘

                    ▲  dashboard reads via polling
                    │
┌───────────────────────────────────────────────────────────────────────┐
│  frontend/adns-frontend/src/App.jsx  —  React dashboard               │
│                                                                       │
│  Live data: setInterval(fetchLatest, 2000) — fires every 2 s          │
│    GET /api/flows           → last MAX_FLOWS=400 live flows            │
│    GET /api/anomalies       → aggregate stats over live buffer         │
│    GET /api/anomalous_flows → live flows where label≠normal/score≥0.6 │
│    GET /api/blocked_ips     → active blocked-IP records                │
│                                                                       │
│  Capture status: setInterval(fetchCaptureStatus, 3000)                │
│    GET /api/capture_status → interface name, tshark found,            │
│      live (running/flows/last ingest/uptime/error),                   │
│      batch (running/batches/last batch/uptime/error)                  │
│                                                                       │
│  Batch Analysis tab: setInterval(fetchBatchSummary, 15000)            │
│    GET /api/batch_summary?window=10m|15m|1h                           │
│      → total_flows, total_bytes, anomaly_count, anomaly_rate,         │
│         proto_breakdown, top_src_ips, top_dst_ips, timeseries         │
└───────────────────────────────────────────────────────────────────────┘
```

---

## Simulation path (bypass)

`POST /simulate` is synchronous and does **not** use the capture agent, the
ingest endpoint, or the background thread pool.  It generates synthetic Flow objects in
memory, scores them immediately via `simulation_detector.predict()`, inserts
Flow + Prediction rows in one transaction, and returns.  The flows are visible
on the next dashboard poll (~0–2 s later).

For streaming simulations (`duration_seconds > 0`) a background thread repeats
the same generate → score → commit cycle every `interval_seconds` (default 1.0 s,
range 0.5–5.0 s) until the deadline, with a cap of 200 flows per batch and a
total cap of 250 flows per one-shot call.

---

## End-to-end latency

Each stage contributes a bounded wait.  The table shows the delay a single flow
accumulates at each step before it is visible on the dashboard.

| Stage | Min | Typical | Max | Controlling variable |
|---|---|---|---|---|
| Capture buffer | ~0 ms | ~1 000 ms | 2 000 ms | `POST_INTERVAL` (2.0 s) |
| HTTP POST to /ingest | <5 ms | <20 ms | 5 000 ms | agent HTTP timeout |
| DB flush + commit | <5 ms | <20 ms | ~100 ms | Postgres latency |
| Thread pool submit | <1 ms | <1 ms | ~5 ms | in-process (no round-trip) |
| Thread pool pickup | <1 ms | ~1–10 ms | ~50 ms | depends on worker load |
| rDNS lookup (cache miss) | 0 ms | 0–500 ms | 500 ms × N flows | `ADNS_RDNS_TIMEOUT_MS` |
| Model inference | <1 ms | <10 ms | ~50 ms | model tier in use |
| DB prediction write | <5 ms | <10 ms | ~50 ms | Postgres latency |
| Frontend poll wait | ~0 ms | ~1 000 ms | 2 000 ms | `setInterval` (2 s) |
| **End-to-end (no rDNS miss)** | **~1 s** | **~3–5 s** | **~10 s** | |
| **End-to-end (cold rDNS, 50 flows)** | — | — | **~35 s** | disable with `ADNS_RDNS_ENABLED=false` |

---

## Batch sizes at every level

| Level | Batch size | Variable | Where enforced |
|---|---|---|---|
| Capture → /ingest | 50 flows | `BATCH_SIZE` | `agent/capture.py:34` |
| /ingest → thread pool | 100 flow IDs per Future | `ADNS_SCORING_BATCH_SIZE` | `api/task_queue.py` |
| Worker DB fetch | 256 rows per SELECT | `ADNS_SCORING_FETCH_CHUNK` | `api/tasks.py:18` |
| Retention delete | 1 000 rows per DELETE | hardcoded | `api/app.py:664` |
| Dashboard /flows | 400 flows returned | `MAX_FLOWS` | `api/app.py:30` |
| Simulation (one-shot) | 5–250 flows | `count`, capped in /simulate | `api/app.py:810` |
| Simulation (streaming) | up to 200 flows per tick | `batch_size` in /simulate | `api/app.py:835` |

---

## Key environment variables

| Variable | Default | Stage |
|---|---|---|
| `BATCH_SIZE` | 50 | Live capture |
| `POST_INTERVAL` | 2.0 s | Live capture |
| `RETRY_DELAY` | 3.0 s | Live capture |
| `API_URL` | http://127.0.0.1:5000/ingest | Live capture |
| `INTERFACE` | eth0 | Both capture agents |
| `TSHARK_BIN` | *(auto-detected)* | Both capture agents |
| `BATCH_WINDOW_SECONDS` | 15 | Batch capture |
| `BATCH_DIR` | *(system temp)* | Batch capture |
| `BATCH_API_URL` | http://127.0.0.1:5000/ingest_batch | Batch capture |
| `ADNS_BATCH_FLOW_RETENTION_MINUTES` | 65 | Ingest (batch) |
| `ADNS_SCORER_WORKERS` | 2 | Thread pool |
| `ADNS_SCORING_BATCH_SIZE` | 100 | Thread pool |
| `ADNS_SCORING_FETCH_CHUNK` | 256 | Scoring |
| `ADNS_RDNS_ENABLED` | true | Scoring |
| `ADNS_RDNS_TIMEOUT_MS` | 500 | Scoring |
| `ADNS_RDNS_CACHE_SIZE` | 500 | Scoring |
| `ADNS_RDNS_CACHE_TTL` | 900 s | Scoring |
| `ADNS_META_ANOMALY_THRESHOLD` | 0.82 | Detection |
| `ADNS_META_WATCH_THRESHOLD` | 0.60 | Detection |
| `ADNS_FLOW_RETENTION_MINUTES` | 30 | Ingest (live) |
| `ADNS_FLOW_RETENTION_MAX_ROWS` | 5 000 | Ingest (live) |
| `ADNS_ADMIN_TOKEN` | *(unset)* | block_ip / unblock_ip (killswitch is not gated) |
| `SQLALCHEMY_DATABASE_URI` | postgresql://adns:adns_password@127.0.0.1/adns | API |

---

## Related documents

- [ADR 0001](0001-microservice-architecture.md) — why the pipeline is split across services
- [ADR 0002](0002-async-scoring-redis-rq.md) — why scoring is async; Redis/RQ → ThreadPoolExecutor migration
- [ADR 0003](0003-three-tier-detection-cascade.md) — the detection cascade and hot reload
- [ADR 0004](0004-postgres-persistence-and-retention.md) — persistence and retention policy
- [ADR 0005](0005-feature-synthesis-for-sparse-telemetry.md) — feature engineering for live telemetry
- [ADR 0006](0006-attack-simulation-subsystem.md) — the simulation subsystem
- [`../ml/model_card.md`](../ml/model_card.md) — model metrics, training data, and limitations
