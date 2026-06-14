# ADNS Agent Brief

Use this page to remind yourself (or other assistants) what lives where in the ADNS stack and how the pieces talk to each other.

## Mission Snapshot
- **Goal**: Demonstrate a modern network anomaly detection loop end to end (capture -> ingest -> score -> visualize) with synthetic attack simulations for workshops.
- **Live topology**: `agent/capture.py` runs `tshark` on `eth0`, POSTs batches to the Flask API (Gunicorn on 127.0.0.1:5000; Nginx proxies `/api/*` to it). The API persists flows in PostgreSQL, enqueues flow IDs on Redis/RQ, and the scoring worker writes `Prediction` rows. The React/Vite dashboard (`frontend/adns-frontend/dist`) is served by Nginx at `http://159.203.105.167/`.
- **Storage**: PostgreSQL holds `flows` + `predictions` (see `api/app.py`). Retention trims anything older than `ADNS_FLOW_RETENTION_MINUTES` or beyond `ADNS_FLOW_RETENTION_MAX_ROWS`. The DB URI now respects `SQLALCHEMY_DATABASE_URI` env (defaults to `postgresql://adns:adns_password@127.0.0.1/adns`).
- **Models**: `api/model_runner.py` loads `model_artifacts/flow_detector.joblib` and `meta_model_combined.joblib` (ExtraTrees + XGBoost) to drive the DetectionEngine, falling back to heuristics if artifacts are absent. Training/data prep lives under `ml/`.

## Component Reference
| Piece | Path | Notes |
| --- | --- | --- |
| Capture agent | `agent/capture.py` | Wraps `tshark` (fields in `TSHARK_FIELDS`), infers ports/services, batches ~50 flows or ~2 s, retries POSTs with backoff. Settings read from env (`API_URL`, `TSHARK_BIN`, `INTERFACE`, `BATCH_SIZE`, `POST_INTERVAL`, `RETRY_DELAY`) with defaults. `requirements.txt` pins `requests`. |
| API | `api/app.py` | Flask + SQLAlchemy; exposes `/health`, `/ingest`, `/flows`, `/anomalies`, `/simulate`. Creates tables, ensures `flows.extra` JSON column **and** enforces unique `predictions.flow_id` on bootstrap. DB URI is env-driven (`SQLALCHEMY_DATABASE_URI`) with local Postgres default. Flow inserts enqueue `tasks.score_flow_batch` via `task_queue.py` with inline scoring fallback if enqueue fails. |
| Task queue | `api/task_queue.py`, `api/tasks.py` | Redis URL from `ADNS_REDIS_URL`. RQ queue name defaults to `flow_scores`. `score_flow_batch` loads flows in chunks (`ADNS_SCORING_FETCH_CHUNK`), skips already scored IDs, and writes `Prediction` rows within app context. |
| Worker | `api/worker.py` | RQ worker bootstrap; set `ADNS_RQ_QUEUE`, `ADNS_REDIS_URL`, `ADNS_RQ_JOB_TIMEOUT` as needed. Uses the same DB URI as the API (`SQLALCHEMY_DATABASE_URI` default/local). Adds optional reverse-DNS enrichment (`ADNS_RDNS_ENABLED`, timeout/cache tunables) before scoring. |
| Detection engine | `api/model_runner.py`, `api/scoring.py` | Combines lightweight flow pipeline and ExtraTrees/XGBoost meta bundle. Builds synthesized features from `Flow.extra` when packet metadata is sparse. |
| Frontend | `frontend/adns-frontend/` | React/Vite dashboard with timeline, severity donut, and attack simulation buttons (calls `/api/simulate`). Build output served from `dist/` via Nginx. The UI uses optional `VITE_API_URL` to set the API base; defaults to relative `/api/*`. |
| Deployment + ops | `deployment/`, `worker/`, `assets/` | Empty placeholders in the repo; no systemd/nginx assets are checked in. Production services live under `/var/www/adns/...` on the host per the notes below. |
| Data + ML lab | `data/`, `outputs/`, `ml/`, `docs/` | Raw datasets (e.g., UNSW-NB15) and zips sit in `data/` (gitignored). Derived CSVs/models in `outputs/` (gitignored). Preprocessing + training scripts in `ml/`, notebooks/research notes in `docs/`. |

## Key Runtime Details
- **Endpoints**:
  - Flask routes are `/health`, `/ingest`, `/flows`, `/anomalies`, `/simulate`; Nginx maps them to `/api/*` for the frontend.
  - `POST /ingest` (list or single flow) -> writes `Flow` rows, enforces retention, enqueues scoring (with inline fallback on failure).
  - `GET /flows` -> last `MAX_FLOWS` rows ordered oldest-first; falls back to canned demo flows when DB empty.
  - `GET /anomalies` -> simple stats derived from current buffer (count, max score, pct > 0.9) or demo stats.
  - `POST /simulate` -> generates synthetic flows (botnet flood, data exfiltration, port scan) and scores inline with `DetectionEngine`.
- **Database**: DSN defaults to `postgresql://adns:adns_password@127.0.0.1/adns` but can be overridden via `SQLALCHEMY_DATABASE_URI`. Tables: `flows` (timestamp/src/dst/proto/bytes/extra JSON) and `predictions` (flow_id unique, score, label, created_at). `init_db()` creates tables, ensures `flows.extra`, and adds a unique index on `predictions.flow_id`, pruning duplicates if needed.
- **Queues**: Redis defaults to `redis://127.0.0.1:6379/0`. Queue names, batch size, fetch chunk, and timeouts are configurable via env (`ADNS_RQ_QUEUE`, `ADNS_RQ_BATCH_SIZE`, `ADNS_SCORING_FETCH_CHUNK`, `ADNS_RQ_JOB_TIMEOUT`).
- **Agent expectations**: Requires `/usr/bin/tshark`, runs with privileges on `eth0`, posts JSON that already includes inferred service + HTTP/DNS metadata so the API can stash it in `flows.extra`. Tuning is via env (`API_URL`, `INTERFACE`, `BATCH_SIZE`, timing knobs).
- **Reverse DNS feature**: `tasks.score_flow_batch` can optionally add `rdns_exists`/`rdns_hash` to flow extras before scoring, using a cached reverse lookup on the peer IP. Control via `ADNS_RDNS_ENABLED` (default true), `ADNS_RDNS_TIMEOUT_MS`, `ADNS_RDNS_CACHE_TTL`, `ADNS_RDNS_CACHE_SIZE`.
- **Retention**: Controlled by `ADNS_FLOW_RETENTION_MINUTES` (default 30) and `ADNS_FLOW_RETENTION_MAX_ROWS` (default 5000), purged during ingest/simulate paths.

## Dev Commands & Checks
- **API**:
  ```bash
  cd api
  python -m venv .venv && source .venv/bin/activate
  pip install -r requirements.txt
  export FLASK_APP=app.py
  export ADNS_REDIS_URL=${ADNS_REDIS_URL:-redis://127.0.0.1:6379/0}
  export SQLALCHEMY_DATABASE_URI=${SQLALCHEMY_DATABASE_URI:-postgresql://adns:adns_password@127.0.0.1/adns}
  flask run  # serves /health on 5000
  ```
- **Worker**: `source api/.venv/bin/activate && python api/worker.py` (honors `ADNS_REDIS_URL`, `ADNS_RQ_QUEUE`, `ADNS_RQ_JOB_TIMEOUT`; DB URI comes from `SQLALCHEMY_DATABASE_URI` or defaults).
- **Agent**: `cd agent && python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt && sudo ./capture.py`. Override via env (`API_URL`, `INTERFACE`, `BATCH_SIZE`, `POST_INTERVAL`, etc.) when pointing at staging/prod.
- **Frontend**: `cd frontend/adns-frontend && npm install && npm run dev` (hot reload) or `npm run build && npm run preview` for production bundle served via Nginx. `dist/` is deployed to `/root/ADNS/frontend/adns-frontend/dist`. Set `VITE_API_URL` before build if the API isn’t on the same origin.
- **One-shot local setup**: `./scripts/setup_local.sh` creates `.venv`, installs API/agent deps and frontend node_modules, and copies `.env.example` to `.env` if missing.
- **ML**: `cd ml && pip install -r requirements.txt` then run preprocess/train scripts (see README for exact commands). Copy resulting `.joblib` files into `api/model_artifacts/`.
- **Testing**: A `pytest` suite lives in `api/tests/` (run `pip install -r api/requirements-test.txt && cd api && python -m pytest`). It uses a throwaway SQLite DB in heuristic mode — no Redis/PostgreSQL/artifacts needed. CI runs it via `.github/workflows/ci.yml`. For the frontend prefer `npx vitest run`; mock external systems (Redis, PostgreSQL, tshark) in unit tests.

## Operational Notes
- Production services live under `/var/www/adns/api/app.py` (Gunicorn on 127.0.0.1:5000; Nginx proxies `/api/*`) and `/var/www/adns/agent/capture.py`. The frontend bundle is served from `/root/ADNS/frontend/adns-frontend/dist` by Nginx at `http://159.203.105.167/`.
- No deployment assets (systemd/nginx units) are tracked in this repo; manage existing units on the host directly (`adns.service`, `adns-worker.service`, `adns-agent.service` if present).
- Secrets/DSNs live in `.env` (gitignored). Rotate any placeholder passwords before sharing images or demos.
- When changing schema or models, run `init_db()` (or migrate) before restarting Gunicorn/RQ so `/ingest` never sees missing columns. Restart the worker after updating model artifacts so the DetectionEngine reloads them.
- **After any implementation change**: restart the relevant service(s) you touched (e.g., `adns.service`, `adns-worker.service`, `adns-agent.service`, or reload the frontend via Nginx) and update/push the GitHub repo so deployment matches source.
- **Git access**: an SSH key is available for pushes; fingerprint `SHA256:+rRkOHASSedkJHfy85SEJEhjs8k7JnpKYybLWGFGM6A`.
