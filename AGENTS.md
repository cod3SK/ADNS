# ADNS Agent Brief

Use this page to remind yourself (or other assistants) what lives where in the ADNS stack and how the pieces talk to each other.

## Mission Snapshot
- **Goal**: Demonstrate a modern network anomaly detection loop end to end (capture → ingest → score → visualize) with synthetic attack simulations for workshops and portfolio review.
- **Live topology (desktop installer)**: `launcher.py` starts Flask in a background thread via Werkzeug. Flask serves both the API and the bundled React `dist/`. An in-process `ThreadPoolExecutor` handles async scoring. Packet capture runs as two embedded agents (`_CaptureAgent`, `_BatchCaptureAgent`) launched automatically on startup via `/capture/autostart`.
- **Storage**: SQLite at `%APPDATA%\ADNS\adns.db` (desktop); PostgreSQL via `SQLALCHEMY_DATABASE_URI` (Docker/server). Tables: `flows` + `predictions`. Retention trims flows older than `ADNS_FLOW_RETENTION_MINUTES` or beyond `ADNS_FLOW_RETENTION_MAX_ROWS`.
- **Models**: `api/model_runner.py` loads `meta_model_combined.joblib` (ExtraTrees + XGBoost ensemble) then `flow_detector.joblib` (calibrated LogisticRegression fallback), falling back to heuristics if artifacts are absent.

## Component Reference
| Piece | Path | Notes |
| --- | --- | --- |
| Live capture agent | `agent/capture.py` | Wraps `tshark` (fields in `TSHARK_FIELDS`), infers ports/services, batches ~50 flows or ~2 s, retries POSTs with backoff. Settings read from env (`API_URL`, `TSHARK_BIN`, `INTERFACE`, `BATCH_SIZE`, `POST_INTERVAL`, `RETRY_DELAY`) with defaults. In the desktop installer this runs as `_CaptureAgent` embedded inside the API process. |
| Batch capture agent | `agent/batch_capture.py` | Standalone script for server/dev use. In the desktop installer the same logic runs as `_BatchCaptureAgent` embedded inside the API process (started automatically on launch). Env: `INTERFACE`, `TSHARK_BIN`, `BATCH_WINDOW_SECONDS` (default 15), `BATCH_DIR`, `BATCH_API_URL`. Each 15-second window produces one pcap; two-pass tshark processing extracts flows. **Requires tshark 4.x**: conv output has no pipe chars in data rows and uses human-readable byte units (`85 kB`, `1530 bytes`); `_BATCH_CONV_RE` and `_parse_tshark_bytes` in `app.py` handle this. |
| API | `api/app.py` | Flask + SQLAlchemy. Key routes: `/health`, `/ingest`, `/ingest_batch`, `/flows`, `/anomalies`, `/anomalous_flows`, `/simulate`, `/batch_summary`, `/capture_status`, `/interfaces`, `/capture/autostart`, `/block_ip`, `/unblock_ip`, `/killswitch`, `/model_status`. `_BatchCaptureAgent` and `_CaptureAgent` classes run in-process; auto-started by the launcher. |
| Task queue | `api/task_queue.py`, `api/tasks.py` | In-process `ThreadPoolExecutor` (default 2 workers, `ADNS_SCORER_WORKERS`). `enqueue_flow_scoring(flow_ids)` splits IDs into 100-ID chunks and submits each to the pool. `score_flow_batch` in `tasks.py` runs inside an explicit Flask app context, optionally enriches with reverse-DNS, and upserts `Prediction` rows. No Redis or external queue required. |
| Detection engine | `api/model_runner.py`, `api/scoring.py` | Three-tier cascade: MetaEnsembleModel (ExtraTrees+XGBoost, `meta_model_combined.joblib`) → FlowModel (calibrated LogisticRegression, `flow_detector.joblib`) → heuristic FlowScorer. Hot-reloads on artifact mtime change. Feature synthesis in `MetaFeatureBuilder` fills the ~46-column vector from sparse live tshark telemetry. |
| Frontend | `frontend/adns-frontend/` | React/Vite dashboard with five tabs: **Dashboard** (metric cards + four charts), **Flows** (filterable flow table with per-row block action), **Flows Manager** (anomalous flows + blocked IPs), **Batch Analysis** (15-min pcap-based summaries), **Settings** (capture pipeline status, model health). Kill switch stays in the top header. Build output served from `dist/` (embedded in the desktop bundle). |
| Desktop launcher | `launcher.py` | PyInstaller entry point. Elevates to admin if needed, installs Npcap if absent, starts Flask in a background thread, waits for `/health`, then opens a pywebview window. Tray icon: left-click to show, right-click → Quit runs `os._exit(0)` after stopping tshark. |
| Desktop build | `ADNS.spec`, `installer.iss`, `scripts/build_installer.ps1` | PyInstaller spec bundles Flask, React dist, ML models, tshark binaries, and all Python deps (including `collect_all("xgboost")` + explicit native DLL). Inno Setup wraps to a single installer. Build script auto-kills any running ADNS process before PyInstaller runs so `dist/ADNS/` is not locked. Run: `pwsh scripts\build_installer.ps1` (or with `-Version X.Y.Z`). |
| Data + ML lab | `data/`, `outputs/`, `ml/`, `docs/` | Raw datasets (e.g., UNSW-NB15, TON_IoT) in `data/` (gitignored). Derived CSVs/models in `outputs/` (gitignored). Preprocessing + training scripts in `ml/`, notebooks in `docs/`. |

## Key Runtime Details
- **Endpoints** (Flask routes; in desktop bundle the `/api` prefix is stripped by `_StripApiPrefix` WSGI middleware):
  - `POST /ingest` — ingest live flow JSON, enforce retention, enqueue scoring
  - `POST /ingest_batch` — ingest batch-capture flows (`source='batch'`), enforce 65-min retention
  - `GET /flows` — last `MAX_FLOWS=400` live flows, oldest-first
  - `GET /anomalies` — aggregate stats over live buffer
  - `GET /anomalous_flows` — live flows where label ≠ normal or score ≥ 0.6
  - `GET /batch_summary?window=10m|15m|1h` — total_flows, total_bytes, anomaly_count, proto_breakdown, top IPs, timeseries
  - `POST /simulate` — generate + score synthetic attack flows inline (types: attack/scanning/dos/ddos/injection)
  - `POST /capture/autostart` — detect default-route interface, start both capture agents
  - `GET /capture_status` — interface, tshark_found, live/batch agent status (running, batches, last ingest, error)
  - `GET /model_status` — probe each ML estimator with a dummy prediction; reports ok/broken/absent
  - `POST /block_ip` — OS-level block (requires `ADNS_ADMIN_TOKEN`)
  - `POST /killswitch` — drop all non-loopback traffic; ungated
- **Database**: DSN defaults to `postgresql://adns:adns_password@127.0.0.1/adns`; SQLite for dev/desktop. `init_db()` creates tables and runs in-code migrations (adds `flows.extra` column, deduplicates `predictions.flow_id`).
- **Scoring async**: `ThreadPoolExecutor` — no Redis/RQ. Jobs are in-process; do not survive a restart. Configurable via `ADNS_SCORER_WORKERS` (default 2) and `ADNS_SCORING_BATCH_SIZE` (default 100).
- **Reverse DNS**: `tasks.score_flow_batch` optionally enriches flows with `rdns_exists`/`rdns_hash`. Control via `ADNS_RDNS_ENABLED` (default true in server; set to false in desktop launcher), `ADNS_RDNS_TIMEOUT_MS`, `ADNS_RDNS_CACHE_TTL`, `ADNS_RDNS_CACHE_SIZE`.
- **Retention**: Live flows controlled by `ADNS_FLOW_RETENTION_MINUTES` (30) and `ADNS_FLOW_RETENTION_MAX_ROWS` (5000). Batch flows have a separate 65-min retention (`ADNS_BATCH_FLOW_RETENTION_MINUTES`).
- **tshark version**: The batch conv parser (`_BATCH_CONV_RE`) targets tshark 4.x format — no pipes in data rows, human-readable byte units. The bundled tshark in the installer is from Wireshark 4.x at build time.

## Dev Commands (Windows)

**Start Flask dev server** (from `X:\ADNS\api\`):
```powershell
& "C:\Users\ruzha\AppData\Local\Programs\Python\Python312\python.exe" -m flask run
```

**Start frontend dev server** (from `X:\ADNS\frontend\adns-frontend\`):
```powershell
npm run dev   # http://localhost:5173 — Vite proxy rewrites /api/* → http://127.0.0.1:5000/*
```

**Run API tests** (from `X:\ADNS\api\`):
```powershell
& "C:\Users\ruzha\AppData\Local\Programs\Python\Python312\python.exe" -m pytest
```
Uses throwaway SQLite + heuristic scorer — no PostgreSQL, Redis, or ML artifacts needed.

**Run the desktop bundle directly** (after building):
```
X:\ADNS\dist\ADNS\ADNS.exe
```

**Build installer**:
```powershell
pwsh scripts\build_installer.ps1              # auto-increments patch version
pwsh scripts\build_installer.ps1 -Version 1.2.3  # explicit version (not yet wired; set VERSION file manually)
```
Output: `X:\ADNS\Output\ADNS_Installer_v<version>.exe`

**Live tshark capture agent** (real traffic, must run as Administrator):
```powershell
$env:INTERFACE = "Wi-Fi"
$env:API_URL = "http://127.0.0.1:5000/ingest"
& "C:\Users\ruzha\AppData\Local\Programs\Python\Python312\python.exe" X:\ADNS\agent\capture.py
```

**Attack simulation** (stdlib CLI, no Flask deps):
```powershell
& "C:\Users\ruzha\AppData\Local\Programs\Python\Python312\python.exe" X:\ADNS\core\attack_generator.py --type ddos --count 80
& "C:\Users\ruzha\AppData\Local\Programs\Python\Python312\python.exe" X:\ADNS\core\attack_generator.py --type injection --duration 120 --interval 1
```
Supported types: `attack`, `scanning`, `dos`, `ddos`, `injection`.

## Operational Notes
- In the desktop bundle, the database lives at `%APPDATA%\ADNS\adns.db` and persists across reinstalls.
- The app self-elevates to admin on startup (required for raw-socket capture and firewall rules via Npcap).
- Secrets/DSNs live in `.env` (gitignored). Rotate placeholder passwords before sharing.
- No `api/worker.py` — the RQ worker was removed when scoring moved to `ThreadPoolExecutor` (see ADR-0002).
- **Git**: repo is at `github.com/OffensiveGeneric/ADNS`.
