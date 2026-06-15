# ADNS — Anomaly Detection Network System

[![CI](https://github.com/OffensiveGeneric/ADNS/actions/workflows/ci.yml/badge.svg)](https://github.com/OffensiveGeneric/ADNS/actions/workflows/ci.yml)

ADNS is an end-to-end demo of a modern network anomaly detection platform. It ingests live packet captures, stores recent flows in PostgreSQL, scores them asynchronously via an in-process thread pool with a DetectionEngine (meta ensemble → sklearn → heuristics), and visualizes detections on a React dashboard. Attack scenarios are driven from the CLI tool in `core/attack_generator.py`, not the dashboard.

## Architecture
<img width="1024" height="559" alt="image" src="https://github.com/user-attachments/assets/3c972f97-f751-4c92-9d10-fb54f326c4b3" />


| Component | Path | Description |
| --- | --- | --- |
| Packet capture agent | `agent/` | `capture.py` wraps `tshark`, normalizes packet metadata into flow JSON, and POSTs batches to `/api/ingest`. |
| Flask API | `api/` | Persists flows/predictions, exposes `/flows`, `/anomalies`, `/simulate`, and submits flow IDs to an in-process thread pool for scoring. |
| Thread pool scorer | `api/task_queue.py`, `api/tasks.py` | `ThreadPoolExecutor` that runs `score_flow_batch` in background threads inside app context. |
| Frontend dashboard | `frontend/adns-frontend/` | Vite/React UI with anomaly charts and severity donut. |
| ML lab | `ml/` | Preprocessing scripts (`preprocess/`), meta-model notebooks, and `train_flow_detector.py` for the live scorer. |
| Model artifacts | `api/model_artifacts/` | `meta_model_combined.joblib` (ExtraTrees+XGBoost) + `flow_detector.joblib` (sklearn pipeline). |
| Attack generator | `core/attack_generator.py` | Stdlib-only CLI; generates synthetic attack flows and POSTs them to `/ingest` for live demo runs. |
| Ops | `deployment/`, `worker/`, `assets/` | Systemd units, scripts, and misc assets. Research docs live in `docs/`. |

Generated datasets live under `data/`, and derived artifacts (clean CSVs, model outputs) live under `outputs/`; both are gitignored to keep the repo lean.

## Design decisions

The significant design choices are recorded as [Architecture Decision Records](design-decisions/)
(ADRs). In brief:

- **[Microservice architecture](design-decisions/0001-microservice-architecture.md)** — capture, API, worker, and UI are split so the privileged Linux-only agent never blocks running the rest of the stack, and each service carries only its own dependencies.
- **[Async scoring with in-process thread pool](design-decisions/0002-async-scoring-redis-rq.md)** — ingestion submits flow IDs to a `ThreadPoolExecutor` and returns immediately; no external queue dependency.
- **[Three-tier detection cascade](design-decisions/0003-three-tier-detection-cascade.md)** — meta ensemble → calibrated sklearn → rule-based heuristic, with hot model reload; the system always produces a score regardless of what is installed.
- **[Persistence, in-code schema management, and retention](design-decisions/0004-postgres-persistence-and-retention.md)** — PostgreSQL (SQLite-substitutable), self-healing schema migrations on startup, and automatic pruning to stay bounded.
- **[Feature synthesis for sparse telemetry](design-decisions/0005-feature-synthesis-for-sparse-telemetry.md)** — live `tshark` data is estimated/hashed into the model's full feature vector; documents the resulting train/serve skew honestly.
- **[Attack simulation subsystem](design-decisions/0006-attack-simulation-subsystem.md)** — `POST /simulate` drives believable threat scenarios through the real scoring path for demos.
- **[Fail-closed admin-token gate](design-decisions/0007-admin-token-gate-for-response-actions.md)** — `/block_ip` and `/unblock_ip` are disabled unless `ADNS_ADMIN_TOKEN` is set, then require a bearer/header token. `/killswitch` is ungated so it works immediately from the dashboard.
- **[Externalized configuration and secrets](design-decisions/0008-externalized-configuration-and-secrets.md)** — credentials come from the environment with demo-only defaults; no real secret lives in source.
- **[Test strategy and CI](design-decisions/0009-test-strategy-and-ci.md)** — tests run against the heuristic + SQLite paths, so the full suite is fast, dependency-light, and runs in CI on every push.

See also the [**model card**](ml/model_card.md) for the detectors' training data, metrics, and limitations.

## Desktop App — Windows (No Setup Required)

If you just want to open ADNS and see it working — no Python, Node.js, or Docker needed:

1. Go to the [Releases page](https://github.com/OffensiveGeneric/ADNS/releases) and download **`ADNS_installer.exe`**.
2. Run the installer and click **Next** through the wizard. No administrator password is required — it installs to your personal user folder.
3. When the wizard finishes, click **Launch ADNS now** (or double-click the desktop shortcut any time after that).

The app opens in its own window with everything running inside it. Your data is saved in `%AppData%\ADNS\adns.db` — uninstalling the app leaves that file in place so you don't lose history.

> The first launch may take a few seconds while the detection engine loads — this is normal.

### Building the installer yourself (developers only)

You will need three free tools installed first:

- [Node.js 18+](https://nodejs.org) — download and run the installer, accept all defaults
- [Python 3.10+](https://python.org/downloads) — download and run the installer; **check "Add Python to PATH"** on the first screen
- [Inno Setup 6](https://jrsoftware.org/isinfo.php) — download and run the installer, accept all defaults

Then open PowerShell in the repo root and run:

```powershell
pip install -r requirements-desktop.txt pyinstaller
pwsh scripts\build_installer.ps1
```

The finished installer is written to `Output\ADNS_installer.exe`. The GitHub Actions workflow (`.github/workflows/build-installer.yml`) runs the same steps automatically whenever a version tag is pushed and attaches the result to the GitHub Release.

## Quickstart — Docker first
Prereqs: Docker + Docker Compose, Git.

```bash
git clone https://github.com/OffensiveGeneric/ADNS.git
cd ADNS
docker compose up --build -d          # API:5000, Frontend:8080, Postgres
```

- Frontend: `http://localhost:8080`
- API health: `curl http://localhost:5000/health`
- Demo traffic: `curl -X POST http://localhost:5000/simulate -H 'Content-Type: application/json' -d '{"type":"ddos","count":50}'`
- Streaming demo traffic (background): `curl -X POST http://localhost:5000/simulate -H 'Content-Type: application/json' -d '{"type":"ddos","duration_seconds":120,"interval_seconds":1}'`
- Supported simulation `type` values: `attack`, `scanning`, `dos`, `ddos`, `injection`.
- Live capture (Linux only): `docker compose --profile agent up -d agent` (uses host network + NET_ADMIN; set `INTERFACE`/`API_URL` in `docker-compose.yml` if needed).

### Local dev (bare metal, optional)
If you prefer running services directly:
- macOS/Linux: `./scripts/setup_local.sh` then start API/agent/frontend with the commands in `AGENTS.md`.
- Windows: `pwsh ./scripts/setup_local.ps1` then use the PowerShell commands in `AGENTS.md`.

Databases:
- Postgres default: `SQLALCHEMY_DATABASE_URI=postgresql://adns:adns_password@127.0.0.1/adns`
- SQLite (no install): set `SQLALCHEMY_DATABASE_URI=sqlite:///./adns.db` in `.env`

## Docker Compose (dev stack)
- Build and run API, frontend, and Postgres: `docker compose up --build` (from repo root). API on `http://localhost:5000`, frontend on `http://localhost:8080`.
- Frontend build arg: override `VITE_API_URL` if you want a different API origin (default `http://localhost:5000`); e.g., `docker compose build --build-arg VITE_API_URL=http://api:5000 frontend`.
- Optional capture agent: `docker compose --profile agent up --build agent` (Linux only, uses `network_mode: host` and `NET_ADMIN` so `tshark` can see host traffic). On macOS/Windows, run the agent on the host instead and point `API_URL` at `http://localhost:5000/ingest`.
- Persistent Postgres data lives in the `pgdata` volume; remove it with `docker volume rm adns_pgdata` if you need a clean slate.
- Common fixes:
  - macOS AirPlay can own port 5000; if `curl localhost:5000/health` returns 403 AirTunes, change the API port mapping (e.g., `5100:5000`), restart compose, and point agent/frontend at the new port.
  - If the UI cannot reach the API, rebuild the frontend with the right base: `docker compose build --no-cache --build-arg VITE_API_URL=http://127.0.0.1:5000 frontend && docker compose up -d frontend` (or `VITE_API_URL=""` to use the nginx `/api` proxy). Verify with `curl http://localhost:8080/api/health`.

### Run locally to monitor your own traffic

1) Install system deps: PostgreSQL (or use SQLite via `SQLALCHEMY_DATABASE_URI=sqlite:///./adns.db`), `tshark`, Python 3.9+, Node.js 18+.  
2) Bootstrap the repo: `./scripts/setup_local.sh` on macOS/Linux or `pwsh ./scripts/setup_local.ps1` on Windows (creates `.venv`, installs API+agent deps, runs `npm install`, and copies `.env.example` to `.env` if missing).  
3) Edit `.env` as needed:
   - Want Postgres? Install it, run `./scripts/setup_postgres_local.sh` (or `pwsh ./scripts/setup_postgres_local.ps1` on Windows) to create the `adns` database/user, then set `SQLALCHEMY_DATABASE_URI` to the printed URL.
   - `SQLALCHEMY_DATABASE_URI` can be set to `sqlite:///./adns.db` for a zero-install database.
   - `VITE_API_URL` only if the frontend will call the API on a different origin.
   - `API_URL`, `INTERFACE`, `BATCH_SIZE`, etc. to control the capture agent.
   - `ADNS_RDNS_ENABLED` and related knobs to include reverse-DNS resolution as a scoring feature.
4) Run services (separate terminals):
   - API: `source .venv/bin/activate && export $(grep -v '^#' .env | xargs) && cd api && flask run`
   - Agent (needs tshark + capture privileges): `source .venv/bin/activate && export $(grep -v '^#' .env | xargs) && cd agent && sudo ./capture.py`
   - Frontend: `cd frontend/adns-frontend && export $(grep -v '^#' ../../.env | xargs) && npm run dev -- --host`
   - On Windows/PowerShell: use `.\.venv\Scripts\Activate.ps1` instead of `source ...`, drop `sudo`, and run the agent from an elevated shell so `tshark` can capture.
   - On WSL with Docker Desktop: `sudo apt-get install -y tshark` inside WSL, then run the agent with explicit paths: `API_URL=http://127.0.0.1:5000/ingest TSHARK_BIN=/usr/bin/tshark INTERFACE=eth0 sudo .venv/bin/python agent/capture.py`. If tshark permissions are already set, drop `sudo`.
   - On macOS with Docker: avoid AirPlay port 5000 conflicts by using the mapped API port (e.g., 5001). Run the agent with preserved envs: `sudo env TSHARK_BIN=/usr/local/bin/tshark API_URL=http://127.0.0.1:5001/ingest INTERFACE=en0 .venv/bin/python agent/capture.py`. If you’ve run Wireshark’s ChmodBPF helper, you can omit `sudo`.

### 0. Dependencies

- PostgreSQL (default URL `postgresql://adns:adns_password@127.0.0.1/adns`) — or SQLite via `SQLALCHEMY_DATABASE_URI=sqlite:///./adns.db`
- `tshark` on any host that runs the capture agent

The commands below assume those services are already running.

### 1. Backend / API

```bash
cd api
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export FLASK_APP=app.py
export SQLALCHEMY_DATABASE_URI=${SQLALCHEMY_DATABASE_URI:-postgresql://adns:adns_password@127.0.0.1/adns}
flask run
```

The API exposes:

- `POST /ingest` — ingest flow JSON (single object or list).
- `GET /flows` & `GET /anomalies` — dashboard data feeds.
- `POST /simulate` — synthesize attack traffic (used by the UI buttons).
  - Accepts `count` for one-shot batches.
  - Accepts `duration_seconds` (and optional `interval_seconds`, default 1.0s) to stream batches in the background for the given duration.

On first run `init_db()` creates tables and adds the `flows.extra` JSON column so the agent’s rich metadata can be stored immediately.

### 2. Packet capture agent

```bash
cd agent
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # only pulls requests
export API_URL=${API_URL:-http://127.0.0.1:5000/ingest}
sudo ./capture.py                 # needs privileges for the interface
```

The agent wraps `tshark`, infers services, batches ~50 flows or 2 seconds, and POSTs them to the API. Production deployments run it under `systemd` (`adns-agent.service`) so it survives reboots.

### 3. Frontend

```bash
cd frontend/adns-frontend
npm install
npm run dev   # for hot reload
npm run build && npm run preview   # for production bundle
```

Building places static assets under `dist/`. Set `VITE_API_URL` before `npm run build` if the UI is hosted separately; the production droplet serves that folder via Nginx at `http://159.203.105.167/`.

### 4. Training & Data Pipelines

```bash
cd ml
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Preprocess UNSW-NB15 CSVs into features
python preprocess/merge_and_clean.py \
  --data_dir ../data/DataSet/UNSW-NB15/"Training and Testing Sets" \
  --out_dir ../outputs/preprocessed

# Train meta models (ExtraTrees + XGBoost ensemble)
python meta/meta_train.py --raw_train ... --raw_test ... --clean_dir ../outputs/preprocessed --out_dir ../outputs/meta

# Train the lightweight flow detector used in production
python train_flow_detector.py \
  --raw_train ../data/DataSet/UNSW-NB15/"Training and Testing Sets"/UNSW_NB15_training-set.csv \
  --raw_test  ../data/DataSet/UNSW-NB15/"Training and Testing Sets"/UNSW_NB15_testing-set.csv \
  --model_out ../api/model_artifacts/flow_detector.joblib
```

Copy the resulting artifacts (both `flow_detector.joblib` and `meta_model_combined.joblib`) into `/var/www/adns/api/model_artifacts/` (or wherever Gunicorn runs) and restart the API so the DetectionEngine reloads them.

## Demo Tips

- The **Threat Timeline** and **Severity Mix** donut help narrate how the model responds as traffic changes.
- To inject synthetic attack traffic, use the CLI tool in `core/attack_generator.py` (requires only stdlib — no Flask deps):

```bash
# One-shot batch of 80 DDoS flows
python core/attack_generator.py --type ddos --count 80

# Stream injection flows for 2 minutes
python core/attack_generator.py --type injection --duration 120 --interval 1

# Supported types: attack, scanning, dos, ddos, injection
```

- `POST /ingest` (which the generator targets) can also be called directly via cURL:

```bash
# Minimal single-flow ingest
curl -X POST http://localhost:5000/ingest -H 'Content-Type: application/json' \
     -d '[{"src_ip":"10.0.0.1","dst_ip":"8.8.8.8","proto":"TCP","bytes":50000}]'
```

## Testing

The API ships with a `pytest` suite that runs against a throwaway SQLite database
in heuristic scoring mode — no PostgreSQL or ML artifacts required:

```bash
cd api
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-test.txt
python -m pytest
```

CI (GitHub Actions, `.github/workflows/ci.yml`) runs these tests plus the frontend
lint/build on every push and pull request.

## Security notes

- **Killswitch** (`POST /killswitch`) is intentionally ungated — it is a
  first-responder action that must work from the dashboard without configuration.
  When triggered it drops all non-loopback traffic via `iptables ! -o/-i lo` on
  Linux, or `netsh advfirewall` block-all rules on Windows, preserving localhost
  so the monitoring stack stays reachable. Requires `NET_ADMIN` (Linux) or an
  Administrator process (Windows).
  **Deployment constraint:** the killswitch only affects the machine the API
  process runs on directly. When the API runs inside Docker on Windows, `sys.platform`
  is `linux` (container OS), so the `netsh` Windows path is never taken; the
  iptables path applies rules inside the container's own network namespace, not
  on the Windows host. Docker Desktop on Windows also uses a WSL2 Linux VM as its
  host layer, so even escaping the container via `nsenter` would only reach the VM,
  not the Windows machine's actual network adapters. The killswitch works as
  intended on a **native Linux deployment** where the API runs directly on the
  host.
- **Block/unblock IP** (`/block_ip`, `/unblock_ip`) require `ADNS_ADMIN_TOKEN` to
  be set **and** the caller to send a matching `Authorization: Bearer <token>` (or
  `X-Admin-Token: <token>`). Without a token those endpoints return HTTP 403 —
  fail closed by default.
- Database credentials are read from the environment (`POSTGRES_USER`,
  `POSTGRES_PASSWORD`, `POSTGRES_DB`). The committed defaults are for local demos
  only — set real values in `.env` before any non-local deployment.

## Contributing

- Python code follows PEP 8; React follows the Vite ESLint defaults.
- Add tests near the subsystem you touch (`api/tests` exists; add `ml/tests`, `frontend/.../__tests__` as needed).
- Keep secrets in `.env` (already gitignored), and add any new large/generated directories to `.gitignore`.
- Use short imperative commit messages and include screenshots or metrics when changing UI/ML behavior.

Questions? See `AGENTS.md` for contributor guidelines or open an issue. Happy hunting!
