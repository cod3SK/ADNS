# ADNS — Anomaly Detection Network System

[![CI](https://github.com/cod3SK/ADNS/actions/workflows/ci.yml/badge.svg)](https://github.com/cod3SK/ADNS/actions/workflows/ci.yml)

ADNS is an end-to-end network anomaly detection platform. It captures live traffic via [NFStream](https://www.nfstream.org/), scores flows against a trained XGBoost + ExtraTrees ensemble, and visualizes detections on a React dashboard. The primary distribution is a Windows desktop installer — one download, no prerequisites.

## Architecture

| Component | Path | Description |
| --- | --- | --- |
| Live capture agent | `api/app.py` (`_NfstreamCaptureAgent`) | Uses NFStream directly on the selected interface. Flows expire via idle/active timeouts (120 s / 1800 s) — same grain as training. |
| Flask API | `api/` | Persists flows/predictions in SQLite, exposes `/flows`, `/anomalies`, `/simulate`, `/capture/autostart`, `/model_status`, and submits scored flows via in-process thread pool. |
| Thread pool scorer | `api/task_queue.py`, `api/tasks.py` | `ThreadPoolExecutor` (default 2 workers) calls `score_flow_batch` in background threads inside app context. |
| NFStream detection engine | `api/model_runner.py`, `api/serving_nfstream.py` | `NfstreamDetectionEngine` reads 21 contract features from `flow.extra`, calls `NfstreamScorer.score_matrix()`. `validate_matrix()` gates every predict call — no silent column reconciliation. |
| Model artifact | `api/model_artifacts/nfstream_model.joblib` | XGBoost + ExtraTrees E3 pooled ensemble. Trained on UNSW-NB15, Gotham Dataset 2025, CIC-IDS2017 Tuesday. Tracked via Git LFS. |
| Feature contract | `ml/adns_flows/schema.py` | 21 `FEATURE_COLUMNS` — the single source of truth used at both training time and serve time. |
| Corpus + training | `ml/` | `ml/adns_flows/extract_nfstream.py` extracts features from PCAPs; `ml/corpus/build_corpus.py` builds labeled parquets; `ml/train_nfstream.py` trains the pooled model. |
| Frontend dashboard | `frontend/adns-frontend/` | Vite/React UI: Dashboard (charts + metrics), Flows, Flows Manager (anomalous flows + blocked IPs), Settings (capture pipeline + model health). |
| Desktop packaging | `ADNS.spec`, `installer.iss`, `scripts/build_installer.ps1` | PyInstaller bundles Flask + NFStream + model artifact + React build; Inno Setup wraps into installer with silent Npcap bundling. |

## Design decisions

Significant choices are recorded as [Architecture Decision Records](design-decisions/) (ADRs):

- **[Async scoring with in-process thread pool](design-decisions/0002-async-scoring-redis-rq.md)** — ingestion submits flow IDs to a `ThreadPoolExecutor`; no external queue dependency.
- **[NFStream detection engine](design-decisions/0003-three-tier-detection-cascade.md)** — single `NfstreamDetectionEngine`; absent model blocks `/capture/autostart` with HTTP 503.
- **[Persistence and retention](design-decisions/0004-postgres-persistence-and-retention.md)** — SQLite (substitutable with PostgreSQL), self-healing schema, automatic pruning.
- **[NFStream feature contract](design-decisions/0005-feature-synthesis-for-sparse-telemetry.md)** — 21 `FEATURE_COLUMNS` extracted identically at corpus-build and serve time; `validate_matrix()` replaces silent reconciliation.
- **[Attack simulation subsystem](design-decisions/0006-attack-simulation-subsystem.md)** — `POST /simulate` drives synthetic threat scenarios through the scoring path for demos.
- **[Fail-closed admin-token gate](design-decisions/0007-admin-token-gate-for-response-actions.md)** — `/block_ip` and `/unblock_ip` require `ADNS_ADMIN_TOKEN`; fail closed by default.
- **[Externalized configuration](design-decisions/0008-externalized-configuration-and-secrets.md)** — credentials and knobs from environment; demo-only defaults.
- **[Test strategy and CI](design-decisions/0009-test-strategy-and-ci.md)** — 141 tests across ML suite and API; runs against real model via pytest without external services.
- **[Windows desktop packaging](design-decisions/0010-windows-desktop-packaging.md)** — PyInstaller + Inno Setup; NFStream + Npcap bundled; single-exe install.
- **[Tabbed left-nav layout](design-decisions/0011-tabbed-navigation-layout.md)** — four-tab nav rail separating visualization, data browsing, active response, and pipeline controls.
- **[Installer versioning](design-decisions/0012-installer-versioning-and-update-safety.md)** — fixed `AppId` GUID, version wired through `iscc /D`, `CloseApplications=yes` for clean updates.

See the [**model card**](ml/model_card.md) for training data, metrics, and limitations.

## Desktop App — Windows (recommended)

If you just want to run ADNS — no Python, Node.js, or Docker needed:

1. Go to the [Releases page](https://github.com/cod3SK/ADNS/releases) and download **`ADNS_Installer_v*.exe`**.
2. Run the installer and click **Next** through the wizard. Installs to your personal user folder — no administrator password required for the install step.
3. If [Npcap](https://npcap.com) is not already on your machine the installer will install it silently.
4. Click **Launch ADNS now** (or the desktop shortcut).

The app opens in its own window — Flask API, React UI, NFStream, and ML model are all bundled. Data is saved in `%AppData%\ADNS\adns.db`.

> First launch may take a few seconds while the detection engine initializes — this is normal.

### Building the installer yourself

| Requirement | Notes |
|---|---|
| [Node.js 18+](https://nodejs.org) | Accept all defaults |
| [Python 3.10+](https://python.org/downloads) | Check "Add Python to PATH" |
| [Inno Setup 6](https://jrsoftware.org/isinfo.php) | Accept all defaults |
| `npcap-installer.exe` in repo root | Download from [npcap.com](https://npcap.com), rename, drop in repo root |

```powershell
git lfs pull                          # fetch nfstream_model.joblib (~102 MB)
pip install -r requirements-desktop.txt pyinstaller
pwsh scripts\build_installer.ps1
```

The finished installer is written to `Output\ADNS_Installer_v*.exe`. The build script handles `npm run build`, PyInstaller, and Inno Setup in sequence.

## Dev setup (Flask API + React frontend)

```powershell
# 1. Python deps
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r api/requirements.txt

# 2. Run the API (from repo root)
cd api
flask run          # http://127.0.0.1:5000

# 3. Frontend dev server (separate terminal)
cd frontend/adns-frontend
npm install
npm run dev        # http://localhost:5173
```

Requires `api/model_artifacts/nfstream_model.joblib` — run `git lfs pull` after clone or train it yourself (see Training below). Without the model, `/capture/autostart` returns HTTP 503.

## Training & corpus pipeline

```powershell
# Raw PCAPs required (local, not in repo).
# UNSW-NB15 (two collection days):
python ml/run_unsw_day1_nfstream.py   # → outputs/corpus/unsw_day1.parquet
python ml/run_unsw_day2_nfstream.py   # → outputs/corpus/unsw_day2.parquet
python ml/combine_unsw_nfstream.py    # → outputs/corpus/unsw_flows.parquet

# Gotham Dataset 2025:
python -m corpus.build_corpus gotham  # → outputs/corpus/gotham_flows.parquet

# CIC-IDS2017 Tuesday:
python -m corpus.build_corpus cic     # → outputs/corpus/cic_tuesday_flows.parquet

# Train E3 pooled model (reads all three parquets above):
python ml/train_nfstream.py           # → api/model_artifacts/nfstream_model.joblib
```

Corpus parquets (~120 MB total) are not in git — reproducible from raw PCAPs (~15 GB, not redistributable). See `CLAUDE.md` §5 for dataset-specific details.

## Demo

Inject synthetic attack traffic via the `/simulate` endpoint:

```bash
curl -X POST http://localhost:5000/simulate \
     -H 'Content-Type: application/json' \
     -d '{"type":"scanning","count":40}'
```

Supported types: `attack`, `scanning`, `dos`, `ddos`, `injection`.

> **Note:** `/simulate` generates flows with synthetic feature values and is useful for UI demonstrations. Real live-capture flows (scored by NFStream) produce accurate anomaly scores; simulated flows use a simplified feature format.

## Testing

```powershell
python -m pytest api/tests/ ml/ -q
# 141 passed (131 ML suite + 10 API absent-model tests)
```

The full suite runs without external services. ML tests cover feature contract parity, corpus labeling, orientation invariance, and grain-parity. API tests cover endpoint behavior and the absent-model loud-failure path.

CI (`.github/workflows/ci.yml`) runs tests and the frontend lint/build on every push.

## Security notes

- **Killswitch** (`POST /killswitch`) drops all non-loopback traffic via `iptables` (Linux) or `netsh advfirewall` (Windows). Ungated — intended as an immediate first-responder action from the dashboard. Requires `NET_ADMIN` (Linux) or Administrator (Windows). Only affects the machine the API runs on directly.
- **Block/unblock IP** require `ADNS_ADMIN_TOKEN` to be set and a matching `Authorization: Bearer <token>` or `X-Admin-Token: <token>` header. Returns HTTP 403 without a token — fail closed by default.
- **Absent model** blocks `/capture/autostart` with HTTP 503 — no silent no-scoring. Run `git lfs pull` or `ml/train_nfstream.py` to resolve.
- Database credentials come from the environment (`SQLALCHEMY_DATABASE_URI`). The SQLite default (`sqlite:///./adns.db`) is fine for local use; set a real URI before any shared deployment.

## Contributing

- Python code follows PEP 8; React follows the Vite ESLint defaults.
- Tests live in `api/tests/` and `ml/adns_flows/tests/` — add tests near the subsystem you touch.
- Keep secrets in `.env` (gitignored). Add large/generated directories to `.gitignore`.
- Use short imperative commit messages; include metrics when changing ML behavior.
