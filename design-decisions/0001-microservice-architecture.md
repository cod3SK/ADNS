# 0001 — Microservice architecture and data flow

- **Status:** Accepted
- **Phase:** 0 — Foundational architecture

## Context

ADNS demonstrates a complete network anomaly-detection loop: capture packets,
turn them into flows, score the flows with ML, and visualize detections. These
concerns have very different runtime profiles and dependencies:

- Packet capture needs elevated OS privileges and `tshark`, and only runs on Linux.
- Scoring is CPU-bound and depends on heavy ML libraries (NumPy, pandas, xgboost).
- The dashboard is a static single-page app.
- Persistence and HTTP serving have their own scaling characteristics.

Coupling all of this into one process would make the privileged, Linux-only
capture path a hard dependency for running the API or UI, and would force the web
tier to carry the ML dependency footprint.

## Decision

Split the system into independent services that communicate over HTTP and a job
queue:

- **`agent/`** — `tshark` wrapper that normalizes packets into flow JSON and POSTs
  batches to the API.
- **`api/`** — Flask app that persists flows, serves the dashboard feeds
  (`/flows`, `/anomalies`), runs simulations, and enqueues scoring work.
- **worker** (`api/worker.py`) — consumes scoring jobs and writes predictions.
- **`frontend/`** — React/Vite dashboard served by Nginx.
- **PostgreSQL** and **Redis** as backing stores.

The whole stack is wired together with Docker Compose; the privileged agent lives
behind an optional `agent` compose profile so the rest of the stack runs anywhere.

## Consequences

- The API and UI run on macOS/Windows/Linux even though capture is Linux-only.
- Each service has a focused dependency set (the frontend image never pulls xgboost).
- The HTTP/queue contract makes each piece independently testable and replaceable.
- Cost: more moving parts to orchestrate and a richer set of failure modes, which
  motivates the resilience decisions in [0002](0002-async-scoring-redis-rq.md) and
  [0003](0003-three-tier-detection-cascade.md).

## Amendment — Windows desktop packaging collapses the service boundary

The system now ships a second distribution path alongside Docker Compose: a
self-contained Windows installer built with PyInstaller + Inno Setup (see
[0010](0010-windows-desktop-packaging.md)). In that mode the "microservices" are
co-located inside one process: Flask serves both the API and the bundled React
`dist/`, the thread-pool scorer runs in-process, and packet capture is handled by
NFStream capturing directly from the network interface (no external process).

The service decomposition still holds conceptually — the code paths are the same —
but the deployment topology reduces to a single executable with no external
dependencies. The Docker Compose stack remains the reference for Linux/CI
deployments where the boundaries matter operationally.
