# 0006 — Attack simulation subsystem (CLI, not UI)

- **Status:** Revised
- **Phase:** 0 — Foundational architecture; UI controls removed in Phase 2

## Context

The project's purpose is to *demonstrate* detection, often in a classroom or
interview setting where there is no real malicious traffic to observe and no
appetite for generating actual attacks. The system needs to show the model
reacting to recognizable threat patterns on demand.

## Decision

A `POST /simulate` endpoint synthesizes labeled attack traffic for five
scenarios — `attack`, `scanning`, `dos`, `ddos`, `injection` — scores flows
through the live `NfstreamDetectionEngine`, and persists predictions. Two modes: a
one-shot `count` batch, and a background streaming mode (`duration_seconds` /
`interval_seconds`). Inputs are clamped (count 5–250, duration ≤ 600s).

The dashboard previously exposed these as clickable buttons, which were
**removed** (Phase 2) because they made the dashboard feel like a toy and
caused out-of-memory conditions when multiple streaming sessions were started
accidentally. Instead, `core/attack_generator.py` is a stdlib-only CLI that
POSTs flow batches directly to `/ingest`, running the same detection path.

The streaming OOM risk was also fixed at the API level: a module-level lock
(`_STREAM_LOCK`) prevents more than one concurrent streaming thread; the
endpoint returns 409 if one is already running. Session identity-map references
are also released with `db.session.expunge_all()` after each batch.

## Consequences

- The dashboard is view-only: charts, severity mix, block/killswitch controls.
- A presenter runs `python core/attack_generator.py --type ddos --duration 120`
  from the terminal; flows show up on the dashboard in real time.
- Flows from the CLI pass through the *same* `/ingest` → scoring path as the
  capture agent, so the demo reflects actual model behavior.
- Cost: losing the one-click UI adds a terminal step for demos. Trade-off
  accepted: the dashboard is cleaner and the OOM risk is gone.

## Known limitation — simulate scores 0.0 post-NFStream-migration

The `generate_attack_flows()` function in `api/app.py` generates synthetic `Flow`
objects whose `extra` dict uses a simplified format (no `_extractor: nfstream`
marker and no 21-column NFStream feature contract). `extra_to_feature_vector()`
returns `None` for these flows, causing `score_many()` to return `(0.0, "normal")`
for all simulated flows.

Real live-captured flows (scored by `_NfstreamCaptureAgent`) carry the full
21-column contract and score correctly.

Fix path: update `generate_attack_flows()` to emit flows with realistic NFStream
contract features (all 21 `FEATURE_COLUMNS` populated in `flow.extra`, with
`_extractor: "nfstream"`).
