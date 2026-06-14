# 0006 — Built-in attack simulation subsystem

- **Status:** Accepted
- **Phase:** 0 — Foundational architecture

## Context

The project's purpose is to *demonstrate* detection, often in a classroom or
interview setting where there is no real malicious traffic to observe and no
appetite for generating actual attacks. The dashboard needs to show the model
reacting to recognizable threat patterns on demand.

## Decision

Add a `POST /simulate` endpoint and matching dashboard controls that synthesize
labeled attack traffic for a fixed catalogue of scenarios: `attack`, `scanning`,
`dos`, `ddos`, and `injection`. Each scenario generates flows with realistic IP
patterns, ports, byte volumes, and service hints (`generate_attack_flows`), scores
them through the live `DetectionEngine`, and persists predictions so the charts
update immediately.

Two modes are supported: a one-shot `count` batch, and a background streaming mode
(`duration_seconds`/`interval_seconds`) that emits batches over time to narrate a
sustained event. Inputs are clamped (count 5–250, duration ≤ 600s) to keep the
demo bounded.

## Consequences

- A presenter can trigger a believable detection story with one click or one cURL.
- Simulated flows pass through the *same* scoring path as real flows, so the demo
  reflects actual model behavior rather than canned results.
- The scenario catalogue is the contract between backend and frontend; the two
  must stay in sync. (A README/code drift here — documenting non-existent
  `botnet_flood`/`data_exfiltration` types — was corrected during Phase 1.)
- Cost: synthetic generators encode assumptions about what each attack "looks
  like," which can bias the demo toward patterns the model already separates well.
