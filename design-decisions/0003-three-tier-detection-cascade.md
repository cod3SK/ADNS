# 0003 — Three-tier detection cascade with hot reload

- **Status:** Accepted
- **Phase:** 0 — Foundational architecture

## Context

The detector needs to run in very different environments: a fully provisioned
deployment with trained model artifacts and xgboost installed, a lighter
deployment with only the legacy sklearn pipeline, and a bare clone with no
artifacts at all (e.g. someone evaluating the project). It must never fail to
start just because a model file or optional dependency is missing.

Operators also need to update models without downtime — retraining and dropping a
new `.joblib` in place should take effect without restarting the worker.

## Decision

`model_runner.DetectionEngine` loads detectors in a priority cascade and falls
through on `FileNotFoundError`:

1. **`meta`** — the combined ExtraTrees + XGBoost ensemble
   (`meta_model_combined.joblib`). Primary when present.
2. **`ml`** — the lightweight calibrated sklearn pipeline
   (`flow_detector.joblib`).
3. **`heuristic`** — a dependency-free rule-based `FlowScorer` (bytes, burst rate,
   direction, protocol) used when no artifacts are available.

The engine records artifact mtimes and `reload_if_stale()` reloads the model when
a file changes, so new artifacts are picked up live.

## Consequences

- The system always starts and produces scores, regardless of what is installed.
- The heuristic tier doubles as a transparent, explainable baseline and as the
  mode used by the test suite (see [0009](0009-test-strategy-and-ci.md)), which is
  why tests can run without xgboost/sklearn.
- Hot reload removes a restart from the model-deployment loop.
- Cost: three scoring implementations must agree on a return contract. They return
  `(score, label)` or `(score, label, attack_label)`, and callers normalize both
  shapes — a small but real source of coupling.
- Limitation: because the meta ensemble loads first when its artifact is present,
  the documented `flow_detector` metrics describe the *fallback* model, not
  necessarily the one serving in a full deployment. See
  [`../ml/model_card.md`](../ml/model_card.md).
