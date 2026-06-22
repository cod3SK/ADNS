# 0003 — Three-tier detection cascade with hot reload

- **Status:** Superseded — see amendment below
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

## Amendment — replaced by NfstreamDetectionEngine (NFStream migration)

The three-tier cascade (`meta` → `ml` → `heuristic`) has been removed. All three
prior model classes (`MetaEnsembleModel`, `DetectionEngine`, `FlowScorer`) are
deleted from `api/model_runner.py`.

**Current detection path:**

`NfstreamDetectionEngine` is the sole scorer. It:
1. Loads `api/model_artifacts/nfstream_model.joblib` (XGBoost + ExtraTrees E3
   pooled, trained on 21 contract features).
2. Calls `NfstreamScorer.score_matrix(X)` which gates every call with
   `validate_matrix()` — raises `SchemaError` on column mismatch (replaces silent
   `_match_shape` padding).
3. Returns `(0.0, "normal")` for any flow whose `extra` does not carry the
   `_extractor: nfstream` marker (pre-migration flows, extraction failures).

**Absent model behavior** (replaces graceful degradation to heuristic):
- `is_model_loaded` is `False`; `model_error` carries the reason.
- `NfstreamDetectionEngine.__init__` logs at `ERROR` level.
- `/capture/autostart` returns HTTP 503 — no silent no-scoring.

**Hot reload removed.** The mtime-based `reload_if_stale()` mechanism is gone.
Model reloads require a process restart.

**Test coverage:** `api/tests/test_absent_model.py` (10 tests) covers the
absent-model loud-failure path. `ml/adns_flows/tests/` (131 tests) covers the
feature contract, orientation, and grain-parity invariants.
