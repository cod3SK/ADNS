# 0009 — Lightweight test strategy and continuous integration

- **Status:** Accepted
- **Phase:** 2 — Quality and confidence

## Context

The project had no automated tests. The behavior most worth protecting — request
validation, the auth gate, simulation, retention/blocking, and the scoring math —
is also the behavior most likely to regress during refactoring. But the full
runtime depends on PostgreSQL, Redis, `tshark`, trained `.joblib` artifacts, and
xgboost. Requiring all of that to run a test would make the suite slow, flaky, and
unlikely to run in CI on every push.

## Decision

Test against real dependencies where practical; isolate only what is unavoidable:

- Run the Flask app against a throwaway **SQLite** database.
- The **real NFStream model** (`nfstream_model.joblib`) is loaded by default in
  the API test suite — it is present via Git LFS. Tests that require an absent
  model monkeypatch `app._simulation_scorer` with a fresh engine pointing at a
  nonexistent path (`tmp_path / "no_model.joblib"`).
- Scoring runs via the in-process `ThreadPoolExecutor` (Redis/RQ was removed — see
  [0002](0002-async-scoring-redis-rq.md)); no queue service is required.

**ML suite** (`ml/adns_flows/tests/`, 131 tests) covers:
- Feature contract: column identity, ordering, `validate_matrix()` pass/fail
- Orientation: canonical src/dst assignment invariants
- Extractor parity: NFStream corpus path vs NFStream serving path produce
  byte-identical feature matrices on the same pcap
- Grain-parity: direct live capture vs windowed path grain comparison
- Corpus labeling: UNSW, Gotham, CIC label helpers (46 tests)

**API suite** (`api/tests/`, 10 tests):
- Absent-model loud-failure: ERROR log, `is_model_loaded=False`, HTTP 503 on
  `/capture/autostart`, `/model_status` reports `"absent"`

**Total: 141 tests.** Run with: `python -m pytest api/tests/ ml/ -q`

GitHub Actions (`.github/workflows/ci.yml`) runs the full suite and frontend
lint/build on every push.

## Consequences

- The full suite runs without external services (no Docker, no Postgres, no
  tshark, no Redis).
- Tests cover the real model path (not a heuristic fallback), so regressions in
  feature extraction or model loading are caught directly.
- Coverage gap (acknowledged): the frozen exe, live NFStream capture on a real
  interface, and the installer are exercised by the manual smoke test
  (`step4_smoke_test.py`), not the unit suite.
