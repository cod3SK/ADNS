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

Test against the system's own graceful-degradation paths instead of its heavy
dependencies:

- Run the Flask app against a throwaway **SQLite** database.
- Force the **heuristic** detection tier by pointing the model-artifact paths at
  non-existent files, so no xgboost/sklearn/joblib model load is needed.
- Disable reverse-DNS and host `nsenter` in the test environment.
- Scoring runs via the in-process `ThreadPoolExecutor` (Redis/RQ was removed — see
  [0002](0002-async-scoring-redis-rq.md)); no queue service is required.

The suite (`api/tests/`) covers endpoints, payload validation, the admin-token
gate, simulation, blocked-IP filtering, the heuristic scorer, and the meta feature
builder. A `requirements-test.txt` pins only the lightweight deps. GitHub Actions
(`.github/workflows/ci.yml`) runs the API tests and the frontend lint/build on
every push and pull request.

## Consequences

- The full suite runs in seconds with no external services, so CI is fast and
  reliable and the README can carry a live build badge.
- The tests double as executable documentation of the degradation paths from
  [0002](0002-async-scoring-redis-rq.md) and [0003](0003-three-tier-detection-cascade.md).
- Coverage gap (acknowledged): the trained models, the Redis/RQ worker, and the
  `tshark` agent are not exercised end-to-end. Those would need integration tests
  with real services (e.g. Compose-based or `testcontainers`) and are out of scope
  for the unit suite.
