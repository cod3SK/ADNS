# 0008 — Externalized configuration and secrets

- **Status:** Accepted
- **Phase:** 1 — Hardening

## Context

Runtime behavior across ADNS — database connection, Redis URL, queue/retention
tuning, reverse-DNS knobs — is already environment-driven, which is good. But the
PostgreSQL credentials were **hard-coded** in `docker-compose.yml` and in the
default DSN (`adns:adns_password`). Committed credentials are a classic "doesn't
understand security hygiene" signal and a real risk if a deployment ever reuses
the default.

## Decision

- Drive the database credentials from `POSTGRES_USER`, `POSTGRES_PASSWORD`, and
  `POSTGRES_DB`, consumed by both the `postgres` service and the
  `SQLALCHEMY_DATABASE_URI` woven for the API/worker.
- Keep the stack runnable out of the box by using Compose default-substitution
  (`${POSTGRES_PASSWORD:-adns_password}`): no `.env` is required for a local demo,
  but real values override cleanly.
- Document every knob, including the new `ADNS_ADMIN_TOKEN`, in `.env.example`, and
  collapse the stale, divergent `api/.env.example` into a pointer to the canonical
  root file.

## Consequences

- No real secret needs to live in version control; the committed defaults are
  explicitly demo-only and documented as such.
- The "clone and `docker compose up`" experience is preserved — externalizing
  config did not add a setup step.
- Cost: default-substitution means a deployer who forgets to set a password still
  gets the weak default rather than a hard failure. The README and `.env.example`
  call this out; a stricter stance (no default, fail if unset) is a reasonable
  future tightening.
