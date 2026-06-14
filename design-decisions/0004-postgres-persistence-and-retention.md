# 0004 — PostgreSQL persistence, in-code schema management, and retention

- **Status:** Accepted
- **Phase:** 0 — Foundational architecture

## Context

Flows and predictions need durable storage that supports concurrent writes from
the API and worker, time-range queries for the dashboard, and an upsert primitive
for idempotent prediction writes. The schema has also evolved over the project's
life (the `flows.extra` JSON column and the unique constraint on
`predictions.flow_id` were added after the first version), so older databases
needed to migrate without manual intervention.

The system ingests continuously, so unbounded growth would eventually exhaust disk
and slow queries.

## Decision

- Use **PostgreSQL** as the primary store via SQLAlchemy, with the DSN supplied by
  `SQLALCHEMY_DATABASE_URI` so SQLite can be substituted for zero-install demos
  and tests.
- Manage schema drift **in code** at startup: `init_db()` creates tables,
  `ensure_flow_extra_column()` adds the JSON column to legacy tables, and
  `ensure_prediction_flow_unique_index()` de-duplicates and enforces uniqueness on
  `predictions.flow_id`.
- Enforce **retention** on ingest/simulate paths: prune flows older than
  `ADNS_FLOW_RETENTION_MINUTES` and trim beyond `ADNS_FLOW_RETENTION_MAX_ROWS`,
  deleting in batches.

## Consequences

- The app self-heals older schemas, so `/ingest` never trips over a missing column.
- The same codebase runs against Postgres in production and SQLite in tests —
  prediction upserts use PostgreSQL `ON CONFLICT` with a portable
  select-then-insert fallback for SQLite.
- The database stays bounded automatically; the demo never silently fills a disk.
- Cost: in-code migration is convenient but not a substitute for a real migration
  tool. Alembic is already a dependency and would be the path forward if the schema
  grows more complex; this is noted as a future evolution rather than implemented now.
