# 0002 — Asynchronous scoring with an in-process thread pool

- **Status:** Accepted (amended — see below)
- **Phase:** 0 — Foundational architecture

## Context

Flow ingestion (`POST /ingest`) is on the hot path: the capture agent posts
batches of ~50 flows every couple of seconds and must not block on model
inference. Scoring, however, is comparatively expensive and depends on heavy ML
libraries. If ingestion waited for scoring, a slow model or a burst of traffic
would back-pressure the agent and risk dropped captures.

At the same time, we did not want a hard dependency on Redis for someone who just
wants to try the project — a down or absent queue should degrade gracefully, not
fail ingestion outright.

## Decision

Decouple ingestion from scoring with a Redis-backed RQ queue:

- `POST /ingest` persists flows, then enqueues their IDs on the `flow_scores`
  queue (`task_queue.enqueue_flow_scoring`) and returns immediately.
- A separate worker (`api/worker.py`) consumes the queue and runs
  `tasks.score_flow_batch`, which loads flows in chunks, scores them, and upserts
  `Prediction` rows.
- If enqueueing raises (e.g. Redis is unreachable), the API falls back to scoring
  **inline** in the request by calling `score_flow_batch` directly.

Batch size, fetch chunk size, queue name, and job timeout are all environment-driven.

## Consequences

- Ingestion stays fast and resilient to scoring slowness under normal operation.
- The system still works end-to-end with Redis stopped — useful for quick local
  demos — at the cost of slower ingestion when the fallback engages.
- Predictions are written idempotently (`ON CONFLICT DO NOTHING` on `flow_id`),
  so a flow scored by both the inline fallback and a later worker run is safe.
- Cost: two code paths reach the scorer (worker and inline), so scoring must be
  free of request-context assumptions. `score_flow_batch` runs inside an explicit
  app context to satisfy this.

## Amendment — replaced with in-process thread pool

Redis/RQ introduced an external service dependency that prevented the stack from
running natively on Windows (no official Redis for Windows) and added operational
complexity for local development. The system is now refactored to target Windows
as its primary platform.

**New approach:** `task_queue.enqueue_flow_scoring` submits chunks to a
`concurrent.futures.ThreadPoolExecutor` (default 2 workers, tunable via
`ADNS_SCORER_WORKERS`). The public interface — `enqueue_flow_scoring(flow_ids)`
— is unchanged, so the rest of the codebase is unaffected. `score_flow_batch` in
`tasks.py` still wraps itself in an explicit Flask app context, making it safe
to call from background threads.

**What was removed:** `api/worker.py` (RQ bootstrap), `redis` and `rq` from
`requirements.txt`, the `redis` and `worker` services from `docker-compose.yml`,
and the two-level inline fallback in `/ingest`.

**Trade-off accepted:** scoring jobs are in-process and do not survive a process
restart (unlike RQ jobs which live in Redis). For this project's scale this is
acceptable. A production deployment requiring durability could reintroduce a queue
without changing `tasks.py` or the `/ingest` endpoint.
