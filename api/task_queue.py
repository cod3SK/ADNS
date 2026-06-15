from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Sequence

logger = logging.getLogger(__name__)

_executor: ThreadPoolExecutor | None = None


def _get_executor() -> ThreadPoolExecutor:
    global _executor
    if _executor is None:
        workers = int(os.environ.get("ADNS_SCORER_WORKERS", "2"))
        _executor = ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="adns-scorer",
        )
        logger.info("initialized scorer thread pool with %d worker(s)", workers)
    return _executor


def enqueue_flow_scoring(flow_ids: Sequence[int]) -> int:
    """Submit flow IDs to the background scorer thread pool. Returns the count submitted."""
    ids = [int(fid) for fid in flow_ids if fid]
    if not ids:
        return 0
    batch_size = int(os.environ.get("ADNS_SCORING_BATCH_SIZE", "100"))
    executor = _get_executor()
    for start in range(0, len(ids), batch_size):
        chunk = ids[start : start + batch_size]
        executor.submit(_run_batch, chunk)
    return len(ids)


def _run_batch(chunk: list[int]) -> None:
    try:
        from tasks import score_flow_batch
        score_flow_batch(chunk)
    except Exception:
        logger.exception("background scorer failed for chunk of %d flow(s)", len(chunk))
