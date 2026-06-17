from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Iterable, Sequence

try:
    from sqlalchemy.dialects.postgresql import insert as pg_insert
except ImportError:  # postgresql dialect not bundled (e.g. PyInstaller SQLite-only build)
    pg_insert = None  # type: ignore[assignment]
from sqlalchemy.exc import IntegrityError

from app import Flow, Prediction, app, db
from model_runner import DetectionEngine
from rdns import ReverseDNSResolver

logger = logging.getLogger(__name__)
detector = DetectionEngine()

SCORING_FETCH_CHUNK = int(os.environ.get("ADNS_SCORING_FETCH_CHUNK", "256"))
RDNS_ENABLED = os.environ.get("ADNS_RDNS_ENABLED", "true").lower() not in {"0", "false", "no"}
RDNS_TIMEOUT = float(os.environ.get("ADNS_RDNS_TIMEOUT_MS", "500")) / 1000.0
RDNS_CACHE_TTL = float(os.environ.get("ADNS_RDNS_CACHE_TTL", "900"))
RDNS_CACHE_SIZE = int(os.environ.get("ADNS_RDNS_CACHE_SIZE", "500"))
resolver = ReverseDNSResolver(cache_ttl=RDNS_CACHE_TTL, cache_size=RDNS_CACHE_SIZE, timeout=RDNS_TIMEOUT) if RDNS_ENABLED else None


def _infer_scanning(flow) -> str | None:
    """
    Lightweight heuristic: flows with explicit scan service or low-bytes hits to many ports
    default to scanning when the model says normal/watch.
    """
    extra = flow.extra or {}
    service = str(extra.get("service", "")).lower()
    if "scan" in service:
        return "scanning"
    try:
        dst_port = int(extra.get("dst_port") or 0)
    except (TypeError, ValueError):
        dst_port = 0
    try:
        src_port = int(extra.get("src_port") or 0)
    except (TypeError, ValueError):
        src_port = 0
    bytes_total = float(flow.bytes or 0.0)
    if (dst_port and dst_port <= 1024) or (src_port and src_port <= 1024):
        if bytes_total <= 20000:
            return "scanning"
    return None


def _chunked(ids: Sequence[int], size: int) -> Iterable[list[int]]:
    for start in range(0, len(ids), size):
        yield list(ids[start : start + size])


def _insert_predictions(records: list[dict]) -> int:
    if not records:
        return 0

    bind = db.session.get_bind()
    dialect = getattr(bind, "dialect", None)
    if pg_insert is not None and dialect and dialect.name == "postgresql":
        stmt = pg_insert(Prediction).values(records).on_conflict_do_nothing(index_elements=["flow_id"])
        result = db.session.execute(stmt)
        return result.rowcount or 0

    # Fallback for dev databases without PostgreSQL features.
    flow_ids = [rec["flow_id"] for rec in records]
    existing = {
        row.flow_id
        for row in db.session.query(Prediction.flow_id).filter(Prediction.flow_id.in_(flow_ids)).all()
    }
    to_insert = [rec for rec in records if rec["flow_id"] not in existing]
    if not to_insert:
        return 0
    try:
        db.session.bulk_insert_mappings(Prediction, to_insert)
        return len(to_insert)
    except IntegrityError:
        db.session.rollback()
        logger.warning("bulk insert hit integrity error on non-postgres backend; retrying row by row")
        inserted = 0
        for rec in to_insert:
            try:
                db.session.add(Prediction(**rec))
                db.session.flush()
                inserted += 1
            except IntegrityError:
                db.session.rollback()
        return inserted


def score_flow_batch(flow_ids: Sequence[int]) -> int:
    """
    Background job entrypoint for scoring a batch of newly ingested flows.
    """
    if not flow_ids:
        return 0

    with app.app_context():
        ids = [int(fid) for fid in flow_ids if fid]
        if not ids:
            return 0

        detector.reload_if_stale()
        scored = 0
        session = db.session

        try:
            for chunk_ids in _chunked(ids, SCORING_FETCH_CHUNK):
                flows: Iterable[Flow] = (
                    session.query(Flow)
                    .filter(Flow.id.in_(chunk_ids))
                    .order_by(Flow.id.asc())
                    .all()
                )
                if not flows:
                    continue

                flows_to_score = list(flows)
                if RDNS_ENABLED:
                    _enrich_with_rdns(flows_to_score)

                predictions = detector.predict_many(session, flows_to_score)
                if len(predictions) != len(flows):
                    raise RuntimeError("detection engine returned mismatched prediction count")
                now = datetime.now(timezone.utc)
                records = []
                for flow, pred in zip(flows, predictions):
                    if len(pred) == 3:
                        score, label, attack_label = pred
                    else:
                        score, label = pred
                        attack_label = None
                    records.append(
                        {
                            "flow_id": flow.id,
                            "score": score,
                            "label": label,
                            "created_at": now,
                        }
                    )
                    candidate_attack = None
                    base_labels = {"normal", "watch", "anomaly"}
                    if label and label.lower() not in base_labels:
                        candidate_attack = label
                    elif attack_label and label and label.lower() != "normal":
                        candidate_attack = attack_label
                    elif label and label.lower() in {"normal", "watch"}:
                        candidate_attack = _infer_scanning(flow)

                    extras = dict(flow.extra or {})
                    if candidate_attack and candidate_attack.lower() not in base_labels:
                        extras["attack_type"] = candidate_attack
                    else:
                        extras.pop("attack_type", None)
                    flow.extra = extras

                inserted = _insert_predictions(records)
                scored += inserted

            if scored:
                session.commit()
                logger.info("scored %d flow(s) via RQ job", scored)
            else:
                session.rollback()
            return scored
        except Exception:
            session.rollback()
            raise
        finally:
            session.remove()


def _enrich_with_rdns(flows: Sequence[Flow]) -> None:
    """
    Add rdns_exists/rdns_hash to flow.extra for scoring. This does not persist changes to DB.
    """
    for flow in flows:
        peer_ip = flow.src_ip or flow.dst_ip
        if resolver is None:
            continue
        rdns_ok = resolver.lookup(peer_ip)
        extra = dict(flow.extra or {})
        extra["rdns_exists"] = bool(rdns_ok)
        # hash the peer hostname presence without storing the name
        extra["rdns_hash"] = 1 if rdns_ok else 0
        flow.extra = extra
