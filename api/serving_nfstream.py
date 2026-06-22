"""NFStream-based serving path for ADNS Phase 3.

Replaces MetaFeatureBuilder + _match_shape for flows captured via NFStream.

Core invariant:
    Features stored in flow.extra come from flow_to_row(adns_flows.schema.Flow).
    The corpus builder uses the same flow_to_row() on the same Flow objects
    from extract_flows_nfstream().  So training features == serving features —
    byte-identical for the same pcap.

The _NfstreamCaptureAgent (api/app.py) stores contract features in flow.extra
via flow_to_extra().  The NfstreamDetectionEngine (api/model_runner.py) reads
them back via extra_to_feature_vector() and calls score_matrix() which calls
validate_matrix() before every predict().  No _match_shape, no silent reconciliation.
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

logger = logging.getLogger(__name__)

# ── adns_flows path resolution ────────────────────────────────────────────────
# Dev:    api/ and ml/ are siblings under ADNS/
# Bundle: both are under sys._MEIPASS/
_THIS_DIR = Path(__file__).resolve().parent

if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
    _ML_DIR = Path(sys._MEIPASS) / "ml"
else:
    _ML_DIR = _THIS_DIR.parent / "ml"

if str(_ML_DIR) not in sys.path:
    sys.path.insert(0, str(_ML_DIR))

# adns_flows.schema has no nfstream dependency — safe at module level.
from adns_flows.schema import FEATURE_COLUMNS, SchemaError, flow_to_row, validate_matrix  # noqa: E402

# ── Model path ────────────────────────────────────────────────────────────────
if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
    _BASE_DIR = Path(sys._MEIPASS)
else:
    _BASE_DIR = _THIS_DIR

DEFAULT_NFSTREAM_MODEL_PATH = _BASE_DIR / "model_artifacts" / "nfstream_model.joblib"


# ── Feature helpers ───────────────────────────────────────────────────────────

def flow_to_extra(flow) -> dict:
    """Convert an adns_flows.schema.Flow to the dict stored in DB flow.extra.

    Stores all 21 FEATURE_COLUMNS as floats so the scoring path can reconstruct
    the feature vector without re-running NFStream on the pcap.
    Also stores port identity fields for display and the extractor tag for routing.
    """
    row = flow_to_row(flow)
    extra: dict = {k: float(row[k]) for k in FEATURE_COLUMNS}
    extra["src_port"] = int(flow.src_port)
    extra["dst_port"] = int(flow.dst_port)
    extra["_extractor"] = "nfstream"
    return extra


def extra_to_feature_vector(extra: dict | None) -> "np.ndarray | None":
    """Reconstruct the (1, 21) float32 feature matrix from a DB flow's stored extra.

    Returns None when:
    - extra is None or missing '_extractor': 'nfstream' marker (pre-Phase-3 flows)
    - any of the 21 contract features is absent or not convertible to float
    """
    if not extra or extra.get("_extractor") != "nfstream":
        return None
    try:
        vec = np.array([[float(extra[k]) for k in FEATURE_COLUMNS]], dtype="float32")
        return vec
    except (KeyError, TypeError, ValueError) as exc:
        logger.debug("extra_to_feature_vector: missing/bad feature: %s", exc)
        return None


# ── Scorer ────────────────────────────────────────────────────────────────────

class NfstreamScorer:
    """Loads the 21-feature NFStream model bundle and scores feature matrices.

    Bundle format: {"xgboost": XGBClassifier, "extra_trees": ExtraTreesClassifier}
    Trained on FEATURE_COLUMNS (21 features, binary label: 0=benign, 1=attack).

    validate_matrix() is called before every predict() call.
    n_features_in_ is checked against len(FEATURE_COLUMNS) — raises SchemaError
    if the model was trained on a different schema.  No silent reconciliation.
    """

    CLASS_LABELS = {0: "normal", 1: "anomaly"}

    def __init__(self, model_path: "str | Path | None" = None) -> None:
        import joblib

        resolved = Path(model_path or DEFAULT_NFSTREAM_MODEL_PATH)
        if not resolved.exists():
            raise FileNotFoundError(f"NFStream model not found: {resolved}")

        payload = joblib.load(resolved)
        if not isinstance(payload, dict):
            raise ValueError("NFStream model artifact must be a dict of estimators")

        self.models: dict[str, object] = {}
        for key in ("xgboost", "extra_trees"):
            if key in payload:
                self.models[key] = payload[key]

        if not self.models:
            raise ValueError("NFStream model bundle contains no supported estimator keys")

        self.anomaly_threshold = float(os.environ.get("ADNS_META_ANOMALY_THRESHOLD", "0.82"))
        self.watch_threshold = float(os.environ.get("ADNS_META_WATCH_THRESHOLD", "0.6"))

    def _label_for(self, prob: float) -> str:
        if prob >= self.anomaly_threshold:
            return "anomaly"
        if prob >= self.watch_threshold:
            return "watch"
        return "normal"

    def score_matrix(self, X: "np.ndarray") -> "list[tuple[float, str]]":
        """Score an (N, 21) float32 matrix. validate_matrix() gates every call.

        Raises SchemaError if:
        - X does not have exactly len(FEATURE_COLUMNS) columns
        - any loaded estimator was trained on a different feature count
        No _match_shape: mismatch is always an error, never silently corrected.
        """
        # Static schema gate: confirms FEATURE_COLUMNS hasn't drifted since import.
        validate_matrix(None, list(FEATURE_COLUMNS))

        if X.shape[1] != len(FEATURE_COLUMNS):
            raise SchemaError(
                f"feature matrix has {X.shape[1]} columns; "
                f"expected {len(FEATURE_COLUMNS)} (FEATURE_COLUMNS)"
            )

        probs_list: list[np.ndarray] = []
        for name, model in self.models.items():
            n_expected = getattr(model, "n_features_in_", len(FEATURE_COLUMNS))
            if X.shape[1] != n_expected:
                raise SchemaError(
                    f"estimator '{name}' was trained on {n_expected} features; "
                    f"feature matrix has {X.shape[1]}. "
                    "Retrain the NFStream model if FEATURE_COLUMNS changed."
                )
            p = np.asarray(model.predict_proba(X), dtype="float32")
            probs_list.append(p)

        if not probs_list:
            return [(0.0, "normal")] * X.shape[0]

        # Binary classifier: average P(class=1) = P(attack) across all models.
        attack_probs = np.zeros(X.shape[0], dtype="float32")
        for p in probs_list:
            attack_probs += p[:, 1] if p.shape[1] > 1 else p[:, 0]
        attack_probs /= len(probs_list)

        return [(float(p), self._label_for(float(p))) for p in attack_probs]
