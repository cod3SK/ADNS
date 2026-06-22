"""Train the 21-feature NFStream model on all three NFStream corpora (E3 pool).

Produces api/model_artifacts/nfstream_model.joblib in the same bundle format
as meta_model_combined.joblib: {"xgboost": XGBClassifier, "extra_trees": ExtraTreesClassifier}

Usage:
    python ml/train_nfstream.py

Parquet inputs (must exist):
    outputs/corpus/unsw_flows.parquet
    outputs/corpus/gotham_flows.parquet
    outputs/corpus/cic_tuesday_flows.parquet

Output:
    api/model_artifacts/nfstream_model.joblib
"""
from __future__ import annotations

import logging
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from adns_flows.schema import FEATURE_COLUMNS

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
PARQUETS = {
    "unsw":   _ROOT / "outputs" / "corpus" / "unsw_flows.parquet",
    "gotham": _ROOT / "outputs" / "corpus" / "gotham_flows.parquet",
    "cic":    _ROOT / "outputs" / "corpus" / "cic_tuesday_flows.parquet",
}
OUT = _ROOT / "api" / "model_artifacts" / "nfstream_model.joblib"


# ── Training helpers ─────────────────────────────────────────────────────────

def _load_corpus(name: str, path: Path) -> pd.DataFrame:
    log.info("loading %s: %s", name, path)
    df = pd.read_parquet(path, columns=list(FEATURE_COLUMNS) + ["label"])
    n_att = int((df["label"] == 1).sum())
    log.info("  %s: %d rows, %d attack (%.2f%%)", name, len(df), n_att, 100 * n_att / max(len(df), 1))
    return df


def _to_xy(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    feat_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    X = df[feat_cols].to_numpy(dtype=np.float32)
    y = df["label"].to_numpy(dtype=np.int32)
    return X, y


def _train_xgb(X_train: np.ndarray, y_train: np.ndarray, seed: int = 42) -> object:
    try:
        from xgboost import XGBClassifier
    except ImportError:
        raise ImportError("xgboost required: pip install xgboost==1.7.6")

    n_neg = float((y_train == 0).sum())
    n_pos = float((y_train == 1).sum())
    scale_pos_weight = n_neg / max(n_pos, 1.0)
    log.info("  xgboost: n_neg=%d n_pos=%d scale_pos_weight=%.3f", int(n_neg), int(n_pos), scale_pos_weight)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = XGBClassifier(
            tree_method="hist",
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            scale_pos_weight=scale_pos_weight,
            random_state=seed,
            verbosity=0,
            eval_metric="aucpr",
            nthread=-1,
        )
        model.fit(X_train, y_train)
    return model


def _train_et(X_train: np.ndarray, y_train: np.ndarray, seed: int = 42) -> object:
    from sklearn.ensemble import ExtraTreesClassifier

    n_neg = float((y_train == 0).sum())
    n_pos = float((y_train == 1).sum())
    class_weight = {0: 1.0, 1: n_neg / max(n_pos, 1.0)}

    model = ExtraTreesClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=2,
        class_weight=class_weight,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def _evaluate(model, X_test: np.ndarray, y_test: np.ndarray, name: str) -> None:
    from sklearn.metrics import average_precision_score, recall_score
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(np.int32)
    pr_auc = float(average_precision_score(y_test, y_prob)) if len(np.unique(y_test)) > 1 else float("nan")
    recall = float(recall_score(y_test, y_pred, pos_label=1, zero_division=0))
    benign_mask = y_test == 0
    fpr = float(((y_pred == 1) & benign_mask).sum() / max(benign_mask.sum(), 1))
    log.info("  %s eval: PR-AUC=%.4f recall=%.4f benign_FPR=%.4f", name, pr_auc, recall, fpr)


# ── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    from sklearn.model_selection import StratifiedShuffleSplit

    # Verify all parquets exist
    missing = [str(p) for p in PARQUETS.values() if not p.exists()]
    if missing:
        log.error("missing parquet(s):\n  %s", "\n  ".join(missing))
        sys.exit(1)

    # Load and pool all three corpora (E3 configuration)
    frames = [_load_corpus(name, path) for name, path in PARQUETS.items()]
    df = pd.concat(frames, ignore_index=True)
    log.info("pooled: %d rows total", len(df))

    X, y = _to_xy(df)
    del df, frames

    n_feat = X.shape[1]
    if n_feat != len(FEATURE_COLUMNS):
        log.error("feature count mismatch: data has %d, FEATURE_COLUMNS has %d", n_feat, len(FEATURE_COLUMNS))
        sys.exit(1)
    log.info("feature matrix: %d rows x %d features", *X.shape)

    # 80/20 stratified split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, test_idx = next(sss.split(X, y))
    X_train, y_train = X[train_idx], y[train_idx]
    X_test,  y_test  = X[test_idx],  y[test_idx]
    log.info("split: train=%d test=%d", len(train_idx), len(test_idx))

    # Train both estimators
    log.info("training XGBoost...")
    xgb_model = _train_xgb(X_train, y_train)
    _evaluate(xgb_model, X_test, y_test, "xgboost")

    log.info("training ExtraTrees...")
    et_model = _train_et(X_train, y_train)
    _evaluate(et_model, X_test, y_test, "extra_trees")

    # Save bundle
    OUT.parent.mkdir(parents=True, exist_ok=True)
    bundle = {"xgboost": xgb_model, "extra_trees": et_model}
    joblib.dump(bundle, OUT)
    log.info("saved: %s", OUT)
    log.info("n_features_in_: xgboost=%d  extra_trees=%d",
             getattr(xgb_model, "n_features_in_", -1),
             getattr(et_model, "n_features_in_", -1))


if __name__ == "__main__":
    main()
