"""
Sanity-check XGBoost classifier for the labeled corpus.

Reports PR-AUC and per-attack-class recall — not accuracy, since the corpus is
typically class-imbalanced and accuracy on an imbalanced set is misleading.

XGBoost version is pinned to 1.7.6 to match the model served in the ADNS API.

Usage
-----
    import pandas as pd
    from corpus.sanity_model import train_sanity_model

    df = pd.read_parquet("outputs/corpus/unsw_flows.parquet")
    results = train_sanity_model(df)
    # results keys: pr_auc, attack_recall, top10_importances,
    #               classification_report, n_train, n_test
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    recall_score,
)
from sklearn.model_selection import StratifiedShuffleSplit

from adns_flows import FEATURE_COLUMNS

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None  # type: ignore[assignment,misc]


def train_sanity_model(
    df: pd.DataFrame,
    test_size: float = 0.2,
    seed: int = 42,
) -> dict:
    """Train a single XGBoost model and return a results dict.

    Parameters
    ----------
    df        : labeled corpus DataFrame (must include FEATURE_COLUMNS + 'label')
    test_size : fraction of data held out for evaluation (default 0.2)
    seed      : random seed for reproducibility

    Returns
    -------
    dict with keys:
        pr_auc                 : precision-recall AUC on the test split
        attack_recall          : recall for label=1 at threshold 0.5
        top10_importances      : [(feature, importance), ...] by gain (top 10)
        classification_report  : sklearn text report
        n_train / n_test       : split sizes
    """
    if XGBClassifier is None:
        raise ImportError(
            "xgboost is required — pip install xgboost==1.7.6"
        )

    feature_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = df["label"].to_numpy(dtype=np.int32)

    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=test_size, random_state=seed
    )
    train_idx, test_idx = next(splitter.split(X, y))
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    n_neg = float((y_train == 0).sum())
    n_pos = float((y_train == 1).sum())
    scale_pos_weight = n_neg / max(n_pos, 1.0)

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
        )
        model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(np.int32)

    pr_auc         = average_precision_score(y_test, y_prob)
    attack_recall  = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
    clf_report     = classification_report(y_test, y_pred)

    importance = dict(zip(feature_cols, model.feature_importances_))
    top10 = sorted(importance.items(), key=lambda kv: kv[1], reverse=True)[:10]

    results = {
        "pr_auc":               round(float(pr_auc), 4),
        "attack_recall":        round(float(attack_recall), 4),
        "top10_importances":    top10,
        "classification_report": clf_report,
        "n_train":              len(X_train),
        "n_test":               len(X_test),
    }

    print("\n=== Sanity model results ===")
    print(f"  PR-AUC         : {results['pr_auc']}")
    print(f"  Attack recall  : {results['attack_recall']}")
    print(f"\n{clf_report}")
    print("  Top-10 feature importances (by gain):")
    for feat, imp in top10:
        print(f"    {feat:<30} {imp:.4f}")

    return results
