"""
Cross-dataset generalization evaluation — UNSW-NB15 vs Gotham Dataset 2025.

Three evaluation configurations
---------------------------------
(A) IN-DOMAIN Gotham
    Train 80% Gotham / test 20% held-out Gotham.
    Ceiling: how well the model can do with matched train/test distributions.

(B) CROSS-DOMAIN (both directions)
    train UNSW  / test Gotham   — real generalization test
    train Gotham / test UNSW    — reverse direction
    For each direction: overall PR-AUC, per-attack_cat recall, AND benign
    false-positive rate (benign FP = fraction of true-benign flows predicted
    as attack). A UNSW-trained model may flag normal IoT traffic as anomalous
    purely from domain-shift in the benign class.

(C) POOLED
    Train 80% (UNSW + Gotham combined) / test 20% held-out from both.
    Expected strongest model; confirms complementary value of both datasets.

Step 4: feature-distribution diagnostic
-----------------------------------------
For each FEATURE_COLUMN, compare benign-only distributions (median + IQR)
between UNSW and Gotham side-by-side. Features whose IQR ranges do not overlap
are flagged as domain-shift drivers — they mechanistically explain cross-domain
recall gaps.

Usage
-----
    python -m corpus.cross_eval \
        --unsw  outputs/corpus/unsw_flows.parquet \
        --gotham outputs/corpus/gotham_flows.parquet

Or from Python:
    from corpus.cross_eval import run_cross_eval
    results = run_cross_eval(unsw_path, gotham_path)
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from adns_flows import FEATURE_COLUMNS

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None  # type: ignore[assignment,misc]

from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedShuffleSplit


# ── model training ─────────────────────────────────────────────────────────

def _train_xgb(
    X_train: np.ndarray,
    y_train: np.ndarray,
    seed: int = 42,
) -> object:
    """Fit an XGBoost classifier (tree_method=hist, xgboost==1.7.6 target)."""
    if XGBClassifier is None:
        raise ImportError("xgboost is required — pip install xgboost==1.7.6")

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
    return model


def _to_xy(
    df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (X, y, attack_cats, src_ips) for a labeled corpus DataFrame."""
    feat_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    X    = df[feat_cols].to_numpy(dtype=np.float32)
    y    = df["label"].to_numpy(dtype=np.int32)
    cats = df["attack_cat"].fillna("").to_numpy(dtype=str)
    ips  = df["src_ip"].to_numpy(dtype=str)
    return X, y, cats, ips


# ── per-attack-cat recall ──────────────────────────────────────────────────

def _per_cat_recall(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    attack_cats: np.ndarray,
) -> dict[str, float]:
    """Return recall per attack_cat for label=1 rows only.

    Benign rows (attack_cat=="") are skipped; their false-positive rate is
    computed separately by _benign_fpr().
    """
    results: dict[str, float] = {}
    attack_mask = y_true == 1
    if not attack_mask.any():
        return results

    unique_cats = sorted(set(attack_cats[attack_mask]) - {""})
    for cat in unique_cats:
        cat_mask = attack_mask & (attack_cats == cat)
        if cat_mask.sum() == 0:
            continue
        recall = recall_score(
            y_true[cat_mask], y_pred[cat_mask],
            pos_label=1, zero_division=0,
        )
        results[cat] = round(float(recall), 4)

    # Overall attack recall across all categories
    results["_overall_attack"] = round(
        float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)), 4,
    )
    return results


def _benign_fpr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """False-positive rate among true-benign flows: FP / (FP + TN)."""
    benign_mask = y_true == 0
    if not benign_mask.any():
        return float("nan")
    false_positives = ((y_pred == 1) & benign_mask).sum()
    true_negatives  = ((y_pred == 0) & benign_mask).sum()
    return round(float(false_positives / max(false_positives + true_negatives, 1)), 4)


def _per_host_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    src_ips: np.ndarray,
) -> dict:
    """Per-source-host attacker recall and benign-host false-positive rate.

    A flood generates millions of flows from a single host.  A deployable
    detector cares whether it caught the *host*, not just scored each SYN.

    attacker_host_recall : fraction of attacking src_ips with >=1 flow flagged
    benign_host_fpr      : fraction of benign src_ips with >=1 flow flagged
    """
    attacker_caught: list[bool] = []
    benign_flagged:  list[bool] = []

    for ip in np.unique(src_ips):
        mask = src_ips == ip
        if y_true[mask].any():
            attacker_caught.append(bool(y_pred[mask].any()))
        else:
            benign_flagged.append(bool(y_pred[mask].any()))

    return {
        "attacker_host_recall": round(
            float(sum(attacker_caught) / max(len(attacker_caught), 1)), 4,
        ),
        "benign_host_fpr": round(
            float(sum(benign_flagged) / max(len(benign_flagged), 1)), 4,
        ),
        "n_attacker_hosts": len(attacker_caught),
        "n_benign_hosts":   len(benign_flagged),
    }


# ── evaluation block ───────────────────────────────────────────────────────

_THRESHOLD = 0.5  # fixed operating threshold for precision/recall reporting


def _evaluate(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    test_cats: np.ndarray,
    train_label: str,
    test_label: str,
    seed: int = 42,
    test_src_ips: np.ndarray | None = None,
) -> dict:
    """Train on (X_train, y_train), evaluate on (X_test, y_test).

    Returns a dict with:
      pr_auc            : PR-AUC on test set (use with prevalence — see below)
      prevalence        : positive-class fraction in test set (trivial-baseline)
      precision_at_t    : precision at THRESHOLD (default 0.5)
      recall_at_t       : recall at THRESHOLD
      threshold         : the fixed threshold used
      benign_fpr        : false-positive rate among true-benign test flows
      per_cat_recall    : dict[attack_cat -> recall] + '_overall_attack'
      host_metrics      : per-source-host attacker recall + benign host FPR
                          (only if test_src_ips is provided)
      n_train / n_test  : split sizes
      train_label       : identifier string for the training set
      test_label        : identifier string for the test set
    """
    if len(np.unique(y_train)) < 2:
        raise ValueError(
            f"Training set '{train_label}' has only one class — cannot train. "
            "Check corpus balance."
        )

    model  = _train_xgb(X_train, y_train, seed=seed)
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= _THRESHOLD).astype(np.int32)

    n_pos      = int(y_test.sum())
    prevalence = float(n_pos) / max(len(y_test), 1)
    pr_auc = (
        float(average_precision_score(y_test, y_prob))
        if len(np.unique(y_test)) > 1 else float("nan")
    )
    prec = float(precision_score(y_test, y_pred, pos_label=1, zero_division=0))
    rec  = float(recall_score(y_test, y_pred, pos_label=1, zero_division=0))

    result = {
        "train_label":      train_label,
        "test_label":       test_label,
        "n_train":          len(X_train),
        "n_test":           len(X_test),
        "prevalence":       round(prevalence, 4),
        "threshold":        _THRESHOLD,
        "pr_auc":           round(pr_auc, 4),
        "precision_at_t":   round(prec, 4),
        "recall_at_t":      round(rec, 4),
        "benign_fpr":       _benign_fpr(y_test, y_pred),
        "per_cat_recall":   _per_cat_recall(y_test, y_pred, test_cats),
    }
    if test_src_ips is not None:
        result["host_metrics"] = _per_host_metrics(y_test, y_pred, test_src_ips)

    return result


# ── feature distribution diagnostic ───────────────────────────────────────

def feature_distribution_compare(
    df_unsw: pd.DataFrame,
    df_gotham: pd.DataFrame,
) -> pd.DataFrame:
    """Compare benign-only feature distributions between UNSW and Gotham.

    For each FEATURE_COLUMN, computes median and IQR (Q25–Q75) for benign
    flows in each dataset. Flags features where the IQR ranges do not overlap
    — these are the strongest domain-shift drivers.

    Returns a DataFrame with columns:
      feature, unsw_median, unsw_q25, unsw_q75,
      gotham_median, gotham_q25, gotham_q75, overlap, shift_flag
    """
    benign_u = df_unsw[df_unsw["label"] == 0]
    benign_g = df_gotham[df_gotham["label"] == 0]

    records = []
    for col in FEATURE_COLUMNS:
        if col not in df_unsw.columns or col not in df_gotham.columns:
            continue

        su = benign_u[col].dropna()
        sg = benign_g[col].dropna()

        u_q25, u_med, u_q75 = float(su.quantile(0.25)), float(su.median()), float(su.quantile(0.75))
        g_q25, g_med, g_q75 = float(sg.quantile(0.25)), float(sg.median()), float(sg.quantile(0.75))

        # IQR ranges overlap iff max(Q25s) <= min(Q75s)
        overlap = max(u_q25, g_q25) <= min(u_q75, g_q75)

        records.append({
            "feature":       col,
            "unsw_median":   round(u_med, 4),
            "unsw_q25":      round(u_q25, 4),
            "unsw_q75":      round(u_q75, 4),
            "gotham_median": round(g_med, 4),
            "gotham_q25":    round(g_q25, 4),
            "gotham_q75":    round(g_q75, 4),
            "overlap":       overlap,
            "shift_flag":    not overlap,
        })

    return pd.DataFrame(records)


# ── main evaluation orchestrator ───────────────────────────────────────────

def run_cross_eval(
    unsw_path: str | Path,
    gotham_path: str | Path,
    test_size: float = 0.20,
    seed: int = 42,
) -> dict:
    """Run all three evaluation configurations and feature diagnostics.

    Parameters
    ----------
    unsw_path   : path to the UNSW labeled corpus parquet
    gotham_path : path to the Gotham labeled corpus parquet
    test_size   : held-out fraction for in-domain / pooled splits (default 0.20)
    seed        : random seed for reproducibility

    Returns
    -------
    dict with keys:
      in_domain_gotham     : result dict from config (A)
      cross_unsw_to_gotham : result dict from config (B) train-UNSW→test-Gotham
      cross_gotham_to_unsw : result dict from config (B) train-Gotham→test-UNSW
      pooled               : result dict from config (C)
      feature_shift        : DataFrame from feature_distribution_compare()
    """
    df_unsw   = pd.read_parquet(unsw_path)
    df_gotham = pd.read_parquet(gotham_path)

    X_u, y_u, cats_u, ips_u = _to_xy(df_unsw)
    X_g, y_g, cats_g, ips_g = _to_xy(df_gotham)

    # (A) In-domain Gotham — 80/20 stratified split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    tr_g, te_g = next(sss.split(X_g, y_g))
    in_domain_gotham = _evaluate(
        X_g[tr_g], y_g[tr_g],
        X_g[te_g], y_g[te_g], cats_g[te_g],
        train_label="Gotham (80%)", test_label="Gotham (20%)", seed=seed,
        test_src_ips=ips_g[te_g],
    )

    # (B) Cross-domain: UNSW → Gotham
    cross_u2g = _evaluate(
        X_u, y_u,
        X_g, y_g, cats_g,
        train_label="UNSW (full)", test_label="Gotham (full)", seed=seed,
        test_src_ips=ips_g,
    )

    # (B) Cross-domain: Gotham → UNSW
    cross_g2u = _evaluate(
        X_g, y_g,
        X_u, y_u, cats_u,
        train_label="Gotham (full)", test_label="UNSW (full)", seed=seed,
        test_src_ips=ips_u,
    )

    # (C) Pooled — 80/20 stratified split on combined corpus
    X_pool    = np.vstack([X_u, X_g])
    y_pool    = np.concatenate([y_u, y_g])
    cats_pool = np.concatenate([cats_u, cats_g])
    ips_pool  = np.concatenate([ips_u, ips_g])

    sss_p = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    tr_p, te_p = next(sss_p.split(X_pool, y_pool))
    pooled = _evaluate(
        X_pool[tr_p], y_pool[tr_p],
        X_pool[te_p], y_pool[te_p], cats_pool[te_p],
        train_label="UNSW+Gotham (80%)", test_label="UNSW+Gotham (20%)", seed=seed,
        test_src_ips=ips_pool[te_p],
    )

    # (Step 4) Feature distribution comparison
    feature_shift = feature_distribution_compare(df_unsw, df_gotham)

    return {
        "in_domain_gotham":     in_domain_gotham,
        "cross_unsw_to_gotham": cross_u2g,
        "cross_gotham_to_unsw": cross_g2u,
        "pooled":               pooled,
        "feature_shift":        feature_shift,
    }


# ── pretty-printer ─────────────────────────────────────────────────────────

def _print_eval_block(label: str, result: dict) -> None:
    t = result.get("threshold", _THRESHOLD)
    prevalence = result.get("prevalence", float("nan"))
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  train={result['train_label']}  test={result['test_label']}")
    print(f"  n_train={result['n_train']:,}  n_test={result['n_test']:,}")
    print(f"{'='*60}")
    print(f"  Prevalence      : {prevalence:.4f}  "
          f"(trivial always-attack baseline PR-AUC)")
    print(f"  PR-AUC          : {result['pr_auc']:.4f}")
    print(f"  Precision @{t}  : {result.get('precision_at_t', float('nan')):.4f}")
    print(f"  Recall    @{t}  : {result.get('recall_at_t', float('nan')):.4f}")
    print(f"  Benign FPR      : {result['benign_fpr']:.4f}  "
          f"(fraction of true-benign flows flagged as attack)")
    hm = result.get("host_metrics")
    if hm:
        print(f"  Attacker-host recall : {hm['attacker_host_recall']:.4f}  "
              f"({hm['n_attacker_hosts']} attacking src_ips)")
        print(f"  Benign-host FPR      : {hm['benign_host_fpr']:.4f}  "
              f"({hm['n_benign_hosts']} benign src_ips)")
    print(f"\n  Per-attack-cat recall:")
    cats = result["per_cat_recall"]
    overall = cats.pop("_overall_attack", None)
    for cat, rec in sorted(cats.items()):
        print(f"    {cat:<30} {rec:.4f}")
    if overall is not None:
        print(f"    {'[OVERALL ATTACK]':<30} {overall:.4f}")
        cats["_overall_attack"] = overall   # restore


def print_results(results: dict) -> None:
    """Print all evaluation results to stdout."""
    _print_eval_block("(A) IN-DOMAIN: train Gotham / test Gotham (ceiling)",
                      results["in_domain_gotham"])
    _print_eval_block("(B) CROSS-DOMAIN: train UNSW / test Gotham",
                      results["cross_unsw_to_gotham"])
    _print_eval_block("(B) CROSS-DOMAIN: train Gotham / test UNSW",
                      results["cross_gotham_to_unsw"])
    _print_eval_block("(C) POOLED: train UNSW+Gotham / test UNSW+Gotham",
                      results["pooled"])

    fs = results["feature_shift"]
    flagged = fs[fs["shift_flag"]]
    print(f"\n{'='*60}")
    print(f"  (Step 4) FEATURE DISTRIBUTION: benign-only, UNSW vs Gotham")
    print(f"{'='*60}")
    print(f"\n  {'Feature':<28} {'UNSW med':>10} {'UNSW IQR':>18} "
          f"{'GTH med':>10} {'GTH IQR':>18}  shift?")
    print("  " + "-" * 90)
    for _, row in fs.iterrows():
        flag = "*** SHIFT" if row["shift_flag"] else ""
        print(
            f"  {row['feature']:<28} "
            f"{row['unsw_median']:>10.3f} [{row['unsw_q25']:.3f}–{row['unsw_q75']:.3f}]  "
            f"{row['gotham_median']:>10.3f} [{row['gotham_q25']:.3f}–{row['gotham_q75']:.3f}]"
            f"  {flag}"
        )
    if len(flagged) == 0:
        print("\n  No non-overlapping features — benign distributions are broadly compatible.")
    else:
        print(f"\n  {len(flagged)} feature(s) with non-overlapping benign IQR (domain-shift drivers):")
        for feat in flagged["feature"].tolist():
            print(f"    {feat}")


# ── CLI ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Cross-dataset generalization evaluation: UNSW vs Gotham",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Prints three evaluation blocks (in-domain, cross-domain, pooled) with
PR-AUC, per-attack-cat recall, and benign FPR, plus the feature
distribution comparison table.
""",
    )
    ap.add_argument("--unsw",   required=False, default=None, metavar="PATH",
                    help="UNSW labeled corpus parquet (omit to run Gotham-only config A)")
    ap.add_argument("--gotham", required=True, metavar="PATH",
                    help="Gotham labeled corpus parquet")
    ap.add_argument("--test-size", type=float, default=0.20, metavar="FRAC",
                    help="Held-out fraction for in-domain/pooled splits (default 0.20)")
    ap.add_argument("--seed",   type=int, default=42,
                    help="Random seed (default 42)")
    ap.add_argument("--out-csv", metavar="PATH",
                    help="Optional: write feature-shift table to CSV")

    args = ap.parse_args()

    if XGBClassifier is None:
        print("ERROR: xgboost is required — pip install xgboost==1.7.6", file=sys.stderr)
        sys.exit(1)

    if args.unsw is None:
        # Gotham-only mode: run config (A) in-domain evaluation only
        print(f"Loading Gotham corpus: {args.gotham}")
        print("(--unsw not provided; running config A in-domain only)")
        df_gotham = pd.read_parquet(args.gotham)
        X_g, y_g, cats_g, ips_g = _to_xy(df_gotham)
        n_attack = int(y_g.sum())
        n_benign = len(y_g) - n_attack
        print(f"  Gotham: {len(X_g):,} rows  "
              f"(attack={n_attack:,}  benign={n_benign:,}  "
              f"prevalence={n_attack/max(len(y_g),1):.3f})")
        sss = StratifiedShuffleSplit(
            n_splits=1, test_size=args.test_size, random_state=args.seed,
        )
        tr_g, te_g = next(sss.split(X_g, y_g))
        result = _evaluate(
            X_g[tr_g], y_g[tr_g],
            X_g[te_g], y_g[te_g], cats_g[te_g],
            train_label="Gotham (80%)", test_label="Gotham (20%)", seed=args.seed,
            test_src_ips=ips_g[te_g],
        )
        _print_eval_block(
            "(A) IN-DOMAIN: train Gotham / test Gotham (ceiling)", result,
        )
    else:
        print(f"Loading UNSW corpus  : {args.unsw}")
        print(f"Loading Gotham corpus: {args.gotham}")
        results = run_cross_eval(
            args.unsw, args.gotham,
            test_size=args.test_size,
            seed=args.seed,
        )
        print_results(results)

        if args.out_csv:
            results["feature_shift"].to_csv(args.out_csv, index=False)
            print(f"\nFeature-shift table written to {args.out_csv}")
