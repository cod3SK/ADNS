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

from adns_flows import FEATURE_COLUMNS, orientation_key

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


_PROTO_NORM: dict[str, str] = {
    "tcp": "TCP", "udp": "UDP",
    "6":   "TCP", "17":  "UDP",
}


def _norm_proto(raw: str) -> str:
    """Normalise a protocol string to 'TCP' or 'UDP' (upper-case canonical form).

    Handles lowercase text ('tcp', 'udp'), numeric codes ('6', '17'), and
    already-normalised values ('TCP', 'UDP').
    """
    return _PROTO_NORM.get(str(raw).strip().lower(), str(raw).strip().upper())


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
            nthread=-1,
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
    agg = (
        pd.DataFrame({"ip": src_ips, "true": y_true, "pred": y_pred})
        .groupby("ip", sort=False)
        .agg(has_attack=("true", "any"), any_flagged=("pred", "any"))
    )
    attackers = agg[agg["has_attack"]]
    benign    = agg[~agg["has_attack"]]

    return {
        "attacker_host_recall": round(
            float(attackers["any_flagged"].mean()) if len(attackers) else float("nan"), 4,
        ),
        "benign_host_fpr": round(
            float(benign["any_flagged"].mean()) if len(benign) else 0.0, 4,
        ),
        "n_attacker_hosts": len(attackers),
        "n_benign_hosts":   len(benign),
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


# ── three-way feature distribution ───────────────────────────────────────

def feature_distribution_compare_three(
    df_unsw: pd.DataFrame,
    df_gotham: pd.DataFrame,
    df_cic: pd.DataFrame,
) -> pd.DataFrame:
    """Three-way benign feature distribution: UNSW vs Gotham vs CIC.

    For each FEATURE_COLUMN, computes median + IQR (Q25–Q75) for benign flows in
    each corpus.  shift_flag=True if ANY pair of corpora has non-overlapping IQR.

    Returns a DataFrame with columns:
      feature,
      unsw_median, unsw_q25, unsw_q75,
      gotham_median, gotham_q25, gotham_q75,
      cic_median, cic_q25, cic_q75,
      shift_flag
    """
    benign_u = df_unsw[df_unsw["label"] == 0]
    benign_g = df_gotham[df_gotham["label"] == 0]
    benign_c = df_cic[df_cic["label"] == 0]

    def _stats(s: pd.Series) -> tuple[float, float, float]:
        if len(s) == 0:
            nan = float("nan")
            return nan, nan, nan
        return float(s.quantile(0.25)), float(s.median()), float(s.quantile(0.75))

    def _overlap(a25: float, a75: float, b25: float, b75: float) -> bool:
        if any(np.isnan(v) for v in [a25, a75, b25, b75]):
            return True
        return max(a25, b25) <= min(a75, b75)

    records = []
    for col in FEATURE_COLUMNS:
        su = benign_u[col].dropna() if col in benign_u.columns else pd.Series(dtype=float)
        sg = benign_g[col].dropna() if col in benign_g.columns else pd.Series(dtype=float)
        sc = benign_c[col].dropna() if col in benign_c.columns else pd.Series(dtype=float)

        u_q25, u_med, u_q75 = _stats(su)
        g_q25, g_med, g_q75 = _stats(sg)
        c_q25, c_med, c_q75 = _stats(sc)

        any_shift = not (
            _overlap(u_q25, u_q75, g_q25, g_q75)
            and _overlap(u_q25, u_q75, c_q25, c_q75)
            and _overlap(g_q25, g_q75, c_q25, c_q75)
        )

        records.append({
            "feature":       col,
            "unsw_median":   round(u_med, 4),
            "unsw_q25":      round(u_q25, 4),
            "unsw_q75":      round(u_q75, 4),
            "gotham_median": round(g_med, 4),
            "gotham_q25":    round(g_q25, 4),
            "gotham_q75":    round(g_q75, 4),
            "cic_median":    round(c_med, 4),
            "cic_q25":       round(c_q25, 4),
            "cic_q75":       round(c_q75, 4),
            "shift_flag":    any_shift,
        })

    return pd.DataFrame(records)


# ── CIC held-out evaluation ───────────────────────────────────────────────

def run_cross_eval_with_cic(
    unsw_path: str | Path,
    gotham_path: str | Path,
    cic_path: str | Path,
    test_size: float = 0.20,
    seed: int = 42,
) -> dict:
    """Run held-out CIC generalization evaluation (E1, E2, E3).

    CIC is the HELD-OUT third environment — the key question is whether a model
    trained ONLY on UNSW+Gotham genuinely generalises.

    E1 (held-out):   train=UNSW+Gotham (full)        → test=CIC (never trained on)
    E2 (in-domain):  train=CIC (80%)                 → test=CIC (20%)  — CIC ceiling
    E3 (pooled all): train=UNSW+Gotham+CIC (80%)     → test=all three (20%)

    Parameters
    ----------
    unsw_path   : UNSW labeled corpus parquet
    gotham_path : Gotham labeled corpus parquet
    cic_path    : CIC-IDS2017 Tuesday labeled corpus parquet (HELD-OUT)
    test_size   : held-out fraction for E2 / E3 splits (default 0.20)
    seed        : random seed

    Returns
    -------
    dict with keys:
      e1_pooled_to_cic    : _evaluate() result dict — key generalization answer
      e2_cic_in_domain    : _evaluate() result dict — CIC ceiling
      e3_pooled_all_three : _evaluate() result dict — three-corpus ceiling
      feature_shift_cic   : DataFrame from feature_distribution_compare_three()
    """
    df_unsw   = pd.read_parquet(unsw_path)
    df_gotham = pd.read_parquet(gotham_path)
    df_cic    = pd.read_parquet(cic_path)

    X_u, y_u, cats_u, ips_u = _to_xy(df_unsw)
    X_g, y_g, cats_g, ips_g = _to_xy(df_gotham)
    X_c, y_c, cats_c, ips_c = _to_xy(df_cic)

    # E1: train=UNSW+Gotham (full), test=CIC — the held-out test
    X_ug = np.vstack([X_u, X_g])
    y_ug = np.concatenate([y_u, y_g])
    e1 = _evaluate(
        X_ug, y_ug,
        X_c, y_c, cats_c,
        train_label="UNSW+Gotham (full)", test_label="CIC (held-out)", seed=seed,
        test_src_ips=ips_c,
    )

    # E2: CIC in-domain ceiling
    sss_c = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    tr_c, te_c = next(sss_c.split(X_c, y_c))
    e2 = _evaluate(
        X_c[tr_c], y_c[tr_c],
        X_c[te_c], y_c[te_c], cats_c[te_c],
        train_label="CIC (80%)", test_label="CIC (20%)", seed=seed,
        test_src_ips=ips_c[te_c],
    )

    # E3: pooled all three
    X_all    = np.vstack([X_u, X_g, X_c])
    y_all    = np.concatenate([y_u, y_g, y_c])
    cats_all = np.concatenate([cats_u, cats_g, cats_c])
    ips_all  = np.concatenate([ips_u, ips_g, ips_c])

    sss_all = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    tr_all, te_all = next(sss_all.split(X_all, y_all))
    e3 = _evaluate(
        X_all[tr_all], y_all[tr_all],
        X_all[te_all], y_all[te_all], cats_all[te_all],
        train_label="UNSW+Gotham+CIC (80%)", test_label="UNSW+Gotham+CIC (20%)", seed=seed,
        test_src_ips=ips_all[te_all],
    )

    feature_shift_cic = feature_distribution_compare_three(df_unsw, df_gotham, df_cic)

    return {
        "e1_pooled_to_cic":    e1,
        "e2_cic_in_domain":    e2,
        "e3_pooled_all_three": e3,
        "feature_shift_cic":   feature_shift_cic,
    }


# ── label-match audit (STEP 0) ─────────────────────────────────────────────

def label_match_audit(
    corpus_day1: str | Path,
    corpus_day2: str | Path,
    label_csv: str | Path,
    time_tol: float = 1.0,
    capture_margin_sec: float = 300.0,
) -> dict:
    """Audit in-window GT attack event coverage in the UNSW corpus.

    The GT CSV (NUSW-NB15_GT.csv) covers all UNSW collection days
    (Jan 22, Jan 23, Feb 9–18, 2015).  Our pcap set only contains Jan 22
    (day-1) and Feb 17 (day-2).  The ~94% overall unmatched rate is therefore
    expected — most GT rows are from other collection days.

    This audit identifies ONLY the GT rows whose stime falls within the actual
    capture period of our pcaps, then checks what fraction of those in-window
    events are represented by >= 1 attack flow in the corpus (matched via
    orientation_key + proto + time window, identical to the build pipeline).

    A high in-window match rate (>= 90%) confirms the 66,607 attack labels are
    sound and that the ~34K remaining in-window unmatched rows are genuinely
    absent from the capture (different hosts / sessions outside the pcap).

    Parameters
    ----------
    corpus_day1, corpus_day2 : parquet paths for the two day builds
    label_csv                : UNSW-NB15 GT CSV (NUSW-NB15_GT.csv)
    time_tol                 : time tolerance matching the build pipeline (1.0s)
    capture_margin_sec       : stime must fall within [ts_min - margin, ts_max + margin]
                               to count as in-window (default 5 min)

    Returns
    -------
    dict with:
      capture_period_day1/2    : (ts_min, ts_max) for each day's corpus
      in_window_total          : GT rows with stime in either capture period
      in_window_matched        : count with >= 1 matching attack flow
      in_window_missed         : count with 0 matching flows
      match_rate               : in_window_matched / in_window_total
      per_cat_match_rates      : dict[attack_cat -> {in_window, matched, match_rate}]
      missed_examples          : up to 10 sample unmatched in-window GT rows
    """
    from datetime import datetime, timezone

    d1 = pd.read_parquet(corpus_day1)
    d2 = pd.read_parquet(corpus_day2)

    p1_min = float(d1["ts"].min()); p1_max = float(d1["ts"].max())
    p2_min = float(d2["ts"].min()); p2_max = float(d2["ts"].max())

    # Build attack-flow index per day: orientation_key -> [(proto_upper, ts)]
    def _build_attack_idx(df: pd.DataFrame) -> dict:
        atk = df[df["label"] == 1]
        idx: dict = {}
        for r in atk.itertuples(index=False):
            k = orientation_key(r.src_ip, int(r.src_port), r.dst_ip, int(r.dst_port))
            idx.setdefault(k, []).append((_norm_proto(r.proto), float(r.ts)))
        return idx

    idx1 = _build_attack_idx(d1)
    idx2 = _build_attack_idx(d2)

    # Load and normalise GT CSV (same alias logic as build_corpus.load_label_index)
    gt = pd.read_csv(label_csv, low_memory=False)
    gt.columns = [c.strip().lower().replace(" ", "_") for c in gt.columns]
    _GT_ALIASES = {
        "source_ip": "srcip", "src_ip": "srcip",
        "destination_ip": "dstip", "dst_ip": "dstip",
        "source_port": "sport", "src_port": "sport",
        "destination_port": "dsport", "dst_port": "dsport",
        "protocol": "proto", "start_time": "stime", "last_time": "ltime",
        "attack_category": "attack_cat",
    }
    gt = gt.rename(columns={k: v for k, v in _GT_ALIASES.items() if k in gt.columns})

    gt["stime"]  = pd.to_numeric(gt["stime"],  errors="coerce")
    gt["ltime"]  = pd.to_numeric(gt["ltime"],  errors="coerce")
    gt["sport"]  = pd.to_numeric(gt.get("sport",  pd.Series(dtype=float)), errors="coerce").fillna(0).astype(int)
    gt["dsport"] = pd.to_numeric(gt.get("dsport", pd.Series(dtype=float)), errors="coerce").fillna(0).astype(int)
    gt = gt.dropna(subset=["stime", "ltime", "srcip", "dstip"])

    # Assign each GT row to a capture day (or None if out-of-window).
    # Use plain column names (no leading underscore) — pandas itertuples
    # silently renames columns starting with '_' and breaks attribute access.
    m1 = capture_margin_sec
    gt["cap_day"] = None
    in_d1 = (gt["stime"] >= p1_min - m1) & (gt["stime"] <= p1_max + m1)
    in_d2 = (gt["stime"] >= p2_min - m1) & (gt["stime"] <= p2_max + m1)
    gt.loc[in_d1, "cap_day"] = "day1"
    gt.loc[in_d2 & ~in_d1, "cap_day"] = "day2"
    gt_iw = gt[gt["cap_day"].notna()].copy()

    # Precompute orientation keys for all in-window GT rows
    gt_iw["orient_key"] = [
        orientation_key(r.srcip, int(r.sport), r.dstip, int(r.dsport))
        for r in gt_iw.itertuples(index=False)
    ]

    # Check match: any corpus attack flow with same key+proto within time window?
    def _is_matched(r) -> bool:
        idx = idx1 if r.cap_day == "day1" else idx2
        candidates = idx.get(r.orient_key, [])
        if not candidates:
            return False
        proto = _norm_proto(r.proto)
        st, lt = float(r.stime) - time_tol, float(r.ltime) + time_tol
        return any(p == proto and st <= ts <= lt for p, ts in candidates)

    gt_iw["is_matched"] = [_is_matched(r) for r in gt_iw.itertuples(index=False)]

    total   = len(gt_iw)
    n_match = int(gt_iw["is_matched"].sum())
    n_miss  = total - n_match

    # Per-attack_cat
    cat_col = "attack_cat" if "attack_cat" in gt_iw.columns else "attackcat"
    per_cat: dict = {}
    for cat, grp in gt_iw.groupby(cat_col, dropna=False):
        cat_str = str(cat).strip() if cat and str(cat).strip() else "(unknown)"
        n  = len(grp)
        m  = int(grp["is_matched"].sum())
        per_cat[cat_str] = {
            "in_window": n,
            "matched":   m,
            "missed":    n - m,
            "match_rate": round(m / max(n, 1), 4),
        }

    missed_cols = [c for c in ["srcip","sport","dstip","dsport","proto","stime","ltime",cat_col]
                   if c in gt_iw.columns]
    missed_examples = (
        gt_iw[~gt_iw["is_matched"]][missed_cols].head(10).to_dict("records")
    )

    def _fmt_epoch(t: float) -> str:
        return datetime.fromtimestamp(t, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    return {
        "capture_period_day1": (p1_min, p1_max),
        "capture_period_day2": (p2_min, p2_max),
        "capture_period_day1_fmt": f"[{_fmt_epoch(p1_min)}, {_fmt_epoch(p1_max)}]",
        "capture_period_day2_fmt": f"[{_fmt_epoch(p2_min)}, {_fmt_epoch(p2_max)}]",
        "in_window_total":    total,
        "in_window_matched":  n_match,
        "in_window_missed":   n_miss,
        "match_rate":         round(n_match / max(total, 1), 4),
        "per_cat_match_rates": per_cat,
        "missed_examples":    missed_examples,
    }


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

    # (A) In-domain UNSW — 80/20 stratified split
    sss_u = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    tr_u, te_u = next(sss_u.split(X_u, y_u))
    in_domain_unsw = _evaluate(
        X_u[tr_u], y_u[tr_u],
        X_u[te_u], y_u[te_u], cats_u[te_u],
        train_label="UNSW (80%)", test_label="UNSW (20%)", seed=seed,
        test_src_ips=ips_u[te_u],
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
        "in_domain_unsw":       in_domain_unsw,
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


def _print_audit_block(audit: dict) -> None:
    """Print label-match audit (STEP 0) to stdout."""
    print(f"\n{'='*60}")
    print("  (STEP 0) LABEL-MATCH AUDIT — in-window GT coverage")
    print(f"{'='*60}")
    print(f"  Day-1 capture period : {audit['capture_period_day1_fmt']}")
    print(f"  Day-2 capture period : {audit['capture_period_day2_fmt']}")
    print()
    total   = audit["in_window_total"]
    matched = audit["in_window_matched"]
    missed  = audit["in_window_missed"]
    rate    = audit["match_rate"]
    print(f"  In-window GT rows    : {total:,}")
    print(f"  Matched (>=1 flow)   : {matched:,}  ({100*rate:.1f}%)")
    print(f"  Missed  (0 flows)    : {missed:,}  ({100*(1-rate):.1f}%)")
    print()
    print(f"  Per-attack-cat (in-window rows):")
    print(f"  {'Category':<25} {'In-win':>7} {'Matched':>8} {'Missed':>7} {'Rate':>7}")
    print("  " + "-" * 58)
    for cat, d in sorted(audit["per_cat_match_rates"].items()):
        print(f"  {cat:<25} {d['in_window']:>7,} {d['matched']:>8,} "
              f"{d['missed']:>7,} {d['match_rate']:>7.1%}")
    if missed > 0 and audit.get("missed_examples"):
        print(f"\n  Sample unmatched in-window rows (up to 10):")
        for ex in audit["missed_examples"]:
            print(f"    {ex}")
    verdict = (
        "PASS — attack labels are sound"
        if rate >= 0.90
        else f"INVESTIGATE — {100*(1-rate):.1f}% of in-window GT events have no matching flow"
    )
    print(f"\n  Verdict: {verdict}")


def print_results(results: dict) -> None:
    """Print all evaluation results to stdout."""
    if "label_match_audit" in results:
        _print_audit_block(results["label_match_audit"])
    _print_eval_block("(A) IN-DOMAIN: train Gotham / test Gotham (ceiling)",
                      results["in_domain_gotham"])
    _print_eval_block("(A) IN-DOMAIN: train UNSW / test UNSW (ceiling)",
                      results["in_domain_unsw"])
    _print_eval_block("(B) CROSS-DOMAIN: train UNSW / test Gotham",
                      results["cross_unsw_to_gotham"])
    _print_eval_block("(B) CROSS-DOMAIN: train Gotham / test UNSW",
                      results["cross_gotham_to_unsw"])
    _print_eval_block("(D) POOLED: train UNSW+Gotham / test UNSW+Gotham",
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


def print_results_with_cic(results: dict) -> None:
    """Print E1 / E2 / E3 evaluation results and three-way feature shift table."""
    _print_eval_block(
        "(E1) HELD-OUT: train UNSW+Gotham / test CIC  [KEY GENERALIZATION TEST]",
        results["e1_pooled_to_cic"],
    )
    _print_eval_block(
        "(E2) IN-DOMAIN: train CIC / test CIC (ceiling)",
        results["e2_cic_in_domain"],
    )
    _print_eval_block(
        "(E3) POOLED ALL THREE: train UNSW+Gotham+CIC / test all",
        results["e3_pooled_all_three"],
    )

    fs = results["feature_shift_cic"]
    flagged = fs[fs["shift_flag"]]
    print(f"\n{'='*60}")
    print(f"  (STEP 5) FEATURE DISTRIBUTION: benign-only, UNSW vs Gotham vs CIC")
    print(f"{'='*60}")
    print(
        f"\n  {'Feature':<28} {'UNSW med':>9} {'UNSW IQR':>16}  "
        f"{'GTH med':>9} {'GTH IQR':>16}  "
        f"{'CIC med':>9} {'CIC IQR':>16}  shift?"
    )
    print("  " + "-" * 115)
    for _, row in fs.iterrows():
        flag = "*** SHIFT" if row["shift_flag"] else ""
        print(
            f"  {row['feature']:<28} "
            f"{row['unsw_median']:>9.3f} [{row['unsw_q25']:.3f}–{row['unsw_q75']:.3f}]  "
            f"{row['gotham_median']:>9.3f} [{row['gotham_q25']:.3f}–{row['gotham_q75']:.3f}]  "
            f"{row['cic_median']:>9.3f} [{row['cic_q25']:.3f}–{row['cic_q75']:.3f}]  "
            f"{flag}"
        )
    if len(flagged) == 0:
        print("\n  No non-overlapping features across any pair.")
    else:
        print(f"\n  {len(flagged)} feature(s) with non-overlapping IQR in at least one pair:")
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
    ap.add_argument("--cic", metavar="PATH", default=None,
                    help="CIC-IDS2017 Tuesday corpus parquet "
                         "(held-out test; triggers E1/E2/E3 configs)")
    ap.add_argument("--test-size", type=float, default=0.20, metavar="FRAC",
                    help="Held-out fraction for in-domain/pooled splits (default 0.20)")
    ap.add_argument("--seed",   type=int, default=42,
                    help="Random seed (default 42)")
    ap.add_argument("--out-csv", metavar="PATH",
                    help="Optional: write feature-shift table to CSV")
    ap.add_argument("--label-csv", metavar="PATH", default=None,
                    help="(STEP 0) UNSW GT CSV path for label-match audit "
                         "(requires --corpus-day1 and --corpus-day2)")
    ap.add_argument("--corpus-day1", metavar="PATH", default=None,
                    help="(STEP 0) unsw_day1.parquet path for label-match audit")
    ap.add_argument("--corpus-day2", metavar="PATH", default=None,
                    help="(STEP 0) unsw_day2.parquet path for label-match audit")

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

        # STEP 0: label-match audit (requires all three extra args)
        if args.label_csv and args.corpus_day1 and args.corpus_day2:
            print("\nRunning label-match audit (STEP 0)…")
            audit = label_match_audit(
                args.corpus_day1, args.corpus_day2, args.label_csv,
            )
            results["label_match_audit"] = audit
        elif any([args.label_csv, args.corpus_day1, args.corpus_day2]):
            print("WARNING: --label-csv, --corpus-day1, and --corpus-day2 must all "
                  "be provided to run the label-match audit. Skipping.")

        print_results(results)

        if args.out_csv:
            results["feature_shift"].to_csv(args.out_csv, index=False)
            print(f"\nFeature-shift table written to {args.out_csv}")

        # CIC held-out eval (E1/E2/E3) — runs after A/B/D if --cic is given
        if args.cic:
            print(f"\nLoading CIC corpus   : {args.cic}")
            cic_results = run_cross_eval_with_cic(
                args.unsw, args.gotham, args.cic,
                test_size=args.test_size,
                seed=args.seed,
            )
            print_results_with_cic(cic_results)
            if args.out_csv:
                cic_csv = args.out_csv.replace(".csv", "_cic.csv")
                cic_results["feature_shift_cic"].to_csv(cic_csv, index=False)
                print(f"\nCIC feature-shift table written to {cic_csv}")
