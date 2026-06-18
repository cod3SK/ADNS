"""
Corpus quality report for the labeled flow corpus.

Detects constant/all-zero features (signals extraction bugs), reports class
balance, per-category counts, and per-feature statistics split by label.

Usage
-----
    import pandas as pd
    from corpus.report import print_report, print_stats
    from corpus.build_corpus import CorpusStats

    df    = pd.read_parquet("outputs/corpus/unsw_flows.parquet")
    stats = CorpusStats(...)   # returned by build_corpus()

    print_stats(stats)   # three-way labeling counters + label-row accounting
    print_report(df)     # feature distribution report
"""
from __future__ import annotations

import pandas as pd

from adns_flows import FEATURE_COLUMNS


def class_balance(df: pd.DataFrame) -> dict[str, int | float]:
    counts = df["label"].value_counts().to_dict()
    total = len(df)
    return {
        "total":      total,
        "benign":     int(counts.get(0, 0)),
        "attack":     int(counts.get(1, 0)),
        "attack_pct": round(100.0 * counts.get(1, 0) / max(total, 1), 2),
    }


def per_attack_cat_counts(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("attack_cat")["label"]
        .agg(count="count", attacks="sum")
        .reset_index()
        .sort_values("count", ascending=False)
    )


def feature_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Per-feature min / median / max split by label=0 and label=1."""
    records = []
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            continue
        for lbl in (0, 1):
            sub = df.loc[df["label"] == lbl, col]
            records.append({
                "feature": col,
                "label":   lbl,
                "min":     sub.min(),
                "median":  sub.median(),
                "max":     sub.max(),
            })
    return pd.DataFrame(records)


def flag_constant_or_zero_features(df: pd.DataFrame) -> list[str]:
    """Return feature names that are constant or all-zero — signal extraction bugs."""
    flagged = []
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            continue
        s = df[col]
        if s.nunique() <= 1 or s.eq(0).all():
            flagged.append(col)
    return flagged


def print_stats(stats: object) -> None:
    """Print the three-way labeling counters and label-row accounting from a CorpusStats."""
    total = getattr(stats, "total_kept", 0)
    n_attack = getattr(stats, "n_attack", 0)
    n_benign = getattr(stats, "n_benign", 0)
    n_dropped = getattr(stats, "n_dropped_unprocessable", 0)
    dropped_reasons = getattr(stats, "dropped_reasons", {})
    label_rows_total    = getattr(stats, "label_rows_total", 0)
    label_rows_matched  = getattr(stats, "label_rows_matched", 0)
    label_rows_unmatched = getattr(stats, "label_rows_unmatched", 0)

    attack_pct = 100.0 * n_attack / max(total, 1)
    benign_pct = 100.0 * n_benign / max(total, 1)
    unmatched_pct = 100.0 * label_rows_unmatched / max(label_rows_total, 1)

    print("\n=== Corpus labeling stats ===")
    print(f"  Kept   attack (label=1) : {n_attack:>8,}  ({attack_pct:.1f}%)")
    print(f"  Kept   benign (label=0) : {n_benign:>8,}  ({benign_pct:.1f}%)  "
          "[includes all no-match flows]")
    print(f"  Dropped (unprocessable) : {n_dropped:>8,}  {dropped_reasons}")

    print(f"\n=== Label-row accounting (GT attack rows) ===")
    print(f"  Total attack rows in GT  : {label_rows_total:,}")
    print(f"  Matched by >=1 flow      : {label_rows_matched:,}")
    print(f"  Unmatched (0 flows)      : {label_rows_unmatched:,}  ({unmatched_pct:.1f}%)")
    if label_rows_total > 0 and label_rows_unmatched / label_rows_total > 0.20:
        print("  [WARNING] >20% attack rows unmatched — check epoch reconstruction")


def print_report(df: pd.DataFrame) -> None:
    bal = class_balance(df)
    print("\n=== Corpus summary ===")
    print(f"  Total rows : {bal['total']:,}")
    print(f"  Benign     : {bal['benign']:,}")
    print(f"  Attack     : {bal['attack']:,}  ({bal['attack_pct']}%)")

    print("\n=== Per-attack-category counts ===")
    cats = per_attack_cat_counts(df)
    print(cats.to_string(index=False))

    bad = flag_constant_or_zero_features(df)
    if bad:
        print(f"\n[WARNING] Constant/all-zero features (possible extraction bug): {bad}")
    else:
        print("\n[OK] No constant/all-zero features detected")

    print("\n=== Feature stats by label ===")
    stats = feature_stats(df)
    print(stats.to_string(index=False))
