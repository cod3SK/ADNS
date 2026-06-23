"""
ADNS Pipeline Diagnostic
========================
Captures 30 s of live benign traffic, extracts with the exact serve-time
NFStream config, compares against the training corpus, and scores it.

Run from the repo root:
    python diag_pipeline.py

Requires admin / elevated process (Npcap capture needs raw socket access).
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent
CORPUS_DIR  = ROOT / "outputs" / "corpus"
MODEL_PATH  = ROOT / "api" / "model_artifacts" / "nfstream_model.joblib"
PCAP_PATH   = ROOT / "live_benign_30s.pcap"
CSV_PATH    = ROOT / "live_benign_extracted.csv"

# Add ml/ to path so we can import adns_flows without installing the package.
sys.path.insert(0, str(ROOT / "ml"))
sys.path.insert(0, str(ROOT / "api"))

TSHARK      = r"C:\Program Files\Wireshark\tshark.exe"
INTERFACE   = r"\Device\NPF_{E466F43A-35D6-409B-AC2B-A026C362E238}"  # Wi-Fi

CORPUS_FILES = {
    "UNSW-NB15":    CORPUS_DIR / "unsw_flows.parquet",
    "Gotham 2025":  CORPUS_DIR / "gotham_flows.parquet",
    "CIC-IDS2017":  CORPUS_DIR / "cic_tuesday_flows.parquet",
}

SCORE_THRESHOLD = 0.82   # production threshold from ADNS

SEP = "=" * 62


def banner(msg: str) -> None:
    print(f"\n{SEP}\n  {msg}\n{SEP}")


# ── STEP 1 — CAPTURE ──────────────────────────────────────────────────────────

def step1_capture() -> None:
    banner("STEP 1 — LIVE CAPTURE (30 s)")

    if not Path(TSHARK).is_file():
        sys.exit(f"[FATAL] tshark not found at {TSHARK}")

    if PCAP_PATH.exists():
        print(f"[SKIP] {PCAP_PATH} already exists ({PCAP_PATH.stat().st_size // 1024} KB) — delete to re-capture")
        return

    cmd = [
        TSHARK,
        "-i", INTERFACE,
        "-w", str(PCAP_PATH),
        "-a", "duration:30",
    ]
    print(f"[RUN ] {' '.join(cmd)}")
    print("       Capturing for 30 seconds — browse normally, nothing malicious …")
    t0 = time.time()
    proc = subprocess.run(
        cmd,
        capture_output=True,
        cwd=str(Path(TSHARK).parent),
        env={**os.environ, "PATH": str(Path(TSHARK).parent) + os.pathsep + os.environ.get("PATH", "")},
    )
    elapsed = time.time() - t0
    if proc.returncode not in (0, 1):   # tshark exits 1 on signal-triggered stop
        print(f"[WARN] tshark exited {proc.returncode}")
        stderr = proc.stderr.decode("utf-8", errors="replace").strip()
        if stderr:
            print("[STDERR]", stderr[:400])

    if PCAP_PATH.exists():
        kb = PCAP_PATH.stat().st_size // 1024
        print(f"[OK  ] Captured {kb} KB in {elapsed:.1f} s -> {PCAP_PATH}")
    else:
        sys.exit("[FATAL] PCAP file not created. Is the process running as Administrator?")


# ── STEP 2 — EXTRACT ─────────────────────────────────────────────────────────

def step2_extract() -> pd.DataFrame:
    banner("STEP 2 — NFStream EXTRACTION")

    from adns_flows.extract_nfstream import extract_flows_nfstream, flows_to_dataframe_nfstream
    from adns_flows.schema import FEATURE_COLUMNS, IDENTITY_COLUMNS, SchemaError, validate_matrix

    print(f"[RUN ] NFStreamer(source={PCAP_PATH}, statistical_analysis=True, n_dissections=0, "
          f"idle_timeout=120, active_timeout=1800, n_meters=1)")

    flows = extract_flows_nfstream(str(PCAP_PATH), n_meters=1)
    print(f"[INFO] Raw flows extracted: {len(flows)}")

    if not flows:
        sys.exit("[FATAL] No flows extracted. Is the PCAP empty or too short?")

    df = flows_to_dataframe_nfstream(flows)

    # Explicit validate_matrix check (the function already calls it internally,
    # but we call it again here to surface any error loudly with context).
    try:
        validate_matrix(df, list(df.columns[len(IDENTITY_COLUMNS):]))
        print(f"[OK  ] validate_matrix: PASS — columns match FEATURE_COLUMNS exactly")
    except SchemaError as exc:
        print(f"[FAIL] validate_matrix: {exc}")
        sys.exit(1)

    nan_report = df[list(FEATURE_COLUMNS)].isna().sum()
    nan_cols = nan_report[nan_report > 0]
    if nan_cols.empty:
        print("[OK  ] NaN check: no NaN values in any feature column")
    else:
        print(f"[WARN] NaN columns detected:\n{nan_cols.to_string()}")

    df.to_csv(str(CSV_PATH), index=False)
    print(f"[OK  ] Saved {len(df)} rows → {CSV_PATH}")
    return df


# ── STEP 3 — LOAD TRAINING CORPUS ────────────────────────────────────────────

def step3_load_corpus() -> pd.DataFrame:
    banner("STEP 3 — TRAINING CORPUS")

    from adns_flows.schema import FEATURE_COLUMNS

    frames = []
    for name, path in CORPUS_FILES.items():
        if not path.exists():
            print(f"[WARN] {name}: {path} not found — skipping")
            continue
        df = pd.read_parquet(str(path), columns=list(FEATURE_COLUMNS) + ["label"])
        attacks = int((df["label"] == 1).sum())
        benign  = int((df["label"] == 0).sum())
        print(f"[OK  ] {name:<16}: {len(df):>9,} rows  attack={attacks:>8,}  benign={benign:>8,}")
        frames.append(df)

    if not frames:
        sys.exit("[FATAL] No corpus parquets found. Expected in outputs/corpus/")

    pool = pd.concat(frames, ignore_index=True)
    print(f"\n[SUM ] Total pooled: {len(pool):,} rows")

    # Protocol breakdown
    proto_map = {0: "OTHER", 1: "ICMP", 6: "TCP", 17: "UDP"}
    proto_counts = pool["proto"].value_counts(normalize=True) * 100
    proto_str = "  ".join(
        f"{proto_map.get(int(p), str(p))}={v:.1f}%"
        for p, v in proto_counts.items()
    )
    print(f"[INFO] Protocol (training): {proto_str}")

    return pool


# ── STEP 4 — FEATURE COMPARISON ──────────────────────────────────────────────

def step4_compare(live_df: pd.DataFrame, corpus_df: pd.DataFrame) -> None:
    banner("STEP 4 — FEATURE COMPARISON")

    from adns_flows.schema import FEATURE_COLUMNS

    # Proto breakdown for live
    proto_map = {0: "OTHER", 1: "ICMP", 6: "TCP", 17: "UDP"}
    proto_counts = live_df["proto"].value_counts(normalize=True) * 100
    proto_str = "  ".join(
        f"{proto_map.get(int(p), str(p))}={v:.1f}%"
        for p, v in proto_counts.items()
    )
    print(f"[INFO] Protocol (live):     {proto_str}\n")

    # Compare top features by training variance (pick 8 most informative)
    corpus_feat = corpus_df[list(FEATURE_COLUMNS)]
    variances   = corpus_feat.var().sort_values(ascending=False)
    top_feats   = list(variances.index[:8]) + ["syn_count", "dst_port_bucket", "proto"]
    seen = set()
    compare_feats = [f for f in top_feats if not (f in seen or seen.add(f))]

    hdr = f"{'Feature':<22}  {'Train mean':>12}  {'Train std':>10}  {'Live mean':>12}  {'Live std':>10}  {'Overlap?':>8}"
    print(hdr)
    print("-" * len(hdr))

    for feat in compare_feats:
        if feat not in FEATURE_COLUMNS:
            continue
        t_vals = corpus_feat[feat].dropna()
        l_vals = live_df[feat].dropna()
        if t_vals.empty or l_vals.empty:
            continue

        t_mean, t_std = t_vals.mean(), t_vals.std()
        l_mean, l_std = l_vals.mean(), l_vals.std()

        # Simple range-overlap check: live mean within ±3σ of training
        t3_lo = t_mean - 3 * t_std
        t3_hi = t_mean + 3 * t_std
        overlap = "YES" if t3_lo <= l_mean <= t3_hi else "SHIFTED"

        print(f"{feat:<22}  {t_mean:>12.2f}  {t_std:>10.2f}  {l_mean:>12.2f}  {l_std:>10.2f}  {overlap:>8}")


# ── STEP 5 — SCORE ────────────────────────────────────────────────────────────

def step5_score(live_df: pd.DataFrame) -> np.ndarray:
    banner("STEP 5 — SCORING LIVE BENIGN TRAFFIC")

    from adns_flows.schema import FEATURE_COLUMNS

    if not MODEL_PATH.exists():
        sys.exit(
            f"[FATAL] Model not found at {MODEL_PATH}\n"
            "        Run:  git lfs pull\n"
            "        Or:   python ml/train_nfstream.py"
        )

    import joblib
    bundle = joblib.load(str(MODEL_PATH))
    print(f"[OK  ] Model loaded: {MODEL_PATH.stat().st_size // (1024*1024)} MB")

    # The model bundle is a dict {"xgb": ..., "et": ...} or a single estimator.
    if isinstance(bundle, dict):
        model = bundle.get("xgb") or bundle.get("et") or next(iter(bundle.values()))
        print(f"[INFO] Using model key: {[k for k,v in bundle.items() if v is model][0]}")
    else:
        model = bundle

    feat_mat = live_df[list(FEATURE_COLUMNS)].values.astype(float)
    print(f"[INFO] Feature matrix: {feat_mat.shape[0]} rows × {feat_mat.shape[1]} cols")

    scores: np.ndarray = model.predict_proba(feat_mat)[:, 1]

    flagged = int((scores > SCORE_THRESHOLD).sum())
    fpr = flagged / len(scores) * 100 if len(scores) else 0.0

    print(f"\n[RESULT] Score distribution (n={len(scores)}):")
    print(f"         Mean  : {scores.mean():.4f}")
    print(f"         Std   : {scores.std():.4f}")
    print(f"         Min   : {scores.min():.4f}")
    print(f"         Max   : {scores.max():.4f}")
    print(f"         FPR (score > {SCORE_THRESHOLD}): {flagged}/{len(scores)} = {fpr:.2f}%")

    # ASCII histogram
    print("\n[HIST ] Score distribution (live benign):")
    bins = np.linspace(0, 1, 11)
    hist, edges = np.histogram(scores, bins=bins)
    max_bar = max(hist) or 1
    bar_width = 40
    for i, count in enumerate(hist):
        lo, hi = edges[i], edges[i+1]
        bar = "█" * int(count / max_bar * bar_width)
        print(f"  [{lo:.1f}–{hi:.1f}]  {bar:<40}  {count:>5}")

    return scores


# ── STEP 6 — SANITY CHECK ────────────────────────────────────────────────────

def step6_sanity(corpus_df: pd.DataFrame) -> None:
    banner("STEP 6 — SANITY CHECK (known attack flow)")

    from adns_flows.schema import FEATURE_COLUMNS

    if not MODEL_PATH.exists():
        print("[SKIP] Model missing — cannot sanity check")
        return

    import joblib
    bundle = joblib.load(str(MODEL_PATH))
    if isinstance(bundle, dict):
        model = next(iter(bundle.values()))
    else:
        model = bundle

    attacks = corpus_df[corpus_df["label"] == 1][list(FEATURE_COLUMNS)]
    if attacks.empty:
        print("[SKIP] No attack rows in pooled corpus — skipping sanity check")
        return

    # Pick 5 diverse attack rows
    sample = attacks.sample(n=min(5, len(attacks)), random_state=42)
    feat_mat = sample.values.astype(float)
    attack_scores = model.predict_proba(feat_mat)[:, 1]

    print(f"[INFO] Attack sample scores (n={len(attack_scores)}):")
    for i, s in enumerate(attack_scores):
        verdict = "GOOD" if s > 0.5 else "MISS"
        print(f"         Row {i+1}: {s:.4f}  [{verdict}]")

    high_recall = float((attack_scores > 0.5).mean()) * 100
    print(f"\n[INFO] Recall on 5-row sample (threshold=0.5): {high_recall:.0f}%")
    discriminates = high_recall >= 80
    print(f"[INFO] Model is discriminating: {'YES' if discriminates else 'NO — model may be broken'}")


# ── FINAL REPORT ─────────────────────────────────────────────────────────────

def final_report(live_df: pd.DataFrame, corpus_df: pd.DataFrame, scores: np.ndarray) -> None:
    banner("ADNS PIPELINE DIAGNOSTIC RESULTS")

    from adns_flows.schema import FEATURE_COLUMNS

    flagged = int((scores > SCORE_THRESHOLD).sum())
    fpr = flagged / len(scores) * 100 if len(scores) else 0.0

    nan_cols = list(live_df[list(FEATURE_COLUMNS)].columns[live_df[list(FEATURE_COLUMNS)].isna().any()])

    print(f"""
1. SCHEMA & EXTRACTION
   live_benign_extracted.csv rows : {len(live_df)}
   All 21 features present        : {'YES' if list(live_df.columns[5:]) == list(FEATURE_COLUMNS) else 'NO'}
   validate_matrix result         : OK
   Any NaN columns                : {', '.join(nan_cols) if nan_cols else 'None'}

2. TRAINING CORPUS STATS
   Total rows pooled              : {len(corpus_df):,}
   Protocol breakdown             : see Step 4 above

3. FEATURE COMPARISON
   (see Step 4 table above)

4. LIVE BENIGN TRAFFIC SCORES
   Mean anomaly score             : {scores.mean():.4f}
   Std                            : {scores.std():.4f}
   Min / Max                      : {scores.min():.4f} / {scores.max():.4f}
   FPR (score > {SCORE_THRESHOLD})           : {flagged}/{len(scores)} = {fpr:.2f}%

5. DIAGNOSIS
   Schema valid?                  : {'GO' if not nan_cols else 'WARNING'}
   FPR on benign < 5%?            : {'GO' if fpr < 5.0 else 'NO-GO — investigate model or threshold'}
   Next step                      : {'Model is working as expected' if fpr < 5.0 else
                                     'FPR > 5% — check feature shift (Step 4) or retrain'}
""")


# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()

    print(SEP)
    print("  ADNS PIPELINE DIAGNOSTIC")
    print(f"  ROOT: {ROOT}")
    print(SEP)

    step1_capture()
    live_df   = step2_extract()
    corpus_df = step3_load_corpus()
    step4_compare(live_df, corpus_df)
    scores    = step5_score(live_df)
    step6_sanity(corpus_df)
    final_report(live_df, corpus_df, scores)
