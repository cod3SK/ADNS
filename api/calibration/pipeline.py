"""Deployment-time calibration pipeline for ADNS.

Full flow: capture → extract → filter → retrain → validate → swap model.
Runs in a daemon thread; state is polled via calibration_routes.

STAGE MAP                     PROGRESS
  idle              —         0
  capturing         §1        5–45
  extracting        §2        48–55
  filtering         §3        58–63
  retraining        §4        65–90
  validating        §5        91–99
  done              —         100
  failed            —         preserved at failure point
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ── Paths (freeze-aware) ──────────────────────────────────────────────────────
# In the frozen exe (PyInstaller), __file__ resolves to a read-only location
# inside sys._MEIPASS.  Writable artifacts (calibrated model, capture PCAP)
# must go to %APPDATA%\ADNS instead.  Corpus parquets are bundled in the exe
# at sys._MEIPASS/corpus/ so calibration works offline.
_FROZEN = getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS")

if _FROZEN:
    _MEIPASS = Path(sys._MEIPASS)
    _APPDATA = Path(os.environ.get("APPDATA") or Path.home()) / "ADNS"
    MODEL_PATH      = _MEIPASS / "model_artifacts" / "nfstream_model.joblib"
    CALIBRATED_PATH = _APPDATA / "nfstream_model_calibrated.joblib"
    INSTANCE_DIR    = _APPDATA / "instance"
    CORPUS_PATHS: dict[str, Path] = {
        "unsw":   _MEIPASS / "corpus" / "unsw_flows.parquet",
        "gotham": _MEIPASS / "corpus" / "gotham_flows.parquet",
        "cic":    _MEIPASS / "corpus" / "cic_tuesday_flows.parquet",
    }
else:
    _ROOT = Path(__file__).resolve().parent.parent.parent
    MODEL_PATH      = _ROOT / "api" / "model_artifacts" / "nfstream_model.joblib"
    CALIBRATED_PATH = _ROOT / "api" / "model_artifacts" / "nfstream_model_calibrated.joblib"
    INSTANCE_DIR    = _ROOT / "api" / "instance"
    CORPUS_PATHS: dict[str, Path] = {
        "unsw":   _ROOT / "outputs" / "corpus" / "unsw_flows.parquet",
        "gotham": _ROOT / "outputs" / "corpus" / "gotham_flows.parquet",
        "cic":    _ROOT / "outputs" / "corpus" / "cic_tuesday_flows.parquet",
    }

CAPTURE_PCAP    = INSTANCE_DIR / "calibration_traffic.pcap"
CALIBRATION_CSV = INSTANCE_DIR / "calibration_flows.csv"

# ── Retrain hyper-parameters ──────────────────────────────────────────────────
ATTACK_SAMPLE_PER_CORPUS = 33_000  # ~100 k total across three corpora
GATE_ATTACK_SCORE_MIN    = 0.80    # Gate 5B: each sample attack must exceed this
GATE_FPR_MAX             = 0.05    # Gate 5C: local benign FPR must be below this
GATE_RECALL_MIN          = 0.90    # Gate 5D: corpus attack recall must be above this

# ── Pipeline state ────────────────────────────────────────────────────────────
_STATE: dict[str, Any] = {
    "stage":             "idle",
    "progress":          0,
    "message":           "",
    "error":             None,
    "result":            None,
    "cancel_requested":  False,
    "capture_duration":  1800,
    "flows_captured":    0,
    "timer_start":       None,
}
_LOCK = threading.Lock()
_pipeline_thread: threading.Thread | None = None


# ── Public API ─────────────────────────────────────────────────────────────────

def get_state() -> dict:
    with _LOCK:
        return dict(_STATE)


def start_pipeline(
    interface: str,
    *,
    capture_duration: int = 1800,
    tshark_bin: str | None = None,
) -> None:
    """Start the calibration pipeline in a background thread.

    Raises RuntimeError if a pipeline run is already in progress.
    """
    global _pipeline_thread
    with _LOCK:
        stage = _STATE["stage"]
        if stage not in ("idle", "done", "failed"):
            raise RuntimeError(f"pipeline already running (stage={stage!r})")
        _STATE.update({
            "stage":             "capturing",
            "progress":          0,
            "message":           "Preparing capture...",
            "error":             None,
            "result":            None,
            "cancel_requested":  False,
            "capture_duration":  capture_duration,
            "flows_captured":    0,
            "timer_start":       time.time(),
        })

    _pipeline_thread = threading.Thread(
        target=_run,
        args=(interface, capture_duration, tshark_bin),
        daemon=True,
        name="calibration-pipeline",
    )
    _pipeline_thread.start()


def cancel_pipeline() -> None:
    with _LOCK:
        _STATE["cancel_requested"] = True


def reset_pipeline() -> None:
    """Reset state to idle (only allowed when not actively running)."""
    with _LOCK:
        if _STATE["stage"] in ("capturing", "extracting", "filtering", "retraining", "validating"):
            _STATE["cancel_requested"] = True
        _STATE.update({"stage": "idle", "progress": 0, "message": "", "error": None, "result": None})


def revert_model() -> None:
    """Restore the pre-calibration model.

    Frozen exe: the bundled base model is always present at MODEL_PATH (read-only,
    never overwritten).  Revert by deleting CALIBRATED_PATH from AppData so the
    serving engine falls back to the bundled model on the next reload.

    Dev mode: restore from the .joblib.backup written during calibration.
    """
    if _FROZEN:
        if not CALIBRATED_PATH.exists():
            raise FileNotFoundError("No calibrated model to revert (not yet calibrated).")
        CALIBRATED_PATH.unlink()
        log.info("reverted: deleted calibrated model at %s", CALIBRATED_PATH)
    else:
        backup = MODEL_PATH.with_suffix(".joblib.backup")
        if not backup.exists():
            raise FileNotFoundError(f"No backup found at {backup}")
        shutil.copy2(str(backup), str(MODEL_PATH))
        log.info("model reverted from backup: %s -> %s", backup, MODEL_PATH)


# ── Private: pipeline orchestration ──────────────────────────────────────────

def _upd(msg: str, progress: int | None = None, **kwargs: Any) -> None:
    with _LOCK:
        _STATE["message"] = msg
        if progress is not None:
            _STATE["progress"] = progress
        _STATE.update(kwargs)
    log.info("[calibration] %s", msg)


def _cancelled() -> bool:
    with _LOCK:
        return bool(_STATE.get("cancel_requested"))


def _run(interface: str, capture_duration: int, tshark_bin: str | None) -> None:
    try:
        INSTANCE_DIR.mkdir(parents=True, exist_ok=True)

        # In dev mode, add ml/ to sys.path so adns_flows is importable.
        # In the frozen exe adns_flows is already compiled into the bundle.
        if not _FROZEN:
            _root = Path(__file__).resolve().parent.parent.parent
            ml_path = str(_root / "ml")
            if ml_path not in sys.path:
                sys.path.insert(0, ml_path)

        pcap_path = _stage_capture(interface, capture_duration, tshark_bin)
        if _cancelled():
            _upd("Cancelled by user.", progress=0, stage="idle")
            return

        local_df = _stage_extract(pcap_path)
        if _cancelled():
            _upd("Cancelled by user.", progress=0, stage="idle")
            return

        benign_df = _stage_filter(local_df)

        calibrated_path, result = _stage_retrain(benign_df)
        _stage_validate(calibrated_path, benign_df, result)

        _upd("Calibration complete.", progress=100, stage="done")

    except Exception as exc:
        log.exception("[calibration] pipeline failed: %s", exc)
        _upd(f"Pipeline failed: {exc}", stage="failed", error=str(exc))


# ── Stage 1: Capture ──────────────────────────────────────────────────────────

def _stage_capture(interface: str, duration: int, tshark_bin: str | None) -> Path:
    _upd(f"Capturing traffic for {duration // 60} min on {interface}...", progress=5, stage="capturing")

    tshark = _find_tshark(tshark_bin)
    CAPTURE_PCAP.unlink(missing_ok=True)

    cmd = [tshark, "-i", interface, "-w", str(CAPTURE_PCAP), "-a", f"duration:{duration}"]
    env = {**os.environ, "PATH": str(Path(tshark).parent) + os.pathsep + os.environ.get("PATH", "")}

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        cwd=str(Path(tshark).parent),
        env=env,
        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
    )

    t_start = time.time()
    deadline = t_start + duration + 30
    while proc.poll() is None and time.time() < deadline:
        if _cancelled():
            proc.terminate()
            raise RuntimeError("Cancelled by user.")
        elapsed = time.time() - t_start
        # Progress 5 → 45 over the capture duration
        pct = 5 + int(min(elapsed / max(duration, 1), 1.0) * 40)
        _upd(
            f"Capturing... {int(elapsed // 60):02d}:{int(elapsed % 60):02d} elapsed "
            f"({duration // 60 - int(elapsed // 60):02d}:{duration % 60 - int(elapsed % 60) % 60:02d} remaining)",
            progress=pct,
        )
        time.sleep(2)

    proc.wait(timeout=15)

    if not CAPTURE_PCAP.exists() or CAPTURE_PCAP.stat().st_size < 100:
        raise RuntimeError(
            "PCAP file not created or empty after capture. "
            "Ensure the process is running as Administrator (Npcap requires elevated access)."
        )

    kb = CAPTURE_PCAP.stat().st_size // 1024
    _upd(f"Capture done: {kb} KB saved.", progress=45)
    return CAPTURE_PCAP


# ── Stage 2: Extract ──────────────────────────────────────────────────────────

def _stage_extract(pcap_path: Path) -> pd.DataFrame:
    _upd("Extracting flows with NFStream...", progress=48, stage="extracting")

    from adns_flows.extract_nfstream import extract_flows_nfstream, flows_to_dataframe_nfstream

    flows = extract_flows_nfstream(str(pcap_path), n_meters=1)
    if not flows:
        raise RuntimeError(
            "No flows extracted from calibration PCAP. "
            "The capture may have been too short or captured no routable traffic."
        )

    df = flows_to_dataframe_nfstream(flows)
    df.to_csv(str(CALIBRATION_CSV), index=False)
    _upd(f"Extracted {len(df)} flows from PCAP.", progress=55)
    return df


# ── Stage 3: Filter ───────────────────────────────────────────────────────────

def _stage_filter(df: pd.DataFrame) -> pd.DataFrame:
    _upd("Applying protocol whitelist and outlier filter...", progress=58, stage="filtering")

    from calibration.whitelist import apply_outlier_drop, apply_whitelist

    kept_wl, n_drop_wl = apply_whitelist(df)
    kept_ol, n_drop_ol, outlier_warn = apply_outlier_drop(kept_wl)

    total_dropped = n_drop_wl + n_drop_ol
    drop_pct = total_dropped / max(len(df), 1) * 100

    _upd(
        f"Filtered: {len(kept_ol)} benign flows kept "
        f"({n_drop_wl} by whitelist, {n_drop_ol} by outlier drop).",
        progress=63,
        flows_captured=len(kept_ol),
    )

    if len(kept_ol) < 10:
        raise RuntimeError(
            f"Too few benign flows after filtering ({len(kept_ol)}). "
            "Extend the capture duration or browse more varied benign traffic."
        )

    with _LOCK:
        _STATE["result"] = {
            "flows_total":     len(df),
            "flows_kept":      len(kept_ol),
            "flows_dropped":   total_dropped,
            "drop_pct":        round(drop_pct, 1),
            "outlier_warning": outlier_warn,
        }

    return kept_ol


# ── Stage 4: Retrain ──────────────────────────────────────────────────────────

def _stage_retrain(benign_df: pd.DataFrame) -> tuple[Path, dict]:
    _upd("Loading corpus attack samples for retraining...", progress=65, stage="retraining")

    from adns_flows.schema import FEATURE_COLUMNS

    # Load and stratified-sample attack rows from each corpus
    attack_frames: list[pd.DataFrame] = []
    for name, path in CORPUS_PATHS.items():
        if not path.exists():
            log.warning("corpus %s not found at %s", name, path)
            continue
        df_c = pd.read_parquet(str(path), columns=list(FEATURE_COLUMNS) + ["label"])
        attacks = df_c[df_c["label"] == 1]
        n = min(ATTACK_SAMPLE_PER_CORPUS, len(attacks))
        attack_frames.append(attacks.sample(n=n, random_state=42))
        log.info("  %s: sampled %d / %d attack rows", name, n, len(attacks))

    if not attack_frames:
        raise RuntimeError(
            "No corpus parquets found. Cannot retrain without attack examples. "
            "Expected at outputs/corpus/{{unsw,gotham,cic_tuesday}}_flows.parquet"
        )

    attacks_df = pd.concat(attack_frames, ignore_index=True)
    attacks_df = attacks_df[list(FEATURE_COLUMNS)].assign(label=1)

    benign_train = benign_df[list(FEATURE_COLUMNS)].assign(label=0)

    train_df = (
        pd.concat([benign_train, attacks_df], ignore_index=True)
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )

    n_ben = int((train_df["label"] == 0).sum())
    n_att = int((train_df["label"] == 1).sum())
    log.info("retrain pool: %d rows (%d benign, %d attack)", len(train_df), n_ben, n_att)
    _upd(f"Retraining on {len(train_df):,} rows ({n_ben} benign, {n_att} attack)...", progress=70)

    X = train_df[list(FEATURE_COLUMNS)].to_numpy(dtype=np.float32)
    y = train_df["label"].to_numpy(dtype=np.int32)
    scale_pos = n_ben / max(n_att, 1.0)

    _upd("Training XGBoost (this may take a few minutes)...", progress=72)
    import warnings

    from xgboost import XGBClassifier
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        xgb = XGBClassifier(
            tree_method="hist",
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            scale_pos_weight=scale_pos,
            random_state=42,
            verbosity=0,
            eval_metric="aucpr",
            nthread=-1,
        )
        xgb.fit(X, y)

    _upd("Training ExtraTrees...", progress=83)
    from sklearn.ensemble import ExtraTreesClassifier
    et = ExtraTreesClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=2,
        class_weight={0: 1.0, 1: scale_pos},
        random_state=42,
        n_jobs=-1,
    )
    et.fit(X, y)

    import joblib
    bundle = {"xgboost": xgb, "extra_trees": et}
    joblib.dump(bundle, str(CALIBRATED_PATH))
    _upd(f"Calibrated model saved to {CALIBRATED_PATH.name}.", progress=90)

    return CALIBRATED_PATH, {"n_benign": n_ben, "n_attack": n_att}


# ── Stage 5: Validate + swap ──────────────────────────────────────────────────

def _stage_validate(model_path: Path, benign_df: pd.DataFrame, retrain_result: dict) -> None:
    _upd("Running validation gates...", progress=91, stage="validating")

    import joblib

    from adns_flows.schema import FEATURE_COLUMNS

    # GATE 5A: Load without error
    try:
        bundle = joblib.load(str(model_path))
        xgb = bundle.get("xgboost") or next(iter(bundle.values()))
        assert xgb is not None
        log.info("GATE 5A PASS: model loads")
    except Exception as exc:
        raise RuntimeError(f"GATE 5A FAIL — model load error: {exc}") from exc

    _upd("Gate 5A: model loads OK.", progress=92)

    # Load a small attack sample for 5B + 5D
    attack_sample_frames: list[pd.DataFrame] = []
    for path in CORPUS_PATHS.values():
        if path.exists():
            df_c = pd.read_parquet(str(path), columns=list(FEATURE_COLUMNS) + ["label"])
            attack_sample_frames.append(df_c[df_c["label"] == 1].head(200))

    # GATE 5B: Attack discrimination
    if attack_sample_frames:
        att_samp = pd.concat(attack_sample_frames).sample(
            n=min(10, sum(len(f) for f in attack_sample_frames)), random_state=42
        )
        X_att = att_samp[list(FEATURE_COLUMNS)].to_numpy(dtype=np.float32)
        scores_att = xgb.predict_proba(X_att)[:, 1]
        bad = [round(float(s), 3) for s in scores_att if s < GATE_ATTACK_SCORE_MIN]
        if bad:
            raise RuntimeError(
                f"GATE 5B FAIL — {len(bad)} attack samples scored below {GATE_ATTACK_SCORE_MIN}: {bad}"
            )
        log.info("GATE 5B PASS: attack scores %s", [round(float(s), 3) for s in scores_att])
    else:
        log.warning("GATE 5B SKIP — no corpus available")

    _upd("Gate 5B: attack discrimination OK.", progress=94)

    # GATE 5C: FPR on calibration benign
    X_ben = benign_df[list(FEATURE_COLUMNS)].to_numpy(dtype=np.float32)
    scores_ben = xgb.predict_proba(X_ben)[:, 1]
    fpr = float((scores_ben > 0.5).sum()) / max(len(scores_ben), 1)
    if fpr >= GATE_FPR_MAX:
        raise RuntimeError(
            f"GATE 5C FAIL — local benign FPR={fpr:.1%} >= {GATE_FPR_MAX:.0%} threshold. "
            "The retrained model still flags too many local benign flows. Try a longer capture."
        )
    log.info("GATE 5C PASS: local benign FPR=%.2f%%", fpr * 100)
    _upd(f"Gate 5C: local FPR {fpr:.1%} — OK.", progress=95)

    # GATE 5D: Corpus recall maintained vs. original model
    old_recall: float | None = None
    new_recall: float | None = None
    if MODEL_PATH.exists() and attack_sample_frames:
        old_bundle = joblib.load(str(MODEL_PATH))
        old_xgb = old_bundle.get("xgboost") or next(iter(old_bundle.values()))
        all_att = pd.concat(attack_sample_frames)
        X_all = all_att[list(FEATURE_COLUMNS)].to_numpy(dtype=np.float32)
        old_recall = float((old_xgb.predict_proba(X_all)[:, 1] > 0.5).sum()) / max(len(X_all), 1)
        new_recall = float((xgb.predict_proba(X_all)[:, 1] > 0.5).sum()) / max(len(X_all), 1)
        if new_recall < GATE_RECALL_MIN:
            raise RuntimeError(
                f"GATE 5D FAIL — corpus attack recall dropped to {new_recall:.1%} "
                f"(was {old_recall:.1%}, minimum {GATE_RECALL_MIN:.0%}). "
                "The calibration corpus may be biased. Revert and recalibrate."
            )
        log.info("GATE 5D PASS: corpus recall %.2f%% (was %.2f%%)", new_recall * 100, old_recall * 100)
    else:
        log.warning("GATE 5D SKIP — original model or corpus not available")

    _upd("Gate 5D: corpus recall maintained.", progress=98)

    # All gates passed — activate the calibrated model.
    if _FROZEN:
        # Frozen exe: MODEL_PATH (bundled) is read-only.  CALIBRATED_PATH is already
        # written to AppData by _stage_retrain.  _do_reload() in calibration_routes
        # will pick it up (CALIBRATED_PATH exists → preferred over bundled model).
        log.info("frozen exe: calibrated model activated at %s", CALIBRATED_PATH)
    else:
        # Dev: back up base model and copy calibrated over it in-place.
        backup = MODEL_PATH.with_suffix(".joblib.backup")
        if MODEL_PATH.exists():
            shutil.copy2(str(MODEL_PATH), str(backup))
            log.info("backed up original model to %s", backup)
        shutil.copy2(str(model_path), str(MODEL_PATH))
        log.info("calibrated model activated: %s -> %s", model_path.name, MODEL_PATH.name)

    # Persist final result
    with _LOCK:
        existing = _STATE.get("result") or {}
        existing.update({
            "local_fpr_pct":     round(fpr * 100, 2),
            "corpus_recall_pct": round((new_recall or 0.0) * 100, 2) if new_recall is not None else None,
            "old_recall_pct":    round((old_recall or 0.0) * 100, 2) if old_recall is not None else None,
            "gates_passed":      ["5A", "5B", "5C", "5D"],
        })
        _STATE["result"] = existing

    _upd("All 4 validation gates passed. Model activated.", progress=99)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _find_tshark(override: str | None) -> str:
    candidates = [
        override or "",
        os.environ.get("TSHARK_BIN", ""),
        r"C:\Program Files\Wireshark\tshark.exe",
        shutil.which("tshark") or "",
    ]
    for c in candidates:
        if c and Path(c).is_file():
            return c
    raise RuntimeError(
        "tshark not found. Install Wireshark or set TSHARK_BIN environment variable."
    )
