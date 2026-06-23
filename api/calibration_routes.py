"""Flask blueprint for deployment-time calibration endpoints.

Endpoints:
  GET  /calibration/status          — poll pipeline state
  POST /calibration/start           — begin guided capture + retrain
  POST /calibration/cancel          — request pipeline cancellation
  POST /calibration/reset           — reset to idle (after done/failed)
  POST /calibration/revert          — restore pre-calibration model backup
  POST /calibration/reload_model    — hot-reload the active model in all engines
  GET  /calibration/first_run_check — returns whether a calibrated model exists
"""
from __future__ import annotations

import logging

from flask import Blueprint, jsonify, request

log = logging.getLogger(__name__)

calibration_bp = Blueprint("calibration", __name__, url_prefix="/calibration")


# ── Status ─────────────────────────────────────────────────────────────────────

@calibration_bp.get("/status")
def cal_status():
    from calibration.pipeline import get_state
    state = get_state()
    # Add time remaining during capture
    if state["stage"] == "capturing" and state.get("timer_start") and state.get("capture_duration"):
        import time
        elapsed = time.time() - state["timer_start"]
        remaining = max(0, state["capture_duration"] - elapsed)
        state["capture_seconds_remaining"] = int(remaining)
    return jsonify(state)


# ── Start ──────────────────────────────────────────────────────────────────────

@calibration_bp.post("/start")
def cal_start():
    from calibration.pipeline import start_pipeline

    body = request.get_json(silent=True) or {}
    interface = body.get("interface", "").strip()
    duration  = int(body.get("capture_duration", 1800))
    tshark    = body.get("tshark_bin", None)

    if not interface:
        return jsonify({"error": "interface is required"}), 400
    if not (60 <= duration <= 7200):
        return jsonify({"error": "capture_duration must be 60–7200 seconds"}), 400

    try:
        start_pipeline(interface, capture_duration=duration, tshark_bin=tshark)
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 409

    return jsonify({"status": "started", "interface": interface, "duration": duration})


# ── Cancel ─────────────────────────────────────────────────────────────────────

@calibration_bp.post("/cancel")
def cal_cancel():
    from calibration.pipeline import cancel_pipeline
    cancel_pipeline()
    return jsonify({"status": "cancel_requested"})


# ── Reset ──────────────────────────────────────────────────────────────────────

@calibration_bp.post("/reset")
def cal_reset():
    from calibration.pipeline import reset_pipeline
    reset_pipeline()
    return jsonify({"status": "reset"})


# ── Revert ─────────────────────────────────────────────────────────────────────

@calibration_bp.post("/revert")
def cal_revert():
    from calibration.pipeline import revert_model
    try:
        revert_model()
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 404
    _do_reload()
    return jsonify({"status": "reverted"})


# ── Reload ─────────────────────────────────────────────────────────────────────

@calibration_bp.post("/reload_model")
def cal_reload():
    """Hot-reload the active model in all detection engine instances."""
    _do_reload()
    return jsonify({"status": "reloaded"})


def _do_reload() -> None:
    """Reload every NfstreamDetectionEngine with the currently active model.

    Prefers the calibrated model if it exists (AppData in frozen exe, or
    api/model_artifacts/ in dev), otherwise uses the base bundled model.
    """
    from calibration.pipeline import CALIBRATED_PATH, MODEL_PATH
    active = str(CALIBRATED_PATH) if CALIBRATED_PATH.exists() else str(MODEL_PATH)
    log.info("reloading detection engines with %s", active)

    try:
        import tasks as tasks_mod
        if hasattr(tasks_mod, "nfstream_detector"):
            tasks_mod.nfstream_detector.reload(active)
            log.info("tasks.nfstream_detector reloaded")
    except Exception as exc:
        log.warning("tasks engine reload failed: %s", exc)

    try:
        import app as app_mod
        if hasattr(app_mod, "_simulation_scorer"):
            app_mod._simulation_scorer.reload(active)
            log.info("app._simulation_scorer reloaded")
    except Exception as exc:
        log.warning("app scorer reload failed: %s", exc)


# ── First-run check ───────────────────────────────────────────────────────────

@calibration_bp.get("/first_run_check")
def cal_first_run_check():
    """Returns whether a calibrated model already exists.

    Used by the frontend to decide whether to show the first-run calibration prompt.
    Paths are freeze-aware (imported from calibration.pipeline).
    """
    from calibration.pipeline import CALIBRATED_PATH, MODEL_PATH
    calibrated_exists = CALIBRATED_PATH.exists()
    base_exists       = MODEL_PATH.exists()
    return jsonify({
        "calibrated_model_exists": calibrated_exists,
        "base_model_exists":       base_exists,
        "suggest_calibration":     base_exists and not calibrated_exists,
    })
