"""Stateful port-scan and SYN-flood detector for ADNS live capture.

Replaces the per-flow _infer_scanning heuristic with a sliding-window
per-IP aggregator that requires sustained behavior before flagging:

  Port scan:  src_ip hits >= SCAN_PORT_THRESHOLD distinct dst_ports
              within SCAN_WINDOW_S seconds -> "scanning"

  SYN flood:  src_ip sends >= FLOOD_PKT_THRESHOLD degenerate TCP flows
              (proto=6, src_pkts <= 2, dst_pkts == 0) within FLOOD_WINDOW_S
              seconds -> "syn_flood"

Both detections are strictly per src_ip and do not cross-count.
"""
from __future__ import annotations

import threading
import time
from collections import defaultdict, deque


class ScanFloodDetector:
    SCAN_WINDOW_S: float = 60.0
    SCAN_PORT_THRESHOLD: int = 15

    FLOOD_WINDOW_S: float = 30.0
    FLOOD_PKT_THRESHOLD: int = 50

    MAX_TRACKED_IPS: int = 8_000

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # per src_ip: {dst_port -> last_seen_epoch_s}
        # Using a dict so repeated hits to the same port don't inflate the count.
        self._port_map: dict[str, dict[int, float]] = defaultdict(dict)
        # per src_ip: deque of epoch seconds for degenerate TCP flows
        self._flood_times: dict[str, deque] = defaultdict(deque)

    def record_and_classify(self, flow, _now: float | None = None) -> str | None:
        """Record this flow in the sliding windows and return a threat class or None.

        Returns one of "scanning", "syn_flood", or None.

        _now: epoch seconds override — used in tests to inject fixed timestamps.
        In production, the flow's own .timestamp is used (falls back to time.time()).
        """
        extra = flow.extra or {}
        src_ip = str(flow.src_ip or "")
        if not src_ip:
            return None

        try:
            dst_port = int(extra.get("dst_port") or 0)
        except (TypeError, ValueError):
            dst_port = 0
        try:
            proto = int(extra.get("proto") or 0)
        except (TypeError, ValueError):
            proto = 0
        try:
            src_pkts = int(extra.get("src_pkts") or 0)
        except (TypeError, ValueError):
            src_pkts = 0
        try:
            dst_pkts = int(extra.get("dst_pkts") or 0)
        except (TypeError, ValueError):
            dst_pkts = 0

        if _now is None:
            ts = getattr(flow, "timestamp", None)
            if ts is not None:
                try:
                    _now = ts.timestamp()
                except (AttributeError, OSError):
                    _now = time.time()
            else:
                _now = time.time()
        now: float = _now

        with self._lock:
            result = self._classify(src_ip, now, dst_port, proto, src_pkts, dst_pkts)
            if len(self._port_map) + len(self._flood_times) > 2 * self.MAX_TRACKED_IPS:
                self._evict_stale(now)
        return result

    def _classify(
        self,
        src_ip: str,
        now: float,
        dst_port: int,
        proto: int,
        src_pkts: int,
        dst_pkts: int,
    ) -> str | None:
        # ── Port scan ─────────────────────────────────────────────────────────
        if dst_port:
            pm = self._port_map[src_ip]
            pm[dst_port] = now
            cutoff = now - self.SCAN_WINDOW_S
            stale = [p for p, t in pm.items() if t < cutoff]
            for p in stale:
                del pm[p]
            if len(pm) >= self.SCAN_PORT_THRESHOLD:
                return "scanning"

        # ── SYN flood ─────────────────────────────────────────────────────────
        if proto == 6 and src_pkts <= 2 and dst_pkts == 0:
            dq = self._flood_times[src_ip]
            dq.append(now)
            cutoff = now - self.FLOOD_WINDOW_S
            while dq and dq[0] < cutoff:
                dq.popleft()
            if len(dq) >= self.FLOOD_PKT_THRESHOLD:
                return "syn_flood"

        return None

    def _evict_stale(self, now: float) -> None:
        """Remove IPs with no active entries — called when MAX_TRACKED_IPS is exceeded."""
        scan_cutoff = now - self.SCAN_WINDOW_S
        flood_cutoff = now - self.FLOOD_WINDOW_S
        dead = [ip for ip, pm in self._port_map.items()
                if not pm or max(pm.values()) < scan_cutoff]
        for ip in dead:
            del self._port_map[ip]
        dead = [ip for ip, dq in self._flood_times.items()
                if not dq or dq[-1] < flood_cutoff]
        for ip in dead:
            del self._flood_times[ip]

    def reset(self) -> None:
        """Clear all tracked state. Used after calibration resets and in tests."""
        with self._lock:
            self._port_map.clear()
            self._flood_times.clear()


# Process-level singleton — imported by tasks.py and app.py.
detector = ScanFloodDetector()
