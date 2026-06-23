"""Protocol whitelist filter + statistical outlier drop for calibration captures.

Keeps only flows whose src_port or dst_port matches a known-safe protocol set,
then drops statistical outliers on total_bytes (> median + 3 std).

These two filters together are the "poison safeguard" described in the spec:
  - Whitelist: attacker can't inject arbitrary traffic into the benign corpus
    by sending traffic to unexpected ports during the calibration window.
  - Outlier drop: catches the few flows that escaped the whitelist but are
    anomalously large (large file downloads, torrents, etc.).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Every port that is expected in benign home/office traffic.
# Flows where EITHER src_port OR dst_port is in this set are kept.
SAFE_PORTS: frozenset[int] = frozenset({
    # DNS / mDNS / LLMNR
    53, 5353, 5355,
    # DHCP
    67, 68,
    # NTP
    123,
    # HTTP / HTTPS
    80, 443, 8080, 8443,
    # QUIC (UDP/443 is included above; 80/443 cover most QUIC)
    # Email
    25, 587, 465, 143, 993, 110, 995,
    # SSH
    22,
    # LDAP / LDAPS
    389, 636,
    # SMB / NetBIOS-NS / NetBIOS-DGM
    137, 138, 139, 445,
    # SNMP
    161, 162,
    # Syslog
    514,
    # NFS / RPC
    111, 2049,
    # UPnP / SSDP
    1900,
    # RDP
    3389,
    # Kerberos
    88,
    # CUPS
    631,
    # ICMP (port 0 appears in some NFStream rows for ICMP flows)
    0,
})

_WARN_DROP_RATE = 0.10   # emit warning if > 10% of flows are dropped by either filter


def apply_whitelist(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Return (kept_df, n_dropped) keeping flows whose src/dst_port is safe."""
    if df.empty:
        return df.copy(), 0

    dst_col = "dst_port" if "dst_port" in df.columns else None
    src_col = "src_port" if "src_port" in df.columns else None

    if dst_col is None and src_col is None:
        log.warning("whitelist: no port columns found — keeping all %d flows", len(df))
        return df.copy(), 0

    mask_dst = df[dst_col].isin(SAFE_PORTS) if dst_col else pd.Series(False, index=df.index)
    mask_src = df[src_col].isin(SAFE_PORTS) if src_col else pd.Series(False, index=df.index)
    keep = mask_dst | mask_src

    kept = df[keep].copy()
    n_dropped = int((~keep).sum())
    log.info("whitelist: kept %d / %d  (dropped %d, %.1f%%)",
             len(kept), len(df), n_dropped, 100 * n_dropped / max(len(df), 1))
    return kept, n_dropped


def apply_outlier_drop(df: pd.DataFrame) -> tuple[pd.DataFrame, int, bool]:
    """Drop flows with total_bytes far above the local 75th percentile.

    Uses an IQR-based threshold so a single extreme outlier does not inflate
    the scale and hide itself (a known failure mode of mean+3σ).

    Threshold: Q3 + max(3 * IQR, 0.5 * Q3)
      - When IQR is small (homogeneous traffic), 0.5*Q3 provides a floor so that
        values merely 50% above Q3 are still accepted.
      - When IQR is large (diverse traffic), 3*IQR is more permissive.

    Returns (kept_df, n_dropped, high_drop_rate_warning).
    The warning fires when > 10% are dropped — suggests unusual activity.
    """
    if df.empty:
        return df.copy(), 0, False

    col = "total_bytes" if "total_bytes" in df.columns else None
    if col is None:
        log.warning("outlier_drop: no total_bytes column — keeping all flows")
        return df.copy(), 0, False

    vals = df[col].astype(float).values
    q1, q3 = float(np.percentile(vals, 25)), float(np.percentile(vals, 75))
    iqr = q3 - q1
    upper = q3 + max(3.0 * iqr, max(q3 * 0.5, 1.0))

    keep = df[col] <= upper
    kept = df[keep].copy()
    n_dropped = int((~keep).sum())
    drop_rate = n_dropped / max(len(df), 1)
    warn = drop_rate > _WARN_DROP_RATE

    if warn:
        log.warning(
            "outlier_drop: drop rate %.1f%% > 10%% — possible large transfers or poisoning "
            "during calibration window (Q3=%.0f, IQR=%.0f, upper=%.0f)",
            drop_rate * 100, q3, iqr, upper,
        )
    else:
        log.info("outlier_drop: kept %d / %d  (dropped %d, %.1f%%)",
                 len(kept), len(df), n_dropped, drop_rate * 100)

    return kept, n_dropped, warn
