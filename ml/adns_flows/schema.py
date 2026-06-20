"""
Canonical feature contract for ADNS v1 flow-statistics detector.

IDENTITY_COLUMNS  — kept for joins/labeling; NEVER fed to the model.
FEATURE_COLUMNS   — authoritative ordered model input list.

ORIENTATION IS PART OF THE FEATURE CONTRACT.
Flow orientation (which endpoint is "src", which is "dst") is as load-bearing
as the column list. All directional features — src_bytes, dst_bytes, src_pkts,
dst_pkts, bytes_ratio, pkts_ratio, src_mean_pkt_size, dst_mean_pkt_size — are
computed AFTER canonicalization. Both tshark passes and the label join must use
the same orientation rule, applied once in assemble.py via
canonicalize_orientation(). The rule is documented below.

All other features are directly observed from tshark conv/flags output or
arithmetic of those observed values. No hashing into invented ranges, no
Zeek-only fields.

Excluded (with rationale):
    conn_state        — Zeek state-machine field; meaningless at tshark serve time
    missed_bytes      — never reliably populated from tshark
    ssl_*, weird_*    — app-layer; deferred to v2 app-layer model
    http_*            — same
    dns_*             — same
    *_ip_bytes        — aliases for *_bytes (TON_IoT artifact); redundant
"""
from __future__ import annotations

import dataclasses
from typing import Sequence

IDENTITY_COLUMNS: tuple[str, ...] = (
    "ts",
    "src_ip",
    "dst_ip",
    "src_port",
    "dst_port",
)

FEATURE_COLUMNS: tuple[str, ...] = (
    "proto",              # numeric: tcp=6, udp=17, icmp=1, other=0
    "duration",           # real flow duration in seconds (from conv output)
    "src_bytes",          # canonical-src→dst bytes (after orientation)
    "dst_bytes",          # canonical-dst→src bytes (after orientation)
    "total_bytes",        # src_bytes + dst_bytes
    "src_pkts",           # canonical-src→dst frames (after orientation)
    "dst_pkts",           # canonical-dst→src frames (after orientation)
    "total_pkts",         # src_pkts + dst_pkts
    "bytes_ratio",        # dst_bytes / (src_bytes + 1)
    "pkts_ratio",         # dst_pkts / (src_pkts + 1)
    "src_mean_pkt_size",  # src_bytes / (src_pkts + 1)
    "dst_mean_pkt_size",  # dst_bytes / (dst_pkts + 1)
    "bytes_per_sec",      # total_bytes / max(duration, 1e-3)
    "pkts_per_sec",       # total_pkts / max(duration, 1e-3)
    "dst_port_bucket",    # 0=well-known(<1024) 1=registered(<49152) 2=ephemeral
    "syn_count",          # TCP SYN bits seen across the flow (both directions)
    "ack_count",
    "rst_count",
    "fin_count",
    "psh_count",
    "urg_count",
)


class SchemaError(ValueError):
    """Raised when a matrix does not conform to the expected FEATURE_COLUMNS schema."""


@dataclasses.dataclass
class Flow:
    # Identity (not model inputs)
    ts: float
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    # Directly observed (canonical-src direction)
    proto: int        # tcp=6, udp=17, icmp=1, other=0
    duration: float
    src_bytes: int
    dst_bytes: int
    src_pkts: int
    dst_pkts: int
    # TCP flag counts (aggregated across both directions; 0 for non-TCP)
    syn_count: int = 0
    ack_count: int = 0
    rst_count: int = 0
    fin_count: int = 0
    psh_count: int = 0
    urg_count: int = 0


def flow_to_row(flow: Flow) -> dict:
    """Return an ordered dict with IDENTITY_COLUMNS then FEATURE_COLUMNS."""
    total_bytes = flow.src_bytes + flow.dst_bytes
    total_pkts = flow.src_pkts + flow.dst_pkts
    dur = max(flow.duration, 1e-3)
    return {
        "ts": flow.ts,
        "src_ip": flow.src_ip,
        "dst_ip": flow.dst_ip,
        "src_port": flow.src_port,
        "dst_port": flow.dst_port,
        "proto": flow.proto,
        "duration": flow.duration,
        "src_bytes": flow.src_bytes,
        "dst_bytes": flow.dst_bytes,
        "total_bytes": total_bytes,
        "src_pkts": flow.src_pkts,
        "dst_pkts": flow.dst_pkts,
        "total_pkts": total_pkts,
        "bytes_ratio": flow.dst_bytes / (flow.src_bytes + 1),
        "pkts_ratio": flow.dst_pkts / (flow.src_pkts + 1),
        "src_mean_pkt_size": flow.src_bytes / (flow.src_pkts + 1),
        "dst_mean_pkt_size": flow.dst_bytes / (flow.dst_pkts + 1),
        "bytes_per_sec": total_bytes / dur,
        "pkts_per_sec": total_pkts / dur,
        "dst_port_bucket": _port_bucket(flow.dst_port),
        "syn_count": flow.syn_count,
        "ack_count": flow.ack_count,
        "rst_count": flow.rst_count,
        "fin_count": flow.fin_count,
        "psh_count": flow.psh_count,
        "urg_count": flow.urg_count,
    }


def _port_bucket(port: int) -> int:
    if port < 1024:
        return 0
    if port < 49152:
        return 1
    return 2


# ── Orientation contract ───────────────────────────────────────────────────

def orientation_key(
    ip_a: str, port_a: int, ip_b: str, port_b: int
) -> tuple[tuple[str, int], tuple[str, int]]:
    """Return an unordered canonical key for a flow 4-tuple.

    The key is the same regardless of which endpoint tshark listed as A or B,
    and regardless of packet direction. Used to join Pass A (conv stats) with
    Pass B (tcp.flags) without relying on directional naming.

    Both components are sorted so min(ep_a, ep_b) is always first.
    """
    a = (ip_a, port_a)
    b = (ip_b, port_b)
    return (min(a, b), max(a, b))


def canonicalize_orientation(
    endpoint_a: tuple[str, int],
    endpoint_b: tuple[str, int],
    prefer_src: str | None = None,
) -> tuple[tuple[str, int], tuple[str, int]]:
    """Return (src_endpoint, dst_endpoint) according to the canonical rule.

    DEFAULT RULE: src = the endpoint with the lower (ip, port) tuple under
    Python's lexicographic ordering of the IP string, then numeric port.
    This ordering is TOTAL (every pair has a definite answer), STABLE (the
    same input always produces the same output), and DETERMINISTIC (no
    randomness or per-run state). The only property that matters is that every
    path — live capture, offline pcap, corpus labeling — applies the SAME rule.

    Tie-break when endpoint_a == endpoint_b (both IP and port identical, which
    is degenerate and should not occur in real traffic): src = endpoint_a.

    prefer_src override: if prefer_src is an IP string matching one endpoint,
    that endpoint is pinned as src. Used by corpus builders to orient flows
    relative to the dataset's documented attacker IP. Live capture never passes
    prefer_src, ensuring both paths default to the same deterministic rule.
    If prefer_src does not match either endpoint, the default rule applies.
    """
    if prefer_src is not None:
        if endpoint_a[0] == prefer_src:
            return (endpoint_a, endpoint_b)
        if endpoint_b[0] == prefer_src:
            return (endpoint_b, endpoint_a)
        # No match — fall through to default rule.

    if endpoint_a <= endpoint_b:
        return (endpoint_a, endpoint_b)
    return (endpoint_b, endpoint_a)


def validate_matrix(data: object, columns: Sequence[str]) -> None:
    """
    Raise SchemaError if `columns` does not exactly match FEATURE_COLUMNS in
    name and order.

    This is the explicit-failure replacement for MetaFeatureBuilder._match_shape
    (api/model_runner.py), which silently pads or truncates. The new pipeline
    fails loud on any schema mismatch so drift is caught at integration time,
    not at prediction time.
    """
    expected = list(FEATURE_COLUMNS)
    got = list(columns)
    if got == expected:
        return
    if len(got) != len(expected):
        raise SchemaError(
            f"column count mismatch: expected {len(expected)}, got {len(got)}"
        )
    mismatches = [
        f"  [{i}] expected {e!r}, got {g!r}"
        for i, (e, g) in enumerate(zip(expected, got))
        if e != g
    ]
    raise SchemaError("column name/order mismatch:\n" + "\n".join(mismatches))
