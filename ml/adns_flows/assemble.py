"""
Join Pass A (conv stats) and Pass B (TCP flags) into bidirectional Flow objects.

Assembly contract:
  1. Both passes are joined on orientation_key (unordered pair) so the join is
     direction-agnostic — it succeeds regardless of which side tshark listed as A.
  2. canonicalize_orientation() is called ONCE per flow to fix src vs dst.
     The default rule (src = lower (ip,port)) is applied identically in live
     capture and offline pcap mode; corpus building may pass prefer_src to pin
     a known attacker IP as src without affecting other flows.
  3. Directional byte/packet counts are assigned to canonical src/dst AFTER
     canonicalization, so derived ratios and mean sizes are consistent.
  4. Rows are sorted by (ts, src_ip, src_port, dst_ip, dst_port) for
     determinism: same pcap in → identical CSV out regardless of parse order.
"""
from __future__ import annotations

import pandas as pd

from .extract import run_pass_a, run_pass_b
from .schema import (
    FEATURE_COLUMNS,
    IDENTITY_COLUMNS,
    Flow,
    canonicalize_orientation,
    flow_to_row,
    orientation_key,
)

_PROTO_CODE: dict[str, int] = {"TCP": 6, "UDP": 17, "ICMP": 1}


def _make_flow(
    conv: dict,
    flags: dict[tuple, dict[str, int]],
    prefer_src: str | None = None,
) -> Flow:
    proto_code = _PROTO_CODE.get(conv["proto"].upper(), 0)
    ep_a: tuple[str, int] = conv["ep_a"]
    ep_b: tuple[str, int] = conv["ep_b"]

    # Canonical src/dst — same rule for live and offline paths.
    src_ep, dst_ep = canonicalize_orientation(ep_a, ep_b, prefer_src=prefer_src)

    # Assign directional counts based on which physical endpoint is canonical src.
    if src_ep == ep_a:
        src_bytes, dst_bytes = conv["bytes_ab"], conv["bytes_ba"]
        src_pkts,  dst_pkts  = conv["pkts_ab"],  conv["pkts_ba"]
    else:
        src_bytes, dst_bytes = conv["bytes_ba"], conv["bytes_ab"]
        src_pkts,  dst_pkts  = conv["pkts_ba"],  conv["pkts_ab"]

    # Flag join on the unordered key — matches regardless of packet direction.
    key = orientation_key(ep_a[0], ep_a[1], ep_b[0], ep_b[1])
    fc = flags.get(key, {})

    return Flow(
        ts=conv.get("rel_start", 0.0),
        src_ip=src_ep[0],
        dst_ip=dst_ep[0],
        src_port=src_ep[1],
        dst_port=dst_ep[1],
        proto=proto_code,
        duration=conv["duration"],
        src_bytes=src_bytes,
        dst_bytes=dst_bytes,
        src_pkts=src_pkts,
        dst_pkts=dst_pkts,
        syn_count=fc.get("syn", 0),
        ack_count=fc.get("ack", 0),
        rst_count=fc.get("rst", 0),
        fin_count=fc.get("fin", 0),
        psh_count=fc.get("psh", 0),
        urg_count=fc.get("urg", 0),
    )


def build_flows(
    conv_flows: list[dict],
    flag_counts: dict[tuple, dict[str, int]],
    prefer_src: str | None = None,
) -> list[Flow]:
    """Join conv dicts with flag counts and return unsorted Flow objects.

    prefer_src: optional IP string to pin as src for all flows in this batch.
    Used by corpus builders to orient flows relative to a known attacker IP.
    Omit (or pass None) for live capture — the default rule applies.
    """
    return [_make_flow(c, flag_counts, prefer_src=prefer_src) for c in conv_flows]


def flows_to_dataframe(flows: list[Flow]) -> pd.DataFrame:
    """Convert Flow objects to a DataFrame in canonical column order, sorted for determinism."""
    all_cols = list(IDENTITY_COLUMNS) + list(FEATURE_COLUMNS)
    if not flows:
        return pd.DataFrame(columns=all_cols)
    df = pd.DataFrame([flow_to_row(f) for f in flows])
    df = df.sort_values(
        ["ts", "src_ip", "src_port", "dst_ip", "dst_port"],
        ascending=True,
    ).reset_index(drop=True)
    return df[all_cols]


def extract_flows(
    tshark_bin: str,
    *,
    pcap: str | None = None,
    iface: str | None = None,
    window_sec: int = 60,
    prefer_src: str | None = None,
) -> list[Flow]:
    """
    Full two-pass extraction over a pcap file or live capture window.

    Both passes receive the same source so the flag join is always coherent.
    Orientation is canonicalized once in build_flows(); prefer_src is forwarded
    unchanged (None for live capture, attacker IP for corpus building).
    Returns Flow objects sorted by (ts, canonical src_ip, src_port, dst_ip,
    dst_port) for deterministic output.
    """
    conv = run_pass_a(tshark_bin, pcap=pcap, iface=iface, window_sec=window_sec)
    flags = run_pass_b(tshark_bin, pcap=pcap, iface=iface, window_sec=window_sec)
    flows = build_flows(conv, flags, prefer_src=prefer_src)
    flows.sort(key=lambda f: (f.ts, f.src_ip, f.src_port, f.dst_ip, f.dst_port))
    return flows
