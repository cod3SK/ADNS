"""NFStream-based flow extractor behind the ADNS feature contract.

Drop-in replacement for the tshark two-pass path (extract.py + assemble.py).
Emits the same canonical Flow objects and passes validate_matrix() on the
resulting DataFrame, so corpus builders and the live serving path can switch
between extractors without touching any downstream code.

Orientation
-----------
NFStream assigns src/dst based on the first-seen packet (initiator = src).
We override this with canonicalize_orientation() — the same function used by
the tshark path — so directional features (src_bytes, dst_bytes, etc.) are
computed relative to the lower (ip, port) endpoint, not the flow initiator.
When orientation flips, src2dst_bytes/packets and dst2src_bytes/packets are
swapped before being assigned to the canonical src/dst fields.

Byte accounting
---------------
NFStream accounting_mode=0 (default) counts Ethernet frame bytes (L2).
tshark -z conv,tcp also counts at L2, so both extractors report identical byte
counts for the same pcap.  See nfstream_config.py for the full note.

TCP flag counts
---------------
bidirectional_*_packets fields count flag bits across both directions.
This matches how parse_flag_lines() aggregates tshark Pass B output.
All six flag fields require statistical_analysis=True (see nfstream_config.py).
"""
from __future__ import annotations

import pandas as pd

from .nfstream_config import make_nfstream_kwargs
from .schema import (
    FEATURE_COLUMNS,
    IDENTITY_COLUMNS,
    Flow,
    canonicalize_orientation,
    flow_to_row,
    validate_matrix,
)

# IP protocol numbers that map to named contract values; everything else → 0.
_KNOWN_PROTOS: frozenset[int] = frozenset({6, 17, 1})  # TCP, UDP, ICMP


def _proto_code(ip_protocol: int) -> int:
    return ip_protocol if ip_protocol in _KNOWN_PROTOS else 0


def _nf_to_flow(nf, *, prefer_src: str | None = None) -> Flow:
    """Convert one NFStream flow object to a canonical Flow.

    NFStream's src/dst (initiator-based) is replaced by canonicalize_orientation()
    to match the tshark path's orientation rule: src = lower (ip, port).
    """
    ep_initiator: tuple[str, int] = (nf.src_ip, nf.src_port)
    ep_receiver:  tuple[str, int] = (nf.dst_ip, nf.dst_port)

    src_ep, dst_ep = canonicalize_orientation(
        ep_initiator, ep_receiver, prefer_src=prefer_src
    )
    flipped = (src_ep != ep_initiator)

    if not flipped:
        src_bytes = nf.src2dst_bytes
        dst_bytes = nf.dst2src_bytes
        src_pkts  = nf.src2dst_packets
        dst_pkts  = nf.dst2src_packets
    else:
        src_bytes = nf.dst2src_bytes
        dst_bytes = nf.src2dst_bytes
        src_pkts  = nf.dst2src_packets
        dst_pkts  = nf.src2dst_packets

    return Flow(
        ts=nf.bidirectional_first_seen_ms / 1000.0,
        src_ip=src_ep[0],
        dst_ip=dst_ep[0],
        src_port=src_ep[1],
        dst_port=dst_ep[1],
        proto=_proto_code(nf.protocol),
        duration=nf.bidirectional_duration_ms / 1000.0,
        src_bytes=src_bytes,
        dst_bytes=dst_bytes,
        src_pkts=src_pkts,
        dst_pkts=dst_pkts,
        syn_count=nf.bidirectional_syn_packets,
        ack_count=nf.bidirectional_ack_packets,
        rst_count=nf.bidirectional_rst_packets,
        fin_count=nf.bidirectional_fin_packets,
        psh_count=nf.bidirectional_psh_packets,
        urg_count=nf.bidirectional_urg_packets,
    )


def extract_flows_nfstream(
    source: str,
    *,
    prefer_src: str | None = None,
    n_meters: int = 1,
) -> list[Flow]:
    """Extract flows from a pcap file or live interface using NFStream.

    source:     pcap file path, OR NPF device name for live capture (Windows).
    prefer_src: optional attacker IP to pin as canonical src (corpus builders only).
                Omit for live inference — the default canonical-orientation rule applies.
    n_meters:   parallelism; 1 for frozen-exe serving safety.
    """
    from nfstream import NFStreamer  # deferred import — child workers must not run this at top level

    kwargs = make_nfstream_kwargs(n_meters=n_meters)
    flows: list[Flow] = []
    for nf in NFStreamer(source=source, **kwargs):
        flows.append(_nf_to_flow(nf, prefer_src=prefer_src))

    flows.sort(key=lambda f: (f.ts, f.src_ip, f.src_port, f.dst_ip, f.dst_port))
    return flows


def flows_to_dataframe_nfstream(flows: list[Flow]) -> pd.DataFrame:
    """Convert a Flow list to a validated DataFrame in canonical column order.

    Calls validate_matrix() so any schema drift raises SchemaError immediately.
    """
    all_cols = list(IDENTITY_COLUMNS) + list(FEATURE_COLUMNS)
    if not flows:
        return pd.DataFrame(columns=all_cols)
    df = pd.DataFrame([flow_to_row(f) for f in flows])
    df = df.sort_values(
        ["ts", "src_ip", "src_port", "dst_ip", "dst_port"],
        ascending=True,
    ).reset_index(drop=True)
    df = df[all_cols]
    validate_matrix(df, list(df.columns[len(IDENTITY_COLUMNS):]))
    return df
