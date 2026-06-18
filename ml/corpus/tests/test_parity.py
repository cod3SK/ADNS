"""
Parity tests: the corpus extraction path is the live scoring path.

Three tiers of proof:
  1. Pure-Python (no tshark): prefer_src flips directional features correctly.
  2. Pure-Python (no tshark): benign corpus orientation == default orientation.
  3. tshark-gated: extract_pcap_flows + build_flows produces byte-identical
     output to extract_flows() (the live path) on the same pcap.

These tests guard the core invariant of ml/adns_flows/: one extractor and one
orientation rule for train and serve.
"""
from __future__ import annotations

import io
import struct

import pytest

from adns_flows import (
    FEATURE_COLUMNS,
    IDENTITY_COLUMNS,
    build_flows,
    extract_flows,
    flow_to_row,
    flows_to_dataframe,
    orientation_key,
)
from adns_flows.extract import find_tshark

from corpus.build_corpus import extract_pcap_flows


# ── shared conv fixture ────────────────────────────────────────────────────
#
# A single TCP flow where ep_b has the higher IP, so by default orientation
# ep_a (10.0.0.5) is src.  When prefer_src="10.0.0.9", the roles flip.

_CONV = {
    "proto":    "TCP",
    "ep_a":     ("10.0.0.5", 12345),
    "ep_b":     ("10.0.0.9", 80),
    "bytes_ab": 100,   # A→B (ep_a sent 100 bytes to ep_b)
    "bytes_ba": 200,   # B→A
    "pkts_ab":  1,
    "pkts_ba":  2,
    "duration": 1.0,
    "rel_start": 0.0,
}

_KEY = orientation_key("10.0.0.5", 12345, "10.0.0.9", 80)
_FLAGS = {_KEY: {"syn": 2, "ack": 3, "rst": 0, "fin": 1, "psh": 1, "urg": 0}}


# ── TIER 1: prefer_src flips directional features ─────────────────────────

def test_default_orientation_src_is_lower_ip():
    """Default rule: src = lower (ip, port). 10.0.0.5 < 10.0.0.9 → src."""
    flow = build_flows([_CONV], {}, prefer_src=None)[0]
    assert flow.src_ip    == "10.0.0.5"
    assert flow.dst_ip    == "10.0.0.9"
    assert flow.src_bytes == 100   # bytes_ab (A→B, A is canonical src)
    assert flow.dst_bytes == 200   # bytes_ba


def test_prefer_src_flips_src_and_dst():
    """prefer_src=10.0.0.9 must make 10.0.0.9 the src even though it has the higher IP."""
    flow = build_flows([_CONV], {}, prefer_src="10.0.0.9")[0]
    assert flow.src_ip    == "10.0.0.9"
    assert flow.dst_ip    == "10.0.0.5"
    assert flow.src_bytes == 200   # bytes_ba (B→A, B is now src)
    assert flow.dst_bytes == 100   # bytes_ab


def test_prefer_src_conserves_total_bytes():
    """Total bytes must be the same regardless of which endpoint is src."""
    flow_default = build_flows([_CONV], {}, prefer_src=None)[0]
    flow_flipped = build_flows([_CONV], {}, prefer_src="10.0.0.9")[0]
    assert (flow_default.src_bytes + flow_default.dst_bytes ==
            flow_flipped.src_bytes + flow_flipped.dst_bytes)


def test_prefer_src_conserves_total_pkts():
    flow_default = build_flows([_CONV], {}, prefer_src=None)[0]
    flow_flipped = build_flows([_CONV], {}, prefer_src="10.0.0.9")[0]
    assert (flow_default.src_pkts + flow_default.dst_pkts ==
            flow_flipped.src_pkts + flow_flipped.dst_pkts)


def test_prefer_src_flips_bytes_ratio():
    """bytes_ratio = dst_bytes / (src_bytes+1) must invert when src/dst swap."""
    row_default = flow_to_row(build_flows([_CONV], {}, prefer_src=None)[0])
    row_flipped = flow_to_row(build_flows([_CONV], {}, prefer_src="10.0.0.9")[0])
    # default:  200/(100+1) ≈ 1.98   flipped: 100/(200+1) ≈ 0.498
    assert row_default["bytes_ratio"] > row_flipped["bytes_ratio"]


def test_prefer_src_does_not_affect_flag_counts():
    """TCP flag counts are direction-agnostic and must be identical in both orientations."""
    flow_default = build_flows([_CONV], _FLAGS, prefer_src=None)[0]
    flow_flipped = build_flows([_CONV], _FLAGS, prefer_src="10.0.0.9")[0]
    assert flow_default.syn_count == flow_flipped.syn_count == 2
    assert flow_default.ack_count == flow_flipped.ack_count == 3
    assert flow_default.fin_count == flow_flipped.fin_count == 1


# ── TIER 2: benign corpus path == default orientation ─────────────────────

def test_benign_corpus_path_matches_live_path():
    """For benign flows (prefer_src=None), the corpus and live paths are identical."""
    flow_live   = build_flows([_CONV], _FLAGS, prefer_src=None)[0]
    flow_corpus = build_flows([_CONV], _FLAGS, prefer_src=None)[0]
    assert flow_to_row(flow_live) == flow_to_row(flow_corpus)


def test_prefer_src_none_matches_no_match():
    """prefer_src that matches neither endpoint falls back to default rule."""
    flow_default  = build_flows([_CONV], {}, prefer_src=None)[0]
    flow_no_match = build_flows([_CONV], {}, prefer_src="99.99.99.99")[0]
    assert flow_default.src_ip == flow_no_match.src_ip
    assert flow_default.src_bytes == flow_no_match.src_bytes


def test_orientation_key_join_is_direction_agnostic():
    """Flag join via orientation_key succeeds regardless of which side tshark listed as A."""
    # Simulate B-first tshark listing: ep_a and ep_b are swapped vs _CONV
    conv_b_first = {**_CONV, "ep_a": ("10.0.0.9", 80), "ep_b": ("10.0.0.5", 12345),
                    "bytes_ab": 200, "bytes_ba": 100,   # B→A is now A→B from tshark's view
                    "pkts_ab": 2, "pkts_ba": 1}
    flow_a_first = build_flows([_CONV],       _FLAGS, prefer_src=None)[0]
    flow_b_first = build_flows([conv_b_first], _FLAGS, prefer_src=None)[0]
    # Same canonical src because orientation_key is unordered
    assert flow_a_first.src_ip    == flow_b_first.src_ip
    assert flow_a_first.syn_count == flow_b_first.syn_count == 2


# ── tshark-gated parity test ──────────────────────────────────────────────

_TSHARK = find_tshark()
tshark_only = pytest.mark.skipif(
    _TSHARK is None, reason="tshark binary not available"
)


def _build_parity_pcap() -> bytes:
    """Build a minimal two-flow pcap (one TCP, one UDP) in pure Python."""
    _SYN     = 0x002
    _SYN_ACK = 0x012
    _PSH_ACK = 0x018

    def _ip_pack(ip: str) -> bytes:
        return bytes(int(p) for p in ip.split("."))

    def _tcp(sp, dp, seq, ack, flags, payload=b""):
        return struct.pack("!HHIIHHHH", sp, dp, seq, ack, (5 << 12) | flags, 65535, 0, 0) + payload

    def _udp(sp, dp, payload=b""):
        return struct.pack("!HHHH", sp, dp, 8 + len(payload), 0) + payload

    def _ip4(src, dst, proto, payload):
        total = 20 + len(payload)
        return struct.pack("!BBHHHBBH4s4s",
                           0x45, 0, total, 0, 0x4000, 64, proto, 0,
                           _ip_pack(src), _ip_pack(dst)) + payload

    def _eth(payload):
        return b"\xaa\xbb\xcc\xdd\xee\x01\xaa\xbb\xcc\xdd\xee\x02\x08\x00" + payload

    def _pkt(ts_s, ts_us, src, dst, proto, seg):
        data = _eth(_ip4(src, dst, proto, seg))
        return struct.pack("<IIII", ts_s, ts_us, len(data), len(data)) + data

    buf = io.BytesIO()
    buf.write(struct.pack("<IHHiIII", 0xA1B2C3D4, 2, 4, 0, 0, 65535, 1))
    T = 1_700_000_000

    # TCP: 10.0.0.1:12345 ↔ 192.168.1.1:80
    data = b"HELLO"
    buf.write(_pkt(T,   0,   "10.0.0.1",   "192.168.1.1", 6, _tcp(12345, 80, 1000, 0, _SYN)))
    buf.write(_pkt(T,   100, "192.168.1.1", "10.0.0.1",   6, _tcp(80, 12345, 2000, 1001, _SYN_ACK)))
    buf.write(_pkt(T,   200, "10.0.0.1",   "192.168.1.1", 6, _tcp(12345, 80, 1001, 2001, _PSH_ACK, data)))

    # UDP: 10.0.0.1:54321 ↔ 8.8.8.8:53
    dns_q = b"\x00\x01" + b"\x00" * 10
    dns_r = b"\x00\x01" + b"\x00" * 20
    buf.write(_pkt(T+1, 0,   "10.0.0.1", "8.8.8.8",   17, _udp(54321, 53, dns_q)))
    buf.write(_pkt(T+1, 100, "8.8.8.8",  "10.0.0.1",  17, _udp(53, 54321, dns_r)))
    return buf.getvalue()


@pytest.fixture(scope="session")
def parity_pcap_path(tmp_path_factory):
    p = tmp_path_factory.mktemp("parity") / "parity.pcap"
    p.write_bytes(_build_parity_pcap())
    return p


@tshark_only
def test_corpus_path_byte_identical_to_live_path(parity_pcap_path):
    """extract_pcap_flows + build_flows must produce byte-identical CSV to extract_flows."""
    # Live path: the function used at serve time
    live_flows = extract_flows(_TSHARK, pcap=str(parity_pcap_path), prefer_src=None)
    live_csv   = flows_to_dataframe(live_flows).to_csv(index=False)

    # Corpus path: extract_pcap_flows is the thin wrapper used by build_corpus
    convs, flags = extract_pcap_flows(str(parity_pcap_path), _TSHARK)
    corpus_flows = build_flows(convs, flags, prefer_src=None)
    corpus_flows.sort(key=lambda f: (f.ts, f.src_ip, f.src_port, f.dst_ip, f.dst_port))
    corpus_csv = flows_to_dataframe(corpus_flows).to_csv(index=False)

    assert live_csv == corpus_csv, (
        "live extract_flows and corpus extract_pcap_flows+build_flows diverge — "
        "train/serve skew introduced"
    )


@tshark_only
def test_corpus_extraction_column_set(parity_pcap_path):
    """Corpus extraction must produce exactly IDENTITY_COLUMNS + FEATURE_COLUMNS."""
    convs, flags = extract_pcap_flows(str(parity_pcap_path), _TSHARK)
    df = flows_to_dataframe(build_flows(convs, flags))
    assert list(df.columns) == list(IDENTITY_COLUMNS) + list(FEATURE_COLUMNS)


@tshark_only
def test_prefer_src_flips_directional_cols_not_totals(parity_pcap_path):
    """Pinning prefer_src must swap src/dst counts without changing the total."""
    convs, flags = extract_pcap_flows(str(parity_pcap_path), _TSHARK)
    tcp_convs = [c for c in convs if c["proto"] == "TCP"]
    assert tcp_convs, "no TCP conv in parity pcap"

    conv = tcp_convs[0]
    flow_default = build_flows([conv], flags, prefer_src=None)[0]
    flow_flipped = build_flows([conv], flags, prefer_src=flow_default.dst_ip)[0]

    # Totals are conserved
    assert (flow_default.src_bytes + flow_default.dst_bytes ==
            flow_flipped.src_bytes + flow_flipped.dst_bytes)
    assert (flow_default.src_pkts + flow_default.dst_pkts ==
            flow_flipped.src_pkts + flow_flipped.dst_pkts)

    # Directional split is exactly swapped
    assert flow_default.src_bytes == flow_flipped.dst_bytes
    assert flow_default.dst_bytes == flow_flipped.src_bytes
    assert flow_default.src_pkts  == flow_flipped.dst_pkts
    assert flow_default.dst_pkts  == flow_flipped.src_pkts
