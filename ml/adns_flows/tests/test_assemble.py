"""
Tests for assemble.py: flow joining, orientation, DataFrame column order, and determinism.
tshark-gated end-to-end tests use the fixture pcap from conftest.py.
"""
from __future__ import annotations

import pytest

from adns_flows.assemble import build_flows, extract_flows, flows_to_dataframe
from adns_flows.extract import find_tshark
from adns_flows.schema import FEATURE_COLUMNS, IDENTITY_COLUMNS, orientation_key

# ── unit tests (no tshark) ────────────────────────────────────────────────
#
# Conv dicts now use neutral names: ep_a, ep_b, bytes_ab, bytes_ba, pkts_ab, pkts_ba.
# Canonical rule: src = lower (ip, port).
# For ("1.2.3.4", 12345) vs ("5.6.7.8", 80): "1.2.3.4" < "5.6.7.8" → src = ep_a.
# For ("1.2.3.4", 54321) vs ("8.8.8.8",  53): "1.2.3.4" < "8.8.8.8"  → src = ep_a.

_SAMPLE_CONVS = [
    {
        "proto": "TCP",
        "ep_a": ("1.2.3.4", 12345),
        "ep_b": ("5.6.7.8", 80),
        "bytes_ab": 1000, "bytes_ba": 2000,
        "pkts_ab": 5, "pkts_ba": 10,
        "duration": 1.5, "rel_start": 0.0,
    },
    {
        "proto": "UDP",
        "ep_a": ("1.2.3.4", 54321),
        "ep_b": ("8.8.8.8", 53),
        "bytes_ab": 60, "bytes_ba": 120,
        "pkts_ab": 1, "pkts_ba": 1,
        "duration": 0.01, "rel_start": 1.0,
    },
]

_TCP_KEY = orientation_key("1.2.3.4", 12345, "5.6.7.8", 80)
_SAMPLE_FLAGS = {
    _TCP_KEY: {"syn": 1, "ack": 5, "rst": 0, "fin": 1, "psh": 2, "urg": 0},
}


def test_build_flows_count():
    flows = build_flows(_SAMPLE_CONVS, _SAMPLE_FLAGS)
    assert len(flows) == 2


def test_build_flows_proto_codes():
    flows = build_flows(_SAMPLE_CONVS, _SAMPLE_FLAGS)
    protos = {f.proto for f in flows}
    assert 6 in protos   # TCP
    assert 17 in protos  # UDP


def test_build_flows_tcp_flag_join():
    flows = build_flows(_SAMPLE_CONVS, _SAMPLE_FLAGS)
    tcp = next(f for f in flows if f.proto == 6)
    assert tcp.syn_count == 1
    assert tcp.ack_count == 5
    assert tcp.fin_count == 1


def test_build_flows_udp_flags_are_zero():
    flows = build_flows(_SAMPLE_CONVS, _SAMPLE_FLAGS)
    udp = next(f for f in flows if f.proto == 17)
    assert udp.syn_count == 0
    assert udp.ack_count == 0
    assert udp.rst_count == 0


def test_build_flows_unmatched_conv_gets_zero_flags():
    """A conv with no matching flag key produces all-zero flag counts."""
    flows = build_flows(_SAMPLE_CONVS, {})
    tcp = next(f for f in flows if f.proto == 6)
    assert tcp.syn_count == 0


def test_build_flows_canonical_src_is_lower_ip():
    """Default rule: lower (ip, port) → src. ep_a=1.2.3.4 wins over ep_b=5.6.7.8."""
    flows = build_flows(_SAMPLE_CONVS, {})
    tcp = next(f for f in flows if f.proto == 6)
    assert tcp.src_ip == "1.2.3.4"
    assert tcp.dst_ip == "5.6.7.8"
    assert tcp.src_bytes == 1000   # bytes_ab (A→B, since ep_a is canonical src)
    assert tcp.dst_bytes == 2000   # bytes_ba


def test_build_flows_prefer_src_flips_orientation():
    """prefer_src pins ep_b as src even though ep_a has the lower IP."""
    flows = build_flows(_SAMPLE_CONVS, {}, prefer_src="5.6.7.8")
    tcp = next(f for f in flows if f.proto == 6)
    assert tcp.src_ip == "5.6.7.8"
    assert tcp.dst_ip == "1.2.3.4"
    # bytes are flipped relative to default orientation
    assert tcp.src_bytes == 2000   # bytes_ba (B→A, ep_b is now src)
    assert tcp.dst_bytes == 1000   # bytes_ab


def test_flows_to_dataframe_column_order():
    flows = build_flows(_SAMPLE_CONVS, _SAMPLE_FLAGS)
    df = flows_to_dataframe(flows)
    expected = list(IDENTITY_COLUMNS) + list(FEATURE_COLUMNS)
    assert list(df.columns) == expected


def test_flows_to_dataframe_row_count():
    flows = build_flows(_SAMPLE_CONVS, _SAMPLE_FLAGS)
    df = flows_to_dataframe(flows)
    assert len(df) == 2


def test_flows_to_dataframe_empty():
    df = flows_to_dataframe([])
    expected = list(IDENTITY_COLUMNS) + list(FEATURE_COLUMNS)
    assert list(df.columns) == expected
    assert len(df) == 0


def test_flows_to_dataframe_deterministic_regardless_of_input_order():
    """Reversed conv input order must produce byte-identical CSV output."""
    flows_fwd = build_flows(_SAMPLE_CONVS, _SAMPLE_FLAGS)
    flows_rev = build_flows(list(reversed(_SAMPLE_CONVS)), _SAMPLE_FLAGS)
    df_fwd = flows_to_dataframe(flows_fwd)
    df_rev = flows_to_dataframe(flows_rev)
    assert df_fwd.to_csv(index=False) == df_rev.to_csv(index=False)


# ── tshark-gated integration tests ────────────────────────────────────────

_TSHARK = find_tshark()
tshark_only = pytest.mark.skipif(
    _TSHARK is None, reason="tshark binary not available"
)


@tshark_only
def test_extract_fixture_pcap_flow_count(fixture_pcap_path):
    flows = extract_flows(_TSHARK, pcap=str(fixture_pcap_path))
    # fixture contains exactly 3 conversations: TCP-normal, TCP-scan, UDP-DNS
    assert len(flows) == 3


@tshark_only
def test_extract_fixture_has_tcp_and_udp(fixture_pcap_path):
    flows = extract_flows(_TSHARK, pcap=str(fixture_pcap_path))
    protos = {f.proto for f in flows}
    assert 6 in protos   # TCP
    assert 17 in protos  # UDP


@tshark_only
def test_extract_directional_split(fixture_pcap_path):
    """Every flow must have non-negative directional counts."""
    flows = extract_flows(_TSHARK, pcap=str(fixture_pcap_path))
    for f in flows:
        assert f.src_pkts >= 0
        assert f.dst_pkts >= 0


@tshark_only
def test_syn_heavy_flow_syn_gt_ack(fixture_pcap_path):
    """The SYN-scan flow (192.168.1.2 <-> 10.0.0.2:22) has syn_count > ack_count."""
    flows = extract_flows(_TSHARK, pcap=str(fixture_pcap_path))
    scan_flow = next(
        (f for f in flows
         if f.proto == 6 and ("192.168.1.2" in (f.src_ip, f.dst_ip))),
        None,
    )
    assert scan_flow is not None, "SYN-scan flow not found in extracted flows"
    assert scan_flow.syn_count > scan_flow.ack_count, (
        f"expected syn_count({scan_flow.syn_count}) > ack_count({scan_flow.ack_count})"
    )


@tshark_only
def test_extract_determinism(fixture_pcap_path, tmp_path):
    """Two runs on the same pcap must produce byte-identical CSV output."""
    flows1 = extract_flows(_TSHARK, pcap=str(fixture_pcap_path))
    flows2 = extract_flows(_TSHARK, pcap=str(fixture_pcap_path))
    csv1 = flows_to_dataframe(flows1).to_csv(index=False)
    csv2 = flows_to_dataframe(flows2).to_csv(index=False)
    assert csv1 == csv2
