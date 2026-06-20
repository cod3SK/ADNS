"""
Tests for extract.py — pure parsing tests run without tshark.
tshark-gated integration tests require tshark on PATH or TSHARK_BIN.
"""
from __future__ import annotations

import pytest

from adns_flows.extract import (
    parse_conv_output,
    parse_flag_lines,
    parse_tshark_bytes,
)
from adns_flows.extract import find_tshark
from adns_flows.schema import orientation_key

# ── parse_tshark_bytes ─────────────────────────────────────────────────────

def test_bytes_unit():
    assert parse_tshark_bytes("1530", "bytes") == 1530


def test_kb_unit():
    assert parse_tshark_bytes("85", "kB") == 85 * 1024


def test_kib_unit():
    assert parse_tshark_bytes("4", "kib") == 4 * 1024


def test_mb_unit():
    assert parse_tshark_bytes("2", "MB") == 2 * 1024 * 1024


def test_gb_unit():
    assert parse_tshark_bytes("1", "GB") == 1 * 1024 * 1024 * 1024


def test_invalid_value_returns_zero():
    assert parse_tshark_bytes("", "bytes") == 0
    assert parse_tshark_bytes("abc", "kB") == 0


# ── parse_conv_output — fixture text, no tshark required ──────────────────
#
# parse_conv_output now returns NEUTRAL names (ep_a, ep_b, bytes_ab, bytes_ba,
# pkts_ab, pkts_ba). No src/dst naming — that happens in assemble.py.
# Fixture represents tshark 4.x output (no pipe chars, human-readable units).

FIXTURE_CONV = """\
=====================================================================================
TCP Conversations
Filter:<No Filter>

                                               |       <-      | |       ->      | |     Total     |    Relative    |   Duration   |
                                               | Frames  Bytes | | Frames  Bytes | | Frames  Bytes |      Start     |              |
10.0.0.1:443           <->  192.168.1.2:58432    12 1530 bytes    58 85 kB    70 87 kB    0.000000  30.1234
fe80::1:8765           <->  2001:db8::1:80         3 300 bytes     5 2 kB      8  3 kB    0.100000  1.0000
=====================================================================================
UDP Conversations
Filter:<No Filter>

                                               |       <-      | |       ->      | |     Total     |    Relative    |   Duration   |
                                               | Frames  Bytes | | Frames  Bytes | | Frames  Bytes |      Start     |              |
192.168.1.1:54321      <->  8.8.8.8:53           1 100 bytes     1 120 bytes    2  220 bytes    2.500000  0.0100
"""


def test_conv_parse_tcp_count():
    flows = parse_conv_output(FIXTURE_CONV)
    assert sum(1 for f in flows if f["proto"] == "TCP") == 2


def test_conv_parse_udp_count():
    flows = parse_conv_output(FIXTURE_CONV)
    assert sum(1 for f in flows if f["proto"] == "UDP") == 1


def test_conv_parse_neutral_keys_present():
    """parse_conv_output must NOT emit src_*/dst_* keys; only neutral ep_a/ep_b."""
    flows = parse_conv_output(FIXTURE_CONV)
    for f in flows:
        assert "ep_a" in f and "ep_b" in f
        assert "bytes_ab" in f and "bytes_ba" in f
        assert "pkts_ab" in f and "pkts_ba" in f
        assert "src_ip" not in f
        assert "dst_ip" not in f


def test_conv_parse_kb_units():
    """bytes_ab for the TCP flow with 85 kB in A→B direction."""
    flows = parse_conv_output(FIXTURE_CONV)
    tcp = next(f for f in flows if f["proto"] == "TCP" and f["ep_a"][0] == "10.0.0.1")
    assert tcp["bytes_ab"] == 85 * 1024


def test_conv_parse_bytes_unit():
    """bytes_ba for the TCP flow with 1530 bytes in B→A direction."""
    flows = parse_conv_output(FIXTURE_CONV)
    tcp = next(f for f in flows if f["proto"] == "TCP" and f["ep_a"][0] == "10.0.0.1")
    assert tcp["bytes_ba"] == 1530


def test_conv_parse_duration():
    flows = parse_conv_output(FIXTURE_CONV)
    tcp = next(f for f in flows if f["proto"] == "TCP" and f["ep_a"][0] == "10.0.0.1")
    assert tcp["duration"] == pytest.approx(30.1234)


def test_conv_parse_rel_start():
    flows = parse_conv_output(FIXTURE_CONV)
    udp = next(f for f in flows if f["proto"] == "UDP")
    assert udp["rel_start"] == pytest.approx(2.5)


def test_conv_parse_udp_direction():
    """A-side is 192.168.1.1; bytes_ab (A→B) = 120, bytes_ba (B→A) = 100."""
    flows = parse_conv_output(FIXTURE_CONV)
    udp = next(f for f in flows if f["proto"] == "UDP")
    assert udp["ep_a"] == ("192.168.1.1", 54321)
    assert udp["ep_b"] == ("8.8.8.8", 53)
    assert udp["bytes_ab"] == 120   # A→B
    assert udp["bytes_ba"] == 100   # B→A
    assert udp["pkts_ab"] == 1
    assert udp["pkts_ba"] == 1


def test_conv_parse_ipv6_address():
    r"""(\S+):(\d+) backtracking must extract IPv6 address and port separately."""
    flows = parse_conv_output(FIXTURE_CONV)
    ipv6 = next(
        (f for f in flows if ":" in f["ep_a"][0] or ":" in f["ep_b"][0]), None
    )
    assert ipv6 is not None
    # fe80::1:8765 → IP=fe80::1, port=8765
    assert "fe80::1" in (ipv6["ep_a"][0], ipv6["ep_b"][0])


def test_conv_parse_ipv6_other_side():
    flows = parse_conv_output(FIXTURE_CONV)
    ipv6 = next(f for f in flows if "fe80" in f["ep_a"][0])
    # 2001:db8::1:80 → IP=2001:db8::1, port=80
    assert ipv6["ep_b"] == ("2001:db8::1", 80)


# ── parse_flag_lines ───────────────────────────────────────────────────────

FLAG_LINES = [
    # ip.src          ip.dst         sport  dport  flags (hex)
    "192.168.1.1\t10.0.0.1\t12345\t80\t0x00000002",   # SYN
    "10.0.0.1\t192.168.1.1\t80\t12345\t0x00000012",   # SYN-ACK
    "192.168.1.1\t10.0.0.1\t12345\t80\t0x00000010",   # ACK
    "192.168.1.1\t10.0.0.1\t12345\t80\t0x00000018",   # PSH+ACK
    "10.0.0.1\t192.168.1.1\t80\t12345\t0x00000018",   # PSH+ACK
    "192.168.1.1\t10.0.0.1\t12345\t80\t0x00000011",   # FIN+ACK
    "10.0.0.1\t192.168.1.1\t80\t12345\t0x00000011",   # FIN+ACK
    "192.168.1.1\t10.0.0.1\t12345\t80\t0x00000010",   # ACK
]


def test_flag_lines_canonical_key():
    counts = parse_flag_lines(FLAG_LINES)
    key = orientation_key("192.168.1.1", 12345, "10.0.0.1", 80)
    assert key in counts


def test_flag_lines_syn_count():
    counts = parse_flag_lines(FLAG_LINES)
    key = orientation_key("192.168.1.1", 12345, "10.0.0.1", 80)
    assert counts[key]["syn"] == 2  # SYN from client + SYN in SYN-ACK


def test_flag_lines_ack_count():
    counts = parse_flag_lines(FLAG_LINES)
    key = orientation_key("192.168.1.1", 12345, "10.0.0.1", 80)
    # SYN-ACK, ACK, PSH+ACK×2, FIN+ACK×2, ACK = 7
    assert counts[key]["ack"] == 7


def test_flag_lines_fin_count():
    counts = parse_flag_lines(FLAG_LINES)
    key = orientation_key("192.168.1.1", 12345, "10.0.0.1", 80)
    assert counts[key]["fin"] == 2


def test_flag_lines_direction_agnostic():
    """Reversed packet direction must produce the same orientation_key."""
    fwd = parse_flag_lines(["192.168.1.1\t10.0.0.1\t12345\t80\t0x00000002"])
    rev = parse_flag_lines(["10.0.0.1\t192.168.1.1\t80\t12345\t0x00000002"])
    assert list(fwd.keys()) == list(rev.keys())


def test_flag_lines_malformed_skipped():
    counts = parse_flag_lines(["only\ttwo\tfields"])
    assert counts == {}


# ── tshark-gated integration tests ────────────────────────────────────────

_TSHARK = find_tshark()
tshark_only = pytest.mark.skipif(
    _TSHARK is None, reason="tshark binary not available"
)


@tshark_only
def test_tshark_pass_a_returns_flows(fixture_pcap_path):
    from adns_flows.extract import run_pass_a
    flows = run_pass_a(_TSHARK, pcap=str(fixture_pcap_path))
    assert len(flows) >= 1


@tshark_only
def test_tshark_pass_b_returns_flags(fixture_pcap_path):
    from adns_flows.extract import run_pass_b
    flags = run_pass_b(_TSHARK, pcap=str(fixture_pcap_path))
    assert len(flags) >= 1
