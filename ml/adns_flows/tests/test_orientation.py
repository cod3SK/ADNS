"""
Orientation canonicalization tests (STEP 4 — orientation as explicit contract).

Verifies:
  - orientation_key is symmetric (unordered)
  - canonicalize_orientation is idempotent: (a,b) and (b,a) → same (src,dst)
  - default rule: src = lower (ip, port) — total, stable, deterministic
  - tie-breaking: same IP, lower port wins; identical endpoints → ep_a is src
  - prefer_src override: pins a named IP as src; falls back to default on no match
  - KNOWN-INITIATOR: same flow listed with A/B sides swapped in tshark conv
    output produces identical final feature rows — pure-Python and tshark-gated
  - DIRECTION-AGNOSTIC JOIN: syn_count is consistent regardless of capture order
"""
from __future__ import annotations

import pytest

from adns_flows.assemble import build_flows
from adns_flows.extract import find_tshark, parse_conv_output
from adns_flows.schema import canonicalize_orientation, orientation_key

# ── orientation_key ────────────────────────────────────────────────────────

def test_orientation_key_symmetric():
    assert (
        orientation_key("10.0.0.5", 12345, "10.0.0.9", 80)
        == orientation_key("10.0.0.9", 80, "10.0.0.5", 12345)
    )


def test_orientation_key_self():
    """Degenerate: same endpoint on both sides still returns a valid key."""
    k = orientation_key("1.2.3.4", 80, "1.2.3.4", 80)
    assert k == (("1.2.3.4", 80), ("1.2.3.4", 80))


def test_orientation_key_min_is_first():
    k = orientation_key("10.0.0.9", 80, "10.0.0.5", 12345)
    assert k[0] == ("10.0.0.5", 12345)  # lower IP → first slot
    assert k[1] == ("10.0.0.9", 80)


# ── canonicalize_orientation — default rule ────────────────────────────────

def test_idempotent_ab():
    a = ("10.0.0.5", 12345)
    b = ("10.0.0.9", 80)
    assert canonicalize_orientation(a, b) == canonicalize_orientation(b, a)


def test_default_lower_ip_is_src():
    a = ("10.0.0.5", 12345)
    b = ("10.0.0.9", 80)
    src, dst = canonicalize_orientation(a, b)
    assert src == a     # "10.0.0.5" < "10.0.0.9"
    assert dst == b


def test_default_higher_ip_first_still_gets_same_src():
    a = ("10.0.0.9", 80)
    b = ("10.0.0.5", 12345)
    src, dst = canonicalize_orientation(a, b)
    assert src == b     # "10.0.0.5" < "10.0.0.9" wins regardless of call order
    assert dst == a


def test_tiebreak_same_ip_lower_port_is_src():
    a = ("10.0.0.1", 80)
    b = ("10.0.0.1", 443)
    src, dst = canonicalize_orientation(a, b)
    assert src == a     # port 80 < 443


def test_tiebreak_same_ip_lower_port_still_wins_when_reversed():
    a = ("10.0.0.1", 443)
    b = ("10.0.0.1", 80)
    src, dst = canonicalize_orientation(a, b)
    assert src == b     # port 80 < 443


def test_degenerate_identical_endpoints():
    a = ("1.2.3.4", 80)
    src, dst = canonicalize_orientation(a, a)
    assert src == a     # tie-break: ep_a when equal


# ── prefer_src override ────────────────────────────────────────────────────

def test_prefer_src_pins_matching_endpoint_a():
    a = ("10.0.0.5", 12345)
    b = ("10.0.0.9", 80)
    # a would be src by default (lower IP); prefer_src agrees → no change
    src, dst = canonicalize_orientation(a, b, prefer_src="10.0.0.5")
    assert src == a


def test_prefer_src_overrides_default_to_pin_endpoint_b():
    a = ("10.0.0.5", 12345)
    b = ("10.0.0.9", 80)
    # default: a is src; prefer_src forces b to be src instead
    src, dst = canonicalize_orientation(a, b, prefer_src="10.0.0.9")
    assert src == b
    assert dst == a


def test_prefer_src_no_match_falls_back_to_default():
    a = ("10.0.0.5", 12345)
    b = ("10.0.0.9", 80)
    src, dst = canonicalize_orientation(a, b, prefer_src="192.168.1.1")
    # prefer_src matched neither endpoint → default rule
    assert src == a     # "10.0.0.5" < "10.0.0.9"


# ── KNOWN-INITIATOR TEST (pure-Python, no tshark) ─────────────────────────
#
# The same physical flow (10.0.0.5:12345 ↔ 10.0.0.9:80) is represented in
# two fixture strings that differ only in which side tshark listed as A:
#   FIXTURE_A_FIRST  — tshark put 10.0.0.5 on the left of <->
#   FIXTURE_B_FIRST  — tshark put 10.0.0.9 on the left of <->
#
# After canonicalize_orientation(), both must yield the same (src, dst),
# the same src_bytes/dst_bytes, and byte-identical feature rows.

_FLOW_BYTES = 500   # A→B bytes in the A-first fixture (same as B→A in B-first)
_FLOW_PKTS  = 5

FIXTURE_A_FIRST = f"""\
TCP Conversations
10.0.0.5:12345  <->  10.0.0.9:80   0 0 bytes   {_FLOW_PKTS} {_FLOW_BYTES} bytes   {_FLOW_PKTS} {_FLOW_BYTES} bytes   0.000000  1.000
"""

# Same conversation but tshark listed 10.0.0.9 on the left.
# bytes_ab column (left→right) is now 0; bytes_ba (right→left) is _FLOW_BYTES.
FIXTURE_B_FIRST = f"""\
TCP Conversations
10.0.0.9:80  <->  10.0.0.5:12345   {_FLOW_PKTS} {_FLOW_BYTES} bytes   0 0 bytes   {_FLOW_PKTS} {_FLOW_BYTES} bytes   0.000000  1.000
"""


def _single_flow(fixture_text: str) -> object:
    convs = parse_conv_output(fixture_text)
    assert len(convs) == 1, f"expected 1 conv, got {len(convs)}"
    flows = build_flows(convs, {})
    assert len(flows) == 1
    return flows[0]


def test_known_initiator_same_src_regardless_of_tshark_a_side():
    """Canonical src must be the same endpoint in both capture orderings."""
    fa = _single_flow(FIXTURE_A_FIRST)
    fb = _single_flow(FIXTURE_B_FIRST)
    assert fa.src_ip == fb.src_ip, f"src_ip mismatch: {fa.src_ip} vs {fb.src_ip}"
    assert fa.dst_ip == fb.dst_ip
    assert fa.src_port == fb.src_port
    assert fa.dst_port == fb.dst_port


def test_known_initiator_same_bytes_regardless_of_tshark_a_side():
    """Directional bytes must be correctly oriented in both capture orderings."""
    fa = _single_flow(FIXTURE_A_FIRST)
    fb = _single_flow(FIXTURE_B_FIRST)
    # Default rule: "10.0.0.5" < "10.0.0.9" → src = 10.0.0.5
    assert fa.src_ip == "10.0.0.5"
    assert fa.src_bytes == _FLOW_BYTES   # 10.0.0.5 → 10.0.0.9
    assert fa.dst_bytes == 0             # no return traffic in fixture
    assert fb.src_bytes == fa.src_bytes, (
        f"src_bytes differ: A-first={fa.src_bytes}, B-first={fb.src_bytes}"
    )
    assert fb.dst_bytes == fa.dst_bytes


def test_known_initiator_identical_feature_rows():
    """Full feature rows must be byte-identical when tshark A-side differs."""
    from adns_flows.assemble import flows_to_dataframe
    fa = _single_flow(FIXTURE_A_FIRST)
    fb = _single_flow(FIXTURE_B_FIRST)
    csv_a = flows_to_dataframe([fa]).to_csv(index=False)
    csv_b = flows_to_dataframe([fb]).to_csv(index=False)
    assert csv_a == csv_b, "feature rows differ between A-first and B-first fixtures"


# ── DIRECTION-AGNOSTIC JOIN: SYN scan ─────────────────────────────────────
#
# A SYN-heavy flow (5 SYNs from 10.0.0.5, 1 RST-ACK from 10.0.0.9) is
# described by two conv fixtures (A-side swapped) and the same flag lines.
# syn_count must equal 5 and be consistent in both representations.

_SYN_FLAG_LINES = [
    "10.0.0.5\t10.0.0.9\t54321\t22\t0x00000002",  # SYN ×5
    "10.0.0.5\t10.0.0.9\t54321\t22\t0x00000002",
    "10.0.0.5\t10.0.0.9\t54321\t22\t0x00000002",
    "10.0.0.5\t10.0.0.9\t54321\t22\t0x00000002",
    "10.0.0.5\t10.0.0.9\t54321\t22\t0x00000002",
    "10.0.0.9\t10.0.0.5\t22\t54321\t0x00000014",  # RST+ACK
]

SCAN_CONV_A_FIRST = """\
TCP Conversations
10.0.0.5:54321  <->  10.0.0.9:22   1 60 bytes   5 300 bytes   6 360 bytes   0.000000  0.500
"""

SCAN_CONV_B_FIRST = """\
TCP Conversations
10.0.0.9:22  <->  10.0.0.5:54321   5 300 bytes   1 60 bytes   6 360 bytes   0.000000  0.500
"""


def _scan_flow(fixture_text: str, flag_lines: list[str]) -> object:
    from adns_flows.extract import parse_flag_lines
    convs = parse_conv_output(fixture_text)
    flags = parse_flag_lines(flag_lines)
    flows = build_flows(convs, flags)
    assert len(flows) == 1
    return flows[0]


def test_syn_scan_syn_count_consistent_across_capture_orderings():
    """syn_count must be 5 in both A-first and B-first representations."""
    fa = _scan_flow(SCAN_CONV_A_FIRST, _SYN_FLAG_LINES)
    fb = _scan_flow(SCAN_CONV_B_FIRST, _SYN_FLAG_LINES)
    assert fa.syn_count == 5
    assert fb.syn_count == 5


def test_syn_scan_canonical_src_is_initiator():
    """Canonical src (10.0.0.5, lower IP) should be the scanner in both orderings."""
    fa = _scan_flow(SCAN_CONV_A_FIRST, _SYN_FLAG_LINES)
    fb = _scan_flow(SCAN_CONV_B_FIRST, _SYN_FLAG_LINES)
    assert fa.src_ip == "10.0.0.5"
    assert fb.src_ip == "10.0.0.5"


def test_syn_scan_src_bytes_consistent():
    """src_bytes (scanner→target) must be the same regardless of A-side."""
    fa = _scan_flow(SCAN_CONV_A_FIRST, _SYN_FLAG_LINES)
    fb = _scan_flow(SCAN_CONV_B_FIRST, _SYN_FLAG_LINES)
    assert fa.src_bytes == fb.src_bytes, (
        f"src_bytes differ: A-first={fa.src_bytes}, B-first={fb.src_bytes}"
    )
    assert fa.dst_bytes == fb.dst_bytes


# ── tshark-gated version of the known-initiator test ──────────────────────

_TSHARK = find_tshark()
tshark_only = pytest.mark.skipif(
    _TSHARK is None, reason="tshark binary not available"
)


@tshark_only
def test_known_initiator_tshark_pcap_same_orientation(
    fixture_pcap_path, initiator_pcap_fwd_path, initiator_pcap_rev_path
):
    """
    Two pcap files carry the same 10.0.0.5→10.0.0.9 flow with opposite packet
    ordering. After extraction, canonical src must be the same in both.
    """
    from adns_flows.assemble import extract_flows

    flows_fwd = extract_flows(_TSHARK, pcap=str(initiator_pcap_fwd_path))
    flows_rev = extract_flows(_TSHARK, pcap=str(initiator_pcap_rev_path))

    assert len(flows_fwd) == 1
    assert len(flows_rev) == 1

    ff, fr = flows_fwd[0], flows_rev[0]
    assert ff.src_ip == fr.src_ip, f"src_ip: fwd={ff.src_ip} rev={fr.src_ip}"
    assert ff.dst_ip == fr.dst_ip
    assert ff.src_bytes == fr.src_bytes, (
        f"src_bytes: fwd={ff.src_bytes} rev={fr.src_bytes}"
    )
    assert ff.dst_bytes == fr.dst_bytes
