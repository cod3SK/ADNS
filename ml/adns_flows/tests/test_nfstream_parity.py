"""
Phase 1 NFStream parity checks.

Step 4.1 — Hand-fixture diff
    Run the 3-flow fixture pcap through extract_flows_nfstream() and verify every
    contract feature against hand-computed L2 byte counts and bidirectional flag
    counts.  No tshark required.

Step 4.2 — Config-parity (feature-neutral parameter)
    Same pcap, n_meters=1 vs n_meters=2.  Feature output must be byte-identical
    because n_meters is a parallelism knob, not a feature-affecting parameter.

Step 4.3 — Determinism
    Same pcap twice → identical DataFrames.

Step 4.4 — Orientation invariance
    initiator_fwd and initiator_rev pcaps (same logical flow, different packet
    order on disk) must produce the same canonical src_ip / src_port after
    canonicalize_orientation().

Tshark-gated cross-extractor check (Step 4.1b)
    When tshark is available, run the same fixture through both extractors and
    compare both flag counts AND byte counts — both must match exactly.

Byte-accounting finding
-----------------------
NFStream accounting_mode=0 and tshark -z conv,tcp both count Ethernet frame
bytes (L2).  They agree on byte counts for the same pcap.

Flag counts (Bucket A — must match tshark exactly)
---------------------------------------------------
Both extractors aggregate bidirectional TCP flag counts.  Both must produce
identical syn/ack/rst/fin/psh/urg values for the same pcap.

Fixture layout (from conftest.py build_fixture_pcap())
------------------------------------------------------
Flow 1: 192.168.1.1:12345 <-> 10.0.0.1:80   TCP  8 packets  T+0..T+0.0008s
  canonical src = (10.0.0.1, 80)   [lower IP string: "10" < "192"]
  NFStream orientation FLIPS (initiator = 192.168.1.1)

  From 192.168.1.1 (NFStream src / canonical dst):
    pkt1 SYN       ETH+IP+TCP          = 54 B
    pkt3 ACK                            = 54 B
    pkt4 PSH-ACK + req(18 B)            = 72 B
    pkt6 FIN-ACK                        = 54 B
    pkt8 ACK                            = 54 B
    Total: 5 pkts, 288 bytes

  From 10.0.0.1 (NFStream dst / canonical src):
    pkt2 SYN-ACK                        = 54 B
    pkt5 PSH-ACK + resp(24 B)           = 78 B
    pkt7 FIN-ACK                        = 54 B
    Total: 3 pkts, 186 bytes

  canonical src_bytes=186, dst_bytes=288, src_pkts=3, dst_pkts=5
  Flags: syn=2, ack=7, rst=0, fin=2, psh=2, urg=0

Flow 2: 192.168.1.2:54321 <-> 10.0.0.2:22  TCP  6 packets  T+1..T+2s
  canonical src = (10.0.0.2, 22)   [lower IP: "10" < "192"]
  NFStream orientation FLIPS (initiator = 192.168.1.2)

  From 192.168.1.2 (NFStream src / canonical dst): 5 SYN  = 5 × 54 = 270 B
  From 10.0.0.2  (NFStream dst / canonical src): 1 RST-ACK = 54 B
  canonical src_bytes=54, dst_bytes=270, src_pkts=1, dst_pkts=5
  Flags: syn=5, ack=1, rst=1, fin=0, psh=0, urg=0

Flow 3: 192.168.1.1:54322 <-> 8.8.8.8:53   UDP  2 packets  T+2
  canonical src = (192.168.1.1, 54322)  ["1" < "8"]
  NFStream orientation SAME (initiator = 192.168.1.1)

  dns_q payload = 12 B → frame = ETH(14)+IP(20)+UDP(8)+12 = 54 B
  dns_r payload = 22 B → frame = ETH(14)+IP(20)+UDP(8)+22 = 64 B
  canonical src_bytes=54, dst_bytes=64, src_pkts=1, dst_pkts=1
  All flag counts = 0 (UDP)
"""
from __future__ import annotations

import pytest

# Skip entire module if nfstream is not installed.
nfstream = pytest.importorskip("nfstream")

from adns_flows.extract import find_tshark
from adns_flows.extract_nfstream import extract_flows_nfstream, flows_to_dataframe_nfstream
from adns_flows.schema import FEATURE_COLUMNS, IDENTITY_COLUMNS

_TSHARK = find_tshark()
tshark_only = pytest.mark.skipif(_TSHARK is None, reason="tshark binary not available")

# ── helpers ───────────────────────────────────────────────────────────────────

def _get_flow(flows, src_ip, dst_ip):
    """Return the first flow with matching canonical src_ip and dst_ip."""
    for f in flows:
        if f.src_ip == src_ip and f.dst_ip == dst_ip:
            return f
    raise AssertionError(f"No flow found with src={src_ip!r} dst={dst_ip!r}. "
                         f"Available: {[(f.src_ip, f.dst_ip) for f in flows]}")


# ── Step 4.1: Hand-fixture diff ───────────────────────────────────────────────

def test_fixture_flow_count(fixture_pcap_path):
    flows = extract_flows_nfstream(str(fixture_pcap_path))
    assert len(flows) == 3, f"expected 3 flows, got {len(flows)}"


def test_fixture_protocols(fixture_pcap_path):
    flows = extract_flows_nfstream(str(fixture_pcap_path))
    protos = {f.proto for f in flows}
    assert 6 in protos,  "TCP (proto=6) not found"
    assert 17 in protos, "UDP (proto=17) not found"


def test_fixture_flow1_canonical_orientation(fixture_pcap_path):
    """Flow 1: NFStream initiator=192.168.1.1; canonical src must be 10.0.0.1:80."""
    flows = extract_flows_nfstream(str(fixture_pcap_path))
    f = _get_flow(flows, "10.0.0.1", "192.168.1.1")
    assert f.src_port == 80
    assert f.dst_port == 12345


def test_fixture_flow1_bytes(fixture_pcap_path):
    """Flow 1 L2 byte counts after canonical orientation flip."""
    flows = extract_flows_nfstream(str(fixture_pcap_path))
    f = _get_flow(flows, "10.0.0.1", "192.168.1.1")
    # canonical src = 10.0.0.1 (server side): SYN-ACK(54) + PSH-ACK+resp(78) + FIN-ACK(54) = 186
    assert f.src_bytes == 186, f"src_bytes={f.src_bytes}, expected 186"
    # canonical dst = 192.168.1.1 (client): SYN(54)+ACK(54)+PSH-ACK+req(72)+FIN-ACK(54)+ACK(54) = 288
    assert f.dst_bytes == 288, f"dst_bytes={f.dst_bytes}, expected 288"


def test_fixture_flow1_packets(fixture_pcap_path):
    flows = extract_flows_nfstream(str(fixture_pcap_path))
    f = _get_flow(flows, "10.0.0.1", "192.168.1.1")
    assert f.src_pkts == 3, f"src_pkts={f.src_pkts}, expected 3 (SYN-ACK, PSH-ACK, FIN-ACK)"
    assert f.dst_pkts == 5, f"dst_pkts={f.dst_pkts}, expected 5 (SYN, ACK, PSH-ACK, FIN-ACK, ACK)"


def test_fixture_flow1_flags(fixture_pcap_path):
    flows = extract_flows_nfstream(str(fixture_pcap_path))
    f = _get_flow(flows, "10.0.0.1", "192.168.1.1")
    # SYN in pkt1(SYN) + pkt2(SYN-ACK) = 2
    assert f.syn_count == 2, f"syn_count={f.syn_count}"
    # ACK in pkt2+pkt3+pkt4+pkt5+pkt6+pkt7+pkt8 = 7
    assert f.ack_count == 7, f"ack_count={f.ack_count}"
    assert f.rst_count == 0
    # FIN in pkt6(FIN-ACK) + pkt7(FIN-ACK) = 2
    assert f.fin_count == 2, f"fin_count={f.fin_count}"
    # PSH in pkt4(PSH-ACK) + pkt5(PSH-ACK) = 2
    assert f.psh_count == 2, f"psh_count={f.psh_count}"
    assert f.urg_count == 0


def test_fixture_flow2_canonical_orientation(fixture_pcap_path):
    """Flow 2: NFStream initiator=192.168.1.2; canonical src must be 10.0.0.2:22."""
    flows = extract_flows_nfstream(str(fixture_pcap_path))
    f = _get_flow(flows, "10.0.0.2", "192.168.1.2")
    assert f.src_port == 22
    assert f.dst_port == 54321


def test_fixture_flow2_syn_heavy(fixture_pcap_path):
    """Flow 2 is a SYN scan: syn_count=5, rst_count=1, src_pkts=1, dst_pkts=5."""
    flows = extract_flows_nfstream(str(fixture_pcap_path))
    f = _get_flow(flows, "10.0.0.2", "192.168.1.2")
    assert f.syn_count == 5, f"syn_count={f.syn_count}"
    assert f.rst_count == 1, f"rst_count={f.rst_count}"
    assert f.ack_count == 1, f"ack_count={f.ack_count} (only the RST-ACK)"
    assert f.fin_count == 0
    assert f.src_pkts == 1, f"src_pkts={f.src_pkts} (1 RST-ACK from 10.0.0.2)"
    assert f.dst_pkts == 5, f"dst_pkts={f.dst_pkts} (5 SYNs from 192.168.1.2)"
    assert f.src_bytes == 54,  f"src_bytes={f.src_bytes}"
    assert f.dst_bytes == 270, f"dst_bytes={f.dst_bytes}"


def test_fixture_flow3_udp_orientation_no_flip(fixture_pcap_path):
    """Flow 3 UDP: canonical src=192.168.1.1:54322 (no flip — lower IP than 8.8.8.8)."""
    flows = extract_flows_nfstream(str(fixture_pcap_path))
    f = _get_flow(flows, "192.168.1.1", "8.8.8.8")
    assert f.proto == 17
    assert f.src_port == 54322
    assert f.dst_port == 53
    assert f.src_bytes == 54, f"src_bytes={f.src_bytes}"   # ETH+IP+UDP+dns_q(12)
    assert f.dst_bytes == 64, f"dst_bytes={f.dst_bytes}"   # ETH+IP+UDP+dns_r(22)
    assert f.src_pkts == 1 and f.dst_pkts == 1


def test_fixture_flow3_udp_flags_zero(fixture_pcap_path):
    flows = extract_flows_nfstream(str(fixture_pcap_path))
    f = _get_flow(flows, "192.168.1.1", "8.8.8.8")
    assert f.syn_count == f.ack_count == f.rst_count == 0
    assert f.fin_count == f.psh_count == f.urg_count == 0


# ── Step 4.2: Config-parity — n_meters is feature-neutral ────────────────────

def test_config_parity_n_meters(fixture_pcap_path):
    """n_meters=1 vs n_meters=2 must produce byte-identical feature output."""
    flows1 = extract_flows_nfstream(str(fixture_pcap_path), n_meters=1)
    flows2 = extract_flows_nfstream(str(fixture_pcap_path), n_meters=2)
    df1 = flows_to_dataframe_nfstream(flows1)
    df2 = flows_to_dataframe_nfstream(flows2)
    assert df1.to_csv(index=False) == df2.to_csv(index=False), (
        "n_meters=1 vs n_meters=2 produced different feature output — "
        "n_meters must be feature-neutral"
    )


# ── Step 4.3: Determinism ─────────────────────────────────────────────────────

def test_determinism_same_pcap_twice(fixture_pcap_path):
    """Two calls on the same pcap must produce byte-identical DataFrames."""
    flows_a = extract_flows_nfstream(str(fixture_pcap_path))
    flows_b = extract_flows_nfstream(str(fixture_pcap_path))
    csv_a = flows_to_dataframe_nfstream(flows_a).to_csv(index=False)
    csv_b = flows_to_dataframe_nfstream(flows_b).to_csv(index=False)
    assert csv_a == csv_b


# ── Step 4.4: Orientation invariance ─────────────────────────────────────────

def test_orientation_fwd_vs_rev(initiator_pcap_fwd_path, initiator_pcap_rev_path):
    """Same logical flow, different initiator order on disk → same canonical src."""
    flows_fwd = extract_flows_nfstream(str(initiator_pcap_fwd_path))
    flows_rev = extract_flows_nfstream(str(initiator_pcap_rev_path))
    assert len(flows_fwd) == 1 and len(flows_rev) == 1
    f_fwd = flows_fwd[0]
    f_rev = flows_rev[0]
    assert f_fwd.src_ip   == f_rev.src_ip,   f"{f_fwd.src_ip!r} != {f_rev.src_ip!r}"
    assert f_fwd.src_port == f_rev.src_port,  f"{f_fwd.src_port} != {f_rev.src_port}"
    assert f_fwd.dst_ip   == f_rev.dst_ip,   f"{f_fwd.dst_ip!r} != {f_rev.dst_ip!r}"
    assert f_fwd.dst_port == f_rev.dst_port,  f"{f_fwd.dst_port} != {f_rev.dst_port}"
    # The canonical rule for (10.0.0.5:12345, 10.0.0.9:80): "10.0.0.5" < "10.0.0.9"
    assert f_fwd.src_ip == "10.0.0.5" and f_fwd.src_port == 12345, (
        f"Expected canonical src=10.0.0.5:12345, got {f_fwd.src_ip}:{f_fwd.src_port}"
    )


# ── Step 4.4b: DataFrame schema passes validate_matrix ───────────────────────

def test_dataframe_schema_valid(fixture_pcap_path):
    """flows_to_dataframe_nfstream must not raise SchemaError."""
    flows = extract_flows_nfstream(str(fixture_pcap_path))
    df = flows_to_dataframe_nfstream(flows)
    expected_cols = list(IDENTITY_COLUMNS) + list(FEATURE_COLUMNS)
    assert list(df.columns) == expected_cols


# ── Step 4.1b: Cross-extractor flag parity (requires tshark) ─────────────────

@tshark_only
def test_flag_counts_match_tshark(fixture_pcap_path):
    """Flag counts from NFStream and tshark must be identical for the same pcap.

    Bucket A (must match): syn/ack/rst/fin/psh/urg counts.
    Bucket B (expected diff): byte counts differ by 14 bytes/packet (L2 vs L3).
    """
    from adns_flows.assemble import extract_flows

    nf_flows = extract_flows_nfstream(str(fixture_pcap_path))
    ts_flows = extract_flows(_TSHARK, pcap=str(fixture_pcap_path))

    # Key both flow lists by canonical (src_ip, src_port, dst_ip, dst_port)
    def _key(f):
        return (f.src_ip, f.src_port, f.dst_ip, f.dst_port)

    nf_map = {_key(f): f for f in nf_flows}
    ts_map = {_key(f): f for f in ts_flows}

    assert set(nf_map) == set(ts_map), (
        f"Flow keys differ:\n  nfstream={sorted(nf_map)}\n  tshark={sorted(ts_map)}"
    )

    for key in nf_map:
        nf = nf_map[key]
        ts = ts_map[key]
        label = f"{key[0]}:{key[1]}→{key[2]}:{key[3]}"
        assert nf.syn_count == ts.syn_count, f"{label} syn: nf={nf.syn_count} ts={ts.syn_count}"
        assert nf.ack_count == ts.ack_count, f"{label} ack: nf={nf.ack_count} ts={ts.ack_count}"
        assert nf.rst_count == ts.rst_count, f"{label} rst: nf={nf.rst_count} ts={ts.rst_count}"
        assert nf.fin_count == ts.fin_count, f"{label} fin: nf={nf.fin_count} ts={ts.fin_count}"
        assert nf.psh_count == ts.psh_count, f"{label} psh: nf={nf.psh_count} ts={ts.psh_count}"
        assert nf.urg_count == ts.urg_count, f"{label} urg: nf={nf.urg_count} ts={ts.urg_count}"
        # Both extractors count Ethernet frame bytes (L2) — values must be equal.
        assert nf.src_bytes == ts.src_bytes, f"{label} src_bytes: nf={nf.src_bytes} ts={ts.src_bytes}"
        assert nf.dst_bytes == ts.dst_bytes, f"{label} dst_bytes: nf={nf.dst_bytes} ts={ts.dst_bytes}"
        assert nf.src_pkts  == ts.src_pkts,  f"{label} src_pkts:  nf={nf.src_pkts} ts={ts.src_pkts}"
        assert nf.dst_pkts  == ts.dst_pkts,  f"{label} dst_pkts:  nf={nf.dst_pkts} ts={ts.dst_pkts}"
