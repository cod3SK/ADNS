"""
Unit tests for the NFStream labeling functions in corpus/build_corpus.py.

All tests run without a real PCAP or live NFStream session — they call
_apply_labels_nf, _apply_labels_gotham_nf, _apply_labels_cic_nf, and
_reorient_flow directly with synthetic Flow objects.

Design invariants verified:
  1. No-match flow → label=0, retained (same as tshark _apply_labels)
  2. Time-window match → label=1, attacker re-oriented as src
  3. Orientation swap via _reorient_flow inverts all directional features
  4. _apply_labels_gotham_nf uses attacker_ips per-flow (mirrors tshark path)
  5. _apply_labels_cic_nf matches by IP+port+time window
  6. All-zero feature detection: syn_count propagates correctly
  7. Empty flow list → empty result, zero stats
"""
from __future__ import annotations

import pytest

from adns_flows.schema import Flow, orientation_key

from corpus.build_corpus import (
    CorpusStats,
    REASON_OTHER,
    TIME_TOLERANCE,
    _apply_labels_cic_nf,
    _apply_labels_gotham_nf,
    _apply_labels_nf,
    _reorient_flow,
    load_label_index,
)

# ── shared test data ───────────────────────────────────────────────────────────

_EPOCH = 1_700_000_000.0

def _flow(
    src_ip="10.1.1.1", src_port=1000,
    dst_ip="10.2.2.2", dst_port=80,
    proto=6, ts=None,
    src_bytes=200, dst_bytes=100,
    src_pkts=2, dst_pkts=1,
    duration=1.0,
    syn_count=3, ack_count=5, rst_count=0, fin_count=1, psh_count=2, urg_count=0,
) -> Flow:
    return Flow(
        ts=ts if ts is not None else _EPOCH + 5.0,
        src_ip=src_ip, dst_ip=dst_ip,
        src_port=src_port, dst_port=dst_port,
        proto=proto, duration=duration,
        src_bytes=src_bytes, dst_bytes=dst_bytes,
        src_pkts=src_pkts, dst_pkts=dst_pkts,
        syn_count=syn_count, ack_count=ack_count,
        rst_count=rst_count, fin_count=fin_count,
        psh_count=psh_count, urg_count=urg_count,
    )


def _unsw_label_row(flow: Flow, row_idx: int = 0, attack_cat: str = "DoS") -> dict:
    """Synthetic UNSW GT label row whose time window contains flow.ts."""
    return {
        "srcip":      flow.dst_ip,   # attacker = dst in default orientation
        "dstip":      flow.src_ip,
        "sport":      flow.dst_port,
        "dsport":     flow.src_port,
        "proto":      "tcp",
        "stime":      flow.ts - 0.5,
        "ltime":      flow.ts + 0.5,
        "label":      1,
        "attack_cat": attack_cat,
        "_row_idx":   row_idx,
    }


# ══════════════════════════════════════════════════════════════════════════════
# _reorient_flow
# ══════════════════════════════════════════════════════════════════════════════

def test_reorient_noop_when_already_src():
    f = _flow(src_ip="10.1.1.1", dst_ip="10.2.2.2")
    result = _reorient_flow(f, prefer_src="10.1.1.1")
    assert result is f   # same object, no swap needed


def test_reorient_noop_when_prefer_src_none():
    f = _flow()
    result = _reorient_flow(f, prefer_src=None)
    assert result is f


def test_reorient_swaps_when_prefer_src_is_dst():
    f = _flow(src_ip="10.1.1.1", src_port=1000,
              dst_ip="10.2.2.2", dst_port=80,
              src_bytes=200, dst_bytes=100,
              src_pkts=2, dst_pkts=1)
    result = _reorient_flow(f, prefer_src="10.2.2.2")
    assert result.src_ip   == "10.2.2.2"
    assert result.dst_ip   == "10.1.1.1"
    assert result.src_port == 80
    assert result.dst_port == 1000
    assert result.src_bytes == 100    # was dst_bytes
    assert result.dst_bytes == 200    # was src_bytes
    assert result.src_pkts  == 1
    assert result.dst_pkts  == 2


def test_reorient_preserves_flag_counts():
    f = _flow(syn_count=3, ack_count=7, rst_count=1, fin_count=2, psh_count=4, urg_count=0)
    result = _reorient_flow(f, prefer_src=f.dst_ip)
    assert result.syn_count == 3
    assert result.ack_count == 7
    assert result.rst_count == 1
    assert result.fin_count == 2
    assert result.psh_count == 4
    assert result.urg_count == 0


def test_reorient_noop_when_prefer_src_matches_neither():
    f = _flow(src_ip="10.1.1.1", dst_ip="10.2.2.2")
    result = _reorient_flow(f, prefer_src="99.99.99.99")
    assert result is f


def test_reorient_preserves_ts_and_proto_and_duration():
    f = _flow(proto=17, duration=2.5, ts=_EPOCH + 10.0)
    result = _reorient_flow(f, prefer_src=f.dst_ip)
    assert result.ts       == f.ts
    assert result.proto    == 17
    assert result.duration == 2.5


# ══════════════════════════════════════════════════════════════════════════════
# _apply_labels_nf — UNSW path
# ══════════════════════════════════════════════════════════════════════════════

def test_nf_no_match_flow_retained_as_benign():
    f = _flow()
    rows, stats = _apply_labels_nf([f], label_index={}, matched_attack_row_indices=set())
    assert len(rows) == 1
    assert rows[0]["label"] == 0
    assert rows[0]["attack_cat"] == ""
    assert stats.n_benign == 1
    assert stats.n_attack == 0
    assert stats.n_dropped_unprocessable == 0


def test_nf_empty_flow_list():
    rows, stats = _apply_labels_nf([], {}, set())
    assert rows == []
    assert stats.n_attack == 0
    assert stats.n_benign == 0
    assert stats.n_dropped_unprocessable == 0


def test_nf_attack_match_produces_label_1_and_reorients():
    f = _flow(src_ip="10.1.1.1", src_port=1000,
              dst_ip="10.2.2.2", dst_port=80, proto=6)
    row = _unsw_label_row(f, row_idx=7, attack_cat="Exploit")
    # attacker in the label row is f.dst_ip ("10.2.2.2")
    key = orientation_key(f.src_ip, f.src_port, f.dst_ip, f.dst_port)
    matched = set()
    rows, stats = _apply_labels_nf([f], {key: [row]}, matched)

    assert len(rows) == 1
    assert rows[0]["label"] == 1
    assert rows[0]["attack_cat"] == "Exploit"
    assert stats.n_attack == 1
    assert stats.n_benign == 0
    assert 7 in matched
    # attacker (10.2.2.2) should now be src after re-orientation
    assert rows[0]["src_ip"] == "10.2.2.2"
    assert rows[0]["dst_ip"] == "10.1.1.1"


def test_nf_time_window_miss_becomes_benign():
    f = _flow()
    key = orientation_key(f.src_ip, f.src_port, f.dst_ip, f.dst_port)
    # Window is far in the future — no match
    row = {
        "srcip": f.dst_ip, "dstip": f.src_ip,
        "sport": f.dst_port, "dsport": f.src_port,
        "proto": "tcp",
        "stime": f.ts + 1000.0, "ltime": f.ts + 1001.0,
        "label": 1, "attack_cat": "Probe", "_row_idx": 2,
    }
    rows, stats = _apply_labels_nf([f], {key: [row]}, set())
    assert rows[0]["label"] == 0
    assert stats.n_benign == 1
    assert stats.n_attack == 0


def test_nf_udp_flow_matched_on_correct_proto():
    f = _flow(proto=17, src_port=54321, dst_port=53)
    key = orientation_key(f.src_ip, f.src_port, f.dst_ip, f.dst_port)
    row = {
        "srcip": f.src_ip, "dstip": f.dst_ip,
        "sport": f.src_port, "dsport": f.dst_port,
        "proto": "udp",
        "stime": f.ts - 0.5, "ltime": f.ts + 0.5,
        "label": 1, "attack_cat": "DNS", "_row_idx": 0,
    }
    rows, stats = _apply_labels_nf([f], {key: [row]}, set())
    assert rows[0]["label"] == 1


def test_nf_proto_mismatch_becomes_benign():
    f = _flow(proto=6)   # TCP
    key = orientation_key(f.src_ip, f.src_port, f.dst_ip, f.dst_port)
    row = {
        "srcip": f.dst_ip, "dstip": f.src_ip,
        "sport": f.dst_port, "dsport": f.src_port,
        "proto": "udp",   # mismatch
        "stime": f.ts - 0.5, "ltime": f.ts + 0.5,
        "label": 1, "attack_cat": "X", "_row_idx": 0,
    }
    rows, stats = _apply_labels_nf([f], {key: [row]}, set())
    assert rows[0]["label"] == 0


def test_nf_multiple_flows_mixed():
    f1 = _flow(src_ip="10.1.1.1", src_port=1000, dst_ip="10.2.2.2", dst_port=80)
    f2 = _flow(src_ip="10.1.1.1", src_port=2000, dst_ip="8.8.8.8",  dst_port=53, proto=17)
    key1 = orientation_key(f1.src_ip, f1.src_port, f1.dst_ip, f1.dst_port)
    attack_row = _unsw_label_row(f1, row_idx=0, attack_cat="DoS")
    matched = set()
    rows, stats = _apply_labels_nf([f1, f2], {key1: [attack_row]}, matched)
    assert stats.n_attack == 1
    assert stats.n_benign == 1
    assert 0 in matched


def test_nf_syn_count_propagates_to_row():
    f = _flow(syn_count=9, proto=6)
    rows, stats = _apply_labels_nf([f], {}, set())
    assert rows[0]["syn_count"] == 9


def test_nf_flow_ts_is_absolute():
    abs_ts = 1_700_500_000.0
    f = _flow(ts=abs_ts)
    rows, stats = _apply_labels_nf([f], {}, set())
    assert rows[0]["ts"] == pytest.approx(abs_ts)


# ══════════════════════════════════════════════════════════════════════════════
# _apply_labels_gotham_nf
# ══════════════════════════════════════════════════════════════════════════════

def test_gotham_nf_benign_pcap():
    f = _flow()
    rows, stats = _apply_labels_gotham_nf([f], is_attack=False, attack_cat="", attacker_ips=[])
    assert rows[0]["label"] == 0
    assert rows[0]["attack_cat"] == ""
    assert stats.n_benign == 1
    assert stats.n_attack == 0


def test_gotham_nf_attack_pcap_attacker_as_src():
    attacker = "10.2.2.2"
    # In f, attacker is dst — expect _reorient_flow to swap it
    f = _flow(src_ip="10.1.1.1", dst_ip=attacker, src_bytes=200, dst_bytes=100)
    rows, stats = _apply_labels_gotham_nf([f], is_attack=True, attack_cat="Mirai",
                                          attacker_ips=[attacker])
    assert rows[0]["label"] == 1
    assert rows[0]["attack_cat"] == "Mirai"
    assert rows[0]["src_ip"] == attacker   # attacker pinned as src
    assert stats.n_attack == 1
    assert stats.n_benign == 0


def test_gotham_nf_attack_pcap_attacker_already_src():
    attacker = "10.1.1.1"
    f = _flow(src_ip=attacker, dst_ip="10.2.2.2")
    rows, stats = _apply_labels_gotham_nf([f], is_attack=True, attack_cat="DDoS",
                                          attacker_ips=[attacker])
    assert rows[0]["src_ip"] == attacker
    assert rows[0]["label"] == 1


def test_gotham_nf_attack_pcap_no_matching_attacker_ip():
    f = _flow(src_ip="10.1.1.1", dst_ip="10.2.2.2")
    # attacker_ips contains a third IP not in this flow → default orientation kept
    rows, stats = _apply_labels_gotham_nf([f], is_attack=True, attack_cat="Scan",
                                          attacker_ips=["192.168.0.1"])
    assert rows[0]["label"] == 1
    assert rows[0]["src_ip"] == "10.1.1.1"   # unchanged, default orientation


def test_gotham_nf_multiple_attacker_ips_second_matches():
    attacker2 = "10.2.2.2"
    f = _flow(src_ip="10.1.1.1", dst_ip=attacker2)
    rows, stats = _apply_labels_gotham_nf(
        [f], is_attack=True, attack_cat="Brute",
        attacker_ips=["192.0.0.1", attacker2],   # second IP matches dst
    )
    assert rows[0]["src_ip"] == attacker2


def test_gotham_nf_empty_flow_list():
    rows, stats = _apply_labels_gotham_nf([], is_attack=True, attack_cat="X", attacker_ips=[])
    assert rows == []
    assert stats.n_attack == 0


# ══════════════════════════════════════════════════════════════════════════════
# _apply_labels_cic_nf
# ══════════════════════════════════════════════════════════════════════════════

_CIC_WINDOW = {
    "attack_cat":  "FTP-BruteForce",
    "proto":       "TCP",
    "attacker_ip": "172.16.0.1",
    "victim_ip":   "192.168.10.50",
    "dst_port":    21,
    "stime":       _EPOCH + 100.0,
    "ltime":       _EPOCH + 200.0,
}


def _cic_flow(ts_offset=150.0, src_is_attacker=True) -> Flow:
    attacker = _CIC_WINDOW["attacker_ip"]
    victim   = _CIC_WINDOW["victim_ip"]
    if src_is_attacker:
        return _flow(src_ip=attacker, src_port=55000,
                     dst_ip=victim, dst_port=_CIC_WINDOW["dst_port"],
                     proto=6, ts=_EPOCH + ts_offset)
    else:
        return _flow(src_ip=victim, src_port=_CIC_WINDOW["dst_port"],
                     dst_ip=attacker, dst_port=55000,
                     proto=6, ts=_EPOCH + ts_offset)


def test_cic_nf_in_window_match():
    f = _cic_flow(ts_offset=150.0)
    cats: set[str] = set()
    rows, stats = _apply_labels_cic_nf([f], [_CIC_WINDOW], cats)
    assert rows[0]["label"] == 1
    assert rows[0]["attack_cat"] == "FTP-BruteForce"
    assert "FTP-BruteForce" in cats
    assert stats.n_attack == 1


def test_cic_nf_in_window_match_reversed_canonical_order():
    # Canonical src = victim (lower IP), attacker = dst — match must still work
    f = _cic_flow(ts_offset=150.0, src_is_attacker=False)
    cats: set[str] = set()
    rows, stats = _apply_labels_cic_nf([f], [_CIC_WINDOW], cats)
    assert rows[0]["label"] == 1
    assert rows[0]["src_ip"] == _CIC_WINDOW["attacker_ip"]   # re-oriented


def test_cic_nf_outside_window_becomes_benign():
    f = _cic_flow(ts_offset=300.0)   # after window end (200.0)
    cats: set[str] = set()
    rows, stats = _apply_labels_cic_nf([f], [_CIC_WINDOW], cats)
    assert rows[0]["label"] == 0
    assert stats.n_benign == 1
    assert stats.n_attack == 0


def test_cic_nf_wrong_dst_port_no_match():
    f = _flow(src_ip=_CIC_WINDOW["attacker_ip"], dst_ip=_CIC_WINDOW["victim_ip"],
              dst_port=22,   # SSH, not FTP
              proto=6, ts=_EPOCH + 150.0)
    cats: set[str] = set()
    rows, stats = _apply_labels_cic_nf([f], [_CIC_WINDOW], cats)
    assert rows[0]["label"] == 0


def test_cic_nf_within_tolerance():
    # ts exactly at window boundary ± TIME_TOLERANCE
    f = _cic_flow(ts_offset=200.0 + TIME_TOLERANCE - 0.001)
    cats: set[str] = set()
    rows, stats = _apply_labels_cic_nf([f], [_CIC_WINDOW], cats)
    assert rows[0]["label"] == 1


def test_cic_nf_empty_window_list():
    f = _cic_flow()
    cats: set[str] = set()
    rows, stats = _apply_labels_cic_nf([f], [], cats)
    assert rows[0]["label"] == 0
    assert stats.n_benign == 1
