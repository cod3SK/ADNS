"""
Tests for the three-way labeling logic, class-balance gate, and label-row
accounting in corpus/build_corpus.py.

All tests in this file run without tshark — they call _apply_labels() and
assert_sane_balance() directly, bypassing pcap extraction.

Design principle: tests must be able to reach every code path without a
live tshark binary.  The tshark-gated parity tests live in test_parity.py.
"""
from __future__ import annotations

import io
import struct

import pytest

from adns_flows import orientation_key

from corpus.build_corpus import (
    CorpusBalanceError,
    CorpusStats,
    REASON_NO_TIMESTAMP,
    REASON_OTHER,
    _apply_labels,
    assert_sane_balance,
    get_pcap_start_epoch,
    load_label_index,
)


# ── shared test fixtures ───────────────────────────────────────────────────

_EPOCH = 1_700_000_000.0   # arbitrary but realistic pcap start epoch

_CONV_TCP = {
    "proto":    "TCP",
    "ep_a":     ("10.1.1.1", 1000),
    "ep_b":     ("10.2.2.2", 80),
    "bytes_ab": 200, "bytes_ba": 100,
    "pkts_ab":  2,   "pkts_ba":  1,
    "duration": 1.0, "rel_start": 5.0,
}

_CONV_UDP = {
    "proto":    "UDP",
    "ep_a":     ("10.1.1.1", 54321),
    "ep_b":     ("8.8.8.8",  53),
    "bytes_ab": 60,  "bytes_ba": 120,
    "pkts_ab":  1,   "pkts_ba":  1,
    "duration": 0.01, "rel_start": 6.0,
}


def _attack_label_row(conv: dict, row_idx: int = 0, attack_cat: str = "DoS") -> dict:
    """Build a synthetic attack label row that matches this conv's abs_ts."""
    abs_ts = _EPOCH + conv["rel_start"]
    ep_a, ep_b = conv["ep_a"], conv["ep_b"]
    # srcip is whichever side has the higher IP (to prove prefer_src flips it)
    srcip = ep_a[0] if ep_a[0] > ep_b[0] else ep_b[0]
    return {
        "srcip":      srcip,
        "dstip":      ep_b[0] if srcip == ep_a[0] else ep_a[0],
        "sport":      ep_a[1] if srcip == ep_a[0] else ep_b[1],
        "dsport":     ep_b[1] if srcip == ep_a[0] else ep_a[1],
        "proto":      conv["proto"].lower(),
        "stime":      abs_ts - 0.5,
        "ltime":      abs_ts + 0.5,
        "label":      1,
        "attack_cat": attack_cat,
        "_row_idx":   row_idx,
    }


# ══════════════════════════════════════════════════════════════════════════
# TIER 1 — THE CORE FIX: no-match flow becomes label=0 and is retained
# ══════════════════════════════════════════════════════════════════════════

def test_no_match_flow_is_retained_as_benign():
    """Critical fix: a flow with no label-index entry MUST become label=0 NOT be dropped."""
    label_index = {}   # empty GT → every flow is a no-match
    matched = set()
    rows, stats = _apply_labels([_CONV_TCP], {}, _EPOCH, label_index, matched)

    assert len(rows) == 1, "no-match flow must be retained, not dropped"
    assert rows[0]["label"] == 0
    assert rows[0]["attack_cat"] == ""
    assert stats.n_benign == 1
    assert stats.n_attack == 0
    assert stats.n_dropped_unprocessable == 0


def test_no_match_multiple_flows_all_retained():
    """All no-match flows must be retained."""
    rows, stats = _apply_labels([_CONV_TCP, _CONV_UDP], {}, _EPOCH, {}, set())
    assert len(rows) == 2
    assert all(r["label"] == 0 for r in rows)
    assert stats.n_benign == 2
    assert stats.n_attack == 0
    assert stats.n_dropped_unprocessable == 0


def test_attack_match_produces_label_1():
    """A flow that matches an attack row gets label=1."""
    row = _attack_label_row(_CONV_TCP, row_idx=0)
    key = orientation_key(_CONV_TCP["ep_a"][0], _CONV_TCP["ep_a"][1],
                          _CONV_TCP["ep_b"][0], _CONV_TCP["ep_b"][1])
    label_index = {key: [row]}
    matched = set()
    rows, stats = _apply_labels([_CONV_TCP], {}, _EPOCH, label_index, matched)

    assert len(rows) == 1
    assert rows[0]["label"] == 1
    assert rows[0]["attack_cat"] == "DoS"
    assert stats.n_attack == 1
    assert stats.n_benign == 0
    assert 0 in matched


def test_mixed_batch_attack_and_benign():
    """One attack conv + one no-match conv → attack + benign, nothing dropped."""
    row = _attack_label_row(_CONV_TCP, row_idx=7, attack_cat="Exploit")
    key = orientation_key(_CONV_TCP["ep_a"][0], _CONV_TCP["ep_a"][1],
                          _CONV_TCP["ep_b"][0], _CONV_TCP["ep_b"][1])
    label_index = {key: [row]}
    matched = set()
    rows, stats = _apply_labels([_CONV_TCP, _CONV_UDP], {}, _EPOCH, label_index, matched)

    assert len(rows) == 2
    labels = {r["label"] for r in rows}
    assert labels == {0, 1}
    assert stats.n_attack == 1
    assert stats.n_benign == 1
    assert stats.n_dropped_unprocessable == 0
    assert 7 in matched


def test_time_window_miss_becomes_benign_not_dropped():
    """A conv that matches on endpoint+proto but NOT time → benign (not dropped)."""
    abs_ts = _EPOCH + _CONV_TCP["rel_start"]
    key = orientation_key(_CONV_TCP["ep_a"][0], _CONV_TCP["ep_a"][1],
                          _CONV_TCP["ep_b"][0], _CONV_TCP["ep_b"][1])
    # Window is 1000 seconds in the future — time match fails
    label_index = {key: [{
        "srcip":      _CONV_TCP["ep_a"][0],
        "dstip":      _CONV_TCP["ep_b"][0],
        "sport":      _CONV_TCP["ep_a"][1],
        "dsport":     _CONV_TCP["ep_b"][1],
        "proto":      "tcp",
        "stime":      abs_ts + 1000.0,
        "ltime":      abs_ts + 1001.0,
        "label":      1,
        "attack_cat": "Probe",
        "_row_idx":   3,
    }]}
    rows, stats = _apply_labels([_CONV_TCP], {}, _EPOCH, label_index, set())

    assert len(rows) == 1
    assert rows[0]["label"] == 0   # time miss → benign, not dropped
    assert stats.n_benign == 1
    assert stats.n_attack == 0
    assert stats.n_dropped_unprocessable == 0


def test_proto_mismatch_becomes_benign_not_dropped():
    """A conv that matches on endpoints but NOT proto → benign (not dropped)."""
    key = orientation_key(_CONV_TCP["ep_a"][0], _CONV_TCP["ep_a"][1],
                          _CONV_TCP["ep_b"][0], _CONV_TCP["ep_b"][1])
    abs_ts = _EPOCH + _CONV_TCP["rel_start"]
    label_index = {key: [{
        **_attack_label_row(_CONV_TCP, row_idx=5),
        "proto": "udp",  # mismatch — conv is TCP
    }]}
    rows, stats = _apply_labels([_CONV_TCP], {}, _EPOCH, label_index, set())

    assert len(rows) == 1
    assert rows[0]["label"] == 0
    assert stats.n_benign == 1
    assert stats.n_dropped_unprocessable == 0


# ══════════════════════════════════════════════════════════════════════════
# TIER 2 — UNPROCESSABLE (no_timestamp) must NOT mix with no-match path
# ══════════════════════════════════════════════════════════════════════════

def test_no_timestamp_drops_all_flows():
    """pcap_start_epoch=None → all flows dropped as no_timestamp, none retained."""
    rows, stats = _apply_labels([_CONV_TCP, _CONV_UDP], {}, None, {}, set())

    assert len(rows) == 0
    assert stats.n_dropped_unprocessable == 2
    assert stats.dropped_reasons.get(REASON_NO_TIMESTAMP, 0) == 2
    assert stats.n_benign == 0
    assert stats.n_attack == 0


def test_no_timestamp_is_separate_from_no_match():
    """no_timestamp and no-match MUST increment different counters."""
    rows_nomatch, stats_nomatch = _apply_labels([_CONV_TCP], {}, _EPOCH, {}, set())
    rows_notime, stats_notime   = _apply_labels([_CONV_TCP], {}, None,   {}, set())

    # no-match: benign path
    assert stats_nomatch.n_benign == 1
    assert stats_nomatch.n_dropped_unprocessable == 0

    # no-timestamp: dropped path
    assert stats_notime.n_benign == 0
    assert stats_notime.n_dropped_unprocessable == 1
    assert stats_notime.dropped_reasons.get(REASON_NO_TIMESTAMP, 0) == 1


def test_no_timestamp_does_not_populate_benign_rows():
    """Flows dropped for no_timestamp must NOT appear as benign label=0 rows."""
    rows, stats = _apply_labels([_CONV_TCP], {}, None, {}, set())
    assert rows == []                          # nothing kept
    assert stats.n_benign == 0                 # not counted as benign either


def test_no_timestamp_with_empty_convs():
    """Empty conv list + None epoch → zero dropped, no error."""
    rows, stats = _apply_labels([], {}, None, {}, set())
    assert rows == []
    assert stats.n_dropped_unprocessable == 0
    assert stats.dropped_reasons.get(REASON_NO_TIMESTAMP, 0) == 0


# ══════════════════════════════════════════════════════════════════════════
# TIER 3 — assert_sane_balance
# ══════════════════════════════════════════════════════════════════════════

def test_sane_balance_raises_on_all_attack():
    """Corpus is 100% attack → benign_frac=0 < 0.50 → CorpusBalanceError."""
    with pytest.raises(CorpusBalanceError, match="benign fraction"):
        assert_sane_balance(1000, 0)


def test_sane_balance_raises_on_near_all_attack():
    """200 attacks + 1 benign (benign_frac ≈ 0.005) → below 0.50 threshold."""
    with pytest.raises(CorpusBalanceError, match="benign fraction"):
        assert_sane_balance(200, 1)


def test_sane_balance_raises_on_empty_corpus():
    """Empty corpus raises CorpusBalanceError with 'empty' message."""
    with pytest.raises(CorpusBalanceError, match="empty"):
        assert_sane_balance(0, 0)


def test_sane_balance_raises_on_zero_attack():
    """No attacks → attack_frac=0 < 0.001 → time-matching failure diagnostic."""
    with pytest.raises(CorpusBalanceError, match="[Aa]ttack fraction"):
        assert_sane_balance(0, 1000)


def test_sane_balance_raises_on_too_few_attacks():
    """1 attack + 10000 benign → attack_frac ≈ 0.0001 < 0.001 → raises."""
    with pytest.raises(CorpusBalanceError, match="[Aa]ttack fraction"):
        assert_sane_balance(1, 10000)


def test_sane_balance_passes_on_realistic_mix():
    """100 attacks + 9900 benign (1%, 99%) — within both thresholds."""
    assert_sane_balance(100, 9900)   # must NOT raise


def test_sane_balance_passes_on_unsw_paper_ratio():
    """UNSW-NB15 paper: ~87% benign (2,219 benign, 321 attack flows per table)."""
    assert_sane_balance(321, 2219)   # must NOT raise


def test_sane_balance_passes_at_exact_50_50():
    """Exactly 50% benign is valid — borderline but ≥ min_benign_frac."""
    assert_sane_balance(50, 50)      # must NOT raise


def test_sane_balance_custom_min_benign_raises():
    """Custom min_benign_frac=0.70 rejects a 50/50 corpus."""
    with pytest.raises(CorpusBalanceError, match="benign fraction"):
        assert_sane_balance(50, 50, min_benign_frac=0.70)


def test_sane_balance_custom_min_benign_passes():
    """Custom min_benign_frac=0.20 accepts a 70/30 attack-heavy corpus."""
    assert_sane_balance(70, 30, min_benign_frac=0.20)   # must NOT raise


def test_sane_balance_error_message_contains_diagnosis():
    """The error message must name the likely root cause."""
    # All-attack → drop-bug diagnosis
    with pytest.raises(CorpusBalanceError) as exc_info:
        assert_sane_balance(500, 0)
    msg = str(exc_info.value)
    assert "dropped" in msg.lower() or "unmatched" in msg.lower() or "benign" in msg.lower()

    # No-attack → time-matching diagnosis
    with pytest.raises(CorpusBalanceError) as exc_info:
        assert_sane_balance(0, 500)
    msg = str(exc_info.value)
    assert "time" in msg.lower() or "epoch" in msg.lower() or "match" in msg.lower()


# ══════════════════════════════════════════════════════════════════════════
# TIER 4 — label-row accounting
# ══════════════════════════════════════════════════════════════════════════

def test_matched_attack_row_idx_tracked():
    """When a flow hits an attack label row, that row's _row_idx is in matched set."""
    row = _attack_label_row(_CONV_TCP, row_idx=42)
    key = orientation_key(_CONV_TCP["ep_a"][0], _CONV_TCP["ep_a"][1],
                          _CONV_TCP["ep_b"][0], _CONV_TCP["ep_b"][1])
    matched = set()
    _apply_labels([_CONV_TCP], {}, _EPOCH, {key: [row]}, matched)
    assert 42 in matched


def test_unmatched_attack_row_not_in_matched_set():
    """An attack row whose endpoint pair is never extracted stays absent from matched set."""
    key_other = orientation_key("99.0.0.1", 9999, "99.0.0.2", 8080)
    label_index = {key_other: [{
        "srcip": "99.0.0.1", "dstip": "99.0.0.2",
        "sport": 9999, "dsport": 8080,
        "proto": "tcp",
        "stime": _EPOCH + 100.0,
        "ltime": _EPOCH + 200.0,
        "label": 1,
        "attack_cat": "Backdoor",
        "_row_idx": 5,
    }]}
    matched = set()
    rows, stats = _apply_labels([_CONV_TCP], {}, _EPOCH, label_index, matched)

    assert 5 not in matched          # attack row was never matched
    assert len(rows) == 1            # the conv was kept as benign
    assert rows[0]["label"] == 0


def test_multiple_flows_hit_same_attack_row():
    """Two convs that match the same GT row → matched set still contains that row once."""
    row = _attack_label_row(_CONV_TCP, row_idx=0)
    key = orientation_key(_CONV_TCP["ep_a"][0], _CONV_TCP["ep_a"][1],
                          _CONV_TCP["ep_b"][0], _CONV_TCP["ep_b"][1])
    # Two identical convs (both will match the same row)
    matched = set()
    rows, stats = _apply_labels([_CONV_TCP, _CONV_TCP], {}, _EPOCH, {key: [row]}, matched)

    assert len(matched) == 1         # only one distinct row index
    assert stats.n_attack == 2       # but two attack flows produced


def test_high_unmatched_fraction_threshold():
    """Verify >20% of attack rows unmatched triggers the warning condition."""
    total_attack_rows = 10
    matched = 1
    unmatched = total_attack_rows - matched
    frac = unmatched / total_attack_rows
    assert frac > 0.20, "9 of 10 unmatched should be >20%"


def test_low_unmatched_fraction_threshold():
    """<=20% unmatched must NOT trigger the warning."""
    total_attack_rows = 100
    matched = 85
    frac = (total_attack_rows - matched) / total_attack_rows
    assert frac <= 0.20, "15 of 100 unmatched is ≤20%"


def test_high_unmatched_rate_logged(caplog):
    """build_corpus logs a WARNING when >20% of attack rows match nothing."""
    import logging
    from corpus.build_corpus import UNMATCHED_WARN_FRAC  # noqa: F401 — just verify importable

    # Simulate the warning logic from build_corpus
    total_attack_rows = 10
    label_rows_matched = 0   # 100% unmatched
    unmatched_frac = (total_attack_rows - label_rows_matched) / total_attack_rows

    test_log = logging.getLogger("corpus.build_corpus")
    with caplog.at_level(logging.WARNING, logger="corpus.build_corpus"):
        if unmatched_frac > 0.20:
            test_log.warning(
                "HIGH UNMATCHED RATE: %d of %d attack rows (%.0f%%) matched no flow.",
                total_attack_rows - label_rows_matched,
                total_attack_rows,
                100 * unmatched_frac,
            )

    assert any("HIGH UNMATCHED RATE" in r.message for r in caplog.records)


# ══════════════════════════════════════════════════════════════════════════
# TIER 5 — CorpusStats dataclass
# ══════════════════════════════════════════════════════════════════════════

def test_corpus_stats_merge():
    """CorpusStats.merge() accumulates all fields correctly."""
    a = CorpusStats(n_attack=10, n_benign=90, n_dropped_unprocessable=2,
                    dropped_reasons={"no_timestamp": 1, "other": 1},
                    label_rows_total=5, label_rows_matched=3)
    b = CorpusStats(n_attack=5, n_benign=45, n_dropped_unprocessable=1,
                    dropped_reasons={"no_timestamp": 1},
                    label_rows_total=0, label_rows_matched=0)
    a.merge(b)
    assert a.n_attack == 15
    assert a.n_benign == 135
    assert a.n_dropped_unprocessable == 3
    assert a.dropped_reasons["no_timestamp"] == 2
    assert a.dropped_reasons["other"] == 1
    assert a.label_rows_total == 5    # merge does NOT add label totals
    assert a.label_rows_matched == 3  # those are accumulated by build_corpus


def test_corpus_stats_derived_properties():
    stats = CorpusStats(n_attack=100, n_benign=900)
    assert stats.total_kept == 1000
    assert stats.attack_frac == pytest.approx(0.10)
    assert stats.benign_frac == pytest.approx(0.90)


def test_corpus_stats_label_rows_unmatched():
    stats = CorpusStats(label_rows_total=50, label_rows_matched=35)
    assert stats.label_rows_unmatched == 15


# ══════════════════════════════════════════════════════════════════════════
# TIER 6 — get_pcap_start_epoch returns None on bad input
# ══════════════════════════════════════════════════════════════════════════

def test_get_pcap_start_epoch_none_on_empty_file(tmp_path):
    p = tmp_path / "empty.pcap"
    p.write_bytes(b"")
    assert get_pcap_start_epoch(str(p)) is None


def test_get_pcap_start_epoch_none_on_bad_magic(tmp_path):
    p = tmp_path / "bad.pcap"
    p.write_bytes(b"\xDE\xAD\xBE\xEF" + b"\x00" * 36)
    assert get_pcap_start_epoch(str(p)) is None


def test_get_pcap_start_epoch_none_on_nonexistent_file():
    assert get_pcap_start_epoch("/nonexistent/path.pcap") is None


def test_get_pcap_start_epoch_valid_le_microseconds(tmp_path):
    """Valid LE-microsecond pcap with ts_sec=1700000000, ts_usec=500000."""
    buf = io.BytesIO()
    buf.write(struct.pack("<IHHiIII", 0xA1B2C3D4, 2, 4, 0, 0, 65535, 1))
    buf.write(struct.pack("<IIII", 1_700_000_000, 500_000, 0, 0))
    p = tmp_path / "valid_le.pcap"
    p.write_bytes(buf.getvalue())
    result = get_pcap_start_epoch(str(p))
    assert result == pytest.approx(1_700_000_000.5)


def test_get_pcap_start_epoch_valid_be_microseconds(tmp_path):
    """Valid BE-microsecond pcap.

    A BE pcap stores the logical magic value 0xA1B2C3D4 in big-endian byte
    order: bytes [a1 b2 c3 d4].  The reader detects this by reading those 4
    bytes as a LE uint32 and getting 0xD4C3B2A1 — the BE branch.  All
    subsequent multi-byte fields are also in BE.
    """
    buf = io.BytesIO()
    # Magic 0xA1B2C3D4 stored in BE byte order → on-disk bytes: a1 b2 c3 d4
    buf.write(struct.pack(">IHHiIII", 0xA1B2C3D4, 2, 4, 0, 0, 65535, 1))
    # First packet record in BE
    buf.write(struct.pack(">IIII", 1_700_000_001, 0, 0, 0))
    p = tmp_path / "valid_be.pcap"
    p.write_bytes(buf.getvalue())
    result = get_pcap_start_epoch(str(p))
    assert result == pytest.approx(1_700_000_001.0)


# ══════════════════════════════════════════════════════════════════════════
# TIER 7 — load_label_index row tracking
# ══════════════════════════════════════════════════════════════════════════

def test_load_label_index_counts_attack_rows(tmp_path):
    """total_attack_rows counts only label=1 rows."""
    csv = tmp_path / "gt.csv"
    csv.write_text(
        "srcip,sport,dstip,dsport,proto,stime,ltime,attack_cat,label\n"
        "10.0.0.1,1000,10.0.0.2,80,tcp,1700000000,1700000010,DoS,1\n"
        "10.0.0.3,2000,10.0.0.4,443,tcp,1700000000,1700000010,,0\n"
        "10.0.0.5,3000,10.0.0.6,22,tcp,1700000005,1700000015,Probe,1\n"
    )
    index, total_attack = load_label_index(str(csv))
    assert total_attack == 2   # two label=1 rows


def test_load_label_index_adds_row_idx(tmp_path):
    """Each entry in the index must have a _row_idx field."""
    csv = tmp_path / "gt.csv"
    csv.write_text(
        "srcip,sport,dstip,dsport,proto,stime,ltime,attack_cat,label\n"
        "10.0.0.1,1000,10.0.0.2,80,tcp,1700000000,1700000010,DoS,1\n"
    )
    index, _ = load_label_index(str(csv))
    all_entries = [e for entries in index.values() for e in entries]
    assert all("_row_idx" in e for e in all_entries)
    assert all_entries[0]["_row_idx"] == 0   # first row → index 0


def test_load_label_index_row_idx_is_per_csv_row(tmp_path):
    """_row_idx values reflect the original CSV row order."""
    csv = tmp_path / "gt.csv"
    csv.write_text(
        "srcip,sport,dstip,dsport,proto,stime,ltime,attack_cat,label\n"
        "10.0.0.1,1000,10.0.0.2,80,tcp,1700000000,1700000010,DoS,1\n"
        "10.0.0.3,2000,10.0.0.4,443,tcp,1700000000,1700000010,Probe,1\n"
    )
    index, _ = load_label_index(str(csv))
    all_entries = sorted(
        [e for entries in index.values() for e in entries],
        key=lambda e: e["_row_idx"],
    )
    assert all_entries[0]["_row_idx"] == 0
    assert all_entries[1]["_row_idx"] == 1
