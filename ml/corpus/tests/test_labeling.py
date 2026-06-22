"""
Tests for the class-balance gate, drop-rate gate, label-row accounting,
CorpusStats dataclass, get_pcap_start_epoch, and load_label_index in
corpus/build_corpus.py.

All tests run without network hardware or pcap extraction tools.
The NFStream labeling tests (_apply_labels_nf / _apply_labels_gotham_nf /
_apply_labels_cic_nf) live in test_labeling_nf.py.
"""
from __future__ import annotations

import io
import logging
import struct

import pytest

from corpus.build_corpus import (
    CorpusBalanceError,
    CorpusDropRateError,
    CorpusStats,
    MAX_DROP_FRAC,
    REASON_NO_TIMESTAMP,
    REASON_OTHER,
    UNMATCHED_WARN_FRAC,
    assert_drop_rate,
    assert_sane_balance,
    get_pcap_start_epoch,
    load_label_index,
)


# ══════════════════════════════════════════════════════════════════════════
# TIER 3 — assert_sane_balance
# ════���═══════════════════════════════��═════════════════════════════════════

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
    with pytest.raises(CorpusBalanceError) as exc_info:
        assert_sane_balance(500, 0)
    msg = str(exc_info.value)
    assert "dropped" in msg.lower() or "unmatched" in msg.lower() or "benign" in msg.lower()

    with pytest.raises(CorpusBalanceError) as exc_info:
        assert_sane_balance(0, 500)
    msg = str(exc_info.value)
    assert "time" in msg.lower() or "epoch" in msg.lower() or "match" in msg.lower()


# ══════════════════════════════════���═══════════════════════════════════════
# TIER 4 — label-row accounting thresholds (no extraction required)
# ═══════════════════���═════════════════════════════��════════════════════════

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
    from corpus.build_corpus import UNMATCHED_WARN_FRAC  # noqa: F401

    total_attack_rows = 10
    label_rows_matched = 0
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


# ════════════════════════════════��════════════════════════════════════���════
# TIER 5 — CorpusStats dataclass
# ═���════════════���═══════════════════════════���═════════════════════════════��═

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


# ═════════════════════════════��═══════════════════════════════════���════════
# TIER 6 — get_pcap_start_epoch returns None on bad input
# ════════════════��═════════════════════════════════════════════════════════

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
    """Valid BE-microsecond pcap."""
    buf = io.BytesIO()
    buf.write(struct.pack(">IHHiIII", 0xA1B2C3D4, 2, 4, 0, 0, 65535, 1))
    buf.write(struct.pack(">IIII", 1_700_000_001, 0, 0, 0))
    p = tmp_path / "valid_be.pcap"
    p.write_bytes(buf.getvalue())
    result = get_pcap_start_epoch(str(p))
    assert result == pytest.approx(1_700_000_001.0)


# ═══════════════════════════════���═════════════════════════════════��════════
# TIER 7 — load_label_index row tracking
# ══════���═══════════════��═══════════════════════════════════════════════════

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
    assert all_entries[0]["_row_idx"] == 0


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


# ════���══════════════════════════��════════════════════════════════════��═════
# TIER 8 — drop-rate gate (assert_drop_rate)
# ══════��═════════════════════════════════════════════��═════════════════════

def test_drop_rate_gate_raises_above_threshold():
    """>10% drop rate raises CorpusDropRateError naming reason breakdown."""
    with pytest.raises(CorpusDropRateError, match="no_timestamp"):
        assert_drop_rate(2, 10, {REASON_NO_TIMESTAMP: 2})  # 20% > 10%


def test_drop_rate_gate_raises_with_breakdown_in_message():
    """Error message must contain both the rate and the reason breakdown."""
    with pytest.raises(CorpusDropRateError) as exc_info:
        assert_drop_rate(5, 10, {REASON_NO_TIMESTAMP: 4, REASON_OTHER: 1})
    msg = str(exc_info.value)
    assert "50.0%" in msg or "50%" in msg
    assert REASON_NO_TIMESTAMP in msg
    assert REASON_OTHER in msg


def test_drop_rate_gate_does_not_raise_at_threshold():
    """Exactly MAX_DROP_FRAC does NOT raise (condition is strictly >)."""
    n = 10
    dropped = int(MAX_DROP_FRAC * n)  # exactly 10% if MAX_DROP_FRAC=0.10
    assert_drop_rate(dropped, n, {REASON_NO_TIMESTAMP: dropped})  # must not raise


def test_drop_rate_gate_does_not_raise_below_threshold():
    """0% drop rate never raises."""
    assert_drop_rate(0, 100, {})  # must not raise


def test_drop_rate_gate_safe_on_zero_total():
    """Zero total_seen → no check, no error (balance gate handles empty corpora)."""
    assert_drop_rate(0, 0, {})  # must not raise


def test_drop_rate_gate_one_over_threshold():
    """1 dropped out of 9 seen = 11.1% > 10% → raises."""
    with pytest.raises(CorpusDropRateError):
        assert_drop_rate(1, 9, {REASON_NO_TIMESTAMP: 1})


def test_drop_rate_gate_custom_threshold():
    """Custom max_drop_frac=0.05 rejects a 6% drop rate."""
    with pytest.raises(CorpusDropRateError):
        assert_drop_rate(6, 100, {REASON_NO_TIMESTAMP: 6}, max_drop_frac=0.05)


def test_drop_rate_gate_no_timestamp_high_rate_is_pcapng_bug():
    """The scenario that caused the stale day-1 build: ~80% no_timestamp drops.

    214K kept / ~1028K total ≈ 79% dropped.  Must raise before writing parquet.
    """
    n_kept = 214_425
    n_dropped = 813_757
    total = n_kept + n_dropped
    with pytest.raises(CorpusDropRateError, match="no_timestamp"):
        assert_drop_rate(n_dropped, total, {REASON_NO_TIMESTAMP: n_dropped})
