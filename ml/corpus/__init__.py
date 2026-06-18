"""
corpus — PCAP-to-labeled-corpus pipeline for ADNS.

Builds labeled training data from raw network captures, using adns_flows for
extraction — the same code path as live scoring.

Supports:
  UNSW-NB15  — time-window label matching against a ground-truth CSV
  Gotham 2025 — directory-based labeling (no GT CSV; see gotham_labels.py)
"""
from .build_corpus import (
    CorpusBalanceError,
    CorpusStats,
    OUTPUT_COLUMNS,
    REASON_EXTRACTION_FAIL,
    REASON_NO_TIMESTAMP,
    REASON_OTHER,
    TIME_TOLERANCE,
    UNMATCHED_WARN_FRAC,
    _apply_labels,
    _apply_labels_gotham,
    assert_sane_balance,
    build_corpus,
    build_corpus_gotham,
    extract_pcap_flows,
    get_pcap_start_epoch,
    load_label_index,
)

__all__ = [
    "CorpusBalanceError",
    "CorpusStats",
    "OUTPUT_COLUMNS",
    "REASON_EXTRACTION_FAIL",
    "REASON_NO_TIMESTAMP",
    "REASON_OTHER",
    "TIME_TOLERANCE",
    "UNMATCHED_WARN_FRAC",
    "_apply_labels",
    "_apply_labels_gotham",
    "assert_sane_balance",
    "build_corpus",
    "build_corpus_gotham",
    "extract_pcap_flows",
    "get_pcap_start_epoch",
    "load_label_index",
]
