"""Tests for schema.py: flow_to_row key order and validate_matrix gating."""
import pytest

from adns_flows.schema import (
    FEATURE_COLUMNS,
    IDENTITY_COLUMNS,
    Flow,
    SchemaError,
    flow_to_row,
    validate_matrix,
)


def _flow(**kw) -> Flow:
    defaults = dict(
        ts=0.0, src_ip="1.2.3.4", dst_ip="5.6.7.8",
        src_port=12345, dst_port=80,
        proto=6, duration=1.0,
        src_bytes=100, dst_bytes=200,
        src_pkts=3, dst_pkts=5,
    )
    defaults.update(kw)
    return Flow(**defaults)


# ── flow_to_row ────────────────────────────────────────────────────────────

def test_flow_to_row_feature_keys_exact_order():
    row = flow_to_row(_flow())
    feature_keys = [k for k in row if k not in set(IDENTITY_COLUMNS)]
    assert feature_keys == list(FEATURE_COLUMNS)


def test_flow_to_row_no_extra_or_missing_keys():
    row = flow_to_row(_flow())
    assert set(row) == set(IDENTITY_COLUMNS) | set(FEATURE_COLUMNS)


def test_flow_to_row_derived_fields():
    f = _flow(src_bytes=100, dst_bytes=200, src_pkts=4, dst_pkts=8, duration=2.0)
    row = flow_to_row(f)
    assert row["total_bytes"] == 300
    assert row["total_pkts"] == 12
    assert row["bytes_ratio"] == pytest.approx(200 / 101)
    assert row["pkts_ratio"] == pytest.approx(8 / 5)
    assert row["src_mean_pkt_size"] == pytest.approx(100 / 5)
    assert row["dst_mean_pkt_size"] == pytest.approx(200 / 9)
    assert row["bytes_per_sec"] == pytest.approx(300 / 2.0)
    assert row["pkts_per_sec"] == pytest.approx(12 / 2.0)


def test_flow_to_row_dst_port_bucket_well_known():
    assert flow_to_row(_flow(dst_port=80))["dst_port_bucket"] == 0


def test_flow_to_row_dst_port_bucket_registered():
    assert flow_to_row(_flow(dst_port=8080))["dst_port_bucket"] == 1


def test_flow_to_row_dst_port_bucket_ephemeral():
    assert flow_to_row(_flow(dst_port=55000))["dst_port_bucket"] == 2


def test_flow_to_row_duration_floor():
    # duration=0 should not divide by zero; bytes_per_sec uses max(dur, 1e-3)
    f = _flow(src_bytes=1000, dst_bytes=0, src_pkts=1, dst_pkts=0, duration=0.0)
    row = flow_to_row(f)
    assert row["bytes_per_sec"] == pytest.approx(1000 / 1e-3)


# ── validate_matrix ────────────────────────────────────────────────────────

def test_validate_matrix_ok():
    validate_matrix(None, list(FEATURE_COLUMNS))  # must not raise


def test_validate_matrix_raises_on_dropped_column():
    cols = list(FEATURE_COLUMNS[:-1])
    with pytest.raises(SchemaError, match="count"):
        validate_matrix(None, cols)


def test_validate_matrix_raises_on_extra_column():
    cols = list(FEATURE_COLUMNS) + ["extra_col"]
    with pytest.raises(SchemaError, match="count"):
        validate_matrix(None, cols)


def test_validate_matrix_raises_on_reordered():
    cols = list(FEATURE_COLUMNS)
    cols[0], cols[1] = cols[1], cols[0]
    with pytest.raises(SchemaError, match="mismatch"):
        validate_matrix(None, cols)


def test_validate_matrix_raises_on_renamed_column():
    cols = list(FEATURE_COLUMNS)
    cols[2] = "src_bytes_WRONG"
    with pytest.raises(SchemaError, match="mismatch"):
        validate_matrix(None, cols)
