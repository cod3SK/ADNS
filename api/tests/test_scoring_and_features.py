"""Unit tests for the heuristic scorer and the meta-model feature builder."""

from datetime import datetime, timezone
from types import SimpleNamespace

from model_runner import MetaFeatureBuilder
from scoring import FlowScorer, is_private_ip


def _flow(**kwargs):
    base = dict(
        timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
        src_ip="192.168.1.5",
        dst_ip="8.8.8.8",
        proto="TCP",
        bytes=1000,
        extra={},
    )
    base.update(kwargs)
    return SimpleNamespace(**base)


# --- FlowScorer ----------------------------------------------------------

def test_is_private_ip():
    assert is_private_ip("10.0.0.1") is True
    assert is_private_ip("192.168.1.1") is True
    assert is_private_ip("8.8.8.8") is False
    assert is_private_ip("not-an-ip") is False


def test_bytes_score_is_monotonic_and_bounded():
    scorer = FlowScorer()
    assert scorer._bytes_score(0) == 0.0
    low = scorer._bytes_score(1000)
    high = scorer._bytes_score(1_000_000)
    assert 0.0 < low < high <= 1.0


def test_direction_bonus_prefers_outbound():
    scorer = FlowScorer()
    outbound = scorer._direction_bonus(_flow(src_ip="10.0.0.1", dst_ip="8.8.8.8"))
    inbound = scorer._direction_bonus(_flow(src_ip="8.8.8.8", dst_ip="10.0.0.1"))
    assert outbound > inbound >= 0.0


def test_proto_bonus_known_and_unknown():
    scorer = FlowScorer()
    assert scorer._proto_bonus("ICMP") > 0.0
    assert scorer._proto_bonus("TCP") == 0.0


def test_stable_jitter_is_deterministic_and_small():
    scorer = FlowScorer()
    flow = _flow()
    j1 = scorer._stable_jitter(flow)
    j2 = scorer._stable_jitter(flow)
    assert j1 == j2
    assert 0.0 <= j1 <= 0.05


# --- MetaFeatureBuilder --------------------------------------------------

def test_feature_builder_row_shape_matches_columns():
    builder = MetaFeatureBuilder()
    frame = builder.build(_flow())
    assert list(frame.columns) == list(MetaFeatureBuilder.FEATURE_COLUMNS)
    assert frame.shape == (1, len(MetaFeatureBuilder.FEATURE_COLUMNS))


def test_feature_builder_encodes_proto_code():
    builder = MetaFeatureBuilder()
    row = builder.build(_flow(proto="TCP")).iloc[0]
    assert row["proto"] == 6.0
    row_udp = builder.build(_flow(proto="UDP")).iloc[0]
    assert row_udp["proto"] == 17.0


def test_feature_builder_outbound_directional_bytes():
    builder = MetaFeatureBuilder()
    # private -> public with only a total byte count splits all bytes to src.
    row = builder.build(_flow(src_ip="10.0.0.1", dst_ip="8.8.8.8", bytes=5000, extra={})).iloc[0]
    assert row["src_bytes"] == 5000.0
    assert row["dst_bytes"] == 0.0


def test_feature_builder_batch_handles_empty():
    builder = MetaFeatureBuilder()
    frame = builder.build_batch([])
    assert list(frame.columns) == list(MetaFeatureBuilder.FEATURE_COLUMNS)
    assert frame.shape[0] == 0
