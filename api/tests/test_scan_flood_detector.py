"""Tests for the stateful scan-flood detector.

Uses a simple MockFlow dataclass to avoid importing SQLAlchemy.
All tests inject _now timestamps so they are deterministic and fast.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from scan_flood_detector import ScanFloodDetector


@dataclass
class MockFlow:
    src_ip: str = "192.168.1.1"
    timestamp: object = None
    extra: dict = field(default_factory=dict)
    bytes: int = 0


def _tcp_flow(src_ip: str, dst_port: int, src_pkts: int = 1, dst_pkts: int = 0) -> MockFlow:
    return MockFlow(
        src_ip=src_ip,
        extra={"proto": 6, "dst_port": dst_port, "src_pkts": src_pkts, "dst_pkts": dst_pkts},
    )


def _udp_flow(src_ip: str, dst_port: int) -> MockFlow:
    return MockFlow(
        src_ip=src_ip,
        extra={"proto": 17, "dst_port": dst_port, "src_pkts": 1, "dst_pkts": 1},
    )


T0 = 1_000_000.0   # arbitrary fixed epoch second for tests


class TestPortScan:
    @pytest.fixture(autouse=True)
    def fresh_detector(self):
        self.d = ScanFloodDetector()

    def test_single_flow_returns_none(self):
        f = _tcp_flow("10.0.0.1", dst_port=22)
        assert self.d.record_and_classify(f, _now=T0) is None

    def test_below_threshold_returns_none(self):
        # 14 unique ports — one below threshold
        for port in range(1, 15):
            f = _tcp_flow("10.0.0.1", dst_port=port)
            result = self.d.record_and_classify(f, _now=T0)
        assert result is None

    def test_at_threshold_returns_scanning(self):
        # 15th unique port crosses the threshold
        for port in range(1, 16):
            f = _tcp_flow("10.0.0.1", dst_port=port)
            result = self.d.record_and_classify(f, _now=T0)
        assert result == "scanning"

    def test_repeated_same_port_does_not_trigger(self):
        # 100 non-degenerate flows to the same port — unique port count stays at 1
        for _ in range(100):
            f = _tcp_flow("10.0.0.1", dst_port=443, src_pkts=5, dst_pkts=3)
            result = self.d.record_and_classify(f, _now=T0)
        assert result is None

    def test_different_ips_do_not_cross_count(self):
        # IP-A hits 15 unique ports → "scanning"; IP-B hits only 5 → None.
        # Proves IP-A's hits don't inflate IP-B's counter.
        for port in range(1, 16):
            self.d.record_and_classify(_udp_flow("10.0.0.1", port), _now=T0)
        # IP-B hits a completely disjoint set of 5 ports
        result_b = None
        for port in range(200, 205):
            result_b = self.d.record_and_classify(_udp_flow("10.0.0.2", port), _now=T0)
        assert result_b is None

    def test_ports_expire_after_window(self):
        # Add 15 unique ports at T0
        for port in range(1, 16):
            self.d.record_and_classify(_tcp_flow("10.0.0.1", port), _now=T0)
        # Now check one more port 61 seconds later — all old entries should have expired
        f = _tcp_flow("10.0.0.1", dst_port=99)
        result = self.d.record_and_classify(f, _now=T0 + 61.0)
        assert result is None

    def test_udp_scan_also_detected(self):
        # Protocol doesn't matter for port-scan — all flows contribute
        for port in range(1, 16):
            result = self.d.record_and_classify(_udp_flow("10.0.0.1", port), _now=T0)
        assert result == "scanning"


class TestSynFlood:
    @pytest.fixture(autouse=True)
    def fresh_detector(self):
        self.d = ScanFloodDetector()

    def _degenerate(self, src_ip: str = "10.0.0.5") -> MockFlow:
        return _tcp_flow(src_ip, dst_port=80, src_pkts=1, dst_pkts=0)

    def test_below_threshold_returns_none(self):
        for _ in range(49):
            result = self.d.record_and_classify(self._degenerate(), _now=T0)
        assert result is None

    def test_at_threshold_returns_syn_flood(self):
        for _ in range(50):
            result = self.d.record_and_classify(self._degenerate(), _now=T0)
        assert result == "syn_flood"

    def test_normal_tcp_does_not_trigger(self):
        # Flows with dst_pkts > 0 are not degenerate
        for _ in range(50):
            f = _tcp_flow("10.0.0.5", dst_port=80, src_pkts=1, dst_pkts=1)
            result = self.d.record_and_classify(f, _now=T0)
        assert result is None

    def test_high_src_pkts_does_not_trigger(self):
        # src_pkts > 2 means established flow — not degenerate
        for _ in range(50):
            f = _tcp_flow("10.0.0.5", dst_port=80, src_pkts=10, dst_pkts=0)
            result = self.d.record_and_classify(f, _now=T0)
        assert result is None

    def test_flood_expires_after_window(self):
        # 50 flows at T0
        for _ in range(50):
            self.d.record_and_classify(self._degenerate(), _now=T0)
        # 31 seconds later — all entries expired; one new flow should not trigger
        result = self.d.record_and_classify(self._degenerate(), _now=T0 + 31.0)
        assert result is None

    def test_different_ips_do_not_cross_count(self):
        # 49 flows from IP-A and 49 from IP-B — neither triggers
        for _ in range(49):
            self.d.record_and_classify(_tcp_flow("10.0.0.5", 80, 1, 0), _now=T0)
            self.d.record_and_classify(_tcp_flow("10.0.0.6", 80, 1, 0), _now=T0)
        result_a = self.d.record_and_classify(_tcp_flow("10.0.0.5", 80, 1, 0), _now=T0)
        result_b = self.d.record_and_classify(_tcp_flow("10.0.0.6", 80, 1, 0), _now=T0)
        # Both hit 50 — both should trigger
        assert result_a == "syn_flood"
        assert result_b == "syn_flood"


class TestReset:
    def test_reset_clears_scan_state(self):
        d = ScanFloodDetector()
        for port in range(1, 16):
            d.record_and_classify(_tcp_flow("10.0.0.1", port), _now=T0)
        d.reset()
        # After reset, 15 more unique ports should not immediately trigger
        # (they re-enter the window from a clean slate)
        results = []
        for port in range(1, 15):
            results.append(d.record_and_classify(_tcp_flow("10.0.0.1", port), _now=T0 + 1))
        assert all(r is None for r in results)

    def test_reset_clears_flood_state(self):
        d = ScanFloodDetector()
        for _ in range(50):
            d.record_and_classify(_tcp_flow("10.0.0.5", 80, 1, 0), _now=T0)
        d.reset()
        # After reset, 49 more flows should not trigger
        for _ in range(49):
            result = d.record_and_classify(_tcp_flow("10.0.0.5", 80, 1, 0), _now=T0 + 1)
        assert result is None


class TestEdgeCases:
    def test_missing_src_ip_returns_none(self):
        d = ScanFloodDetector()
        f = MockFlow(src_ip="", extra={"proto": 6, "dst_port": 80, "src_pkts": 1, "dst_pkts": 0})
        assert d.record_and_classify(f, _now=T0) is None

    def test_empty_extra_returns_none(self):
        d = ScanFloodDetector()
        f = MockFlow(src_ip="10.0.0.1", extra={})
        assert d.record_and_classify(f, _now=T0) is None

    def test_bad_port_coercion(self):
        d = ScanFloodDetector()
        f = MockFlow(src_ip="10.0.0.1", extra={"proto": "6", "dst_port": "not-a-port", "src_pkts": "1", "dst_pkts": "0"})
        assert d.record_and_classify(f, _now=T0) is None

    def test_string_proto_coerced(self):
        d = ScanFloodDetector()
        # NFStream may store proto as string "6" — should be coerced correctly
        for _ in range(50):
            f = MockFlow(src_ip="10.0.0.5", extra={"proto": "6", "dst_port": 80, "src_pkts": 1, "dst_pkts": 0})
            result = d.record_and_classify(f, _now=T0)
        assert result == "syn_flood"
