"""Unit tests for tshark parsing helpers in api/app.py."""

import time
import pytest
from app import (
    _ts_safe_float,
    _ts_safe_int,
    _ts_proto,
    _ts_service,
    _parse_tshark_line,
    _build_tshark_cmd,
    _tshark_env,
    _TSHARK_FIELDS,
)


class TestTsSafeFloat:
    def test_valid_string_converts(self):
        assert _ts_safe_float("1234567890.123", 0.0) == pytest.approx(1234567890.123)

    def test_empty_string_returns_fallback(self):
        assert _ts_safe_float("", 42.0) == 42.0

    def test_non_numeric_returns_fallback(self):
        assert _ts_safe_float("not_a_number", -1.0) == -1.0


class TestTsSafeInt:
    def test_decimal_string(self):
        assert _ts_safe_int("443") == 443

    def test_hex_string(self):
        assert _ts_safe_int("0x01bb") == 443

    def test_empty_returns_none(self):
        assert _ts_safe_int("") is None

    def test_invalid_returns_none(self):
        assert _ts_safe_int("abc") is None


class TestTsProto:
    def test_numeric_6_maps_to_tcp(self):
        assert _ts_proto("6") == "TCP"

    def test_numeric_17_maps_to_udp(self):
        assert _ts_proto("17") == "UDP"

    def test_numeric_1_maps_to_icmp(self):
        assert _ts_proto("1") == "ICMP"

    def test_empty_returns_other(self):
        assert _ts_proto("") == "OTHER"

    def test_named_proto_uppercased(self):
        assert _ts_proto("gre") == "GRE"


class TestTsService:
    def test_http_method_wins(self):
        assert _ts_service("TCP", 12345, 80, None, "GET", None) == "http"

    def test_https_by_dst_port_443(self):
        assert _ts_service("TCP", 0, 443, None, None, None) == "https"

    def test_dns_by_dst_port_53(self):
        assert _ts_service("UDP", 0, 53, None, None, None) == "dns"

    def test_ssh_by_port_22(self):
        assert _ts_service("TCP", 0, 22, None, None, None) == "ssh"

    def test_unknown_port_falls_back_to_proto(self):
        result = _ts_service("UDP", 0, 9999, None, None, None)
        assert result == "udp"


class TestParseTsharkLine:
    def _make_line(self, overrides=None):
        fields = {
            "ts": str(time.time()),
            "src": "192.168.1.1",
            "dst": "8.8.8.8",
            "proto": "6",
            "len": "1500",
            "tcp_sport": "12345",
            "tcp_dport": "443",
            "udp_sport": "",
            "udp_dport": "",
            "dns_name": "",
            "dns_qtype": "",
            "dns_qclass": "",
            "dns_rcode": "",
            "http_method": "",
            "http_uri": "",
            "http_ua": "",
            "http_status": "",
            "http_clen": "",
            "ssl_ver": "",
            "ssl_cipher": "",
        }
        if overrides:
            for k, v in overrides.items():
                fields[k] = v
        return "\t".join(list(fields.values()))

    def test_valid_line_returns_dict(self):
        line = self._make_line()
        rec = _parse_tshark_line(line)
        assert rec is not None
        assert rec["src_ip"] == "192.168.1.1"
        assert rec["dst_ip"] == "8.8.8.8"
        assert rec["proto"] == "TCP"
        assert rec["bytes"] == 1500

    def test_missing_src_returns_none(self):
        line = self._make_line({"src": ""})
        assert _parse_tshark_line(line) is None

    def test_missing_dst_returns_none(self):
        line = self._make_line({"dst": ""})
        assert _parse_tshark_line(line) is None

    def test_short_line_padded_gracefully(self):
        short = "1234567890.0\t10.0.0.1\t10.0.0.2"
        rec = _parse_tshark_line(short)
        assert rec is not None
        assert rec["src_ip"] == "10.0.0.1"

    def test_https_service_detected_via_port(self):
        line = self._make_line({"tcp_dport": "443"})
        rec = _parse_tshark_line(line)
        assert rec["service"] == "https"


class TestBuildTsharkCmd:
    def test_contains_interface_flag(self):
        cmd = _build_tshark_cmd("tshark.exe", r"\Device\NPF_{ABC}")
        assert "-i" in cmd
        assert r"\Device\NPF_{ABC}" in cmd

    def test_fields_flag_count(self):
        cmd = _build_tshark_cmd("tshark.exe", "eth0")
        field_flags = [c for c in cmd if c == "-e"]
        assert len(field_flags) == len(_TSHARK_FIELDS)

    def test_tab_separator_set(self):
        cmd = _build_tshark_cmd("tshark.exe", "eth0")
        assert "separator=\t" in cmd or r"separator=\t" in " ".join(cmd)


class TestTsharkEnv:
    def test_tshark_dir_prepended_to_path(self):
        env = _tshark_env(r"C:\Program Files\Wireshark\tshark.exe")
        assert r"C:\Program Files\Wireshark" in env["PATH"]
        assert env["PATH"].startswith(r"C:\Program Files\Wireshark")

    def test_wireshark_run_key_set(self):
        env = _tshark_env(r"C:\tools\tshark.exe")
        assert "WIRESHARK_RUN_FROM_BUILD_DIRECTORY" in env
