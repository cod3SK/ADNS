"""Unit tests for dumpcap integration — interface enumeration and capture agent."""

import json
import re
import subprocess
import sys
from unittest.mock import MagicMock, patch

import pytest


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_run_result(stdout="", returncode=0):
    r = MagicMock(spec=subprocess.CompletedProcess)
    r.stdout = stdout.encode("utf-8") if isinstance(stdout, str) else stdout
    r.returncode = returncode
    return r


# Realistic tshark -D output (what dumpcap produces and tshark forwards)
_SAMPLE_IFACE_OUTPUT = """\
1. \\Device\\NPF_{AAA} (Wi-Fi)
2. \\Device\\NPF_{BBB} (Ethernet)
3. \\Device\\NPF_{CCC} (Local Area Connection* 3)
4. \\Device\\NPF_Loopback (Adapter for loopback traffic capture)
"""


# ── Interface line parsing (the regex inside list_interfaces) ─────────────────

class TestInterfaceLineParsing:
    """The regex `r'(\\d+)\\.\\s+(\\S+)(?:\\s+\\((.+)\\))?'` is the core parser."""

    PATTERN = re.compile(r"(\d+)\.\s+(\S+)(?:\s+\((.+)\))?")

    def _parse(self, line):
        m = self.PATTERN.match(line.strip())
        if not m:
            return None
        idx, dev, name = int(m.group(1)), m.group(2), m.group(3) or m.group(2)
        return {"index": idx, "device": dev, "name": name}

    def test_standard_line_with_friendly_name(self):
        r = self._parse(r"1. \Device\NPF_{AAA} (Wi-Fi)")
        assert r == {"index": 1, "device": r"\Device\NPF_{AAA}", "name": "Wi-Fi"}

    def test_loopback_line(self):
        r = self._parse(r"4. \Device\NPF_Loopback (Adapter for loopback traffic capture)")
        assert r["index"] == 4
        assert "Loopback" in r["device"]
        assert r["name"] == "Adapter for loopback traffic capture"

    def test_line_without_friendly_name_falls_back_to_device(self):
        r = self._parse(r"2. \Device\NPF_{BBB}")
        assert r["name"] == r"\Device\NPF_{BBB}"

    def test_line_with_spaces_in_friendly_name(self):
        r = self._parse(r"3. \Device\NPF_{CCC} (Local Area Connection* 3)")
        assert r["name"] == "Local Area Connection* 3"

    def test_malformed_line_returns_none(self):
        assert self._parse("not an interface line") is None

    def test_full_sample_output_parses_all_four(self):
        pattern = self.PATTERN
        results = []
        for line in _SAMPLE_IFACE_OUTPUT.strip().splitlines():
            m = pattern.match(line.strip())
            if m:
                results.append(int(m.group(1)))
        assert results == [1, 2, 3, 4]


# ── /interfaces endpoint ──────────────────────────────────────────────────────

class TestInterfacesEndpoint:
    def test_returns_list_on_success(self, client, app):
        with patch("app._find_tshark", return_value="tshark.exe"), \
             patch("app.subprocess.run", return_value=_make_run_result(_SAMPLE_IFACE_OUTPUT)):
            with app.app_context():
                res = client.get("/interfaces")
        assert res.status_code == 200
        data = res.get_json()
        assert isinstance(data, list)
        assert len(data) == 4

    def test_returns_503_when_tshark_not_found(self, client, app):
        with patch("app._find_tshark", return_value=None):
            with app.app_context():
                res = client.get("/interfaces")
        assert res.status_code == 503
        assert res.get_json()["interfaces"] == []

    def test_returns_504_on_timeout(self, client, app):
        with patch("app._find_tshark", return_value="tshark.exe"), \
             patch("app.subprocess.run", side_effect=subprocess.TimeoutExpired("tshark", 5)):
            with app.app_context():
                res = client.get("/interfaces")
        assert res.status_code == 504

    def test_returns_500_on_unexpected_error(self, client, app):
        with patch("app._find_tshark", return_value="tshark.exe"), \
             patch("app.subprocess.run", side_effect=OSError("permission denied")):
            with app.app_context():
                res = client.get("/interfaces")
        assert res.status_code == 500

    def test_empty_stdout_returns_empty_list(self, client, app):
        with patch("app._find_tshark", return_value="tshark.exe"), \
             patch("app.subprocess.run", return_value=_make_run_result("")):
            with app.app_context():
                res = client.get("/interfaces")
        assert res.status_code == 200
        assert res.get_json() == []

    def test_interface_fields_present(self, client, app):
        with patch("app._find_tshark", return_value="tshark.exe"), \
             patch("app.subprocess.run", return_value=_make_run_result(_SAMPLE_IFACE_OUTPUT)):
            with app.app_context():
                ifaces = client.get("/interfaces").get_json()
        first = ifaces[0]
        assert {"index", "device", "name"} == set(first.keys())

    def test_wifi_interface_identified(self, client, app):
        with patch("app._find_tshark", return_value="tshark.exe"), \
             patch("app.subprocess.run", return_value=_make_run_result(_SAMPLE_IFACE_OUTPUT)):
            with app.app_context():
                ifaces = client.get("/interfaces").get_json()
        names = [i["name"] for i in ifaces]
        assert "Wi-Fi" in names


# ── Agent status — tshark_found reflects dumpcap availability ─────────────────

class TestAgentStatus:
    def test_tshark_found_true_when_binary_exists(self, client, app):
        with patch("app._find_tshark", return_value="tshark.exe"):
            with app.app_context():
                body = client.get("/agent/status").get_json()
        assert body["tshark_found"] is True

    def test_tshark_found_false_when_binary_missing(self, client, app):
        with patch("app._find_tshark", return_value=None):
            with app.app_context():
                body = client.get("/agent/status").get_json()
        assert body["tshark_found"] is False

    def test_agent_start_missing_interface_returns_400(self, client, app):
        with app.app_context():
            res = client.post(
                "/agent/start",
                data=json.dumps({}),
                content_type="application/json",
            )
        assert res.status_code == 400
        assert "interface" in res.get_json()["error"]
