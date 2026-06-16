"""Unit tests for launcher.py utility functions."""

import os
import socket
import sys
import threading
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import launcher


class TestPortInUse:
    def test_closed_port_returns_false(self):
        # Port 19999 is almost certainly free in CI
        assert launcher._port_in_use(19999) is False

    def test_open_port_returns_true(self):
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("127.0.0.1", 0))
        srv.listen(1)
        port = srv.getsockname()[1]
        try:
            assert launcher._port_in_use(port) is True
        finally:
            srv.close()


class TestIsAdmin:
    def test_returns_bool(self):
        result = launcher._is_admin()
        assert isinstance(result, bool)


class TestNpcapInstalled:
    def test_returns_true_when_registry_key_found(self):
        with patch("launcher.winreg.OpenKey", return_value=MagicMock()):
            assert launcher._npcap_installed() is True

    def test_returns_false_when_registry_missing(self):
        with patch("launcher.winreg.OpenKey", side_effect=OSError):
            assert launcher._npcap_installed() is False

    def test_falls_back_to_npf_key(self):
        call_count = {"n": 0}
        def side_effect(hkey, key):
            call_count["n"] += 1
            if "npcap" in key:
                raise OSError
            return MagicMock()
        with patch("launcher.winreg.OpenKey", side_effect=side_effect):
            result = launcher._npcap_installed()
        assert result is True
        assert call_count["n"] == 2


class TestResourcePath:
    def test_dev_mode_uses_file_dir(self):
        path = launcher.resource_path("dist")
        assert os.path.isabs(path)
        assert "dist" in path

    def test_bundle_mode_uses_meipass(self, tmp_path):
        sys._MEIPASS = str(tmp_path)
        try:
            path = launcher.resource_path("dist")
            assert path == str(tmp_path / "dist") or path.startswith(str(tmp_path))
        finally:
            del sys._MEIPASS


class TestDataDir:
    def test_creates_adns_subdir(self, tmp_path):
        with patch.dict(os.environ, {"APPDATA": str(tmp_path)}):
            d = launcher._data_dir()
        assert os.path.isdir(d)
        assert d.endswith("ADNS")


class TestStripApiPrefix:
    def _make_app(self):
        calls = []
        def wsgi(environ, start_response):
            calls.append(environ.get("PATH_INFO"))
            return []
        return launcher._StripApiPrefix(wsgi), calls

    def test_strips_api_prefix(self):
        mw, calls = self._make_app()
        mw({"PATH_INFO": "/api/flows"}, None)
        assert calls[0] == "/flows"

    def test_bare_api_maps_to_root(self):
        mw, calls = self._make_app()
        mw({"PATH_INFO": "/api"}, None)
        assert calls[0] == "/"

    def test_non_api_path_unchanged(self):
        mw, calls = self._make_app()
        mw({"PATH_INFO": "/health"}, None)
        assert calls[0] == "/health"
