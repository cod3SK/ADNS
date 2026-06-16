"""Desktop launcher for ADNS.

Starts the Flask API in a background thread, waits for it to be ready,
then opens a native pywebview window — no browser required.

The _StripApiPrefix WSGI middleware rewrites /api/flows -> /flows etc. so
the React build (which uses /api/* paths via the Vite proxy convention)
works against Flask's routes without any route changes.

Closing the window minimizes to tray. Left-click the tray icon to restore;
right-click → Quit to exit completely.
"""

import atexit
import ctypes
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import winreg


def resource_path(relative: str) -> str:
    """Resolve a path that works both in dev and inside a PyInstaller bundle."""
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, relative)
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), relative)


def _data_dir() -> str:
    base = os.environ.get("APPDATA") or os.path.expanduser("~")
    path = os.path.join(base, "ADNS")
    os.makedirs(path, exist_ok=True)
    return path


class _StripApiPrefix:
    """WSGI middleware that strips the /api prefix before routing to Flask."""

    def __init__(self, wsgi_app):
        self._app = wsgi_app

    def __call__(self, environ, start_response):
        path = environ.get("PATH_INFO", "")
        if path.startswith("/api/"):
            environ["PATH_INFO"] = path[4:]   # /api/flows -> /flows
        elif path == "/api":
            environ["PATH_INFO"] = "/"
        return self._app(environ, start_response)


_flask_server = None


def _start_flask(data_dir: str) -> None:
    global _flask_server
    api_dir = resource_path("api")
    if api_dir not in sys.path:
        sys.path.insert(0, api_dir)

    # SQLite stored in user's AppData so it survives re-installs
    os.environ.setdefault(
        "SQLALCHEMY_DATABASE_URI",
        "sqlite:///{}".format(os.path.join(data_dir, "adns.db")),
    )
    os.environ.setdefault("ADNS_RDNS_ENABLED", "false")
    # Point Flask's static-serving route at the bundled React build
    os.environ["ADNS_FRONTEND_DIST"] = resource_path("dist")

    from app import app  # noqa: PLC0415  (deferred so env vars are set first)

    app.wsgi_app = _StripApiPrefix(app.wsgi_app)

    from werkzeug.serving import make_server  # noqa: PLC0415
    server = make_server("127.0.0.1", 5000, app, threaded=True)
    _flask_server = server
    server.serve_forever()
    server.server_close()  # release the socket so the port is immediately reusable


def _wait_for_api(url: str, timeout: float = 15.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url, timeout=1)
            return True
        except Exception:
            time.sleep(0.2)
    return False


def _fatal(msg: str) -> None:
    try:
        ctypes.windll.user32.MessageBoxW(0, msg, "ADNS — Startup Error", 0x10)
    except Exception:
        print(msg, file=sys.stderr)
    sys.exit(1)


def _port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def _is_admin() -> bool:
    try:
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:
        return False


def _elevate() -> None:
    params = " ".join(f'"{a}"' for a in sys.argv[1:]) if len(sys.argv) > 1 else ""
    ctypes.windll.shell32.ShellExecuteW(None, "runas", sys.executable, params, None, 1)
    sys.exit(0)


def _npcap_installed() -> bool:
    for key in (
        r"SYSTEM\CurrentControlSet\Services\npcap",
        r"SYSTEM\CurrentControlSet\Services\npf",
    ):
        try:
            winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key)
            return True
        except OSError:
            continue
    return False


def _ensure_npcap() -> None:
    if _npcap_installed():
        return
    installer = resource_path("npcap-installer.exe")
    if os.path.isfile(installer):
        ok = ctypes.windll.user32.MessageBoxW(
            0,
            "ADNS requires Npcap for network packet capture.\n\nClick OK to install it now.",
            "ADNS — Npcap Required",
            0x01,
        )
        if ok == 1:
            subprocess.run([installer, "/S"], check=True, timeout=120)
        else:
            sys.exit(0)
    else:
        _fatal(
            "Npcap is required for network packet capture but is not installed.\n\n"
            "Download and install Npcap from https://npcap.com, then restart ADNS."
        )


def _build_tray(window):
    """Create and return a pystray Icon wired to the webview window."""
    import pystray
    from PIL import Image

    icon_path = resource_path("assets/icon.ico")
    image = Image.open(icon_path)

    def on_show(icon=None, item=None):
        window.show()

    def on_quit(icon, item):
        icon.stop()
        srv = _flask_server
        if srv is not None:
            srv.shutdown()
        window.destroy()

    menu = pystray.Menu(
        pystray.MenuItem("Open ADNS", on_show, default=True),
        pystray.MenuItem("Quit", on_quit),
    )

    return pystray.Icon("ADNS", image, "ADNS", menu)


def main() -> None:
    if sys.platform == "win32" and not _is_admin():
        _elevate()
        return

    _ensure_npcap()

    data_dir = _data_dir()

    if _port_in_use(5000):
        _fatal(
            "Port 5000 is already in use.\n\n"
            "Another application (or a leftover ADNS process) is running on port 5000.\n"
            "Open Task Manager, find the process using port 5000, and close it, then try again."
        )

    t = threading.Thread(target=_start_flask, args=(data_dir,), daemon=True)
    t.start()

    if not _wait_for_api("http://127.0.0.1:5000/health"):
        _fatal(
            "ADNS failed to start.\n\n"
            "Port 5000 may already be in use.\n"
            "Check Task Manager and close anything running on port 5000."
        )

    # Flask is up — app module is imported. Register cleanup so tshark
    # (a child subprocess) is terminated on exit; Windows does not auto-kill
    # child processes when the parent exits.
    def _stop_capture():
        try:
            from app import _capture_agent  # noqa: PLC0415
            _capture_agent.stop()
        except Exception:
            pass

    atexit.register(_stop_capture)

    import webview  # noqa: PLC0415  (not available in test env)

    window = webview.create_window(
        "ADNS — Anomaly Detection Network System",
        "http://127.0.0.1:5000",
        width=1400,
        height=900,
        min_size=(1024, 600),
    )

    tray = _build_tray(window)

    def on_closing():
        window.hide()
        return False  # cancel the close; window stays alive hidden

    window.events.closing += on_closing

    tray_thread = threading.Thread(target=tray.run, daemon=True)
    tray_thread.start()

    webview.start()

    # webview.start() returns when window.destroy() is called from the tray.
    # Flask and the tray thread are daemon threads and exit with the process.
    # atexit handler above ensures tshark is terminated first.
    sys.exit(0)


if __name__ == "__main__":
    main()
