"""Desktop launcher for ADNS.

Starts the Flask API in a background thread, waits for it to be ready,
then opens a native pywebview window — no browser required.

The _StripApiPrefix WSGI middleware rewrites /api/flows -> /flows etc. so
the React build (which uses /api/* paths via the Vite proxy convention)
works against Flask's routes without any route changes.
"""

import os
import sys
import threading
import time
import urllib.request


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


def _start_flask(data_dir: str) -> None:
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

    from werkzeug.serving import run_simple  # noqa: PLC0415
    run_simple(
        "127.0.0.1", 5000, app,
        use_reloader=False,
        use_debugger=False,
        threaded=True,
    )


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
        import ctypes
        ctypes.windll.user32.MessageBoxW(0, msg, "ADNS — Startup Error", 0x10)
    except Exception:
        print(msg, file=sys.stderr)
    sys.exit(1)


def main() -> None:
    data_dir = _data_dir()

    t = threading.Thread(target=_start_flask, args=(data_dir,), daemon=True)
    t.start()

    if not _wait_for_api("http://127.0.0.1:5000/health"):
        _fatal(
            "ADNS failed to start.\n\n"
            "Port 5000 may already be in use.\n"
            "Check Task Manager and close anything running on port 5000."
        )

    import webview  # noqa: PLC0415  (not available in test env)

    webview.create_window(
        "ADNS — Anomaly Detection Network System",
        "http://127.0.0.1:5000",
        width=1400,
        height=900,
        min_size=(1024, 600),
    )
    webview.start()


if __name__ == "__main__":
    main()
