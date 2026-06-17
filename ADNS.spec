# PyInstaller spec for ADNS desktop application.
# Run from repo root after:
#   npm run build  (inside frontend/adns-frontend)
#   pip install -r requirements-desktop.txt

import glob
import os

# Bundle tshark + its DLLs so the app captures traffic without a separate Wireshark install.
# Npcap (the packet capture driver) must still be installed on the target machine.
# Bundle npcap installer if present in repo root (download from https://npcap.com).
_npcap_datas = []
_npcap_installer = os.path.join(os.path.abspath("."), "npcap-installer.exe")
if os.path.isfile(_npcap_installer):
    _npcap_datas.append((_npcap_installer, "."))

_WIRESHARK_DIR = r"C:\Program Files\Wireshark"
_tshark_datas = []
if os.path.isdir(_WIRESHARK_DIR):
    for _f in glob.glob(os.path.join(_WIRESHARK_DIR, "*.dll")):
        _tshark_datas.append((_f, "tshark"))
    _tshark_exe = os.path.join(_WIRESHARK_DIR, "tshark.exe")
    if os.path.isfile(_tshark_exe):
        _tshark_datas.append((_tshark_exe, "tshark"))
    _dumpcap_exe = os.path.join(_WIRESHARK_DIR, "dumpcap.exe")
    if os.path.isfile(_dumpcap_exe):
        _tshark_datas.append((_dumpcap_exe, "tshark"))

from PyInstaller.utils.hooks import collect_all, collect_data_files

block_cipher = None

# Collect all sklearn files — it uses a lot of data files and Cython extensions
sklearn_datas, sklearn_binaries, sklearn_hiddenimports = collect_all("sklearn")
webview_datas, webview_binaries, webview_hiddenimports = collect_all("webview")
pystray_datas, pystray_binaries, pystray_hiddenimports = collect_all("pystray")

# collect_all("xgboost") misses the native DLL — add it explicitly.
import xgboost as _xgb
_xgb_lib = os.path.join(os.path.dirname(_xgb.__file__), "lib", "xgboost.dll")
_xgboost_binaries = [(_xgb_lib, "xgboost/lib")] if os.path.isfile(_xgb_lib) else []

a = Analysis(
    ["launcher.py"],
    pathex=["api"],          # so 'from app import ...' resolves
    binaries=sklearn_binaries + webview_binaries + pystray_binaries + _xgboost_binaries,
    datas=[
        # React production build
        ("frontend/adns-frontend/dist", "dist"),
        # Trained model artifacts (included if present; app degrades gracefully without them)
        ("api/model_artifacts", "model_artifacts"),
        # Flask app source files (all modules in api/)
        ("api/*.py", "api"),
        # App icon (used by the desktop shortcut)
        ("assets/icon.ico", "assets"),
    ] + sklearn_datas + webview_datas + pystray_datas + _tshark_datas + _npcap_datas,
    hiddenimports=[
        # Flask ecosystem
        "flask_cors",
        "flask_sqlalchemy",
        "sqlalchemy.dialects.sqlite",
        "sqlalchemy.dialects.sqlite.pysqlite",
        "sqlalchemy.dialects.postgresql",
        "sqlalchemy.dialects.postgresql.base",
        "sqlalchemy.pool.impl",
        # ML stack
        "joblib",
        "numpy",
        "pandas",
        "xgboost",
        # pywebview Windows backends
        "webview.platforms.winforms",
        "webview.platforms.edgechromium",
        "clr",
        "pystray",
        "pystray._win32",
        "PIL",
        "PIL.Image",
    ] + sklearn_hiddenimports + webview_hiddenimports + pystray_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "psycopg2",
        "psycopg2_binary",
        "alembic",
        "tkinter",
        "matplotlib",
        "IPython",
        "jupyter",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="ADNS",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,        # no terminal window on Windows
    icon="assets/icon.ico",
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="ADNS",
)
