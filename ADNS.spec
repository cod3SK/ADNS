# PyInstaller spec for ADNS desktop application.
# Run from repo root after:
#   npm run build  (inside frontend/adns-frontend)
#   pip install -r requirements-desktop.txt

import os

# Bundle npcap installer if present in repo root (download from https://npcap.com).
# Npcap (the packet capture driver) must still be installed on the target machine.
_npcap_datas = []
_npcap_installer = os.path.join(os.path.abspath("."), "npcap-installer.exe")
if os.path.isfile(_npcap_installer):
    _npcap_datas.append((_npcap_installer, "."))

from PyInstaller.utils.hooks import collect_all, collect_data_files
import importlib.util as _ilu

block_cipher = None

# Collect all sklearn files — it uses a lot of data files and Cython extensions
sklearn_datas, sklearn_binaries, sklearn_hiddenimports = collect_all("sklearn")
webview_datas, webview_binaries, webview_hiddenimports = collect_all("webview")
pystray_datas, pystray_binaries, pystray_hiddenimports = collect_all("pystray")

# collect_all bundles xgboost Python files but misses the native DLL on Windows — add it explicitly.
import xgboost as _xgb
xgboost_datas, xgboost_binaries, xgboost_hiddenimports = collect_all("xgboost")
_xgb_lib = os.path.join(os.path.dirname(_xgb.__file__), "lib", "xgboost.dll")
if os.path.isfile(_xgb_lib) and not any(_xgb_lib == src for src, _ in xgboost_binaries):
    xgboost_binaries.append((_xgb_lib, "xgboost/lib"))

# NFStream: collect Python package + _lib_engine.pyd (CFFI extension at site-packages root).
# collect_all("nfstream") finds only the Python package — the native .pyd is one level up.
nfstream_datas, nfstream_binaries_pkg, nfstream_hiddenimports = collect_all("nfstream")
_lib_engine_spec = _ilu.find_spec("_lib_engine")
_nfstream_extra_binaries = []
if _lib_engine_spec and _lib_engine_spec.origin:
    _nfstream_extra_binaries.append((_lib_engine_spec.origin, "."))

a = Analysis(
    ["launcher.py"],
    pathex=["api", "ml"],    # api: 'from app import ...' resolves; ml: adns_flows package
    binaries=(sklearn_binaries + webview_binaries + pystray_binaries + xgboost_binaries
              + nfstream_binaries_pkg + _nfstream_extra_binaries),
    datas=[
        # React production build
        ("frontend/adns-frontend/dist", "dist"),
        # Trained model artifacts — REQUIRED for detection.  Absent model blocks
        # /capture/autostart with HTTP 503 (not silent).  See api/model_runner.py.
        # nfstream_model.joblib is tracked via Git LFS; run `git lfs pull` after clone.
        ("api/model_artifacts", "model_artifacts"),
        # Corpus parquets bundled for offline calibration (stage 4 retrain).
        # pipeline.py resolves these at sys._MEIPASS/corpus/ in the frozen exe.
        ("outputs/corpus/unsw_flows.parquet",        "corpus"),
        ("outputs/corpus/gotham_flows.parquet",      "corpus"),
        ("outputs/corpus/cic_tuesday_flows.parquet", "corpus"),
        # Flask app source files (all modules in api/ and calibration sub-package)
        ("api/*.py", "api"),
        ("api/calibration/*.py", "api/calibration"),
        # App icon (used by the desktop shortcut)
        ("assets/icon.ico", "assets"),
    ] + sklearn_datas + webview_datas + pystray_datas + xgboost_datas
      + nfstream_datas + _npcap_datas,
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
        # pywebview Windows backends
        "webview.platforms.winforms",
        "webview.platforms.edgechromium",
        "clr",
        "pystray",
        "pystray._win32",
        "PIL",
        "PIL.Image",
        # adns_flows shared extractor (ml/adns_flows/ — on pathex, but api/*.py are data
        # files so PyInstaller won't trace their imports automatically)
        "adns_flows",
        "adns_flows.schema",
        "adns_flows.extract_nfstream",
        "adns_flows.nfstream_config",
        # NFStream serving module (api/ data file, imports not auto-traced)
        "serving_nfstream",
        # Calibration pipeline (api/calibration/ sub-package)
        "calibration",
        "calibration.pipeline",
        "calibration.whitelist",
        # Additional api/ modules (data files, not auto-traced)
        "calibration_routes",
        "scan_flood_detector",
        # NFStream sub-modules (collect_all may miss lazy-imported ones)
        "nfstream",
        "nfstream.streamer",
        "nfstream.meter",
        "nfstream.plugin",
        "nfstream.engine",
        "nfstream.utils",
        # multiprocessing spawn protocol (NFStream meter workers use spawn on Windows)
        "multiprocessing.spawn",
        "multiprocessing.forkserver",
        "multiprocessing.popen_spawn_win32",
    ] + sklearn_hiddenimports + webview_hiddenimports + pystray_hiddenimports
      + xgboost_hiddenimports + nfstream_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=["pyi_hooks/rthook_nfstream_npcap.py"],
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
