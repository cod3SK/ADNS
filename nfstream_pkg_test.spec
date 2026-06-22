"""Minimal PyInstaller spec to test NFStream packaging feasibility on Windows."""
import os
import site

sp = site.getsitepackages()[1]

from PyInstaller.utils.hooks import collect_all
nfstream_datas, nfstream_binaries, nfstream_hiddenimports = collect_all("nfstream")

# _lib_engine.pyd lives at site-packages root (not inside nfstream package)
# Must be added manually as a binary so PyInstaller bundles it.
_lib_engine = os.path.join(sp, "_lib_engine.pyd")

# wpcap.dll from Npcap — bundle it next to _lib_engine.pyd so Windows finds it
# when loading the frozen app (application directory is always in DLL search path).
_wpcap = r"C:\Windows\System32\Npcap\wpcap.dll"
_packet = r"C:\Windows\System32\Npcap\packet.dll"

# cffi backend
_cffi_backend = os.path.join(sp, "cffi", "_cffi_backend.pyd")

extra_binaries = []
if os.path.isfile(_lib_engine):
    extra_binaries.append((_lib_engine, "."))
if os.path.isfile(_wpcap):
    extra_binaries.append((_wpcap, "."))
if os.path.isfile(_packet):
    extra_binaries.append((_packet, "."))
if os.path.isfile(_cffi_backend):
    extra_binaries.append((_cffi_backend, "cffi"))

block_cipher = None

a = Analysis(
    ["nfstream_pkg_test.py"],
    pathex=[],
    binaries=nfstream_binaries + extra_binaries,
    datas=nfstream_datas,
    hiddenimports=nfstream_hiddenimports + ["cffi", "_cffi_backend", "dpkt"],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=["rthook_nfstream.py"],
    excludes=["tkinter", "matplotlib", "IPython"],
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
    name="nfstream_pkg_test",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="nfstream_pkg_test",
)
