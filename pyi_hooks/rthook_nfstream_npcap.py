# Runtime hook: add Npcap DLL directory so _lib_engine.pyd can load wpcap.dll -> packet.dll.
# Npcap secure-mode installs DLLs into System32\\Npcap\\ which is not in the default
# DLL search path for spawned processes. os.add_dll_directory() fixes this.
import os

_NPCAP_DIR = r"C:\Windows\System32\Npcap"
if os.path.isdir(_NPCAP_DIR):
    try:
        os.add_dll_directory(_NPCAP_DIR)
    except (AttributeError, OSError):
        pass
