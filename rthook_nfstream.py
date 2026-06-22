"""PyInstaller runtime hook: DLL search path for NFStream + multiprocessing notes.

Why this hook exists
--------------------
_lib_engine.pyd (NFStream's native engine) links against wpcap.dll + packet.dll
from Npcap.  In the onedir layout those DLLs land in _MEIPASS (_internal/).
Python 3.8+ loads extension modules with LOAD_LIBRARY_SEARCH_DEFAULT_DIRS which
does NOT automatically include _MEIPASS, so the DLLs go unfound even when they
are physically present in the same folder.  Calling add_dll_directory() here runs
before any import fires, INCLUDING in child processes spawned by NFStream's
multiprocessing.get_context('spawn'), because the bootloader executes all runtime
hooks at process startup.

Why we do NOT call multiprocessing.freeze_support() here
---------------------------------------------------------
freeze_support() must be called inside `if __name__ == '__main__':` in the entry
script — NOT in a runtime hook.  The hook runs in every process (workers too);
calling freeze_support() or set_start_method() here would conflict with the
worker setup that multiprocessing already performs.  The correct guard is in
nfstream_pkg_test.py (and launcher.py):

    if __name__ == '__main__':
        import multiprocessing
        multiprocessing.freeze_support()
        main()

Why we do NOT call set_start_method() here
------------------------------------------
NFStream calls multiprocessing.get_context('spawn') directly, so the default
start method is irrelevant.  Calling set_start_method() in a runtime hook that
also runs inside worker processes would raise RuntimeError ("context has already
been set").
"""

import os
import sys

if sys.platform == "win32" and hasattr(sys, "_MEIPASS"):
    _meipass = sys._MEIPASS
    if os.path.isdir(_meipass):
        os.add_dll_directory(_meipass)
