# 0010 — Windows desktop packaging with PyInstaller and Inno Setup

- **Status:** Accepted
- **Phase:** 3 — Desktop packaging and distribution

## Context

The Docker Compose stack is the right target for Linux deployments and CI, but it
is a high barrier for the primary audience of this project: a reviewer, recruiter,
or interviewer on a Windows machine who wants to open the app without installing
Python, Node.js, Docker, or any other toolchain. The system also requires elevated
OS privileges at runtime (raw-socket access for Npcap, firewall rule writes for the
killswitch) and depends on `tshark`/`dumpcap` binaries that most Windows users do
not have.

## Decision

Produce a self-contained Windows installer that bundles every dependency:

- **PyInstaller** compiles the Flask app, ML models (`flow_detector.joblib`,
  `meta_model_combined.joblib`), and all Python dependencies into a standalone
  `dist/ADNS/` directory. The React production build (`frontend/adns-frontend/dist/`)
  and the tshark/dumpcap binaries (including ~50 Wireshark DLLs) are included via
  the `.spec` file's `datas` list.
- **Inno Setup 6** wraps `dist/ADNS/` into a single `ADNS_installer.exe`. It
  installs to `%LocalAppData%\ADNS` (no UAC prompt for the install step itself) and
  creates desktop and Start Menu shortcuts.
- **Npcap** is bundled as `npcap-installer.exe` inside the Inno Setup package. A
  Pascal script (`NpcapMissing()`) checks the registry at install time and only
  extracts and runs the Npcap installer silently when Npcap is absent.
- **UAC self-elevation:** `ADNS.exe` detects at startup whether it is running with
  administrator privileges. If not, it re-launches itself via `ShellExecuteW` with
  the `runas` verb, so the user sees a single UAC prompt rather than a confusing
  failure when the app tries to open a raw socket or write a firewall rule.
- **Bundled tshark preferred over system tshark:** when running as administrator
  the app prefers the bundled `_internal/tshark/tshark.exe` over any system
  installation, so the capture path is deterministic regardless of what Wireshark
  version (if any) the user has installed.
- The build pipeline is a single PowerShell script (`scripts/build_installer.ps1`)
  that runs `npm run build`, `pyinstaller ADNS.spec --clean`, and `iscc` in
  sequence. The version number is passed as a `/D` define so Inno Setup stamps it
  into the installer metadata and Add/Remove Programs.

## Consequences

- End-users install ADNS with a single `.exe` and no prerequisites beyond a
  Windows machine; Npcap is handled automatically.
- The whole Flask + React + ML stack runs inside one process. The service
  decomposition from [0001](0001-microservice-architecture.md) is preserved in
  code but collapsed at the deployment boundary.
- The bundled tshark means the installer is large (~120 MB compressed) and the
  Wireshark DLLs must be kept up to date when the tshark version changes.
- PyInstaller startup is slower than a native binary (~3–5 s on first launch while
  the bootloader unpacks to a temp directory). This is documented in the README.
- The build requires four tools (Node.js, Python + pyinstaller, Inno Setup,
  `npcap-installer.exe`) and is Windows-only. The CI workflow handles this on a
  `windows-latest` runner.
