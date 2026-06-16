# Changelog

All notable changes to ADNS are documented here.

## [0.0.1] — 2026-06-16

### Changed

**UI — left nav rail with tabbed layout**
- Replaced the single-page scrolling layout with a persistent left sidebar nav (158 px) and four tabs:
  - **Dashboard** — signal summary metric cards (anomalies, max score, % anomalous) + four charts (anomalous flows, threat timeline, severity mix donut, anomaly score over recent flows)
  - **Flows** — recent flows table with source-IP filter and per-row block action
  - **Flows Manager** — anomalous flow list + blocked IPs panel with unblock actions
  - **Settings** — capture pipeline (tshark status, interface selector, start/stop capture)
- Kill switch button stays in the top header as a global emergency control
- `app-shell` is now `height: 100vh; overflow: hidden`; only the tab content area scrolls
- Metric cards switch from a vertical sidebar stack to a horizontal three-column row on the Dashboard tab

**Installer — update reliability**
- Added `AppId` GUID (`{8EC917E9-8DB8-4681-A41E-2A03D9FEFE33}`) so Windows always recognises reinstalls and updates as the same application and never creates a duplicate Add/Remove Programs entry
- Added `CloseApplications=yes` so the installer prompts the user to close a running ADNS instance before overwriting files (previously the executable could be silently skipped if the app was open)
- Wired the `$Version` parameter from `build_installer.ps1` through to Inno Setup via `/DMyAppVersion` so Add/Remove Programs reflects the real version number instead of a hardcoded string
- Default version in `build_installer.ps1` changed from `1.0.0` to `0.0.1`

### Fixed

- Block IP inline handler extracted to a shared `blockIp()` function — both the Flows table and Flows Manager table now share the same code path
- `Th` and `Td` helper components now forward the `className` prop (was previously silently ignored)
