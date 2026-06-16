# 0012 — Installer versioning and update safety

- **Status:** Accepted
- **Phase:** 3 — Desktop packaging and distribution

## Context

The initial Inno Setup script (introduced with [0010](0010-windows-desktop-packaging.md))
had three update-safety gaps that would have caused silent failures or user
confusion the first time a new installer was distributed:

1. **No `AppId`** — Inno Setup derives the application identity from `AppName` when
   no explicit GUID is given. Any future rename or typo in `AppName` would cause
   Windows to treat the new installer as a different application, leaving the old
   entry in Add/Remove Programs alongside the new one.

2. **Hardcoded `AppVersion=1.0.0`** — the build script accepted a `$Version`
   parameter but never forwarded it to `iscc`, so the version displayed in
   Add/Remove Programs was always `1.0.0` regardless of what was actually shipped.

3. **No `CloseApplications`** — if the user ran a new installer while ADNS was
   open, Windows would lock `ADNS.exe` and the installer would silently skip
   overwriting it while still reporting success. The user would believe they had
   updated but the old binary would still be running.

## Decision

Three targeted fixes to `installer.iss` and `scripts/build_installer.ps1`:

- **`AppId={{8EC917E9-8DB8-4681-A41E-2A03D9FEFE33}`** — a fixed GUID generated
  once and committed. Inno Setup uses this as the stable identity key in the
  Windows registry, regardless of what `AppName` says. The double `{{` is Inno
  Setup's escape for a literal brace in the `[Setup]` section.

- **`AppVersion={#MyAppVersion}`** with `/DMyAppVersion=$Version` passed to `iscc`
  from the build script. The `$Version` parameter defaults to `0.0.1` and is the
  single source of truth; it flows into the installer metadata, Add/Remove Programs,
  and the Inno Setup wizard title without any manual editing of the `.iss` file.

- **`CloseApplications=yes`** — Inno Setup will detect running processes that hold
  files in the install directory, show the user a list, and wait for them to be
  closed (or offer to close them automatically) before proceeding with the file
  copy. This ensures the binary overwrite always succeeds.

To ship a new version: `pwsh scripts\build_installer.ps1 -Version X.Y.Z`.

## Consequences

- Updates install cleanly over existing installations without duplicate registry
  entries or leftover files.
- The version in Add/Remove Programs matches the version shipped, making it
  straightforward to verify which build is installed.
- `CloseApplications=yes` adds a brief pause to the install flow when ADNS is
  running, but the alternative — a silent partial update — is worse in every way.
- The `AppId` GUID must never be changed. Changing it would break the update chain
  for all existing installations (same effect as removing it). It is intentionally
  not templated or auto-generated per build.
