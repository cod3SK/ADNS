# Build the ADNS Windows installer locally.
# Run from the repo root: pwsh scripts\build_installer.ps1
#
# Prerequisites:
#   - Node.js 18+
#   - Python 3.10+
#   - Inno Setup 6    https://jrsoftware.org/isinfo.php
#   - pip install -r requirements-desktop.txt pyinstaller
#   - npcap-installer.exe in repo root
#       Download from https://npcap.com, rename to npcap-installer.exe

param(
    [string]$Version = "0.0.1"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path $PSScriptRoot -Parent

Write-Host "==> Building ADNS $Version installer" -ForegroundColor Cyan

# --- Preflight checks ---------------------------------------------------
$missing = @()

if (-not (Test-Path "$Root\npcap-installer.exe")) {
    $missing += "npcap-installer.exe  (download from https://npcap.com, rename, place in repo root)"
}

$iscc = "C:\Program Files (x86)\Inno Setup 6\ISCC.exe"
if (-not (Test-Path $iscc)) {
    $missing += "Inno Setup 6  (https://jrsoftware.org/isinfo.php)"
}

if ($missing.Count -gt 0) {
    Write-Host "`nPreflight failed - missing required components:" -ForegroundColor Red
    $missing | ForEach-Object { Write-Host "  - $_" -ForegroundColor Red }
    exit 1
}
# ------------------------------------------------------------------------

# 1. React frontend build
Write-Host "`n[1/3] Building React frontend..." -ForegroundColor Yellow
$npmAvailable = $null -ne (Get-Command npm -ErrorAction SilentlyContinue)
$distExists   = Test-Path "$Root\frontend\adns-frontend\dist\index.html"

if ($npmAvailable) {
    Push-Location "$Root\frontend\adns-frontend"
    npm install
    if ($LASTEXITCODE -ne 0) { throw "npm install failed" }
    npm run build
    if ($LASTEXITCODE -ne 0) { throw "npm run build failed" }
    Pop-Location
} elseif ($distExists) {
    Write-Host "  npm not found — using existing dist build at frontend/adns-frontend/dist" -ForegroundColor Yellow
} else {
    throw "npm is not installed and no pre-built dist found.`nInstall Node.js 18+ (https://nodejs.org) or copy a pre-built dist/ into frontend/adns-frontend/dist."
}

# 2. PyInstaller
Write-Host "`n[2/3] Running PyInstaller..." -ForegroundColor Yellow
Set-Location $Root
pyinstaller ADNS.spec --clean -y
if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed" }

# 3. Inno Setup
Write-Host "`n[3/3] Building installer with Inno Setup..." -ForegroundColor Yellow
& $iscc /DMyAppVersion=$Version installer.iss
if ($LASTEXITCODE -ne 0) { throw "Inno Setup failed" }

$output = "$Root\Output\ADNS_installer.exe"
if (Test-Path $output) {
    $size = [math]::Round((Get-Item $output).Length / 1MB, 1)
    Write-Host "`nDone! $output ($size MB)" -ForegroundColor Green
} else {
    throw "Installer not found at expected path: $output"
}
