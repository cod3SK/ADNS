# Build the ADNS Windows installer locally.
# Run from the repo root: pwsh scripts\build_installer.ps1
#
# Prerequisites:
#   - Node.js 18+
#   - Python 3.10+
#   - Inno Setup 6 (https://jrsoftware.org/isinfo.php)
#   - pip install -r requirements-desktop.txt pyinstaller

param(
    [string]$Version = "1.0.0"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path $PSScriptRoot -Parent

Write-Host "==> Building ADNS $Version installer" -ForegroundColor Cyan

# 1. React frontend build
Write-Host "`n[1/3] Building React frontend..." -ForegroundColor Yellow
Push-Location "$Root\frontend\adns-frontend"
npm ci
if ($LASTEXITCODE -ne 0) { throw "npm ci failed" }
npm run build
if ($LASTEXITCODE -ne 0) { throw "npm run build failed" }
Pop-Location

# 2. PyInstaller
Write-Host "`n[2/3] Running PyInstaller..." -ForegroundColor Yellow
Set-Location $Root
pyinstaller ADNS.spec --clean
if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed" }

# 3. Inno Setup
Write-Host "`n[3/3] Building installer with Inno Setup..." -ForegroundColor Yellow
$iscc = "C:\Program Files (x86)\Inno Setup 6\ISCC.exe"
if (-not (Test-Path $iscc)) {
    throw "Inno Setup not found at $iscc. Download from https://jrsoftware.org/isinfo.php"
}
& $iscc installer.iss
if ($LASTEXITCODE -ne 0) { throw "Inno Setup failed" }

$output = "$Root\Output\ADNS_installer.exe"
if (Test-Path $output) {
    $size = [math]::Round((Get-Item $output).Length / 1MB, 1)
    Write-Host "`nDone! $output ($size MB)" -ForegroundColor Green
} else {
    throw "Installer not found at expected path: $output"
}
