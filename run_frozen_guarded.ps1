#!/usr/bin/env pwsh
<#
.SYNOPSIS
    S1 safety guard for frozen NFStream exe tests.

.DESCRIPTION
    Launches the frozen exe, polls the child-process tree every $PollSec seconds,
    and kills the entire tree (taskkill /T /F) if:
      - child count exceeds $MaxChildren  (fork-bomb guard)
      - wall time exceeds $TimeoutSec

    Stdout/stderr are captured to <exe>.stdout.log / <exe>.stderr.log and
    echoed to the console at the end.

    Kill-tree helper (copy-paste to terminate manually):
        taskkill /T /F /PID <rootPid>

.PARAMETER Exe
    Path to the frozen exe. Default: dist_test\nfstream_pkg_test\nfstream_pkg_test.exe

.PARAMETER ExeArgs
    Arguments forwarded to the exe.

.PARAMETER MaxChildren
    Kill the tree if child-process count exceeds this.  Default: 12

.PARAMETER TimeoutSec
    Kill the tree if wall time exceeds this.  Default: 180

.PARAMETER PollSec
    How often to sample the child count.  Default: 3

.EXAMPLE
    # pcap mode (STEP 3)
    .\run_frozen_guarded.ps1

    # live mode (STEP 4) — 90 s capture
    .\run_frozen_guarded.ps1 -ExeArgs @("--mode","live","--duration","90") -TimeoutSec 150
#>
param(
    [string]  $Exe         = "X:\ADNS\dist_test\nfstream_pkg_test\nfstream_pkg_test.exe",
    [string[]]$ExeArgs     = @(),
    [int]     $MaxChildren = 12,
    [int]     $TimeoutSec  = 180,
    [int]     $PollSec     = 3
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Continue"

# ── helpers ────────────────────────────────────────────────────────────────────

function Get-Descendants {
    param([int]$RootPid)
    $snapshot = @(Get-CimInstance Win32_Process -Property ProcessId,ParentProcessId `
                  -ErrorAction SilentlyContinue)
    $queue  = [System.Collections.Generic.Queue[int]]::new()
    $seen   = [System.Collections.Generic.HashSet[int]]::new()
    $result = [System.Collections.Generic.List[int]]::new()
    $queue.Enqueue($RootPid)
    while ($queue.Count -gt 0) {
        $p = $queue.Dequeue()
        if ($seen.Add($p)) {
            foreach ($child in ($snapshot | Where-Object { $_.ParentProcessId -eq $p })) {
                $queue.Enqueue([int]$child.ProcessId)
            }
        }
    }
    $seen.Remove($RootPid) | Out-Null
    foreach ($id in $seen) { $result.Add([int]$id) }
    # Always return an unambiguous int array so .Count is always valid
    return ,[int[]]$result.ToArray()
}

# ── pre-flight ─────────────────────────────────────────────────────────────────

Write-Host ""
Write-Host "=== NFStream Frozen Exe Safety Guard ===" -ForegroundColor Cyan
Write-Host "  Exe     : $Exe"
Write-Host "  Args    : $($ExeArgs -join ' ')"
Write-Host "  Limits  : max_children=$MaxChildren  timeout=${TimeoutSec}s  poll=${PollSec}s"
Write-Host ""

if (-not (Test-Path $Exe)) {
    Write-Error "Exe not found: $Exe"
    exit 2
}

$outLog = Join-Path (Split-Path $Exe) "nfstream_pkg_test.stdout.log"
$errLog = Join-Path (Split-Path $Exe) "nfstream_pkg_test.stderr.log"

# ── launch ─────────────────────────────────────────────────────────────────────

$proc = Start-Process `
    -FilePath $Exe `
    -ArgumentList $ExeArgs `
    -RedirectStandardOutput $outLog `
    -RedirectStandardError  $errLog `
    -NoNewWindow `
    -PassThru

$rootPid = $proc.Id
Write-Host "  PID     : $rootPid"
Write-Host "  Kill cmd: taskkill /T /F /PID $rootPid" -ForegroundColor Yellow
Write-Host "  Stdout  : $outLog"
Write-Host "  Stderr  : $errLog"
Write-Host ""

# ── monitor loop ───────────────────────────────────────────────────────────────

$t0       = Get-Date
$peak     = 0
$killed   = $false
$timeline = [System.Collections.Generic.List[string]]::new()

while ($true) {
    Start-Sleep -Seconds $PollSec
    $elapsed = [int]((Get-Date) - $t0).TotalSeconds

    if ($proc.HasExited) {
        Write-Host "[${elapsed}s] Exited — code=$($proc.ExitCode)" -ForegroundColor Green
        break
    }

    if ($elapsed -ge $TimeoutSec) {
        Write-Host "[${elapsed}s] TIMEOUT (${TimeoutSec}s) — killing tree" -ForegroundColor Red
        & taskkill /T /F /PID $rootPid 2>$null
        $killed = $true
        break
    }

    $kids = @(Get-Descendants -RootPid $rootPid)
    $n    = [int]$kids.Count
    if ($n -gt $peak) { $peak = $n }
    $timeline.Add("${elapsed}s:$n")
    Write-Host "[${elapsed}s] children=$n  peak=$peak"

    if ($n -gt $MaxChildren) {
        Write-Host "[${elapsed}s] CHILD LIMIT ($n > $MaxChildren) — killing tree" -ForegroundColor Red
        & taskkill /T /F /PID $rootPid 2>$null
        $killed = $true
        break
    }
}

# ── orphan check ───────────────────────────────────────────────────────────────

Start-Sleep -Milliseconds 800
$orphanPids = @(Get-Descendants -RootPid $rootPid) |
    Where-Object { $null -ne (Get-Process -Id $_ -ErrorAction SilentlyContinue) }
$orphanCount = [int]@($orphanPids).Count
if ($orphanCount -gt 0) {
    Write-Host "Orphaned child PIDs: $($orphanPids -join ', ')" -ForegroundColor Red
    # Clean them up automatically
    foreach ($pid in $orphanPids) {
        & taskkill /F /PID $pid 2>$null | Out-Null
    }
    Write-Host "Orphans killed."
}

# ── report ─────────────────────────────────────────────────────────────────────

Write-Host ""
Write-Host "=== RESULTS ===" -ForegroundColor Cyan
Write-Host "  Peak child count  : $peak"
Write-Host "  Child timeline    : $($timeline -join ', ')"
Write-Host "  Orphans after exit: $orphanCount"
Write-Host "  Guard triggered   : $killed"
$exitCode = if ($proc.HasExited) { $proc.ExitCode } else { -1 }
Write-Host "  Exit code         : $exitCode"
Write-Host ""

Write-Host "--- stdout ---" -ForegroundColor DarkGray
if (Test-Path $outLog) { Get-Content $outLog } else { Write-Host "(no stdout log)" }
Write-Host ""
Write-Host "--- stderr ---" -ForegroundColor DarkGray
if (Test-Path $errLog) {
    $errContent = Get-Content $errLog
    if ($errContent) { $errContent } else { Write-Host "(empty)" }
} else { Write-Host "(no stderr log)" }
Write-Host ""

# Exit 1 if guard fired, orphans remain, or exe itself failed
$finalCode = if ($killed -or $orphanCount -gt 0 -or $exitCode -ne 0) { 1 } else { 0 }
Write-Host "Guard exit code: $finalCode" -ForegroundColor $(if ($finalCode -eq 0) { 'Green' } else { 'Red' })
exit $finalCode
