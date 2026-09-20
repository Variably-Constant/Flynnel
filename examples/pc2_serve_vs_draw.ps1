# Whether serving a confirmed record routes any worse than drawing one.
#
# Two arms of the same binary. "serve" points every process at one
# directory, so after two starts it reads a stored record. "draw" gives
# each process a fresh directory, so every start measures for itself.
# Nothing else differs.
#
# The shared directory is warmed before the trials and those runs are
# not counted. What the startup draw costs is a separate figure measured
# elsewhere; the question here is whether the thresholds a served record
# carries route worse than the ones a fresh draw would have produced,
# and a warmup inside the sample would answer a different question.
#
# Each process measures its own control window beside its loaded one and
# reports the ratio, so a trial that lands during a build storm moves
# both and largely cancels. The ratio is what the arms are compared on.
#
# Arm order alternates by trial. A window measured first in its trial is
# worth a few per cent more, and a fixed order would give that to one
# arm every time.

param(
    [Parameter(Mandatory = $true)][string]$Sha,
    [int]$Trials = 20,
    [int]$WindowS = 2,
    [int]$Load = 12,
    [int]$Reps = 512,
    # Ceiling on warm-up draws, not a target. The loop stops as soon as
    # the record serves; this only bounds a host whose draws never agree.
    [int]$Warmup = 8,
    # Suffixes the log so two runs against one tree do not interleave
    # into one file. A PowerShell pipeline into Add-Content holds the log
    # for the life of the child writing to it, so a second run sharing
    # the name produces a file neither run can read and a reader cannot
    # separate.
    [string]$Tag = ''
)

$ErrorActionPreference = 'Continue'
$PSDefaultParameterValues['Add-Content:Encoding'] = 'utf8'
$PSDefaultParameterValues['Out-File:Encoding'] = 'utf8'

# Busy cores over a short interval, counting every process on the box.
#
# A count of cargo, rustc, cl and link names reads near zero on a host
# saturated by anything that is not a compile: pc2 has been observed at
# 99 per cent CPU with that count at 1. A run claiming a quiet box has
# to measure the box, not a list of names it expects to find.
#
# PercentIdleTime on the _Total instance is a 100 ns counter, so the
# idle fraction is its delta over the timestamp delta, and busy cores is
# the core count times what is left.
function Get-BusyCores {
    $q = "SELECT PercentIdleTime,Timestamp_Sys100NS FROM Win32_PerfRawData_PerfOS_Processor WHERE Name='_Total'"
    $a = Get-CimInstance -Query $q
    Start-Sleep -Milliseconds 250
    $b = Get-CimInstance -Query $q
    $dt = $b.Timestamp_Sys100NS - $a.Timestamp_Sys100NS
    if ($dt -le 0) { return -1 }
    $idle = ($b.PercentIdleTime - $a.PercentIdleTime) / $dt
    $cores = [int]$env:NUMBER_OF_PROCESSORS
    # Floored at zero: the idle fraction exceeds one by a sliver when the
    # samples straddle a counter update, and a negative busy count is
    # read as a failed sample by every reader here, which discards the
    # quietest trials. -1 stays the failed read.
    #
    # The same function lives in pc2_timing_guard.ps1, which says it is
    # there so every harness reports one quantity. Two copies is one
    # more than that.
    [Math]::Round([Math]::Max(0, $cores * (1 - $idle)), 2)
}

$log = if ($Tag) { "C:\Temp\serve_vs_draw_${Sha}_$Tag.log" } else { "C:\Temp\serve_vs_draw_$Sha.log" }
$tree = "C:\Projects\Flynnel-l1-$Sha"

. C:\Temp\pc2_timing_guard.ps1

trap {
    "AB_ABORTED $($_.Exception.Message)" | Add-Content -Path $log
    Exit-TimingRun -Log $log
    exit 7
}

if (-not (Test-Path $tree)) {
    "MISSING_TREE $tree" | Out-File -FilePath $log
    exit 9
}

$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$shared = "C:\Temp\abserve_${Sha}_$stamp"
New-Item -ItemType Directory -Force -Path $shared | Out-Null

"AB_START $(Get-Date -Format o) tree=$tree trials=$Trials window=${WindowS}s load=$Load reps=$Reps shared=$shared" |
    Out-File -FilePath $log

Set-Location $tree
Remove-Item Env:\FLYNNEL_HOST_PROFILE_NS -ErrorAction SilentlyContinue
$env:FLYNNEL_OCCUPANCY = '1'

cargo build --release --example throughput_under_load *>&1 | Add-Content -Path $log
if ($LASTEXITCODE -ne 0) {
    "BUILD_FAILED $LASTEXITCODE" | Add-Content -Path $log
    exit 8
}

$exe = ".\target\release\examples\throughput_under_load.exe"

# The build is contention tolerant and ran above. Everything past here is
# timed, so the box has to be clear before it starts and the run has to
# be announced while it lasts.
if (-not (Enter-TimingRun -What "flynnel serve-vs-draw A/B, $Trials trials" -Log $log `
          -Tree $tree -Who 'Flynnel-Scholar')) {
    exit 3
}

# Bring the shared directory to the state the serve arm is about: a
# stored record another draw has agreed on.
#
# Warmed until the record actually serves rather than for a fixed count.
# Two draws confirm only if they agree, and a disagreeing pair leaves the
# record provisional, so a fixed two left the first trial drawing and the
# run short one serve of what it asserted.
$env:FLYNNEL_CALIBRATION_DIR = $shared
$warmed = $false
foreach ($w in 1..$Warmup) {
    "WARM $w" | Add-Content -Path $log
    & $exe $WindowS $Load 1 0 0 $Reps 0 indexed *>&1 | Add-Content -Path $log
    $tail = Get-Content $log | Select-String -Pattern 'READ from the stored record'
    if ($tail) { $warmed = $true; "WARM_CONFIRMED after $w draws" | Add-Content -Path $log; break }
}
if (-not $warmed) {
    "WARM_NEVER_CONFIRMED after $Warmup draws; the serve arm would draw like the other one" |
        Add-Content -Path $log
    Exit-TimingRun -Log $log
    exit 4
}

foreach ($t in 1..$Trials) {
    $order = if ($t % 2 -eq 1) { @('serve', 'draw') } else { @('draw', 'serve') }
    foreach ($arm in $order) {
        if ($arm -eq 'serve') {
            $env:FLYNNEL_CALIBRATION_DIR = $shared
        } else {
            $fresh = "C:\Temp\abdraw_${Sha}_${stamp}_t${t}"
            New-Item -ItemType Directory -Force -Path $fresh | Out-Null
            $env:FLYNNEL_CALIBRATION_DIR = $fresh
        }

        $foreign = @(Get-Process -Name cargo,rustc,cl,link -ErrorAction SilentlyContinue).Count
        $busy = Get-BusyCores
        "ARM $arm trial=$t position=$([array]::IndexOf($order, $arm) + 1) foreign=$foreign busy_cores=$busy" |
            Add-Content -Path $log

        & $exe $WindowS $Load 1 0 0 $Reps 0 indexed *>&1 | Add-Content -Path $log
        if ($LASTEXITCODE -ne 0) {
            "ARM_FAILED $arm trial=$t exit=$LASTEXITCODE" | Add-Content -Path $log
        }

        # Sampled again on the way out. The entry figure says the box was
        # clear when the window opened and nothing about the rest of it,
        # so a trial can begin quiet and be measured through a storm: one
        # run here cleared its gate and then read a median of 3.71 busy
        # cores across its arms, with a worst arm at all 24. A reader
        # takes the larger of the two.
        #
        # Settled first, because this arm's own load reads 15.67 cores
        # while it runs and the threads are not reclaimed the instant the
        # process returns. Without the pause the exit figure measures the
        # harness winding down and votes good trials out.
        Start-Sleep -Milliseconds 600
        "ARM_END $arm trial=$t busy_cores=$(Get-BusyCores)" | Add-Content -Path $log
    }
    Write-Output "trial $t/$Trials done"
}

$written = Get-Content $log
$arms = @($written | Select-String -Pattern '^ARM ').Count
$retained = @($written | Select-String -Pattern '^retained ').Count
$reads = @($written | Select-String -Pattern 'READ from the stored record').Count
"ROW_CENSUS arms=$arms retained=$retained reads=$reads expected=$($Trials * 2)" |
    Add-Content -Path $log
"AB_DONE $(Get-Date -Format o)" | Add-Content -Path $log
Exit-TimingRun -Log $log

# The serve arm must actually have served. Without this the two arms are
# the same arm twice and the comparison reports a clean null about a
# mechanism that never ran.
if ($reads -lt $Trials) {
    "SERVE_ARM_DID_NOT_SERVE reads=$reads over $Trials trials" | Add-Content -Path $log
    exit 6
}
exit 0
