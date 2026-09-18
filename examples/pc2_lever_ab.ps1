# One lever on against the same code with it off.
#
# Two arms differing in a single environment variable, alternating which
# goes first, each process measuring its own control window beside its
# loaded one. Writes the ARM / ARM_END / control / throughput / retained
# lines the paired report reads, so a lever rotation and a serve rotation
# are read by the same tool.
#
# Every process gets a FRESH calibration directory. A shared one would
# let a record drawn under one arm serve the other, which makes the
# thresholds a confound rather than a constant.
#
# The lever is read once per process and cached, so an arm is a process
# rather than a window. That is why the arms are separate runs of the
# binary and not two windows inside one.

param(
    [Parameter(Mandatory = $true)][string]$Sha,
    [Parameter(Mandatory = $true)][string]$Lever,
    [int]$Trials = 40,
    [int]$WindowS = 2,
    [int]$Load = 12,
    [int]$Reps = 512,
    [string]$Entry = 'indexed',
    # Per-item work and whether it varies. A lever that improves a
    # spread is not consulted where classify_observed returns on the
    # mean alone, and uniform work has a spread of zero, so the light
    # uniform point prices the instrument in a regime that never reads
    # what it produces.
    [int]$Irregular = 0,
    # Burners spin and sleep in phase rather than burning throughout.
    # A lever that weighs a batch by its on-core share reads nothing
    # under steady contention, because a share common to every batch
    # divides out of a weighted total over a weighted count. Alternating
    # puts contended and quiet batches in one window.
    [int]$DutyMs = 0,
    [double]$SettleCores = 1.4,
    [int]$SettleTries = 12,
    [string]$Tag = ''
)

$ErrorActionPreference = 'Continue'
$PSDefaultParameterValues['Add-Content:Encoding'] = 'utf8'
$PSDefaultParameterValues['Out-File:Encoding'] = 'utf8'

$short = ($Lever -replace '^FLYNNEL_LEVER_', '').ToLower()
$log = if ($Tag) { "C:\Temp\lever_${short}_${Sha}_$Tag.log" } else { "C:\Temp\lever_${short}_$Sha.log" }
$tree = "C:\Projects\Flynnel-l1-$Sha"

. C:\Temp\pc2_timing_guard.ps1

trap {
    "LEVER_ABORTED $($_.Exception.Message)" | Add-Content -Path $log
    Exit-TimingRun -Log $log
    exit 7
}

if (-not (Test-Path $tree)) {
    "MISSING_TREE $tree" | Out-File -FilePath $log
    exit 9
}

$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
"LEVER_START $(Get-Date -Format o) tree=$tree lever=$Lever trials=$Trials window=${WindowS}s load=$Load reps=$Reps entry=$Entry" |
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

if (-not (Enter-TimingRun -What "flynnel lever $short, $Trials trials" -Log $log)) {
    exit 3
}

foreach ($t in 1..$Trials) {
    $order = if ($t % 2 -eq 1) { @('1', '0') } else { @('0', '1') }
    foreach ($arm in $order) {
        $dir = "C:\Temp\levercal_${short}_${stamp}_t${t}_$arm"
        New-Item -ItemType Directory -Force -Path $dir | Out-Null
        $env:FLYNNEL_CALIBRATION_DIR = $dir
        Set-Item -Path "Env:\$Lever" -Value $arm

        $settle = 0
        do {
            Start-Sleep -Milliseconds 700
            $busy = Get-BusyCores
            $settle++
        } while ($busy -gt $SettleCores -and $settle -lt $SettleTries)

        $foreign = @(Get-Process -Name cargo,rustc,cl,link -ErrorAction SilentlyContinue).Count
        "ARM $short=$arm trial=$t position=$([array]::IndexOf($order, $arm) + 1) foreign=$foreign busy_cores=$busy" |
            Add-Content -Path $log

        & $exe $WindowS $Load 1 0 $DutyMs $Reps $Irregular $Entry *>&1 | Add-Content -Path $log
        if ($LASTEXITCODE -ne 0) {
            "ARM_FAILED $short=$arm trial=$t exit=$LASTEXITCODE" | Add-Content -Path $log
        }

        # Settled before the exit reading so it measures the box rather
        # than this arm's own threads being reclaimed.
        Start-Sleep -Milliseconds 600
        "ARM_END $short=$arm trial=$t busy_cores=$(Get-BusyCores)" | Add-Content -Path $log
    }
    Write-Output "trial $t/$Trials done"
}

$written = Get-Content $log
$arms = @($written | Select-String -Pattern '^ARM ').Count
$retained = @($written | Select-String -Pattern '^retained ').Count
$failed = @($written | Select-String -Pattern '^ARM_FAILED ').Count

# The binary refuses when a lever is on and its counter never moved, so
# a failed arm is the engagement assertion firing rather than a crash.
# Counted here so the log states it instead of leaving a reader to infer
# it from an exit code nobody kept.
$engaged = @($written | Select-String -Pattern 'oncore_items=[1-9]').Count
"ROW_CENSUS arms=$arms retained=$retained failed=$failed engaged_rows=$engaged expected=$($Trials * 2)" |
    Add-Content -Path $log
"LEVER_DONE $(Get-Date -Format o)" | Add-Content -Path $log
Exit-TimingRun -Log $log

if ($retained -eq 0) {
    "NO_RETAINED_ROWS nothing here parses as a measurement" | Add-Content -Path $log
    exit 6
}
if ($failed -gt 0) {
    "ARMS_REFUSED $failed of $arms; a lever that did not engage cannot be compared" |
        Add-Content -Path $log
    exit 5
}
exit 0
