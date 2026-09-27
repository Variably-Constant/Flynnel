# What the calibration refusal lever costs, on the shipped serve policy.
#
# The lever gates prefers_incumbent, which decides which of two records
# stands. With it on, the cheapest dispatch cost seen for a stamp
# survives. With it off, the most recent draw replaces whatever was
# there. The two arms therefore end holding different records, and what
# a later process routes on is the difference.
#
# Reaching it needs a fresh stamp directory per arm rather than a long
# run in one. On the shipped policy the comparison is reached at the
# SECOND start of a directory: the first stores a provisional record,
# the second draws and offers it against that incumbent, and from the
# third the record is confirmed and serves without offering anything.
#
# So each arm gets a fresh directory, several seeding processes to
# exercise the lever and settle a record, then measuring processes that
# serve it and report a control window beside a loaded one.
#
# Arm order alternates by trial, since a window measured first in its
# trial is worth a few percent more.

param(
    [Parameter(Mandatory = $true)][string]$Sha,
    [int]$Trials = 20,
    # Processes that draw and offer, per arm per trial. Three is the
    # smallest that reaches the comparison more than once: starts two
    # and three both offer only if the record has not confirmed, so this
    # is raised rather than assumed adequate and the census reports how
    # often the lever actually fired.
    [int]$Seeds = 3,
    [int]$WindowS = 2,
    [int]$Load = 12,
    [int]$Reps = 512,
    # Busy-core ceiling the box must fall under after the seeding runs
    # before an arm is measured, and how many polls to spend waiting.
    # The ceiling matches the one the report filters on, so a trial the
    # harness accepts is one the report will keep.
    [double]$SettleCores = 1.4,
    [int]$SettleTries = 12
)

$ErrorActionPreference = 'Continue'
$PSDefaultParameterValues['Add-Content:Encoding'] = 'utf8'
$PSDefaultParameterValues['Out-File:Encoding'] = 'utf8'

$log = "C:\Temp\refusal_ab_$Sha.log"
$tree = "C:\Projects\Flynnel-l1-$Sha"

. C:\Temp\pc2_timing_guard.ps1

trap {
    "REFUSAL_ABORTED $($_.Exception.Message)" | Add-Content -Path $log
    Exit-TimingRun -Log $log
    exit 7
}

if (-not (Test-Path $tree)) {
    "MISSING_TREE $tree" | Out-File -FilePath $log
    exit 9
}

$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
"REFUSAL_START $(Get-Date -Format o) tree=$tree trials=$Trials seeds=$Seeds window=${WindowS}s load=$Load" |
    Out-File -FilePath $log

Set-Location $tree
Remove-Item Env:\FLYNNEL_HOST_PROFILE_NS -ErrorAction SilentlyContinue
Remove-Item Env:\FLYNNEL_SERVE_POLICY -ErrorAction SilentlyContinue
$env:FLYNNEL_OCCUPANCY = '1'

cargo build --release --example throughput_under_load *>&1 | Add-Content -Path $log
if ($LASTEXITCODE -ne 0) {
    "BUILD_FAILED $LASTEXITCODE" | Add-Content -Path $log
    exit 8
}
$exe = ".\target\release\examples\throughput_under_load.exe"

if (-not (Enter-TimingRun -What "flynnel calibration-refusal A/B, $Trials trials" -Log $log)) {
    exit 3
}

foreach ($t in 1..$Trials) {
    $order = if ($t % 2 -eq 1) { @('1', '0') } else { @('0', '1') }
    foreach ($refusal in $order) {
        $dir = "C:\Temp\refcal_${stamp}_t${t}_r$refusal"
        New-Item -ItemType Directory -Force -Path $dir | Out-Null
        $env:FLYNNEL_CALIBRATION_DIR = $dir
        $env:FLYNNEL_LEVER_CALIBRATION_REFUSAL = $refusal

        # Seeding runs settle the record under this lever setting. Their
        # windows are written but carry no ARM line, so the report skips
        # them and only the measuring run is compared.
        foreach ($s in 1..$Seeds) {
            "SEED refusal=$refusal trial=$t n=$s" | Add-Content -Path $log
            & $exe 1 2 1 0 0 64 0 indexed *>&1 | Add-Content -Path $log
        }

        # The seeding runs above load the box and their threads are not
        # reclaimed the instant each process returns, so an arm sampled
        # straight after them reads the seeds rather than the host. A
        # first run of this harness kept 5 trials of 20 on that account.
        # Waited out rather than slept through: the box is polled until
        # it settles or the patience runs out, and the figure that
        # decided is written either way.
        $settle = 0
        do {
            Start-Sleep -Milliseconds 700
            $busy = Get-BusyCores
            $settle++
        } while ($busy -gt $SettleCores -and $settle -lt $SettleTries)
        "SETTLED after $settle polls at $busy busy cores" | Add-Content -Path $log

        $foreign = @(Get-Process -Name cargo,rustc,cl,link -ErrorAction SilentlyContinue).Count
        "ARM refusal=$refusal trial=$t position=$([array]::IndexOf($order, $refusal) + 1) foreign=$foreign busy_cores=$busy" |
            Add-Content -Path $log
        & $exe $WindowS $Load 1 0 0 $Reps 0 indexed *>&1 | Add-Content -Path $log
        if ($LASTEXITCODE -ne 0) {
            "ARM_FAILED refusal=$refusal trial=$t exit=$LASTEXITCODE" | Add-Content -Path $log
        }

        # The window is judged on the worse of its two ends: a quiet gate
        # is checked once at entry and a trial can begin clear and be
        # measured through a storm.
        #
        # Settled first, because this arm's own load reads about 15 cores
        # while it runs and the threads are not reclaimed the instant the
        # process returns. Without the pause the exit figure measures the
        # harness winding down and votes good trials out.
        Start-Sleep -Milliseconds 600
        "ARM_END refusal=$refusal trial=$t busy_cores=$(Get-BusyCores)" |
            Add-Content -Path $log
    }
    Write-Output "trial $t/$Trials done"
}

$written = Get-Content $log
$arms = @($written | Select-String -Pattern '^ARM ').Count
$kept = @($written | Select-String -Pattern 'the cheaper record stands').Count
$reads = @($written | Select-String -Pattern 'READ from the stored record').Count
"ROW_CENSUS arms=$arms kept=$kept reads=$reads expected_arms=$($Trials * 2)" |
    Add-Content -Path $log
"REFUSAL_DONE $(Get-Date -Format o)" | Add-Content -Path $log
Exit-TimingRun -Log $log

# The lever must have fired somewhere, and the measuring runs must have
# served something. Without both, the two arms are the same arm twice
# and the comparison reports a null about a switch that never moved.
if ($kept -eq 0) {
    "LEVER_NEVER_FIRED no record was ever kept over an offer" | Add-Content -Path $log
    exit 6
}
if ($reads -eq 0) {
    "NOTHING_SERVED the measuring runs all drew for themselves" | Add-Content -Path $log
    exit 5
}
exit 0
