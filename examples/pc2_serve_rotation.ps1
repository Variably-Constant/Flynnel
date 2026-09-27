# What each serve policy costs in dispatch, not in start time.
#
# The start-cost half is measured: spread never admits so every process
# draws, any always admits, occupancy admits above its floor. Serving
# saves about 10 ms a start.
#
# That half cannot choose an arm. A served record's thresholds are not a
# fresh draw's, so an arm can save 10 ms at startup and route worse for
# the life of the process. This measures the second half.
#
# One binary throughout. The arms differ only in FLYNNEL_SERVE_POLICY,
# so a difference between them is the policy and not the build.
#
# The profile is left unpinned on purpose: pinning installs a fixed
# profile and draws nothing, which is the mechanism under test. Each arm
# gets a fresh calibration directory so its first process draws rather
# than inheriting another arm's record.
#
# Arm order rotates per trial. A window is worth a few percent more in
# one position, and three arms held in a fixed order would give the
# first position to one of them every time.

param(
    [Parameter(Mandatory = $true)][string]$Tip,
    [int]$WindowS = 3,
    [int]$Trials = 6,
    [int]$Load = 12,
    [int]$Reps = 512,
    [string]$Entry = 'indexed',
    # One comma-separated string, split here, rather than a string[].
    # An array parameter crossing the ssh path and the parameter binder
    # arrives as a single element holding the whole comma-joined text.
    # That value then went into FLYNNEL_SERVE_POLICY, the lever refused
    # it as unknown and ran the shipped arm, and twelve trials measured
    # the default while the log's header claimed five arms.
    [string]$Arms = 'spread,any,occupancy',
    # Comma-separated values for FLYNNEL_LEVER_CALIBRATION_REFUSAL,
    # crossed with the arms above. One value leaves it out of the
    # comparison, which is the usual case.
    #
    # Crossing rather than a separate run because the refusal gates
    # prefers_incumbent, which only runs where a record is offered
    # against an incumbent. Under a serve policy that admits nothing the
    # comparison is never reached, so rotating the refusal alone would
    # give two arms of identical rows and read as a clean null.
    [string]$Refusal = '1'
)

$ErrorActionPreference = 'Continue'
$PSDefaultParameterValues['Add-Content:Encoding'] = 'utf8'
$PSDefaultParameterValues['Out-File:Encoding'] = 'utf8'

$log = "C:\Temp\serve_rotation_$Tip.log"
$tree = "C:\Projects\Flynnel-l1-$Tip"

. C:\Temp\pc2_timing_guard.ps1

trap {
    "ROTATION_ABORTED $($_.Exception.Message)" | Add-Content -Path $log
    Exit-TimingRun -Log $log
    exit 7
}

if (-not (Test-Path $tree)) {
    "MISSING_TREE $tree" | Out-File -FilePath $log
    exit 9
}

$policies = @($Arms -split ',' | ForEach-Object { $_.Trim() } | Where-Object { $_ })
$refusals = @($Refusal -split ',' | ForEach-Object { $_.Trim() } | Where-Object { $_ })
if ($policies.Count -eq 0 -or $refusals.Count -eq 0) {
    "NO_ARMS policies=$Arms refusal=$Refusal split to nothing" | Out-File -FilePath $log
    exit 9
}

# One arm per combination, named so the log says which pair produced a
# row. A single refusal value leaves the name as the policy alone, so
# logs from runs that did not cross the two still read the same.
$armList = @()
foreach ($p in $policies) {
    foreach ($r in $refusals) {
        $armList += if ($refusals.Count -eq 1) { $p } else { "$p/refusal=$r" }
    }
}

$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
# The split count is printed, not the raw string. A header echoing what
# was asked for cannot show that the split produced one arm, which is
# the failure it is here to make visible.
"ROTATION_START $(Get-Date -Format o) tip=$Tip window=${WindowS}s trials=$Trials load=$Load reps=$Reps entry=$Entry arm_count=$($armList.Count) arms=$($armList -join '|')" |
    Out-File -FilePath $log

Set-Location $tree
Remove-Item Env:\FLYNNEL_HOST_PROFILE_NS -ErrorAction SilentlyContinue

cargo build --release --example throughput_under_load *>&1 | Add-Content -Path $log
if ($LASTEXITCODE -ne 0) {
    "BUILD_FAILED $LASTEXITCODE" | Add-Content -Path $log
    exit 8
}

if (-not (Enter-TimingRun -What "flynnel serve-policy rotation, $($armList.Count) arms over $Trials trials" -Log $log)) {
    exit 3
}

foreach ($t in 1..$Trials) {
    # Rotate rather than reverse: reversing leaves a middle arm where it
    # was, and with three arms every one must take every position.
    $shift = ($t - 1) % $armList.Count
    $order = @()
    foreach ($i in 0..($armList.Count - 1)) { $order += $armList[($i + $shift) % $armList.Count] }

    foreach ($arm in $order) {
        # The arm names a directory and spread:<n> carries a colon,
        # which Windows will not take in a path. Substituted only for
        # the directory; the arm goes to the process and the log as
        # written, so a reader never sees a name the run did not use.
        $safe = $arm -replace '[:/=]', '-'
        $dir = "C:\Temp\servecal_${stamp}_${safe}_t$t"
        New-Item -ItemType Directory -Force -Path $dir | Out-Null
        $env:FLYNNEL_CALIBRATION_DIR = $dir

        # The arm name carries both settings where they are crossed.
        # Split back rather than tracking them separately: one name in
        # the log and one name in the environment cannot drift apart.
        if ($arm -match '^(.+)/refusal=(.+)$') {
            $env:FLYNNEL_SERVE_POLICY = $Matches[1]
            $env:FLYNNEL_LEVER_CALIBRATION_REFUSAL = $Matches[2]
        } else {
            $env:FLYNNEL_SERVE_POLICY = $arm
            Remove-Item Env:\FLYNNEL_LEVER_CALIBRATION_REFUSAL -ErrorAction SilentlyContinue
        }

        $foreign = @(Get-Process -Name cargo,rustc,cl,link -ErrorAction SilentlyContinue).Count
        "ARM $arm trial=$t position=$([array]::IndexOf($order, $arm) + 1) foreign=$foreign" |
            Add-Content -Path $log

        & ".\target\release\examples\throughput_under_load.exe" `
            $WindowS $Load 1 0 0 $Reps 0 $Entry *>&1 | Add-Content -Path $log
        if ($LASTEXITCODE -ne 0) {
            "ARM_FAILED $arm trial=$t exit=$LASTEXITCODE" | Add-Content -Path $log
        }
    }
    Write-Output "trial $t/$Trials done"
}

$written = Get-Content $log
$retained = @($written | Select-String -Pattern '^retained ').Count
$arms = @($written | Select-String -Pattern '^ARM ').Count
"ROW_CENSUS arms=$arms retained=$retained expected=$($armList.Count * $Trials)" |
    Add-Content -Path $log
"ROTATION_DONE $(Get-Date -Format o)" | Add-Content -Path $log
Exit-TimingRun -Log $log

if ($retained -eq 0) {
    "NO_RETAINED_ROWS nothing here parses as a measurement" | Add-Content -Path $log
    exit 6
}
exit 0
