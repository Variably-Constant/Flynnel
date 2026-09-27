# What a calibration draw costs, and what the read that replaces it
# costs, over enough processes to see a distribution.
#
# The age bound makes a record expire, so a process whose record has
# aged out pays a draw instead of a read. The difference between the two
# is the bound's whole cost, and a doc comment asserted it was
# acceptable before anyone measured it.
#
# One draw per process against a fresh directory. A directory holding a
# record makes the first call a read, so a single process cannot yield
# two draws and a loop inside one would report a draw and some copies.
#
# The directories are stamped rather than deleted: each is the record
# that run drew, they cost a kilobyte, and nothing here needs the
# authority to remove a tree.

param(
    [Parameter(Mandatory = $true)][string]$Sha,
    [int]$Runs = 20
)

$ErrorActionPreference = 'Continue'
$PSDefaultParameterValues['Add-Content:Encoding'] = 'utf8'
$PSDefaultParameterValues['Out-File:Encoding'] = 'utf8'

$log = "C:\Temp\calibration_cost_$Sha.log"
$tree = "C:\Projects\Flynnel-l1-$Sha"

. C:\Temp\pc2_timing_guard.ps1

trap {
    "COST_ABORTED $($_.Exception.Message)" | Add-Content -Path $log
    Exit-TimingRun -Log $log
    exit 7
}

if (-not (Test-Path $tree)) {
    "MISSING_TREE $tree" | Out-File -FilePath $log
    exit 9
}

$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$root = "C:\Temp\costcal_${Sha}_$stamp"

"COST_START $(Get-Date -Format o) tree=$tree runs=$Runs root=$root" |
    Out-File -FilePath $log

Set-Location $tree
# A pin would install a profile and measure nothing, and the example
# refuses on it; cleared here so an inherited one does not end the run
# twenty times over.
Remove-Item Env:\FLYNNEL_HOST_PROFILE_NS -ErrorAction SilentlyContinue

# The spread and the occupancy the draw ran at, on stderr from the
# library itself. A draw that is not served back was refused on the
# spread against PROVISIONAL_SPREAD_PER_MILLE, and the profile the
# caller receives carries neither figure, so without this the log shows
# a refusal and cannot show what it refused.
$env:FLYNNEL_OCCUPANCY = '1'

cargo build --release --example calibration_cost *>&1 | Add-Content -Path $log
if ($LASTEXITCODE -ne 0) {
    "BUILD_FAILED $LASTEXITCODE" | Add-Content -Path $log
    exit 8
}

# The build is contention tolerant and ran above. Everything below is a
# draw being timed, so the box has to be clear and the run announced.
if (-not (Enter-TimingRun -What "flynnel calibration draw cost, $Runs processes" -Log $log)) {
    exit 3
}

$failed = 0
foreach ($i in 1..$Runs) {
    $dir = Join-Path $root "run_$i"
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    $env:FLYNNEL_CALIBRATION_DIR = $dir

    # What was on the host as this run starts. A draw measures the host,
    # so what else was on it decides what the draw cost, and a figure
    # without that beside it cannot be read as either a quiet cost or a
    # busy one.
    #
    # Busy cores beside the process count because the count names four
    # images and reads near zero on a box saturated by anything else:
    # observed at 99 percent CPU with that count at 1.
    $foreign = @(Get-Process -Name cargo,rustc,cl,link -ErrorAction SilentlyContinue).Count
    "RUN $i foreign=$foreign busy_cores=$(Get-BusyCores)" | Add-Content -Path $log

    & ".\target\release\examples\calibration_cost.exe" *>&1 | Add-Content -Path $log
    if ($LASTEXITCODE -ne 0) {
        "RUN_FAILED $i exit=$LASTEXITCODE" | Add-Content -Path $log
        $failed++
    }
    Write-Output "run $i/$Runs done"
}

$written = Get-Content $log
$rows = @($written | Select-String -Pattern '^calibration draw_ns=').Count
"ROW_CENSUS runs=$Runs rows=$rows failed=$failed" | Add-Content -Path $log
"COST_DONE $(Get-Date -Format o)" | Add-Content -Path $log
Exit-TimingRun -Log $log

if ($rows -eq 0) {
    "NO_COST_ROWS nothing in this log parses as a timing" | Add-Content -Path $log
    exit 6
}
exit $failed
