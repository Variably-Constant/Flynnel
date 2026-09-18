# Whether a stored calibration ever reaches a later process.
#
# One directory across every run, which is what the store sees in
# production and what a fresh-directory probe cannot show: a draw is
# published for the next process, so a run that never shares a directory
# measures a first start over and over.
#
# The library says which of the two happened on stderr under
# FLYNNEL_OCCUPANCY. Counting those lines is the whole measurement: a
# READ is a record that served, a draw is one that did not, and a run
# where nothing serves prints no READ at all.

param(
    [Parameter(Mandatory = $true)][string]$Sha,
    [int]$Runs = 6,
    [int]$WindowS = 1,
    [int]$Load = 2,
    [int]$Reps = 64
)

$ErrorActionPreference = 'Continue'
$PSDefaultParameterValues['Add-Content:Encoding'] = 'utf8'
$PSDefaultParameterValues['Out-File:Encoding'] = 'utf8'

$log = "C:\Temp\serve_confirms_$Sha.log"
$tree = "C:\Projects\Flynnel-l1-$Sha"

. C:\Temp\pc2_timing_guard.ps1

trap {
    "CONFIRM_ABORTED $($_.Exception.Message)" | Add-Content -Path $log
    Exit-TimingRun -Log $log
    exit 7
}

if (-not (Test-Path $tree)) {
    "MISSING_TREE $tree" | Out-File -FilePath $log
    exit 9
}

$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$dir = "C:\Temp\confirmcal_${Sha}_$stamp"
New-Item -ItemType Directory -Force -Path $dir | Out-Null

"CONFIRM_START $(Get-Date -Format o) tree=$tree runs=$Runs dir=$dir" |
    Out-File -FilePath $log

Set-Location $tree
Remove-Item Env:\FLYNNEL_HOST_PROFILE_NS -ErrorAction SilentlyContinue
$env:FLYNNEL_OCCUPANCY = '1'
$env:FLYNNEL_CALIBRATION_DIR = $dir

cargo build --release --example throughput_under_load *>&1 | Add-Content -Path $log
if ($LASTEXITCODE -ne 0) {
    "BUILD_FAILED $LASTEXITCODE" | Add-Content -Path $log
    exit 8
}

if (-not (Enter-TimingRun -What "flynnel serve confirmation, $Runs processes on one directory" -Log $log)) {
    exit 3
}

foreach ($i in 1..$Runs) {
    $foreign = @(Get-Process -Name cargo,rustc,cl,link -ErrorAction SilentlyContinue).Count
    "RUN $i foreign=$foreign" | Add-Content -Path $log

    $t0 = Get-Date
    & ".\target\release\examples\throughput_under_load.exe" `
        $WindowS $Load 1 0 0 $Reps 0 indexed *>&1 | Add-Content -Path $log
    $ms = [int]((Get-Date) - $t0).TotalMilliseconds
    "RUN_WALL $i ms=$ms exit=$LASTEXITCODE" | Add-Content -Path $log
    Write-Output "run $i/$Runs done"
}

$written = Get-Content $log
$reads = @($written | Select-String -Pattern 'READ from the stored record').Count
$provisional = @($written | Select-String -Pattern 'stored and provisional').Count
$confirmed = @($written | Select-String -Pattern 'agreeing draw\(s\) and serves').Count
"ROW_CENSUS runs=$Runs reads=$reads provisional=$provisional confirmed=$confirmed" |
    Add-Content -Path $log
"CONFIRM_DONE $(Get-Date -Format o)" | Add-Content -Path $log
Exit-TimingRun -Log $log

# A run where nothing ever served is the state this is here to detect,
# so it exits non-zero rather than reporting a clean zero.
if ($reads -eq 0) {
    "NO_READS no process served a stored record" | Add-Content -Path $log
    exit 6
}
exit 0
