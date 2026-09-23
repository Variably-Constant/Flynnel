# The oversubscription reproducer in three arms, one process per arm:
# every park lever off, the join park that parks always, and the join
# park gated on the process being oversubscribed. The order of the arms
# rotates by round, so no arm always follows another. Every arm sets all
# three switches to 0 or 1 and never leaves one unset. The slot park is
# held at 0 in every arm: four arms on two hosts put it with the arm that
# had no switch on.
#
# Each process runs examples/oversubscribed_caller with its defaults: a
# quiet phase and a loaded phase alternating within the process, the
# parallel dispatch and the same work serial alternating within each
# phase. It prints its own summary and a JOIN_PARKS line carrying the
# switches it read, the parks and the long yields, which is what says
# whether an arm reached its mechanism.
#
# Runs under the measurement lease, from a wrapper that takes it, and
# announces in presence for its length. A child of this script still
# running at its deadline is stopped and reported as TIMEOUT: it is this
# script's own process, started here and nowhere else.
#
# Writes the lines the scratchpad's oversub_levers_report.py reads, as
# the Linux guest's runner does.
#
#   powershell -File pc2_oversub_levers.ps1 -Tree <dir> -Log <path> [-Rounds 3] [-Seconds 600]

param(
    [Parameter(Mandatory = $true)][string]$Tree,
    [Parameter(Mandatory = $true)][string]$Log,
    [int]$Rounds = 3,
    [int]$Seconds = 600
)

$ErrorActionPreference = 'Continue'
$PSDefaultParameterValues['Add-Content:Encoding'] = 'utf8'
function Say($text) { Add-Content -Path $Log -Value $text }

Set-Content -Path $Log -Encoding utf8 -Value ("OVERSUB_LEVERS_START " + (Get-Date -Format 'o') + " tree=$Tree rounds=$Rounds")
$exe = Join-Path $Tree 'target\release\examples\oversubscribed_caller.exe'
if (-not (Test-Path $exe)) { Say "NO_EXE $exe, so nothing was built to time"; exit 2 }
$head = (& git -C $Tree rev-parse --short HEAD 2>&1 | Out-String).Trim()
Say ("EXE $exe head=$head written=" + (Get-Item $exe).LastWriteTime.ToString('o'))

$arms = @(
    @{ Name = 'off';     Join = '0'; Oversub = '0' },
    @{ Name = 'join';    Join = '1'; Oversub = '0' },
    @{ Name = 'oversub'; Join = '0'; Oversub = '1' }
)

& C:\Temp\presence.ps1 -Claim "Flynnel park levers, oversubscribed_caller in three arms at $head, timing, 18 spinners in each process, Flynnel-Scholar" -OwnerPid $PID 2>&1 |
    ForEach-Object { Say $_ }
try {
    for ($r = 0; $r -lt $Rounds; $r++) {
        for ($k = 0; $k -lt $arms.Count; $k++) {
            $arm = $arms[($k + $r) % $arms.Count]
            $env:FLYNNEL_LEVER_JOIN_PARK = $arm.Join
            $env:FLYNNEL_LEVER_SLOT_PARK_NOW = '0'
            $env:FLYNNEL_LEVER_JOIN_PARK_OVERSUBSCRIBED = $arm.Oversub
            Say ("ARM $($arm.Name) round=$($r + 1) join_park=$($arm.Join) slot_park_now=0 " +
                 "join_park_oversubscribed=$($arm.Oversub) " + (Get-Date -Format 'HH:mm:ss'))
            $out = "C:\Temp\oversub_levers_$($arm.Name)_r$($r + 1).out"
            $err = "C:\Temp\oversub_levers_$($arm.Name)_r$($r + 1).err"
            $child = Start-Process -FilePath $exe -PassThru -WindowStyle Hidden `
                -RedirectStandardOutput $out -RedirectStandardError $err
            $null = $child.Handle
            $finished = $child.WaitForExit($Seconds * 1000)
            if (-not $finished) {
                Stop-Process -Id $child.Id -Force
                Say "TIMEOUT $($arm.Name) round=$($r + 1) pid=$($child.Id) after $Seconds s, stopped"
            }
            foreach ($file in @($out, $err)) {
                if (Test-Path $file) { Get-Content $file | ForEach-Object { Say ([string]$_) } }
            }
            $code = if ($finished) { $child.ExitCode } else { 'none' }
            Say ("ARM_END $($arm.Name) round=$($r + 1) exit=$code " + (Get-Date -Format 'HH:mm:ss'))
        }
    }
}
finally {
    Remove-Item Env:\FLYNNEL_LEVER_JOIN_PARK -ErrorAction SilentlyContinue
    Remove-Item Env:\FLYNNEL_LEVER_SLOT_PARK_NOW -ErrorAction SilentlyContinue
    Remove-Item Env:\FLYNNEL_LEVER_JOIN_PARK_OVERSUBSCRIBED -ErrorAction SilentlyContinue
    & C:\Temp\presence.ps1 -Release -OwnerPid $PID 2>&1 | ForEach-Object { Say $_ }
    Say ("OVERSUB_LEVERS_END " + (Get-Date -Format 'o'))
}
