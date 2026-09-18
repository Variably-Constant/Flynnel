# Wait for the box, announce the run, and release on every path out.
#
# Dot-sourced by each timing harness so the four of them cannot drift
# apart on the part that affects other agents. pc2 is shared: one timing
# job at a time is a constraint on the box, not on one agent's jobs.
#
#   . C:\Temp\pc2_timing_guard.ps1
#   if (-not (Enter-TimingRun -What "what this measures" -Log $log)) { exit 3 }
#   ... timed work ...
#   Exit-TimingRun -Log $log
#
# Exit-TimingRun must also be called from the harness's trap. It is safe
# to call when no claim is held and safe to call twice.

$script:TimingClaimHeld = $false

# Busy cores over a short interval, counting every process on the box.
#
# Lives here so every harness reports the same quantity. A count of
# cargo, rustc, cl and link names reads near zero on a box saturated by
# anything that is not a compile, observed at 99 per cent CPU with that
# count at 1, and Win32_Processor LoadPercentage read 100 on the same
# instant this read 4.33.
#
# PercentIdleTime on the _Total instance is a 100 ns counter, so the
# idle fraction is its delta over the timestamp delta.
function Get-BusyCores {
    $q = "SELECT PercentIdleTime,Timestamp_Sys100NS FROM Win32_PerfRawData_PerfOS_Processor WHERE Name='_Total'"
    $a = Get-CimInstance -Query $q
    Start-Sleep -Milliseconds 250
    $b = Get-CimInstance -Query $q
    $dt = $b.Timestamp_Sys100NS - $a.Timestamp_Sys100NS
    if ($dt -le 0) { return -1 }
    $idle = ($b.PercentIdleTime - $a.PercentIdleTime) / $dt
    [Math]::Round([int]$env:NUMBER_OF_PROCESSORS * (1 - $idle), 2)
}

function Enter-TimingRun {
    param(
        [Parameter(Mandatory = $true)][string]$What,
        [Parameter(Mandatory = $true)][string]$Log,
        # The caller's own pid. Defaults to this session's, which is the
        # harness when this file is dot-sourced into it.
        [int]$OwnerPid = $PID,
        [double]$MaxIdleCores = 1.2,
        # Ceiling on the wait. awaitquiet's own default is 180, which is
        # right for a run nobody is watching and wrong for a smoke test.
        [double]$WaitMinutes = 180
    )

    # What was actually measured, not what the tree is called.
    #
    # These trees are named for the commit they were cloned at and then
    # have files pushed into them, so the name goes stale the first time
    # a source file moves. A log naming only the path attributes its
    # numbers to a commit the tree no longer holds. HEAD plus the
    # modified paths says what the run really built.
    $head = (& git rev-parse --short HEAD 2>&1 | Out-String).Trim()
    $dirty = @(& git status --porcelain --untracked-files=no 2>&1 |
        ForEach-Object { $_.ToString().Trim() })
    "SOURCE_STATE head=$head dirty=$($dirty.Count) in $(Get-Location)" |
        Add-Content -Path $Log
    foreach ($d in $dirty) { "SOURCE_DIRTY $d" | Add-Content -Path $Log }

    # awaitquiet reads the presence file, the process table and a total
    # CPU aggregate. -SelfRoot subtracts the caller's own subtree, since
    # a harness that loads the box on purpose would defer to itself.
    #
    # Its output goes to its own file by redirection rather than through
    # a pipeline into the run log. A pipeline is held until the child
    # exits, so a wait of any length shows nothing at all while it lasts
    # and a run deferring correctly cannot be told from one wedged on a
    # lock. Redirection writes as it goes.
    # -MaxIdleCores above this host's own floor. Sampled over five
    # seconds with nothing of mine running, the persistent consumers are
    # lql-server at 0.75 cores and System at 0.08, so a bare box sits
    # near 0.83 and awaitquiet's 0.5 default can never be met here. A
    # run left on the default waits out its whole timeout and then
    # declines to measure, which reads as a busy host rather than as an
    # unreachable threshold.
    #
    # 1.2 clears that floor and stays under the 1.4 a neighbour's gate
    # uses, so this never measures on a box busier than they would take.
    $waitLog = [System.IO.Path]::ChangeExtension($Log, 'awaitquiet.log')
    "AWAITING_QUIET $(Get-Date -Format o) max_idle_cores=$MaxIdleCores progress in $waitLog" |
        Add-Content -Path $Log
    & powershell -NoProfile -ExecutionPolicy Bypass -File 'C:\Temp\awaitquiet.ps1' `
        -SelfRoot $OwnerPid -MaxIdleCores $MaxIdleCores -Minutes $WaitMinutes > $waitLog 2>&1
    $quiet = ($LASTEXITCODE -eq 0)
    Get-Content $waitLog -ErrorAction SilentlyContinue | Select-Object -Last 4 |
        Add-Content -Path $Log
    if (-not $quiet) {
        "NOT_QUIET awaitquiet gave up; declining to measure rather than measure badly" |
            Add-Content -Path $Log
        return $false
    }
    "QUIET_REACHED $(Get-Date -Format o)" | Add-Content -Path $Log

    $claim = "$What, TIMINGS, needs a quiet box, and this line stands until it exits, Flynnel-Scholar"
    & powershell -NoProfile -ExecutionPolicy Bypass -File 'C:\Temp\presence.ps1' `
        -Claim $claim -OwnerPid $OwnerPid *>&1 | Add-Content -Path $Log
    $script:TimingClaimHeld = $true
    return $true
}

function Exit-TimingRun {
    param(
        [Parameter(Mandatory = $true)][string]$Log,
        [int]$OwnerPid = $PID
    )
    if (-not $script:TimingClaimHeld) { return }
    & powershell -NoProfile -ExecutionPolicy Bypass -File 'C:\Temp\presence.ps1' `
        -Release -OwnerPid $OwnerPid *>&1 | Add-Content -Path $Log
    $script:TimingClaimHeld = $false
}
