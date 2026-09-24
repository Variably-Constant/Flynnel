# Wait for the box, announce the run, and release on every path out.
#
# Dot-sourced by each timing harness so the four of them cannot drift
# apart on the part that affects other agents. pc2 is shared: one timing
# job at a time is a constraint on the box, not on one agent's jobs.
#
#   . C:\Temp\pc2_timing_guard.ps1
#   if (-not (Enter-TimingRun -What "what this measures" -Log $log `
#             -Tree 'C:\Temp\the-tree-being-measured' -Who 'Your-Name')) { exit 3 }
#   ... timed work ...
#   Exit-TimingRun -Log $log
#
# -Tree is the tree whose commit the log records. It is read with git -C, so
# a harness that sets no location still gets its own provenance rather than
# the dot-sourcing directory's; a tree that gives no head is refused, since a
# provenance line holding a git error attributes the numbers to nothing.
#
# -Who is the agent the presence claim names, so an operator reading the
# presence list knows whose run is holding the box.
#
# A run that is not timing anything - a build, a bake, a correctness gate -
# holds the box just as long and declares itself through the other pair:
#
#   . C:\Temp\pc2_timing_guard.ps1
#   if (-not (Enter-BoxRun -What "what this run is" -Log $log `
#             -Who 'Your-Name')) { exit 3 }
#   ... the work ...
#   Exit-BoxRun -Log $log
#
# Enter-TimingRun calls Enter-BoxRun itself, so a timing run declares once and
# a caller must not call both. Neither takes the measurement lease, and
# neither asks for it unless the caller passes -RequireLease $true. That lease
# is C:\Temp\pc2_lease.py, whose holder must be the parent of the work, so a
# launch that uses it wraps the runner rather than the runner taking it -
#
#   schtasks /run -> python pc2_lease.py run --who '<you>' -- <the runner>
#
# and with -RequireLease $true both entry points refuse a span that is not
# under it. Each refusal writes its own line, so NO_LEASE, NO_PROVENANCE and
# NOT_QUIET are told apart by a caller that only sees the exit code; the
# BOX_LEASE line records the lease's state for every span either way.
#
# -BoundOn says what the run's wall is bound on and picks the CPU ceiling from
# it: host, the default, waits for a box quiet enough that its cores are not
# deciding the answer; device waits only for a box that is not saturated,
# because background CPU does not reach a wall read off device clocks. A
# caller passing -MaxIdleCores gets that number and -BoundOn does not apply.
#
# Exit-TimingRun must also be called from the harness's trap. It is safe
# to call when no claim is held and safe to call twice.

$script:TimingClaimHeld = $false
$script:BoxClaimHeld = $false

# Whether this process is running under C:\Temp\pc2_lease.py's measurement
# lease: true, false, or nothing where the question could not be answered.
#
# Three answers rather than two. A check that reads "I could not tell" as
# "yes" lets an undeclared span run believing it is serialized, and one that
# reads it as "no" refuses a run that is correctly wrapped; the caller chooses
# which of those it would rather have.
#
# The lease sets no environment variable on its child, so the relation is read
# from the process tree: its own status verb names the holding pid, and this
# process is under it when that pid is one of its ancestors. Ancestors rather
# than the parent, because the holder is the parent of the runner and the
# runner is the parent of whatever dot-sources this file.
function Test-UnderMeasurementLease {
    try {
        $status = & python 'C:\Temp\pc2_lease.py' status 2>&1 | Out-String
    } catch {
        return $null
    }
    # Free is tested first and the holder is matched on its whole phrase,
    # because the same output lists the queue behind the lease as "1. pid N"
    # and a looser pattern would read the first waiter as the holder.
    if ($status -match 'the lease is free') {
        return $false
    }
    if ($status -match 'held by pid (\d+)') {
        $held = [int]$Matches[1]
    } else {
        return $null
    }
    $walk = $PID
    for ($hops = 0; $hops -lt 12; $hops++) {
        if ($walk -eq $held) { return $true }
        $proc = Get-CimInstance Win32_Process -Filter "ProcessId = $walk" -ErrorAction SilentlyContinue
        if (-not $proc -or -not $proc.ParentProcessId -or $proc.ParentProcessId -eq 0) { return $false }
        $walk = [int]$proc.ParentProcessId
    }
    return $null
}

# Declare a span on the box, without waiting for it to be quiet.
#
# A build, a bake or a correctness gate holds the box as long as a timing run
# and has had nothing to announce itself with, so a neighbour reads occupancy
# off CPU and cannot tell a gap between two phases from an ending. This writes
# the span where a neighbour reads it, carrying what the run is for and whom
# to ask about it.
#
# It does not take the measurement lease. pc2_lease.py holds that for exactly
# the life of its child and requires the holder to be the child's parent,
# which a function dot-sourced into the work can never be; its header forbids
# hand-rolling the take and release for that reason. So with -RequireLease
# $true this refuses unless it is already under the lease, which turns the
# nesting that file requires - the task, then the lease, then the work - into
# something checked rather than hoped for.
function Enter-BoxRun {
    param(
        [Parameter(Mandatory = $true)][string]$What,
        [Parameter(Mandatory = $true)][string]$Log,
        # The agent whose span this is, so the presence line says who to ask.
        [string]$Who = 'an agent that did not name itself',
        [int]$OwnerPid = $PID,
        # Whether a span that is not under the measurement lease is refused.
        # Only a caller whose launch is wrapped in the lease passes true. The
        # lease's state reaches the log either way, so an unserialized span is
        # never silent.
        [bool]$RequireLease = $false
    )
    $under = Test-UnderMeasurementLease
    $said = if ($null -eq $under) { 'could not be read' } elseif ($under) { 'held' } else { 'not held' }
    "BOX_LEASE $said at $(Get-Date -Format o)" | Add-Content -Path $Log
    if ($RequireLease -and $under -ne $true) {
        ("NO_LEASE this span is not under C:\Temp\pc2_lease.py, which is the only registry that " +
         "serializes; wrap the launch as: python C:\Temp\pc2_lease.py run --who '<you>' -- <command>") |
            Add-Content -Path $Log
        return $false
    }
    $claim = "$What, and this line stands until it exits, $Who"
    & powershell -NoProfile -ExecutionPolicy Bypass -File 'C:\Temp\presence.ps1' `
        -Claim $claim -OwnerPid $OwnerPid *>&1 | Add-Content -Path $Log
    $script:BoxClaimHeld = $true
    return $true
}

# Withdraw the declaration. Safe when none is held and safe to call twice,
# which a harness needs because its trap and its normal path both reach here
# on some exit orders.
function Exit-BoxRun {
    param(
        [Parameter(Mandatory = $true)][string]$Log,
        [int]$OwnerPid = $PID
    )
    if (-not $script:BoxClaimHeld) { return }
    & powershell -NoProfile -ExecutionPolicy Bypass -File 'C:\Temp\presence.ps1' `
        -Release -OwnerPid $OwnerPid *>&1 | Add-Content -Path $Log
    $script:BoxClaimHeld = $false
}

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
    # The idle fraction exceeds one by a sliver when the two samples
    # straddle a counter update, and the busier the box the less room
    # there is for that to happen - so a quiet box is where it shows.
    # Busy cores cannot be negative. Floored at zero rather than
    # returned negative, because a reader treating a negative as a
    # failed sample discards the quietest trials it has: 28 of 136
    # readings on one 40-trial rotation came back between -0.24 and
    # -0.40 while only 7 were genuinely above the gate.
    #
    # -1 stays the failed read, returned above when the interval carried
    # no ticks at all.
    $busy = [int]$env:NUMBER_OF_PROCESSORS * (1 - $idle)
    [Math]::Round([Math]::Max(0, $busy), 2)
}

function Enter-TimingRun {
    param(
        [Parameter(Mandatory = $true)][string]$What,
        [Parameter(Mandatory = $true)][string]$Log,
        # The tree whose commit this log records. Read with git -C, so the
        # provenance belongs to the tree being measured rather than to
        # whatever directory the harness happened to be dot-sourced from.
        # The default is that directory, which is what a harness that sets
        # its location already gets; one that does not is refused below
        # rather than given a git error where a commit should be.
        [string]$Tree = (Get-Location).Path,
        # The agent whose run this is, so the presence line says who to ask
        # about it. A claim is worth keeping beside a measurement because it
        # carries intent, and intent belongs to somebody.
        [string]$Who = 'an agent that did not name itself',
        # The caller's own pid. Defaults to this session's, which is the
        # harness when this file is dot-sourced into it.
        [int]$OwnerPid = $PID,
        # What the run's wall is bound on, which is what decides how much
        # background CPU it can carry. A host-bound run reads a wall the box's
        # cores set. A device-bound one reads device clocks plus a short host
        # tail, and background CPU does not reach it.
        [ValidateSet('host', 'device')][string]$BoundOn = 'host',
        # Below zero takes the ceiling from -BoundOn. A caller naming a number
        # gets that number.
        [double]$MaxIdleCores = -1,
        # Ceiling on the wait. awaitquiet's own default is 180, which is
        # right for a run nobody is watching and wrong for a smoke test.
        [double]$WaitMinutes = 180,
        # Passed to [Enter-BoxRun]: whether a run that is not under the
        # measurement lease is refused. Only a caller whose launch is wrapped
        # in the lease passes true.
        [bool]$RequireLease = $false
    )

    if ($MaxIdleCores -lt 0) {
        $MaxIdleCores = if ($BoundOn -eq 'device') { 20.0 } else { 1.2 }
    }

    # What was actually measured, not what the tree is called.
    #
    # These trees are named for the commit they were cloned at and then
    # have files pushed into them, so the name goes stale the first time
    # a source file moves. A log naming only the path attributes its
    # numbers to a commit the tree no longer holds. HEAD plus the
    # modified paths says what the run really built.
    # Read against the named tree rather than the current directory, and
    # refused when no head comes back. A log whose provenance line holds a
    # git error attributes its numbers to nothing, and every reader after
    # it has to take the tree's name on trust - which is the one thing the
    # name cannot be trusted for.
    $head = (& git -C $Tree rev-parse --short HEAD 2>&1 | Out-String).Trim()
    if ($head -notmatch '^[0-9a-f]{7,40}$') {
        "NO_PROVENANCE $Tree gave no head ($head); declining to measure rather than measure unattributably" |
            Add-Content -Path $Log
        return $false
    }
    $dirty = @(& git -C $Tree status --porcelain --untracked-files=no 2>&1 |
        ForEach-Object { $_.ToString().Trim() })
    "SOURCE_STATE head=$head dirty=$($dirty.Count) in $Tree" |
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
    #
    # A device-bound run takes 20.0 instead, which is where a load ladder
    # stopped rather than where its numbers moved. Over rungs of 0, 1, 2, 4,
    # 8, 12, 16 and 20 delivered cores and back down, 100 passes a rung, every
    # one of the wave calibration's nine figures stayed inside the spread its
    # unloaded rung produced, with the box at 22.05 of 24 busy at the top; the
    # narrow lane's root step read 40960 ns on all 100 passes of every rung.
    # Nothing is measured above 20 delivered cores, which is why the ceiling
    # sits there and not higher.
    $waitLog = [System.IO.Path]::ChangeExtension($Log, 'awaitquiet.log')
    "AWAITING_QUIET $(Get-Date -Format o) bound_on=$BoundOn max_idle_cores=$MaxIdleCores progress in $waitLog" |
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

    # One declaration per span, made through the call a build makes, so a
    # timing run appears in the presence record once rather than twice and a
    # caller adding Enter-BoxRun beside this would not double it.
    if (-not (Enter-BoxRun -What "$What, TIMINGS, needs a quiet box" -Log $Log `
              -Who $Who -OwnerPid $OwnerPid -RequireLease $RequireLease)) {
        return $false
    }
    $script:TimingClaimHeld = $true
    return $true
}

function Exit-TimingRun {
    param(
        [Parameter(Mandatory = $true)][string]$Log,
        [int]$OwnerPid = $PID
    )
    if (-not $script:TimingClaimHeld) { return }
    $script:TimingClaimHeld = $false
    Exit-BoxRun -Log $Log -OwnerPid $OwnerPid
}
