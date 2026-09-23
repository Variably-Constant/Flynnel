# The native chunk runner against the two ways the same work could be
# done without it: Lightfold's evaluator on the calling thread, and
# Flynnel's declared kernel driven through the engine.
#
# Three arms per cell, in one process with both modules loaded:
#
#   POOL    Invoke-LightfoldParallel -Plane Compiled with LIGHTFOLD_POOL
#           set to flynnel, which runs Lightfold's compiled body over
#           chunks through flynnel_run_chunks_v1 on Flynnel's pool
#   SERIAL  the same call with LIGHTFOLD_POOL set to inline, on the
#           calling thread
#   ENGINE  Invoke-FlynnelMap with the matching declared operation
#
# The five bodies are the ones whose answer is the same IEEE operation on
# both sides, per Lightfold's compiled plane and Flynnel's MapOp. Before
# anything is timed, every row's three answers are compared bit for bit,
# and the POOL and SERIAL arms are asked through -Stats which pool ran and
# what each phase cost, so a row cannot be fast because it computed
# something else or because the pool quietly declined.
#
# Lightfold reads LIGHTFOLD_POOL on every dispatch, so the variable is set
# between calls and never inside a timed body. Within a row the arms
# alternate call by call: each round runs every arm once, in an order that
# rotates by round and by row, and every timed call follows a full
# collection, so the heap and the box's drift land on every arm alike. A
# PowerShell-only control is timed at both ends of every row, which gives
# how far the process moved while the row was taken. The load pass repeats
# every row under burners on half the logical processors, with its own
# controls.
#
# The arms include what each route costs to hand its answer back.
# Invoke-FlynnelMap returns one array; whatever Invoke-LightfoldParallel
# writes is written and discarded inside the timed body, because that is
# what a caller pays.
#
# Under Windows PowerShell 5.1 the two modules share one Pwrs.Runtime and
# Lightfold declines the pool there, so the POOL arm is refused rather
# than timed as a second serial arm.
#
#   pwsh -NoProfile -File ChunkRunnerArms.ps1 -FlynnelModule <dir>
#        -LightfoldModule <dir> [-Sizes 1000,100000,1000000] [-Repeats 7]
#        [-Tag name] [-OutDir dir] [-LightfoldCommit id]

param(
    [Parameter(Mandatory)][string]$FlynnelModule,
    [Parameter(Mandatory)][string]$LightfoldModule,
    [int[]]$Sizes = @(1000, 100000, 1000000),
    [int]$Repeats = 7,
    # Seconds of rest after every timed call, so a pool's spinning workers
    # have parked before the next arm starts.
    [double]$Cooldown = 0.1,
    [string]$Tag = 'chunkrunner',
    [string]$OutDir = $PSScriptRoot,
    # The Lightfold commit the module was built from. Lightfold's tree is
    # not this one, so nothing here can read it; a run without it says so.
    [string]$LightfoldCommit = '',
    # Burners for the load pass: half the logical processors, the ratio
    # KernelShapes uses. Zero skips the load pass.
    [int]$LoadThreads = [int]([Environment]::ProcessorCount / 2)
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# ----------------------------------------------------------------------
# The burners, and the trap that stops them
# ----------------------------------------------------------------------
#
# First in the script because a trap covers its whole scope, including
# what runs before the trap statement: a failure earlier than the
# functions it calls would otherwise raise a second error inside the
# trap and lose the first.

$script:Burners = @()
# Read from the platform rather than $IsWindows, which Windows PowerShell
# does not define and which strict mode refuses to read unset.
$onWindows = [Environment]::OSVersion.Platform -eq [PlatformID]::Win32NT
$script:BurnerShell = if ($onWindows) { 'powershell' } else { 'pwsh' }
$script:BurnerHides = $onWindows

function Start-Burners {
    if ($LoadThreads -le 0) { return }
    $spin = 'while ($true) { $null = [Math]::Sqrt([Environment]::TickCount) }'
    $arguments = @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-Command', $spin)
    $script:Burners = 1..$LoadThreads | ForEach-Object {
        if ($script:BurnerHides) {
            Start-Process -FilePath $script:BurnerShell -ArgumentList $arguments -WindowStyle Hidden -PassThru
        } else {
            Start-Process -FilePath $script:BurnerShell -ArgumentList $arguments -PassThru
        }
    }
    Start-Sleep -Milliseconds 500
}

function Stop-Burners {
    foreach ($burner in $script:Burners) {
        if ($burner -and -not $burner.HasExited) {
            Stop-Process -Id $burner.Id -Force -ErrorAction SilentlyContinue
        }
    }
    $script:Burners = @()
}

# A burner left running holds the box against every other agent on it,
# so stopping them is bound to the script ending, not only to the happy
# path. The error is rethrown by value so its diagnostic survives.
trap {
    Stop-Burners
    Write-Host ("FAULT " + $_.Exception.GetType().Name + ": " + $_.Exception.Message)
    Write-Host ("FAULT_AT " + $_.InvocationInfo.PositionMessage)
    throw $_
}

# ----------------------------------------------------------------------
# The two modules, and what this host can run
# ----------------------------------------------------------------------

foreach ($pair in @(@('Flynnel', $FlynnelModule), @('Lightfold', $LightfoldModule))) {
    $manifest = Join-Path $pair[1] "$($pair[0]).psd1"
    if (-not (Test-Path $manifest)) { throw "no manifest at $manifest" }
    Import-Module $manifest -Force -ErrorAction Stop
}

$edition = $PSVersionTable.PSEdition
$poolPossible = $edition -eq 'Core'
if (-not $poolPossible) {
    $refusal = 'POOL ARM REFUSED on {0} {1}: the two modules share one Pwrs.Runtime here and ' +
        'Lightfold declines the pool, so a POOL row would be a second serial row.'
    Write-Host ($refusal -f $edition, $PSVersionTable.PSVersion)
}

# The commit of the tree the Flynnel module was built in, which is what the
# run timed, whichever tree this script itself was read from.
$commit = (& git -C $FlynnelModule rev-parse --short HEAD 2>&1 | Out-String).Trim()
if ($LASTEXITCODE -ne 0) {
    Write-Host "FLYNNEL COMMIT UNKNOWN: $FlynnelModule is not inside a git work tree ($commit)"
    $commit = $null
}
$scriptParam = (Get-Command Invoke-LightfoldParallel).Parameters['Script']
if (-not $scriptParam) { throw 'Invoke-LightfoldParallel has no -Script parameter' }
$scriptIsBlock = $scriptParam.ParameterType -eq [scriptblock]

$hostInfo = [PSCustomObject]@{
    Machine         = [Environment]::MachineName
    Edition         = $edition
    Version         = $PSVersionTable.PSVersion.ToString()
    ProcessorCount  = [Environment]::ProcessorCount
    FlynnelCommit   = $commit
    FlynnelModule   = (Resolve-Path $FlynnelModule).Path
    LightfoldModule = (Resolve-Path $LightfoldModule).Path
    LightfoldCommit = if ($LightfoldCommit) { $LightfoldCommit } else { $null }
}
if (-not $LightfoldCommit) {
    Write-Host 'LIGHTFOLD COMMIT UNKNOWN: -LightfoldCommit was not given, so this run cannot say which Lightfold build it timed.'
}

# ----------------------------------------------------------------------
# Timing, the control and the burners
# ----------------------------------------------------------------------

# A full collection, so every timed call starts from the same heap.
function Invoke-Collection {
    [System.GC]::Collect()
    [System.GC]::WaitForPendingFinalizers()
    [System.GC]::Collect()
}

# One timed call of a body, after a full collection.
function Measure-Once {
    param([Parameter(Mandatory)][scriptblock]$Body)
    Invoke-Collection
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $null = & $Body
    $sw.Stop()
    $sw.Elapsed.TotalMilliseconds
}

# The middle value, the lower of the two middle values for an even count.
function Get-Median {
    param([Parameter(Mandatory)][double[]]$Values)
    $sorted = [double[]]($Values | Sort-Object)
    $sorted[[int][Math]::Floor(($sorted.Length - 1) / 2)]
}

function Measure-Cell {
    param([Parameter(Mandatory)][scriptblock]$Body, [int]$Warmup = 2)
    for ($i = 0; $i -lt $Warmup; $i++) { $null = & $Body }
    $samples = New-Object double[] $Repeats
    for ($i = 0; $i -lt $Repeats; $i++) { $samples[$i] = Measure-Once -Body $Body }
    $sorted = [double[]]($samples | Sort-Object)
    [PSCustomObject]@{ MedianMs = (Get-Median $samples); MinMs = $sorted[0]; MaxMs = $sorted[-1] }
}

function Start-Cooldown {
    if ($Cooldown -gt 0) { Start-Sleep -Milliseconds ([int]($Cooldown * 1000)) }
}

# Arithmetic neither module can reach, the same at every row's two ends.
$controlBody = {
    $acc = 0.0
    for ($i = 1; $i -le 200000; $i++) { $acc += [Math]::Sqrt($i) }
    $acc
}

# ----------------------------------------------------------------------
# The bodies and their arms
# ----------------------------------------------------------------------

$bodies = @(
    @{ Name = 'Square';     Text = '$_ * $_';            Op = 'Square' }
    @{ Name = 'Sqrt';       Text = '[Math]::Sqrt($_)';   Op = 'Sqrt' }
    @{ Name = 'Abs';        Text = '[Math]::Abs($_)';    Op = 'Abs' }
    @{ Name = 'Negate';     Text = '-$_';                Op = 'Negate' }
    @{ Name = 'Reciprocal'; Text = '1 / $_';             Op = 'Reciprocal' }
)

# Inputs chosen so every body is defined on every element and none
# answers a special value: positive, away from zero, not integral.
function New-Input {
    param([int]$Count)
    $x = New-Object double[] $Count
    for ($i = 0; $i -lt $Count; $i++) { $x[$i] = 1.5 + ($i % 997) * 0.37 }
    , $x
}

function Get-Script {
    param([string]$Text)
    if ($scriptIsBlock) { [scriptblock]::Create($Text) } else { $Text }
}

# What a command wrote, as one double[]: a single array written whole
# and a stream of numbers written one by one both arrive as the same.
function ConvertTo-Doubles {
    param([object[]]$Written)
    $flat = if ($Written.Count -eq 1 -and $Written[0] -is [array]) { $Written[0] } else { $Written }
    , [double[]]$flat
}

# The answer an arm gives, as a double[], outside any timed body.
function Get-Answer {
    param([string]$Arm, [hashtable]$Body, [double[]]$X)
    switch ($Arm) {
        'POOL' {
            $env:LIGHTFOLD_POOL = 'flynnel'
            ConvertTo-Doubles @(Invoke-LightfoldParallel -Script (Get-Script $Body.Text) -Items $X -Plane Compiled)
        }
        'SERIAL' {
            $env:LIGHTFOLD_POOL = 'inline'
            ConvertTo-Doubles @(Invoke-LightfoldParallel -Script (Get-Script $Body.Text) -Items $X -Plane Compiled)
        }
        'ENGINE' {
            ConvertTo-Doubles @(Invoke-FlynnelMap -InputObject $X -Operation $Body.Op)
        }
    }
}

# A Lightfold arm's own report of one run: which pool evaluated the items
# and what each phase cost. The output is collected before it is searched,
# so the write phase is not charged for the search.
function Get-RunReport {
    param([string]$Arm, [hashtable]$Body, [double[]]$X)
    $env:LIGHTFOLD_POOL = if ($Arm -eq 'POOL') { 'flynnel' } else { 'inline' }
    Invoke-Collection
    $written = @(Invoke-LightfoldParallel -Script (Get-Script $Body.Text) -Items $X -Plane Compiled -Stats)
    $run = @($written.Where({ $_ -isnot [double] -and
                              ($_.PSObject.TypeNames -contains 'Lightfold.ParallelRun' -or
                               $_.GetType().FullName -eq 'Lightfold.ParallelRun') }))
    if ($run.Count -ne 1) { throw "$Arm $($Body.Name): -Stats gave $($run.Count) ParallelRun objects, not one" }
    $run[0]
}

# A reported time rounded for the table, or $null when the report carried
# none, so a phase that was not measured never reads as one that took no
# time.
function Get-Rounded {
    param($Value)
    if ($null -eq $Value) { $null } else { [Math]::Round([double]$Value, 3) }
}

# The phase times of one report.
function Get-Phases {
    param($Report)
    [PSCustomObject]@{
        SplitMs  = Get-Rounded $Report.SplitMs
        DecodeMs = Get-Rounded $Report.DecodeMs
        EvalMs   = Get-Rounded $Report.EvalMs
        WriteMs  = Get-Rounded $Report.WriteMs
        TotalMs  = Get-Rounded $Report.TotalMs
    }
}

function Test-SameBits {
    param([double[]]$A, [double[]]$B)
    if ($A.Length -ne $B.Length) { return $false }
    for ($i = 0; $i -lt $A.Length; $i++) {
        if ([BitConverter]::DoubleToInt64Bits($A[$i]) -ne [BitConverter]::DoubleToInt64Bits($B[$i])) {
            return $false
        }
    }
    $true
}

# The timed body of one arm. The variable is set before the body runs,
# so the body is the call alone.
function Get-ArmBody {
    param([string]$Arm, [hashtable]$Body, [double[]]$X)
    $bodyScript = Get-Script $Body.Text
    switch ($Arm) {
        'POOL'   { { Invoke-LightfoldParallel -Script $bodyScript -Items $X -Plane Compiled }.GetNewClosure() }
        'SERIAL' { { Invoke-LightfoldParallel -Script $bodyScript -Items $X -Plane Compiled }.GetNewClosure() }
        'ENGINE' { $op = $Body.Op; { Invoke-FlynnelMap -InputObject $X -Operation $op }.GetNewClosure() }
    }
}

function Set-ArmEnvironment {
    param([string]$Arm)
    if ($Arm -eq 'POOL') { $env:LIGHTFOLD_POOL = 'flynnel' }
    elseif ($Arm -eq 'SERIAL') { $env:LIGHTFOLD_POOL = 'inline' }
}

# One row under the current load. Every round runs each arm once, in an
# order that rotates by round and by row, and the control is timed at both
# ends of the row.
function Measure-Row {
    param([Parameter(Mandatory)][hashtable]$Body, [Parameter(Mandatory)][double[]]$X, [int]$Rotation)
    $armBodies = @{}
    foreach ($arm in $arms) {
        $armBodies[$arm] = Get-ArmBody -Arm $arm -Body $Body -X $X
        Set-ArmEnvironment -Arm $arm
        for ($w = 0; $w -lt 2; $w++) { $null = & $armBodies[$arm] }
    }
    $first = (Measure-Cell -Body $controlBody).MedianMs
    $samples = @{}
    foreach ($arm in $arms) { $samples[$arm] = New-Object double[] $Repeats }
    for ($r = 0; $r -lt $Repeats; $r++) {
        for ($k = 0; $k -lt $arms.Count; $k++) {
            $arm = $arms[($k + $r + $Rotation) % $arms.Count]
            Set-ArmEnvironment -Arm $arm
            $samples[$arm][$r] = Measure-Once -Body $armBodies[$arm]
            Start-Cooldown
        }
    }
    $last = (Measure-Cell -Body $controlBody).MedianMs
    [PSCustomObject]@{ Samples = $samples; ControlFirstMs = $first; ControlLastMs = $last }
}

# The median over rounds of one arm's time over another's in the same
# round, so a round the process slowed moves both sides of its ratio.
function Get-PairRatio {
    param([Parameter(Mandatory)][double[]]$Over, [Parameter(Mandatory)][double[]]$Under)
    $ratios = New-Object double[] $Over.Length
    for ($i = 0; $i -lt $Over.Length; $i++) { $ratios[$i] = $Over[$i] / $Under[$i] }
    [Math]::Round((Get-Median $ratios), 3)
}

# A pass's figures added to its row under a prefix: each arm's median,
# the paired ratios, the control's drift across the row, and every sample.
function Add-Pass {
    param([Parameter(Mandatory)]$Row, [string]$Prefix, [Parameter(Mandatory)]$Measured)
    $s = $Measured.Samples
    $hasPool = $s.ContainsKey('POOL')
    $drift = if ($Measured.ControlFirstMs -gt 0) {
        [Math]::Round(100.0 * ($Measured.ControlLastMs - $Measured.ControlFirstMs) / $Measured.ControlFirstMs, 2)
    } else { $null }
    $kept = [ordered]@{}
    foreach ($arm in $arms) { $kept[$arm] = @($s[$arm] | ForEach-Object { [Math]::Round($_, 4) }) }
    $values = [ordered]@{
        "${Prefix}PoolMs"           = if ($hasPool) { [Math]::Round((Get-Median $s['POOL']), 4) } else { $null }
        "${Prefix}SerialMs"         = [Math]::Round((Get-Median $s['SERIAL']), 4)
        "${Prefix}EngineMs"         = [Math]::Round((Get-Median $s['ENGINE']), 4)
        "${Prefix}SerialOverPool"   = if ($hasPool) { Get-PairRatio $s['SERIAL'] $s['POOL'] } else { $null }
        "${Prefix}EngineOverPool"   = if ($hasPool) { Get-PairRatio $s['ENGINE'] $s['POOL'] } else { $null }
        "${Prefix}EngineOverSerial" = Get-PairRatio $s['ENGINE'] $s['SERIAL']
        "${Prefix}DriftPct"         = $drift
        "${Prefix}Samples"          = [PSCustomObject]$kept
    }
    foreach ($name in $values.Keys) { $Row | Add-Member -NotePropertyName $name -NotePropertyValue $values[$name] }
}

# ----------------------------------------------------------------------
# The run
# ----------------------------------------------------------------------

$arms = if ($poolPossible) { @('POOL', 'SERIAL', 'ENGINE') } else { @('SERIAL', 'ENGINE') }
$lightfoldArms = @($arms | Where-Object { $_ -ne 'ENGINE' })
$inputs = @{}
foreach ($size in $Sizes) { $inputs[$size] = New-Input -Count $size }

# Every row is checked, and its phase reports read, before the first timed
# call: the answers agree bit for bit and each Lightfold arm ran where it
# was sent.
$rows = @()
foreach ($size in $Sizes) {
    $x = $inputs[$size]
    foreach ($b in $bodies) {
        $answers = @{}
        foreach ($arm in $arms) { $answers[$arm] = Get-Answer -Arm $arm -Body $b -X $x }
        foreach ($arm in $arms) {
            if (-not (Test-SameBits $answers[$arm] $answers['ENGINE'])) {
                throw "$($b.Name) at ${size}: the $arm answer differs from ENGINE's in its bits; nothing is timed"
            }
        }
        $reports = @{}
        foreach ($arm in $lightfoldArms) { $reports[$arm] = Get-RunReport -Arm $arm -Body $b -X $x }
        if ($poolPossible -and [string]$reports['POOL'].Pool -notmatch 'Flynnel') {
            throw "$($b.Name) at ${size}: POOL reported pool '$($reports['POOL'].Pool)', so it did not run on Flynnel"
        }
        if ([string]$reports['SERIAL'].Pool -match 'Flynnel') {
            throw "$($b.Name) at ${size}: SERIAL reported pool '$($reports['SERIAL'].Pool)'"
        }
        $rows += [PSCustomObject]@{
            Body         = $b.Name
            Size         = $size
            PoolRan      = if ($poolPossible) { [string]$reports['POOL'].Pool } else { $null }
            SerialRan    = [string]$reports['SERIAL'].Pool
            PoolPhases   = if ($poolPossible) { Get-Phases $reports['POOL'] } else { $null }
            SerialPhases = Get-Phases $reports['SERIAL']
        }
    }
}
$answers = $null
$reports = $null
Write-Host ("checked {0} row(s) before any timing: every answer matches ENGINE's bits, and each Lightfold arm ran where it was sent" -f $rows.Count)

$rowIndex = 0
$controlFirst = Measure-Cell -Body $controlBody -Warmup 20
foreach ($row in $rows) {
    $b = $bodies | Where-Object { $_.Name -eq $row.Body }
    Write-Host ("quiet {0} at {1}" -f $row.Body, $row.Size)
    Add-Pass -Row $row -Prefix '' -Measured (Measure-Row -Body $b -X $inputs[$row.Size] -Rotation $rowIndex)
    $rowIndex++
}
$controlLast = Measure-Cell -Body $controlBody

# The load pass: every row again under the burners, each with a control at
# both ends.
$loadedControlFirst = $null
$loadedControlLast = $null
if ($LoadThreads -gt 0) {
    Write-Host ("starting {0} burner(s) for the load pass" -f $LoadThreads)
    Start-Burners
    $loadedControlFirst = Measure-Cell -Body $controlBody
    foreach ($row in $rows) {
        $b = $bodies | Where-Object { $_.Name -eq $row.Body }
        Write-Host ("loaded {0} at {1}" -f $row.Body, $row.Size)
        Add-Pass -Row $row -Prefix 'Loaded' -Measured (Measure-Row -Body $b -X $inputs[$row.Size] -Rotation $rowIndex)
        $rowIndex++
    }
    $loadedControlLast = Measure-Cell -Body $controlBody
    Stop-Burners
    Write-Host 'burners stopped'
}

$result = [PSCustomObject]@{
    Tag                  = $Tag
    When                 = (Get-Date -Format 'o')
    Host                 = $hostInfo
    Repeats              = $Repeats
    Arms                 = $arms
    ControlFirstMs       = [Math]::Round($controlFirst.MedianMs, 4)
    ControlLastMs        = [Math]::Round($controlLast.MedianMs, 4)
    LoadThreads          = $LoadThreads
    LoadedControlFirstMs = if ($loadedControlFirst) { [Math]::Round($loadedControlFirst.MedianMs, 4) } else { $null }
    LoadedControlLastMs  = if ($loadedControlLast) { [Math]::Round($loadedControlLast.MedianMs, 4) } else { $null }
    Rows                 = $rows
}
$stamp = (Get-Date -Format 'yyyyMMdd-HHmmss')
$outFile = Join-Path $OutDir "$Tag-$($hostInfo.Machine)-$edition-$stamp.json"
$result | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $outFile -Encoding utf8

Write-Host ''
Write-Host ("host {0} {1} {2}, {3} cpu(s), Flynnel {4}, Lightfold {5}" -f $hostInfo.Machine, $edition,
    $hostInfo.Version, $hostInfo.ProcessorCount, $commit,
    $(if ($LightfoldCommit) { $LightfoldCommit } else { 'unknown' }))
Write-Host ("control {0} ms then {1} ms" -f $result.ControlFirstMs, $result.ControlLastMs)
if ($LoadThreads -gt 0) {
    Write-Host ("load pass: {0} burner(s), control {1} ms quiet against {2} ms loaded" -f
        $LoadThreads, $result.ControlFirstMs, $result.LoadedControlFirstMs)
}
Write-Host ''
Write-Host 'quiet'
$rows | Format-Table Body, Size, PoolMs, SerialMs, EngineMs, SerialOverPool, EngineOverPool,
    EngineOverSerial, DriftPct -AutoSize | Out-Host
if ($LoadThreads -gt 0) {
    Write-Host 'loaded'
    $rows | Format-Table Body, Size, LoadedPoolMs, LoadedSerialMs, LoadedEngineMs, LoadedSerialOverPool,
        LoadedEngineOverPool, LoadedEngineOverSerial, LoadedDriftPct -AutoSize | Out-Host
}
Write-Host 'phases of one call per Lightfold arm, from its -Stats report, taken before any timing'
$phaseRows = foreach ($row in $rows) {
    foreach ($pair in @(@('POOL', $row.PoolPhases), @('SERIAL', $row.SerialPhases))) {
        if ($null -ne $pair[1]) {
            [PSCustomObject]@{
                Body = $row.Body; Size = $row.Size; Arm = $pair[0]
                SplitMs = $pair[1].SplitMs; DecodeMs = $pair[1].DecodeMs; EvalMs = $pair[1].EvalMs
                WriteMs = $pair[1].WriteMs; TotalMs = $pair[1].TotalMs
            }
        }
    }
}
$phaseRows | Format-Table -AutoSize | Out-Host
Write-Host 'Each ratio is the median over rounds of one arm over another in the same round. SerialOverPool'
Write-Host 'above one means the pool arm beat Lightfold on one thread; EngineOverPool and EngineOverSerial'
Write-Host 'below one mean the declared kernel driven through the engine beat that arm. DriftPct is how far'
Write-Host "the control moved across the row, a floor under the row's error."
Write-Host "RESULT $outFile"
