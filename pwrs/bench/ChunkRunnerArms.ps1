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
# both sides, per Lightfold's compiled plane and Flynnel's MapOp. Before a
# cell is timed its three answers are compared bit for bit, and the POOL
# and SERIAL arms are asked through -Stats which pool actually ran, so a
# row cannot be fast because it computed something else or because the
# pool quietly declined.
#
# Lightfold reads LIGHTFOLD_POOL on every dispatch, so the variable is set
# between arms and never inside a timed body. The order of the three arms
# rotates by row so that a position does not stand in for an arm. A
# PowerShell-only control is timed at both ends of every row, which gives
# how far the box moved while the row was taken. The load pass repeats
# every arm under burners on half the logical processors and is read
# against a loaded control.
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
    [double]$Cooldown = 0.5,
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

$repoRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$commit = (& git -C $repoRoot rev-parse --short HEAD 2>&1 | Out-String).Trim()
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

function Measure-Cell {
    param([Parameter(Mandatory)][scriptblock]$Body, [int]$Warmup = 2)
    for ($i = 0; $i -lt $Warmup; $i++) { $null = & $Body }
    $samples = New-Object double[] $Repeats
    for ($i = 0; $i -lt $Repeats; $i++) {
        $sw = [System.Diagnostics.Stopwatch]::StartNew()
        $null = & $Body
        $sw.Stop()
        $samples[$i] = $sw.Elapsed.TotalMilliseconds
    }
    $sorted = $samples | Sort-Object
    [PSCustomObject]@{ MedianMs = $sorted[[int]($Repeats / 2)]; MinMs = $sorted[0]; MaxMs = $sorted[-1] }
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

# Which pool a Lightfold arm actually ran on, from its own report.
function Get-PoolUsed {
    param([string]$Arm, [hashtable]$Body, [double[]]$X)
    $env:LIGHTFOLD_POOL = if ($Arm -eq 'POOL') { 'flynnel' } else { 'inline' }
    $run = @(Invoke-LightfoldParallel -Script (Get-Script $Body.Text) -Items $X -Plane Compiled -Stats |
        Where-Object { $_.PSObject.TypeNames -contains 'Lightfold.ParallelRun' -or
                       $_.GetType().FullName -eq 'Lightfold.ParallelRun' })
    if ($run.Count -ne 1) { throw "$Arm $($Body.Name): -Stats gave $($run.Count) ParallelRun objects, not one" }
    [string]$run[0].Pool
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

# ----------------------------------------------------------------------
# The run
# ----------------------------------------------------------------------

$arms = if ($poolPossible) { @('POOL', 'SERIAL', 'ENGINE') } else { @('SERIAL', 'ENGINE') }
$rows = @()
$rowIndex = 0
$controlFirst = Measure-Cell -Body $controlBody -Warmup 20
Start-Cooldown

foreach ($size in $Sizes) {
    $x = New-Input -Count $size
    foreach ($b in $bodies) {
        # Correctness first: the answers agree and each Lightfold arm ran
        # where it was asked to.
        $answers = @{}
        foreach ($arm in $arms) { $answers[$arm] = Get-Answer -Arm $arm -Body $b -X $x }
        foreach ($arm in $arms) {
            if (-not (Test-SameBits $answers[$arm] $answers['ENGINE'])) {
                throw "$($b.Name) at ${size}: the $arm answer differs from ENGINE's in its bits; nothing is timed"
            }
        }
        $poolUsed = @{}
        foreach ($arm in $arms | Where-Object { $_ -ne 'ENGINE' }) {
            $poolUsed[$arm] = Get-PoolUsed -Arm $arm -Body $b -X $x
        }
        if ($poolPossible -and $poolUsed['POOL'] -notmatch 'Flynnel') {
            throw "$($b.Name) at ${size}: POOL reported pool '$($poolUsed['POOL'])', so it did not run on Flynnel"
        }
        if ($poolUsed['SERIAL'] -match 'Flynnel') {
            throw "$($b.Name) at ${size}: SERIAL reported pool '$($poolUsed['SERIAL'])'"
        }

        # The arm order rotates by row.
        $order = @(for ($k = 0; $k -lt $arms.Count; $k++) { $arms[($k + $rowIndex) % $arms.Count] })
        $rowIndex++
        Write-Host ("row {0} at {1}: order {2}" -f $b.Name, $size, ($order -join ','))
        $rowControlFirst = Measure-Cell -Body $controlBody
        $cells = @{}
        foreach ($arm in $order) {
            Set-ArmEnvironment -Arm $arm
            $cells[$arm] = Measure-Cell -Body (Get-ArmBody -Arm $arm -Body $b -X $x)
            Start-Cooldown
        }
        $rowControlLast = Measure-Cell -Body $controlBody
        $drift = if ($rowControlFirst.MedianMs -gt 0) {
            [Math]::Round(100.0 * ($rowControlLast.MedianMs - $rowControlFirst.MedianMs) / $rowControlFirst.MedianMs, 2)
        } else { $null }
        $poolMs = if ($cells.ContainsKey('POOL')) { [Math]::Round($cells['POOL'].MedianMs, 4) } else { $null }
        $serialMs = [Math]::Round($cells['SERIAL'].MedianMs, 4)
        $engineMs = [Math]::Round($cells['ENGINE'].MedianMs, 4)
        $serialOverPool = $null
        $engineOverPool = $null
        if ($null -ne $poolMs -and $poolMs -gt 0) {
            $serialOverPool = [Math]::Round($serialMs / $poolMs, 3)
            $engineOverPool = [Math]::Round($engineMs / $poolMs, 3)
        }
        $rows += [PSCustomObject]@{
            Body           = $b.Name
            Size           = $size
            PoolMs         = $poolMs
            SerialMs       = $serialMs
            EngineMs       = $engineMs
            SerialOverPool = $serialOverPool
            EngineOverPool = $engineOverPool
            RowDriftPct    = $drift
            PoolRan        = if ($poolUsed.ContainsKey('POOL')) { $poolUsed['POOL'] } else { $null }
            SerialRan      = $poolUsed['SERIAL']
        }
    }
}
$controlLast = Measure-Cell -Body $controlBody

# The load pass: every arm again under the burners, against a loaded
# control taken at both ends.
$loadedControlFirst = $null
$loadedControlLast = $null
if ($LoadThreads -gt 0) {
    Write-Host ("starting {0} burner(s) for the load pass" -f $LoadThreads)
    Start-Burners
    $loadedControlFirst = Measure-Cell -Body $controlBody
    foreach ($row in $rows) {
        $x = New-Input -Count $row.Size
        $b = $bodies | Where-Object { $_.Name -eq $row.Body }
        foreach ($arm in $arms) {
            Set-ArmEnvironment -Arm $arm
            $cell = Measure-Cell -Body (Get-ArmBody -Arm $arm -Body $b -X $x)
            $row | Add-Member -NotePropertyName ("Loaded{0}Ms" -f ($arm.Substring(0, 1) + $arm.Substring(1).ToLower())) `
                -NotePropertyValue ([Math]::Round($cell.MedianMs, 4))
            Start-Cooldown
        }
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
$rows | Format-Table Body, Size, PoolMs, SerialMs, EngineMs, SerialOverPool, EngineOverPool,
    RowDriftPct, PoolRan -AutoSize
Write-Host 'SerialOverPool above one means the pool arm beat Lightfold on one thread; EngineOverPool'
Write-Host 'above one means it beat the declared kernel driven through the engine. RowDriftPct is'
Write-Host "how far the control moved across the row's arms, a floor under the row's error."
Write-Host "RESULT $outFile"
