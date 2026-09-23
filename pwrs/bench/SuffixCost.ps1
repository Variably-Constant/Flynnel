# What the size-suffix transform costs a bind, in one process.
#
# The module puts a transform on its byte sizes and large counts, and
# PWRS's TransformBase crosses into Rust for every value bound to such a
# parameter, a number included. The first two cells are each timed against
# a control that differs from them by the transform alone:
#
#   direct   the transform's Transform method on a number, against a no-op
#            ArgumentTransformationAttribute called the same way
#   number   a function whose parameter carries the transform, bound to a
#            number, against the same function whose parameter carries none,
#            and a third carrying the no-op transform, which splits what the
#            binder charges for any transform from what this one adds
#   text     the same two functions bound to '64KB'. Windows PowerShell 5.1
#            refuses that text without the transform, so there the text
#            cell runs alone and has no control
#   cmdlet   New-FlynnelPlan with -BatchSize bound, as shipped, timed alone
#            as the whole call the transform is part of. -BatchSize is
#            mandatory, so the cmdlet has no call without it to pair with
#
# A sample is a batch of calls in one loop. Every round times each cell
# once, in an order that rotates by round, after a full collection, and a
# paired cell's cost is read as the median over rounds of its difference
# from its control in the same round. A PowerShell-only control is timed at
# both ends of each pass, and the load pass repeats every round under
# burners on half the logical processors.
#
# Run it with -NonInteractive, so a call that lacks a mandatory parameter
# fails rather than waiting at a prompt.
#
#   pwsh -NoProfile -NonInteractive -File SuffixCost.ps1 -Module <dir> [-Rounds 21] [-Batch 2000]
#
# Prints, per pass: a `CONTROL` line with the control's two readings in
# milliseconds; a `CELL` line per cell with its per-call microseconds as
# median, min and max over rounds; and a `DIFF` line per pair with the
# median over rounds of the per-call difference in microseconds.

param(
    [Parameter(Mandatory)][string]$Module,
    [int]$Rounds = 21,
    [int]$Batch = 2000,
    [int]$LoadThreads = [int]([Environment]::ProcessorCount / 2)
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# ----------------------------------------------------------------------
# The burners, and the trap that stops them
# ----------------------------------------------------------------------

$script:Burners = @()
$onWindows = [Environment]::OSVersion.Platform -eq [PlatformID]::Win32NT
$script:BurnerShell = if ($onWindows) { 'powershell' } else { 'pwsh' }

function Start-Burners {
    if ($LoadThreads -le 0) { return }
    $spin = 'while ($true) { $null = [Math]::Sqrt([Environment]::TickCount) }'
    $arguments = @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-Command', $spin)
    $script:Burners = 1..$LoadThreads | ForEach-Object {
        if ($onWindows) {
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

trap {
    Stop-Burners
    Write-Host ("FAULT " + $_.Exception.GetType().Name + ": " + $_.Exception.Message)
    Write-Host ("FAULT_AT " + $_.InvocationInfo.PositionMessage)
    throw $_
}

# ----------------------------------------------------------------------
# The module, the transform and its controls
# ----------------------------------------------------------------------

Import-Module (Join-Path $Module 'Flynnel.psd1') -Force -ErrorAction Stop
$edition = $PSVersionTable.PSEdition
Write-Host ("HOST {0} {1} {2} cpus={3} module={4}" -f [Environment]::MachineName, $edition,
    $PSVersionTable.PSVersion, [Environment]::ProcessorCount, (Resolve-Path $Module).Path)

$carried = @((Get-Command New-FlynnelPlan).Parameters['BatchSize'].Attributes | Where-Object {
    $null -ne $_.GetType().BaseType -and $_.GetType().BaseType.FullName -eq 'Pwrs.TransformBase'
})
if ($carried.Count -ne 1) {
    throw "New-FlynnelPlan -BatchSize carries $($carried.Count) PWRS transforms, not one"
}
$transform = $carried[0]
Write-Host ("STEP transform {0}" -f $transform.GetType().FullName)

Add-Type -TypeDefinition @'
using System.Management.Automation;
public sealed class FlynnelBenchNoopTransform : ArgumentTransformationAttribute
{
    public override object Transform(EngineIntrinsics engineIntrinsics, object inputData)
    {
        return inputData;
    }
}
'@
$noop = New-Object FlynnelBenchNoopTransform
$engine = $ExecutionContext
Write-Host 'STEP no-op transform compiled'
$direct = $transform.Transform($engine, 100000)
Write-Host ("STEP direct transform answered {0} as {1}" -f $direct, $direct.GetType().Name)

# A function with one dynamic parameter N of type long, carrying the given
# transform, or none when it is $null. Both functions build the parameter
# the same way, so they differ by the transform alone.
function New-Probe {
    param($Attribute)
    $attr = $Attribute
    {
        [CmdletBinding()]
        param()
        dynamicparam {
            $attributes = New-Object 'System.Collections.ObjectModel.Collection[System.Attribute]'
            $attributes.Add((New-Object System.Management.Automation.ParameterAttribute))
            if ($null -ne $attr) { $attributes.Add($attr) }
            $dictionary = New-Object System.Management.Automation.RuntimeDefinedParameterDictionary
            $dictionary.Add('N', (New-Object System.Management.Automation.RuntimeDefinedParameter('N', [long], $attributes)))
            $dictionary
        }
        end { $PSBoundParameters['N'] }
    }.GetNewClosure()
}
# The probes are plain parameter blocks when this host resolves the
# generated attribute's type by name, which costs a bind far less than a
# dynamic parameter does, and dynamic parameters otherwise. The form is
# printed, because the two carry different overheads around the same
# transform.
$typeName = $transform.GetType().FullName
$probeForm = 'static'
try {
    $withTransform = [scriptblock]::Create("param([$typeName()][long]`$N) `$N")
    if ((& $withTransform -N 1) -ne 1) { throw 'the static probe did not bind 1' }
    $withNone = [scriptblock]::Create('param([long]$N) $N')
    $withNoop = [scriptblock]::Create('param([FlynnelBenchNoopTransform()][long]$N) $N')
} catch [System.Management.Automation.RuntimeException] {
    $probeForm = 'dynamic'
    Write-Host ("PROBE_FORM_DYNAMIC {0} is not resolvable by name here: {1}" -f $typeName, $_.Exception.Message)
    $withTransform = New-Probe -Attribute $transform
    $withNone = New-Probe -Attribute $null
    $withNoop = New-Probe -Attribute $noop
}
if ((& $withNoop -N 100000) -ne 100000) { throw 'the no-op probe did not bind 100000' }
Write-Host "STEP probes built, $probeForm parameters"

# The value every cell binds, checked through both probes before timing.
if ((& $withNone -N 100000) -ne 100000) { throw 'the plain probe did not bind 100000' }
Write-Host 'STEP plain probe bound a number'
if ((& $withTransform -N 100000) -ne 100000) { throw 'the transform did not pass 100000 through' }
Write-Host 'STEP transform probe bound a number'
if ((& $withTransform -N '64KB') -ne 65536) { throw "the transform did not read '64KB' as 65536" }
Write-Host 'STEP transform probe read 64KB'
$textWithoutTransform = $true
try {
    if ((& $withNone -N '64KB') -ne 65536) { throw "the plain probe bound '64KB' to something other than 65536" }
} catch [System.Management.Automation.ParameterBindingException] {
    $textWithoutTransform = $false
    Write-Host "TEXT_REFUSED_WITHOUT_TRANSFORM this host does not bind '64KB' unaided, so the text cell runs without a control"
}
Write-Host ("STEP plain probe on 64KB: {0}" -f $(if ($textWithoutTransform) { 'bound' } else { 'refused' }))

# ----------------------------------------------------------------------
# The cells
# ----------------------------------------------------------------------
#
# Each cell is one loop of $Batch calls, so the per-call figure carries
# no invocation of the cell itself.

$cells = [ordered]@{
    direct_transform = { param($n) for ($i = 0; $i -lt $n; $i++) { $null = $transform.Transform($engine, 100000) } }.GetNewClosure()
    direct_noop      = { param($n) for ($i = 0; $i -lt $n; $i++) { $null = $noop.Transform($engine, 100000) } }.GetNewClosure()
    number_transform = { param($n) for ($i = 0; $i -lt $n; $i++) { $null = & $withTransform -N 100000 } }.GetNewClosure()
    number_none      = { param($n) for ($i = 0; $i -lt $n; $i++) { $null = & $withNone -N 100000 } }.GetNewClosure()
    number_noop      = { param($n) for ($i = 0; $i -lt $n; $i++) { $null = & $withNoop -N 100000 } }.GetNewClosure()
    text_transform   = { param($n) for ($i = 0; $i -lt $n; $i++) { $null = & $withTransform -N '64KB' } }.GetNewClosure()
    cmdlet_batchsize = { param($n) for ($i = 0; $i -lt $n; $i++) { $null = New-FlynnelPlan -KOuter 10 -BatchSize 100000 } }
}
if ($textWithoutTransform) {
    $cells['text_none'] = { param($n) for ($i = 0; $i -lt $n; $i++) { $null = & $withNone -N '64KB' } }.GetNewClosure()
}
$pairs = @(
    @('direct_transform', 'direct_noop'),
    @('number_transform', 'number_none'),
    @('number_transform', 'number_noop'),
    @('number_noop', 'number_none')
)
if ($textWithoutTransform) { $pairs += , @('text_transform', 'text_none') }

$controlBody = {
    $acc = 0.0
    for ($i = 1; $i -le 200000; $i++) { $acc += [Math]::Sqrt($i) }
    $acc
}

function Invoke-Collection {
    [System.GC]::Collect()
    [System.GC]::WaitForPendingFinalizers()
    [System.GC]::Collect()
}

function Get-Median {
    param([Parameter(Mandatory)][double[]]$Values)
    $sorted = [double[]]($Values | Sort-Object)
    $sorted[[int][Math]::Floor(($sorted.Length - 1) / 2)]
}

# PowerShell compiles a script block after sixteen calls, so every body is
# called twenty times before anything is timed.
$warmCalls = 20

function Measure-Control {
    for ($i = 0; $i -lt $warmCalls; $i++) { $null = & $controlBody }
    $samples = New-Object double[] 7
    for ($i = 0; $i -lt 7; $i++) {
        Invoke-Collection
        $sw = [System.Diagnostics.Stopwatch]::StartNew()
        $null = & $controlBody
        $sw.Stop()
        $samples[$i] = $sw.Elapsed.TotalMilliseconds
    }
    Get-Median $samples
}

# One pass: every round times each cell once, in an order that rotates by
# round, and the figures are printed under the pass's name.
function Invoke-Pass {
    param([Parameter(Mandatory)][string]$Pass)
    $names = @($cells.Keys)
    foreach ($name in $names) {
        for ($w = 0; $w -lt $warmCalls; $w++) { & $cells[$name] 5 }
        Write-Host "STEP $Pass warmed $name"
    }
    $first = Measure-Control
    $perCall = @{}
    foreach ($name in $names) { $perCall[$name] = New-Object double[] $Rounds }
    for ($r = 0; $r -lt $Rounds; $r++) {
        for ($k = 0; $k -lt $names.Count; $k++) {
            $name = $names[($k + $r) % $names.Count]
            Invoke-Collection
            $sw = [System.Diagnostics.Stopwatch]::StartNew()
            & $cells[$name] $Batch
            $sw.Stop()
            $perCall[$name][$r] = $sw.Elapsed.TotalMilliseconds * 1000.0 / $Batch
        }
        Write-Host ("STEP {0} round {1} of {2}" -f $Pass, ($r + 1), $Rounds)
    }
    $last = Measure-Control
    Write-Host ("CONTROL {0} first {1:F4} last {2:F4}" -f $Pass, $first, $last)
    foreach ($name in $names) {
        $sorted = [double[]]($perCall[$name] | Sort-Object)
        Write-Host ("CELL {0} {1} {2:F4} {3:F4} {4:F4}" -f $Pass, $name, (Get-Median $perCall[$name]),
            $sorted[0], $sorted[-1])
    }
    foreach ($pair in $pairs) {
        $diff = New-Object double[] $Rounds
        for ($r = 0; $r -lt $Rounds; $r++) { $diff[$r] = $perCall[$pair[0]][$r] - $perCall[$pair[1]][$r] }
        Write-Host ("DIFF {0} {1} {2} {3:F4}" -f $Pass, $pair[0], $pair[1], (Get-Median $diff))
    }
}

Invoke-Pass -Pass quiet
if ($LoadThreads -gt 0) {
    Write-Host ("LOAD {0} burner(s)" -f $LoadThreads)
    Start-Burners
    Invoke-Pass -Pass loaded
    Stop-Burners
}
Write-Host 'SUFFIX_COST_DONE'
