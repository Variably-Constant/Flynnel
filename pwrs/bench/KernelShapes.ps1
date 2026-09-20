# What the declared kernels cost, measured three ways in one run.
#
# The three arms, per kernel, all in this process and interleaved:
#
#   flynnel  the kernel as a script would call it, the pool deciding
#            its own width.
#   serial   the same kernel under a plan pinned to one worker. Flynnel
#            running its own code down one lane, so the difference
#            between this and the arm above is dispatch and nothing
#            else.
#   native   the PowerShell way to get the same answer: Measure-Object,
#            Sort-Object, Select-String, Get-FileHash. What a script
#            would otherwise have written.
#
# The control is a fixed arithmetic loop this module cannot reach,
# benched first and again last. Its drift across the run is the
# position effect: how much of a row's difference is where the row sat
# rather than what it did. A control outside a few per cent means the
# box moved under the run and no row in it is readable.
#
# The empty cell is the cheapest cmdlet in the module, which does no
# work. It separates the cost of crossing the boundary from the cost of
# the work behind it, so a kernel that looks slow can be read as slow
# to dispatch or slow to compute.
#
# The anchor is a fixed-size reduce over a fixed buffer. It is never
# given an arm, never changed, and benched first in every run, and its
# median is stored against the commit. Every table prints the anchor's
# drift from the previous build beside each row's own difference.
#
# The control bounds position inside one run. The anchor bounds drift
# between builds, which the control cannot see: a branchy microsecond
# cell in quartz moved 53 per cent across a rebuild that touched
# neither measured function, while that run's controls held inside 6
# per cent. Stationary cells in the same rebuild moved under 10 per
# cent, so the anchor speaks only for the straight-scan rows and each
# row says whether it is one.
#
# Ratios are read within one run against that run's control. A figure
# from another run, or another build, is a different measurement.

[CmdletBinding()]
param(
    # Where the built module folder is. The same variable the suites
    # use, so a bench and a test run measure the same binary.
    [string]$Module = $env:PWRS_MODULE,
    # How many timed repeats per cell. The median is reported.
    [int]$Repeats = 7,
    # Seconds of idle between arms, so one arm's cache and turbo state
    # is not the next arm's starting condition.
    [double]$Cooldown = 0.75,
    # Where the per-commit anchor medians accumulate.
    [string]$AnchorStore = (Join-Path $PSScriptRoot 'anchor-medians.json'),
    # Where this run's table and raw cells land.
    [string]$OutDir = $PSScriptRoot,
    # A name for this run in the output file.
    [string]$Tag = 'kernelshapes'
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# ----------------------------------------------------------------------
# The module under measurement
# ----------------------------------------------------------------------

if (-not $Module) {
    throw 'PWRS_MODULE is not set and -Module was not given. A bench that imported an ' +
          'installed copy would measure something other than the build in front of it.'
}
$manifest = Join-Path $Module 'Flynnel.psd1'
if (-not (Test-Path $manifest)) { throw "no manifest at $manifest" }
Import-Module $manifest -Force -ErrorAction Stop

$repoRoot = Split-Path $PSScriptRoot -Parent
$commit = (& git -C $repoRoot rev-parse --short HEAD 2>&1 | Out-String).Trim()
$dirty = @(& git -C $repoRoot status --porcelain 2>&1).Count

$hostInfo = [PSCustomObject]@{
    Machine        = [Environment]::MachineName
    Edition        = $PSVersionTable.PSEdition
    Version        = $PSVersionTable.PSVersion.ToString()
    Platform       = if ($IsWindows -or $PSVersionTable.PSEdition -eq 'Desktop') { 'Windows' }
                     elseif ($IsLinux) { 'Linux' } else { 'Other' }
    ProcessorCount = [Environment]::ProcessorCount
    Commit         = $commit
    Dirty          = $dirty
}

# ----------------------------------------------------------------------
# Timing
# ----------------------------------------------------------------------

function Measure-Cell {
    param(
        [Parameter(Mandatory)][scriptblock]$Body,
        [int]$Warmup = 2
    )
    for ($i = 0; $i -lt $Warmup; $i++) { $null = & $Body }
    $samples = New-Object double[] $Repeats
    for ($i = 0; $i -lt $Repeats; $i++) {
        $sw = [System.Diagnostics.Stopwatch]::StartNew()
        $null = & $Body
        $sw.Stop()
        $samples[$i] = $sw.Elapsed.TotalMilliseconds
    }
    $sorted = $samples | Sort-Object
    [PSCustomObject]@{
        MedianMs = $sorted[[int]($Repeats / 2)]
        MinMs    = $sorted[0]
        MaxMs    = $sorted[-1]
    }
}

function Start-Cooldown {
    if ($Cooldown -gt 0) { Start-Sleep -Milliseconds ([int]($Cooldown * 1000)) }
}

# ----------------------------------------------------------------------
# The control, the empty cell and the anchor
# ----------------------------------------------------------------------

# Arithmetic the module cannot reach, at a fixed size. Its only job is
# to be the same work at both ends of the run.
$controlBody = {
    $acc = 0.0
    for ($i = 1; $i -le 200000; $i++) { $acc += [Math]::Sqrt($i) }
    $acc
}

# The cheapest cmdlet the module has: a table lookup with no dispatch
# and no allocation behind it.
$emptyBody = { Get-FlynnelKBand -KOuter 8 }

# The anchor. Fixed buffer, fixed size, fixed operation, no arm. Do not
# change any of the three: its whole value is that it is the same cell
# in every build, and a changed anchor is a new series with no history.
$anchorBuffer = 1..65536 | ForEach-Object { [double]($_ % 1024) }
$anchorBody = { Measure-FlynnelReduce -InputObject $anchorBuffer -Operation Sum }

Write-Host 'anchor first, before anything else in this run'
$anchor = Measure-Cell -Body $anchorBody
Start-Cooldown
$controlFirst = Measure-Cell -Body $controlBody
Start-Cooldown
$empty = Measure-Cell -Body $emptyBody
Start-Cooldown

# ----------------------------------------------------------------------
# The kernels and their three arms
# ----------------------------------------------------------------------

$n = 200000
$data = 1..$n | ForEach-Object { [double]($_ % 997) }
$other = 1..$n | ForEach-Object { [double](($_ * 7) % 501) }
$serialPlan = New-FlynnelPlan -KOuter 10 -BatchSize $n -Workers 1

$work = Join-Path ([System.IO.Path]::GetTempPath()) "flynnel-bench-$PID"
New-Item -ItemType Directory -Path $work -Force | Out-Null
$files = @()
foreach ($i in 1..64) {
    $p = Join-Path $work "f$i.bin"
    Set-Content -LiteralPath $p -Value (('payload ' * 4096) + $i) -NoNewline
    $files += $p
}
$textBody = (1..40000 | ForEach-Object { "row $_ value ABCD trailing" }) -join "`n"

# Straight records whether this row's shape is the anchor's shape. The
# anchor is a straight scan, so it speaks for the kernels whose inner
# loop is also a straight scan and not for the ones whose inner loop
# branches on the data. A row it does not speak for is readable within
# this run and carries no cross-build claim.
$kernels = @(
    @{ Name = 'Map.Square'; Straight = $true
       Flynnel = { Invoke-FlynnelMap -InputObject $data -Operation Square }
       Serial  = { Invoke-FlynnelMap -InputObject $data -Operation Square -Plan $serialPlan }
       Native  = { $data | ForEach-Object { $_ * $_ } } }

    @{ Name = 'Zip.Add'; Straight = $true
       Flynnel = { Invoke-FlynnelZip -Left $data -Right $other -Operation Add }
       Serial  = { Invoke-FlynnelZip -Left $data -Right $other -Operation Add -Plan $serialPlan }
       Native  = { $out = New-Object double[] $n
                   for ($i = 0; $i -lt $n; $i++) { $out[$i] = $data[$i] + $other[$i] }
                   $out } }

    @{ Name = 'Reduce.Sum'; Straight = $true
       Flynnel = { Measure-FlynnelReduce -InputObject $data -Operation Sum }
       Serial  = { Measure-FlynnelReduce -InputObject $data -Operation Sum -Plan $serialPlan }
       Native  = { $data | Measure-Object -Sum } }

    @{ Name = 'Reduce.Variance'; Straight = $true
       Flynnel = { Measure-FlynnelReduce -InputObject $data -Operation Variance }
       Serial  = { Measure-FlynnelReduce -InputObject $data -Operation Variance -Plan $serialPlan }
       Native  = { $m = ($data | Measure-Object -Average).Average
                   ($data | ForEach-Object { ($_ - $m) * ($_ - $m) } |
                       Measure-Object -Sum).Sum / $n } }

    @{ Name = 'PrefixSum'; Straight = $true
       Flynnel = { Get-FlynnelPrefixSum -InputObject $data }
       Serial  = { Get-FlynnelPrefixSum -InputObject $data -Plan $serialPlan }
       Native  = { $out = New-Object double[] $n; $a = 0.0
                   for ($i = 0; $i -lt $n; $i++) { $a += $data[$i]; $out[$i] = $a }
                   $out } }

    @{ Name = 'DotProduct'; Straight = $true
       Flynnel = { Get-FlynnelDotProduct -Left $data -Right $other }
       Serial  = { Get-FlynnelDotProduct -Left $data -Right $other -Plan $serialPlan }
       Native  = { $a = 0.0
                   for ($i = 0; $i -lt $n; $i++) { $a += $data[$i] * $other[$i] }
                   $a } }

    @{ Name = 'Histogram'; Straight = $false
       Flynnel = { Get-FlynnelHistogram -InputObject $data -Bins 32 }
       Serial  = { Get-FlynnelHistogram -InputObject $data -Bins 32 -Plan $serialPlan }
       Native  = { $b = New-Object long[] 32
                   foreach ($x in $data) { $b[[Math]::Min(31, [int]($x / 32))]++ }
                   $b } }

    @{ Name = 'Sort'; Straight = $false
       Flynnel = { Sort-FlynnelArray -InputObject $data }
       Serial  = { Sort-FlynnelArray -InputObject $data -Plan $serialPlan }
       Native  = { $data | Sort-Object } }

    @{ Name = 'FileHash'; Straight = $false
       Flynnel = { Measure-FlynnelFileHash -Path $files }
       Serial  = { Measure-FlynnelFileHash -Path $files -Plan $serialPlan }
       Native  = { Get-FileHash -LiteralPath $files -Algorithm SHA256 } }

    @{ Name = 'FileLine'; Straight = $false
       Flynnel = { Measure-FlynnelFileLine -Path $files }
       Serial  = { Measure-FlynnelFileLine -Path $files -Plan $serialPlan }
       Native  = { $files | ForEach-Object { (Get-Content -LiteralPath $_).Count } } }

    @{ Name = 'SearchFile'; Straight = $false
       Flynnel = { Search-FlynnelFile -Pattern 'payload' -Path $files }
       Serial  = { Search-FlynnelFile -Pattern 'payload' -Path $files -Plan $serialPlan }
       Native  = { Select-String -LiteralPath $files -Pattern 'payload' -SimpleMatch } }

    @{ Name = 'SearchText'; Straight = $false
       Flynnel = { Search-FlynnelText -Text $textBody -Pattern 'ABCD' }
       Serial  = { Search-FlynnelText -Text $textBody -Pattern 'ABCD' -Plan $serialPlan }
       Native  = { [regex]::Matches($textBody, 'ABCD') } }

    @{ Name = 'TextCount'; Straight = $false
       Flynnel = { Measure-FlynnelTextCount -Text $textBody }
       Serial  = { Measure-FlynnelTextCount -Text $textBody -Plan $serialPlan }
       Native  = { $textBody.Split(@(' ', "`n"), 'RemoveEmptyEntries').Count } }

    @{ Name = 'SplitText'; Straight = $false
       Flynnel = { Split-FlynnelText -Text $textBody -Separator ' ' }
       Serial  = { Split-FlynnelText -Text $textBody -Separator ' ' -Plan $serialPlan }
       Native  = { $textBody.Split(' ') } }

    @{ Name = 'UpdateText'; Straight = $false
       Flynnel = { Update-FlynnelText -Text $textBody -Operation Replace `
                       -Pattern 'ABCD' -Replacement 'WXYZ' }
       Serial  = { Update-FlynnelText -Text $textBody -Operation Replace `
                       -Pattern 'ABCD' -Replacement 'WXYZ' -Plan $serialPlan }
       Native  = { $textBody.Replace('ABCD', 'WXYZ') } }
)

# Arms interleave rather than running all of one arm then all of the
# next: a box that drifts during the run then moves every arm together
# instead of loading the arm that ran while it drifted.
$rows = @()
foreach ($k in $kernels) {
    Write-Host ("cell {0}" -f $k.Name)
    $f = Measure-Cell -Body $k.Flynnel
    Start-Cooldown
    $s = Measure-Cell -Body $k.Serial
    Start-Cooldown
    $nat = Measure-Cell -Body $k.Native
    Start-Cooldown
    $rows += [PSCustomObject]@{
        Kernel          = $k.Name
        FlynnelMs       = [Math]::Round($f.MedianMs, 4)
        SerialMs        = [Math]::Round($s.MedianMs, 4)
        NativeMs        = [Math]::Round($nat.MedianMs, 4)
        VsSerial        = if ($f.MedianMs -gt 0) {
                              [Math]::Round($s.MedianMs / $f.MedianMs, 3)
                          } else { $null }
        VsNative        = if ($f.MedianMs -gt 0) {
                              [Math]::Round($nat.MedianMs / $f.MedianMs, 3)
                          } else { $null }
        AnchorSpeaksFor = $k.Straight
    }
}

$controlLast = Measure-Cell -Body $controlBody

# ----------------------------------------------------------------------
# The two figures every row is read against
# ----------------------------------------------------------------------

$controlDriftPct = if ($controlFirst.MedianMs -gt 0) {
    [Math]::Round(
        100.0 * ($controlLast.MedianMs - $controlFirst.MedianMs) / $controlFirst.MedianMs, 2)
} else { $null }

# The anchor's drift is against the previous build, so the store is
# keyed by commit and the comparison is to the most recent entry from a
# different one. Comparing to the same commit would measure the box,
# which the control already does.
$store = @()
if (Test-Path $AnchorStore) {
    $store = @(Get-Content -LiteralPath $AnchorStore -Raw | ConvertFrom-Json)
}
$previous = $store |
    Where-Object { $_.Commit -ne $commit -and $_.Machine -eq $hostInfo.Machine -and
                   $_.Edition -eq $hostInfo.Edition } |
    Select-Object -Last 1

$anchorDriftPct = $null
if ($previous) {
    $anchorDriftPct = [Math]::Round(
        100.0 * ($anchor.MedianMs - $previous.AnchorMs) / $previous.AnchorMs, 2)
}

# A row is not comparable across builds when the anchor moved further
# than the row's own difference did: whatever the row shows could be
# the build rather than the work. A row the anchor does not speak for
# is never comparable across builds whatever the anchor did.
foreach ($row in $rows) {
    $verdict = 'within-run-only'
    if ($null -ne $anchorDriftPct -and $row.AnchorSpeaksFor) {
        $ownDiffPct = if ($row.VsSerial) {
            [Math]::Abs(100.0 * ($row.VsSerial - 1.0))
        } else { 0.0 }
        $verdict = if ([Math]::Abs($anchorDriftPct) -gt $ownDiffPct) {
            'not-comparable-across-builds'
        } else {
            'comparable-across-builds'
        }
    }
    $row | Add-Member -NotePropertyName CrossBuild -NotePropertyValue $verdict
}

# ----------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------

$result = [PSCustomObject]@{
    Tag             = $Tag
    When            = (Get-Date -Format 'o')
    Host            = $hostInfo
    Repeats         = $Repeats
    AnchorMs        = [Math]::Round($anchor.MedianMs, 4)
    AnchorDriftPct  = $anchorDriftPct
    AnchorAgainst   = if ($previous) { $previous.Commit } else { $null }
    ControlFirstMs  = [Math]::Round($controlFirst.MedianMs, 4)
    ControlLastMs   = [Math]::Round($controlLast.MedianMs, 4)
    ControlDriftPct = $controlDriftPct
    EmptyCallMs     = [Math]::Round($empty.MedianMs, 6)
    Rows            = $rows
}

$stamp = (Get-Date -Format 'yyyyMMdd-HHmmss')
$outFile = Join-Path $OutDir "$Tag-$($hostInfo.Machine)-$($hostInfo.Edition)-$stamp.json"
$result | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $outFile -Encoding utf8

$store = @($store) + [PSCustomObject]@{
    Commit   = $commit
    Machine  = $hostInfo.Machine
    Edition  = $hostInfo.Edition
    When     = (Get-Date -Format 'o')
    AnchorMs = [Math]::Round($anchor.MedianMs, 4)
}
$store | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $AnchorStore -Encoding utf8

Write-Host ''
Write-Host ("host {0} {1} {2}, {3} cpu(s), commit {4}{5}" -f
    $hostInfo.Machine, $hostInfo.Edition, $hostInfo.Platform,
    $hostInfo.ProcessorCount, $commit, $(if ($dirty) { " (dirty=$dirty)" } else { '' }))
Write-Host ("control {0} ms then {1} ms, drift {2}%" -f
    $result.ControlFirstMs, $result.ControlLastMs, $controlDriftPct)
Write-Host ("empty call {0} ms" -f $result.EmptyCallMs)
if ($null -ne $anchorDriftPct) {
    Write-Host ("anchor {0} ms, {1}% from {2}" -f
        $result.AnchorMs, $anchorDriftPct, $previous.Commit)
} else {
    Write-Host ("anchor {0} ms, no earlier build on this host to compare with" -f
        $result.AnchorMs)
}
Write-Host ''
$rows | Format-Table Kernel, FlynnelMs, SerialMs, NativeMs, VsSerial, VsNative,
    AnchorSpeaksFor, CrossBuild -AutoSize
Write-Host ''
Write-Host 'VsSerial is how many times faster the pool is than one worker running the same'
Write-Host 'kernel. VsNative is against the PowerShell way to get the same answer.'
Write-Host 'AnchorSpeaksFor is false where the inner loop branches on the data, which is'
Write-Host 'the shape the anchor was measured not to stand in for.'
Write-Host ("raw cells in {0}" -f $outFile)

Remove-Item -LiteralPath $work -Recurse -Force -ErrorAction SilentlyContinue
