# What the declared kernels cost, measured three ways in one run.
#
# The arms, per kernel, all in this process and interleaved:
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
#   pool     the same kernel routed onto the process-wide IO pool with
#            -UseIoPool instead of the arena. One kernel has it, and
#            only where the process was launched with a pool to route
#            to. Without one the cmdlet warns and hashes on the arena,
#            so an unguarded arm would put the arena's figure in the
#            pool column and read as the route costing nothing.
#
# The pool arm runs quiet and loaded both. Contention is the whole of
# its case: routing reads off the arena is a claim about what happens
# when the arena has other work, so a quiet box is the one condition
# that cannot test it.
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
    #
    # The default sits in the tree, which is right for a repository
    # built in place and wrong for a per-commit clone: a fresh tree has
    # an empty store, so every run reports "no earlier build on this
    # host to compare with" and the cross-build drift the anchor exists
    # to carry can never accumulate. A clone-per-commit workflow has to
    # pass a path outside the tree.
    [string]$AnchorStore = (Join-Path $PSScriptRoot 'anchor-medians.json'),
    # Where each run's per-kernel medians accumulate, so run-to-run
    # spread can be read rather than the within-run spread standing in
    # for it.
    [string]$RunStore = (Join-Path $PSScriptRoot 'run-medians.json'),
    # Where this run's table and raw cells land.
    [string]$OutDir = $PSScriptRoot,
    # A name for this run in the output file.
    [string]$Tag = 'kernelshapes',
    # Ceiling on the warm-up, which stops early once the box's own
    # readings settle. Not a limit on what the run measures: the cells
    # are timed either way and the warm-up line says whether it settled.
    [double]$WarmupMaxSeconds = 60,
    # How close two consecutive warm-up blocks must read before the box
    # counts as settled.
    [double]$WarmupTolerancePct = 3,
    # How many burner processes contend for the box during the load
    # arm. Half the logical processors, which is the ratio the
    # campaign's own loaded rotations used: 12 burners on a 24-thread
    # host. Enough that the pool must share and not so much that it is
    # starved into measuring the scheduler's queueing instead of its
    # work. Zero skips the load arm entirely.
    [int]$LoadThreads = [int]([Environment]::ProcessorCount / 2)
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

# Whether this process has a process-wide IO pool, read before any
# kernel dispatches.
#
# It is a property of how the bench was launched rather than something
# it can arrange. global_io_pool() is a OnceLock filled when the pool
# starts, so a process that has already dispatched cannot gain one, and
# New-FlynnelIoPool hands back its own object instead of installing the
# global. FLYNNEL_SCHED_SMT_AS_IO has to be set before this process
# starts or there is nothing to route to.
#
# Read once here rather than at the cell that uses it, so every row in
# a run has the same answer and the run says which answer it had.
$ioPool = Get-FlynnelIoPool -WarningAction SilentlyContinue -ErrorAction SilentlyContinue
$hasIoPool = $null -ne $ioPool
if ($hasIoPool) {
    Write-Host ("io pool present with {0} worker(s); the pool arm runs" -f $ioPool.WorkerCount)
} else {
    Write-Host ('NO IO POOL in this process, so the pool arm is SKIPPED and its column ' +
                'will read empty. Set FLYNNEL_SCHED_SMT_AS_IO=1 before launching to measure it.')
}

# The commit describes the tree this script sits in. It describes the
# binary only when the module was built from that same tree, and this
# fleet clones a tree per commit, so pointing -Module at a different
# clone is easy and silent. It has already happened: a run stamped
# 54d4238 measured a module built at 5c9be71, and its anchor went into
# the store under the wrong commit, where a later run would have read
# it as a cross-build comparison it is not.
#
# Reported rather than refused, because measuring one build's module
# with a later build's harness is a legitimate thing to want. What is
# not legitimate is not knowing which you did.
$moduleRoot = (Resolve-Path $Module).Path
$treeRoot = (Resolve-Path $repoRoot).Path
$commitDescribesModule = $moduleRoot.StartsWith($treeRoot, [StringComparison]::OrdinalIgnoreCase)
if (-not $commitDescribesModule) {
    Write-Host ("PROVENANCE: the module at {0} is outside the tree this script runs from ({1}), " -f
        $moduleRoot, $treeRoot)
    Write-Host ("  so commit {0} names the harness and not the binary being measured." -f $commit)
    Write-Host '  The anchor for this run is stored against that commit, and a later run comparing'
    Write-Host '  against it is comparing harnesses rather than builds.'
}

# The platform, read without naming $IsWindows or $IsLinux: Windows
# PowerShell 5.1 defines neither, and strict mode throws on reading a
# variable that was never set.
$onWindows = $PSVersionTable.PSEdition -eq 'Desktop' -or
    [bool](Get-Variable -Name IsWindows -ValueOnly -ErrorAction SilentlyContinue)
$onLinux = [bool](Get-Variable -Name IsLinux -ValueOnly -ErrorAction SilentlyContinue)
$platform = if ($onWindows) { 'Windows' } elseif ($onLinux) { 'Linux' } else { 'Other' }

$hostInfo = [PSCustomObject]@{
    Machine        = [Environment]::MachineName
    Edition        = $PSVersionTable.PSEdition
    Version        = $PSVersionTable.PSVersion.ToString()
    Platform       = $platform
    ProcessorCount = [Environment]::ProcessorCount
    Commit         = $commit
    Dirty          = $dirty
    ModulePath     = $moduleRoot
    # False means Commit above is the harness's, not the binary's.
    CommitDescribesModule = $commitDescribesModule
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
    $sorted = @($samples | Sort-Object)
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
# The load arm
# ----------------------------------------------------------------------

# A quiet row says what a change costs when nothing else wants the
# cores. The criterion this crate is held to is about the loaded case:
# never slower than the current system, faster under load. A table with
# only the quiet arm cannot speak to it.
#
# Burners are separate processes rather than threads so they contend
# for cores the way a neighbour does, and so a wedged one can be seen
# and killed in a process listing rather than being invisible inside
# this one.

$script:Burners = @()

# The host that runs a burner, and whether it can be asked to hide.
#
# Windows PowerShell is only on Windows, and -WindowStyle is refused
# outright by every other edition rather than ignored, so naming either
# unconditionally ends the run before a single cell is timed. That is
# how this arm first failed on the Linux guest: eight burners asked
# for, none started, and the whole bench gone at the load arm.
$script:BurnerShell = if ($onWindows) { 'powershell' } else { 'pwsh' }
$script:BurnerHides = $onWindows

function Start-Burners {
    if ($LoadThreads -le 0) { return }
    $spin = 'while ($true) { $null = [Math]::Sqrt([Environment]::TickCount) }'
    $arguments = @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-Command', $spin)
    $script:Burners = 1..$LoadThreads | ForEach-Object {
        if ($script:BurnerHides) {
            Start-Process -FilePath $script:BurnerShell -ArgumentList $arguments `
                -WindowStyle Hidden -PassThru
        } else {
            Start-Process -FilePath $script:BurnerShell -ArgumentList $arguments -PassThru
        }
    }
    # Long enough for the scheduler to have placed them, short enough
    # not to dominate the run.
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

# A burner left running would hold the box against every other agent on
# it, so the stop is bound to the script ending rather than only to the
# happy path.
#
# The error is printed and then rethrown by value. A bare `throw` in a
# trap does not rethrow what arrived: it raises ScriptHalted, which
# replaces the diagnostic with a word that names nothing. The first
# version of this trap did exactly that and cost a whole bench run's
# evidence.
#
# The trap covers the whole script, lines above Stop-Burners' definition
# included, so it calls the function only once it exists; otherwise the
# missing function's error would replace the one being reported.
trap {
    if (Get-Command -Name Stop-Burners -CommandType Function -ErrorAction SilentlyContinue) { Stop-Burners }
    Write-Host ("FAULT " + $_.Exception.GetType().Name + ": " + $_.Exception.Message)
    Write-Host ("FAULT_AT " + $_.InvocationInfo.PositionMessage)
    throw $_
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

# The box is warmed until its own readings stop moving, so the first
# cell is timed at the clock the rest of the run will hold rather than
# at whatever the host was doing beforehand.
#
# The warm-up runs the anchor body, which dispatches across the pool,
# because what the run does to the box is multi-core work and that is
# the state the cells have to be timed in. Three seconds of the
# single-threaded control body was tried first: it raised one core's
# clock, the anchor read its lowest of five runs, and the closing
# control then read 74 per cent slower than the opening one, because
# the minutes of kernel cells in between heated the package and the
# warm-up had not.
#
# Convergence rather than a duration, since how long a host takes is a
# property of the host: warm in half-second blocks until two
# consecutive block medians agree within WarmupTolerancePct, and report
# what happened either way. A run that never converges is not stopped,
# because a warm-up is not the measurement; the line says so and the
# control at both ends remains the check.
$warmBlocks = @()
$warmClock = [System.Diagnostics.Stopwatch]::StartNew()
$warmSettled = $false
while ($warmClock.Elapsed.TotalSeconds -lt $WarmupMaxSeconds) {
    $block = [System.Diagnostics.Stopwatch]::StartNew()
    $samples = @()
    while ($block.Elapsed.TotalMilliseconds -lt 500) {
        $one = [System.Diagnostics.Stopwatch]::StartNew()
        $null = & $anchorBody
        $one.Stop()
        $samples += $one.Elapsed.TotalMilliseconds
    }
    $block.Stop()
    # A block that fits one call of the anchor holds one sample, and a
    # sort of one element yields a scalar, which has no Count under
    # strict mode.
    $sorted = @($samples | Sort-Object)
    $warmBlocks += $sorted[[int]($sorted.Count / 2)]
    if ($warmBlocks.Count -ge 2) {
        $a = $warmBlocks[-2]
        $b = $warmBlocks[-1]
        if ($a -gt 0 -and ([Math]::Abs($b - $a) / $a * 100.0) -le $WarmupTolerancePct) {
            $warmSettled = $true
            break
        }
    }
}
$warmClock.Stop()
$warmMoved = if ($warmBlocks.Count -ge 2 -and $warmBlocks[-2] -gt 0) {
    [Math]::Round([Math]::Abs($warmBlocks[-1] - $warmBlocks[-2]) / $warmBlocks[-2] * 100.0, 2)
} else { -1 }
Write-Host ("warmed the box for {0:N1} s over {1} block(s); settled={2}, last two blocks differ by {3}%" -f
    $warmClock.Elapsed.TotalSeconds, $warmBlocks.Count, $warmSettled, $warmMoved)

Write-Host 'anchor first, before anything else in this run'
$anchor = Measure-Cell -Body $anchorBody
Start-Cooldown
# PowerShell interprets a script block for its first sixteen calls and
# compiles it after that, and the control is the one body this run
# calls more than sixteen times: measured on Windows PowerShell 5.1,
# its median reads 19.3 ms over calls one to seven, 16.6 over eight to
# fourteen and 15.8 from the fifteenth call on. The warm-up above
# already carries it past that threshold; the count here holds whatever
# the warm-up's duration does not.
$controlFirst = Measure-Cell -Body $controlBody -Warmup 20
Start-Cooldown
$empty = Measure-Cell -Body $emptyBody
Start-Cooldown

# ----------------------------------------------------------------------
# The kernels and their three arms
# ----------------------------------------------------------------------

$n = 200000
# Typed arrays, built by index. A typed array crosses as one pinned
# copy; anything else, including the Object[] that
# `1..$n | ForEach-Object {...}` produces, is read element by element.
# Building the data the convenient way measured the slow path and
# called it the kernel's cost.
$data = [double[]]::new($n)
$other = [double[]]::new($n)
for ($i = 0; $i -lt $n; $i++) {
    $data[$i] = $i % 997
    $other[$i] = ($i * 7) % 501
}
# The same numbers as a boxed Object[], for the cell that prices what
# the convenient way costs.
$dataUntyped = 1..$n | ForEach-Object { [double]($_ % 997) }
# A buffer of its own for the in-place cell, which mutates what it is
# given. Squaring it repeatedly overflows to infinity, which costs the
# same to compute and keeps the cell comparable.
$inPlace = [double[]]::new($n)
for ($i = 0; $i -lt $n; $i++) { $inPlace[$i] = 1.0000001 }
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

# ----------------------------------------------------------------------
# What the boundary costs at the kernels' own width
# ----------------------------------------------------------------------
#
# The empty call above prices a cmdlet with no arguments. It says
# nothing about carrying 200,000 doubles, and without these three the
# array rows read as statements about the pool when most of each one
# is the crossing. That is an instrument whose absence reads as data,
# so it is measured rather than reasoned about.
#
#   in       the array in, one number back. Input crossing plus one
#            add an element.
#   inOut    the array in, the array back. Adds the return.
#   untyped  the same as `in` over a boxed Object[], which is what the
#            convenient way of building an array in the shell gives.

Write-Host 'crossing cells'
$crossingIn = Measure-Cell -Body { Measure-FlynnelReduce -InputObject $data -Operation Sum }
Start-Cooldown
$crossingInOut = Measure-Cell -Body { Invoke-FlynnelMap -InputObject $data -Operation Abs }
Start-Cooldown
$crossingUntyped = Measure-Cell -Body {
    Measure-FlynnelReduce -InputObject $dataUntyped -Operation Sum
}
Start-Cooldown

# ----------------------------------------------------------------------
# What a record per row costs, isolated
# ----------------------------------------------------------------------
#
# Every other kernel's row count is fixed by the hardware or by a path
# list whose input crossing already dominates. The histogram's is Bins,
# which the caller names and nothing else bounds, so it is the one place
# the return shape can be varied with everything else held still: both
# cells bin the same array over the same range and differ only in what
# crosses back. The gap divided by Bins is the per-record cost at this
# host's own width, rather than the 1712 ns the sibling module measured
# on a bare pipeline.

$histBins = 50000
Write-Host 'record-shape cells'
$histRows = Measure-Cell -Body {
    Get-FlynnelHistogram -InputObject $data -Bins $histBins
}
Start-Cooldown
$histArray = Measure-Cell -Body {
    Get-FlynnelHistogram -InputObject $data -Bins $histBins -AsArray
}
Start-Cooldown

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

    # The same arithmetic as Map.Square with the return taken out.
    # Its own buffer, because it mutates what it is given and the
    # shared one feeds every other cell.
    @{ Name = 'Map.Square.InPlace'; Straight = $true
       Flynnel = { Update-FlynnelArray -InputObject $inPlace -Operation Square }
       Serial  = { Update-FlynnelArray -InputObject $inPlace -Operation Square -Plan $serialPlan }
       Native  = { for ($i = 0; $i -lt $n; $i++) { $inPlace[$i] = $inPlace[$i] * $inPlace[$i] } } }

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

    # The only kernel with a fourth arm. -UseIoPool reads the files on
    # the process-wide pool instead of the arena, which costs a String
    # clone and a channel per file against a dispatch that already
    # spreads one task per file. Whether that earns itself back is what
    # this arm is for, and the answer decides three more file kernels
    # that have deliberately not been routed yet.
    @{ Name = 'FileHash'; Straight = $false
       Flynnel = { Measure-FlynnelFileHash -Path $files }
       Serial  = { Measure-FlynnelFileHash -Path $files -Plan $serialPlan }
       Pool    = { Measure-FlynnelFileHash -Path $files -UseIoPool }
       Native  = { Get-FileHash -LiteralPath $files -Algorithm SHA256 } }

    @{ Name = 'FileLine'; Straight = $false
       Flynnel = { Measure-FlynnelFileLine -Path $files }
       Serial  = { Measure-FlynnelFileLine -Path $files -Plan $serialPlan }
       # Wrapped in @(): a one-line file makes Get-Content answer a
       # bare string, which has no Count under strict mode.
       Native  = { $files | ForEach-Object { @(Get-Content -LiteralPath $_).Count } } }

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
       # The separators are cast: an untyped array picks the
       # Split(char[], int) overload and the options argument lands on
       # a count parameter.
       Native  = { $textBody.Split([string[]]@(' ', "`n"),
                       [System.StringSplitOptions]::RemoveEmptyEntries).Count } }

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
    # The control at both ends of this row's own arms, as well as at
    # both ends of the run. The two answer different questions and the
    # run-wide pair cannot answer this one: a ratio between arms is
    # exposed only to what the box did across those arms, which is tens
    # of seconds, while the run-wide figure covers minutes and reads
    # far larger. Reading a row against the run-wide drift discards
    # rows that were never at risk.
    $rowControlFirst = Measure-Cell -Body $controlBody
    Start-Cooldown
    $f = Measure-Cell -Body $k.Flynnel
    Start-Cooldown
    $s = Measure-Cell -Body $k.Serial
    Start-Cooldown
    $nat = Measure-Cell -Body $k.Native
    Start-Cooldown
    # The fourth arm runs on the one kernel that has it, and only when
    # the process has a pool to route to. Without one the cmdlet warns
    # and hashes on the arena, so timing it would put the arena's own
    # figure in the pool column and read as the route costing nothing.
    $pool = $null
    if ($hasIoPool -and $k.ContainsKey('Pool')) {
        $pool = Measure-Cell -Body $k.Pool
        Start-Cooldown
    }
    # Computed before the object, since an if is a statement and cannot
    # sit as a value in a literal.
    $poolMs = $null
    $vsPool = $null
    if ($pool) {
        $poolMs = [Math]::Round($pool.MedianMs, 4)
        # Flynnel is the denominator here as it is for the other two
        # arms, so above one always means the arm is slower than the
        # arena and the column reads the same way across the table.
        if ($f.MedianMs -gt 0) {
            $vsPool = [Math]::Round($pool.MedianMs / $f.MedianMs, 3)
        }
    }
    $rowControlLast = Measure-Cell -Body $controlBody
    $rowDriftPct = if ($rowControlFirst.MedianMs -gt 0) {
        [Math]::Round(
            100.0 * ($rowControlLast.MedianMs - $rowControlFirst.MedianMs) /
                $rowControlFirst.MedianMs, 2)
    } else { $null }
    $rows += [PSCustomObject]@{
        Kernel          = $k.Name
        FlynnelMs       = [Math]::Round($f.MedianMs, 4)
        SerialMs        = [Math]::Round($s.MedianMs, 4)
        NativeMs        = [Math]::Round($nat.MedianMs, 4)
        PoolMs          = $poolMs
        VsSerial        = if ($f.MedianMs -gt 0) {
                              [Math]::Round($s.MedianMs / $f.MedianMs, 3)
                          } else { $null }
        VsNative        = if ($f.MedianMs -gt 0) {
                              [Math]::Round($nat.MedianMs / $f.MedianMs, 3)
                          } else { $null }
        VsPool          = $vsPool
        AnchorSpeaksFor = $k.Straight
        # How far the box moved across this row's own arms. This is a
        # floor under the row's error and not the error itself: the
        # control body is not the kernel, so it carries the box and
        # nothing the kernel brings. BoundPct below is the readable one.
        RowDriftPct     = $rowDriftPct
    }
    Write-Host ("  row drift {0}% across this row's arms" -f $rowDriftPct)
}

$controlLast = Measure-Cell -Body $controlBody

# ----------------------------------------------------------------------
# The load arm
# ----------------------------------------------------------------------
#
# The quiet and loaded passes cannot interleave, because the burners
# are either running or not. So the loaded pass is its own phase with
# its own control at both ends: a loaded row is read against the loaded
# control, never against the quiet one.
#
# The flynnel arm is repeated under load, and so is the IO-pool arm
# where a kernel has one. The serial arm is the same code down one lane
# and the native arm is not this crate, so what the criterion asks
# about is how the arena's own figure moves when the box is contended.
#
# The IO-pool arm belongs here rather than only in the quiet pass
# because contention is the whole of its case: routing reads off the
# arena is a claim about what happens when the arena has other work,
# and a quiet box is the one condition where that claim cannot be
# tested. A route that wins only on an idle machine has not won.

$loadedControlFirst = $null
$loadedControlLast = $null
$loaded = @{}
$loadedPool = @{}
if ($LoadThreads -gt 0) {
    Write-Host ("starting {0} burner(s) for the load arm" -f $LoadThreads)
    Start-Burners
    $loadedControlFirst = Measure-Cell -Body $controlBody
    Start-Cooldown
    foreach ($k in $kernels) {
        Write-Host ("loaded cell {0}" -f $k.Name)
        $loaded[$k.Name] = Measure-Cell -Body $k.Flynnel
        Start-Cooldown
        if ($hasIoPool -and $k.ContainsKey('Pool')) {
            Write-Host ("loaded pool cell {0}" -f $k.Name)
            $loadedPool[$k.Name] = Measure-Cell -Body $k.Pool
            Start-Cooldown
        }
    }
    $loadedControlLast = Measure-Cell -Body $controlBody
    Stop-Burners
    Write-Host 'burners stopped'
}

foreach ($row in $rows) {
    $cell = $loaded[$row.Kernel]
    if ($cell) {
        # Computed before the call: an if is a statement, and as an
        # argument it has to be assigned first or wrapped in $().
        $loadedMs = [Math]::Round($cell.MedianMs, 4)
        $cost = $null
        if ($row.FlynnelMs -gt 0) {
            $cost = [Math]::Round($cell.MedianMs / $row.FlynnelMs, 3)
        }
        $row | Add-Member -NotePropertyName LoadedMs -NotePropertyValue $loadedMs
        $row | Add-Member -NotePropertyName LoadCost -NotePropertyValue $cost
    } else {
        # Named rather than left absent, so a table without the load
        # arm cannot be read as one where load cost nothing.
        $row | Add-Member -NotePropertyName LoadedMs -NotePropertyValue $null
        $row | Add-Member -NotePropertyName LoadCost -NotePropertyValue $null
    }

    # The loaded IO-pool figure, and the arena's loaded figure beside
    # it. Read against each other rather than against the quiet pair:
    # the two passes have different controls and a loaded row compared
    # to a quiet one carries the burners in the ratio.
    $poolCell = $loadedPool[$row.Kernel]
    $loadedPoolMs = $null
    $loadedVsPool = $null
    if ($poolCell) {
        $loadedPoolMs = [Math]::Round($poolCell.MedianMs, 4)
        if ($row.LoadedMs -and $row.LoadedMs -gt 0) {
            $loadedVsPool = [Math]::Round($poolCell.MedianMs / $row.LoadedMs, 3)
        }
    }
    $row | Add-Member -NotePropertyName LoadedPoolMs -NotePropertyValue $loadedPoolMs
    $row | Add-Member -NotePropertyName LoadedVsPool -NotePropertyValue $loadedVsPool
}

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
#
# Windows PowerShell 5.1's ConvertFrom-Json writes a JSON array as one
# object where PowerShell 7 enumerates it; ForEach-Object unrolls it in
# both, so each store reads back as its records.
$store = @()
if (Test-Path $AnchorStore) {
    $store = @(Get-Content -LiteralPath $AnchorStore -Raw | ConvertFrom-Json | ForEach-Object { $_ })
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
# Across runs, which is a different question from across samples
# ----------------------------------------------------------------------
#
# The repeats inside one invocation are samples, not runs. They share a
# process: the same warmed allocator, the same calibration, the same
# code placement. A median over seven of them says how tight this
# process was, not how tight the figure is.
#
# So each run appends its medians here, keyed by commit, machine and
# host edition, and the table reports how many runs are on record and
# how far apart they fell. One run on record prints a null rather than
# a zero spread, because a single point has no spread and a zero would
# read as a tight one.

# Named runRows and not runStore: PowerShell variable names are case
# insensitive, so a local $runStore and the $RunStore parameter are one
# variable, and assigning the local destroys the path the parameter
# held. The same collision is why the rows below read back without
# their properties.
$runRows = @()
if (Test-Path $RunStore) {
    $runRows = @(Get-Content -LiteralPath $RunStore -Raw | ConvertFrom-Json | ForEach-Object { $_ })
}
$runId = [Guid]::NewGuid().ToString('N').Substring(0, 8)
foreach ($row in $rows) {
    $runRows = @($runRows) + [PSCustomObject]@{
        RunId     = $runId
        Commit    = $commit
        Machine   = $hostInfo.Machine
        Edition   = $hostInfo.Edition
        When      = (Get-Date -Format 'o')
        Kernel    = $row.Kernel
        # How many timed samples the medians below are of. A run of one
        # is a single sample wearing the word median, and comparing it
        # with a run of seven measures the difference between two
        # estimators rather than between two runs, so the spread below
        # pairs only runs that took the same number.
        Repeats   = $Repeats
        FlynnelMs = $row.FlynnelMs
        SerialMs  = $row.SerialMs
        VsSerial  = $row.VsSerial
        LoadedMs  = $row.LoadedMs
    }
}
foreach ($row in $rows) {
    $mine = @($runRows | Where-Object {
        $_.Commit -eq $commit -and $_.Machine -eq $hostInfo.Machine -and
        $_.Edition -eq $hostInfo.Edition -and $_.Kernel -eq $row.Kernel -and
        $_.PSObject.Properties.Name -contains 'Repeats' -and $_.Repeats -eq $Repeats -and
        $null -ne $_.FlynnelMs -and $_.FlynnelMs -gt 0
    })
    $row | Add-Member -NotePropertyName Runs -NotePropertyValue $mine.Count
    if ($mine.Count -lt 2) {
        # One point has no spread. A zero here would be read as
        # agreement between runs that never happened.
        $row | Add-Member -NotePropertyName RunSpread -NotePropertyValue $null
    } else {
        $values = @($mine | ForEach-Object FlynnelMs | Sort-Object)
        $row | Add-Member -NotePropertyName RunSpread -NotePropertyValue (
            [Math]::Round($values[-1] / $values[0], 3))
    }
    # The ratio is what a reader acts on, so the ratio is what needs a
    # bound. Its numerator and denominator move together in part, so
    # the spread of the ratio is its own measurement and not something
    # the two arms' spreads can be combined into.
    $ratios = @($mine | Where-Object {
        $null -ne $_.VsSerial -and $_.VsSerial -gt 0
    } | ForEach-Object VsSerial | Sort-Object)
    $ratioSpreadPct = if ($ratios.Count -lt 2) { $null } else {
        [Math]::Round(100.0 * ($ratios[-1] - $ratios[0]) / $ratios[0], 1)
    }
    # The row control moves with the box and the ratio moves with
    # everything the box does not cover, so neither bounds the other and
    # the readable bound is whichever is wider. Where the row has been
    # run once, the control is all there is and the column says so by
    # carrying the control's own figure.
    $bound = if ($null -eq $row.RowDriftPct) { 0.0 } else {
        [Math]::Abs([double]$row.RowDriftPct)
    }
    if ($null -ne $ratioSpreadPct -and $ratioSpreadPct -gt $bound) {
        $bound = $ratioSpreadPct
    }
    $row | Add-Member -NotePropertyName RatioSpreadPct -NotePropertyValue $ratioSpreadPct
    $row | Add-Member -NotePropertyName BoundPct -NotePropertyValue ([Math]::Round($bound, 1))
    # A ratio sits this far from parity; when the bound reaches that far
    # the row has not measured which side of parity it is on, whatever
    # the ratio prints.
    $distancePct = if ($null -ne $row.VsSerial -and $row.VsSerial -gt 0) {
        [Math]::Abs(100.0 * ($row.VsSerial - 1.0))
    } else { $null }
    # Computed before it is handed over, because a parenthesised `if`
    # in an argument position is parsed as a command named `if`, which
    # is a name nothing answers to. It passes a syntax check and fails
    # only when the line runs, and everything in this block runs after
    # the last cell is timed.
    $signHeld = $null
    if ($null -ne $distancePct) { $signHeld = ($distancePct -gt $bound) }
    $row | Add-Member -NotePropertyName SignHeld -NotePropertyValue $signHeld
}
$runRows | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $RunStore -Encoding utf8

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
    CrossingInMs    = [Math]::Round($crossingIn.MedianMs, 4)
    CrossingInOutMs = [Math]::Round($crossingInOut.MedianMs, 4)
    CrossingUntypedMs = [Math]::Round($crossingUntyped.MedianMs, 4)
    RecordBins        = $histBins
    RecordRowsMs      = [Math]::Round($histRows.MedianMs, 4)
    RecordArrayMs     = [Math]::Round($histArray.MedianMs, 4)
    RecordNs          = [Math]::Round(
                            ($histRows.MedianMs - $histArray.MedianMs) * 1e6 / $histBins, 1)
    LoadThreads     = $LoadThreads
    LoadedControlFirstMs = if ($loadedControlFirst) {
                               [Math]::Round($loadedControlFirst.MedianMs, 4)
                           } else { $null }
    LoadedControlLastMs  = if ($loadedControlLast) {
                               [Math]::Round($loadedControlLast.MedianMs, 4)
                           } else { $null }
    # What the burners did to a cell the module cannot reach. Every
    # LoadCost in the table is read against this: a row that slowed by
    # less than the control did was not slowed by the load.
    ControlLoadCost = if ($loadedControlFirst -and $controlFirst.MedianMs -gt 0) {
                          [Math]::Round($loadedControlFirst.MedianMs / $controlFirst.MedianMs, 3)
                      } else { $null }
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
Write-Host ("crossing at {0} elements: {1} ms in, {2} ms in and out, {3} ms in untyped" -f
    $n, $result.CrossingInMs, $result.CrossingInOutMs, $result.CrossingUntypedMs)
Write-Host ("  so a typed array costs {0} ns an element in, and an untyped one {1} ns" -f
    [Math]::Round($result.CrossingInMs * 1e6 / $n, 1),
    [Math]::Round($result.CrossingUntypedMs * 1e6 / $n, 1))
Write-Host ("{0} histogram bins: {1} ms a record each against {2} ms in one, {3}x" -f
    $result.RecordBins, $result.RecordRowsMs, $result.RecordArrayMs,
    $(if ($result.RecordArrayMs -gt 0) {
          [Math]::Round($result.RecordRowsMs / $result.RecordArrayMs, 1)
      } else { 'n/a' }))
Write-Host ("  so a pipeline record costs {0} ns here" -f $result.RecordNs)
if ($null -ne $anchorDriftPct) {
    Write-Host ("anchor {0} ms, {1}% from {2}" -f
        $result.AnchorMs, $anchorDriftPct, $previous.Commit)
} else {
    Write-Host ("anchor {0} ms, no earlier build on this host to compare with" -f
        $result.AnchorMs)
}
if ($LoadThreads -gt 0) {
    Write-Host ("load arm: {0} burner(s), control {1} ms quiet against {2} ms loaded, {3}x" -f
        $LoadThreads, $result.ControlFirstMs, $result.LoadedControlFirstMs,
        $result.ControlLoadCost)
} else {
    Write-Host 'load arm: skipped, LoadThreads is zero'
}
Write-Host ''
$rows | Format-Table Kernel, FlynnelMs, SerialMs, NativeMs, LoadedMs, VsSerial, VsNative,
    LoadCost, RowDriftPct, RatioSpreadPct, BoundPct, SignHeld, Runs, RunSpread,
    AnchorSpeaksFor, CrossBuild -AutoSize
Write-Host ''
Write-Host 'Runs is how many invocations of this script are on record for this commit, host'
Write-Host 'and edition, and RunSpread is the widest over the narrowest of their medians.'
Write-Host 'The repeats inside one invocation are samples sharing a process, not runs; a'
Write-Host 'RunSpread of null means one run is on record and there is nothing to compare.'
Write-Host 'VsSerial is how many times faster the pool is than one worker running the same'
Write-Host 'kernel. VsNative is against the PowerShell way to get the same answer.'
Write-Host "RowDriftPct is the control measured at both ends of that row's own arms, so it"
Write-Host 'is how far the box moved while the row was being taken. It is not the error on'
Write-Host "the row's ratio: the control is a different body, so it carries none of what the"
Write-Host 'kernel itself brings to a reading. Measured on the Linux guest, thirteen of'
Write-Host 'sixteen rows moved further between runs than their own control reported, one of'
Write-Host 'them by 878% against a control that said 11%. The run-wide control drift above'
Write-Host 'covers minutes and bounds comparisons between rows; reading one row against it'
Write-Host 'discards rows that were never exposed to it.'
Write-Host 'RatioSpreadPct is how far VsSerial itself moved across the runs on record, which'
Write-Host 'is the only figure that has the kernel in it. BoundPct is the wider of the two,'
Write-Host 'and SignHeld is false where the ratio sits closer to parity than BoundPct, which'
Write-Host 'means this row has not established which side of parity it is on. A SignHeld of'
Write-Host 'null is a row measured once, where nothing yet bounds the ratio.'
Write-Host 'LoadCost is the same kernel under contention over itself on a quiet box, and is'
Write-Host 'read against ControlLoadCost above: a row that rose by less than the control did'
Write-Host 'was not slowed by the load. A null LoadCost means the arm did not run, which is'
Write-Host 'not the same as a load that cost nothing.'
Write-Host 'AnchorSpeaksFor is false where the inner loop branches on the data, which is'
Write-Host 'the shape the anchor was measured not to stand in for.'
Write-Host ("raw cells in {0}" -f $outFile)

Remove-Item -LiteralPath $work -Recurse -Force -ErrorAction SilentlyContinue
