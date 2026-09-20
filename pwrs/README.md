# Flynnel for PowerShell

Flynnel is a K-aware, NUMA-aware work-stealing scheduler. This is that
scheduler as PowerShell commands and objects, bound straight to the
Rust with [PWRS](https://crates.io/crates/PoWerRuSt). The cmdlets are
the library. Nothing here shells out to anything.

Sixty-five commands, thirty-eight object types and twenty-two
enumerations. Every command answers to a shorter name with the `Fly`
prefix: `Measure-FlyReduce` is `Measure-FlynnelReduce`.

Windows x64 and Linux x64 in one module, on PowerShell 7 and Windows
PowerShell 5.1.

## The one rule

Anything over many items crosses the boundary in a single call.

That is not a style preference. It is what the boundary costs, measured
in the sibling module's `bench/CallShapes.ps1`:

| shape | PowerShell 7.6 | Windows PowerShell 5.1 |
|---|---|---|
| one method call | 1907 ns | 651 ns |
| the same call, batched a thousand at a time | 4.5 ns | 3.0 ns |
| one ring push and pop per item | 4848 ns | 2013 ns |
| 256 items packed into one byte array | 39 ns | 22 ns |
| one pipeline record per item | 1712 ns | 7955 ns |

Dividing the first row by the second: batching makes a call 424 times
cheaper per item on PowerShell 7.6 and 217 times cheaper on Windows
PowerShell. Dividing one second by the last row: a cmdlet emitting one
record per item reaches 584 thousand records a second on PowerShell
7.6 and 126 thousand on Windows PowerShell, before anything it calls
has done any work.

So an array arrives as an array:

```powershell
$squared = Invoke-FlynnelMap -InputObject $numbers -Operation Square
$total   = Measure-FlynnelReduce -InputObject $numbers -Operation Sum
```

not one item at a time.

### Two things decide what a call costs

Measured on pc2, PowerShell 7.6, 200,000 doubles, in
`bench/KernelShapes.ps1`:

| crossing | ns per element |
|---|---|
| typed array in | 0.7 |
| typed array in and out | 25.0 |
| `Object[]` in | 235.8 |

**Pass a typed array.** `[double[]]$x` crosses as one pinned copy.
Anything else, including the `Object[]` that
`1..$n | ForEach-Object { ... }` produces, is read element by element
and costs 337 times as much. If the array was built in the shell, cast
it once:

```powershell
$x = [double[]]$x      # once, then every call is on the fast path
```

**A bulk cmdlet answers one array, not a stream.** `$y = Invoke-FlynnelMap ...`
gives the array as before. `Invoke-FlynnelMap ... | ForEach-Object`
receives that array as a single item rather than one element at a time.

The answer is also a typed array, so feeding one kernel's output into
the next stays on the fast path.

What that is worth, same bench, before and after the return was
batched:

| kernel | per-record return | one array | times |
|---|---|---|---|
| Measure-FlynnelReduce Sum | 38.87 ms | 0.13 ms | 299 |
| Get-FlynnelDotProduct | 75.43 ms | 0.19 ms | 397 |
| Get-FlynnelHistogram | 39.11 ms | 0.27 ms | 145 |
| Get-FlynnelPrefixSum | 91.17 ms | 0.68 ms | 134 |
| Sort-FlynnelArray | 97.43 ms | 2.64 ms | 37 |
| Invoke-FlynnelMap Square | 114.62 ms | 5.18 ms | 22 |

## What it does not do

It never runs a PowerShell script block on a Flynnel worker.

A script block runs only on the thread that owns the pipeline. The
binding framework's pipeline token is `!Send`, and the managed side
refuses a stream call reached from another thread. So the work that
runs on the pool is work this module declares: the kernels over arrays,
files and text, and the bodies the other families take by name.

## Getting started

```powershell
Import-Module ./Flynnel/Flynnel.psd1

Get-FlynnelCpuInfo                          # what the scheduler sees
Get-FlynnelTopology                         # nodes, cores, distances
New-FlynnelPlan -KOuter 8 -BatchSize 100000 | Resolve-FlynnelPlan
```

`Resolve-FlynnelPlan` is the one to reach for when the question is what
a plan does on this host. It answers every resolved figure in one call,
where reading them method by method is twenty crossings for one answer.

## The families

**The host.** What the scheduler reads before it sizes anything:
processors, NUMA layout, measured inter-core latency, and the L3 ways
this process can reserve. `Get-FlynnelCpuInfo`, `Get-FlynnelTopology`,
`Get-FlynnelNumaDistance`, `Get-FlynnelNodeCpu`,
`Get-FlynnelLatencyTable`, `Get-FlynnelHwClass`,
`Get-FlynnelCacheAllocation`, `New-FlynnelCacheReservation`.

**Plans.** A plan is the unit every other family takes. Each `With`
method answers a fresh plan and leaves the one it was called on alone,
so a base plan can be varied without being consumed.
`New-FlynnelPlan`, `Resolve-FlynnelPlan`, `Get-FlynnelKBand`,
`Get-FlynnelDispatchProfile`.

**The pool.** The arena, its workers, and the spin, split and IO dials.
Every setter reads its value back and warns when the pool clamped it.
`Start-FlynnelPool`, `Get-FlynnelPool`, `Get-FlynnelWorker`, the spin
and split pairs, `Start-FlynnelSplitObserver`, `New-FlynnelIoPool`,
`Get-FlynnelIoPool`.

**Kernels.** The work that actually runs on the pool.

Arrays and numbers: `Invoke-FlynnelMap` over fourteen element-wise
operations, `Invoke-FlynnelZip` over six pairwise ones,
`Measure-FlynnelReduce` over seven reductions, `Get-FlynnelPrefixSum`,
`Get-FlynnelHistogram`, `Get-FlynnelDotProduct`, `Sort-FlynnelArray`.

Files: `Measure-FlynnelFileHash` and `Test-FlynnelFileHash` in BLAKE3,
`Search-FlynnelFile`, `Measure-FlynnelFileLine`,
`Measure-FlynnelFileByte`.

Text: `Search-FlynnelText`, `Measure-FlynnelTextCount`,
`Split-FlynnelText`, `Update-FlynnelText`.

Each takes an optional `-Plan`, and `-Verbose` reports the plan that
ran the work, the workers it resolved to and the leaves it asked for.

**Observation.** The dispatch counters, the pool's leaf statistics,
occupancy, and which shape the last reduce took.
`Get-FlynnelTraceState`, `Get-FlynnelTrace`, `Clear-FlynnelTrace`,
`Request-FlynnelTraceFlush`, `Get-FlynnelLeafStat`,
`Reset-FlynnelLeafStat`, `Measure-FlynnelOccupancy`,
`Get-FlynnelThreadTick`, `Get-FlynnelReducePath`, `Get-FlynnelSpread`.

**Calibration.** What the scheduler measured about this host, what it
decided from it, and how to make it measure again.
`Get-FlynnelCalibration` answers all of it in one call.
`Get-FlynnelClassThreshold`, `Get-FlynnelHostDispatch`,
`Get-FlynnelKGating`, the seed-hysteresis pair,
`Get-FlynnelWorkloadClass`, the `Measure-` forms of each, and the
persisted store: `Get-FlynnelHostStamp`,
`Get-FlynnelCalibrationStore`, `Get-FlynnelCpuCalibration`,
`Get-FlynnelAccelCalibration`.

## Two conventions worth knowing before you read a number

**An unmeasured figure is null, never zero.** A thread clock the
platform does not have, a leaf spread below the sample floor, a reduce
path on a thread that has not run one: each is null with a reason
beside it. A zero here would read as a measurement, and telling those
apart is the whole value of the column.

**Every calibrated figure says where it came from.** `Source` is
`Measured` only when this module ran the calibration in this process,
and then it carries when. `Stored` came from the persisted table with
the time that record holds. `Default` equals the constant the crate
ships. `Unattributed` is neither, which means something set it and this
module cannot say what.

A getter that measures is named `Measure-`, not `Get-`, with one
exception forced by the crate: `Get-FlynnelHostDispatch` reads a
profile the scheduler measures on its first call in a process, whoever
makes it, so the row says whether the call that produced it is the one
that paid.

## Building and testing

```powershell
cargo pwrs build --release --manifest-dir pwrs
cargo pwrs test  --release --manifest-dir pwrs
```

The suites import the module the build produced, from `PWRS_MODULE`. An
absent variable is a failure rather than a reason to fall back to an
installed copy, which would test something other than the build in
front of you.

`bench/KernelShapes.ps1` times every kernel against its own one-worker
arm, against the PowerShell-native equivalent, and against a control
cell benched at both ends of the run, with a load arm and its own
loaded control, and a fixed anchor whose drift from the previous build
is printed beside each row.

`src/bin/census.rs`, behind the `census` cargo feature, reads the
scheduler's own source and fails when a public item is neither bound
nor recorded in `census.toml` with one of six fixed reasons.

## License

MIT. Copyright (c) 2026 Markus Newton.
