# Flynnel for PowerShell

Flynnel is a K-aware, NUMA-aware work-stealing scheduler. This is that
scheduler as PowerShell commands and objects, bound straight to the
Rust with [PWRS](https://crates.io/crates/PoWerRuSt). The cmdlets are
the library. Nothing here shells out to anything.

Ninety-three commands, sixty-five object types and twenty-nine
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

200,000 doubles, in `bench/KernelShapes.ps1`, on both hosts:

| crossing | pc2, ns/elem | Zen 3 guest, ns/elem |
|---|---|---|
| typed array in | 1.3 | 1.6 |
| typed array in and out | 1.8 | 9.4 |
| `Object[]` in | 232.8 | 482.1 |

The penalty for an untyped array is 185 times on one host and 296 on
the other. It is large on both and its size is not a constant, so
read the ratio rather than either number.

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

**A cmdlet whose row count the caller names offers both shapes.**
`Get-FlynnelHistogram` answers one `Flynnel.HistogramBin` per bin,
which is what filtering and formatting want, or with `-AsArray` one
`Flynnel.Histogram` carrying the whole count array and the range
beside it. Bins is a number the caller picks and nothing else bounds,
so at a large one the records cost more than the binning.

The same binning over the same array both ways, so the only difference
is what crosses back. Two hosts, each with its control drift beside it
because a row without one is not readable:

| 50,000 bins | pc2, PowerShell 7.6 | Zen 3 guest, pwsh 7.6.5 |
|---|---|---|
| one record per bin | 22.29 ms | 37.29 ms |
| one record for the histogram | 5.14 ms | 3.82 ms |
| times | 4.3 | 9.8 |
| one pipeline record | 343 ns | 669 ns |
| control drift over the run | 1.75% | -2.6% |

**A record costs about twice as much on one of these hosts as on the
other**, so 343 ns is a fact about pc2 rather than about the module,
and the switch is worth more where records are dearer. Both figures
are the marginal cost of a record carrying a real object, measured
against its own alternative in the same process. The 1712 ns in the
table above is a different thing again: a bare pipeline record with
nothing behind it, from the sibling module's bench.

**A kernel that only reads can change the array in place.**
`Update-FlynnelArray` runs the same operations as `Invoke-FlynnelMap`
over a pin of the caller's own buffer and writes nothing back. It
needs a typed array and it refuses anything else rather than quietly
copying.

What the three together are worth, same bench, 200,000 doubles:

| kernel | first measured | now | times |
|---|---|---|---|
| Measure-FlynnelReduce Sum | 38.87 ms | 0.11 ms | 353 |
| Invoke-FlynnelZip Add | 145.90 ms | 0.48 ms | 304 |
| Invoke-FlynnelMap Square | 114.62 ms | 0.38 ms | 302 |
| Update-FlynnelArray Square | 114.62 ms | 0.09 ms | 1273 |

Three separate things got it there: the return is one array instead of
200,000 records, the input is a pinned typed array instead of a boxed
collection, and the work is dispatched in chunks of at least 256
elements instead of one task per element.

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
`Get-FlynnelHistogram`, `Get-FlynnelDotProduct`, `Invoke-FlynnelSort`
(which also answers to `Sort-FlynnelArray` and `Sort-FlyArray`).

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
`Get-FlynnelThreadTick`, `Get-FlynnelReducePath`, `Get-FlynnelSpread`,
`Get-FlynnelCallSite`, `Reset-FlynnelCallSite`,
`Set-FlynnelTraceState`.

`Set-FlynnelTraceState` is the one that could not exist until the
crate stopped latching its flag: the ring was armed by an environment
variable read once, and a module cannot set the environment of a
process it is already inside.

`Get-FlynnelSpread` answers two statistics over a caller's samples,
not one. The spread reads the extremes and one stalled sample moves
it; the interquartile range reads the middle half and does not. A run
with a wide spread and a narrow range was steady with an interruption
in it, which is a different finding from one that was not steady.

`Get-FlynnelCallSite` is the per-location half: the scheduler keeps a
classifier per source location, so two callers of the same kernel with
different workloads each get their own rather than averaging into one.
`Reset-FlynnelCallSite` clears what they learned, which is what lets
two arms of a comparison run in the same process.

**Calibration.** What the scheduler measured about this host, what it
decided from it, and how to make it measure again.
`Get-FlynnelCalibration` answers all of it in one call.
`Get-FlynnelClassThreshold`, `Get-FlynnelHostDispatch`,
`Get-FlynnelKGating`, the seed-hysteresis pair,
`Get-FlynnelWorkloadClass`, the `Measure-` forms of each, and the
persisted store: `Get-FlynnelHostStamp`,
`Get-FlynnelCalibrationStore`, `Get-FlynnelCpuCalibration`,
`Get-FlynnelAccelCalibration`, `Clear-FlynnelCalibrationStore`.

The store is shared by every process on the host, so clearing it asks
first. It does not delete the file: other processes hold it mapped,
and it is cleared by publishing a zeroed record under the same writer
lease and the same lock every reader uses.

A stored CPU record says what its trust verdict is made of.
`IsTrustworthy` is `Samples` and `Confirmations` both above zero, not
a spread test, because reproducibility is a property of two draws and
no statistic over one draw's samples substitutes for a second draw
agreeing. `OccupancyPerMille` is beside them: the spread says whether
the samples agreed with each other, and only occupancy says whether
they agreed on the wrong number because a neighbour held half the
machine.

**Rings.** The scheduler's own in-process queues, each bound over a
byte payload because a script has no Rust type to offer.
`New-FlynnelRing` is the general multi-producer multi-consumer shape;
`New-FlynnelSpscRing` is the cheapest, with no compare-and-swap on
either side; `New-FlynnelMpscRing` shares one ring between producers
and `New-FlynnelComposedMpsc` gives each its own;
`New-FlynnelComposedMpmc` is the N-by-M grid of them.
`New-FlynnelInjector` is the fork queue on its own and
`New-FlynnelNotifyRing` is the hub that wakes a parked consumer.
`Send-FlynnelItem` and `Receive-FlynnelItem` are the pipeline forms.

Two things decide how a script uses them.

A ring **refuses**; it does not park. A full ring hands the item back
and the caller retries, backs off or drops it, so a slow stage does
not slow its upstream by itself. Every push answers `Accepted` and,
when it did not, carries the item it refused - a full ring costs a
retry and never costs data, including through `Send-FlynnelItem`,
which writes a refused item back to the pipeline.

And the blocking forms the crate carries are deliberately **not**
bound. `push_blocking`, `pop_blocking`, `NotifySender::send` and
`NotifyReceiver::recv` each wait inside Rust with no way to see the
pipeline's stopping flag, so calling one from a script would hang a
host that Ctrl-C cannot reach. A script that wants to wait writes the
loop, where its own `Start-Sleep` and Ctrl-C both work.

The two ends of an SPSC ring come back as two objects rather than
one, because the ring is only correct with one thread on each and two
objects put that in the script's hands rather than in a doc comment.
Each handle carries `Role` and `Index`, so the grid's output is split
with `Where-Object Role -eq Producer` rather than by counting.

**Backends.** What devices this host has and what each can do.
`Get-FlynnelBackend` writes a row for every backend the build carries,
on every host. `Test-FlynnelBackend` runs the host's probe again.
`Get-FlynnelAccelOp` lists the accelerator operations registered in
the process and `Get-FlynnelAccelTarget` says where one would run
under a plan, without running it.

An absent device is a row saying so, never a missing row. This module
ships with every backend feature on, so the code is compiled in
everywhere and absence is always a runtime fact; a missing row would
read exactly like a capability nobody bound.

Three columns, because they are three questions and they come apart
in both directions. `Registered` is whether an implementation is in
the process registry. `Available` is whether the host's probe finds
the runtime, and a CUDA runtime can load on a machine with no card.
`Detected` is whether the crate's own sweep listed it. Each row also
carries the `Probe` text, so a reading can be argued with rather than
only believed.

`CapabilitiesKnown` says whether the four capability columns came
from an implementation at all. Zero is otherwise ambiguous: an
unregistered backend has none to ask, and the CPU backend has a
genuine measured zero for host-to-device bandwidth.

There is no `Register-FlynnelBackend`. Registering takes a Rust
implementation of the backend trait, and registering an accelerator
operation takes its CPU implementation as a closure; a script has
neither. Both absences are decisions, not gaps.

**Levers.** The runtime switches, the width this process may use, and
the policy deciding which stored calibration it serves.
`Get-FlynnelLever` writes every switch with the value in force, the
variable that sets it, its default and what it was measured to cost.
`Set-FlynnelLever` writes one. `Get-FlynnelAllowedWidth`,
`Get-FlynnelServePolicy` and `Get-FlynnelOccupancyFloor` read the
three that carry more than a boolean.

**Every lever latches on its first read** and holds for the life of
the process. Setting the variable after that changes the variable and
not the behavior, which is why `Set-FlynnelLever` does not simply
write and return: it writes, reads the effective value back, and warns
by name when the two disagree. The row it returns is what is in force,
never what was asked for.

Reading also resolves. A switch nothing has touched is fixed at
whatever its variable says the moment you read it, so a script that
means to change one sets it first.

No cmdlet claims to know whether a lever has already resolved, because
nothing outside the function owning a `OnceLock` can ask it.
`EffectiveMatchesVariable` is what they report instead: false is proof
the lever resolved before the variable was last written, and true is
not proof of the opposite. The column is named for what it measures.

Each switch that ships on carries its measured price with the
conditions of the measurement. The two that ship off carry none, which
is the honest state rather than an omission.

**Verification.** `New-FlynnelVerifyChain` makes a chain that absorbs
chunks and answers one 32-byte root; two chains fed the same bytes in
the same order root the same, which is how a CPU trace and a device
trace are checked for being bit-exact without holding both in memory.
`Compare-FlynnelVerifyChain` says not just that two chains disagree
but where: the first index whose chunks differ.

The index is exact. The crate's chain answers a root and nothing
else, so the module keeps a BLAKE3 digest of each chunk beside it:
32 bytes a chunk, and the chunks themselves are not kept. Two chains
of different lengths that agree on everything they both hold is a
different finding from a chunk that differs, and the row separates
them.

A chain answers its root once and then holds it. The crate's
`finalize` consumes the hasher, so asking twice would answer
thirty-two zero bytes, which reads as a root and is not one.

**The mode region.** `run_in_region` enters a tile mode, runs a
bounded body and exits, with the exit paired to the enter by a guard
that fires on an unwinding panic as well as a normal return.
`Test-FlynnelModeRegion` checks that pairing rather than assuming it,
including through a body that panics on purpose, because a guard that
has never seen a panic is untested. It counts through a backend of
the module's own: the scalar fallback's enter and exit are no-ops, so
a run through it cannot tell a paired exit from no exit at all.

`Get-FlynnelMatrixBackend` writes one row today, the scalar fallback.
The crate carries the substrate and no tile backend implements it
yet, so a host with AMX or SME has nothing here to select, and that
is a row saying so rather than an empty listing, which would read as
a family that failed to enumerate.

**The hybrid shapes, which measure rather than transform.**
`Measure-FlynnelHybridJoin` runs two halves of one declared operation
concurrently, the first on the calling thread and the second on the
plan's backend, and reports what each cost and what the pair cost.
`Measure-FlynnelHybridPlacement` runs the side the call site has
learned to prefer at this size and says which that was.
`Measure-FlynnelHybridSplit` divides a range by the per-item
throughputs it has measured and reports the division.
`Measure-FlynnelHybridPipeline` runs a three-stage CPU-device-CPU
pipeline over a sequence and reports the cost per input.

`Measure-` is the verb because these do not transform your data. A
hybrid shape splits one call between the calling thread and one
backend thread; `Invoke-FlynnelMap` puts the whole pool on the same
work. The work these run is declared and synthetic, sized by `-Count`
and weighted by `-Repetitions`, and nothing crosses the boundary but
the report.

With no device registered the backend half is the CPU backend reached
through a thread hand-off. Both halves still run concurrently and the
timings are real; what they are not is a device reading, so every row
carries `BackendIsCpu`.

`Measure-FlynnelHybridSplit` divides by the ratio of the two measured
per-item costs, so on a deviceless host it settles at an even split
and stays there: the backend side is the CPU backend running the same
body, and each side's clock starts inside its own half, so the thread
hand-off falls outside both readings. `-BackendRepetitions` is how to
see the model respond, by making the backend side dearer per item.
The recorded cost is a whole number of nanoseconds, so at the default
of one repetition both sides truncate to the same integer and no
difference can be resolved whatever the real one is.

The placement model races a cold size bucket and times both sides,
runs only the cheaper side once the bucket is warm, and re-races every
thirty-second call so it tracks drift. Racing is the calibration: a
bucket pays double work once rather than needing an offline pass. The
learned state hangs off the caller's source location, which for these
cmdlets is one file, so every script in a session shares one site per
cmdlet keyed by `log2(Count)`. `Get-FlynnelCallSite` reads it and
`Reset-FlynnelCallSite` clears it.

There is no `Invoke-FlynnelAccelOp`. Every accelerator op the crate
registers takes its arguments as host pointers that its CPU
implementation dereferences, under a contract that the buffers stay
live, correctly sized and unaliased for the whole call. Invoking one
from a script means handing over the address of a pinned array.
`Get-FlynnelAccelTarget` answers where an op would route, which is the
part of the question that needs no pointer.

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
