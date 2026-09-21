---
title: How To Use The PowerShell Module
weight: 3
---

Flynnel as PowerShell commands and objects, bound straight to the Rust with [PWRS](https://crates.io/crates/PoWerRuSt). The cmdlets are the library: nothing here shells out, and no result is parsed from text.

Eighty commands, fifty-four object types and twenty-six enumerations. Every command answers to a shorter name with the `Fly` prefix, so `Measure-FlyReduce` is `Measure-FlynnelReduce`.

Windows x64 and Linux x64 in one module, on PowerShell 7 and Windows PowerShell 5.1.

## Getting it

The module crate is `pwrs/` in this repository. It is not a workspace member and it depends on the scheduler by path, so it builds from its own directory:

```powershell
cargo pwrs build --release --manifest-dir pwrs
Import-Module .\pwrs\target\pwrs\Flynnel\Flynnel.psd1
```

`cargo pwrs test --release --manifest-dir pwrs` runs the Pester suites against the built module in both hosts.

## The one rule

**Anything over many items crosses in one call.**

Measured at this boundary in the sibling SubEtha module's `bench/CallShapes.ps1`, on PowerShell 7.6 and Windows PowerShell 5.1:

| what | 7.6 | 5.1 |
|---|---|---|
| a method call | 1907 ns | 651 ns |
| the same call batched 1000 at a time | 4.5 ns | 3.0 ns |
| one pipeline record | 1712 ns | 7955 ns |
| a ring, one Push or Pop per item | 4848 ns | 2013 ns |
| a ring, 256 items packed into one call | 39 ns | 22 ns |

Dividing the first row by the second gives 424 on 7.6 and 217 on 5.1: that is what batching is worth per item at the boundary itself.

The pipeline row is separate and is paid before any cmdlet body runs. One second divided by 1712 ns is about 584,000 records on 7.6; divided by 7955 ns it is about 126,000 on 5.1. Those are divisions of a measured per-record cost, not a statement about what any particular cmdlet does with a record once it has one.

So every cmdlet and method that can take many items takes them all in one call. Per-record forms exist where a script genuinely wants to interleave with other pipeline stages, and each says what a record costs in its own help.

## Pass a typed array

`[double[]]$x` crosses as one pinned copy. An untyped `@(...)` of boxed objects crosses element by element, and the penalty measured on the two hosts is 185 times on one and 296 on the other.

```powershell
# one pinned crossing
[double[]]$data = 1..200000
$sum = Measure-FlynnelReduce -InputObject $data -Operation Sum

# the same numbers, boxed, and two orders of magnitude more expensive
$sum = Measure-FlynnelReduce -InputObject @(1..200000) -Operation Sum
```

A kernel's answer is also a typed array, so one kernel's output feeds the next without a conversion in between.

## A first path through it

Read the host, build a plan, run a kernel on it.

```powershell
# What the scheduler reads before it sizes anything.
Get-FlynnelTopology
Get-FlynnelCpuInfo

# A plan is the unit every other family takes. Each With method
# answers a fresh plan and leaves the one it was called on alone.
$plan = New-FlynnelPlan -KOuter 8 -BatchSize 200000
$resolved = Resolve-FlynnelPlan -Plan $plan
$resolved.Workers
$resolved.LeafItems

# The pool the plan dispatches into, started deliberately rather than
# on the first dispatch, so the start is not inside a measurement.
Start-FlynnelPool

# The work.
[double[]]$data = 1..200000
Invoke-FlynnelMap -InputObject $data -Operation Sqrt -Plan $plan
```

`-Verbose` on any kernel reports the plan that ran it, the workers it resolved to and the leaves it asked for.

## Rings, when work moves between stages

The scheduler's own in-process queues are bound over a byte payload, because a script has no Rust type to offer.

Two things decide how a script uses them.

**A ring refuses; it does not park.** A full ring hands the item back and the caller retries, backs off or drops it. A push answers `Accepted`, and when it did not, carries the item it refused, so a full ring costs a retry and never costs data.

```powershell
$ring = New-FlynnelRing -Capacity 1024
$outcomes = $ring.PushMany($batch)
$retry = $outcomes | Where-Object { -not $_.Accepted } | ForEach-Object { $_.Item }
```

**The blocking forms are not bound.** The crate carries `push_blocking`, `pop_blocking`, `NotifySender::send` and `NotifyReceiver::recv`; none reaches PowerShell. Each waits inside Rust with no way to see the pipeline's stopping flag, so calling one would hang a host that Ctrl-C cannot reach, and `push_blocking` on a ring nobody drains spins forever by construction. A script that wants to wait writes the loop, where its own `Start-Sleep` and Ctrl-C both work.

```powershell
$deadline = (Get-Date).AddSeconds(30)
while ((Get-Date) -lt $deadline) {
    $got = $consumer.PopMany(256)
    if ($got.Count -eq 0) { Start-Sleep -Milliseconds 5; continue }
    # ... work on $got ...
}
```

The two ends of a single-producer single-consumer ring come back as two objects, because the ring is only correct with one thread on each:

```powershell
$producer, $consumer = New-FlynnelSpscRing -Capacity 1024
```

Every handle carries `Role` and `Index`, so a grid's output is split by role rather than by counting:

```powershell
$all = New-FlynnelComposedMpmc -Capacity 256 -Producers 4 -Consumers 2
$producers = $all | Where-Object Role -eq Producer
$consumers = $all | Where-Object Role -eq Consumer
```

Disposing a handle frees its ring. `Dispose` is the generated one and the garbage collector reaches it too, so a `try`/`finally` is the ordinary idiom and a forgotten handle is not a permanent leak.

## Asking what this machine can do, before asking it to

Four questions answer on any host, including one with no accelerator of any kind. Each is worth asking before the work rather than after it.

```powershell
# Which backends exist, which are reachable, and which this host's own
# sweep found. Three separate columns, because they come apart.
Get-FlynnelBackend | Select-Object Kind, Registered, Available, Detected

# What bounds one piece of GPU work. On Windows this is the timeout
# detection and recovery setting; Applies false means nothing bounds it.
$w = Get-FlynnelPeerWatchdog
if ($w.Applies) { "device work must finish inside $($w.DelaySeconds) s" }
else            { "no watchdog: $($w.Basis)" }

# Which cross-process deque a dispatch of this shape would use, and
# whether a measurement or the fixed rule decided.
Get-FlynnelCrossProcessRoute -ArgsInlineBytes 8 -NDrainThreads 4 -ExpectedBurstSize 32 |
    Select-Object Variant, FromExplicitCell, HeuristicVariant, PayloadFits

# How a wave should keep its frontier, from costs you measured.
Get-FlynnelWavePlan -Width 32 -BarrierNs 4000 -GenerationNs 90000 |
    Select-Object Frontier, RebalanceEvery, CostPerGenerationNs, GlobalCostNs
```

Every one of these is a decision or a reading, never a launch. The watchdog reading loads NVML once and caches it, because neither the hardware nor the driver configuration can change under a running process; the other three touch nothing outside this process.

**A failed read is not an absent answer.** Where the watchdog's driver model or registry read fails, the documented delay is taken and `Basis` names which read failed. A watchdog that is present and treated as absent ends in a device reset; one treated as present only shortens slices.

## Starting a GPU peer

```powershell
if (-not (Get-FlynnelBackend | Where-Object { $_.Kind -eq 'Cuda' -and $_.Available })) {
    throw 'no CUDA driver on this host'
}

try {
    $peer = New-FlynnelGpuPeer -Config (New-FlynnelGpuPeerConfig -Lanes 4 -VramBlocks 32)
    if ($peer.TeamNarrowed) {
        "asked for $($peer.BlocksPerLaneRequested) blocks a lane, ran $($peer.TeamSize)"
    }
    if ($peer.DisplacedForeignContext) {
        'your own CUDA context was replaced by the device primary one'
    }
} finally {
    Remove-FlynnelGpuPeer | Out-Null
}
```

One peer per process: it owns the device context, the mapped region and the resident kernel, and a second start is refused rather than quietly contending. Teardown is a command and not a `Dispose`, because a handle the garbage collector releases would free a device context at a moment nothing chose.

## Measuring where work should run

The hybrid commands do not transform your data; `Invoke-FlynnelMap` does that, and on a host with no device it is faster at it, because it puts the whole pool on the work while a hybrid shape splits one call two ways. What the hybrid commands produce is the placement reading.

```powershell
# Which side this call site has learned to prefer at this size.
1..20 | ForEach-Object { Measure-FlynnelHybridPlacement -Count 65536 -Operation Sqrt } |
    Group-Object Placement | Select-Object Name, Count
```

The first call in a size bucket comes back `Race`: with nothing measured there is nothing to choose on, so both sides run and both are timed. Later calls in the same bucket run one side, and every thirty-second call races again so the model follows drift.

On a host with no registered device the backend side is the CPU backend reached through a thread hand-off, so the model settles on `Cpu`. Every row carries `BackendIsCpu`, so a reading taken that way is never mistaken for a device measurement.

## Reading a number the module gives you

Two conventions matter before quoting anything.

**A counter that has measured nothing says so.** The pool's burst ratio starts at 0.5 with nothing pushed, which is not a measurement of a half-burst workload; `HasPushed` is what tells the two apart. A ring's stat row carries `DepthKnown`, because the SPSC, MPSC, composed and grid handles expose no reader for the ring behind them and a zero depth from one of those would otherwise read as an empty ring.

**A trust verdict says what it is made of.** A stored CPU calibration's `IsTrustworthy` is `Samples` and `Confirmations` both above zero, not a spread test: reproducibility is a property of two draws, and no statistic over one draw's samples substitutes for a second draw agreeing. `OccupancyPerMille` sits beside them, because the spread says whether the samples agreed with each other and only occupancy says whether they agreed on the wrong number because a neighbour held half the machine.

`Get-FlynnelSpread` answers two statistics over a caller's samples rather than one. The spread reads the extremes and one stalled sample moves it; the interquartile range reads the middle half and does not. A run with a wide spread and a narrow range was steady with an interruption in it, which is a different finding from one that was not steady.

## What the module does not do

It never runs a PowerShell script block on a Flynnel worker.

A script block runs only on the thread that owns the pipeline: the binding framework's pipeline token is `!Send`, and the managed side refuses a stream call reached from another thread. Work that runs on the pool is therefore work this module declares: the kernels over arrays, files and text, the accelerator ops, and the bodies the racing and hybrid commands take by name.

The adjacent rule is a contract rather than something the compiler enforces. `PsObject` is `Send + Sync` and its `get` and `pin` call the managed vtable on `&self`, so a `Send + Sync` closure *can* capture one and call it, and one such call attaches that worker to the runtime for the life of the process, making it a GC root suspended at safepoints. So: resolve every managed value before the parallel section and hand the closure plain Rust data.

## Where to look next

- [PowerShell Module Reference](../reference/PowerShell-Module-Reference/): every command by family, with its objects and enumerations.
- [JobPlan Reference](../reference/JobPlan-Reference/): what a plan carries and how it resolves.
- [Environment Variables](../reference/Environment-Variables/): the switches the scheduler reads at startup, including `FLYNNEL_SCHED_SMT_AS_IO` and `FLYNNEL_TRACE`.
