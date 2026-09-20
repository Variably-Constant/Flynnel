---
title: PowerShell Module Reference
weight: 10
---

Every command the Flynnel PowerShell module exports, by family, with the objects and enumerations each one deals in. For the task-oriented introduction see [How To Use The PowerShell Module](../how-to/How-To-Use-The-PowerShell-Module/).

Eighty commands, fifty-four object types, twenty-six enumerations. Every command answers to a shorter name with the `Fly` prefix; the alias is listed beside each.

`Get-Help <command> -Full` carries the parameters, the examples and what each parameter means. This page is the map, not a substitute for it.

## The host — 8 commands

What the scheduler reads before it sizes anything.

| command | alias |
|---|---|
| `Get-FlynnelCpuInfo` | `Get-FlyCpuInfo` |
| `Get-FlynnelTopology` | `Get-FlyTopology` |
| `Get-FlynnelNumaDistance` | `Get-FlyNumaDistance` |
| `Get-FlynnelNodeCpu` | `Get-FlyNodeCpu` |
| `Get-FlynnelLatencyTable` | `Get-FlyLatencyTable` |
| `Get-FlynnelHwClass` | `Get-FlyHwClass` |
| `Get-FlynnelCacheAllocation` | `Get-FlyCacheAllocation` |
| `New-FlynnelCacheReservation` | `New-FlyCacheReservation` |

Objects: `Flynnel.CpuInfo`, `Flynnel.Topology`, `Flynnel.NumaDistance`, `Flynnel.NodeCpus`, `Flynnel.LatencyTable`, `Flynnel.HwClassInfo`, `Flynnel.CacheAllocation`, `Flynnel.CacheReservation`.

Enumerations: `Flynnel.Vendor`, `Flynnel.NumaSource`, `Flynnel.ClusterSource`, `Flynnel.HwClass`.

`Get-FlynnelLatencyTable` is calibrated once per process by a ping-pong sweep and cached, so the first call in a session pays for the sweep and later calls do not.

A host with no resctrl is not an error: `Get-FlynnelCacheAllocation` comes back with `Supported` false and zero counts, because an absent row and an absent facility read alike to a script. `New-FlynnelCacheReservation` returns the ways when the reservation is released, disposed or collected, so a session that ends without releasing does not leave a class of service standing.

## Plans — 5 commands

A plan is the unit every other family takes. Each `With` method answers a fresh plan and leaves the one it was called on alone, so a base plan can be varied without being consumed.

| command | alias |
|---|---|
| `New-FlynnelPlan` | `New-FlyPlan` |
| `Update-FlynnelPlan` | `Update-FlyPlan` |
| `Resolve-FlynnelPlan` | `Resolve-FlyPlan` |
| `Get-FlynnelKBand` | `Get-FlyKBand` |
| `Get-FlynnelDispatchProfile` | `Get-FlyDispatchProfile` |

Objects: `Flynnel.Plan`, `Flynnel.ResolvedPlan`, `Flynnel.ProfileRow`.

Enumerations: `Flynnel.DispatchProfile`, `Flynnel.SchedTier`, `Flynnel.BisectVariant`, `Flynnel.DequeTier`, `Flynnel.LeafShape`, `Flynnel.CooperativeRouting`, `Flynnel.VariantRouting`, `Flynnel.WorkloadShapeKind`.

## The pool — 13 commands

The arena a plan dispatches into, and the dials that change how it waits.

| command | alias |
|---|---|
| `Start-FlynnelPool` | `Start-FlyPool` |
| `Get-FlynnelPool` | `Get-FlyPool` |
| `Get-FlynnelWorker` | `Get-FlyWorker` |
| `Get-FlynnelSpinWindow` | `Get-FlySpinWindow` |
| `Set-FlynnelSpinWindow` | `Set-FlySpinWindow` |
| `Set-FlynnelSpinAdaptive` | `Set-FlySpinAdaptive` |
| `Reset-FlynnelSpinStats` | `Reset-FlySpinStats` |
| `Get-FlynnelSplitMultiplier` | `Get-FlySplitMultiplier` |
| `Set-FlynnelSplitMultiplier` | `Set-FlySplitMultiplier` |
| `Reset-FlynnelSplitStats` | `Reset-FlySplitStats` |
| `Start-FlynnelSplitObserver` | `Start-FlySplitObserver` |
| `New-FlynnelIoPool` | `New-FlyIoPool` |
| `Get-FlynnelIoPool` | `Get-FlyIoPool` |

Objects: `Flynnel.Pool`, `Flynnel.WorkerStat`, `Flynnel.SpinState`, `Flynnel.SplitState`, `Flynnel.IoPool`.

**Every dial here is process-wide.** A script that changes one changes it for everything in the session that dispatches through this module's library. Each `Set-` command reads its value back and writes what is now in force, so a caller never has to assume the write landed.

`Get-FlynnelWorker` writes one row per worker in one call rather than one call per worker. The table runs past the pool's workers into the external slots a foreign thread pushes through; `IsWorker` tells them apart, and emitting them unmarked would give a caller a set of permanently idle workers that do not exist.

`Get-FlynnelSpinWindow` carries `TotalIdleYields` beside the window, so the counter and the window it is judged against come back together.

`Start-FlynnelSplitObserver` runs on the IO pool and resubmits itself each window, so without an IO pool it cannot start. The crate's own start is a silent no-op in that case; this warns instead, because a multiplier nothing is retuning reads exactly like one that is.

`Get-FlynnelIoPool` writes a warning and nothing when the process has no global IO pool, rather than an empty pool: an empty pool and an absent one are different states and a null answer cannot say which.

## Kernels — 17 commands

The work that actually runs on the pool. Each takes an optional `-Plan`, and `-Verbose` reports the plan that ran the work, the workers it resolved to and the leaves it asked for.

Arrays and numbers:

| command | alias |
|---|---|
| `Invoke-FlynnelMap` | `Invoke-FlyMap` |
| `Update-FlynnelArray` | `Update-FlyArray` |
| `Invoke-FlynnelZip` | `Invoke-FlyZip` |
| `Measure-FlynnelReduce` | `Measure-FlyReduce` |
| `Get-FlynnelPrefixSum` | `Get-FlyPrefixSum` |
| `Get-FlynnelHistogram` | `Get-FlyHistogram` |
| `Get-FlynnelDotProduct` | `Get-FlyDotProduct` |
| `Sort-FlynnelArray` | `Sort-FlyArray` |

Files:

| command | alias |
|---|---|
| `Measure-FlynnelFileHash` | `Measure-FlyFileHash` |
| `Test-FlynnelFileHash` | `Test-FlyFileHash` |
| `Search-FlynnelFile` | `Search-FlyFile` |
| `Measure-FlynnelFileLine` | `Measure-FlyFileLine` |
| `Measure-FlynnelFileByte` | `Measure-FlyFileByte` |

Text:

| command | alias |
|---|---|
| `Search-FlynnelText` | `Search-FlyText` |
| `Measure-FlynnelTextCount` | `Measure-FlyTextCount` |
| `Split-FlynnelText` | `Split-FlyText` |
| `Update-FlynnelText` | `Update-FlyText` |

Objects: `Flynnel.Reduction`, `Flynnel.HistogramBin`, `Flynnel.Histogram`, `Flynnel.FileHash`, `Flynnel.HashCheck`, `Flynnel.FileMatch`, `Flynnel.FileMeasure`, `Flynnel.TextMatch`, `Flynnel.TextMeasure`.

Enumerations: `Flynnel.MapOp` (fourteen element-wise operations), `Flynnel.ZipOp` (six pairwise), `Flynnel.ReduceOp` (seven reductions), `Flynnel.TextTransform`.

`Invoke-FlynnelMap` needs a typed array and refuses anything else rather than quietly boxing: the measured penalty for an untyped array is 185 times on one host and 296 on the other.

`Get-FlynnelHistogram` takes `-AsArray` to answer bare counts instead of one `Flynnel.HistogramBin` record per bin. At 50,000 bins the record shape is the pipeline's cost, not the kernel's.

## Observation — 13 commands

The dispatch counters, the pool's leaf statistics, occupancy, call sites, and which shape the last reduce took.

| command | alias |
|---|---|
| `Get-FlynnelTraceState` | `Get-FlyTraceState` |
| `Set-FlynnelTraceState` | `Set-FlyTraceState` |
| `Get-FlynnelTrace` | `Get-FlyTrace` |
| `Clear-FlynnelTrace` | `Clear-FlyTrace` |
| `Request-FlynnelTraceFlush` | `Request-FlyTraceFlush` |
| `Get-FlynnelLeafStat` | `Get-FlyLeafStat` |
| `Reset-FlynnelLeafStat` | `Reset-FlyLeafStat` |
| `Measure-FlynnelOccupancy` | `Measure-FlyOccupancy` |
| `Get-FlynnelThreadTick` | `Get-FlyThreadTick` |
| `Get-FlynnelReducePath` | `Get-FlyReducePath` |
| `Get-FlynnelSpread` | `Get-FlySpread` |
| `Get-FlynnelCallSite` | `Get-FlyCallSite` |
| `Reset-FlynnelCallSite` | `Reset-FlyCallSite` |

Objects: `Flynnel.TraceState`, `Flynnel.TraceCounters`, `Flynnel.LeafStatRow`, `Flynnel.Occupancy`, `Flynnel.Spread`, `Flynnel.CallSite`.

Enumerations: `Flynnel.NoReadingReason`, `Flynnel.ReducePath`.

`Set-FlynnelTraceState` could not exist until the crate stopped latching its flag: the ring was armed by an environment variable read once, and a module cannot set the environment of a process it is already inside. `Get-FlynnelTraceState` reports `EnabledBy` as real provenance — the variable until a setter has decided, and the command after.

`Get-FlynnelSpread` answers two statistics rather than one. The spread reads the extremes and one stalled sample moves it; the interquartile range reads the middle half and does not. A run with a wide spread and a narrow range was steady with an interruption in it, which is a different finding from one that was not steady.

`Get-FlynnelCallSite` is the per-location half: the scheduler keeps a classifier per source location, so two callers of the same kernel with different workloads each get their own rather than averaging into one. `Reset-FlynnelCallSite` clears what they learned, which is what lets two arms of a comparison run in the same process. It supports `-WhatIf` and asks first.

## Calibration — 15 commands

What the scheduler measured about this host, what it decided from it, and how to make it measure again.

| command | alias |
|---|---|
| `Get-FlynnelCalibration` | `Get-FlyCalibration` |
| `Get-FlynnelClassThreshold` | `Get-FlyClassThreshold` |
| `Measure-FlynnelClassThreshold` | `Measure-FlyClassThreshold` |
| `Get-FlynnelHostDispatch` | `Get-FlyHostDispatch` |
| `Measure-FlynnelHostDispatch` | `Measure-FlyHostDispatch` |
| `Get-FlynnelKGating` | `Get-FlyKGating` |
| `Measure-FlynnelKGating` | `Measure-FlyKGating` |
| `Get-FlynnelSeedHysteresis` | `Get-FlySeedHysteresis` |
| `Set-FlynnelSeedHysteresis` | `Set-FlySeedHysteresis` |
| `Get-FlynnelWorkloadClass` | `Get-FlyWorkloadClass` |
| `Get-FlynnelHostStamp` | `Get-FlyHostStamp` |
| `Get-FlynnelCalibrationStore` | `Get-FlyCalibrationStore` |
| `Get-FlynnelCpuCalibration` | `Get-FlyCpuCalibration` |
| `Get-FlynnelAccelCalibration` | `Get-FlyAccelCalibration` |
| `Clear-FlynnelCalibrationStore` | `Clear-FlyCalibrationStore` |

Objects: `Flynnel.Calibration`, `Flynnel.ClassThresholds`, `Flynnel.ThresholdCalibration`, `Flynnel.HostDispatch`, `Flynnel.KGatingResult`, `Flynnel.Stamp`, `Flynnel.StoreInfo`, `Flynnel.CpuRecord`, `Flynnel.AccelRecord`.

Enumerations: `Flynnel.Source`, `Flynnel.KGating`, `Flynnel.AccelKind`, `Flynnel.WorkloadClass`.

`Get-FlynnelCalibration` answers all of it in one call.

The store is shared by every process on the host, so `Clear-FlynnelCalibrationStore` asks first and supports `-WhatIf`. It does not delete the file: other processes hold it mapped, and it is cleared by publishing a zeroed record under the same writer lease and the same lock every reader uses.

A stored CPU record says what its trust verdict is made of. `IsTrustworthy` is `Samples` and `Confirmations` both above zero, not a spread test, because reproducibility is a property of two draws and no statistic over one draw's samples substitutes for a second draw agreeing. `OccupancyPerMille` is beside them: the spread says whether the samples agreed with each other, and only occupancy says whether they agreed on the wrong number because a neighbour held half the machine.

## Rings — 9 commands

The scheduler's own in-process queues, each bound over a byte payload because a script has no Rust type to offer.

| command | alias | shape |
|---|---|---|
| `New-FlynnelRing` | `New-FlyRing` | multi-producer multi-consumer, both ends on one object |
| `New-FlynnelSpscRing` | `New-FlySpscRing` | one producer, one consumer; no compare-and-swap on either side |
| `New-FlynnelMpscRing` | `New-FlyMpscRing` | one shared ring behind n producers |
| `New-FlynnelComposedMpsc` | `New-FlyComposedMpsc` | one dedicated ring per producer, read round-robin |
| `New-FlynnelComposedMpmc` | `New-FlyComposedMpmc` | an n-by-m grid of dedicated rings |
| `New-FlynnelInjector` | `New-FlyInjector` | the fork queue on its own, under the steal protocol |
| `New-FlynnelNotifyRing` | `New-FlyNotifyRing` | a ring that wakes a parked consumer on every send |
| `Send-FlynnelItem` | `Send-FlyItem` | pipeline push |
| `Receive-FlynnelItem` | `Receive-FlyItem` | pipeline pop |

Objects: `Flynnel.Ring`, `Flynnel.SpscProducer`, `Flynnel.SpscConsumer`, `Flynnel.MpscProducer`, `Flynnel.MpscConsumer`, `Flynnel.ComposedConsumer`, `Flynnel.GridProducer`, `Flynnel.GridConsumer`, `Flynnel.Injector`, `Flynnel.NotifySender`, `Flynnel.NotifyReceiver`, plus `Flynnel.PushOutcome`, `Flynnel.PopOutcome` and `Flynnel.RingStat`.

Enumerations: `Flynnel.PushKind`, `Flynnel.PopKind`, `Flynnel.RingRole`.

Methods on a push side: `Push`, `PushMany`, `Stat`. On a pop side: `Pop`, `PopMany`, `Stat`. A ring or an injector, which drives both ends, also has `Len`, `IsEmpty` and `IsFull`. A notify sender adds `Shutdown`. Every handle carries `Id`, `Role` and, where it is one of a set, `Index`.

**A ring refuses; it does not park.** A full ring hands the item back and the caller decides. A push answers `Accepted`, and when it did not, carries the item it refused — so a full ring costs a retry and never costs data, including through `Send-FlynnelItem`, which writes a refused item back to the pipeline instead of dropping it.

**A batch stops at the first refusal** rather than skipping past it, because a ring is ordered and carrying on would deliver later items ahead of an earlier one. The outcomes come back one per item attempted, so their count says how far it got.

**The blocking forms are not bound.** `push_blocking`, `pop_blocking`, `Injector::push`, `NotifySender::send` and `NotifyReceiver::recv` each wait inside Rust with no way to see the pipeline's stopping flag; a script calling one would hang a host that Ctrl-C cannot reach, and `push_blocking` on a ring nobody drains spins forever by construction. The try-forms are what is bound.

**`PopKind.Closed` is declared and never answered.** A notify hub's shutdown flag is private and has no reader, and the one call that would distinguish a closed hub from an empty one is a send, which on a live hub enqueues the probe and delivers a phantom item to whichever consumer reaches it. So an exhausted hub reads `Empty`, and a script learns of the shutdown from its own `Shutdown` call or from a send answering `Closed`.

**`DepthKnown` on a stat row** says whether `Depth`, `IsEmpty` and `IsFull` mean anything for that shape. Only the general ring, the injector and the notify handles expose a reader for the ring behind them; on the rest a zero depth would otherwise read as an empty ring.

The single-owner shapes are a contract the module cannot enforce. An SPSC producer and consumer carry a `Cell` cursor and no synchronization because the crate expects exactly one thread on each; handing the same producer to two runspaces corrupts the ring and nothing here can detect it. The two ends come back as two objects for that reason.

## Conventions across every family

**A counter that has measured nothing says so.** The pool's burst ratio starts at 0.5 with nothing pushed, and `HasPushed` is what tells that from a measured half. A ring's `DepthKnown` does the same job for a depth of zero.

**Many items cross in one call.** Measured at this boundary: a method call costs 1907 ns on PowerShell 7.6 and 651 ns on Windows PowerShell 5.1; the same call batched a thousand at a time costs 4.5 ns and 3.0 ns. One pipeline record costs 1712 ns and 7955 ns respectively, before any cmdlet body runs. Per-record forms exist where a script wants to interleave, and each says what a record costs in its own help.

**No cmdlet takes a script block to run on a worker.** A script block runs only on the thread that owns the pipeline. Work that runs on the pool is work this module declares.

## See also

- [How To Use The PowerShell Module](../how-to/How-To-Use-The-PowerShell-Module/)
- [JobPlan Reference](JobPlan-Reference/)
- [Sched Module Reference](Sched-Module-Reference/)
- [Environment Variables](Environment-Variables/)
