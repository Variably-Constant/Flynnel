---
title: PowerShell Module Reference
weight: 10
---

Every command the Flynnel PowerShell module exports, by family, with the objects and enumerations each one deals in. For the task-oriented introduction see [How To Use The PowerShell Module](../how-to/How-To-Use-The-PowerShell-Module/).

Eighty commands, fifty-four object types, twenty-six enumerations. Every command answers to a shorter name with the `Fly` prefix; the alias is listed beside each.

`Get-Help <command> -Full` carries the parameters, the examples and what each parameter means. This page is the map, not a substitute for it.

## The host - 8 commands

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

## Plans - 5 commands

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

## The pool - 13 commands

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

## Kernels - 17 commands

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

## Observation - 13 commands

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

`Set-FlynnelTraceState` could not exist until the crate stopped latching its flag: the ring was armed by an environment variable read once, and a module cannot set the environment of a process it is already inside. `Get-FlynnelTraceState` reports `EnabledBy` as real provenance: the variable until a setter has decided, and the command after.

`Get-FlynnelSpread` answers two statistics rather than one. The spread reads the extremes and one stalled sample moves it; the interquartile range reads the middle half and does not. A run with a wide spread and a narrow range was steady with an interruption in it, which is a different finding from one that was not steady.

`Get-FlynnelCallSite` is the per-location half: the scheduler keeps a classifier per source location, so two callers of the same kernel with different workloads each get their own rather than averaging into one. `Reset-FlynnelCallSite` clears what they learned, which is what lets two arms of a comparison run in the same process. It supports `-WhatIf` and asks first.

## Calibration - 15 commands

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

## Rings - 9 commands

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

**A ring refuses; it does not park.** A full ring hands the item back and the caller decides. A push answers `Accepted`, and when it did not, carries the item it refused, so a full ring costs a retry and never costs data, including through `Send-FlynnelItem`, which writes a refused item back to the pipeline instead of dropping it.

**A batch stops at the first refusal** rather than skipping past it, because a ring is ordered and carrying on would deliver later items ahead of an earlier one. The outcomes come back one per item attempted, so their count says how far it got.

**The blocking forms are not bound.** `push_blocking`, `pop_blocking`, `Injector::push`, `NotifySender::send` and `NotifyReceiver::recv` each wait inside Rust with no way to see the pipeline's stopping flag; a script calling one would hang a host that Ctrl-C cannot reach, and `push_blocking` on a ring nobody drains spins forever by construction. The try-forms are what is bound.

**`PopKind.Closed` is declared and never answered.** A notify hub's shutdown flag is private and has no reader, and the one call that would distinguish a closed hub from an empty one is a send, which on a live hub enqueues the probe and delivers a phantom item to whichever consumer reaches it. So an exhausted hub reads `Empty`, and a script learns of the shutdown from its own `Shutdown` call or from a send answering `Closed`.

**`DepthKnown` on a stat row** says whether `Depth`, `IsEmpty` and `IsFull` mean anything for that shape. Only the general ring, the injector and the notify handles expose a reader for the ring behind them; on the rest a zero depth would otherwise read as an empty ring.

The single-owner shapes are a contract the module cannot enforce. An SPSC producer and consumer carry a `Cell` cursor and no synchronization because the crate expects exactly one thread on each; handing the same producer to two runspaces corrupts the ring and nothing here can detect it. The two ends come back as two objects for that reason.

## Backends and accelerator ops - 4 commands

What devices this host has, and where an operation would run. This family inspects; it launches nothing.

| command | alias | answers |
|---|---|---|
| `Get-FlynnelBackend` | `Get-FlyBackend` | a row per backend kind, on every host |
| `Test-FlynnelBackend` | `Test-FlyBackend` | runs the probe now rather than at startup |
| `Get-FlynnelAccelOp` | `Get-FlyAccelOp` | every registered operation and its bindings |
| `Get-FlynnelAccelTarget` | `Get-FlyAccelTarget` | where one would route under a plan |

Objects: `Flynnel.Backend`, `Flynnel.BackendProbe`, `Flynnel.AccelOp`, `Flynnel.AccelTarget`. Enumeration: `Flynnel.BackendKind`.

**An absent device is a row, never a missing row.** The module ships with every backend feature on, so the code for each is present everywhere and absence is always a runtime fact.

**Registered, Available and Detected are three questions.** Registered is whether an implementation is in the process registry; Available is whether the host's probe found the runtime; Detected is whether the crate's own sweep listed it. They come apart in both directions: a CUDA runtime loads on a machine with no card, and a consumer-registered backend is registered with no probe having passed. Each row also carries the text of what its probe looked at.

**A capability of zero is not a capability.** An unregistered backend has no implementation to ask, so its capability columns are zeros; `CapabilitiesKnown` separates that from a measured zero, which the CPU backend genuinely has for host-to-device bandwidth.

There is no Register cmdlet for either a backend or an accelerator op. `register_backend` takes an `Arc<dyn DispatchBackend>` and `register_accel_op` takes a Rust closure; a script has neither, and the Get that replaces each says so.

## Hybrid - 4 commands

The CPU half and the device half of one call, and what the scheduler learns from running them.

| command | alias | shape |
|---|---|---|
| `Measure-FlynnelHybridJoin` | `Measure-FlyHybridJoin` | two halves concurrently at a fixed share |
| `Measure-FlynnelHybridPlacement` | `Measure-FlyHybridPlacement` | the side this call site has learned to prefer |
| `Measure-FlynnelHybridSplit` | `Measure-FlyHybridSplit` | a range divided by measured per-item cost |
| `Measure-FlynnelHybridPipeline` | `Measure-FlyHybridPipeline` | a three-stage CPU-device-CPU pipeline |

Objects: `Flynnel.HybridJoin`, `Flynnel.HybridPlacement`, `Flynnel.HybridSplit`, `Flynnel.HybridPipeline`. Enumeration: `Flynnel.Placement`.

**These measure; they do not transform your data.** A hybrid shape splits one call between the calling thread and one backend thread, and `Invoke-FlynnelMap` puts the whole pool on the same work. The work these run is declared and synthetic, sized by `-Count` and weighted by `-Repetitions`, and nothing crosses the boundary but the report.

**With no device registered the backend half is the CPU backend.** Both halves still run concurrently and the timings are real; every row carries `BackendIsCpu` so a reading taken that way is never read as a device measurement.

**Racing is the calibration.** A cold size bucket runs both sides and times each, a warm bucket runs only the cheaper one, and every thirty-second call re-races so the model tracks drift. A bucket pays double work once rather than needing an offline pass.

**The learned state belongs to the cmdlet, not to your script.** The site is this module's own source location, so every caller in a session shares one per size bucket. `Get-FlynnelCallSite` reads it and `Reset-FlynnelCallSite` clears it.

**A cold size does not start even.** `Measure-FlynnelHybridSplit` reads the site's overall ratio at a size it has no data for, so an even first share means the site as a whole is even. The model records per-item cost as a whole number of nanoseconds, so at the default of one repetition both sides truncate to the same integer and no difference can be resolved; `-BackendRepetitions` makes the backend side dearer per item, which is how the model can be seen to respond on a host with no device.

## The GPU peer - 7 commands

The GPU joins the scheduler as a shared-memory peer over a driver-registered mapped region.

| command | alias | answers |
|---|---|---|
| `Get-FlynnelPeerWatchdog` | `Get-FlyPeerWatchdog` | what bounds a piece of device work, on any host |
| `Get-FlynnelWavePlan` | `Get-FlyWavePlan` | how a wave should keep its frontier |
| `Get-FlynnelLinalgMethod` | `Get-FlyLinalgMethod` | which method a decomposition of this size and batch takes |
| `New-FlynnelGpuPeerConfig` | `New-FlyGpuPeerConfig` | the settings a peer starts from |
| `New-FlynnelGpuPeer` | `New-FlyGpuPeer` | starts the peer |
| `Get-FlynnelGpuPeer` | `Get-FlyGpuPeer` | the live peer, or a row saying there is none |
| `Remove-FlynnelGpuPeer` | `Remove-FlyGpuPeer` | tears it down now |

Objects: `Flynnel.PeerWatchdog`, `Flynnel.WavePlan`, `Flynnel.LinalgChoice`, `Flynnel.GpuPeerConfig`, `Flynnel.GpuPeer`. Enumerations: `Flynnel.DriverModel`, `Flynnel.Frontier`, `Flynnel.LinalgOp`.

**Three of these need no device.** The watchdog reading is the driver model and the registry, the wave planner is a cost model over numbers the caller supplies, and the linalg method is a choice made from the size and the batch. All three answer the same on a machine with no card, which is what makes them the ones a script can rely on before a peer exists.

**A failed read is treated as covered.** Where the driver model or the TDR setting cannot be read, the documented delay is taken and `Basis` says which read failed. The direction is deliberate: a watchdog that is present and treated as absent ends in a device reset, while one treated as present only shortens slices.

**`DelayNs` is null, never zero.** A bound of two seconds and no bound at all must not be the same column value, and `Applies` is what a script branches on.

**One peer per process.** It owns a device context, a mapped region and a resident kernel; a start made while one runs is refused. Teardown is a cmdlet rather than a Dispose on a handle, because a handle released by the garbage collector would free a device context at a moment nothing chose.

**Starting a peer replaces a caller's own CUDA context.** The peer works on the device primary context. `DisplacedForeignContext` says when that happened, so a consumer learns it here rather than at a launch far from the cause.

**The team width is clamped to the device.** A team wider than the device loses ranks at its barrier. `TeamSize` is what ran, `BlocksPerLaneRequested` is what was asked for, and `TeamNarrowed` says whether the clamp fired.

## Verification and the mode region - 4 commands

| command | alias | answers |
|---|---|---|
| `New-FlynnelVerifyChain` | `New-FlyVerifyChain` | a chain that roots a sequence of chunks |
| `Compare-FlynnelVerifyChain` | `Compare-FlyVerifyChain` | whether two chains agree, and where they first do not |
| `Get-FlynnelMatrixBackend` | `Get-FlyMatrixBackend` | every registered tile backend |
| `Test-FlynnelModeRegion` | `Test-FlyModeRegion` | whether a region's exit pairs its enter |

Objects: `Flynnel.VerifyChain`, `Flynnel.VerifyComparison`, `Flynnel.MatrixBackend`, `Flynnel.ModeRegionCheck`. Enumeration: `Flynnel.VerifyHasher`.

Methods on a chain: `Add`, `AddMany`, `Root`.

**The comparison names an index, not just a disagreement.** The crate's chain answers a root and nothing else, so the module keeps a digest of each chunk beside it. A chain answers its root once and holds it, because the crate's `finalize` consumes the hasher and a second call would answer thirty-two zero bytes.

**A prefix is not a corrupted chunk.** Two chains of different lengths that agree on everything they both hold report `LengthsDiffer` with no diverging index.

**Two hashers root differently over the same bytes.** A comparison across them says nothing about the traces, so `HashersDiffer` is a column and the cmdlet warns.

**`Get-FlynnelMatrixBackend` writes one row today, the scalar fallback.** The crate carries the CGRA substrate and no tile backend implements it yet, so a host with AMX or SME has nothing here to select. That is a row saying so rather than an empty listing, which would read as a family that failed to enumerate.

## Levers - 5 commands

The runtime switches, the width this process is allowed, and which stored calibration it will serve.

| command | alias | shape |
|---|---|---|
| `Get-FlynnelLever` | `Get-FlyLever` | every switch, or one by name |
| `Set-FlynnelLever` | `Set-FlyLever` | writes one, then reads back what took effect |
| `Get-FlynnelAllowedWidth` | `Get-FlyAllowedWidth` | the width this process may use |
| `Get-FlynnelOccupancyFloor` | `Get-FlyOccupancyFloor` | the floor as a lever row |
| `Get-FlynnelServePolicy` | `Get-FlyServePolicy` | which stored calibration is served to peers |

Objects: `Flynnel.Lever`, `Flynnel.AllowedWidth`, `Flynnel.ServePolicyState`.

**A lever resolves once, on first read, and stays resolved for the life of the process.** Each one is held in a `OnceLock`. Setting the environment variable after something has read it changes the variable and not the behavior.

**So `Set-FlynnelLever` writes, reads the effective value back, and warns by name when the two disagree.** That disagreement is the case where the lever had already resolved. `Get-FlynnelLever` reports the same comparison as a column, so a script can see before it sets.

**Reading resolves too.** A `Get` on a lever nothing has touched fixes it at whatever the variable says at that moment. Set first, then read.

**Agreement is not proof the lever is still unresolved.** Disagreement proves it resolved before the variable was last written; agreement is consistent with both. The column is named for what it measures.

## Cross-process - 3 commands

Work that crosses a process boundary, and how it is routed there.

| command | alias | shape |
|---|---|---|
| `Get-FlynnelCrossProcessVariant` | `Get-FlyCrossProcessVariant` | which deque variant a shape would use |
| `Get-FlynnelCrossProcessRoute` | `Get-FlyCrossProcessRoute` | the routing table behind that answer |
| `Get-FlynnelPassRegistry` | `Get-FlyPassRegistry` | the passes this process has registered |

Objects: `Flynnel.DequeVariantInfo`, `Flynnel.CrossProcessRoute`, `Flynnel.PassRegistry`.

**The wire carries an id, never code.** A cross-process job cannot carry a closure, because the peer cannot dereference a pointer into this process's heap. It carries `(closure_id, args)` and the peer looks the id up in its own pass registry. That is the same shape the accelerator ops use at the device boundary, and it is what makes the family reachable from a script at all: a script names a pass the peer already holds.

**All three answer without a peer process existing.** The variant a dispatch of a given shape would use is decided from the shape and the host, which is what a script sizing a dispatch wants before starting one; the registry reading is a reading of this process.

**Submitting work to a peer is not bound.** Nor is the calibration that re-measures the routing table. Both need a second process.

## Racing - 2 commands

Several attempts at one piece of work, and what taking the first of them buys.

| command | alias | shape |
|---|---|---|
| `Measure-FlynnelRaceAny` | `Measure-FlyRaceAny` | keeps the first arm home, signals the rest to stop |
| `Measure-FlynnelExploreSelect` | `Measure-FlyExploreSelect` | runs every arm to the end, picks the fastest |

Objects: `Flynnel.RaceOutcome`, `Flynnel.RaceArm`.

**They answer opposite questions and are worth running as a pair.** The race reports `TailRatio`, the slowest arm's time over the winner's, which is what hedging trimmed on this host; 1.0 is a race that saved nothing. The exploration cancels nothing, because a slow explorer that finds the best answer is the point of that shape, so what it reports is what exploring costs.

**The call returns when every arm has returned.** Cancelling a loser stops it spending more; it does not hand the call back early. `SlowestArmNs` is what the call actually waited for.

**`CancelledEarly` zero means different things in the two shapes.** On a race it means every loser finished before the winner's signal reached it, which is a fact about how even this host is. On an exploration it is the shape: nothing is canceled. Raise `-Count` or `-Repetitions` to give a signal time to arrive.

**Only two of the crate's nine racing entry points are bound.** The other seven need an arm that can decline a contract it failed, refute a peer, or disagree with one. Every body this module can offer is a declared deterministic kernel, so a cmdlet over `race_agree` would always answer unanimous - a property of the binding rather than of the work.

## The Flynnel drive

A running scheduler, browsed. `Import-Module` creates it; there is nothing to mount.

```
Flynnel:\
  host\         topology, cpu, latency, cache
  pool\         summary, spin, split, workers\<n>
  sites\        one per call site the scheduler has materialised
  backends\     one per backend kind, present on this host or not
  calibration\  summary, thresholds
  trace\        state
  peer\         summary, while a GPU peer is running
```

**Read only, and by not implementing rather than by refusing.** There is no `New-Item`, `Remove-Item` or `Set-Content`: the binding framework's own defaults answer an error for every write. A drive that could change the scheduler would be a second way to do what the `Set-` commands already do, and two ways to write one setting is how they drift apart.

**Every leaf is the object its command writes**, built by the same function - `Flynnel:\host\cpu` answers the same `Flynnel.CpuInfo` as `Get-FlynnelCpuInfo`. The suite compares them field by field, which is what keeps them together.

**A leaf therefore has no `Name`.** A `Flynnel.CpuInfo` carries no such property and adding one would make the drive's object differ from the command's. `PSChildName` is the name, supplied by the engine from the path. Containers do carry `Name`, because their object is the provider's own.

**A level that cannot be read says so.** `peer\` exists on a host with no GPU and enumerates nothing; `host\latency` is a leaf that exists and holds no rows where the ping-pong sweep could not run. A missing path and a missing reading look alike to a script, and only one of them is worth retrying.

**`pool\workers` is built in one pass**, not one call into Rust per child. Its children are matched against the names the level lists, so `5` finds worker five and `05` finds nothing. The external slots a foreign thread pushes through are left out: they are real rows and they are not workers, and `Get-FlynnelWorker -IncludeExternalSlot` is where a caller who wants them asks.

## Conventions across every family

**A counter that has measured nothing says so.** The pool's burst ratio starts at 0.5 with nothing pushed, and `HasPushed` is what tells that from a measured half. A ring's `DepthKnown` does the same job for a depth of zero.

**Many items cross in one call.** Measured at this boundary: a method call costs 1907 ns on PowerShell 7.6 and 651 ns on Windows PowerShell 5.1; the same call batched a thousand at a time costs 4.5 ns and 3.0 ns. One pipeline record costs 1712 ns and 7955 ns respectively, before any cmdlet body runs. Per-record forms exist where a script wants to interleave, and each says what a record costs in its own help.

**No cmdlet takes a script block to run on a worker.** A script block runs only on the thread that owns the pipeline. Work that runs on the pool is work this module declares.

## See also

- [How To Use The PowerShell Module](../how-to/How-To-Use-The-PowerShell-Module/)
- [JobPlan Reference](JobPlan-Reference/)
- [Sched Module Reference](Sched-Module-Reference/)
- [Environment Variables](Environment-Variables/)
