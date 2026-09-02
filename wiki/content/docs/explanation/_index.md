---
title: Explanation
weight: 3
sidebar:
  open: true
---

Discussion that clarifies and illuminates. The "why" behind the code: design choices, taxonomy, comparisons, internals.

{{< cards >}}
  {{< card link="Architecture-Overview/" title="Architecture Overview" subtitle="The layered design from `JobPlan` down to in-house Chase-Lev deques. Includes the adaptive layer (K_gating tag swap, WorkloadClass migration) and why JobPlan::new is adaptive by default." >}}
  {{< card link="Extended-Flynn-Taxonomy/" title="Extended Flynn Taxonomy" subtitle="Why the crate is named after Flynn and how the eight-axis mapping (SISD / SIMD / MISD / MIMD / SIMC / MIMC / SIMT / MIMT) organises every primitive." >}}
  {{< card link="Comparison-To-Other-Schedulers/" title="Comparison To Other Schedulers" subtitle="Flynnel vs rayon, Cilk, tokio, std::thread, crossbeam - where each one fits and where Flynnel's per-call execution-class plan plus in-house lock-free primitives change the math." >}}
  {{< card link="Internals-Work-Stealing/" title="Internals: Work-Stealing Algorithm" subtitle="The low-level mechanics inside the CPU arena: in-house Chase-Lev (chase_lev_local), KHL/Fcl K_inner=3 backings via AdaptiveWorker, FlynnelRing mailbox, Injector, NotifyHub, 4-state latch, JEC sleep protocol. For contributors." >}}
  {{< card link="NUMA-And-Topology/" title="NUMA and Topology" subtitle="How Flynnel probes the host hardware (CPUID, sysfs, GetLogicalProcessorInformationEx) to drive arena partitioning and SMT classification." >}}
  {{< card link="Shared-Memory-Worker-Backend/" title="Shared-Memory Worker Backend" subtitle="The off-process equivalent of `CpuBackend`: same DispatchBackend trait, peer-process execution over a lock-free MMF ring at ~800 ns per call." >}}
{{< /cards >}}
