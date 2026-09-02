---
title: Reference
weight: 4
sidebar:
  open: true
---

Information-oriented technical descriptions. Look up a type, a primitive, an env var, or a measured number.

{{< cards >}}
  {{< card link="JobPlan-Reference/" title="JobPlan Reference" subtitle="Every field on the plan struct, the four `DispatchProfile` defaults, the `bare` escape hatch, and the power-user override builders." >}}
  {{< card link="Foundation-Types-Reference/" title="Foundation Types Reference" subtitle="The vocabulary every Flynnel call site speaks: `Variant`, `SchedTier`, `HwClass`, `DispatchProfile`, `OpClass`, `WorkloadClass`, `WorkloadShape`." >}}
  {{< card link="Sched-Module-Reference/" title="Sched Module Reference" subtitle="Every primitive in `flynnel::sched`, organised by Flynn axis. The crate-root re-exports plus the adaptive dispatcher, in-house Chase-Lev / Vyukov / Lamport rings, NotifyHub, Injector, K_gating, WorkloadShape surfaces." >}}
  {{< card link="Backend-System/" title="Backend System" subtitle="The `DispatchBackend` trait + registry, plus the runtime backend-tag swap surface (`migrate_backend`, `resolve_active_backend`). How CPU is the default and how consumers plug CUDA / ROCm / Metal / TPU / WASM / shared-memory." >}}
  {{< card link="Environment-Variables/" title="Environment Variables" subtitle="Every env var Flynnel honors at startup. All optional; defaults are tuned for general-purpose CPU compute." >}}
  {{< card link="Glossary/" title="Glossary" subtitle="Terms specific to Flynnel and the broader extended-Flynn-taxonomy vocabulary." >}}
  {{< card link="Benchmarks/" title="Benchmarks" subtitle="The internal bench harness organized by category (dispatch overhead / data-parallel / deque backing / cross-mode dispatch), with per-host transparency about which `JobPlan` shape generated each cell." >}}
{{< /cards >}}
