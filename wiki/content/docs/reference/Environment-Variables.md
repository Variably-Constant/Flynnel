---
title: Environment Variables
weight: 5
---

Flynnel honors the following environment variables at startup. All are optional; defaults are tuned for general-purpose CPU compute on modern hosts.

## Scheduler tuning

### `FLYNNEL_SCHED_WORKERS=N`

Explicit per-NUMA-node worker count. `N` is a positive integer; values less than 1 are ignored. On multi-NUMA hosts the total worker count is `N * num_nodes`.

Default: all logical threads per node (from the node's CPU mask; see [`cpu_info()`](NUMA-And-Topology.md#cpuinfo)).

Use this to:
- Cap worker count below the physical-core count (give other processes headroom).
- Force a specific count for reproducibility (benchmark runs).

### `FLYNNEL_SCHED_PHYSICAL_ONLY=on|1|true`

Restrict the arena to **physical cores only** (SMT siblings stay parked unless a `JobPlan::with_smt()` request activates them per-call).

Default: **off**. The default pool uses all logical threads, which gives the best out-of-the-box throughput on most workloads. Set this env var when you have IMUL-saturated work (multi-precision arithmetic, tight Karatsuba loops, NTT butterflies) where the SMT sibling contests the same execution port and adds no architectural throughput.

### `FLYNNEL_SCHED_SMT=off|0|false`

Back-compat alias for `FLYNNEL_SCHED_PHYSICAL_ONLY=on`. Set this to opt OUT of SMT-sibling usage.

Default: SMT **on**. Older code that set `FLYNNEL_SCHED_SMT=on` is now a no-op (the new default already activates SMT). Set `FLYNNEL_SCHED_SMT=off` to get the previous conservative-default behavior.

### `FLYNNEL_SCHED_PIN=on|1|true`

Enable per-worker CPU affinity pinning. Each primary worker pins to `core_ids[i % core_ids.len()]` from `core_affinity::get_core_ids()`.

Default: **off** (since 2026-05-16). The previous default was pinning-on; empirical measurement showed pinning hurts wall-clock perf under any concurrent system load because pinned workers cannot migrate to an idle CPU. Set to `on` only on dedicated bench rigs or strict-NUMA experiments.

`FLYNNEL_SCHED_PIN=off` is also recognized explicitly (idempotent with the default).

### `FLYNNEL_SCHED_SMT_AS_IO=on|1|true`

Enable the SMT-sibling [`IoPool`](Sched-Module-Reference.md#io_pool) for non-compute roles.

Default: off (the pool is not created; `global_io_pool()` returns `None`).

When enabled, parks one worker per physical core on the SMT sibling and uses it for:
- Background calibration runs ([`spawn_calibration`](Sched-Module-Reference.md#spawn_calibration))
- BLAKE3 verification of stripe outputs ([`VerifyChain::submit_chunk`](Sched-Module-Reference.md#verifychain))
- Prefetch sweeps ([`prefetch_into_l2` / `prefetch_into_l3`](Sched-Module-Reference.md#prefetch))
- Background memory zeroing ([`bg_zero::prepare`](Sched-Module-Reference.md#bg_zero))
- Background steal-rate observer ([`spawn_observer`](Sched-Module-Reference.md#split_observer))

Do NOT submit compute work to this pool: the IO workers sit on SMT siblings of the compute cores and contest the same IMUL/FMA execution units when both pools are busy.

### `FLYNNEL_SPIN_WINDOW_ROUNDS=N`

Override the JEC sleep protocol's spin-window (the number of extra yield rounds a sleepy worker spins before locking the condvar). Read by [`src/sched/jec_sleep.rs`](https://github.com/markusmcnugen/flynnel/blob/main/src/sched/jec_sleep.rs).

Default: 500 rounds (~500 us on a 16T Zen+ pool), confirmed as the host-best window by a `[100, 150, 200, 300, 500, 800]` sweep on a Colab Xeon Cascade Lake (12T, 2026-06-08): flynnel-default Heavy/100k medians ran 6.81ms at spin=100, 6.40ms at spin=200, 6.34ms at spin=500 (best), 6.35ms at spin=800.

Setting this variable pins the window and turns the adaptive controller off.

### `FLYNNEL_ADAPTIVE_SPIN=1`

Opt in to the adaptive spin-window controller in [`src/sched/jec_sleep.rs`](https://github.com/markusmcnugen/flynnel/blob/main/src/sched/jec_sleep.rs): when parks dominate (a bursty workload keeps missing the window) the controller shrinks the window toward the 8-round floor; when rescues dominate it grows back toward the tuned default. Bounded by the default, so a throughput workload never regresses. Equivalent to calling `set_spin_adaptive(true)` at startup.

Default: off (the fixed 500-round window).

### `FLYNNEL_TRACE=on|1|true`

Enable tracing of scheduler events (job submit, worker steal, latch transitions). Read by [`src/sched/trace.rs`](https://github.com/markusmcnugen/flynnel/blob/main/src/sched/trace.rs).

Default: off. Tracing adds a measurable overhead (atomic increments per event); enable only when diagnosing scheduler behavior.

### `FLYNNEL_TRACE_DISPATCH=<any value>`

Enable per-`join_in_worker` dispatch tracing to stderr. Accumulates three process-wide counters: `JOIN_CALL_COUNT` (number of `join_in_worker` invocations), `JOIN_A_BODY_NS` (cycles spent in the `a` closure), `JOIN_WAIT_NS` (cycles spent in the wait loop after `a` returned). Compute "wait fraction" as `JOIN_WAIT_NS / (JOIN_A_BODY_NS + JOIN_WAIT_NS)`; high fraction means dispatch overhead is the bottleneck. Read by [`src/sched/arena.rs`](https://github.com/markusmcnugen/flynnel/blob/main/src/sched/arena.rs) via `dispatch_trace_enabled()`; the check is cached in a `OnceLock<bool>` so the hot path pays one Relaxed load per call.

Default: off.

### `FLYNNEL_LOCKLATCH_DIAGNOSE=1|true`

Enable per-`LockLatch::wait()` entry/exit diagnostic logging to stderr. Read by [`src/sched/latch.rs`](https://github.com/markusmcnugen/flynnel/blob/main/src/sched/latch.rs) via `locklatch_diagnose_enabled()`; the check is cached in a `OnceLock<bool>` so the hot wait path pays one Relaxed load per call.

Default: off. Enable only when diagnosing external-dispatch fallback wait behavior.

### `FLYNNEL_ENABLE_FLAT_FANOUT=<any value except 0/false/empty>`

Enable the flat-fanout path in `reduce_chunks` ([`src/sched/par_iter.rs`](https://github.com/markusmcnugen/flynnel/blob/main/src/sched/par_iter.rs)). The default path is bisect-only: the bench audit measured flat structurally slower for the characterized reduce_chunks workloads because the per-round `external_dispatch` + `LockLatch` overhead dominates. Set only for in-source A/B experiments.

Default: off.

### `FLYNNEL_REDUCE_CHUNKS_CHUNKS=N`

Bench-driven audit hook: override the reduce_chunks target chunk count. Read by [`src/sched/par_iter.rs`](https://github.com/markusmcnugen/flynnel/blob/main/src/sched/par_iter.rs). Used by the chunk-count investigation harness; production callers do not set this.

Default: unset (use the plan-derived chunk count).

## TPU JAX backend (feature `tpu-jax-reference`)

### `TPU_NAME`

External env var set by Google Cloud TPU VM images. Read by `detect::tpu_available()` as a positive signal that a TPU device should be reachable from this host. Not set or written by Flynnel; honored if present.

## Detection probes

The detection helpers in `flynnel::backend::detect` do not read env vars directly (they probe shared libraries and device files). The probes are cached in process-level `OnceLock`s and run at most once per process per probe.

## Quick reference

| Variable | Default | Effect |
|----------|---------|--------|
| `FLYNNEL_SCHED_WORKERS=N` | (logical threads per node) | Per-node worker count override |
| `FLYNNEL_SCHED_PHYSICAL_ONLY=on` | off | Restrict pool to physical cores only |
| `FLYNNEL_SCHED_SMT=off` | (effective: on) | Back-compat alias for `PHYSICAL_ONLY=on` |
| `FLYNNEL_SCHED_PIN=on` | off | Enable CPU affinity pinning |
| `FLYNNEL_SCHED_SMT_AS_IO=on` | off | Enable SMT-sibling IoPool |
| `FLYNNEL_SPIN_WINDOW_ROUNDS=N` | 500 | JEC sleep spin-window override (pins the window, adaptation off) |
| `FLYNNEL_ADAPTIVE_SPIN=1` | off | Opt in to the adaptive spin-window controller |
| `FLYNNEL_TRACE=on` | off | Enable scheduler-event tracing |
| `FLYNNEL_TRACE_DISPATCH=<any>` | off | Per-`join_in_worker` dispatch trace to stderr |
| `FLYNNEL_LOCKLATCH_DIAGNOSE=1` | off | Per-`LockLatch::wait()` diagnostic to stderr |
| `FLYNNEL_ENABLE_FLAT_FANOUT=1` | off | Flat-fanout path in reduce_chunks (bench-only) |
| `FLYNNEL_REDUCE_CHUNKS_CHUNKS=N` | unset | Bench-driven audit hook: override reduce_chunks target |
| `TPU_NAME=...` | (read-only probe) | Signals TPU presence to detection probe |
