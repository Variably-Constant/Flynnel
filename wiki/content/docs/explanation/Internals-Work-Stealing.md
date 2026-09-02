---
title: "Internals: Work-Stealing"
weight: 4
---

The low-level mechanics inside Flynnel's CPU arena. This page is for contributors and curious users; nothing here is required for normal use.

## The five primitives

A work-stealing pool is built from interlocking primitives. Flynnel's are all in-house (`pub(crate)` for the worker-only ones; `pub` for the user-facing ring/notify primitives):

| Primitive | File | Role |
|-----------|------|------|
| `chase_lev_local::Worker<T>` | [`src/sched/chase_lev_local.rs`](https://github.com/markusmcnugen/flynnel/blob/main/src/sched/chase_lev_local.rs) | Per-worker, owner-side LIFO push/pop; thieves steal from the FIFO end. In-house Chase-Lev implementation per Vafeiadis et al.; exposes `slot_ptr` for prefetch wiring the upstream crossbeam version doesn't. The production worker pool layers KHL (per-slot Vyukov, K_inner=3) and Fcl (counter-only, K_inner=3) backings on top, swapped at runtime via `AdaptiveWorker`'s AtomicU32 tag. |
| `injector::Injector<JobRef>` | [`src/sched/injector.rs`](https://github.com/markusmcnugen/flynnel/blob/main/src/sched/injector.rs) | Per-arena MPMC queue for external (non-worker-thread) submissions. Wraps `FlynnelRing` (Vyukov per-slot sequence) with the same Success/Empty/Retry steal surface as the upstream crossbeam Injector but reduced wrapper overhead. |
| `flynnel_ring::FlynnelRing<JobRef>` | [`src/sched/flynnel_ring.rs`](https://github.com/markusmcnugen/flynnel/blob/main/src/sched/flynnel_ring.rs) | Per-worker mailbox: bounded MPMC ring used by `push_to_mailbox` for cross-worker hand-offs in recursive splits. |
| `CoreLatch` | [`src/sched/latch.rs`](https://github.com/markusmcnugen/flynnel/blob/main/src/sched/latch.rs) | Per-job one-shot signal with a 4-state machine. |
| `Sleep` (JEC) | [`src/sched/jec_sleep.rs`](https://github.com/markusmcnugen/flynnel/blob/main/src/sched/jec_sleep.rs) | Per-arena Jobs Event Counter wake protocol; tracks `awake_but_idle` vs `sleeping` worker counts so producers can skip unpark syscalls when enough workers are already spinning. Port of `rayon-core-1.13.0::sleep::{counters,mod}`. |
| `Parker` | [`src/sched/sleep.rs`](https://github.com/markusmcnugen/flynnel/blob/main/src/sched/sleep.rs) | Per-worker yield-spin-then-park primitive. Wraps `std::thread::park` + `wake_counter: AtomicU64` for a permit-based race-free wake. Used for the SMT-sibling gate (siblings park whenever `smt_requests == 0`) AND for the `NotifyHub` per-consumer wake; the main-loop sleep path uses the JEC `Sleep` above. |
| `notify_ring::NotifyHub<T>` | [`src/sched/notify_ring.rs`](https://github.com/markusmcnugen/flynnel/blob/main/src/sched/notify_ring.rs) | Blocking send/recv wrapper over `FlynnelRing` + per-consumer `Parker`. Used by the IO pool, GPU/WASM backend workers, and the `hybrid_pipeline` stage hand-off. No Mutex on the hot wake path (`Box<[OnceLock<Arc<Parker>>]>` is pre-allocated at hub construction). |

Above them, [`LocalArena`](#localarena) in `src/sched/arena_local.rs` is the per-NUMA-node worker pool; [`NumaArena`](https://github.com/markusmcnugen/flynnel/blob/main/src/sched/arena_numa.rs) composes one `LocalArena` per node.

## `JobRef` two-word vtable

Defined in [`src/sched/job.rs`](https://github.com/markusmcnugen/flynnel/blob/main/src/sched/job.rs).

```rust
pub(crate) struct JobRef {
    pointer: *const (),
    execute_fn: unsafe fn(*const ()),
    pub k_outer: u8,
    pub numa_hint: u8,
    pub variant: u8,
    pub reserved: u8,
}
```

Two-word handle plus four tag bytes in the alignment slack. Direct adaptation of rayon-core's `JobRef`. The vtable indirection means thieves can classify a job (read tags) without dereferencing the captured-state pointer (avoiding the cache miss until they decide to take it).

`Send + Sync` because every `Job` impl guarantees the needed bounds at construction time. Concrete `Job` impls in the same file:

- `StackJob` - stack-resident job for `join` (caller's stack frame holds the captured state; latch ensures the worker never reads after free). This is the only `Job` impl Flynnel ships - every entry point reaches the work-stealing pool through it.

## `CoreLatch` state machine

```text
  UNSET ----get_sleepy()---> SLEEPY ----fall_asleep()---> SLEEPING
    |                          |                              |
    |                          |                              |
    v                          v                              v
   SET (Latch::set called; publisher observed prior state)
```

Stored as an `AtomicU8` with constants `UNSET=0`, `SLEEPY=1`, `SLEEPING=2`, `SET=3`. Transitions are CAS with `SeqCst` on success, `Relaxed` on failure.

The two-phase sleep handshake is what avoids the lost-wakeup window:

1. Owner calls `get_sleepy()` to declare intent to park. CAS `UNSET -> SLEEPY`.
2. Owner calls `fall_asleep()` immediately before parking. CAS `SLEEPY -> SLEEPING`.
3. Publisher calls `set()` to flip to `SET`. The atomic returns the prior state; if it was `SLEEPING`, the publisher must wake the parked thread.

If a publisher fires between steps 1 and 2, the owner sees its `fall_asleep` CAS fail and skips parking. If a publisher fires between steps 2 and 3, the publisher sees prior=`SLEEPING` and issues an unpark. No window is unprotected.

### The self-invalidation contract

`Latch::set` takes `*const Self` rather than `&self` because the publishing CAS may wake a thread that immediately deallocates the latch (when a `StackJob` finishes and its parent frame returns). Implementations MUST read every field they need BEFORE the publishing store.

Flynnel ships three wake-capable wrappers that honor this discipline: `SpinLatch` (clones its `Arc<Parker>` before the publishing store; used by the external-dispatch slot pool), `CountLatch` (N-participant; the final decrementer clones the parker then publishes; used by the cooperative flat fan-out), and `LockLatch` (mutex-free cross-thread waiter for callers outside the pool).

## Sleep protocol: JEC + spin floor

The main worker loop uses the **JEC (Jobs Event Counter)** sleep protocol from [`src/sched/jec_sleep.rs`](https://github.com/markusmcnugen/flynnel/blob/main/src/sched/jec_sleep.rs), a verbatim port of `rayon-core-1.13.0::sleep::{counters,mod}`. The protocol's central trick is to track two separate worker counts: `awake_but_idle` (spinning in `no_work_found`) and `sleeping` (parked on a `Mutex<Condvar>`). When a producer posts new work, it consults both counters and skips the unpark syscall if there are already enough idle-but-awake workers to absorb the dispatched jobs. The hot path is producer-side `new_internal_jobs(num_jobs, queue_was_empty)`, which:

1. Increments the global JEC if any worker is in the `Sleepy` phase (signals sleepy workers to re-search before sleeping).
2. If the queue was non-empty, wakes `min(num_jobs, num_sleepers)` parked workers.
3. If the queue was empty, wakes `max(num_jobs - awake_but_idle, 0)` capped at `num_sleepers`.

The four-phase consumer state machine (`Active -> Idle -> Sleepy -> Sleeping`) escalates only after `ROUNDS_UNTIL_SLEEPY` (32) yields of finding no work, then a further spin-window's worth of yields (default 500, tunable via `FLYNNEL_SPIN_WINDOW_ROUNDS` / `set_spin_window`) with the worker counted as `Sleepy` before it finally locks its mutex and waits on the condvar. The JEC rescue lets producers pull `Sleepy` workers back into the search loop without paying the park / unpark roundtrip.

The `Parker` primitive in [`src/sched/sleep.rs`](https://github.com/markusmcnugen/flynnel/blob/main/src/sched/sleep.rs) is wired to the SMT-sibling gate (siblings park whenever `smt_requests == 0` and unpark when it goes positive). That path is structurally separate from the main-loop wake decision and uses stdlib `park`/`unpark` because its permit-based race resolution is enough for an on/off SMT toggle.

### Yield-spin floor

Each worker tier has a documented spin floor:

| Tier | `spin_rounds` | Rationale |
|------|--------------|-----------|
| `Inline` | 0 | No scheduler involvement. |
| `Local` | 8 | Sub-microsecond work avoids the park / unpark syscall (~5 us cost). |
| `Hierarchical` | 32 | Multi-microsecond work amortizes the park / unpark pair. |
| `Federated` | 0 | Millisecond-scale jobs; throughput beats latency. |

The actual loop body:

```rust
for _ in 0..spin_rounds {
    std::thread::yield_now();
    if predicate() { return true; }
    if is_shutdown() { return false; }
}
std::thread::park();
predicate()
```

After the yield-spin window, the worker enters real park. The CoreLatch two-phase handshake makes the park safe.

## `LocalArena`

Defined in [`src/sched/arena_local.rs`](https://github.com/markusmcnugen/flynnel/blob/main/src/sched/arena_local.rs).

### Construction

```rust
pub fn new(n_workers: usize) -> Arc<Self>
pub fn with_cpu_set(n_workers, cpu_set: Option<Vec<CoreId>>) -> Arc<Self>
pub fn with_smt_extension(primary_count, smt_extension, cpu_set) -> Arc<Self>
```

`with_smt_extension` is the production constructor. It spawns `primary_count + smt_extension` worker threads:

- Workers `[0..primary_count)` are **primaries**: they run unconditionally and form the always-on pool.
- Workers `[primary_count..primary_count + smt_extension)` are **SMT siblings**: they park whenever the arena's `smt_requests` counter is 0, and wake when it becomes positive (via `acquire_smt`).

The total worker count is sized to the host's logical-thread count when the SMT extension is enabled; otherwise to the physical-core count.

### Worker loop

Each worker runs:

1. **SMT-sibling gate**. If `is_primary == false` AND `smt_requests == 0`, park.
2. **Steal-stash drain**. K_inner=3 batch leftovers from a prior successful steal (locality-warm, coherence already paid).
3. **Mailbox pop**. Owner-directed hand-offs from peers. (The wait-loop counterpart, `WorkerCtx::find_work`, additionally gates this pop on the process-global `MAILBOX_EVER_USED` flag.)
4. **Local deque tier walk**. LIFO from this worker's own per-tier deques, SmtLocal through Public.
5. **Injector steal**. From the arena's global MPMC injector.
6. **Peer-steal probe**. Adaptive probe of peer deques across the tiers the steal discipline allows. Strategy: prefer the last-successful victim (warm L2 + recursive-split bursts), else xorshift-random pick. Probe count is clamped (`PROBE_LARGE = 4` for pools larger than 8 workers; full walk for smaller pools).
7. **JEC sleep**. The worker calls `sleep.no_work_found(idle_state, has_injected_jobs)` after the steal probes come up empty. Internally the call yields `ROUNDS_UNTIL_SLEEPY` (32) times while counted as `awake_but_idle`, then announces itself as `Sleepy` by incrementing JEC, yields a further spin-window's worth of rounds (default 500; producers have that window to bump JEC back and rescue it), then finally locks its mutex and waits on the condvar as `Sleeping`. Producer-side `new_internal_jobs` re-engages the worker through the same channel.

### Key load-bearing constants

| Constant | Value | Rationale |
|----------|-------|-----------|
| `LOCAL_SPIN_ROUNDS` | 64 | ~64 us hot window; covers most Criterion inter-iteration gaps so workers don't park-and-rewake every iteration. |
| `PROBE_FULL_CUTOFF` | 8 | At pool sizes <= 8 (Zen+ physical-core count), walk all peers per loop iteration. |
| `PROBE_LARGE` | 4 | At pool sizes > 8, clamp probes to 4 per loop iteration to keep deque-head cache traffic bounded. |
| `MIN_LEAF_ITEMS` | 256 (in par_iter) | SLAW bisect floor; below this a chunk runs serially. |

### Stats observability

`WorkerStats` lives at the arena level; one `Arc<WorkerStats>` per worker. Fields:

- `local_pops: AtomicU64` - local-deque LIFO pops.
- `peer_steal_hits: AtomicU64` - successful peer steals.
- `peer_steal_misses: AtomicU64` - peer-probe rounds that returned no work.

`#[repr(align(128))]` pads each `WorkerStats` to a 128-byte boundary so adjacent workers' counters never share a 128-byte block (Intel L1 hardware prefetcher fetches in pairs of 64-byte lines; 128 is the effective false-sharing unit on modern CPUs).

The background observer (`split_observer::spawn_observer`) samples these every interval and tunes `split_multiplier` accordingly.

## `NumaArena`

Defined in [`src/sched/arena_numa.rs`](https://github.com/markusmcnugen/flynnel/blob/main/src/sched/arena_numa.rs).

Composes one `LocalArena` per NUMA node. On single-NUMA hosts (most desktops) it collapses to a single underlying `LocalArena`; cross-node code paths are dead branches with zero overhead. On multi-NUMA hosts (Genoa, dual-socket Xeon / Threadripper) it routes work to the caller's current-thread node by default and rebalances via cross-node steal when one node is idle.

Per-node worker count is sized from [`cpu_info()`](NUMA-And-Topology.md#cpuinfo) and per-node CPU mask (from `numa_topology().cpus_in_node(node)`). Each sub-arena's `with_smt_extension` constructor takes a CPU set restricted to its node's CPUs.

## Why this design

The structure is borrowed largely from rayon-core 1.13; the design has been battle-hardened across a decade of edge cases. Flynnel reshapes it around two distinct extensions:

1. **K-axis tier picker** at the entry point. `pick_tier` reads `JobPlan` and the cached NUMA topology to decide whether to even touch the arena. This avoids paying scheduling overhead on micro-jobs where serial execution beats every parallel strategy.
2. **Per-NUMA arena composition**. `NumaArena` collapses on single-node hosts and routes intelligently on multi-node hosts. The composition is built on top of the same `LocalArena` primitive so single-NUMA users pay nothing for the multi-NUMA code path.

Everything else (the worker loop, the latch, the deque, the JEC sleep protocol) is the rayon-core lineage with minor adjustments (env-var-driven pinning default off, SMT-sibling extension model that runs on the simpler `Parker` path on top of the JEC main loop, adaptive peer-probe with last-victim). [`src/sched/jec_sleep.rs`](https://github.com/markusmcnugen/flynnel/blob/main/src/sched/jec_sleep.rs) is a verbatim port of `rayon-core-1.13.0::sleep::{counters,mod}`.

## Reading order for contributors

1. `src/sched/job.rs` - `JobRef` + `StackJob` shape. The smallest piece.
2. `src/sched/latch.rs` - `CoreLatch` 4-state machine + the SpinLatch / CountLatch / LockLatch wrappers.
3. `src/sched/jec_sleep.rs` - the JEC `Sleep` and the four-phase consumer state machine. This is the main-loop sleep path.
4. `src/sched/sleep.rs` - `Parker` yield-spin + park. The SMT-sibling gate and the NotifyHub consumer wake.
5. `src/sched/chase_lev_local.rs` - in-house Chase-Lev `Worker`/`Stealer`/`Steal` with `slot_ptr` exposure.
6. `src/sched/flynnel_ring.rs` + `src/sched/injector.rs` - in-house Vyukov MPMC ring + Injector facade.
7. `src/sched/notify_ring.rs` - blocking notify-channel built on FlynnelRing + Parker.
8. `src/sched/adaptive_worker.rs` - AtomicU32-tag dispatch between KHL (per-slot Vyukov) and Fcl (counter-only) K_inner=3 backings.
9. `src/sched/deque.rs` - `steal_retry` helper wrapping `chase_lev_local`.
10. `src/sched/arena_local.rs` - the worker loop and `LocalArena`.
11. `src/sched/arena_numa.rs` - multi-NUMA composition.
12. `src/sched/arena.rs` - `join` / `join_context` entry points.
13. `src/sched/par_iter.rs` - bisect + SLAW + heartbeat / token-bucket / tiny-tasks splitters.
