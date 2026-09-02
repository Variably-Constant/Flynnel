//! Public scheduler entry points: `join` and the `join_default` /
//! `join_context` conveniences.
//!
//! The dispatch surface that algorithm code (Karatsuba, NTT,
//! transcendentals) calls. Picks a [`SchedTier`] from
//! the [`JobPlan`] + cached NUMA topology and runs the work on the
//! corresponding execution path.
//!
//! ## Tier status
//!
//! - **Inline**: serial in caller, K <= 4.
//! - **Local**: work-stealing across the worker pool (unpinned by
//!   default; see `FLYNNEL_SCHED_PIN` below) with cooperative wait
//!   (parent steals while child runs). K = 5..7.
//! - **Hierarchical**: per-NUMA arenas with leader-driven cross-
//!   NUMA steal. Currently routes through the same Local
//!   dispatch path.
//! - **Federated**: FLINT-style pull pool. Currently routes
//!   through the same Local dispatch path.
//!
//! ## Worker count
//!
//! Default `global_local_arena()` worker count = ALL logical
//! threads per node (= 16 on R7 2700), matching rayon's
//! convention; SMT-2 siblings gain 10-30% on latency-bound work
//! and break even on most other shapes. For IMUL-saturated work
//! where a sibling contests the same execution port, restrict to
//! physical cores via env:
//!
//! - `FLYNNEL_SCHED_WORKERS=N` - exact count per node
//! - `FLYNNEL_SCHED_PHYSICAL_ONLY=on` (or `FLYNNEL_SCHED_SMT=off`)
//!   - physical cores only
//! - `FLYNNEL_SCHED_PIN=on|1|true` - enable per-worker CPU pinning
//!   (default is unpinned). The rayon-crossover bench measured
//!   unpinned workers 1.2-1.6x faster than pinned under any
//!   competing system load, because the OS places threads on idle
//!   SMT siblings. Cache-locality wins from pinning only realize
//!   on dedicated bench hardware with zero competing load; set
//!   this env var when running on such a rig.
//!

use std::sync::{Arc, OnceLock};
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

use crate::sched::arena_local::{WorkerCtx, current_worker_ctx};
use crate::sched::arena_numa::NumaArena;
use crate::sched::job::{JobRef, NUMA_HINT_ANY, StackJob};
use crate::sched::latch::{CoreLatch, LockLatch, SpinLatch};
use crate::sched::sleep::Parker;
use crate::sched::plan::{JobPlan, SchedTier, pick_tier};
use crate::numa_topology::numa_topology;

/// Fork-join primitive: run `a` and `b` and return both results.
///
/// Dispatch behavior per tier (as selected by [`pick_tier`]):
///
/// - `SchedTier::Inline` runs `a` then `b` serially in the caller
///   (see [`inline_join_context`]).
/// - `SchedTier::Local`, `SchedTier::Hierarchical`, and
///   `SchedTier::Federated` all route through
///   [`local_join_context`], which pushes the right half onto a
///   worker deque and runs the left half cooperatively while
///   waiting on the right's latch.
///
/// # Determinism contract
///
/// The returned `(RA, RB)` tuple preserves **algebraic** order
/// (left-half first, right-half second) for non-commutative
/// reductions regardless of which thread executed which half, so
/// bit-exact reproducibility holds across the serial and
/// work-stealing tiers.
pub fn join<A, B, RA, RB>(plan: &JobPlan, a: A, b: B) -> (RA, RB)
where
    A: FnOnce() -> RA + Send,
    B: FnOnce() -> RB + Send,
    RA: Send,
    RB: Send,
{
    // `join` is a convenience wrapper over `join_context` for
    // callers that don't care about the migrated/stolen flag.
    join_context(plan, move |_| a(), move |_| b())
}

/// Fork-join variant that exposes the `migrated`/`stolen` flag to
/// each closure. Mirrors rayon-core's `join_context`.
///
/// `a` is called with `injected: bool` - `true` iff this entire
/// `join_context` was cold-injected from outside the worker pool
/// (rayon's "in_worker_cold" pattern). For nested in-worker calls
/// `injected` is `false`.
///
/// `b` is called with `stolen: bool` - `true` iff `b` was dequeued
/// and executed by a peer worker (i.e., somebody stole the
/// right-half job). `false` iff the originating worker popped the
/// job back from its own local deque and ran it inline.
///
/// The flag is the key signal for adaptive splitters: when work is
/// being stolen there is steal pressure, so the splitter should
/// subdivide more aggressively to feed hungry workers. See
/// `par_iter::for_each_chunk` for the bisect-side use of this.
pub fn join_context<A, B, RA, RB>(plan: &JobPlan, a: A, b: B) -> (RA, RB)
where
    A: FnOnce(bool) -> RA + Send,
    B: FnOnce(bool) -> RB + Send,
    RA: Send,
    RB: Send,
{
    let tier = pick_tier(plan, numa_topology());
    match tier {
        SchedTier::Inline => inline_join_context(a, b),
        SchedTier::Local => local_join_context(plan, a, b),
        SchedTier::Hierarchical | SchedTier::Federated => local_join_context(plan, a, b),
    }
}

/// Lazily-initialized process-global NUMA-aware arena. On single-
/// NUMA hosts this is a single sub-arena; on multi-NUMA hosts
/// (Colab Genoa, dual-socket Xeon / Threadripper) it has one
/// sub-arena per NUMA node, each pinned to its node's CPUs.
///
/// Worker count per node picked from [`crate::cpu_info`]
/// + per-node CPU mask with env overrides (see module doc).
pub fn global_local_arena() -> &'static Arc<NumaArena> {
    static ARENA: OnceLock<Arc<NumaArena>> = OnceLock::new();
    ARENA.get_or_init(|| {
        let per_node_override = pick_worker_count_per_node();
        NumaArena::new(per_node_override)
    })
}

/// Decide how many workers each NUMA sub-arena should spawn.
///
/// 1. `FLYNNEL_SCHED_WORKERS=N` (explicit positive integer) means
///    N PER NODE (so total = N * num_nodes).
/// 2. `FLYNNEL_SCHED_PHYSICAL_ONLY=on|1|true` => `None`
///    (NumaArena uses physical cores in each node). Use this for
///    IMUL-saturated workloads where the SMT sibling contests the
///    same execution port and adds no architectural throughput
///    (e.g., multi-precision arithmetic, tight Karatsuba loops).
/// 3. `FLYNNEL_SCHED_SMT=off|0|false` => same as
///    `FLYNNEL_SCHED_PHYSICAL_ONLY=on` (back-compat alias).
/// 4. Default: `Some(logical_threads_per_node)`. Matches rayon's
///    convention of using all logical threads. SMT-2 siblings
///    typically gain 10-30% on latency-bound work and break even
///    on most other workloads on modern Zen / Skylake silicon.
fn pick_worker_count_per_node() -> Option<usize> {
    if let Ok(v) = std::env::var("FLYNNEL_SCHED_WORKERS")
        && let Ok(n) = v.parse::<usize>()
        && n >= 1
    {
        return Some(n);
    }
    let physical_only = std::env::var("FLYNNEL_SCHED_PHYSICAL_ONLY")
        .map(|v| v.eq_ignore_ascii_case("on") || v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    let smt_explicitly_off = std::env::var("FLYNNEL_SCHED_SMT")
        .map(|v| {
            let lv = v.to_ascii_lowercase();
            lv == "off" || lv == "0" || lv == "false"
        })
        .unwrap_or(false);
    if physical_only || smt_explicitly_off {
        return None;
    }
    let topo = numa_topology();
    // Use number of CPUs in node 0 as the per-node sizing hint. On
    // asymmetric NUMA hosts this may over- or under-allocate for
    // other nodes; for symmetric hosts (the common case) it's
    // exact. NumaArena clamps to >= 1.
    let cpus_node0 = topo.cpus_in_node(0).len();
    Some(cpus_node0.max(1))
}

/// Local-tier fork-join: submit the right half to the worker pool,
/// run the left half inline, wait on the right's latch.
///
/// # Dispatch
///
/// Two paths, picked by `current_worker_ctx()`:
///
/// **Fast (in-worker) path.** The calling thread is a Flynnel
/// scheduler worker. Push `job_b` onto the caller's own local
/// Chase-Lev deque (LIFO, ~5 ns), run `a` inline, then look for
/// `job_b` back on the local deque. If it's still there nobody
/// stole it; execute it inline. If a thief took it, drain other
/// local work (LIFO) and steal until the latch is set. This is
/// the rayon `join_context` shape (rayon-core/src/join/mod.rs:132)
/// and the primary perf win of in-worker dispatch.
///
/// **Slow (external) path.** The calling thread is not a worker
/// (typical from Criterion main, application code, etc.). Push
/// `job_b` to the NUMA arena's injector (~5-30 µs), run `a`,
/// then cooperatively steal via `arena.try_run_one` until the
/// latch is set.
///
/// # Lifetime + panic safety
///
/// The right-half `StackJob` lives on this function's stack frame
/// while its `JobRef` is in the arena. To ensure the worker never
/// dereferences a freed StackJob, we wait on the latch before
/// returning **on every path**, including panic. `catch_unwind`
/// around `a()` makes the panic path explicit: catch -> wait ->
/// resume.
fn local_join_context<A, B, RA, RB>(plan: &JobPlan, a: A, b: B) -> (RA, RB)
where
    A: FnOnce(bool) -> RA + Send,
    B: FnOnce(bool) -> RB + Send,
    RA: Send,
    RB: Send,
{
    let ctx_ptr = current_worker_ctx();
    if !ctx_ptr.is_null() {
        // FAST PATH: already in a worker. Push job_b to OUR own
        // local Chase-Lev deque, run a inline, drain.
        // SAFETY: ctx_ptr was set by worker_loop on this same
        // thread and is valid until worker_loop returns.
        let ctx = unsafe { &*ctx_ptr };
        // `injected = false` because the caller is already a
        // running worker (this is a nested in-worker join).
        return join_in_worker(ctx, plan, a, b, false);
    }
    external_dispatch(plan, a, b)
}

/// Push a job to the worker's local deque, honoring any
/// caller-supplied `deque_tier_hint` in the plan. `None` falls
/// through to `WorkerCtx::push` (default tier = Public per
/// `DequeTier::default`); `Some(tier)` routes via `push_tier`
/// so the steal discipline pins the work to peers at the chosen
/// distance or wider.
#[inline]
/// Pushes `job` to the hinted tier (Public by default). `Err(job)`
/// when that deque is full: the caller runs the job inline instead
/// of waiting for a thief.
fn push_with_tier_hint(ctx: &WorkerCtx, job: JobRef, plan: &JobPlan) -> Result<(), JobRef> {
    match plan.deque_tier_hint {
        Some(tier) => ctx.try_push_tier(job, tier)?,
        None => ctx.try_push(job)?,
    }
    // External slot ctx: broadcast-wake all primaries after the
    // push. A single-sleeper wake random-picks the slot's deque
    // at ~1/36 and may re-sleep without finding the right-half;
    // waking N primaries multiplies the first-round hit
    // probability by N. Without the broadcast, slot-pushed
    // right-halves sit unstolen (measured as full serialization
    // on an n=5 heavy fan-out: 227s = 5 x 45s).
    if ctx.is_external_slot {
        ctx.sleep.new_internal_jobs(ctx.stealers.len() as u32, false);
    }
    Ok(())
}

/// Fast-path body: run `join(a, b)` from within a worker that is
/// already in `WorkerCtx`. Push `job_b` to the worker's local
/// Chase-Lev deque (LIFO, single-owner-writer, ~5 ns), run `a`
/// inline, then drain the local deque until `job_b.latch` is set.
///
/// # Lifetime + panic safety
///
/// `job_b` lives on this function's stack frame. We wait on its
/// latch before returning on every path, including panic. A panic
/// from `a` is captured via `catch_unwind`; we still complete the
/// latch wait so the worker that may have stolen `job_b` doesn't
/// dereference freed memory, then resume the panic for the caller.
// === Per-join_in_worker dispatch trace (FLYNNEL_TRACE_DISPATCH=1) ===
//
// Three counters accumulated process-wide for diagnostic attribution
// of where the bisect's wall-clock per iter goes:
//
//   JOIN_CALL_COUNT        - number of join_in_worker invocations
//   JOIN_A_BODY_NS         - sum of cycles spent in `a()` (left half work
//                            for this frame, which for reduce_inner is
//                            itself the recursive sub-tree)
//   JOIN_WAIT_NS           - sum of cycles spent in the wait loop AFTER
//                            `a()` returned (find_work probing + stolen
//                            job execution + yield_now spins)
//
// Compute "wait fraction" = JOIN_WAIT_NS / (JOIN_A_BODY_NS + JOIN_WAIT_NS).
// High fraction == dispatch overhead is the bottleneck. Low fraction ==
// the bisect is balanced and dispatch is amortized fine.
//
// Zero overhead when env var unset: `if traced` short-circuits before
// any TSC read.
static JOIN_CALL_COUNT: AtomicU64 = AtomicU64::new(0);
static JOIN_A_BODY_NS: AtomicU64 = AtomicU64::new(0);
static JOIN_WAIT_NS: AtomicU64 = AtomicU64::new(0);
// Sub-split of JOIN_WAIT_NS: time inside `unsafe { job.execute() }`
// (productive stealing of cross-worker jobs while waiting) vs time
// in the `yield_now()` path (no work available -- pure dispatch
// waste). Together these sum to JOIN_WAIT_NS minus the small
// `is_set()` poll cost and the small `find_work()` probe cost itself.
static JOIN_WAIT_STEAL_NS: AtomicU64 = AtomicU64::new(0);
static JOIN_WAIT_IDLE_NS: AtomicU64 = AtomicU64::new(0);

// Process-wide cache for FLYNNEL_TRACE_DISPATCH. The env-var lookup
// itself acquires a process-wide env lock and walks the env vector;
// per-call cost ~100-300 ns measured. Caching to a OnceLock<bool>
// reduces it to a single Acquire load on subsequent calls. join_in_worker
// is on the bisect hot path (10^4+ calls per inline_collapse bench cell)
// so the per-call save compounds.
static DISPATCH_TRACE_ENABLED: OnceLock<bool> = OnceLock::new();

#[inline(always)]
fn dispatch_trace_enabled() -> bool {
    *DISPATCH_TRACE_ENABLED.get_or_init(|| {
        std::env::var_os("FLYNNEL_TRACE_DISPATCH").is_some()
    })
}

#[inline(always)]
fn dispatch_tsc() -> u64 {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        // SAFETY: _rdtsc has no preconditions on x86_64.
        std::arch::x86_64::_rdtsc()
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        std::time::Instant::now().elapsed().as_nanos() as u64
    }
}

/// Snapshot + reset the dispatch trace counters. Returns
/// `(calls, a_body_cycles_total, wait_cycles_total)`. Useful as a
/// per-bench-group accessor that prints the cumulative state then
/// clears it for the next bench group.
pub fn dispatch_trace_snapshot() -> (u64, u64, u64) {
    (
        JOIN_CALL_COUNT.swap(0, Relaxed),
        JOIN_A_BODY_NS.swap(0, Relaxed),
        JOIN_WAIT_NS.swap(0, Relaxed),
    )
}

/// Snapshot + reset the wait-loop sub-counters: returns
/// `(steal_cycles, idle_cycles)` accumulated since the previous
/// snapshot. Sum of these two plus the small `is_set()` /
/// `find_work()` probe cost adds up to `JOIN_WAIT_NS` from
/// `dispatch_trace_snapshot()`. The split tells us how much of
/// the wait time is productive cross-worker rebalancing vs pure
/// dispatch waste (idle yield_now spinning).
pub fn dispatch_trace_wait_snapshot() -> (u64, u64) {
    (
        JOIN_WAIT_STEAL_NS.swap(0, Relaxed),
        JOIN_WAIT_IDLE_NS.swap(0, Relaxed),
    )
}

#[inline]
fn join_in_worker<A, B, RA, RB>(
    ctx: &WorkerCtx,
    plan: &JobPlan,
    a: A,
    b: B,
    a_injected: bool,
) -> (RA, RB)
where
    A: FnOnce(bool) -> RA + Send,
    B: FnOnce(bool) -> RB + Send,
    RA: Send,
    RB: Send,
{
    // Per-fork latch: CoreLatch, not SpinLatch. A SpinLatch here
    // (parker.unpark on Latch::set) measures 1.5-1.8x slower on
    // bisect-heavy real-world workloads because the same Parker
    // is shared with worker_loop's idle-sleep path: peer-wake
    // calls from arena_local::wake_one_peer (legitimately
    // targeting the worker_loop idle sleeper) collide with the
    // SpinLatch-park waiter, ~1us of spurious wake per collision,
    // compounded across the 32-fork bisect depth typical at that
    // shape. The wait loop below peer-helps via find_work and
    // does not reach a park branch on those workloads. SpinLatch
    // fits sites that own a dedicated Parker not shared with the
    // worker pool's idle-sleep coordinator; the join_in_worker
    // wait loop is not one of those sites.
    let job_b = StackJob::new(b, CoreLatch::new());
    // SAFETY: job_b lives on this stack frame; the latch-wait
    // below keeps it alive until the worker finishes touching it.
    let job_b_ref = unsafe {
        job_b.as_job_ref(
            plan.k_outer,
            plan.numa_hint.unwrap_or(NUMA_HINT_ANY as u32) as u8,
            plan.variant,
        )
    };
    let job_b_id = job_b_ref.id();
    // SIMC/MIMC mailbox routing for the right-half of join, gated
    // on a cheap is_empty() check against the SMT sibling's
    // mailbox. The intuition:
    //
    // - When the sibling's mailbox is EMPTY, the sibling is either
    //   idle (parked / spinning) or executing - in either case it
    //   has no other queued mailbox work, so handing it the
    //   right-half via mailbox jumps the queue ahead of any deque
    //   tier and consumes the caller's L1d-warm state with zero
    //   coherence transfer (the SMT pair shares L1d).
    // - When the sibling's mailbox is NON-EMPTY, the sibling is
    //   backlogged. Forcing more work into the mailbox concentrates
    //   load on one SMT pair and starves the other workers. The
    //   right-half goes to the owner's own deque instead, where the
    //   full pool can steal from it.
    //
    // The unconditional version of this routing regressed
    // realistic_bench Compute/100k 2.27x because every join right-
    // half got pinned to one specific worker; the is_empty() gate
    // restores broad-steal semantics for backlogged-sibling cases.
    let sibling = ctx.index ^ 1;
    // Mailbox-route only when the caller opted in via
    // `plan.use_mailbox_routing` AND the sibling has truly nothing
    // queued (mailbox empty AND SmtLocal deque empty). Without the
    // opt-in flag, the default path is a regular deque push so
    // broad work-stealing is preserved for latency-bound + IMUL-
    // saturated workloads where mailbox concentration hurts.
    let try_mailbox = plan.use_mailbox_routing
        && sibling < ctx.peer_mailboxes.len()
        && sibling < ctx.stealers.len()
        && ctx.peer_mailboxes[sibling].is_empty()
        && ctx.stealers[sibling][crate::sched::deque_tier::DequeTier::SmtLocal.idx()].is_empty();
    let refused = if try_mailbox {
        match ctx.push_to_mailbox(sibling, job_b_ref) {
            Ok(()) => None,
            // Mailbox full - fall through to deque push, honoring
            // any caller-supplied deque_tier_hint.
            Err(returned_job) => push_with_tier_hint(ctx, returned_job, plan).err(),
        }
    } else {
        push_with_tier_hint(ctx, job_b_ref, plan).err()
    };
    if refused.is_some() {
        // Deque full: run both halves here. Waiting for a thief to
        // free a slot deadlocks when every thief is an owner waiting
        // on its own full deque (measured: all 16 rings at 256 on a
        // 65,536-item min_leaf=1 collect).
        let a_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| a(a_injected)));
        // SAFETY: the refused JobRef was never published, so job_b
        // is untouched and uniquely ours to consume.
        let rb = unsafe { job_b.run_inline(false) };
        return match a_result {
            Ok(ra) => (ra, rb),
            Err(payload) => {
                drop(rb);
                std::panic::resume_unwind(payload);
            }
        };
    }
    crate::sched::trace::emit(crate::sched::trace::TraceEvent::JoinPush, 0);

    let traced = dispatch_trace_enabled();
    let t_a_start = if traced { dispatch_tsc() } else { 0 };
    let a_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| a(a_injected)));
    let t_a_end = if traced { dispatch_tsc() } else { 0 };
    crate::sched::trace::emit(crate::sched::trace::TraceEvent::JoinWaitBegin, 0);

    // Wait loop. The most likely outcome (LIFO, no thief): the
    // first ctx.find_work() returns OUR job_b. We special-case
    // that via `id()` match and run it inline (stolen=false). If a
    // thief stole job_b first, we keep finding work until either
    // we observe job_b.latch set (the thief ran it) or we find
    // another job to execute.
    loop {
        if job_b.latch.is_set() {
            crate::sched::trace::emit(crate::sched::trace::TraceEvent::JoinWaitEnd, 1);
            if traced {
                let t_wait_end = dispatch_tsc();
                JOIN_CALL_COUNT.fetch_add(1, Relaxed);
                JOIN_A_BODY_NS.fetch_add(t_a_end.wrapping_sub(t_a_start), Relaxed);
                JOIN_WAIT_NS.fetch_add(t_wait_end.wrapping_sub(t_a_end), Relaxed);
            }
            // Latch set by a thief that already executed job_b.
            // SAFETY: latch is set => StackJob::execute finished
            // => result slot populated.
            let rb = unsafe { job_b.into_result() };
            return match a_result {
                Ok(ra) => (ra, rb),
                Err(payload) => {
                    drop(rb);
                    std::panic::resume_unwind(payload);
                }
            };
        }
        if let Some(job) = ctx.find_work() {
            if job.id() == job_b_id {
                crate::sched::trace::emit(crate::sched::trace::TraceEvent::JoinWaitEnd, 0);
                if traced {
                    let t_wait_end = dispatch_tsc();
                    JOIN_CALL_COUNT.fetch_add(1, Relaxed);
                    JOIN_A_BODY_NS.fetch_add(t_a_end.wrapping_sub(t_a_start), Relaxed);
                    JOIN_WAIT_NS.fetch_add(t_wait_end.wrapping_sub(t_a_end), Relaxed);
                }
                // Got our own job_b back. Run inline with
                // stolen=false to signal "no thief touched this".
                // SAFETY: the JobRef matches job_b's data
                // pointer; nobody else holds a JobRef for it
                // (we just popped the unique copy out of the
                // local deque), so consuming job_b is sound.
                let rb = unsafe { job_b.run_inline(false) };
                return match a_result {
                    Ok(ra) => (ra, rb),
                    Err(payload) => {
                        drop(rb);
                        std::panic::resume_unwind(payload);
                    }
                };
            }
            // Productive stealing: bracket the execute call so
            // dispatch_trace_wait_snapshot() can attribute cycles
            // to JOIN_WAIT_STEAL_NS.
            let t_steal_start = if traced { dispatch_tsc() } else { 0 };
            // SAFETY: JobRef contract; execute the popped job
            // exactly once.
            unsafe { job.execute() };
            if traced {
                JOIN_WAIT_STEAL_NS.fetch_add(
                    dispatch_tsc().wrapping_sub(t_steal_start),
                    Relaxed,
                );
            }
        } else {
            // Idle path: find_work returned None, no work to
            // steal. The yield_now is pure dispatch waste --
            // attribute its cycles to JOIN_WAIT_IDLE_NS so the
            // breakdown distinguishes productive stealing from
            // worker starvation.
            let t_idle_start = if traced { dispatch_tsc() } else { 0 };
            std::thread::yield_now();
            if traced {
                JOIN_WAIT_IDLE_NS.fetch_add(
                    dispatch_tsc().wrapping_sub(t_idle_start),
                    Relaxed,
                );
            }
        }
    }
}

/// Slow-path body: dispatch a join from outside the worker pool.
///
/// Builds a single wrapper [`StackJob`] whose closure runs the
/// entire `join_in_worker(ctx, plan, a, b)` body, injects it into
/// the NUMA arena, and waits on its latch. While waiting the
/// caller thread participates in work-stealing via
/// `arena.try_run_one`, which keeps the pool fed when the
/// scheduler has no other sleepers.
///
/// Every recursive `sched::join` issued from inside `a` or `b`
/// will then be inside a worker context and use the fast path.
/// This is rayon's `in_worker_cold` pattern.
fn external_dispatch<A, B, RA, RB>(plan: &JobPlan, a: A, b: B) -> (RA, RB)
where
    A: FnOnce(bool) -> RA + Send,
    B: FnOnce(bool) -> RB + Send,
    RA: Send,
    RB: Send,
{
    let arena = global_local_arena();
    // Consult `effective_use_smt` (not raw `use_smt`) so the
    // observer's measured per-leaf variance can suppress SMT
    // activation on uniform-cost workloads where SMT siblings
    // contest the same execution unit. The plan's `use_smt` is
    // the prior (set by the DispatchProfile); the observer
    // corrects it based on measured leaf cv^2.
    //
    // When the result is false the call is a no-op - no siblings
    // wake, the pool runs at primary-only width.
    //
    // SMT-on dispatch fires per external_dispatch + is on the per-
    // join hot path. To avoid a Vec::with_capacity allocation on
    // every SMT-on call (single-NUMA hosts = 99% of dev / laptop /
    // single-socket-server hardware), we hold the guard inline on
    // the stack via `try_acquire_smt_single` when possible and only
    // fall back to the allocating multi-NUMA Vec path on multi-node
    // hosts. Both bindings live until end-of-scope so the guards
    // remain held for the duration of the wrapper job.
    let _smt_guard_inline: Option<crate::sched::arena_local::SmtGuard>;
    let _smt_guards_multi: Vec<crate::sched::arena_local::SmtGuard>;
    if plan.effective_use_smt() {
        if arena.is_single_numa() {
            _smt_guard_inline = arena.try_acquire_smt_single();
            _smt_guards_multi = Vec::new();
        } else {
            _smt_guard_inline = None;
            _smt_guards_multi = arena.acquire_smt();
        }
    } else {
        _smt_guard_inline = None;
        _smt_guards_multi = Vec::new();
    }

    // Slot-pool wrap-and-park fast path. Build the join body as a
    // single StackJob with a SpinLatch wired to a caller-thread
    // Parker. Push the job to the slot's deque (whose stealer is
    // registered in arena.stealers so primaries can see + steal
    // it). Broadcast-wake all primaries. Park the caller on the
    // SpinLatch handshake until a primary completes the join.
    //
    // The body runs entirely on the primary that steals it, using
    // the primary's own WorkerCtx (8MB stack). Running it on the
    // caller instead (caller temporarily becomes a worker via TLS
    // ctx install) overflows the caller's 1MB Windows stack when
    // nested join_in_worker recurses on deep bisects.
    //
    // For an NMFD-shape workload (5 items x 45s, depth 2 bisect):
    // 1 primary runs probe + leftmost item, 3 other primaries
    // steal the bisect right-halves via the slot's registered
    // stealer + normal find_work scanning. Critical path
    // approaches 90s (probe 45s + parallel bisect 45s).
    //
    // Falls through to the legacy wrapper-job + injector +
    // LockLatch path if all slots are currently claimed.
    let arena_for_slot: Arc<crate::sched::arena_local::LocalArena> = arena
        .single_node_arc()
        .unwrap_or_else(|| arena.node_arc(plan.numa_hint));
    if let Some(_guard) = arena_for_slot.try_claim_external_slot() {
        let slot_ctx = _guard.ctx_ref();
        // Caller-thread Parker for the SpinLatch wake. Captures
        // the calling thread's Thread handle at construction;
        // SpinLatch::set unparks this exact thread when the
        // primary completes the job.
        //
        // Wrap-and-park unconditionally: no depth-switch to a
        // caller-as-worker variant on small batch_size. Measured
        // on the NMFD workload, wrap-and-park runs 96s vs the
        // caller-as-worker path's 161s. Criterion's continuous-
        // iter pattern keeps primaries hot, which flatters
        // caller-as-worker on small N; a real one-off workload
        // (NMFD) has primaries parked between calls, where
        // caller-as-worker's eager own-pop loses the race vs
        // broadcast-wake + primary steal. Caller-parks-while-
        // primary-runs is strictly better for the cold-cache case.
        let parker = Arc::new(Parker::new(
            crate::sched::arena_local::LOCAL_SPIN_ROUNDS,
        ));
        let job = StackJob::new(
            move |_stolen: bool| -> (RA, RB) {
                let primary_ctx_ptr = current_worker_ctx();
                debug_assert!(
                    !primary_ctx_ptr.is_null(),
                    "wrap-and-park body must run on a worker thread"
                );
                // SAFETY: ctx lives for the duration of the
                // worker that runs this job.
                let primary_ctx = unsafe { &*primary_ctx_ptr };
                join_in_worker(primary_ctx, plan, a, b, true)
            },
            SpinLatch::new(parker.clone()),
        );
        unsafe {
            let r = job.as_job_ref(
                plan.k_outer,
                plan.numa_hint.unwrap_or(NUMA_HINT_ANY as u32) as u8,
                plan.variant,
            );
            // Push to slot's Public deque. Slot's stealer is in
            // arena.stealers from arena init -- primaries' random
            // victim pick can land on this slot and steal the job.
            slot_ctx.workers
                [crate::sched::deque_tier::DequeTier::Public.idx()]
                .push(r);
        }
        // Broadcast-wake all primaries so they wake + scan
        // immediately, dropping expected steal latency from a
        // single-sleeper wake's ~1/(n+slots) probability per
        // find_work round to near-1 within a few rounds.
        slot_ctx.sleep.new_internal_jobs(
            slot_ctx.stealers.len() as u32,
            false,
        );
        // Park caller on the SpinLatch via the sleep handshake
        // (UNSET -> SLEEPY -> SLEEPING -> set-wakes). Spin
        // briefly first to absorb fast-completion races before
        // dropping into park.
        const SLOT_WAIT_SPIN: usize = 256;
        let mut spun = 0usize;
        loop {
            if job.latch.is_set() {
                break;
            }
            if spun < SLOT_WAIT_SPIN {
                std::hint::spin_loop();
                spun += 1;
                continue;
            }
            if !job.latch.get_sleepy() {
                spun = 0;
                continue;
            }
            if job.latch.is_set() {
                job.latch.wake_up();
                break;
            }
            if !job.latch.fall_asleep() {
                spun = 0;
                continue;
            }
            let _unparked = parker.park_until(|| job.latch.is_set());
            job.latch.wake_up();
            spun = 0;
        }
        // SAFETY: latch is set => StackJob::execute finished =>
        // result slot populated.
        let result = unsafe { job.into_result() };
        drop(_guard);
        return result;
    }

    // Slot pool exhausted: fall through to the wrapper-job path
    // (the original external_dispatch design). Rare under normal
    // load -- only triggers when >32 caller threads concurrently
    // call sched::join from outside the pool. Tag with
    // core::hint::cold_path so LLVM (>= 21 since rust 1.96) reorders
    // basic blocks to keep the slot-pool fast path icache-warm and
    // pushes this wrapper-allocation + LockLatch park-handshake
    // sequence into a cold section.
    core::hint::cold_path();
    let wrapper = StackJob::new(
        move |_stolen: bool| -> (RA, RB) {
            let ctx_ptr = current_worker_ctx();
            debug_assert!(
                !ctx_ptr.is_null(),
                "external_dispatch wrapper must run on a worker thread"
            );
            let ctx = unsafe { &*ctx_ptr };
            join_in_worker(ctx, plan, a, b, true)
        },
        LockLatch::new(),
    );
    unsafe {
        let r = wrapper.as_job_ref(
            plan.k_outer,
            plan.numa_hint.unwrap_or(NUMA_HINT_ANY as u32) as u8,
            plan.variant,
        );
        arena.submit(r, plan.numa_hint);
    }
    let predicted_ns = plan.estimated_per_item_ns
        .map(|n| (n as u64).saturating_mul(plan.batch_size.max(1) as u64));
    let spin_cycles = match predicted_ns {
        Some(ns) => (ns / 3).clamp(1_000, 500_000) as usize,
        None => 200_000,
    };
    for _ in 0..spin_cycles {
        if wrapper.latch.is_set() {
            return unsafe { wrapper.into_result() };
        }
        std::hint::spin_loop();
    }
    wrapper.latch.wait();
    unsafe { wrapper.into_result() }
}

/// Run both closures serially on the caller thread. No scheduler
/// state touched. Cheapest possible join: zero allocations, zero
/// atomics, zero context switches. Both closures receive
/// `false` for their migrated/stolen flag (serial execution = no
/// steal pressure).
#[inline]
fn inline_join_context<A, B, RA, RB>(a: A, b: B) -> (RA, RB)
where
    A: FnOnce(bool) -> RA,
    B: FnOnce(bool) -> RB,
{
    let ra = a(false);
    let rb = b(false);
    (ra, rb)
}

/// Convenience: build a default plan for `(k_outer, batch_size)`
/// and call [`join`]. Use this when you do not need to customize
/// hw_class / variant / numa_hint.
pub fn join_default<A, B, RA, RB>(
    k_outer: u8,
    batch_size: u32,
    a: A,
    b: B,
) -> (RA, RB)
where
    A: FnOnce() -> RA + Send,
    B: FnOnce() -> RB + Send,
    RA: Send,
    RB: Send,
{
    let plan = JobPlan::new(k_outer, batch_size);
    join(&plan, a, b)
}

/// Silence the unused-import warning when no caller has reached
/// for `NUMA_HINT_ANY` from this module. The constant is re-
/// exported at the crate root via `crate::sched::NUMA_HINT_ANY`.
#[allow(dead_code)]
const _NUMA_HINT_ANY_REEXPORT: u8 = NUMA_HINT_ANY;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU32, Ordering};

    use crate::foundation::Variant;
    use crate::sched::plan::HwClass;

    #[test]
    fn inline_join_returns_both_results() {
        let plan = JobPlan::new(2, 1);
        let (ra, rb) = join(&plan, || 10u32, || 20u32);
        assert_eq!(ra, 10);
        assert_eq!(rb, 20);
    }

    #[test]
    fn inline_join_runs_a_before_b_when_serial() {
        // The serial dispatch runs a first, then b. Closures that
        // observe shared state must see the order (10, 20).
        let order = Arc::new(AtomicU32::new(0));
        let order_a = Arc::clone(&order);
        let order_b = Arc::clone(&order);
        let plan = JobPlan::new(2, 1);
        let (ra, rb) = join(
            &plan,
            move || order_a.fetch_add(10, Ordering::SeqCst),
            move || order_b.fetch_add(20, Ordering::SeqCst),
        );
        // a saw 0 (the original); b saw 10 (after a added 10).
        assert_eq!(ra, 0);
        assert_eq!(rb, 10);
        assert_eq!(order.load(Ordering::SeqCst), 30);
    }

    #[test]
    fn inline_join_propagates_panic_from_a() {
        let plan = JobPlan::new(2, 1);
        let r = std::panic::catch_unwind(|| {
            join::<_, _, u32, u32>(&plan, || panic!("a-side panic"), || 7u32)
        });
        assert!(r.is_err(), "a-side panic must propagate to caller");
    }

    #[test]
    fn join_default_builds_plan_and_runs() {
        let (ra, rb) = join_default(8, 1024, || 1u32, || 2u32);
        assert_eq!((ra, rb), (1, 2));
    }

    #[test]
    fn join_with_each_tier_band_dispatches_correctly() {
        // K = 2 (Inline band), K = 6 (Local band), K = 9
        // (Hierarchical or Local depending on NUMA), K = 13
        // (Federated band). All four tiers currently route through
        // the inline path; the contract is "results returned, no
        // crash" for each band.
        for k in [2u8, 6, 9, 13] {
            let plan = JobPlan::new(k, 1024)
                .with_hw_class(HwClass::Scalar)
                .with_variant(Variant::Faithful);
            let (ra, rb) = join(&plan, || k as u32, || (k as u32) << 1);
            assert_eq!(ra, k as u32);
            assert_eq!(rb, (k as u32) << 1);
        }
    }

    #[test]
    fn join_with_numa_hint_does_not_crash_single_node_host() {
        // The numa_hint is currently a tag; on a single-NUMA host
        // it is recorded in the JobPlan and ignored by the
        // dispatcher. Verify no crash.
        let plan = JobPlan::new(6, 1024).with_numa_hint(0);
        let (ra, rb) = join(&plan, || 1u32, || 2u32);
        assert_eq!((ra, rb), (1, 2));
    }
}
