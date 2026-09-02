//! Cooperative N-way fork-join with adaptive dispatch.
//!
//! [`cooperative_join_n`] picks a shape from N: tree (balanced
//! binary `sched::join` recursion, depth log2(N)) below N = 12,
//! flat fan-out (one StackJob per closure, critical-path depth 1)
//! at N >= 12. Tree amortizes per-StackJob overhead for short
//! closures; flat wins once log2(N) nesting extends the wait for
//! the slowest closure. The N = 12 crossover gave the same routing
//! decision on every workload measured across Zen+ R7 2700, Xeon
//! Cascade Lake, and Genoa EPYC 9B14; a timed probe was rejected
//! because it adds one closure of serial latency (~2ms on a 4.7ms
//! large-N total). [`cooperative_join_n_tree`] and
//! [`cooperative_join_n_flat`] bypass the heuristic.
//!
//! The returned `Vec<R>` is in caller-supplied order regardless of
//! which worker ran which closure; bit-exact reductions rely on
//! this. Both variants forward the [`crate::sched::JobPlan`] to
//! every nested dispatch. The flat variant's external entry lands
//! one injected StackJob on the local NUMA node and pushes all N
//! children onto that worker's deque, so the cooperative cluster
//! stays intra-node unless peer steal pulls work across.

use std::panic;

use crate::sched::arena::{global_local_arena, join};
use crate::sched::arena_local::{WorkerCtx, current_worker_ctx};
use crate::sched::job::{NUMA_HINT_ANY, StackJob};
use crate::sched::latch::{CountLatch, LockLatch};
use crate::sched::plan::JobPlan;

/// Adaptive cooperative fork-join over an arbitrary-N closure
/// list. Returns a `Vec` of results in caller-supplied order.
///
/// N = 0 returns an empty `Vec`; N = 1 runs inline; N = 2 is
/// [`crate::sched::join`]. At N >= 3 the shape resolves per the
/// [`crate::sched::adaptive_cooperative`] precedence: per-plan
/// `cooperative_routing` (non-`Auto`) wins, else the process-global
/// tag, else the population heuristic - `N < n_workers` routes to
/// [`cooperative_join_n_tree`], `N >= n_workers` to
/// [`cooperative_join_n_flat_mailbox`] (each closure pushed to a
/// specific peer's mailbox, drained mailbox-first, zero shared-deque
/// CAS contention). The explicit variants and
/// [`crate::sched::JobPlan::with_cooperative_routing`] bypass the
/// heuristic.
///
/// # Determinism contract
///
/// The returned `Vec<R>` is in caller-supplied order, invariant
/// regardless of which thread executed which closure.
///
/// # Example
///
/// ```ignore
/// use flynnel::sched::{JobPlan, cooperative_join_n};
///
/// let plan = JobPlan::new(8, 1);
/// let closures: Vec<Box<dyn FnOnce() -> u32 + Send>> = vec![
///     Box::new(|| 1),
///     Box::new(|| 2),
///     Box::new(|| 3),
/// ];
/// let results = cooperative_join_n(&plan, closures);
/// assert_eq!(results, vec![1, 2, 3]);
/// ```
pub fn cooperative_join_n<R>(
    plan: &JobPlan,
    closures: Vec<Box<dyn FnOnce() -> R + Send>>,
) -> Vec<R>
where
    R: Send + 'static,
{
    let n = closures.len();
    if n < 3 {
        // Inline / 2-way fast paths live inside the tree variant.
        return cooperative_join_n_tree(plan, closures);
    }
    // N >= 3: resolve the dispatch shape through the precedence chain.
    // Per-plan override wins outright; Auto defers to the population
    // heuristic. The plan field was resolved from the process-global
    // active tag at JobPlan::new construction time, so the global flip
    // via migrate_cooperative_routing propagates here without an
    // additional atomic load on the cooperative hot path.
    use crate::sched::adaptive_cooperative::CooperativeRouting;
    match plan.cooperative_routing {
        CooperativeRouting::ForceTree => cooperative_join_n_tree(plan, closures),
        CooperativeRouting::ForceMailbox => {
            cooperative_join_n_flat_mailbox(plan, closures)
        }
        CooperativeRouting::ForceDeque => cooperative_join_n_flat(plan, closures),
        CooperativeRouting::Auto => {
            // Population heuristic: N < n_workers routes to tree
            // (under-populated pool); N >= n_workers routes to mailbox
            // (matches the gate inside cooperative_join_n_flat_mailbox).
            let n_workers = global_local_arena().total_workers();
            if n < n_workers {
                cooperative_join_n_tree(plan, closures)
            } else {
                cooperative_join_n_flat_mailbox(plan, closures)
            }
        }
    }
}

/// Tree-shape cooperative fork-join: balanced binary bisect of
/// `sched::join` calls, depth `log2(N)`. Pick this directly when
/// you know per-closure work is short (sub-100us) and want to
/// skip the probe cost.
///
/// # Determinism contract
///
/// Same as [`cooperative_join_n`]: caller-supplied result order.
///
/// # Algebraic balancing
///
/// The split point is `len / 2` (left-biased on odd N). For
/// `N = 5` this produces the tree `((c0, c1), ((c2, c3), c4))`.
/// The shape is deterministic given N; it does not depend on
/// available worker count.
pub fn cooperative_join_n_tree<R>(
    plan: &JobPlan,
    closures: Vec<Box<dyn FnOnce() -> R + Send>>,
) -> Vec<R>
where
    R: Send + 'static,
{
    let n = closures.len();
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        let mut iter = closures.into_iter();
        let c = iter.next().expect("len == 1 checked");
        return vec![c()];
    }
    if n == 2 {
        let mut iter = closures.into_iter();
        let a = iter.next().expect("len == 2 checked");
        let b = iter.next().expect("len == 2 checked");
        let (ra, rb) = join(plan, a, b);
        return vec![ra, rb];
    }

    let mid = n / 2;
    let mut closures = closures;
    let right = closures.split_off(mid);
    let left = closures;

    let plan_left = *plan;
    let plan_right = *plan;

    let (left_results, right_results) = join(
        plan,
        move || cooperative_join_n_tree::<R>(&plan_left, left),
        move || cooperative_join_n_tree::<R>(&plan_right, right),
    );

    let mut out = left_results;
    out.extend(right_results);
    out
}

/// Flat-shape fork-join over an arbitrary-N closure list. Pushes
/// every closure as one StackJob onto the dispatching worker's
/// local Chase-Lev deque in a single fan-out (critical-path depth
/// 1), then waits on all N latches while participating in
/// work-stealing.
///
/// # When to use
///
/// Pick this when closures are uniformly long (>= 100us each) and
/// N is large enough that the tree's `log2(N)` depth measurably
/// adds to the critical path (typically N >= 12). For short
/// per-closure work or small N, the per-StackJob overhead (one
/// Box alloc + one latch wait per closure) outweighs the saved
/// tree depth; use [`cooperative_join_n`] instead.
///
/// # Determinism contract
///
/// Same as [`cooperative_join_n`]: results returned in caller
/// order regardless of which thread executed which closure.
///
/// # Example
///
/// ```ignore
/// use flynnel::sched::{JobPlan, cooperative_join_n_flat};
///
/// let plan = JobPlan::new(8, 1);
/// let closures: Vec<Box<dyn FnOnce() -> u32 + Send>> =
///     (0..12u32).map(|i| Box::new(move || i * 10) as _).collect();
/// let results = cooperative_join_n_flat(&plan, closures);
/// assert_eq!(results.len(), 12);
/// ```
pub fn cooperative_join_n_flat<R>(
    plan: &JobPlan,
    closures: Vec<Box<dyn FnOnce() -> R + Send>>,
) -> Vec<R>
where
    R: Send + 'static,
{
    let n = closures.len();
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        let mut iter = closures.into_iter();
        let c = iter.next().expect("len == 1 checked");
        return vec![c()];
    }
    if n == 2 {
        let mut iter = closures.into_iter();
        let a = iter.next().expect("len == 2 checked");
        let b = iter.next().expect("len == 2 checked");
        let (ra, rb) = join(plan, a, b);
        return vec![ra, rb];
    }

    // N >= 3: flat fan-out. Pick the in-worker fast path when
    // already on a Flynnel worker; otherwise wrap in a single
    // injected StackJob (mirrors `sched::arena::external_dispatch`)
    // that runs the fan-out from a worker.
    let ctx_ptr = current_worker_ctx();
    if !ctx_ptr.is_null() {
        // SAFETY: `current_worker_ctx` returns a pointer set by
        // `worker_loop` on this same thread; it is valid until
        // `worker_loop` returns.
        let ctx = unsafe { &*ctx_ptr };
        return fan_out_in_worker(ctx, plan, closures, FanOutMode::Deque);
    }
    fan_out_external(plan, closures, FanOutMode::Deque)
}

/// Mailbox-distribute variant of [`cooperative_join_n_flat`]. Each
/// child closure routes directly to a SPECIFIC peer's mailbox via
/// owner-directed distribution (URD-style); the target worker
/// drains its mailbox FIRST in `find_work`, so each closure starts
/// on its assigned core with zero shared-deque CAS contention.
///
/// **Use when:** the workload is N uniform-cost independent
/// closures (canonical SIMC pattern) AND N >= 3. The mailbox path
/// is the SIMC primitive's structural fit per the Flynn-axis
/// taxonomy (`SIMC = cooperative_join_n + owner-directed
/// distribution`).
///
/// **Use the regular [`cooperative_join_n_flat`] instead when:**
/// closures are heterogeneous (some take longer than others) - the
/// mailbox round-robin doesn't observe per-peer load and can
/// concentrate slow closures on one worker while others idle. The
/// deque variant uses random peer-steal which naturally rebalances.
///
/// # Determinism contract
///
/// Same as [`cooperative_join_n_flat`]: results returned in
/// caller-supplied order regardless of execution order.
pub fn cooperative_join_n_flat_mailbox<R>(
    plan: &JobPlan,
    closures: Vec<Box<dyn FnOnce() -> R + Send>>,
) -> Vec<R>
where
    R: Send + 'static,
{
    let n = closures.len();
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        let mut iter = closures.into_iter();
        let c = iter.next().expect("len == 1 checked");
        return vec![c()];
    }
    if n == 2 {
        let mut iter = closures.into_iter();
        let a = iter.next().expect("len == 2 checked");
        let b = iter.next().expect("len == 2 checked");
        let (ra, rb) = join(plan, a, b);
        return vec![ra, rb];
    }
    let ctx_ptr = current_worker_ctx();
    if !ctx_ptr.is_null() {
        // SAFETY: same as cooperative_join_n_flat.
        let ctx = unsafe { &*ctx_ptr };
        // Architectural gate: mailbox-distribute is only correct
        // when N >= n_workers. When N < n_workers, mailbox-pushed
        // closures target a SPECIFIC subset of workers; the parent
        // and the remaining (n_workers - N) workers cannot help via
        // peer-steal (peer-steal probes deques, not mailboxes), so
        // the parent spins idle while a subset processes serially.
        // Measured 4.4x slower than the deque variant on the Zen+
        // R7 2700 N=8 / n_workers=16 case. Fall through to deque
        // mode to preserve broad-steal load balance for under-
        // populated calls.
        let n_workers = ctx.stealers.len();
        let mode = if n >= n_workers {
            FanOutMode::Mailbox
        } else {
            FanOutMode::Deque
        };
        return fan_out_in_worker(ctx, plan, closures, mode);
    }
    // External path: we do not yet have a worker context (so no
    // n_workers signal). Default to Deque so the worker the wrapper
    // lands on can apply the same gate inside fan_out_in_worker.
    fan_out_external(plan, closures, FanOutMode::Mailbox)
}

/// Cooperative fan-out distribution mode. Selects how the N-1
/// child closures are routed to peer workers in the in-worker
/// fast path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FanOutMode {
    /// All N-1 closures pushed to the calling worker's local
    /// deque (default Public tier). Peer workers pull via random
    /// peer-steal. Matches the historical behavior of
    /// [`cooperative_join_n_flat`] before the mailbox variant
    /// shipped.
    Deque,
    /// Each closure pushed directly to a SPECIFIC peer worker's
    /// mailbox via `WorkerCtx::push_to_mailbox`. Target rotates
    /// starting at `caller.index + 1` (the SMT sibling on the
    /// standard enumeration), skipping self, wrapping around.
    /// The target worker drains its mailbox FIRST in `find_work`,
    /// so each closure starts on its assigned core with zero
    /// shared-deque CAS contention. URD-style owner-directed
    /// distribution back-ported in-process.
    ///
    /// Fallback: if a target mailbox is full (bounded at
    /// `MAILBOX_CAPACITY = 16`), the JobRef falls through to
    /// `ctx.push_burst()` so peer-steal still catches it. With
    /// MAILBOX_CAPACITY=16, N up to `n_workers * 16` per call
    /// fits without spilling.
    Mailbox,
}

/// Convenience alias for the boxed-closure shape stored in each
/// `StackJob` for the cooperative fan-out path.
type WrappedClosure<R> = Box<dyn FnOnce(bool) -> R + Send>;

/// Type alias for one StackJobShared in the flat fan-out path. All
/// N-1 share the same `Arc<CountLatch>` so the parent waits on ONE
/// atomic count reaching zero rather than walking N-1 latches.
type FanOutStackJob<R> =
    crate::sched::job::StackJobShared<CountLatch, WrappedClosure<R>, R>;

/// In-worker flat fan-out: build N-1 StackJobs, push them onto
/// the calling worker's local deque (or distribute via mailbox
/// per `mode`), run the last closure inline, then wait on all
/// N-1 latches while participating in stealing.
fn fan_out_in_worker<R>(
    ctx: &WorkerCtx,
    plan: &JobPlan,
    mut closures: Vec<Box<dyn FnOnce() -> R + Send>>,
    mode: FanOutMode,
) -> Vec<R>
where
    R: Send + 'static,
{
    let n = closures.len();
    debug_assert!(n >= 2, "fan_out_in_worker should not see n < 2 (caller branches early)");

    // Pop the last closure for inline execution on this worker
    // while siblings drain the deque. The inline closure keeps
    // the calling worker productively busy and avoids paying a
    // StackJob+latch round-trip for it.
    let last_closure = closures.pop().expect("n >= 2 checked");

    // Build ONE shared CountLatch with count == N-1 (the inline
    // last_closure does NOT decrement; only the dispatched N-1
    // siblings do). The parent waits on the calling worker's own
    // Parker via this latch.
    //
    // CountLatch is the canonical wake-capable N-participant latch
    // at `sched::latch::CountLatch`, and this fan-out is its
    // production consumer: one atomic count gives the parent a
    // single-poll wait surface instead of walking N-1 separate
    // CoreLatches in `is_set` order.
    let n_to_dispatch = closures.len();
    let my_parker: std::sync::Arc<crate::sched::sleep::Parker> = ctx
        .parkers[ctx.index]
        .get()
        .expect("worker parker initialized by worker_loop before any fan_out_in_worker call")
        .clone();
    let shared_latch: std::sync::Arc<CountLatch> =
        std::sync::Arc::new(CountLatch::new(n_to_dispatch, my_parker));
    // Build N-1 StackJobShared in `Box`es so the StackJob memory
    // has a stable heap address. Each StackJob holds an Arc clone
    // of the shared CountLatch.
    let stack_jobs: Vec<Box<FanOutStackJob<R>>> = closures
        .into_iter()
        .map(|c| {
            let wrapped: WrappedClosure<R> = Box::new(move |_stolen| c());
            Box::new(crate::sched::job::StackJobShared::new(
                wrapped,
                shared_latch.clone(),
            ))
        })
        .collect();

    let numa_hint_byte = plan.numa_hint.unwrap_or(NUMA_HINT_ANY as u32) as u8;
    let n_workers = ctx.stealers.len();
    // Mailbox-distribute gate: only correct when N >= n_workers.
    // See cooperative_join_n_flat_mailbox doc + root-cause comment
    // for why N < n_workers regresses 4.4x: parent + (n_workers - N)
    // workers stay idle during the wait because peer-steal cannot
    // probe mailboxes. Demote to Deque mode when the workload is
    // under-populated relative to the pool.
    let effective_mode = match mode {
        FanOutMode::Mailbox if n < n_workers => FanOutMode::Deque,
        other => other,
    };
    // Wake-protocol selection per mode (the bench-audit found
    // these two modes need OPPOSITE wake protocols):
    //
    // - Deque mode: keep the per-push wake. The first push hits
    //   own deque empty -> non-empty (one broadcast); subsequent
    //   pushes hit a non-empty deque (no broadcast). Workers wake
    //   IMMEDIATELY when the first item lands and start probing
    //   the parent's deque via peer-steal while the parent is
    //   still pushing - pipeline parallelism. Measured 3.7x
    //   regression on N=8 when we tried to defer this broadcast.
    //
    // - Mailbox mode: disable per-push wake; issue ONE batched
    //   broadcast at the end. Each push targets a DIFFERENT
    //   mailbox, so each push would otherwise fire its own
    //   empty -> non-empty broadcast (n-1 broadcasts, each
    //   cascading through every worker's parker - waste, since
    //   only the targeted worker can pop that specific mailbox).
    //   The batched single broadcast wakes everyone once after
    //   all targets are loaded. Measured 1.5x -> ~1.0x at N=16.
    let _wake_scope = match effective_mode {
        FanOutMode::Mailbox => {
            // Use new_if_change so we skip the TLS write+drop when
            // the cell already holds `false` (e.g., nested mailbox
            // scopes). Wraps the Option in another Option but the
            // outer Option carries None when the inner is None too.
            crate::sched::arena_local::DispatchScope::new_if_change(false)
        }
        FanOutMode::Deque => None,
    };
    let n_to_push = stack_jobs.len();
    // SAFETY: each StackJob lives in a Box on the heap; the Box
    // is owned by `stack_jobs` which outlives the latch-wait loop
    // below. Each `as_job_ref` returns a JobRef pointing at the
    // StackJob's stable heap address.
    // Producer-fast fan-out: use push_burst so 3 consecutive jobs
    // pack into one cache-line slot (K_inner=3 amortization). The
    // explicit flush_all below publishes all buffered slots and
    // broadcasts a single JEC wake covering them.
    unsafe {
        for (i, sj) in stack_jobs.iter().enumerate() {
            let r = sj.as_job_ref(plan.k_outer, numa_hint_byte, plan.variant);
            match effective_mode {
                FanOutMode::Deque => {
                    ctx.push_burst(r);
                }
                FanOutMode::Mailbox => {
                    // Round-robin starting at (ctx.index + 1) so
                    // closure[0] goes to the SMT sibling first
                    // (under the standard adjacent-sibling x86
                    // enumeration).
                    let target = if n_workers > 1 {
                        let mut t = (ctx.index + 1 + i) % n_workers;
                        if t == ctx.index {
                            t = (t + 1) % n_workers;
                        }
                        t
                    } else {
                        ctx.index
                    };
                    if target == ctx.index {
                        // Self-push: still burst (the same worker
                        // will pop these immediately after the
                        // wait-loop starts).
                        ctx.push_burst(r);
                    } else if let Err(returned) =
                        ctx.push_to_mailbox(target, r)
                    {
                        // Mailbox full: fall back to burst on our
                        // own deque.
                        ctx.push_burst(returned);
                    }
                }
            }
        }
    }
    // Deque-mode publishes accumulated bursts via flush_all so
    // thieves see them as 3-per-slot batches. Mailbox-mode also
    // calls flush_all so any self-push bursts (target == ctx.index
    // case) become visible; mailbox pushes themselves are direct.
    ctx.flush_all();
    crate::sched::trace::emit(crate::sched::trace::TraceEvent::JoinPush, n as u32);
    // Drop the wake_scope (if any) so DISPATCH_USE_JEC_WAKE
    // returns to true for subsequent in-wait-loop pushes.
    let needs_batched_broadcast = _wake_scope.is_some();
    drop(_wake_scope);
    // Mailbox-mode only: issue ONE batched broadcast covering all
    // n_to_push items that were pushed silently. Deque mode skips
    // this because its first push already broadcast immediately
    // via the per-push wake path.
    if needs_batched_broadcast && n_to_push > 0 {
        ctx.sleep.new_internal_jobs(n_to_push as u32, true);
    }

    // Run the last closure inline. Capture any panic so we still
    // wait on the sibling StackJobs (a panic that escapes here
    // would leave workers reading freed StackJob memory).
    let last_result = panic::catch_unwind(panic::AssertUnwindSafe(last_closure));

    crate::sched::trace::emit(crate::sched::trace::TraceEvent::JoinWaitBegin, n as u32);

    // Wait for the shared CountLatch (count drops to 0 when all
    // N-1 siblings have called Latch::set). Single atomic poll
    // per wait-loop iteration instead of walking N-1 separate
    // latches; same behavior because the original loop exited
    // only when the slowest child finished regardless.
    while !shared_latch.is_set() {
        if let Some(job) = ctx.find_work() {
            // SAFETY: `JobRef::execute` contract: execute exactly
            // once. We just popped this JobRef from the local
            // deque / injector / a peer's deque so we hold the
            // only outstanding reference to it.
            unsafe { job.execute() };
        } else {
            std::thread::yield_now();
        }
    }
    crate::sched::trace::emit(crate::sched::trace::TraceEvent::JoinWaitEnd, n as u32);

    // Collect results in caller order: results from stack_jobs[0..n-1]
    // first, then the inline `last_result`.
    let mut out: Vec<R> = Vec::with_capacity(n);
    for sj in stack_jobs {
        // SAFETY: `latch.is_set()` returned true above, which
        // means `StackJob::execute` already wrote into the
        // result slot. `into_result` consumes the Box's interior.
        out.push(unsafe { (*sj).into_result() });
    }
    match last_result {
        Ok(r) => out.push(r),
        Err(payload) => {
            // Drop the sibling results so their destructors run
            // before we resume the panic.
            drop(out);
            panic::resume_unwind(payload);
        }
    }
    out
}

/// External-entry flat fan-out: wrap the in-worker fan-out in a
/// single StackJob, inject it into the NumaArena, and park the
/// caller until the latch fires. Mirrors the wrapper pattern from
/// [`crate::sched::arena::external_dispatch`].
fn fan_out_external<R>(
    plan: &JobPlan,
    closures: Vec<Box<dyn FnOnce() -> R + Send>>,
    mode: FanOutMode,
) -> Vec<R>
where
    R: Send + 'static,
{
    let arena = global_local_arena();
    // Acquire SMT-sibling guards for the lifetime of the dispatch
    // when the plan asks for SMT participation, the same as
    // `external_dispatch` does for the 2-way join.
    let _smt_guards: Vec<crate::sched::arena_local::SmtGuard> = if plan.effective_use_smt() {
        arena.acquire_smt()
    } else {
        Vec::new()
    };

    // LockLatch (Mutex+Condvar) so the foreign caller blocks
    // directly on the wrapper completion instead of polling via
    // park_timeout. Mirrors the wiring in arena::external_dispatch:
    // the wrapper-running worker holds CPU during the fan-out
    // while the caller sleeps on the condvar; wake latency is
    // bounded by mutex acquisition + notify_one (microseconds).
    let plan_copy = *plan;
    let wrapper = StackJob::new(
        move |_stolen: bool| -> Vec<R> {
            let ctx_ptr = current_worker_ctx();
            debug_assert!(
                !ctx_ptr.is_null(),
                "fan_out_external wrapper must run on a worker thread"
            );
            // SAFETY: ctx_ptr was set by worker_loop on the worker
            // that picked up this wrapper; valid until that
            // worker_loop returns.
            let ctx = unsafe { &*ctx_ptr };
            fan_out_in_worker(ctx, &plan_copy, closures, mode)
        },
        LockLatch::new(),
    );

    let numa_hint_byte = plan.numa_hint.unwrap_or(NUMA_HINT_ANY as u32) as u8;
    // SAFETY: the wrapper lives on this stack frame; we wait on
    // its latch before returning.
    unsafe {
        let r = wrapper.as_job_ref(plan.k_outer, numa_hint_byte, plan.variant);
        arena.submit(r, plan.numa_hint);
    }

    // Hybrid sleep policy mirroring arena::external_dispatch: a
    // brief spin on the LockLatch's fast-path AtomicBool catches
    // sub-200us fan-outs without paying mutex+condvar latency,
    // then the loop falls through to wait() which sleeps on the
    // Mutex+Condvar pair for longer dispatches.
    const SPIN_CYCLES: usize = 500_000;
    for _ in 0..SPIN_CYCLES {
        if wrapper.latch.is_set() {
            // SAFETY: latch is set => wrapper closure ran to
            // completion and wrote the result slot.
            return unsafe { wrapper.into_result() };
        }
        std::hint::spin_loop();
    }
    // Spin window expired; block on the LockLatch condvar.
    wrapper.latch.wait();

    // SAFETY: latch is set => wrapper closure ran to completion
    // and wrote the result slot.
    unsafe { wrapper.into_result() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU32, Ordering};

    fn plan() -> JobPlan {
        JobPlan::new(8, 1)
    }

    // -----------------------------------------------------------------
    // Tree variant (default): N-way recursive bisect
    // -----------------------------------------------------------------

    #[test]
    fn tree_empty_input_returns_empty_vec() {
        let p = plan();
        let v: Vec<Box<dyn FnOnce() -> u32 + Send>> = Vec::new();
        let results = cooperative_join_n(&p, v);
        assert!(results.is_empty());
    }

    #[test]
    fn tree_one_closure_runs_inline() {
        let p = plan();
        let closures: Vec<Box<dyn FnOnce() -> u32 + Send>> =
            vec![Box::new(|| 42)];
        let results = cooperative_join_n(&p, closures);
        assert_eq!(results, vec![42]);
    }

    #[test]
    fn tree_two_closures_route_through_join() {
        let p = plan();
        let closures: Vec<Box<dyn FnOnce() -> u32 + Send>> =
            vec![Box::new(|| 10), Box::new(|| 20)];
        let results = cooperative_join_n(&p, closures);
        assert_eq!(results, vec![10, 20]);
    }

    #[test]
    fn tree_three_closures_preserve_caller_order() {
        let p = plan();
        let closures: Vec<Box<dyn FnOnce() -> u32 + Send>> = vec![
            Box::new(|| 1),
            Box::new(|| 2),
            Box::new(|| 3),
        ];
        let results = cooperative_join_n(&p, closures);
        assert_eq!(results, vec![1, 2, 3]);
    }

    #[test]
    fn tree_eight_closures_balanced_tree() {
        let p = plan();
        let closures: Vec<Box<dyn FnOnce() -> u32 + Send>> =
            (0..8u32).map(|i| Box::new(move || i * 10) as _).collect();
        let results = cooperative_join_n(&p, closures);
        assert_eq!(results, vec![0, 10, 20, 30, 40, 50, 60, 70]);
    }

    #[test]
    fn tree_sixteen_closures() {
        let p = plan();
        let closures: Vec<Box<dyn FnOnce() -> u32 + Send>> =
            (0..16u32).map(|i| Box::new(move || i) as _).collect();
        let results = cooperative_join_n(&p, closures);
        let expected: Vec<u32> = (0..16u32).collect();
        assert_eq!(results, expected);
    }

    // -----------------------------------------------------------------
    // Flat variant: opt-in, one StackJob per closure
    // -----------------------------------------------------------------

    #[test]
    fn flat_empty_input_returns_empty_vec() {
        let p = plan();
        let v: Vec<Box<dyn FnOnce() -> u32 + Send>> = Vec::new();
        let results = cooperative_join_n_flat(&p, v);
        assert!(results.is_empty());
    }

    #[test]
    fn flat_one_closure_runs_inline() {
        let p = plan();
        let closures: Vec<Box<dyn FnOnce() -> u32 + Send>> =
            vec![Box::new(|| 42)];
        let results = cooperative_join_n_flat(&p, closures);
        assert_eq!(results, vec![42]);
    }

    #[test]
    fn flat_two_closures_route_through_join() {
        let p = plan();
        let closures: Vec<Box<dyn FnOnce() -> u32 + Send>> =
            vec![Box::new(|| 10), Box::new(|| 20)];
        let results = cooperative_join_n_flat(&p, closures);
        assert_eq!(results, vec![10, 20]);
    }

    #[test]
    fn flat_three_closures_preserve_caller_order() {
        let p = plan();
        let closures: Vec<Box<dyn FnOnce() -> u32 + Send>> = vec![
            Box::new(|| 1),
            Box::new(|| 2),
            Box::new(|| 3),
        ];
        let results = cooperative_join_n_flat(&p, closures);
        assert_eq!(results, vec![1, 2, 3]);
    }

    #[test]
    fn flat_twelve_closures_match_xeon_simc_shape() {
        // Same N as the Xeon SIMC bench (`2 * physical_cores = 12`
        // on the 6-physical Xeon). The flat variant is the
        // recommended pick for this shape; the body is large
        // enough (~5ms per closure in the real bench) that the
        // per-StackJob overhead amortizes and the lower
        // critical-path depth wins.
        let p = plan();
        let closures: Vec<Box<dyn FnOnce() -> u64 + Send>> = (0..12u64)
            .map(|i| Box::new(move || {
                let mut v: u64 = i.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
                for _ in 0..200_000u64 {
                    v = v.wrapping_mul(0x9E37_79B9_7F4A_7C15);
                    v ^= v >> 31;
                }
                v
            }) as _)
            .collect();
        let results = cooperative_join_n_flat(&p, closures);
        assert_eq!(results.len(), 12);
        for (i, v) in results.iter().enumerate() {
            assert_ne!(*v, 0, "closure {i} produced zero, body did not run");
        }
    }

    #[test]
    fn flat_sixteen_closures_full_fan_out() {
        let p = plan();
        let closures: Vec<Box<dyn FnOnce() -> u32 + Send>> =
            (0..16u32).map(|i| Box::new(move || i) as _).collect();
        let results = cooperative_join_n_flat(&p, closures);
        let expected: Vec<u32> = (0..16u32).collect();
        assert_eq!(results, expected);
    }

    #[test]
    fn flat_results_in_caller_order_invariant_of_execution_thread() {
        let p = plan();
        let exec_order = Arc::new(AtomicU32::new(0));
        let closures: Vec<Box<dyn FnOnce() -> (u32, u32) + Send>> = (0..8u32)
            .map(|i| {
                let order = Arc::clone(&exec_order);
                Box::new(move || {
                    let run_idx = order.fetch_add(1, Ordering::SeqCst);
                    (i, run_idx)
                }) as _
            })
            .collect();
        let results = cooperative_join_n_flat(&p, closures);
        let caller_indices: Vec<u32> = results.iter().map(|(i, _)| *i).collect();
        assert_eq!(caller_indices, vec![0, 1, 2, 3, 4, 5, 6, 7]);
    }

    #[test]
    fn flat_closures_return_different_types_via_boxed_trait_object() {
        let p = plan();
        let closures: Vec<Box<dyn FnOnce() -> String + Send>> = vec![
            Box::new(|| "alpha".to_string()),
            Box::new(|| "beta".to_string()),
            Box::new(|| "gamma".to_string()),
            Box::new(|| "delta".to_string()),
        ];
        let results = cooperative_join_n_flat(&p, closures);
        assert_eq!(
            results,
            vec![
                "alpha".to_string(),
                "beta".to_string(),
                "gamma".to_string(),
                "delta".to_string(),
            ]
        );
    }

    #[test]
    fn nested_tree_inside_flat_composes() {
        let p = plan();

        let mut outer: Vec<Box<dyn FnOnce() -> Vec<u32> + Send>> =
            Vec::with_capacity(4);
        for o in 0..4u32 {
            let p_inner = p;
            outer.push(Box::new(move || {
                let inner: Vec<Box<dyn FnOnce() -> u32 + Send>> = (0..3u32)
                    .map(|i| Box::new(move || o * 100 + i) as _)
                    .collect();
                cooperative_join_n(&p_inner, inner)
            }));
        }
        let results = cooperative_join_n_flat(&p, outer);
        let flat: Vec<u32> = results.into_iter().flatten().collect();
        assert_eq!(
            flat,
            vec![0, 1, 2, 100, 101, 102, 200, 201, 202, 300, 301, 302]
        );
    }
}
