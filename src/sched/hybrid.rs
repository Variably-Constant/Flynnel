//! Hybrid CPU+backend dispatch: the MIMT (Multiple Instruction,
//! Multiple Threads) entry point.
//!
//! [`join_hybrid`] runs two closures concurrently - the CPU half on
//! the calling thread (or work-stealing arena) and the GPU / TPU
//! half on whatever [`crate::backend::DispatchBackend`] the
//! [`JobPlan`] hint selects. Results return as a `(cpu_result,
//! gpu_result)` tuple, mirroring [`crate::sched::arena::join`]'s
//! shape so consumers see a familiar surface.
//!
//! ## Breakeven consideration
//!
//! The hybrid path is worth it only when the GPU half's per-call
//! work dominates the backend's launch latency. `join_hybrid`
//! does NOT gate this internally: it always resolves the backend
//! from the plan (falling through to the CPU backend when no
//! hint is set or the hinted backend is not registered) and
//! dispatches the second closure via
//! [`crate::backend::DispatchBackend::dispatch_one`]. Callers
//! that want a breakeven gate should compare
//! [`JobPlan::estimated_total_ns`] against
//! [`crate::backend::BackendCapabilities::launch_latency_ns`]
//! themselves and call [`crate::sched::arena::join`] instead
//! when the ratio is unfavorable. A rule-of-thumb ratio is
//! `estimated_total_ns >= backend.launch_latency_ns * 4`;
//! empirical breakeven varies by backend and workload.
//!
//! ## Execution model
//!
//! - The GPU half runs on a dedicated host-side thread spawned via
//!   [`crate::backend::DispatchBackend::dispatch_one`]. For real
//!   GPU backends that thread typically issues a kernel launch and
//!   waits for completion via the backend's synchronization
//!   primitives.
//! - The CPU half runs on the calling thread (so the caller can
//!   inline it through any in-flight closure context).
//! - A bounded notify ring carries the GPU result back to
//!   the calling thread, where it pairs with the CPU result.
//!
//! ## Panic propagation
//!
//! Either half panicking propagates through [`join_hybrid`] - the
//! GPU half's panic is captured on the spawned thread and
//! re-raised on the calling thread after the CPU half completes.

use crate::sched::call_site::Placement;
use crate::sched::notify_ring::{NotifyHub, NotifySendResult};
use crate::sched::plan::JobPlan;

/// Run `cpu_work` and `gpu_work` concurrently. `cpu_work` runs on
/// the calling thread; `gpu_work` runs on whichever backend
/// [`JobPlan::pick_backend`] selects (the CPU backend if no hint
/// is set or the hinted backend isn't registered).
///
/// Returns `(cpu_result, gpu_result)` in caller-supplied order.
///
/// # Panics
///
/// Resumes a panic from either half. A panic from the GPU half is
/// re-raised on the calling thread after the CPU half completes.
///
/// # Example
///
/// ```no_run
/// use flynnel::sched::{JobPlan, hybrid::join_hybrid};
/// use flynnel::Backend;
///
/// let plan = JobPlan::new(8, 1024).with_backend(Backend::Cuda { device_id: 0 });
/// let (cpu_sum, gpu_sum) = join_hybrid(
///     &plan,
///     || (0..512).sum::<u64>(),
///     || (512..1024).sum::<u64>(),
/// );
/// assert_eq!(cpu_sum + gpu_sum, (0..1024).sum::<u64>());
/// ```
pub fn join_hybrid<RA, RB, A, B>(plan: &JobPlan, cpu_work: A, gpu_work: B) -> (RA, RB)
where
    A: FnOnce() -> RA,
    B: FnOnce() -> RB + Send + 'static,
    RA: 'static,
    RB: Send + 'static,
{
    let backend = plan.pick_backend();
    // SPSC channel: 2-slot capacity (only one item ever sent),
    // 1 consumer (the calling thread). Built on FlynnelRing +
    // Parker via NotifyHub.
    let hub = NotifyHub::<std::thread::Result<RB>>::new(2, 1);
    let tx_for_backend = hub.sender();
    let rx = hub.register_consumer();
    backend.dispatch_one(Box::new(move || {
        let result =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(gpu_work));
        // Disregard send errors: the receiver may have already
        // dropped if the calling thread panicked itself.
        drop(tx_for_backend.send(result));
    }));
    // Run the CPU half on the calling thread. Wrap in
    // catch_unwind so we can propagate the GPU panic (if any)
    // after.
    let cpu_outcome =
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(cpu_work));
    let gpu_outcome = rx.recv().expect("GPU half disconnected before sending");
    match (cpu_outcome, gpu_outcome) {
        (Ok(a), Ok(b)) => (a, b),
        (Err(p), _) => std::panic::resume_unwind(p),
        (_, Err(p)) => std::panic::resume_unwind(p),
    }
}

/// Run one closure on the plan's backend via
/// [`crate::backend::DispatchBackend::dispatch_one`], blocking the
/// calling thread until the result arrives. Returns the result plus
/// the end-to-end wall time in nanoseconds (queueing + execution +
/// hand-back: the cost a placement decision actually pays).
fn run_on_backend<R, G>(plan: &JobPlan, work: G) -> (R, u64)
where
    R: Send + 'static,
    G: FnOnce() -> R + Send + 'static,
{
    let backend = plan.pick_backend();
    let hub = NotifyHub::<std::thread::Result<R>>::new(2, 1);
    let tx = hub.sender();
    let rx = hub.register_consumer();
    let t0 = std::time::Instant::now();
    backend.dispatch_one(Box::new(move || {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(work));
        drop(tx.send(result));
    }));
    let outcome = rx.recv().expect("backend half disconnected before sending");
    let wall_ns = t0.elapsed().as_nanos() as u64;
    match outcome {
        Ok(r) => (r, wall_ns),
        Err(p) => std::panic::resume_unwind(p),
    }
}

/// Learned-placement hybrid dispatch: run ONE of two equivalent
/// implementations of the same computation, choosing the side this
/// call site has measured to be faster at this batch size.
///
/// `cpu_impl` and `gpu_impl` MUST compute the same result; which one
/// executes is a performance decision, not a semantic one. The
/// model:
///
/// - **Cold size bucket** (either side unmeasured for
///   `log2(plan.batch_size)`): run BOTH concurrently in the
///   [`join_hybrid`] shape and time each. Racing IS the
///   calibration: the first call pays double work exactly once per
///   bucket instead of requiring an offline calibration pass. The
///   CPU side's result is returned (deterministic winner selection;
///   both results are equal by contract).
/// - **Warm bucket**: run only the side with the lower end-to-end
///   EWMA, timing it to keep the model fresh.
/// - **Every 32nd call per bucket**: re-race both sides so the
///   model tracks drift (thermal limits, contention, clock changes).
///
/// The EWMAs live on a per-call-site
/// [`crate::sched::call_site::CallSiteState`] (or the caller's own
/// via [`JobPlan::with_site`]), keyed by log2 size bucket, and
/// measure END-TO-END wall time: whatever transfer work the closure
/// performs is inside its own measurement, so no separate transfer
/// model or data-residency tracking exists or is needed. The
/// corresponding boundary: dispatches whose transfer cost depends
/// on buffer reuse ACROSS calls (DAG-style residency) are outside
/// this model's vocabulary; placement there stays a caller decision
/// via [`join_hybrid`] and [`JobPlan::backend_hint`].
///
/// Returns the result plus the [`Placement`] that was executed,
/// so callers can log the learned routing.
///
/// # Panics
///
/// Propagates a panic from whichever implementation ran (both, in
/// race mode, with the CPU side's panic taking precedence).
#[track_caller]
pub fn hybrid_auto<R, C, G>(plan: &JobPlan, cpu_impl: C, gpu_impl: G) -> (R, Placement)
where
    R: Send + 'static,
    C: FnOnce() -> R,
    G: FnOnce() -> R + Send + 'static,
{
    let plan_owned = plan.with_site_if_none(crate::sched::call_site::caller_site());
    let plan = &plan_owned;
    let site = plan
        .site
        .expect("with_site_if_none attached a site above")
        .get();
    let batch = plan.batch_size;
    match site.choose_placement(batch) {
        Placement::Cpu => {
            let t0 = std::time::Instant::now();
            let r = cpu_impl();
            site.record_placement(batch, Some(t0.elapsed().as_nanos() as u64), None);
            (r, Placement::Cpu)
        }
        Placement::Backend => {
            let (r, wall_ns) = run_on_backend(plan, gpu_impl);
            site.record_placement(batch, None, Some(wall_ns));
            (r, Placement::Backend)
        }
        Placement::Race => {
            let ((cpu_r, cpu_ns), (_gpu_r, gpu_ns)) = join_hybrid(
                plan,
                || {
                    let t0 = std::time::Instant::now();
                    let r = cpu_impl();
                    (r, t0.elapsed().as_nanos() as u64)
                },
                move || {
                    let t0 = std::time::Instant::now();
                    let r = gpu_impl();
                    (r, t0.elapsed().as_nanos() as u64)
                },
            );
            site.record_placement(batch, Some(cpu_ns), Some(gpu_ns));
            (cpu_r, Placement::Race)
        }
    }
}

/// Measured outcome of one [`hybrid_auto_split`] dispatch.
#[derive(Debug, Clone, Copy)]
pub struct SplitReport {
    /// Items the CPU side processed.
    pub cpu_items: usize,
    /// Items the backend side processed.
    pub backend_items: usize,
    /// CPU-side wall time in nanoseconds.
    pub cpu_ns: u64,
    /// Backend-side end-to-end wall time in nanoseconds.
    pub backend_ns: u64,
    /// The CPU share (parts-per-1000) this dispatch used.
    pub cpu_share_per_mille: u32,
}

/// Learned-ratio hybrid split over one divisible slice: the CPU
/// side and the backend side each process a contiguous sub-slice,
/// sized by the per-item throughputs this call site has measured
/// (even split until both sides have data). Both halves run
/// concurrently in the [`join_hybrid`] shape; the measured per-item
/// costs update the site's split model for the next call.
///
/// `cpu_fn` and `gpu_fn` MUST apply the same per-item
/// transformation; the split boundary is a performance decision.
///
/// # Panics
///
/// Propagates a panic from either side.
#[track_caller]
pub fn hybrid_auto_split<T, CF, GF>(
    plan: &JobPlan,
    items: &mut [T],
    cpu_fn: CF,
    gpu_fn: GF,
) -> SplitReport
where
    T: Send + 'static,
    CF: FnOnce(&mut [T]),
    GF: FnOnce(&mut [T]) + Send + 'static,
{
    let plan_owned = plan.with_site_if_none(crate::sched::call_site::caller_site());
    let plan = &plan_owned;
    let site = plan
        .site
        .expect("with_site_if_none attached a site above")
        .get();

    let n = items.len();
    let share = site.split_cpu_share_per_mille();
    let mid = ((n as u64 * share as u64) / 1000) as usize;
    let mid = mid.clamp(usize::from(n > 1), n.saturating_sub(usize::from(n > 1)));
    let (cpu_part, backend_part) = items.split_at_mut(mid);
    let cpu_items = cpu_part.len();
    let backend_items = backend_part.len();

    // Transport the backend sub-slice as (address, len): the boxed
    // closure dispatch_one consumes must be 'static, and a borrowed
    // slice is not. Same lifetime-laundering pattern as
    // `par_iter::reduce_chunks_flat`, with the same argument: this
    // function blocks on the backend result before returning, so
    // the borrow outlives every access made through the raw parts.
    let backend_addr = backend_part.as_mut_ptr() as usize;
    let backend_len = backend_part.len();

    let (cpu_ns_out, backend_ns_out) = join_hybrid(
        plan,
        || {
            let t0 = std::time::Instant::now();
            if !cpu_part.is_empty() {
                cpu_fn(cpu_part);
            }
            t0.elapsed().as_nanos() as u64
        },
        move || {
            let t0 = std::time::Instant::now();
            if backend_len > 0 {
                // SAFETY: address + length name the live
                // `backend_part` sub-slice split off above; the
                // caller frame blocks inside join_hybrid until this
                // closure completes, so the reconstruction never
                // outlives the borrow. The two sub-slices are
                // disjoint by construction of split_at_mut.
                let slice: &mut [T] = unsafe {
                    std::slice::from_raw_parts_mut(backend_addr as *mut T, backend_len)
                };
                gpu_fn(slice);
            }
            t0.elapsed().as_nanos() as u64
        },
    );

    site.record_split(cpu_items, cpu_ns_out, backend_items, backend_ns_out);
    SplitReport {
        cpu_items,
        backend_items,
        cpu_ns: cpu_ns_out,
        backend_ns: backend_ns_out,
        cpu_share_per_mille: share,
    }
}

/// Three-stage CPU-GPU-CPU pipeline. Each input flows through
/// `pre_cpu` → `gpu` → `post_cpu`, and the three stages run on
/// dedicated OS threads connected by depth-2 bounded channels so
/// stage\[N+1\] of an earlier pipeline position overlaps stage\[N\]
/// of a later one. After pipeline-fill the steady-state
/// throughput is `1 / max(t_pre_cpu, t_gpu, t_post_cpu)` per
/// input - the smaller stages hide entirely behind the largest.
///
/// The coupled MIMT shape (CPU propose -> GPU likelihood -> CPU
/// accept, and the like); [`join_hybrid`] handles one (cpu, gpu)
/// pair without pipelining across iterations.
///
/// Inter-stage channels are bounded at depth 2: pipeline-fill
/// costs one stage time, steady state is bounded by the slowest
/// stage. `_plan.backend_hint` is informational only; the `gpu`
/// closure decides which backend it invokes (move an
/// `Arc<YourBackend>` into it).
///
/// # Panics
///
/// A stage-thread panic propagates to the caller when that
/// stage's `join` is awaited; the other stages run to completion
/// first.
///
/// # Example
///
/// ```no_run
/// use flynnel::sched::{JobPlan, hybrid::hybrid_pipeline};
///
/// let plan = JobPlan::new(8, 1024);
/// // 10 iterations of a synthetic MCMC-shaped pipeline.
/// let results: Vec<f32> = hybrid_pipeline(
///     &plan,
///     0..10,
///     |seed: i32| -> Vec<f32> {
///         // CPU propose: branchy adaptive step
///         (0..256).map(|i| (seed as f32) + (i as f32) * 0.01).collect()
///     },
///     |proposal: Vec<f32>| -> f32 {
///         // GPU likelihood: launch a kernel; synthetic sum here
///         proposal.iter().sum()
///     },
///     |loglik: f32| -> f32 {
///         // CPU accept/reject
///         if loglik > 0.0 { loglik } else { 0.0 }
///     },
/// );
/// assert_eq!(results.len(), 10);
/// ```
pub fn hybrid_pipeline<I, F1, F2, F3, A, B, R>(
    _plan: &JobPlan,
    inputs: I,
    pre_cpu: F1,
    gpu: F2,
    post_cpu: F3,
) -> Vec<R>
where
    I: IntoIterator + Send + 'static,
    I::Item: Send + 'static,
    F1: FnMut(I::Item) -> A + Send + 'static,
    F2: FnMut(A) -> B + Send + 'static,
    F3: FnMut(B) -> R + Send + 'static,
    A: Send + 'static,
    B: Send + 'static,
    R: Send + 'static,
{
    // _plan is held for trait-stability and so callers compose
    // hybrid_pipeline the same way they compose join_hybrid /
    // cooperative_join_n / race_variants. The current
    // implementation runs the three stages on stock OS threads;
    // backend selection and NUMA pinning are decisions the gpu
    // closure makes from its captured backend reference.

    // Three SPSC hand-off channels backed by NotifyHub
    // (FlynnelRing + Parker). Each: 2-slot ring (ping-pong
    // depth) + 1 consumer (next stage / collector).
    let hub_ab = NotifyHub::<A>::new(2, 1);
    let hub_bc = NotifyHub::<B>::new(2, 1);
    let hub_out = NotifyHub::<R>::new(2, 1);

    let tx_ab = hub_ab.sender();
    let tx_bc = hub_bc.sender();
    let tx_out = hub_out.sender();

    let hub_ab_for_gpu = hub_ab.clone();
    let hub_bc_for_post = hub_bc.clone();

    // Clone an extra handle per stage so each stage holds its OWN
    // shutdown-on-drop guard for its output hub. RAII drop runs
    // on the unwind path, so a panic in any stage closure
    // signals end-of-stream to its downstream consumer without
    // requiring an explicit catch_unwind. Mirrors the panic-safety
    // crossbeam channels gave us via Sender's Drop impl.
    let hub_ab_shutdown = hub_ab.clone();
    let hub_bc_shutdown = hub_bc.clone();
    let hub_out_shutdown = hub_out.clone();

    // Stage 1: pre_cpu producer.
    let pre_handle = std::thread::Builder::new()
        .name("flynnel-hybrid-pre".into())
        .spawn({
            let mut pre_cpu = pre_cpu;
            move || {
                // Panic-safety: if pre_cpu panics, the guard's
                // Drop signals end-of-stream to the GPU stage.
                let _ab_shutdown = hub_ab_shutdown.shutdown_on_drop();
                for item in inputs.into_iter() {
                    let a = pre_cpu(item);
                    if !matches!(tx_ab.send(a), NotifySendResult::Ok) {
                        break;
                    }
                }
            }
        })
        .expect("spawn flynnel-hybrid-pre");

    // Stage 2: gpu.
    let gpu_handle = std::thread::Builder::new()
        .name("flynnel-hybrid-gpu".into())
        .spawn({
            let mut gpu = gpu;
            move || {
                let _bc_shutdown = hub_bc_shutdown.shutdown_on_drop();
                let rx_ab = hub_ab_for_gpu.register_consumer();
                while let Some(a) = rx_ab.recv() {
                    let b = gpu(a);
                    if !matches!(tx_bc.send(b), NotifySendResult::Ok) {
                        break;
                    }
                }
            }
        })
        .expect("spawn flynnel-hybrid-gpu");

    // Stage 3: post_cpu.
    let post_handle = std::thread::Builder::new()
        .name("flynnel-hybrid-post".into())
        .spawn({
            let mut post_cpu = post_cpu;
            move || {
                let _out_shutdown = hub_out_shutdown.shutdown_on_drop();
                let rx_bc = hub_bc_for_post.register_consumer();
                while let Some(b) = rx_bc.recv() {
                    let r = post_cpu(b);
                    if !matches!(tx_out.send(r), NotifySendResult::Ok) {
                        break;
                    }
                }
            }
        })
        .expect("spawn flynnel-hybrid-post");

    // Collect on the calling thread. NotifyReceiver::recv
    // returns None when the hub is shut down and the ring is
    // drained, which fires after stage 3 finishes producing.
    let rx_out = hub_out.register_consumer();
    let mut results = Vec::new();
    while let Some(r) = rx_out.recv() {
        results.push(r);
    }

    // Join the worker threads; propagate panics as-is.
    pre_handle
        .join()
        .unwrap_or_else(|p| std::panic::resume_unwind(p));
    gpu_handle
        .join()
        .unwrap_or_else(|p| std::panic::resume_unwind(p));
    post_handle
        .join()
        .unwrap_or_else(|p| std::panic::resume_unwind(p));

    results
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{
        Backend, BackendCapabilities, DispatchBackend, register_backend,
    };
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU32, Ordering};

    /// Stub backend whose `dispatch_one` runs the closure on a
    /// dedicated thread. Lets us observe MIMT timing without a
    /// real GPU.
    struct ThreadStub;
    impl DispatchBackend for ThreadStub {
        fn id(&self) -> Backend {
            Backend::Custom(424242)
        }
        fn capabilities(&self) -> BackendCapabilities {
            BackendCapabilities {
                simt_width: 64,
                max_threads_in_flight: 8192,
                launch_latency_ns: 5000,
                h2d_bw_bytes_per_sec: 10_000_000_000,
            }
        }
        fn dispatch_parallel_for(&self, _count: u32, _work: &(dyn Fn(u32) + Send + Sync)) {}
        fn dispatch_one(&self, work: Box<dyn FnOnce() + Send>) {
            std::thread::spawn(work);
        }
    }

    fn install_stub() {
        register_backend(Arc::new(ThreadStub));
    }

    #[test]
    fn hybrid_returns_both_results_in_caller_order() {
        install_stub();
        let plan = JobPlan::new(8, 64).with_backend(Backend::Custom(424242));
        let (a, b) = join_hybrid(&plan, || 7u32, || 11u32);
        assert_eq!(a, 7);
        assert_eq!(b, 11);
    }

    #[test]
    fn hybrid_runs_both_halves_to_completion() {
        install_stub();
        let plan = JobPlan::new(8, 64).with_backend(Backend::Custom(424242));
        let cpu_seen = Arc::new(AtomicU32::new(0));
        let gpu_seen = Arc::new(AtomicU32::new(0));
        let c = Arc::clone(&cpu_seen);
        let g = Arc::clone(&gpu_seen);
        let (cr, gr) = join_hybrid(
            &plan,
            move || {
                c.store(1, Ordering::SeqCst);
                100u32
            },
            move || {
                g.store(1, Ordering::SeqCst);
                200u32
            },
        );
        assert_eq!(cr, 100);
        assert_eq!(gr, 200);
        assert_eq!(cpu_seen.load(Ordering::SeqCst), 1);
        assert_eq!(gpu_seen.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn hybrid_with_no_hint_falls_back_to_cpu_backend() {
        // No backend_hint: routing picks the CPU backend. The
        // closures still run; correctness is the same as the
        // hinted case.
        let plan = JobPlan::new(8, 64);
        let (a, b) = join_hybrid(&plan, || 1u32 + 2, || 3u32 + 4);
        assert_eq!(a, 3);
        assert_eq!(b, 7);
    }

    #[test]
    fn hybrid_with_unregistered_hint_falls_back_to_cpu() {
        // Hint a backend id we never register. pick_backend
        // returns the CPU backend, and the hybrid call still
        // completes.
        let plan = JobPlan::new(8, 64).with_backend(Backend::Custom(987_654_321));
        let (a, b) = join_hybrid(&plan, || "left", || "right");
        assert_eq!(a, "left");
        assert_eq!(b, "right");
    }

    #[test]
    #[cfg(panic = "unwind")]
    fn hybrid_propagates_cpu_panic() {
        install_stub();
        let plan = JobPlan::new(8, 64).with_backend(Backend::Custom(424242));
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            join_hybrid::<u32, u32, _, _>(
                &plan,
                || panic!("cpu side"),
                || 42u32,
            );
        }));
        assert!(result.is_err());
    }

    // ---- hybrid_pipeline tests ---------------------------------

    #[test]
    fn pipeline_emits_results_in_order() {
        let plan = JobPlan::new(8, 64);
        let out = hybrid_pipeline(
            &plan,
            0..8i32,
            |seed| seed * 2,
            |a| a + 100,
            |b| b - 50,
        );
        // Expected: ((i*2) + 100) - 50 = i*2 + 50  for i in 0..8
        let expected: Vec<i32> = (0..8).map(|i| i * 2 + 50).collect();
        assert_eq!(out, expected);
    }

    #[test]
    fn pipeline_handles_empty_input() {
        let plan = JobPlan::new(8, 64);
        let out: Vec<u32> = hybrid_pipeline(
            &plan,
            std::iter::empty::<u32>(),
            |x| x,
            |x| x,
            |x| x,
        );
        assert!(out.is_empty());
    }

    #[test]
    fn pipeline_actually_overlaps_stages() {
        use std::time::{Duration, Instant};
        const N: usize = 8;
        const STAGE_MS: u64 = 30;
        let plan = JobPlan::new(8, 64);
        let t0 = Instant::now();
        let out = hybrid_pipeline(
            &plan,
            (0..N).collect::<Vec<usize>>(),
            |x| { std::thread::sleep(Duration::from_millis(STAGE_MS)); x },
            |x| { std::thread::sleep(Duration::from_millis(STAGE_MS)); x },
            |x| { std::thread::sleep(Duration::from_millis(STAGE_MS)); x },
        );
        let elapsed = t0.elapsed();
        assert_eq!(out.len(), N);
        // Sequential would be 3 stages * N items * STAGE_MS = 720 ms.
        // Pipelined fills in 3 stages then emits one per STAGE_MS:
        // 3*STAGE_MS + (N-1)*STAGE_MS = (N+2)*STAGE_MS = 300 ms.
        // Allow generous slack (2x) for OS scheduler jitter on Windows.
        let pipelined_ms = (N as u64 + 2) * STAGE_MS;
        assert!(
            elapsed.as_millis() < (pipelined_ms * 2) as u128,
            "pipeline did not overlap: {} ms exceeds 2x ideal {} ms",
            elapsed.as_millis(),
            pipelined_ms
        );
        // And it must beat fully-sequential by a clear margin.
        let sequential_ms = 3 * N as u64 * STAGE_MS;
        assert!(
            elapsed.as_millis() < sequential_ms as u128,
            "pipeline took {} ms, not faster than sequential {} ms",
            elapsed.as_millis(),
            sequential_ms
        );
    }

    #[test]
    #[cfg(panic = "unwind")]
    fn pipeline_propagates_stage_panic() {
        let plan = JobPlan::new(8, 64);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            hybrid_pipeline::<_, _, _, _, i32, i32, i32>(
                &plan,
                0..4i32,
                |x| x + 1,
                |x| {
                    if x == 2 {
                        panic!("gpu stage boom");
                    }
                    x + 1
                },
                |x| x + 1,
            )
        }));
        assert!(result.is_err());
    }
}
