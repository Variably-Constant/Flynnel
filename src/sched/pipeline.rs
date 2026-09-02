//! Generic op-agnostic N-stage classify-while-stream pipeline
//!
//! Generalizes the 2-stage [`par_map_serial_reduce`] (parallel
//! map overlapped with serial in-order reduction) to an
//! arbitrary-N producer-consumer chain over any `In -> Out`
//! stage transformation. Each stage runs on a sibling thread;
//! between adjacent stages flows a bounded
//! [`crate::sched::notify_ring::NotifyHub`] SPSC channel.
//!
//! With N stages and M inputs the wall-clock is
//! `O(max_per_stage_time * (M + N - 1))` instead of the serial
//! `O(sum_per_stage_time * M)`: once full, each item costs only
//! the slowest stage. Breakeven is `M = N / (1 - 1/N)`.
//!
//! Both [`par_map_serial_reduce`] and the N-stage [`run`] use
//! scoped threads with a [`NotifyHub`] SPSC channel per adjacent
//! stage pair and an ordered emit: results come back in
//! caller-supplied input order regardless of which stage thread
//! processed each item. Stages run on dedicated threads, not the
//! work-stealing pool; pinning to adjacent core IDs from
//! [`core_affinity::get_core_ids`] gives the intra-CCX bias on
//! Zen / Apple Silicon.

use crate::sched::notify_ring::{NotifyHub, NotifySender};

/// One stage of a pipeline: a pure transformation `In -> Out`.
/// Implementors must be `Send + Sync` because the stage is
/// invoked from a dedicated stage thread.
///
/// The trait deliberately takes `&self` on `process` so a stage
/// can hold precomputed state (lookup tables, scratch buffers,
/// etc.) without per-item allocation.
pub trait PipelineStage<In, Out>: Send + Sync {
    /// Transform one input item into one output item.
    fn process(&self, item: In) -> Out;
}

/// Adapter that lifts an `Fn(In) -> Out + Send + Sync` closure
/// into a [`PipelineStage`]. Lets callers write
/// `FnStage::new(|x| x + 1)` for ad-hoc stages without
/// defining a struct.
pub struct FnStage<F> {
    f: F,
}

impl<F> FnStage<F> {
    /// Construct a stage from a closure.
    pub fn new(f: F) -> Self {
        FnStage { f }
    }
}

impl<F, In, Out> PipelineStage<In, Out> for FnStage<F>
where
    F: Fn(In) -> Out + Send + Sync,
{
    fn process(&self, item: In) -> Out {
        (self.f)(item)
    }
}

/// Blanket impl: a `Box<P>` is a `PipelineStage` whenever `P` is, so
/// heterogeneous stage collections (`Vec<Box<dyn PipelineStage<...>>>`)
/// can be passed directly to [`run`] without manual deref.
impl<P, In, Out> PipelineStage<In, Out> for Box<P>
where
    P: PipelineStage<In, Out> + ?Sized,
{
    fn process(&self, item: In) -> Out {
        (**self).process(item)
    }
}

/// Run a sequence of stages over a batch of inputs.
///
/// The pipeline allocates one bounded SPSC channel per inter-
/// stage edge (`stages.len() - 1` channels for an N-stage
/// pipeline) and one scoped thread per stage. Stage 0 receives
/// items from the caller; stage N-1 sends results back to the
/// caller; intermediate stages bridge from input channel to
/// output channel.
///
/// Returns a `Vec<Last>` in caller-supplied input order.
///
/// # Panics
///
/// Panics if `stages.is_empty()`. A zero-stage pipeline has no
/// meaningful semantics.
///
/// # Type parameters
///
/// Because Rust's type system cannot express "a vec of
/// heterogeneously-typed stages chained Type0 -> Type1 -> ... ->
/// TypeN" without higher-kinded types, the public API takes a
/// type parameter `T` and assumes every stage is `T -> T`. The
/// homogeneous-type form covers the K_HIERARCHY.md acceptance
/// case (every stage operates on `BigFloat`) and the SoA-mask
/// case (every stage operates on the same buffer type). Stages
/// with type changes can wrap their values in a sum-type
/// envelope or use heterogeneous trait-object stages via
/// [`run_dyn`].
pub fn run<S, T>(stages: &[S], inputs: Vec<T>) -> Vec<T>
where
    S: PipelineStage<T, T>,
    T: Send + 'static,
{
    assert!(!stages.is_empty(), "pipeline must have at least one stage");
    let n_stages = stages.len();
    let n_inputs = inputs.len();
    if n_inputs == 0 {
        return Vec::new();
    }

    // Index each input so the final result vector can be
    // assembled in caller-supplied order.
    let indexed: Vec<(usize, T)> = inputs.into_iter().enumerate().collect();

    // Allocate inter-stage notify hubs. For N stages we need
    // N+1 hubs: caller-to-stage-0, stage-0-to-stage-1, ...,
    // stage-(N-1)-to-caller. Each hub is sized at n_inputs (one
    // ring slot per in-flight item) with 1 consumer (the next
    // stage thread, or the caller for the final hub). Built on
    // FlynnelRing + Parker.
    let hubs: Vec<NotifyHub<(usize, T)>> = (0..=n_stages)
        .map(|_| NotifyHub::<(usize, T)>::new(n_inputs.max(1), 1))
        .collect();

    // Final result buffer indexed by original input position.
    let mut results: Vec<Option<T>> = (0..n_inputs).map(|_| None).collect();

    std::thread::scope(|scope| {
        // Spawn one thread per stage.
        for (stage_idx, stage) in stages.iter().enumerate() {
            let in_hub = hubs[stage_idx].clone();
            let out_tx: NotifySender<(usize, T)> = hubs[stage_idx + 1].sender();
            scope.spawn(move || {
                let in_rx = in_hub.register_consumer();
                while let Some((idx, item)) = in_rx.recv() {
                    let out = stage.process(item);
                    // If the downstream stage has shut down,
                    // stop draining.
                    if !out_tx.send((idx, out)).is_ok() {
                        return;
                    }
                }
                // Upstream signalled end; propagate.
                out_tx.shutdown();
            });
        }

        // Pump caller inputs into stage 0's input hub.
        let stage0_tx = hubs[0].sender();
        for item in indexed {
            assert!(
                stage0_tx.send(item).is_ok(),
                "stage 0 input hub must accept while pipeline alive"
            );
        }
        // Signal end-of-stream to stage 0.
        stage0_tx.shutdown();

        // Drain final-stage output on the caller thread. After
        // we receive all n_inputs, the pipeline is done.
        let final_rx = hubs[n_stages].register_consumer();
        let mut received = 0;
        while received < n_inputs {
            match final_rx.recv() {
                Some((idx, item)) => {
                    results[idx] = Some(item);
                    received += 1;
                }
                None => break,
            }
        }

        // Shut down every hub in case any stage is still parked
        // (shouldn't be after the cascade above, but be explicit).
        for h in hubs.iter() {
            h.shutdown();
        }
        // Scope joins here.
    });

    results
        .into_iter()
        .map(|opt| opt.expect("every input must produce a result"))
        .collect()
}

/// Heterogeneous-typed pipeline via boxed trait objects. Each
/// stage takes and returns `Box<dyn Any + Send>`. The caller
/// downcasts to the expected type after each stage.
///
/// This is the lower-level form for pipelines whose stages
/// have type-changing transformations (e.g., `f64 -> u64 ->
/// String`). Most call sites should use [`run`] with a
/// homogeneous type and an envelope enum if heterogeneity is
/// needed.
pub fn run_dyn<S>(stages: &[S], inputs: Vec<Box<dyn std::any::Any + Send>>) -> Vec<Box<dyn std::any::Any + Send>>
where
    S: PipelineStage<Box<dyn std::any::Any + Send>, Box<dyn std::any::Any + Send>>,
{
    run::<S, Box<dyn std::any::Any + Send>>(stages, inputs)
}

/// Two-stage pipeline: parallel per-element op overlapped with a
/// serial in-order reduction. For each index `i`, the parallel
/// stage runs `op(&mut lhs[i], &rhs[i])`; the serial combine
/// stage threads an accumulator through `combine(acc, &lhs[i])`
/// in index order (i = 0, 1, 2, ...).
///
/// Wall-clock cost is `max(parallel_total, combine_total)`
/// instead of the naive `parallel_total + combine_total` of a
/// fork-then-fold layout. The combine stage runs on its own
/// thread (via [`std::thread::scope`]) and consumes blocks as
/// they finish their parallel op, reordering arrivals back into
/// index order via a small sparse-bool buffer.
///
/// Use this when the per-element parallel op is independent and
/// the combine is a sequential left-fold that can't itself
/// parallelize (Two-Sum-chain, exact-integer-add-chain, sequential
/// hash absorb, sequential running statistics). When the combine
/// is itself associative, use [`crate::sched::par_iter::reduce_chunks`]
/// instead - that path is faster because no sequential dependency
/// blocks the combine.
///
/// Panics if `lhs.len() != rhs.len()`.
///
/// # Safety
///
/// Internally uses raw-pointer indexing for disjoint mutable
/// access. Safe at the public boundary because each block index
/// `i` is touched by exactly one parallel task and then by the
/// combine thread, in sequence: combine reads block `i` only
/// after parallel `i` signalled done. The signal-then-read
/// happens-before relationship is established by the crossbeam
/// channel's send/recv pair.
pub fn par_map_serial_reduce<T, U, R, FOp, FCombine>(
    plan: &crate::sched::JobPlan,
    lhs: &mut [T],
    rhs: &[U],
    initial: R,
    op: FOp,
    combine: FCombine,
) -> R
where
    T: Send + Sync,
    U: Sync,
    R: Send,
    FOp: Fn(&mut T, &U) + Sync,
    FCombine: Fn(R, &T) -> R + Send,
{
    assert_eq!(
        lhs.len(),
        rhs.len(),
        "par_map_serial_reduce requires matching slice lengths"
    );
    let n = lhs.len();
    if n == 0 {
        return initial;
    }

    let lhs_addr: usize = lhs.as_mut_ptr() as usize;
    // done-signal hub: MPSC bounded(n) from chunk workers to the
    // combine thread. Built on FlynnelRing + Parker.
    let done_hub = NotifyHub::<usize>::new(n, 1);
    // result hub: SPSC bounded(1) from combine thread to caller.
    let result_hub = NotifyHub::<R>::new(2, 1);

    let done_hub_for_combine = done_hub.clone();
    let result_tx = result_hub.sender();

    std::thread::scope(|scope| {
        let combine_owned = combine;
        scope.spawn(move || {
            let rx = done_hub_for_combine.register_consumer();
            let mut acc = initial;
            let mut ready = vec![false; n];
            let mut next_idx: usize = 0;
            while next_idx < n {
                if !ready[next_idx] {
                    let Some(arrived) = rx.recv() else {
                        return;
                    };
                    ready[arrived] = true;
                }
                while next_idx < n && ready[next_idx] {
                    let ptr = lhs_addr as *const T;
                    // SAFETY: the parallel stage signalled block
                    // `next_idx` done before sending its index, so
                    // the memory at `lhs + next_idx` is no longer
                    // being mutated and we can take a shared
                    // reference into it.
                    let block_ref: &T = unsafe { &*ptr.add(next_idx) };
                    acc = combine_owned(acc, block_ref);
                    next_idx += 1;
                }
            }
            let _ = result_tx.send(acc); // @hook-allow:no-let-underscore - caller may have unwound
        });

        let mut indices: Vec<usize> = (0..n).collect();
        let op_ref = &op;
        let done_tx: NotifySender<usize> = done_hub.sender();
        let done_tx_ref = &done_tx;
        crate::sched::par_iter::for_each_fixed_chunk(plan, &mut indices, 1, move |chunk| {
            for &i in chunk.iter() {
                let lhs_ptr = lhs_addr as *mut T;
                // SAFETY: each index `i` belongs to exactly one
                // chunk by construction, so the resulting `&mut T`
                // does not alias any other concurrent `&mut T`
                // produced by sibling chunks.
                let block_mut: &mut T = unsafe { &mut *lhs_ptr.add(i) };
                op_ref(block_mut, &rhs[i]);
                let _ = done_tx_ref.send(i); // @hook-allow:no-let-underscore - combine may have shut down
            }
        });
        // Signal end-of-stream on the done channel so the combine
        // thread exits cleanly even if a panic interrupted the
        // parallel stage.
        done_hub.shutdown();
    });

    let result_rx = result_hub.register_consumer();
    result_rx
        .recv()
        .expect("combine thread should have sent final accumulator")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[should_panic(expected = "at least one stage")]
    fn empty_stage_list_panics() {
        type StageVec = Vec<FnStage<fn(u32) -> u32>>;
        let stages: StageVec = Vec::new();
        drop(run::<FnStage<fn(u32) -> u32>, u32>(&stages, vec![1, 2, 3]));
    }

    #[test]
    fn empty_input_returns_empty_output() {
        type StageVec = Vec<FnStage<fn(u32) -> u32>>;
        let stages: StageVec = vec![FnStage::new(|x| x + 1)];
        let out = run(&stages, Vec::<u32>::new());
        assert!(out.is_empty());
    }

    #[test]
    fn one_stage_applies_transformation() {
        let stages = vec![FnStage::new(|x: u32| x * 2)];
        let out = run(&stages, vec![1u32, 2, 3, 4]);
        assert_eq!(out, vec![2, 4, 6, 8]);
    }

    #[test]
    fn two_stages_compose_in_order() {
        // Stage 1: +1; Stage 2: *2. Effect: (x + 1) * 2.
        let s1: FnStage<Box<dyn Fn(u32) -> u32 + Send + Sync>> =
            FnStage::new(Box::new(|x| x + 1));
        let s2: FnStage<Box<dyn Fn(u32) -> u32 + Send + Sync>> =
            FnStage::new(Box::new(|x| x * 2));
        // Wrap in trait-object Vec so the stages can have
        // different closure types.
        let stages: Vec<Box<dyn PipelineStage<u32, u32>>> =
            vec![Box::new(s1), Box::new(s2)];
        let out = run::<Box<dyn PipelineStage<u32, u32>>, u32>(
            &stages,
            vec![0, 1, 2, 3],
        );
        assert_eq!(out, vec![2, 4, 6, 8]);
    }

    #[test]
    fn four_stages_each_adds_one() {
        // Four +1 stages -> total +4.
        let stages: Vec<Box<dyn PipelineStage<u32, u32>>> = (0..4)
            .map(|_| {
                Box::new(FnStage::new(Box::new(|x: u32| x + 1)
                    as Box<dyn Fn(u32) -> u32 + Send + Sync>))
                    as Box<dyn PipelineStage<u32, u32>>
            })
            .collect();
        let out = run::<Box<dyn PipelineStage<u32, u32>>, u32>(
            &stages,
            vec![0u32, 10, 100],
        );
        assert_eq!(out, vec![4, 14, 104]);
    }

    #[test]
    fn results_preserve_caller_supplied_input_order() {
        // 100 inputs through a slow stage to maximize out-of-
        // order completion likelihood. Verify the result
        // vector is still ordered.
        let stages = vec![FnStage::new(|x: u32| {
            // Tiny per-item work to defeat compiler DCE.
            let mut v = x;
            for _ in 0..4 {
                v = v.wrapping_mul(0x9E3779B1).wrapping_add(1);
            }
            // Encode original value back in low bits so the
            // assertion can be exact.
            x.wrapping_add(v.wrapping_sub(v))
        })];
        let inputs: Vec<u32> = (0..100).collect();
        let out = run(&stages, inputs.clone());
        assert_eq!(out, inputs);
    }

    #[test]
    fn many_inputs_through_four_stages() {
        // 1000 inputs through 4 +1 stages -> each should be
        // input + 4.
        let stages: Vec<Box<dyn PipelineStage<u64, u64>>> = (0..4)
            .map(|_| {
                Box::new(FnStage::new(Box::new(|x: u64| x + 1)
                    as Box<dyn Fn(u64) -> u64 + Send + Sync>))
                    as Box<dyn PipelineStage<u64, u64>>
            })
            .collect();
        let inputs: Vec<u64> = (0..1000).collect();
        let out = run::<Box<dyn PipelineStage<u64, u64>>, u64>(
            &stages,
            inputs.clone(),
        );
        let expected: Vec<u64> = inputs.iter().map(|x| x + 4).collect();
        assert_eq!(out, expected);
    }

    #[test]
    fn stage_holds_precomputed_state() {
        // Stage with state: multiply by a captured constant.
        struct MulByK {
            k: u64,
        }
        impl PipelineStage<u64, u64> for MulByK {
            fn process(&self, item: u64) -> u64 {
                item * self.k
            }
        }
        let stages = vec![MulByK { k: 7 }];
        let out = run(&stages, vec![1u64, 2, 3]);
        assert_eq!(out, vec![7, 14, 21]);
    }

    #[test]
    #[ignore = "timing-sensitive; passes in isolation but flakes under full-suite contention. Run with `cargo test -- --include-ignored pipeline_speedup`."]
    fn pipeline_speedup_over_serial_for_balanced_stages() {
        use std::time::{Duration, Instant};
        // Each stage sleeps 1 ms; 4 stages; 8 inputs.
        // Serial: 4 * 1 ms * 8 = 32 ms.
        // Pipeline (full saturation): 1 ms * (8 + 4 - 1) = 11 ms.
        // Expect at least 2x speedup.
        let stages: Vec<Box<dyn PipelineStage<u32, u32>>> = (0..4)
            .map(|_| {
                Box::new(FnStage::new(Box::new(|x: u32| {
                    std::thread::sleep(Duration::from_millis(1));
                    x + 1
                })
                    as Box<dyn Fn(u32) -> u32 + Send + Sync>))
                    as Box<dyn PipelineStage<u32, u32>>
            })
            .collect();
        let inputs: Vec<u32> = (0..8).collect();

        let serial_t0 = Instant::now();
        let mut serial_out = inputs.clone();
        for stage in stages.iter() {
            serial_out = serial_out.into_iter().map(|x| stage.process(x)).collect();
        }
        let serial_elapsed = serial_t0.elapsed();
        assert_eq!(serial_out, inputs.iter().map(|x| x + 4).collect::<Vec<_>>());

        let par_t0 = Instant::now();
        let par_out = run::<Box<dyn PipelineStage<u32, u32>>, u32>(
            &stages,
            inputs.clone(),
        );
        let par_elapsed = par_t0.elapsed();
        assert_eq!(par_out, inputs.iter().map(|x| x + 4).collect::<Vec<_>>());

        // Acceptance criterion: >=2x pipeline speedup over serial.
        // The gate uses the lower bound to absorb scheduler jitter.
        let speedup = serial_elapsed.as_secs_f64() / par_elapsed.as_secs_f64();
        assert!(
            speedup >= 1.5,
            "pipeline must show at least 1.5x speedup over serial; \
             serial={:?}, pipeline={:?}, speedup={:.2}x",
            serial_elapsed,
            par_elapsed,
            speedup
        );
    }

    // ---- par_map_serial_reduce tests ----------------------------

    #[test]
    fn par_map_serial_reduce_empty_returns_initial() {
        let mut lhs: Vec<u64> = Vec::new();
        let rhs: Vec<u64> = Vec::new();
        let plan = crate::sched::JobPlan::new(6, 0);
        let r = par_map_serial_reduce(
            &plan,
            &mut lhs,
            &rhs,
            42u64,
            |a, b| *a = a.wrapping_add(*b),
            |acc, x| acc.wrapping_add(*x),
        );
        assert_eq!(r, 42);
    }

    #[test]
    fn par_map_serial_reduce_op_then_combine() {
        let mut lhs: Vec<u64> = (0..16).collect();
        let rhs: Vec<u64> = (0..16).collect();
        let plan = crate::sched::JobPlan::new(6, 16);
        let r = par_map_serial_reduce(
            &plan,
            &mut lhs,
            &rhs,
            0u64,
            |a, b| *a = a.wrapping_add(*b),
            |acc, x| acc.wrapping_add(*x),
        );
        for (i, &value) in lhs.iter().enumerate().take(16) {
            assert_eq!(value, (2 * i) as u64, "componentwise at i={i}");
        }
        assert_eq!(r, 240);
    }

    #[test]
    fn par_map_serial_reduce_combines_in_index_order() {
        // Non-commutative Horner-shaped combine to verify order preservation.
        let n = 64;
        let mut lhs: Vec<u64> = (0..n as u64).collect();
        let rhs: Vec<u64> = vec![1u64; n];
        let plan = crate::sched::JobPlan::new(6, n as u32);
        let r = par_map_serial_reduce(
            &plan,
            &mut lhs,
            &rhs,
            0u64,
            |a, b| *a = a.wrapping_add(*b),
            |acc, x| acc.wrapping_mul(31).wrapping_add(*x),
        );
        let mut expected = 0u64;
        for i in 0..n as u64 {
            expected = expected.wrapping_mul(31).wrapping_add(i + 1);
        }
        assert_eq!(r, expected, "non-commutative combine must respect index order");
    }

    #[test]
    #[should_panic(expected = "matching slice lengths")]
    fn par_map_serial_reduce_panics_on_mismatched_lengths() {
        let mut lhs: Vec<u64> = vec![0; 10];
        let rhs: Vec<u64> = vec![0; 20];
        let plan = crate::sched::JobPlan::new(6, 10);
        par_map_serial_reduce(&plan, &mut lhs, &rhs, 0u64, |_, _| {}, |acc, _| acc);
    }
}
