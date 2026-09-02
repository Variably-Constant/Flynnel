//! Unified user-facing dispatch API across the Flynn taxonomy.
//!
//! The user describes their workload declaratively and the
//! framework picks the right Flynn-axis entry point - SISD
//! (inline) vs MIMD (work-steal) vs SIMC (cooperative fan-out) vs
//! MISD (variant race) - based on a single
//! [`crate::sched::workload_shape::WorkloadShape`] hint.
//!
//! ## The Flynn-axis dispatch table
//!
//! | WorkloadShape             | Flynn axis | Underlying primitive                                   | Dispatch method                     |
//! |---------------------------|------------|--------------------------------------------------------|-------------------------------------|
//! | `Streaming`               | SISD       | inline execution (`op()`)                              | [`AdaptiveDispatcher::execute_streaming`]      |
//! | `ProducerFast { burst }`  | SIMC       | [`crate::sched::cooperative_join_n_flat`]              | [`AdaptiveDispatcher::execute_cooperative`]    |
//! | `WorkSteal { n, batch }`  | MIMD       | [`crate::sched::par_iter::for_each_chunk`]             | [`AdaptiveDispatcher::execute_for_each`]       |
//! | `Cooperative { n_cores }` | SIMC/MIMC  | [`crate::sched::cooperative::cooperative_join_n_flat_mailbox`] | [`AdaptiveDispatcher::execute_cooperative_mailbox`] |
//!
//! MISD variant racing has no `execute_*` shortcut here: call
//! [`crate::sched::race::race_variants`] directly (three typed
//! closures, each polling a
//! [`crate::sched::race::CancelToken`]).
//!
//! The K_gating axis (PerSlot vs CounterOnly) swaps at runtime via
//! [`AdaptiveDispatcher::migrate_k_gating`]; measured on Zen+ R7
//! 2700 the per-push AtomicU32 Acquire load adds 0.02 ns over
//! direct dispatch (noise floor).
//!
//! ```ignore
//! use flynnel::sched::dispatch::AdaptiveDispatcher;
//! use flynnel::sched::workload_shape::WorkloadShape;
//!
//! let results = AdaptiveDispatcher::new()
//!     .with_shape(WorkloadShape::ProducerFast { burst: 64 })
//!     .execute_cooperative(closures);
//! ```
//!
//! No deque, K-axis, or Flynn-axis name appears in user code; the
//! user describes the workload and the framework dispatches.

#![allow(clippy::missing_errors_doc)]

use crate::backend::{Backend, BackendRef};
use crate::dispatch_profile::DispatchProfile;
use crate::foundation::Variant;
use crate::sched::JobPlan;
use crate::sched::adaptive_backend::{
    active_backend_id, migrate_backend, resolve_active_backend,
};
use crate::sched::adaptive_profile::{
    WorkloadClass, active_dispatch_profile, migrate_dispatch_profile,
    migrate_workload_class,
};
use crate::sched::k_gating::KGating;
use crate::sched::workload_shape::WorkloadShape;

/// User-facing adaptive dispatcher. Carries the WorkloadShape hint
/// and routes the dispatch through the matching Flynn-axis
/// primitive.
pub struct AdaptiveDispatcher {
    shape: WorkloadShape,
    variant: Variant,
    k_outer: u8,
    use_smt: bool,
    /// Optional per-call WorkloadClass override; takes precedence
    /// over the global active DispatchProfile when set.
    explicit_class: Option<WorkloadClass>,
}

impl Default for AdaptiveDispatcher {
    fn default() -> Self {
        Self::new()
    }
}

impl AdaptiveDispatcher {
    /// Fresh dispatcher with default settings: `WorkSteal` shape
    /// (MIMD baseline), faithful variant, k_outer=4, no SMT,
    /// no explicit workload-class (consults global active
    /// DispatchProfile).
    pub fn new() -> Self {
        Self {
            shape: WorkloadShape::WorkSteal {
                n_consumers: num_cpus(),
                batch_size: 256,
            },
            variant: Variant::Faithful,
            k_outer: 4,
            use_smt: false,
            explicit_class: None,
        }
    }

    /// Builder: install a declarative workload-shape hint. The
    /// shape selects which Flynn-axis primitive the framework
    /// routes to AND the low-level K-axis knobs (K_gating,
    /// mailbox routing, oversubscription) via
    /// [`WorkloadShape::hints`].
    #[inline]
    pub fn with_shape(mut self, shape: WorkloadShape) -> Self {
        self.shape = shape;
        self
    }

    /// Builder: select the [`Variant`] (correctness tier).
    #[inline]
    pub fn with_variant(mut self, variant: Variant) -> Self {
        self.variant = variant;
        self
    }

    /// Builder: set the [`crate::sched::JobPlan::k_outer`] hint
    /// (log2 of operand size in limbs).
    #[inline]
    pub fn with_k_outer(mut self, k_outer: u8) -> Self {
        self.k_outer = k_outer;
        self
    }

    /// Builder: opt in to SMT-sibling participation. The
    /// dispatched plan's `use_smt=true` raises the arena's SMT
    /// request counter; siblings join the work-stealing loop for
    /// the duration of the dispatch.
    #[inline]
    pub fn with_smt(mut self) -> Self {
        self.use_smt = true;
        self
    }

    /// Build the underlying `JobPlan` from the dispatcher's
    /// settings. Maps the shape hint through
    /// [`JobPlan::with_workload_shape`] and the workload-class
    /// hint (explicit or global) through [`JobPlan::set_profile`]
    /// derived knobs.
    fn build_plan(&self, batch_size: u32) -> JobPlan {
        // Pick the effective DispatchProfile: explicit per-call
        // class wins; otherwise read the global active tag.
        let profile = match self.explicit_class {
            Some(c) => c.to_dispatch_profile(),
            None => active_dispatch_profile(),
        };
        // Build a profile-derived base plan so use_smt +
        // oversubscription_log2 + estimated_per_item_ns are set
        // according to the active class. The explicit flag mirrors
        // whether the CALLER supplied a workload class: a
        // with_workload_class dispatcher is pinned; one consulting
        // the global default stays overridable by per-site learning.
        let mut plan = JobPlan::set_profile_with(
            self.k_outer,
            batch_size,
            profile,
            self.explicit_class.is_some(),
        )
        .with_variant(self.variant)
        .with_workload_shape(self.shape);
        if self.use_smt {
            plan = plan.with_smt();
        }
        plan
    }

    /// SIMC dispatch: run a list of closures cooperatively across
    /// workers (fan-out + per-closure result). Maps to
    /// [`crate::sched::cooperative_join_n_flat`] which uses
    /// `push_burst` + flush to unlock the K_inner=3 amortization
    /// (3 jobs per cache-line transfer).
    ///
    /// Returns results in caller order regardless of execution
    /// thread.
    pub fn execute_cooperative<R>(
        self,
        closures: Vec<Box<dyn FnOnce() -> R + Send>>,
    ) -> Vec<R>
    where
        R: Send + 'static,
    {
        let plan = self.build_plan(closures.len() as u32);
        crate::sched::cooperative_join_n_flat(&plan, closures)
    }

    /// SIMC/MIMC dispatch with mailbox routing: variant of
    /// `execute_cooperative` that hands each closure directly to
    /// a specific worker's mailbox (URD-style owner-directed
    /// distribution). Use when N >= n_workers AND closures are
    /// uniform-cost. Internally gated to demote to deque mode
    /// when N < n_workers.
    pub fn execute_cooperative_mailbox<R>(
        self,
        closures: Vec<Box<dyn FnOnce() -> R + Send>>,
    ) -> Vec<R>
    where
        R: Send + 'static,
    {
        let plan = self
            .build_plan(closures.len() as u32)
            .with_mailbox_routing(true);
        crate::sched::cooperative::cooperative_join_n_flat_mailbox(&plan, closures)
    }

    /// MIMD dispatch: run an in-place op over `items` via
    /// recursive work-stealing bisection through
    /// [`crate::sched::par_iter::for_each_chunk`]. Callers who
    /// need finer control over the leaf floor (heavy per-element
    /// work like matmul, spmv, LU update) should build a `JobPlan`
    /// with an explicit `estimated_per_item_ns` hint and call
    /// [`crate::sched::par_iter::for_each_chunk_indexed_min_leaf`]
    /// directly.
    #[track_caller]
    pub fn execute_for_each<T, F>(self, items: &mut [T], op: F)
    where
        T: Send,
        F: Fn(&mut [T]) + Sync + Send,
    {
        let plan = self.build_plan(items.len() as u32);
        crate::sched::par_iter::for_each_chunk(&plan, items, op);
    }

    /// SISD dispatch: run a single closure inline on the caller's
    /// thread. No scheduler involvement; useful as the
    /// degenerate-case fall-through.
    pub fn execute_streaming<R, F>(self, op: F) -> R
    where
        F: FnOnce() -> R + Send,
        R: Send,
    {
        op()
    }

    /// Backend-adaptive indexed parallel-for: invokes `work(i)` for
    /// `i` in `0..count`, routed through the global active
    /// backend's `DispatchBackend::dispatch_parallel_for` impl.
    ///
    /// When the active backend is CPU, this routes through the
    /// work-stealing arena (same as the per-slice
    /// [`Self::execute_for_each`] path). When active is CUDA /
    /// ROCm / Metal / TPU, this dispatches via the GPU/TPU
    /// runtime if a registered backend is available; falls back
    /// to CPU gracefully when the active backend is not
    /// registered.
    ///
    /// Returns `(fell_back_to_cpu, ())` so the caller can
    /// observe the routing decision for telemetry.
    pub fn execute_indexed<F>(self, count: u32, work: F) -> bool
    where
        F: Fn(u32) + Send + Sync,
    {
        let (backend, fell_back) = resolve_active_backend();
        backend.dispatch_parallel_for(count, &work);
        fell_back
    }

    /// Migrate the active K_gating across all workers in the
    /// global LocalArena. Each worker's per-tier AdaptiveWorker
    /// flips its active tag with a single Release-store; new
    /// pushes route to the new backing immediately.
    ///
    /// Use this when the application observes a workload shift
    /// (e.g., the burst-vs-single profile crosses a threshold)
    /// and wants to switch the per-op deque primitive without
    /// respawning the worker pool. Migration cost: one
    /// Release-store per worker per tier (~30 atomic stores on
    /// a 16-worker arena with 4 tiers); per-op cost on new
    /// pushes is unchanged from the existing AtomicU32-tag
    /// dispatch.
    pub fn migrate_k_gating(&self, gating: KGating) {
        let arena = crate::sched::arena::global_local_arena();
        arena.migrate_all_workers_k_gating(gating);
    }

    /// Migrate the active global [`DispatchProfile`]. One AtomicU8
    /// Release-store; subsequent dispatches that consult the
    /// active profile (via [`active_dispatch_profile`]) see the
    /// new value. Per-op cost on the deque hot path is unchanged
    /// because the profile is only read at plan-construction time.
    #[inline]
    pub fn migrate_dispatch_profile(&self, profile: DispatchProfile) {
        migrate_dispatch_profile(profile);
    }

    /// High-level workload-class migration. Maps the class to a
    /// [`DispatchProfile`] then calls
    /// [`Self::migrate_dispatch_profile`].
    ///
    /// Use this when the application observes a workload shift
    /// (Light -> Compute, Compute -> Heavy, Heavy -> Memory) and
    /// wants the scheduler's default plan to retune SMT
    /// activation + oversubscription + cost estimate without
    /// touching the per-call API.
    #[inline]
    pub fn migrate_workload_class(&self, class: WorkloadClass) {
        migrate_workload_class(class);
    }

    /// Read the active global [`DispatchProfile`].
    #[inline]
    pub fn active_dispatch_profile(&self) -> DispatchProfile {
        active_dispatch_profile()
    }

    /// Migrate the global active backend (CPU / CUDA / TPU / etc.)
    /// via one AtomicU32 Release-store. Subsequent dispatches
    /// route through the new backend's `DispatchBackend`
    /// implementation; if the requested backend is not registered,
    /// dispatches gracefully fall back to CPU (the always-
    /// auto-registered default).
    ///
    /// Per-op cost on the deque hot path: zero. Per-dispatch cost:
    /// one Acquire-load on the active-backend tag + one registry
    /// lookup (HashMap probe) at execute-entry. Migration cost:
    /// one Release-store (~1 ns).
    #[inline]
    pub fn migrate_backend(&self, backend: Backend) {
        migrate_backend(backend);
    }

    /// Read the active global backend id.
    #[inline]
    pub fn active_backend_id(&self) -> Backend {
        active_backend_id()
    }

    /// Resolve the active backend to a concrete `BackendRef`.
    /// Returns `(backend, fell_back_to_cpu)`; the second tuple
    /// element is true when the requested active backend was not
    /// registered and CPU was used as fallback. Useful for
    /// observability and telemetry around backend migration
    /// readiness.
    #[inline]
    pub fn resolve_active_backend(&self) -> (BackendRef, bool) {
        resolve_active_backend()
    }

    /// Builder: install a declarative WorkloadClass hint. Sets the
    /// underlying DispatchProfile on the JobPlan that this
    /// dispatcher constructs. The plan-level profile takes
    /// precedence over the global active profile for THIS
    /// dispatch only; subsequent dispatchers without the explicit
    /// class still read the global active tag.
    #[inline]
    pub fn with_workload_class(mut self, class: WorkloadClass) -> Self {
        // Stash the explicit class in the variant tag's high bits
        // via a sibling field. The implementation keeps the field
        // strictly as a hint, applied in build_plan when present.
        self.explicit_class = Some(class);
        self
    }
}

fn num_cpus() -> u32 {
    std::thread::available_parallelism()
        .map(|n| n.get() as u32)
        .unwrap_or(4)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_dispatcher_constructs() {
        let d = AdaptiveDispatcher::new();
        assert_eq!(d.k_outer, 4);
        assert!(!d.use_smt);
        assert_eq!(d.variant, Variant::Faithful);
    }

    #[test]
    fn builder_chains() {
        let d = AdaptiveDispatcher::new()
            .with_shape(WorkloadShape::ProducerFast { burst: 32 })
            .with_variant(Variant::Fast)
            .with_k_outer(8)
            .with_smt();
        assert_eq!(d.k_outer, 8);
        assert!(d.use_smt);
        assert_eq!(d.variant, Variant::Fast);
        assert!(matches!(d.shape, WorkloadShape::ProducerFast { burst: 32 }));
    }

    #[test]
    fn execute_streaming_runs_inline() {
        let d = AdaptiveDispatcher::new()
            .with_shape(WorkloadShape::Streaming);
        let r = d.execute_streaming(|| 42u32);
        assert_eq!(r, 42);
    }

    #[test]
    fn execute_cooperative_round_trips_caller_order() {
        let d = AdaptiveDispatcher::new()
            .with_shape(WorkloadShape::ProducerFast { burst: 8 });
        let closures: Vec<Box<dyn FnOnce() -> u32 + Send>> =
            (0..8u32).map(|i| Box::new(move || i) as _).collect();
        let results = d.execute_cooperative(closures);
        assert_eq!(results, (0..8u32).collect::<Vec<_>>());
    }

    #[test]
    fn execute_for_each_in_place_updates_items() {
        let d = AdaptiveDispatcher::new()
            .with_shape(WorkloadShape::WorkSteal { n_consumers: 4, batch_size: 64 });
        let mut items: Vec<u32> = (0..256).collect();
        d.execute_for_each(&mut items, |slice| {
            for x in slice.iter_mut() {
                *x = x.wrapping_mul(2);
            }
        });
        for (i, x) in items.iter().enumerate() {
            assert_eq!(*x, (i as u32).wrapping_mul(2));
        }
    }
}
