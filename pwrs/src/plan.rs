//! Job plans: what the scheduler decides before it dispatches
//! anything.
//!
//! A plan is the unit every other family takes. It is a value in the
//! crate and it is a value here: each `With` method answers a fresh
//! plan and leaves the one it was called on alone, which is what lets
//! a script keep a base plan and vary it.
//!
//!     $base = New-FlynnelPlan -KOuter 8 -BatchSize 100000 -Profile Streaming
//!     $wide = $base.WithSmt().WithWorkers(24)
//!     Resolve-FlynnelPlan -Plan $wide
//!
//! `Resolve-FlynnelPlan` is the one to reach for when the question is
//! what a plan does on this host: it answers every resolved figure in
//! one call, where reading them method by method is twenty crossings
//! for one answer.

use pwrs::prelude::*;

use crate::types::{
    BisectVariant, CooperativeRouting, DequeTier, DispatchProfile, HwClass, LeafShape, SchedTier,
    Variant,
};

/// The error for a plan argument the scheduler cannot take.
fn arg_err(message: impl Into<String>) -> PsError {
    PsError::new(
        ErrorCategory::InvalidArgument,
        "FlynnelArgument",
        message.into(),
    )
}

/// A scheduling plan for one dispatch: the data size, the batch, and
/// every hint the caller has given about how the work should run.
///
/// Each `With` method answers a fresh plan and leaves this one alone,
/// so a base plan can be varied without being consumed.
#[psclass(name = "Flynnel.JobPlan", mode = proxy)]
pub struct Plan {
    #[psfield(skip)]
    pub(crate) inner: flynnel::JobPlan,
}

impl Plan {
    pub(crate) fn of(inner: flynnel::JobPlan) -> Self {
        Self { inner }
    }
}

/// The operations of a `Flynnel.JobPlan`. Every `With` method answers
/// a fresh plan; every other method reads this one.
#[psmethods]
impl Plan {
    // -- what the plan was built with ---------------------------------

    /// Log2 of the data size the plan was built for.
    pub fn k_outer(&self) -> PsResult<u8> {
        Ok(self.inner.k_outer)
    }

    /// The batch size the plan was built for.
    pub fn batch_size(&self) -> PsResult<u32> {
        Ok(self.inner.batch_size)
    }

    /// Whether a dispatch profile was named when the plan was built.
    ///
    /// A plan does not retain the profile itself. Naming one sets the
    /// SMT request, the cost estimate and the oversubscription and is
    /// then dissolved into them, so the profile is an input and what
    /// it did is read from those three rather than from a field.
    pub fn profile_explicit(&self) -> PsResult<bool> {
        Ok(self.inner.profile_explicit)
    }

    /// The accuracy variant the work is asked for.
    pub fn variant(&self) -> PsResult<Variant> {
        Ok(self.inner.variant.into())
    }

    /// The kernel target class.
    pub fn hw_class(&self) -> PsResult<HwClass> {
        Ok(self.inner.hw_class.into())
    }

    /// The caller's leaf-shape hint.
    pub fn leaf_shape(&self) -> PsResult<LeafShape> {
        Ok(self.inner.leaf_shape.into())
    }

    /// Whether the right-half push is pinned to a coherence tier, and
    /// to which. Null when the plan leaves it to the default.
    pub fn deque_tier_hint(&self) -> PsResult<Option<DequeTier>> {
        Ok(self.inner.deque_tier_hint.map(Into::into))
    }

    /// Whether this plan pins the SMT-sibling mailbox route.
    pub fn use_mailbox_routing(&self) -> PsResult<bool> {
        Ok(self.inner.use_mailbox_routing)
    }

    /// Whether the caller pinned a worker count rather than letting
    /// the plan resolve one.
    pub fn caller_pinned(&self) -> PsResult<bool> {
        Ok(self.inner.caller_pinned())
    }

    // -- the builders, each answering a fresh plan ---------------------

    /// A plan with the SMT siblings asked for.
    pub fn with_smt(&self) -> PsResult<Plan> {
        Ok(Plan::of(self.inner.with_smt()))
    }

    // The argument is `hw_class` and not `class` because the
    // generated shell writes the Rust parameter name straight into
    // C#, where `class` is a keyword: it produced
    // `WithHwClass(HwClass class)` and a CS1001 at that column.
    /// A plan targeting `hw_class`.
    pub fn with_hw_class(&self, hw_class: HwClass) -> PsResult<Plan> {
        Ok(Plan::of(self.inner.with_hw_class(hw_class.into())))
    }

    /// A plan asking for `variant` accuracy.
    pub fn with_variant(&self, variant: Variant) -> PsResult<Plan> {
        Ok(Plan::of(self.inner.with_variant(variant.into())))
    }

    /// A plan hinted to a NUMA node.
    pub fn with_numa_hint(&self, node: u32) -> PsResult<Plan> {
        Ok(Plan::of(self.inner.with_numa_hint(node)))
    }

    /// A plan carrying the caller's leaf shape, which lets the
    /// classifier route correctly on the first call.
    pub fn with_leaf_shape(&self, shape: LeafShape) -> PsResult<Plan> {
        Ok(Plan::of(self.inner.with_leaf_shape(shape.into())))
    }

    /// A plan carrying a per-item cost estimate in nanoseconds. Set
    /// this and the task overhead together: the leaf-width model needs
    /// both, and with only one it falls back to the minimum leaf.
    pub fn with_estimated_per_item_ns(&self, ns: u32) -> PsResult<Plan> {
        Ok(Plan::of(self.inner.with_estimated_per_item_ns(ns)))
    }

    /// A plan carrying what one task costs to hand off, in
    /// nanoseconds.
    pub fn with_task_overhead_ns(&self, ns: u32) -> PsResult<Plan> {
        Ok(Plan::of(self.inner.with_task_overhead_ns(ns)))
    }

    /// A plan carrying how long one task runs, in nanoseconds.
    pub fn with_task_span_ns(&self, ns: u32) -> PsResult<Plan> {
        Ok(Plan::of(self.inner.with_task_span_ns(ns)))
    }

    /// A plan carrying how many tasks there effectively are.
    pub fn with_effective_task_count(&self, count: u32) -> PsResult<Plan> {
        Ok(Plan::of(self.inner.with_effective_task_count(count)))
    }

    /// A plan carrying log2 of the in-kernel lane count.
    pub fn with_k_inner_log2(&self, log2: u8) -> PsResult<Plan> {
        Ok(Plan::of(self.inner.with_k_inner_log2(log2)))
    }

    /// A plan carrying a per-element cost in nanoseconds.
    pub fn with_cost_ns_per_elem(&self, ns: u32) -> PsResult<Plan> {
        Ok(Plan::of(self.inner.with_cost_ns_per_elem(ns)))
    }

    /// A plan carrying how long a worker spins before it yields.
    pub fn with_spin_before_yield_ns(&self, ns: u64) -> PsResult<Plan> {
        Ok(Plan::of(self.inner.with_spin_before_yield_ns(ns)))
    }

    /// A plan carrying log2 of the leaves wanted per worker.
    pub fn with_oversubscription_log2(&self, log2: u8) -> PsResult<Plan> {
        Ok(Plan::of(self.inner.with_oversubscription_log2(log2)))
    }

    /// A plan pinned to a worker count, which stops the plan
    /// resolving one and makes CallerPinned true.
    pub fn with_workers(&self, workers: u32) -> PsResult<Plan> {
        Ok(Plan::of(self.inner.with_workers(workers)))
    }

    /// A plan pinned to a bisect variant.
    pub fn with_bisect_variant(&self, variant: BisectVariant) -> PsResult<Plan> {
        Ok(Plan::of(self.inner.with_bisect_variant(variant.into())))
    }

    /// A plan whose right-half push is pinned to one coherence tier.
    pub fn with_deque_tier_hint(&self, tier: DequeTier) -> PsResult<Plan> {
        Ok(Plan::of(self.inner.with_deque_tier_hint(tier.into())))
    }

    /// A plan with the SMT-sibling mailbox route turned on or off.
    pub fn with_mailbox_routing(&self, enable: bool) -> PsResult<Plan> {
        Ok(Plan::of(self.inner.with_mailbox_routing(enable)))
    }

    /// A plan pinning the shape an N-way cooperative join takes.
    pub fn with_cooperative_routing(&self, routing: CooperativeRouting) -> PsResult<Plan> {
        Ok(Plan::of(
            self.inner.with_cooperative_routing(routing.into()),
        ))
    }

    // A workload shape is a shape name plus the numbers that shape
    // needs, so each is its own method rather than one method with
    // five arguments of which four are ignored. Each sets the mailbox
    // route, the oversubscription and the burst path together.

    /// A plan shaped for single-producer streaming: orchestration
    /// only, no burst and no mailbox route.
    pub fn with_streaming_shape(&self) -> PsResult<Plan> {
        Ok(Plan::of(self.inner.with_workload_shape(
            flynnel::sched::workload_shape::WorkloadShape::Streaming,
        )))
    }

    /// A plan shaped for a producer that emits `burst` jobs between
    /// waits, which takes the burst push path.
    pub fn with_producer_fast_shape(&self, burst: u32) -> PsResult<Plan> {
        Ok(Plan::of(self.inner.with_workload_shape(
            flynnel::sched::workload_shape::WorkloadShape::ProducerFast { burst },
        )))
    }

    /// A plan shaped for independent consumers stealing from each
    /// other, given how many there are and what each handles.
    pub fn with_work_steal_shape(&self, n_consumers: u32, batch_size: u32) -> PsResult<Plan> {
        Ok(Plan::of(self.inner.with_workload_shape(
            flynnel::sched::workload_shape::WorkloadShape::WorkSteal {
                n_consumers,
                batch_size,
            },
        )))
    }

    /// A plan shaped for cooperative cross-core work over `n_cores`,
    /// which turns the owner-directed mailbox route on.
    pub fn with_cooperative_shape(&self, n_cores: u32) -> PsResult<Plan> {
        Ok(Plan::of(self.inner.with_workload_shape(
            flynnel::sched::workload_shape::WorkloadShape::Cooperative { n_cores },
        )))
    }

    /// A plan shaped for racing `n_variants` implementations of the
    /// same work, where each push is a distinct entry rather than a
    /// burst.
    pub fn with_variant_race_shape(&self, n_variants: u32) -> PsResult<Plan> {
        Ok(Plan::of(self.inner.with_workload_shape(
            flynnel::sched::workload_shape::WorkloadShape::VariantRace { n_variants },
        )))
    }

    // -- what it resolves to on this host ------------------------------

    /// The per-element cost in force, from the caller's estimate or
    /// the profile's default. Null when the profile supplies none and
    /// the caller set none.
    pub fn effective_ns_per_elem(&self) -> PsResult<Option<u32>> {
        Ok(self.inner.effective_ns_per_elem())
    }

    /// Log2 of the leaves per worker in force.
    pub fn effective_oversubscription_log2(&self) -> PsResult<u8> {
        Ok(self.inner.effective_oversubscription_log2())
    }

    /// The leaves per worker in force.
    pub fn effective_leaves_per_worker(&self) -> PsResult<u64> {
        Ok(self.inner.effective_leaves_per_worker() as u64)
    }

    /// Whether the SMT siblings are actually used, which the profile
    /// can decline even when the caller asked.
    pub fn effective_use_smt(&self) -> PsResult<bool> {
        Ok(self.inner.effective_use_smt())
    }

    /// How long a worker spins before yielding under this plan.
    pub fn effective_spin_before_yield_ns(&self) -> PsResult<u64> {
        Ok(self.inner.effective_spin_before_yield_ns())
    }

    /// The whole job's estimated cost in nanoseconds, from the
    /// per-item estimate and the batch. Null without an estimate.
    pub fn estimated_total_ns(&self) -> PsResult<Option<u64>> {
        Ok(self.inner.estimated_total_ns())
    }

    /// The in-kernel lane count this plan implies.
    pub fn k_inner_lanes(&self) -> PsResult<u64> {
        Ok(self.inner.k_inner_lanes() as u64)
    }

    /// The workers this plan resolves to right now.
    ///
    /// Starts the arena if it is not already running, so this is not a
    /// free inspection on a process that has not dispatched yet.
    pub fn resolved_workers(&self) -> PsResult<u64> {
        Ok(self.inner.resolved_workers() as u64)
    }

    /// The workers this plan would use given an arena of
    /// `arena_workers`, without starting anything.
    pub fn effective_workers(&self, arena_workers: u64) -> PsResult<u64> {
        Ok(self.inner.effective_workers(arena_workers as usize) as u64)
    }

    /// The leaf count the Tiny-Tasks model recommends for `workers`.
    /// Null unless the plan carries both a per-item cost and a task
    /// overhead, which is the gate the model needs.
    pub fn optimal_chunk_count(&self, workers: u64) -> PsResult<Option<u32>> {
        Ok(self.inner.optimal_chunk_count(workers as usize))
    }

    /// The same for an explicit item count `n`.
    pub fn optimal_chunk_count_for(&self, workers: u64, n: u64) -> PsResult<Option<u32>> {
        Ok(self
            .inner
            .optimal_chunk_count_for(workers as usize, n as usize))
    }

    /// The scheduler tier this plan picks against this host's
    /// topology.
    pub fn tier(&self) -> PsResult<SchedTier> {
        Ok(flynnel::sched::plan::pick_tier(&self.inner, flynnel::numa_topology()).into())
    }

    /// The backend this plan would dispatch to.
    ///
    /// Starts the arena if it is not already running.
    pub fn backend_name(&self) -> PsResult<String> {
        Ok(format!("{:?}", self.inner.pick_backend().id()))
    }
}

/// Everything a plan answers about this host, in one object.
#[psclass(name = "Flynnel.ResolvedPlan")]
#[derive(Clone, Default)]
pub struct ResolvedPlan {
    /// Log2 of the data size.
    pub k_outer: u8,
    /// The batch size.
    pub batch_size: u32,
    /// Whether a dispatch profile was named when the plan was built.
    /// The profile itself is not retained: it sets the SMT request,
    /// the cost estimate and the oversubscription below and is
    /// dissolved into them.
    pub profile_explicit: bool,
    /// The accuracy variant.
    pub variant: Variant,
    /// The kernel target class.
    pub hw_class: HwClass,
    /// The caller's leaf-shape hint.
    pub leaf_shape: LeafShape,
    /// The scheduler tier picked against this host's topology.
    pub tier: SchedTier,
    /// Workers resolved right now.
    pub resolved_workers: u64,
    /// Whether the caller pinned that count.
    pub caller_pinned: bool,
    /// Whether SMT siblings are actually used.
    pub effective_use_smt: bool,
    /// The per-element cost in force, null when there is none.
    pub effective_ns_per_elem: Option<u32>,
    /// Log2 of the leaves per worker.
    pub effective_oversubscription_log2: u8,
    /// Leaves per worker.
    pub effective_leaves_per_worker: u64,
    /// Spin before yield, nanoseconds.
    pub effective_spin_before_yield_ns: u64,
    /// The job's estimated cost, null without a per-item estimate.
    pub estimated_total_ns: Option<u64>,
    /// In-kernel lanes.
    pub k_inner_lanes: u64,
    /// The Tiny-Tasks leaf count for the resolved worker count, null
    /// unless both the per-item cost and the task overhead are set.
    pub optimal_chunk_count: Option<u32>,
    /// Which backend the plan would dispatch to.
    pub backend: String,
    /// Whether the right-half push is pinned to a tier.
    pub deque_tier_hint: Option<DequeTier>,
    /// Whether the SMT-sibling mailbox route is pinned on.
    pub use_mailbox_routing: bool,
}

/// Builds a job plan: the data size, the batch, and how the work
/// should be classified.
///
/// One of Profile, Bare or neither. With Profile the plan takes that
/// dispatch profile and its defaults; with Bare it takes none of the
/// host's adaptive state, which is what a measurement wants when the
/// adaptive layer is the thing under test; with neither it takes the
/// process's current adaptive profile.
///
/// # Examples
///
/// `New-FlynnelPlan -KOuter 8 -BatchSize 100000`
///
/// `New-FlynnelPlan -KOuter 10 -BatchSize 1000000 -Profile Streaming`
///
/// `New-FlynnelPlan -KOuter 6 -BatchSize 4096 -Bare -PerItemNs 200 -TaskOverheadNs 900`
#[cmdlet(
    verb = "New",
    noun = "FlynnelPlan",
    alias = "New-FlyPlan",
    output = ["Flynnel.JobPlan"]
)]
#[derive(Default)]
pub struct NewFlynnelPlan {
    /// Log2 of the data size, which picks the tier band.
    #[param(mandatory, position = 0)]
    pub k_outer: u8,
    /// How many items the dispatch covers.
    #[param(mandatory, position = 1)]
    pub batch_size: u32,
    /// The dispatch profile to build with. Absent, the plan takes the
    /// process's current adaptive profile.
    #[param]
    pub profile: Option<DispatchProfile>,
    /// Build with none of the host's adaptive state, for a
    /// measurement whose subject is that state.
    #[param]
    pub bare: bool,
    /// The per-item cost estimate in nanoseconds. Set it with
    /// TaskOverheadNs or the leaf-width model has nothing to solve.
    #[param]
    pub per_item_ns: Option<u32>,
    /// What one task costs to hand off, in nanoseconds.
    #[param]
    pub task_overhead_ns: Option<u32>,
    /// The caller's leaf shape, which routes correctly on call one.
    #[param]
    pub leaf_shape: Option<LeafShape>,
    /// Ask for the SMT siblings.
    #[param]
    pub smt: bool,
    /// Pin the worker count rather than resolving one.
    #[param]
    pub workers: Option<u32>,
}

impl Cmdlet for NewFlynnelPlan {
    fn process(&mut self, ps: &Pipeline<'_>) -> PsResult<()> {
        if self.bare && self.profile.is_some() {
            return Err(arg_err(
                "Bare and Profile ask for different things: Bare takes none of the host's \
                 adaptive state and Profile names one. Pass one or neither.",
            )
            .terminating());
        }
        let mut plan = match (self.bare, self.profile) {
            (true, _) => flynnel::JobPlan::bare(self.k_outer, self.batch_size),
            (false, Some(profile)) => {
                flynnel::JobPlan::set_profile(self.k_outer, self.batch_size, profile.into())
            }
            (false, None) => flynnel::JobPlan::new(self.k_outer, self.batch_size),
        };
        if let Some(ns) = self.per_item_ns {
            plan = plan.with_estimated_per_item_ns(ns);
        }
        if let Some(ns) = self.task_overhead_ns {
            plan = plan.with_task_overhead_ns(ns);
        }
        if let Some(shape) = self.leaf_shape {
            plan = plan.with_leaf_shape(shape.into());
        }
        if self.smt {
            plan = plan.with_smt();
        }
        if let Some(workers) = self.workers {
            if workers == 0 {
                return Err(arg_err("Workers must be at least one").terminating());
            }
            plan = plan.with_workers(workers);
        }
        // A per-item cost without the task overhead leaves the
        // Tiny-Tasks model unsolvable, and the plan then uses the
        // minimum leaf. Saying so costs nothing and is the difference
        // between a tuned plan and one that looks tuned.
        if self.per_item_ns.is_some() != self.task_overhead_ns.is_some() {
            pwrs::warning!(
                ps,
                "the leaf-width model needs both PerItemNs and TaskOverheadNs; with one of \
                 them the plan falls back to the minimum leaf and OptimalChunkCount answers \
                 nothing"
            )?;
        }
        ps.write(Plan::of(plan))
    }
}

/// Answers everything a plan resolves to on this host in one object:
/// the workers, the tier, the leaf width, the backend and the rest.
///
/// Reading these one method at a time is one crossing each; this is
/// one crossing for all of them, which is the difference the module's
/// own measurements are about.
///
/// Starts the arena if it is not already running, because the worker
/// count and the backend are answers about a live pool.
///
/// # Examples
///
/// `Resolve-FlynnelPlan -Plan $plan`
///
/// `New-FlynnelPlan -KOuter 8 -BatchSize 100000 | Resolve-FlynnelPlan`
#[cmdlet(
    verb = "Resolve",
    noun = "FlynnelPlan",
    alias = "Resolve-FlyPlan",
    output = ["Flynnel.ResolvedPlan"]
)]
#[derive(Default)]
pub struct ResolveFlynnelPlan {
    /// The plan to resolve. Optional in the type because a proxy
    /// class has no default to derive; mandatory to the binder, which
    /// is what guarantees it is here.
    #[param(mandatory, position = 0, value_from_pipeline)]
    pub plan: Option<Plan>,
}

impl Cmdlet for ResolveFlynnelPlan {
    fn process(&mut self, ps: &Pipeline<'_>) -> PsResult<()> {
        let Some(plan) = self.plan.as_ref() else {
            return Err(arg_err("Plan is required").terminating());
        };
        let p = &plan.inner;
        let workers = p.resolved_workers();
        ps.write(ResolvedPlan {
            k_outer: p.k_outer,
            batch_size: p.batch_size,
            profile_explicit: p.profile_explicit,
            variant: p.variant.into(),
            hw_class: p.hw_class.into(),
            leaf_shape: p.leaf_shape.into(),
            tier: flynnel::sched::plan::pick_tier(p, flynnel::numa_topology()).into(),
            resolved_workers: workers as u64,
            caller_pinned: p.caller_pinned(),
            effective_use_smt: p.effective_use_smt(),
            effective_ns_per_elem: p.effective_ns_per_elem(),
            effective_oversubscription_log2: p.effective_oversubscription_log2(),
            effective_leaves_per_worker: p.effective_leaves_per_worker() as u64,
            effective_spin_before_yield_ns: p.effective_spin_before_yield_ns(),
            estimated_total_ns: p.estimated_total_ns(),
            k_inner_lanes: p.k_inner_lanes() as u64,
            optimal_chunk_count: p.optimal_chunk_count(workers),
            backend: format!("{:?}", p.pick_backend().id()),
            deque_tier_hint: p.deque_tier_hint.map(Into::into),
            use_mailbox_routing: p.use_mailbox_routing,
        })
    }
}

/// Reads which scheduler tier a data size falls in before any other
/// input is considered.
///
/// This is the band alone. What a whole plan picks, after the batch,
/// the cost estimate, the hardware class and the topology have been
/// taken into account, is the Tier on Resolve-FlynnelPlan.
///
/// # Examples
///
/// `Get-FlynnelKBand -KOuter 8`
///
/// `0..14 | ForEach-Object { Get-FlynnelKBand -KOuter $_ }`
#[cmdlet(
    verb = "Get",
    noun = "FlynnelKBand",
    alias = "Get-FlyKBand",
    output = ["Flynnel.SchedTier"]
)]
#[derive(Default)]
pub struct GetFlynnelKBand {
    /// Log2 of the data size.
    #[param(mandatory, position = 0, value_from_pipeline)]
    pub k_outer: u8,
}

impl Cmdlet for GetFlynnelKBand {
    fn process(&mut self, ps: &Pipeline<'_>) -> PsResult<()> {
        let tier: SchedTier = flynnel::sched::plan::kband_for(self.k_outer).into();
        ps.write(tier)
    }
}

/// One dispatch profile and the defaults it carries.
#[psclass(name = "Flynnel.ProfileRow")]
#[derive(Clone, Default)]
pub struct ProfileRow {
    /// The profile.
    pub profile: DispatchProfile,
    /// Its default per-element cost in nanoseconds, null where it
    /// supplies none.
    pub default_ns_per_elem: Option<u32>,
    /// Log2 of its default leaves per worker.
    pub default_oversubscription_log2: u8,
    /// Whether it wakes the SMT siblings.
    pub is_latency_bound: bool,
    /// Whether it routes through the SMT-sibling mailbox.
    pub use_mailbox_routing: bool,
    /// Which coherence tier it pins the right-half push to, null for
    /// the default broad steal.
    pub deque_tier_hint: Option<DequeTier>,
}

/// Reads the whole dispatch-profile table: for each profile, the cost
/// estimate, the oversubscription, whether it wakes the SMT siblings
/// and which deque tier it prefers.
///
/// This is the table that decides SMT for every plan that does not
/// override it, so it is the first thing to read when a plan resolves
/// its workers in a way that surprises you.
///
/// # Examples
///
/// `Get-FlynnelDispatchProfile`
///
/// `Get-FlynnelDispatchProfile | Where-Object IsLatencyBound`
#[cmdlet(
    verb = "Get",
    noun = "FlynnelDispatchProfile",
    alias = "Get-FlyDispatchProfile",
    output = ["Flynnel.ProfileRow"]
)]
#[derive(Default)]
pub struct GetFlynnelDispatchProfile {}

impl Cmdlet for GetFlynnelDispatchProfile {
    fn process(&mut self, ps: &Pipeline<'_>) -> PsResult<()> {
        use flynnel::DispatchProfile as P;
        for profile in [
            P::LatencyBound,
            P::PortBound,
            P::MemoryBound,
            P::Streaming,
            P::Unspecified,
        ] {
            ps.write(ProfileRow {
                profile: profile.into(),
                default_ns_per_elem: profile.default_ns_per_elem(),
                default_oversubscription_log2: profile.default_oversubscription_log2(),
                is_latency_bound: profile.is_latency_bound(),
                use_mailbox_routing: profile.use_mailbox_routing(),
                deque_tier_hint: profile.deque_tier_hint().map(Into::into),
            })?;
        }
        Ok(())
    }
}
