//! Job plans: what the scheduler decides before it dispatches
//! anything.
//!
//! A plan is a value. It carries what the caller asked for, nothing
//! derived, and it crosses into any cmdlet that takes `-Plan`:
//!
//!     $p = New-FlynnelPlan -KOuter 8 -BatchSize 100000 -Smt -Workers 4
//!     $p = $p | Update-FlynnelPlan -Variant Faithful
//!     Invoke-FlynnelMap -InputObject $x -Operation Sqrt -Plan $p
//!
//! A value and not a fluent object, because the binding framework
//! cannot pass an object holding Rust-only state back into a cmdlet:
//! such a class has no properties to rebuild it from and says so. The
//! choice is between chained builders that no cmdlet can accept and a
//! value every cmdlet can, and a value is the more useful half. It is
//! also the shape a PowerShell reader expects, since parameters are
//! how this shell configures anything.
//!
//! `New-FlynnelPlan` takes the options a caller reaches for most.
//! `Update-FlynnelPlan` takes every option there is and answers a
//! modified copy, leaving the plan it was given alone.
//!
//! `Resolve-FlynnelPlan` is the one to reach for when the question is
//! what a plan does on this host: it answers every resolved figure in
//! one call, and those figures live there rather than on the plan
//! because they are answers about a host, not things the caller said.

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

/// The shape of the work a plan describes.
///
/// A shape is a name plus the numbers that name needs, so the numbers
/// ride beside it and the plan refuses a shape whose numbers are
/// absent rather than substituting a zero for them.
#[psenum(name = "Flynnel.WorkloadShape")]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum WorkloadShapeKind {
    /// Single-producer streaming: orchestration only, no burst and no
    /// mailbox route.
    #[default]
    Streaming,
    /// A producer emitting several jobs between waits, which takes the
    /// burst push path. Needs ShapeBurst.
    ProducerFast,
    /// Independent consumers stealing from each other. Needs
    /// ShapeConsumers and ShapeBatchSize.
    WorkSteal,
    /// Cooperative cross-core work, which turns the owner-directed
    /// mailbox route on. Needs ShapeCores.
    Cooperative,
    /// Racing several implementations of the same work, where each
    /// push is a distinct entry rather than a burst. Needs
    /// ShapeVariants.
    VariantRace,
}

/// A scheduling plan: the data size, the batch, and every hint the
/// caller has given about how the work should run.
///
/// Every field is what the caller asked for. What the host resolves it
/// to is Resolve-FlynnelPlan's answer, not a property here, so a plan
/// never carries a figure that depends on which machine read it.
#[psclass(name = "Flynnel.JobPlan")]
#[derive(Clone, Default)]
pub struct Plan {
    /// Log2 of the data size, which picks the tier band.
    pub k_outer: u8,
    /// How many items the dispatch covers.
    pub batch_size: u32,
    /// The dispatch profile named when the plan was built. Null means
    /// none was named and the plan takes the process's current
    /// adaptive profile.
    pub profile: Option<DispatchProfile>,
    /// Whether the plan was built with none of the host's adaptive
    /// state, for a measurement whose subject is that state.
    pub bare: bool,
    /// Whether the caller asked for the SMT siblings. What the plan
    /// actually does with them is EffectiveUseSmt on the resolution,
    /// because a profile can overrule the request.
    pub smt: bool,
    /// A pinned worker count, which stops the plan resolving one.
    pub workers: Option<u32>,
    /// A NUMA node to prefer.
    pub numa_hint: Option<u32>,
    /// The kernel target class.
    pub hw_class: Option<HwClass>,
    /// The accuracy variant the work is asked for.
    pub variant: Option<Variant>,
    /// The caller's leaf shape, which lets the classifier route
    /// correctly on the first call rather than after it has learned.
    pub leaf_shape: Option<LeafShape>,
    /// The per-item cost estimate in nanoseconds. The leaf-width model
    /// needs this and TaskOverheadNs together.
    pub per_item_ns: Option<u32>,
    /// What one task costs to hand off, in nanoseconds.
    pub task_overhead_ns: Option<u32>,
    /// How long one task runs, in nanoseconds.
    pub task_span_ns: Option<u32>,
    /// How many tasks there effectively are.
    pub effective_task_count: Option<u32>,
    /// Log2 of the in-kernel lane count.
    pub k_inner_log2: Option<u8>,
    /// A per-element cost in nanoseconds.
    pub cost_ns_per_elem: Option<u32>,
    /// How long a worker spins before it yields, in nanoseconds.
    pub spin_before_yield_ns: Option<u64>,
    /// Log2 of the leaves wanted per worker.
    pub oversubscription_log2: Option<u8>,
    /// A pinned bisect variant.
    pub bisect_variant: Option<BisectVariant>,
    /// A coherence tier to pin the right-half push to.
    pub deque_tier_hint: Option<DequeTier>,
    /// Whether the SMT-sibling mailbox route is pinned on or off.
    pub mailbox_routing: Option<bool>,
    /// The shape an N-way cooperative join takes.
    pub cooperative_routing: Option<CooperativeRouting>,
    /// The workload shape, which sets the mailbox route, the
    /// oversubscription and the burst path together.
    pub shape: Option<WorkloadShapeKind>,
    /// Jobs a ProducerFast shape emits between waits.
    pub shape_burst: Option<u32>,
    /// Consumers a WorkSteal shape has.
    pub shape_consumers: Option<u32>,
    /// What each WorkSteal consumer handles.
    pub shape_batch_size: Option<u32>,
    /// Cores a Cooperative shape spans.
    pub shape_cores: Option<u32>,
    /// Implementations a VariantRace shape races.
    pub shape_variants: Option<u32>,
}

impl Plan {
    /// The crate's own plan, built from what the caller asked for.
    ///
    /// Rebuilt on each use rather than held, because the crate's
    /// `JobPlan` also carries a per-call-site identity the scheduler
    /// attaches itself, and a plan kept across calls would carry one
    /// site's identity into another's statistics.
    pub(crate) fn to_job_plan(&self) -> PsResult<flynnel::JobPlan> {
        let mut plan = match (self.bare, self.profile) {
            (true, _) => flynnel::JobPlan::bare(self.k_outer, self.batch_size),
            (false, Some(profile)) => {
                flynnel::JobPlan::set_profile(self.k_outer, self.batch_size, profile.into())
            }
            (false, None) => flynnel::JobPlan::new(self.k_outer, self.batch_size),
        };
        if self.smt {
            plan = plan.with_smt();
        }
        if let Some(v) = self.hw_class {
            plan = plan.with_hw_class(v.into());
        }
        if let Some(v) = self.variant {
            plan = plan.with_variant(v.into());
        }
        if let Some(v) = self.numa_hint {
            plan = plan.with_numa_hint(v);
        }
        if let Some(v) = self.leaf_shape {
            plan = plan.with_leaf_shape(v.into());
        }
        if let Some(v) = self.per_item_ns {
            plan = plan.with_estimated_per_item_ns(v);
        }
        if let Some(v) = self.task_overhead_ns {
            plan = plan.with_task_overhead_ns(v);
        }
        if let Some(v) = self.task_span_ns {
            plan = plan.with_task_span_ns(v);
        }
        if let Some(v) = self.effective_task_count {
            plan = plan.with_effective_task_count(v);
        }
        if let Some(v) = self.k_inner_log2 {
            plan = plan.with_k_inner_log2(v);
        }
        if let Some(v) = self.cost_ns_per_elem {
            plan = plan.with_cost_ns_per_elem(v);
        }
        if let Some(v) = self.spin_before_yield_ns {
            plan = plan.with_spin_before_yield_ns(v);
        }
        if let Some(v) = self.oversubscription_log2 {
            plan = plan.with_oversubscription_log2(v);
        }
        if let Some(v) = self.bisect_variant {
            plan = plan.with_bisect_variant(v.into());
        }
        if let Some(v) = self.deque_tier_hint {
            plan = plan.with_deque_tier_hint(v.into());
        }
        if let Some(v) = self.mailbox_routing {
            plan = plan.with_mailbox_routing(v);
        }
        if let Some(v) = self.cooperative_routing {
            plan = plan.with_cooperative_routing(v.into());
        }
        // A shape sets the mailbox route, the oversubscription and the
        // burst path together, so it is applied after the options it
        // would otherwise overwrite one at a time.
        if let Some(shape) = self.shape {
            plan = plan.with_workload_shape(self.workload_shape(shape)?);
        }
        // Applied after the shape, so a worker count the caller pinned
        // survives a shape that implies a different one.
        if let Some(v) = self.workers {
            plan = plan.with_workers(v);
        }
        Ok(plan)
    }

    /// The crate's workload shape, refusing a shape whose numbers are
    /// missing rather than standing a zero in for them.
    fn workload_shape(
        &self,
        shape: WorkloadShapeKind,
    ) -> PsResult<flynnel::sched::workload_shape::WorkloadShape> {
        use flynnel::sched::workload_shape::WorkloadShape as W;
        let need = |value: Option<u32>, name: &str| -> PsResult<u32> {
            match value {
                Some(v) => Ok(v),
                None => Err(arg_err(format!(
                    "the {shape:?} shape needs {name}; a shape is a name plus the numbers it \
                     needs, and a zero here would be a number nobody gave"
                ))
                .terminating()),
            }
        };
        Ok(match shape {
            WorkloadShapeKind::Streaming => W::Streaming,
            WorkloadShapeKind::ProducerFast => W::ProducerFast {
                burst: need(self.shape_burst, "ShapeBurst")?,
            },
            WorkloadShapeKind::WorkSteal => W::WorkSteal {
                n_consumers: need(self.shape_consumers, "ShapeConsumers")?,
                batch_size: need(self.shape_batch_size, "ShapeBatchSize")?,
            },
            WorkloadShapeKind::Cooperative => W::Cooperative {
                n_cores: need(self.shape_cores, "ShapeCores")?,
            },
            WorkloadShapeKind::VariantRace => W::VariantRace {
                n_variants: need(self.shape_variants, "ShapeVariants")?,
            },
        })
    }
}

/// Everything a plan resolves to on this host.
#[psclass(name = "Flynnel.ResolvedPlan")]
#[derive(Clone, Default)]
pub struct ResolvedPlan {
    /// Log2 of the data size the plan was built for.
    pub k_outer: u8,
    /// The batch size the plan was built for.
    pub batch_size: u32,
    /// Whether a dispatch profile was named when the plan was built.
    ///
    /// A plan does not retain the profile itself once it reaches the
    /// scheduler. Naming one sets the SMT request, the cost estimate
    /// and the oversubscription and is then dissolved into them, so
    /// the profile is an input and what it did is read from those
    /// three rather than from a field.
    pub profile_explicit: bool,
    /// The accuracy variant in force.
    pub variant: Variant,
    /// The kernel target class in force.
    pub hw_class: HwClass,
    /// The leaf shape in force.
    pub leaf_shape: LeafShape,
    /// The tier this plan dispatches at on this host.
    pub tier: SchedTier,
    /// How many workers it resolves to.
    pub resolved_workers: u64,
    /// Whether the caller pinned that count rather than letting the
    /// plan resolve one.
    pub caller_pinned: bool,
    /// Whether the SMT siblings are actually asked for, which the
    /// profile can decide against the caller's request.
    pub effective_use_smt: bool,
    /// The per-element cost in force, from the caller's estimate or
    /// the profile's default. Null when neither supplies one.
    pub effective_ns_per_elem: Option<u32>,
    /// Log2 of the leaves wanted per worker, in force.
    pub effective_oversubscription_log2: u8,
    /// Leaves per worker, in force.
    pub effective_leaves_per_worker: u64,
    /// How long a worker spins before it yields, in force.
    pub effective_spin_before_yield_ns: u64,
    /// The whole dispatch's estimated cost in nanoseconds. Null when
    /// the leaf-width model has nothing to solve.
    pub estimated_total_ns: Option<u64>,
    /// In-kernel lanes.
    pub k_inner_lanes: u64,
    /// The leaf count the width model recommends. Null when the model
    /// is unsolvable, which is not the same as a count of zero.
    pub optimal_chunk_count: Option<u32>,
    /// The backend this plan picks.
    pub backend: String,
    /// The coherence tier the right-half push is pinned to, if any.
    pub deque_tier_hint: Option<DequeTier>,
    /// Whether the SMT-sibling mailbox route is on.
    pub use_mailbox_routing: bool,
}

/// Builds a job plan.
///
/// Takes the options a caller reaches for most. Update-FlynnelPlan
/// takes every option there is, including these, and answers a
/// modified copy.
///
/// # Examples
///
/// `New-FlynnelPlan -KOuter 8 -BatchSize 100000`
///
/// `New-FlynnelPlan -KOuter 8 -BatchSize 100000 -Profile Streaming -Workers 4`
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
        if self.workers == Some(0) {
            return Err(arg_err("Workers must be at least one").terminating());
        }
        // A per-item cost without the task overhead leaves the
        // leaf-width model unsolvable, and the plan then uses the
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
        let plan = Plan {
            k_outer: self.k_outer,
            batch_size: self.batch_size,
            profile: self.profile,
            bare: self.bare,
            smt: self.smt,
            workers: self.workers,
            leaf_shape: self.leaf_shape,
            per_item_ns: self.per_item_ns,
            task_overhead_ns: self.task_overhead_ns,
            ..Plan::default()
        };
        // Built once here, so an argument the crate refuses is refused
        // now rather than at whichever kernel first used the plan.
        plan.to_job_plan()?;
        ps.write(plan)
    }
}

/// Answers a copy of a plan with the options given changed, and leaves
/// the plan it was given alone.
///
/// Every option a plan can carry is here, including the ones
/// New-FlynnelPlan also takes. An option not given is left as it was,
/// so this is a change rather than a rebuild. There is deliberately no
/// way to clear an option: a plan built without it is the way back,
/// and a switch that both sets and clears cannot say which a caller
/// meant.
///
/// # Examples
///
/// `$p | Update-FlynnelPlan -Variant Faithful -Workers 8`
///
/// `Update-FlynnelPlan -Plan $p -Shape WorkSteal -ShapeConsumers 8 -ShapeBatchSize 64`
#[cmdlet(
    verb = "Update",
    noun = "FlynnelPlan",
    alias = "Update-FlyPlan",
    output = ["Flynnel.JobPlan"]
)]
#[derive(Default)]
pub struct UpdateFlynnelPlan {
    /// The plan to copy and change.
    #[param(mandatory, position = 0, value_from_pipeline)]
    pub plan: Option<Plan>,
    /// Ask for the SMT siblings, or stop asking.
    #[param]
    pub smt: Option<bool>,
    /// Pin the worker count rather than resolving one.
    #[param]
    pub workers: Option<u32>,
    /// A NUMA node to prefer.
    #[param]
    pub numa_hint: Option<u32>,
    /// The kernel target class.
    #[param]
    pub hw_class: Option<HwClass>,
    /// The accuracy variant the work is asked for.
    #[param]
    pub variant: Option<Variant>,
    /// The caller's leaf shape.
    #[param]
    pub leaf_shape: Option<LeafShape>,
    /// The per-item cost estimate in nanoseconds.
    #[param]
    pub per_item_ns: Option<u32>,
    /// What one task costs to hand off, in nanoseconds.
    #[param]
    pub task_overhead_ns: Option<u32>,
    /// How long one task runs, in nanoseconds.
    #[param]
    pub task_span_ns: Option<u32>,
    /// How many tasks there effectively are.
    #[param]
    pub effective_task_count: Option<u32>,
    /// Log2 of the in-kernel lane count.
    #[param]
    pub k_inner_log2: Option<u8>,
    /// A per-element cost in nanoseconds.
    #[param]
    pub cost_ns_per_elem: Option<u32>,
    /// How long a worker spins before it yields, in nanoseconds.
    #[param]
    pub spin_before_yield_ns: Option<u64>,
    /// Log2 of the leaves wanted per worker.
    #[param]
    pub oversubscription_log2: Option<u8>,
    /// A pinned bisect variant.
    #[param]
    pub bisect_variant: Option<BisectVariant>,
    /// A coherence tier to pin the right-half push to.
    #[param]
    pub deque_tier_hint: Option<DequeTier>,
    /// Turn the SMT-sibling mailbox route on or off.
    #[param]
    pub mailbox_routing: Option<bool>,
    /// The shape an N-way cooperative join takes.
    #[param]
    pub cooperative_routing: Option<CooperativeRouting>,
    /// The workload shape.
    #[param]
    pub shape: Option<WorkloadShapeKind>,
    /// Jobs a ProducerFast shape emits between waits.
    #[param]
    pub shape_burst: Option<u32>,
    /// Consumers a WorkSteal shape has.
    #[param]
    pub shape_consumers: Option<u32>,
    /// What each WorkSteal consumer handles.
    #[param]
    pub shape_batch_size: Option<u32>,
    /// Cores a Cooperative shape spans.
    #[param]
    pub shape_cores: Option<u32>,
    /// Implementations a VariantRace shape races.
    #[param]
    pub shape_variants: Option<u32>,
}

impl Cmdlet for UpdateFlynnelPlan {
    fn process(&mut self, ps: &Pipeline<'_>) -> PsResult<()> {
        let Some(base) = self.plan.as_ref() else {
            return Err(arg_err("Plan is required").terminating());
        };
        if self.workers == Some(0) {
            return Err(arg_err("Workers must be at least one").terminating());
        }
        let mut next = base.clone();
        if let Some(v) = self.smt {
            next.smt = v;
        }
        // Each option is taken only when it was given, so an absent
        // one leaves what the plan already carried.
        if self.workers.is_some() {
            next.workers = self.workers;
        }
        if self.numa_hint.is_some() {
            next.numa_hint = self.numa_hint;
        }
        if self.hw_class.is_some() {
            next.hw_class = self.hw_class;
        }
        if self.variant.is_some() {
            next.variant = self.variant;
        }
        if self.leaf_shape.is_some() {
            next.leaf_shape = self.leaf_shape;
        }
        if self.per_item_ns.is_some() {
            next.per_item_ns = self.per_item_ns;
        }
        if self.task_overhead_ns.is_some() {
            next.task_overhead_ns = self.task_overhead_ns;
        }
        if self.task_span_ns.is_some() {
            next.task_span_ns = self.task_span_ns;
        }
        if self.effective_task_count.is_some() {
            next.effective_task_count = self.effective_task_count;
        }
        if self.k_inner_log2.is_some() {
            next.k_inner_log2 = self.k_inner_log2;
        }
        if self.cost_ns_per_elem.is_some() {
            next.cost_ns_per_elem = self.cost_ns_per_elem;
        }
        if self.spin_before_yield_ns.is_some() {
            next.spin_before_yield_ns = self.spin_before_yield_ns;
        }
        if self.oversubscription_log2.is_some() {
            next.oversubscription_log2 = self.oversubscription_log2;
        }
        if self.bisect_variant.is_some() {
            next.bisect_variant = self.bisect_variant;
        }
        if self.deque_tier_hint.is_some() {
            next.deque_tier_hint = self.deque_tier_hint;
        }
        if self.mailbox_routing.is_some() {
            next.mailbox_routing = self.mailbox_routing;
        }
        if self.cooperative_routing.is_some() {
            next.cooperative_routing = self.cooperative_routing;
        }
        if self.shape.is_some() {
            next.shape = self.shape;
        }
        if self.shape_burst.is_some() {
            next.shape_burst = self.shape_burst;
        }
        if self.shape_consumers.is_some() {
            next.shape_consumers = self.shape_consumers;
        }
        if self.shape_batch_size.is_some() {
            next.shape_batch_size = self.shape_batch_size;
        }
        if self.shape_cores.is_some() {
            next.shape_cores = self.shape_cores;
        }
        if self.shape_variants.is_some() {
            next.shape_variants = self.shape_variants;
        }
        // Refused here rather than at whichever kernel first used it.
        next.to_job_plan()?;
        ps.write(next)
    }
}

/// Answers everything a plan resolves to on this host in one object:
/// the workers, the tier, the leaf width, the backend and the rest.
///
/// Reading these one at a time would be one crossing each; this is one
/// crossing for all of them. They live here rather than on the plan
/// because they are answers about a host, and a plan that carried them
/// would be a different value on every machine that read it.
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
    /// The plan to resolve.
    #[param(mandatory, position = 0, value_from_pipeline)]
    pub plan: Option<Plan>,
}

impl Cmdlet for ResolveFlynnelPlan {
    fn process(&mut self, ps: &Pipeline<'_>) -> PsResult<()> {
        let Some(plan) = self.plan.as_ref() else {
            return Err(arg_err("Plan is required").terminating());
        };
        let p = plan.to_job_plan()?;
        let workers = p.resolved_workers();
        ps.write(ResolvedPlan {
            k_outer: p.k_outer,
            batch_size: p.batch_size,
            profile_explicit: p.profile_explicit,
            variant: p.variant.into(),
            hw_class: p.hw_class.into(),
            leaf_shape: p.leaf_shape.into(),
            tier: flynnel::sched::plan::pick_tier(&p, flynnel::numa_topology()).into(),
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
