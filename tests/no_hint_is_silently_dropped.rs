//! Every hint a caller can set survives to the plan, and the ones that
//! steer routing still steer it.
//!
//! A caller who sets a hint and gets no error is entitled to assume it
//! took effect. `JobPlan` has twenty-three builders and no way to
//! report that one of them was ignored, so a hint that stops working
//! is invisible from the outside: the call compiles, the plan is
//! returned, the work runs, and the routing is simply not what was
//! asked for.
//!
//! That is not hypothetical. `with_leaf_shape` set the field correctly
//! while an inherited estimate overrode the classification it implied,
//! so the plan reported carrying the shape and routed as though none
//! had been given. Any test asserting only that the field reads back
//! would have passed throughout.
//!
//! So this file checks two different things:
//!
//! - Every builder sets the field it names. This catches a hint
//!   dropped outright.
//! - Every hint that changes a routing decision still changes it, and
//!   the direction is asserted. This catches a hint that is stored and
//!   then ignored, which is the failure that actually happened.
//!
//! All twenty-three builders are covered. Three of them have no field
//! of their own to read back and are asserted on what they resolve to
//! instead: `with_workload_shape` writes three other hints, and the two
//! site builders differ only in whether they defer to a site already
//! attached.

use flynnel::backend::Backend;
use flynnel::sched::adaptive_cooperative::CooperativeRouting;
use flynnel::sched::call_site::caller_site;
use flynnel::sched::deque_tier::DequeTier;
use flynnel::sched::k_gating::KGating;
use flynnel::sched::plan::BisectVariant;
use flynnel::sched::workload_shape::WorkloadShape;
use flynnel::{DispatchProfile, HwClass, JobPlan, LeafShape, Variant};

fn base() -> JobPlan {
    JobPlan::new(6, 4096)
}

// ---------------------------------------------------------------------
// Every builder sets the field it names.
// ---------------------------------------------------------------------

/// A hint dropped outright would leave the field at its default. These
/// set each one to a value distinguishable from that default and read
/// it back.
#[test]
fn every_builder_sets_the_field_it_names() {
    assert_eq!(base().with_hw_class(HwClass::Avx2).hw_class, HwClass::Avx2);
    assert_eq!(base().with_variant(Variant::Correct).variant, Variant::Correct);
    assert_eq!(base().with_numa_hint(1).numa_hint, Some(1));
    assert!(base().with_smt().use_smt);
    assert_eq!(
        base().with_estimated_per_item_ns(4242).estimated_per_item_ns,
        Some(4242)
    );
    assert_eq!(base().with_task_overhead_ns(77).task_overhead_ns, Some(77));
    assert_eq!(base().with_task_span_ns(88).task_span_ns, Some(88));
    assert_eq!(base().with_effective_task_count(99).effective_task_count, Some(99));
    assert_eq!(base().with_k_inner_log2(5).k_inner_log2, Some(5));
    assert_eq!(base().with_spin_before_yield_ns(1234).spin_before_yield_ns, Some(1234));
    assert_eq!(base().with_oversubscription_log2(3).oversubscription_log2, Some(3));
    assert_eq!(base().with_workers(7).worker_cap, Some(7));
    assert_eq!(base().with_leaf_shape(LeafShape::Gather).leaf_shape, LeafShape::Gather);
    assert_eq!(
        base().with_deque_tier_hint(DequeTier::SmtLocal).deque_tier_hint,
        Some(DequeTier::SmtLocal)
    );
    assert!(base().with_mailbox_routing(true).use_mailbox_routing);
    assert!(!base().with_mailbox_routing(false).use_mailbox_routing);
    // Auto is the default, so both non-default variants are set here:
    // a builder that ignored its argument would still read back Auto.
    assert_eq!(base().with_k_gating(KGating::PerSlot).k_gating, KGating::PerSlot);
    assert_eq!(base().with_k_gating(KGating::CounterOnly).k_gating, KGating::CounterOnly);
    assert_eq!(
        base().with_cooperative_routing(CooperativeRouting::ForceTree).cooperative_routing,
        CooperativeRouting::ForceTree
    );
    assert_eq!(
        base().with_backend(Backend::Cuda { device_id: 1 }).backend_hint,
        Some(Backend::Cuda { device_id: 1 })
    );
    assert_eq!(
        base().with_bisect_variant(BisectVariant::RayonStyleReplenish).bisect_variant,
        Some(BisectVariant::RayonStyleReplenish)
    );
    assert_eq!(base().with_cost_ns_per_elem(321).estimated_per_item_ns, Some(321));
    assert!(base().with_site(caller_site()).site.is_some());
}

/// `with_cost_ns_per_elem` is the other name for the per-item cost, and
/// it has to carry the same authority.
///
/// The two builders write the same field, but the field alone is not
/// what decides anything - an estimate the caller supplied outranks the
/// classifier, and an estimate the plan inherited does not. If only one
/// of the two set that flag, the other would look identical on
/// inspection and be overridden in routing, which is the shape of the
/// leaf-shape defect exactly.
#[test]
fn both_names_for_the_per_item_cost_carry_the_same_authority() {
    for n in [8u32, 64, 512, 4096] {
        let by_estimate = JobPlan::new(0, n)
            .with_leaf_shape(LeafShape::LatencyCompute)
            .with_estimated_per_item_ns(900);
        let by_cost = JobPlan::new(0, n)
            .with_leaf_shape(LeafShape::LatencyCompute)
            .with_cost_ns_per_elem(900);
        assert_eq!(
            by_estimate.estimated_per_item_ns, by_cost.estimated_per_item_ns,
            "n={n}: the two builders must store the same figure"
        );
        assert_eq!(
            by_estimate.use_smt, by_cost.use_smt,
            "n={n}: with_cost_ns_per_elem must reach routing the same way \
             with_estimated_per_item_ns does; a figure that is stored but \
             not marked as the caller's is silently outranked"
        );
        assert_eq!(
            by_estimate.oversubscription_log2, by_cost.oversubscription_log2,
            "n={n}: the two builders must produce the same split budget"
        );
    }
}

/// A profile the caller named survives a cost hint given after it.
///
/// The cost builders re-run the classifier so a caller's figure beats
/// the guess `JobPlan::new` made from size alone. A profile the caller
/// set is not that guess. Both are the caller speaking, and answering
/// the second by discarding the first drops a hint that was accepted
/// without complaint - the plan would report the profile it was given
/// and route by the one inferred instead.
#[test]
fn a_profile_the_caller_named_survives_a_later_cost_hint() {
    for profile in [DispatchProfile::MemoryBound, DispatchProfile::LatencyBound] {
        let named = JobPlan::set_profile(8, 1024, profile);
        for plan in [
            named.with_cost_ns_per_elem(80),
            named.with_estimated_per_item_ns(80),
        ] {
            assert_eq!(
                plan.use_smt,
                named.use_smt,
                "{profile:?}: a cost hint re-classified over the profile the \
                 caller set; smt went from {} to {}",
                named.use_smt,
                plan.use_smt
            );
            assert_eq!(
                plan.estimated_per_item_ns,
                Some(80),
                "{profile:?}: the cost hint itself must still land"
            );
        }
    }
}

/// A declared workload shape resolves to the low-level hints it names.
///
/// This builder is a bundle: it writes `k_gating`, `use_mailbox_routing`
/// and `oversubscription_log2` from the shape rather than storing a
/// shape of its own. So there is no field to read back, and a shape
/// that resolved to nothing would be indistinguishable from one that
/// was never given. What is asserted is that two different shapes
/// produce two different plans, and that the burst size inside a shape
/// reaches the split budget.
#[test]
fn a_declared_workload_shape_resolves_to_the_hints_it_names() {
    let streaming = base().with_workload_shape(WorkloadShape::Streaming);
    let bursty = base().with_workload_shape(WorkloadShape::ProducerFast { burst: 64 });

    assert_ne!(
        (streaming.k_gating, streaming.use_mailbox_routing, streaming.oversubscription_log2),
        (bursty.k_gating, bursty.use_mailbox_routing, bursty.oversubscription_log2),
        "two different shapes produced the same hints, so the shape was \
         not consulted"
    );
    assert_eq!(
        bursty.k_gating,
        KGating::PerSlot,
        "a producer-fast shape names per-slot gating"
    );

    // A shape is the caller describing the workload, so the factor it
    // implies has to carry a caller's authority the same way an
    // explicit oversubscription hint does.
    let bigger = base().with_workload_shape(WorkloadShape::ProducerFast { burst: 4096 });
    let (small, large) = (
        bursty
            .oversubscription_log2
            .expect("a producer-fast shape must resolve to a steal-headroom factor"),
        bigger
            .oversubscription_log2
            .expect("a producer-fast shape must resolve to a steal-headroom factor"),
    );
    assert!(
        large >= small,
        "a larger burst must not ask for less steal headroom: {large} against {small}"
    );
}

/// `with_site_if_none` defers to a site already attached, and attaches
/// one when there is not.
///
/// Both halves matter. The generic dispatch entries call this on every
/// plan that passes through them, so a version that overwrote would
/// silently replace a caller's explicit `with_site` with the entry's
/// own location, and the per-site state a caller was accumulating would
/// go to a different site from then on.
#[test]
fn attaching_a_site_only_when_absent_defers_to_one_already_there() {
    let outer = caller_site();
    let inner = caller_site();
    assert_ne!(outer, inner, "two distinct source lines must be two sites");

    let plan = base().with_site(outer).with_site_if_none(inner);
    assert_eq!(plan.site, Some(outer), "an explicit site must survive an inner attachment");

    let plan = base().with_site_if_none(inner);
    assert_eq!(plan.site, Some(inner), "a plan with no site must take the one offered");

    // And the explicit builder replaces, which is what makes it the
    // caller's override rather than a second if-none.
    let plan = base().with_site(inner).with_site(outer);
    assert_eq!(plan.site, Some(outer), "with_site is a replace, not an if-none");
}

// ---------------------------------------------------------------------
// Hints that steer routing still steer it.
// ---------------------------------------------------------------------

/// A leaf shape decides whether the SMT siblings are engaged, and must
/// do so at every size a caller might pass.
///
/// This is the case that failed: the field was set, and an estimate the
/// caller never supplied overrode what it implied. Small batches are
/// included deliberately, because that is where the override bit.
#[test]
fn a_leaf_shape_steers_smt_at_every_size() {
    for n in [8u32, 64, 512, 4096, 65_536] {
        assert!(
            JobPlan::new(0, n).with_leaf_shape(LeafShape::LatencyCompute).use_smt,
            "n={n}: LatencyCompute must engage the siblings"
        );
        assert!(
            JobPlan::new(0, n).with_leaf_shape(LeafShape::Gather).use_smt,
            "n={n}: Gather must engage the siblings"
        );
        assert!(
            !JobPlan::new(0, n).with_leaf_shape(LeafShape::PortCompute).use_smt,
            "n={n}: PortCompute must keep them parked"
        );
        assert!(
            !JobPlan::new(0, n).with_leaf_shape(LeafShape::Streaming).use_smt,
            "n={n}: Streaming must keep them parked"
        );
    }
}

/// An explicit profile decides the same knobs, and must not be undone
/// by the batch size either.
#[test]
fn an_explicit_profile_steers_smt_at_every_size() {
    for n in [8u32, 64, 4096, 65_536] {
        assert!(
            JobPlan::set_profile(0, n, DispatchProfile::LatencyBound).use_smt,
            "n={n}: LatencyBound engages the siblings"
        );
        assert!(
            !JobPlan::set_profile(0, n, DispatchProfile::PortBound).use_smt,
            "n={n}: PortBound parks them"
        );
    }
}

/// A worker cap of one runs the body on the calling thread.
///
/// This is what a caller uses to keep work on one thread while holding
/// something that cannot cross threads - a lock, a thread-local, a
/// context bound to the caller. If the cap were dropped the body would
/// fan out, which is a correctness problem for that caller rather than
/// a performance one.
///
/// Asserted behaviorally, by running a dispatch and observing which
/// threads the body ran on, because the cap is honored at dispatch
/// rather than in tier selection. Nothing public reports the decision,
/// so watching the work is the only way a consumer could check it
/// either.
#[test]
fn a_worker_cap_of_one_runs_the_body_on_the_calling_thread() {
    use std::collections::HashSet;
    use std::sync::Mutex;
    use std::thread::ThreadId;

    let plan = JobPlan::new(6, 65_536).with_workers(1);
    assert_eq!(plan.worker_cap, Some(1));

    let here = std::thread::current().id();
    let seen: Mutex<HashSet<ThreadId>> = Mutex::new(HashSet::new());
    let mut items: Vec<u64> = (0..65_536u64).collect();
    flynnel::for_each_chunk(&plan, &mut items, |chunk| {
        for c in chunk.iter_mut() {
            *c = c.wrapping_mul(31).wrapping_add(7);
        }
        seen.lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(std::thread::current().id());
    });

    let threads = seen.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    assert_eq!(
        threads.len(),
        1,
        "a cap of one must not fan out; the body ran on {} threads",
        threads.len()
    );
    assert!(
        threads.contains(&here),
        "a cap of one must run on the calling thread, not one pool worker"
    );
}

/// Setting a hint twice keeps the second, and setting two different
/// hints keeps both. A builder that reset its neighbours would drop a
/// hint the caller had already given.
#[test]
fn hints_compose_rather_than_overwrite_each_other() {
    let plan = base()
        .with_numa_hint(1)
        .with_k_inner_log2(4)
        .with_workers(3)
        .with_variant(Variant::Correct)
        .with_leaf_shape(LeafShape::Gather)
        .with_estimated_per_item_ns(500);

    assert_eq!(plan.numa_hint, Some(1), "numa hint lost to a later builder");
    assert_eq!(plan.k_inner_log2, Some(4), "k_inner lost to a later builder");
    assert_eq!(plan.worker_cap, Some(3), "worker cap lost to a later builder");
    assert_eq!(plan.variant, Variant::Correct, "variant lost to a later builder");
    assert_eq!(plan.leaf_shape, LeafShape::Gather, "leaf shape lost to a later builder");
    assert_eq!(
        plan.estimated_per_item_ns,
        Some(500),
        "an explicit estimate must win over the shape's default"
    );

    // Last write wins for the same knob.
    assert_eq!(base().with_workers(2).with_workers(5).worker_cap, Some(5));
}

/// The order two hints are given in does not change the plan.
///
/// Both of these set an estimate and a shape; a builder that consulted
/// the other's field at construction time would produce different plans
/// from the same pair, and a caller has no reason to expect that.
#[test]
fn the_order_hints_are_given_does_not_change_the_plan() {
    let a = base()
        .with_leaf_shape(LeafShape::LatencyCompute)
        .with_estimated_per_item_ns(900);
    let b = base()
        .with_estimated_per_item_ns(900)
        .with_leaf_shape(LeafShape::LatencyCompute);

    assert_eq!(a.use_smt, b.use_smt, "smt decision depends on builder order");
    assert_eq!(a.leaf_shape, b.leaf_shape);
    assert_eq!(a.estimated_per_item_ns, b.estimated_per_item_ns);
    assert_eq!(
        a.oversubscription_log2, b.oversubscription_log2,
        "oversubscription depends on builder order"
    );
}
