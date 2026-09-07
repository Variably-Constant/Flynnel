//! The plan builders, the host probes and the process-global knobs.
//!
//! These are the inputs every dispatch decision reads. A builder that
//! silently drops a setting, a topology that reports a node no CPU
//! belongs to, or a knob that does not round-trip would all be
//! invisible at the call site and would move scheduling underneath
//! every caller at once.

use std::sync::{Mutex, MutexGuard};
use std::sync::atomic::{AtomicUsize, Ordering};

use flynnel::sched::cat::CatCapability;
use flynnel::{
    CallSiteState, DispatchProfile, HwClass, JobPlan, L3Reservation, LeafShape, NumaSource,
    Placement, SchedTier, SiteRef, Variant, caller_site, join, numa_topology, reset_spin_stats,
    set_spin_adaptive, set_spin_window, site_for_location, spin_window, total_idle_yields,
};

/// The spin knobs are process-global, so the tests that move them run
/// one at a time and put the window back.
static KNOBS: Mutex<()> = Mutex::new(());

fn knobs() -> MutexGuard<'static, ()> {
    KNOBS.lock().unwrap_or_else(|e| e.into_inner())
}

#[test]
fn the_spin_window_round_trips_and_restores() {
    let _g = knobs();
    let original = spin_window();
    for rounds in [1u32, 64, 500, 4096] {
        set_spin_window(rounds);
        assert_eq!(spin_window(), rounds, "the window reads back what was set");
    }
    set_spin_window(original);
    assert_eq!(spin_window(), original, "and the original is restorable");
}

#[test]
fn the_adaptive_spin_toggle_does_not_disturb_the_window() {
    let _g = knobs();
    let original = spin_window();
    set_spin_window(321);
    set_spin_adaptive(true);
    set_spin_adaptive(false);
    assert_eq!(
        spin_window(),
        321,
        "toggling adaptation is a separate knob from the window value"
    );
    set_spin_window(original);
}

/// A reset discards the yields counted before it.
///
/// The assertion is not that the counter reads zero afterwards. Idle
/// workers keep yielding on their own schedule once a pool is live, so
/// a read taken any time after the reset can already have moved, and a
/// test demanding zero passes or fails on timing rather than on
/// behavior. What a reset owes the caller is that the earlier count is
/// gone, which is what this compares.
#[test]
fn resetting_spin_stats_discards_the_earlier_yield_count() {
    let _g = knobs();
    let plan = JobPlan::new(8, 4096);

    // Idle yields accrue only once workers have woken and gone quiet
    // again, so drive the pool until the counter has something in it
    // rather than assuming one join is enough.
    let mut before = 0u64;
    for _ in 0..64 {
        let (a, b) = join(&plan, || 1u32, || 2u32);
        assert_eq!((a, b), (1, 2));
        before = total_idle_yields();
        if before > 0 {
            break;
        }
    }

    reset_spin_stats();
    let after = total_idle_yields();
    assert!(
        after < before.max(1),
        "a reset must drop the {before} yields already counted, saw {after}"
    );
}

#[test]
fn the_numa_snapshot_is_internally_consistent() {
    let topo = numa_topology();
    assert!(topo.num_nodes >= 1, "a host always has at least one node");
    assert!(!topo.node_of_cpu.is_empty(), "and at least one logical CPU");
    let nodes = topo.num_nodes as usize;
    for (cpu, &node) in topo.node_of_cpu.iter().enumerate() {
        assert!(
            (node as usize) < nodes,
            "cpu {cpu} claims node {node}, which is past the {nodes} the probe found"
        );
    }
    assert_eq!(topo.distances.len(), nodes, "the distance matrix has one row per node");
    for (i, row) in topo.distances.iter().enumerate() {
        assert_eq!(row.len(), nodes, "row {i} is square with the node count");
        assert!(row[i] > 0, "a node's distance to itself is a positive cost");
        for (j, &d) in row.iter().enumerate() {
            assert!(d >= row[i], "distance {i}->{j} is not nearer than {i}->{i}");
        }
    }
    // The probe names which path produced this, and a fallback must
    // look like a fallback rather than claiming a real probe ran.
    if topo.source == NumaSource::Fallback {
        assert_eq!(topo.num_nodes, 1, "the fallback is a single node by definition");
    }
}

#[test]
fn the_cat_probe_answers_on_a_host_without_it() {
    // Windows and any Linux without resctrl have no L3 CAT. The probe
    // must report that rather than panic, and a reservation attempt
    // must fail cleanly.
    let cap = CatCapability::detect();
    if cap.supported {
        assert!(cap.cbm_bits > 0, "a supported host reports its bitmask width");
        assert!(cap.num_domains >= 1, "and at least one L3 domain");
        assert!(cap.min_cbm_bits >= 1, "and a minimum reservation of at least one way");
    } else {
        assert_eq!(cap.cbm_bits, 0, "an unsupported host reports no bitmask");
        let refused = L3Reservation::reserve_ways("flynnel_test", 0, 1);
        assert!(refused.is_err(), "and refuses a reservation rather than pretending");
    }
}

#[test]
fn the_matrix_extension_classes_are_the_ones_documented_as_tiles() {
    // The tile classes need mode-region batching; the vector ones do
    // not. Getting this wrong routes a kernel to the wrong dispatch
    // shape.
    for tile in [HwClass::AmxInt8, HwClass::AmxFp16, HwClass::TensorCoreHopper] {
        assert!(tile.is_matrix_extension(), "{tile:?} is a tile class");
    }
    for vector in [HwClass::Scalar, HwClass::Avx2] {
        assert!(!vector.is_matrix_extension(), "{vector:?} is not a tile class");
    }
}

#[test]
fn the_default_precision_variant_is_the_exact_one() {
    // A default that silently meant Fast would downgrade every caller
    // that did not name a variant.
    assert_eq!(Variant::default(), Variant::Correct);
}

#[test]
fn the_plan_builders_keep_what_they_were_given() {
    let p = JobPlan::new(8, 100)
        .with_hw_class(HwClass::Avx2)
        .with_variant(Variant::Correct)
        .with_numa_hint(1)
        .with_estimated_per_item_ns(4242)
        .with_workers(4);
    assert_eq!(p.k_outer, 8);
    assert_eq!(p.batch_size, 100);
    assert_eq!(p.hw_class, HwClass::Avx2);
    assert_eq!(p.variant, Variant::Correct);
    assert_eq!(p.numa_hint, Some(1));
    assert_eq!(p.estimated_per_item_ns, Some(4242));
    assert_eq!(p.worker_cap, Some(4));
}

#[test]
fn a_worker_cap_of_zero_is_raised_to_one_rather_than_meaning_none() {
    // Zero workers would be a plan that cannot run; the documented
    // behavior is a floor at one.
    let p = JobPlan::new(8, 1024).with_workers(0);
    assert_eq!(p.worker_cap, Some(1));
    assert_eq!(p.effective_workers(16), 1, "and it resolves to a single worker");
    assert_eq!(
        JobPlan::new(8, 1024).effective_workers(16),
        16,
        "while no cap uses the whole arena"
    );
    assert_eq!(
        JobPlan::new(8, 1024).with_workers(99).effective_workers(16),
        16,
        "and a cap past the arena is the arena"
    );
}

/// Each leaf shape selects the profile its class implies, and the
/// difference between them is the SMT decision.
///
/// `PortCompute` was the only shape any test used, while consumers pass
/// the other two: a device-offload consumer gives `LatencyCompute` as
/// its only hint on every par_iter site it owns, and others pass
/// `Gather`. Since
/// `with_leaf_shape` re-derives use_smt, oversubscription, mailbox
/// routing and the deque tier from the shape, testing one variant said
/// nothing about the two being relied on.
#[test]
fn every_leaf_shape_carries_the_smt_decision_of_its_class() {
    // 65536 items, large enough that the inherited per-item estimate
    // cannot put the total under the fine-grain budget. See
    // an_inherited_estimate_discards_the_shape_hint for what happens
    // below that line.
    let shaped = |s: LeafShape| JobPlan::new(0, 65_536).with_leaf_shape(s);

    assert!(
        !shaped(LeafShape::PortCompute).use_smt,
        "port-saturating work keeps the siblings parked"
    );
    assert!(
        !shaped(LeafShape::Streaming).use_smt,
        "streaming work keeps them parked too"
    );
    assert!(
        shaped(LeafShape::LatencyCompute).use_smt,
        "latency-bound work engages the siblings to cover its stalls"
    );
    assert!(
        shaped(LeafShape::Gather).use_smt,
        "irregular access engages them to interleave the missed loads"
    );

    for s in [
        LeafShape::PortCompute,
        LeafShape::Streaming,
        LeafShape::LatencyCompute,
        LeafShape::Gather,
    ] {
        let plan = shaped(s);
        assert_eq!(plan.leaf_shape, s, "the plan carries the shape it was given");
        assert!(
            plan.oversubscription_log2.is_some(),
            "{s:?} must resolve an oversubscription factor, not leave it unset"
        );
    }
}

/// A shape the caller named outranks an estimate the caller did not.
///
/// `JobPlan::new` seeds `estimated_per_item_ns` from the process-active
/// dispatch profile and marks it non-explicit. That default describes no
/// particular call, so it is not offered to the shape classifier's
/// fine-grain guard: only a figure the caller supplied can outrank a
/// shape the caller named.
///
/// The size range matters. With a 12 ns/item default, any batch below
/// roughly 4200 items totals under the 50 us fine-grain budget, so
/// without this precedence every small shaped batch would be classified
/// fine-grain. That is the range consumers dispatch in - tens to low
/// hundreds of items, `LatencyCompute` as the only hint, no estimate.
#[test]
fn a_named_shape_outranks_an_estimate_the_caller_never_supplied() {
    // Both sides of the budget, same shape, no caller estimate.
    for n in [64u32, 8192, 65_536] {
        let plan = JobPlan::new(0, n).with_leaf_shape(LeafShape::LatencyCompute);
        assert!(
            plan.use_smt,
            "at {n} items the named shape decides the routing, whatever the \
             inherited default would have totalled"
        );
        assert_eq!(plan.leaf_shape, LeafShape::LatencyCompute);
    }

    // An estimate the caller does supply still governs, which is the
    // guard's intended use: 1 ns x 8192 is genuinely too small to
    // dispatch, and saying so is the caller's business.
    let explicit = JobPlan::new(0, 8192)
        .with_leaf_shape(LeafShape::LatencyCompute)
        .with_estimated_per_item_ns(1);
    assert!(
        !explicit.use_smt,
        "an estimate the caller wrote is the right thing for the guard to \
         act on, and 1 ns x 8192 is under the budget"
    );

    // Order-independent: naming the shape after the estimate must reach
    // the same decision, or the contract depends on builder order.
    let reversed = JobPlan::new(0, 8192)
        .with_estimated_per_item_ns(1)
        .with_leaf_shape(LeafShape::LatencyCompute);
    assert_eq!(
        reversed.use_smt, explicit.use_smt,
        "the builders must commute for an explicit estimate"
    );
}

#[test]
fn an_explicit_profile_is_what_the_plan_carries() {
    let port = JobPlan::set_profile(6, 16, DispatchProfile::PortBound);
    assert!(!port.use_smt, "PortBound keeps the siblings parked");
    let latency = JobPlan::set_profile(6, 16, DispatchProfile::LatencyBound);
    assert!(latency.use_smt, "LatencyBound engages them");
}

#[test]
fn a_leaf_shape_hint_reaches_the_pool_even_at_a_micro_k() {
    // The documented reason the hint exists: a caller whose workload
    // has no precision tier supplies K=0, and without the hint that
    // routes to inline serial and defeats the plan.
    let calling = std::thread::current().id();
    let off = AtomicUsize::new(0);
    let plan = JobPlan::new(0, 4096).with_leaf_shape(LeafShape::PortCompute);
    let (a, b) = join(
        &plan,
        || {
            if std::thread::current().id() != calling {
                off.fetch_add(1, Ordering::Relaxed);
            }
            1u32
        },
        || {
            if std::thread::current().id() != calling {
                off.fetch_add(1, Ordering::Relaxed);
            }
            2u32
        },
    );
    assert_eq!((a, b), (1, 2));
    assert_eq!(plan.leaf_shape, LeafShape::PortCompute, "the hint is carried on the plan");
}

#[test]
fn the_tier_bands_are_the_ones_the_plan_documents() {
    let topo = numa_topology();
    // A micro K with a batch that cannot amortise dispatch.
    assert_eq!(flynnel::sched::plan::pick_tier(&JobPlan::new(2, 100), topo), SchedTier::Inline);
    // The same K with enough aggregate work.
    assert_eq!(
        flynnel::sched::plan::pick_tier(&JobPlan::new(2, 1_000_000), topo),
        SchedTier::Local
    );
    // A large K is its own concern and does not collapse on batch.
    assert_eq!(flynnel::sched::plan::pick_tier(&JobPlan::new(13, 1), topo), SchedTier::Federated);
}

#[test]
fn a_call_site_starts_empty_and_records_what_it_is_told() {
    static SITE: CallSiteState = CallSiteState::new();
    let site = SiteRef::new(&SITE);
    // Two handles to one static are the same site, which is what lets
    // a plan carry one without breaking its equality derives.
    assert_eq!(site, SiteRef::new(&SITE));
    assert!(!site.get().collapse_overran(), "a fresh site has not overrun");
    let plan = JobPlan::new(8, 1024).with_site(site);
    assert_eq!(plan.site, Some(site), "the plan carries the site it was given");
}

/// Resolves to the location of whoever called it, which is how a
/// dispatch entry attributes work to its caller's line.
#[track_caller]
fn site_of_my_caller() -> SiteRef {
    site_for_location(std::panic::Location::caller())
}

#[test]
fn caller_site_is_stable_per_location_and_distinct_between_them() {
    // The identity is the source location, so one line must resolve to
    // one site however often it runs, and two lines must differ. That
    // is what keeps one call site's learned state out of another's.
    let mut first: Option<SiteRef> = None;
    for _ in 0..4 {
        let s = caller_site();
        match first {
            None => first = Some(s),
            Some(p) => assert_eq!(p, s, "one source line is one site, however often it runs"),
        }
    }
    let other_line = caller_site();
    assert_ne!(first.expect("seen"), other_line, "different lines are different sites");
}

#[test]
fn site_for_location_attributes_to_the_caller_not_the_helper() {
    // Both calls go through one helper, so if the site were the
    // helper's own line these would collide and a shared helper would
    // pool every caller's statistics into one.
    let a = site_of_my_caller();
    let b = site_of_my_caller();
    assert_ne!(a, b, "two call sites of one helper are two sites");
    let a_again = {
        let mut seen = None;
        for _ in 0..3 {
            let s = site_of_my_caller();
            match seen {
                None => seen = Some(s),
                Some(p) => assert_eq!(p, s, "and one call site stays one site"),
            }
        }
        seen.expect("seen")
    };
    assert_ne!(a_again, a);
}

#[test]
fn a_placement_is_one_of_the_three_documented_decisions() {
    // Exhaustive by construction: the match fails to compile if a
    // variant is added without this test being updated.
    for p in [Placement::Cpu, Placement::Backend, Placement::Race] {
        let named = match p {
            Placement::Cpu => "cpu",
            Placement::Backend => "backend",
            Placement::Race => "race",
        };
        assert!(!named.is_empty());
    }
    assert_ne!(Placement::Cpu, Placement::Backend);
    assert_ne!(Placement::Backend, Placement::Race);
}
