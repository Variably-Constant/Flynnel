//! E2E verification that flipping the adaptive-tag atomics changes
//! downstream JobPlan behavior on the very next call.
//!
//! Closes the read-the-code vs run-the-code gap from the audit:
//! the infrastructure is wired (atomic Release on migrate, atomic
//! Acquire on read), but until this example ran the live binary and
//! observed the plan fields changing, "trust the adaptive system"
//! was a code-review claim only.
//!
//! Run:
//! ```sh
//! cargo run --example migrate_observability --release
//! ```
//!
//! Asserts (every print is checked, panic on mismatch):
//!   1. migrate_dispatch_profile(X) -> active_dispatch_profile() == X
//!      on the very next call (process-global atomic Release/Acquire).
//!   2. JobPlan::set_profile(k, n, active_dispatch_profile()) yields
//!      plan fields that match the active profile's derived knobs:
//!      - Streaming     -> use_smt=false, oversubscription_log2=1
//!      - LatencyBound  -> use_smt=true,  oversubscription_log2=2
//!      - MemoryBound   -> use_smt=true,  oversubscription_log2=1
//!      - PortBound     -> use_smt=false, oversubscription_log2=1
//!   3. After migrate -> read -> build-plan, an actual reduce_chunks
//!      call completes successfully with the migrated plan.
//!   4. cooperative_routing migration is similarly observable.
//!   5. variant_routing migration is similarly observable.

use flynnel::dispatch_profile::DispatchProfile;
use flynnel::sched::JobPlan;
use flynnel::sched::adaptive_cooperative::{
    CooperativeRouting, active_cooperative_routing, migrate_cooperative_routing,
};
use flynnel::sched::adaptive_profile::{
    WorkloadClass, active_dispatch_profile, active_workload_class,
    migrate_dispatch_profile, migrate_workload_class, reset_auto_classify_state,
    tick_auto_classify,
};
use flynnel::sched::adaptive_variant_routing::{
    VariantRouting, active_variant_routing, cpuid_default_routing, migrate_variant_routing,
};
use flynnel::sched::par_iter::reduce_chunks;
use flynnel::sched::split_observer::{record_leaf_time_ns, reset_leaf_stats};

fn assert_profile_round_trip(target: DispatchProfile) {
    migrate_dispatch_profile(target);
    let observed = active_dispatch_profile();
    assert_eq!(
        observed, target,
        "migrate({target:?}) -> active() must return {target:?} on next call; got {observed:?}",
    );
    println!("    [ok] migrate({target:?}) -> active() reports {observed:?}");
}

fn assert_plan_matches_profile(
    target: DispatchProfile,
    expected_smt: bool,
    expected_oversub: u8,
) {
    migrate_dispatch_profile(target);
    let plan = JobPlan::set_profile(6, 1024 * 1024, active_dispatch_profile());
    assert_eq!(
        plan.use_smt, expected_smt,
        "{target:?}: plan.use_smt = {} (expected {expected_smt})",
        plan.use_smt
    );
    assert_eq!(
        plan.oversubscription_log2,
        Some(expected_oversub),
        "{target:?}: plan.oversubscription_log2 = {:?} (expected Some({expected_oversub}))",
        plan.oversubscription_log2,
    );
    println!(
        "    [ok] {target:?}: plan.use_smt={} oversub_log2={:?}",
        plan.use_smt, plan.oversubscription_log2,
    );
}

fn assert_reduce_chunks_executes_with_active_profile(target: DispatchProfile) {
    migrate_dispatch_profile(target);
    let plan = JobPlan::set_profile(6, 100_000, active_dispatch_profile());
    let input: Vec<u64> = (0..100_000u64).collect();
    let parallel: u64 = reduce_chunks(
        &plan,
        &input,
        || 0u64,
        |acc, chunk| acc + chunk.iter().sum::<u64>(),
        |a, b| a + b,
    );
    let serial: u64 = input.iter().sum();
    assert_eq!(
        parallel, serial,
        "{target:?}: reduce_chunks returned {parallel}, expected {serial}",
    );
    println!(
        "    [ok] {target:?}: reduce_chunks({{n=100k, sum}}) = {parallel} (matches serial)",
    );
}

fn assert_cooperative_routing_round_trip(target: CooperativeRouting) {
    migrate_cooperative_routing(target);
    let observed = active_cooperative_routing();
    assert_eq!(
        observed, target,
        "migrate_cooperative({target:?}) -> active() must return {target:?}; got {observed:?}",
    );
    println!("    [ok] migrate_cooperative({target:?}) -> active() reports {observed:?}");
}

fn assert_variant_routing_round_trip(target: VariantRouting) {
    migrate_variant_routing(target);
    let observed = active_variant_routing();
    // `Auto` is documented as deferring to the CPUID-resolved
    // default: active_variant_routing() returns cpuid_default_routing()
    // when the underlying tag is Auto. The assertion respects that.
    let expected = if target == VariantRouting::Auto {
        cpuid_default_routing()
    } else {
        target
    };
    assert_eq!(
        observed, expected,
        "migrate_variant({target:?}) -> active() must return {expected:?}; got {observed:?}",
    );
    println!(
        "    [ok] migrate_variant({target:?}) -> active() reports {observed:?} (expected {expected:?})"
    );
}

fn main() {
    println!("=== migrate_observability: E2E verification of adaptive-tag flow ===\n");

    println!("[1] migrate_dispatch_profile -> active_dispatch_profile round-trip");
    for p in [
        DispatchProfile::Streaming,
        DispatchProfile::LatencyBound,
        DispatchProfile::MemoryBound,
        DispatchProfile::PortBound,
    ] {
        assert_profile_round_trip(p);
    }

    println!("\n[2] JobPlan::set_profile derives knobs from active profile");
    // Expected knob derivations from src/dispatch_profile.rs:
    //   Streaming    -> SMT off, oversub_log2=1
    //   LatencyBound -> SMT on,  oversub_log2=2
    //   MemoryBound  -> SMT on,  oversub_log2=1
    //   PortBound    -> SMT off, oversub_log2=1
    assert_plan_matches_profile(DispatchProfile::Streaming, false, 1);
    assert_plan_matches_profile(DispatchProfile::LatencyBound, true, 2);
    assert_plan_matches_profile(DispatchProfile::MemoryBound, true, 1);
    assert_plan_matches_profile(DispatchProfile::PortBound, false, 1);

    println!("\n[3] reduce_chunks executes correctly under each migrated profile");
    for p in [
        DispatchProfile::Streaming,
        DispatchProfile::LatencyBound,
        DispatchProfile::MemoryBound,
        DispatchProfile::PortBound,
    ] {
        assert_reduce_chunks_executes_with_active_profile(p);
    }

    println!("\n[4] migrate_cooperative_routing round-trip");
    for r in [
        CooperativeRouting::ForceDeque,
        CooperativeRouting::ForceMailbox,
        CooperativeRouting::ForceTree,
        CooperativeRouting::Auto,
    ] {
        assert_cooperative_routing_round_trip(r);
    }

    println!("\n[5] migrate_variant_routing round-trip");
    for r in [
        VariantRouting::Default,
        VariantRouting::ComputeBatchAdaptive,
        VariantRouting::Auto,
    ] {
        assert_variant_routing_round_trip(r);
    }

    println!(
        "\n[6] OBSERVER-DRIVEN migration: tick_auto_classify ingests leaf-time stats\n    and migrates the active WorkloadClass automatically without any direct\n    migrate_workload_class() call. This is the closing-loop adaptive path."
    );
    assert_observer_drives_migration_to_latency_bound();
    assert_observer_drives_migration_to_fine_grain();

    println!("\n=== All E2E adaptive-tag observability assertions passed ===");
}

/// Feed the global leaf-stats accumulator per-leaf timing samples that
/// look LIKE a LatencyBound workload (mean_ns ~ 1500 ns, well above
/// the 500-ns LatencyBound threshold in `classify_observed`), then
/// call `tick_auto_classify()` and assert that the active class
/// migrated AUTOMATICALLY based on what the observer SAW, not what
/// the test directly set.
///
/// `active_workload_class()` is derived from
/// `active_dispatch_profile()` (the migration target), so the
/// observable class is LatencyBound when the observer migrated into
/// the LatencyBound DispatchProfile.
fn assert_observer_drives_migration_to_latency_bound() {
    // Start at PortBound (the observable identity of both
    // FineGrain and PortBound, since they share DispatchProfile).
    migrate_workload_class(WorkloadClass::PortBound);
    reset_leaf_stats();
    reset_auto_classify_state();
    assert_eq!(
        active_workload_class(),
        WorkloadClass::PortBound,
        "test setup: active class must start at PortBound"
    );

    // Feed 128 HIGH-VARIANCE samples (mean ~1500 ns).
    // `classify_observed` keys on BOTH mean and cv^2:
    //   mean_ns >= 500 AND cv2_per_mille >= 500 -> LatencyBound
    //   mean_ns >= 500 AND cv2_per_mille <  500 -> MemoryBound/Streaming
    // Uniform high-cost samples (all 1500 ns) -> cv2=0 -> Streaming.
    // Long-dep-chain workloads have HIGH per-leaf variance
    // (some leaves chase faster than others), so the simulated
    // signal mixes 500 ns and 2500 ns samples (mean 1500, var huge).
    for i in 0..128 {
        let ns = if i % 2 == 0 { 500 } else { 2500 };
        record_leaf_time_ns(ns);
    }

    // Closing-loop observer tick. Production path: `spawn_observer()`
    // ticks this in the background; here we call it inline so the
    // example is single-threaded + deterministic.
    tick_auto_classify();

    let migrated = active_workload_class();
    assert_eq!(
        migrated,
        WorkloadClass::LatencyBound,
        "OBSERVER-DRIVEN: after feeding 128 leaf samples @ 1500 ns and one tick, active class must have AUTO-migrated to LatencyBound; got {migrated:?}",
    );
    println!(
        "    [ok] observer saw 128 leaves @ 1500 ns -> auto-migrated PortBound -> LatencyBound"
    );
}

/// Symmetric down-migration: feed FineGrain-shaped samples (~20 ns
/// per leaf). FineGrain -> PortBound via the shared DispatchProfile,
/// so the observable migration is LatencyBound -> PortBound. Proves
/// the closing-loop responds to changing workload shape both up AND
/// down, not just monotonically.
fn assert_observer_drives_migration_to_fine_grain() {
    // Setup: start at LatencyBound (where the previous test left
    // the global tag).
    migrate_workload_class(WorkloadClass::LatencyBound);
    reset_leaf_stats();
    reset_auto_classify_state();
    assert_eq!(
        active_workload_class(),
        WorkloadClass::LatencyBound,
        "test setup: active class must start at LatencyBound for the down-migration test"
    );

    // Feed 128 samples of ~20 ns. classify_observed maps mean_ns
    // < 50 to FineGrain; LatencyBound -> FineGrain is bucket
    // distance 3 -> fast-adapt fires on one tick. FineGrain and
    // PortBound share DispatchProfile, so the observable post-tick
    // class is PortBound.
    for _ in 0..128 {
        record_leaf_time_ns(20);
    }
    tick_auto_classify();

    let migrated = active_workload_class();
    assert_eq!(
        migrated,
        WorkloadClass::PortBound,
        "OBSERVER-DRIVEN: after feeding 128 leaf samples @ 20 ns and one tick, observer must have AUTO-migrated to FineGrain (observable as PortBound via shared DispatchProfile); got {migrated:?}",
    );
    println!(
        "    [ok] observer saw 128 leaves @ 20 ns -> auto-migrated LatencyBound -> FineGrain (observable as PortBound)"
    );
}
