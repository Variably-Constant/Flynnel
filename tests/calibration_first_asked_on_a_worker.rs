//! A process whose first query of the host dispatch profile comes from a
//! pool worker whose peers have parked. The dispatch cost it ends up with
//! must be a dispatch: a job handed to another thread and a latch waited
//! on, not the worker's own inline pass, which pops the half it just
//! pushed.
//!
//! The module's chunk runner puts a query there: it starts its dispatch
//! from an outside caller's join, so the dispatch runs on a worker, and
//! whichever of its paths first needs the profile asks on that worker.
//! The join here asks in the same position directly.
//!
//! A test binary of its own, because the profile is measured once per
//! process by its first query, and any other test that queried first
//! would take the measurement from the calling thread instead.
//!
//! Its assertion compares two timings, so it holds only in an optimized
//! build: in a debug build a worker's inline pass costs about what a
//! dispatch does, and the two cannot be told apart. The gate runs it
//! under the release-test profile.

use std::time::Duration;

use flynnel::sched::par_iter::{
    calibrate_host_dispatch, host_dispatch_profile, measured_collapse_threshold_ns,
};
use flynnel::{JobPlan, LeafShape};

#[test]
fn a_profile_first_asked_for_on_a_worker_whose_peers_parked_is_a_dispatch() {
    // A calibration directory of this test's own, so no stored record
    // serves and the query below has to measure.
    let dir = std::env::temp_dir().join(format!("flynnel-cal-first-on-worker-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("this test's calibration directory can be made");
    // SAFETY: set before this process's first dispatch and first
    // profile query, while no other thread of the test runs.
    unsafe {
        std::env::set_var("FLYNNEL_CALIBRATION_DIR", &dir);
        std::env::remove_var("FLYNNEL_HOST_PROFILE_NS");
    }

    let workers = flynnel::sched::arena::global_local_arena().local_worker_count();
    // Long past any spin window, so every worker has parked, as they have
    // in a process whose pool sat idle before its first dispatch.
    std::thread::sleep(Duration::from_millis(500));
    let before = measured_collapse_threshold_ns();

    // The chunk runner's outer plan: an explicit leaf shape with a batch
    // of eight, which the pool always takes. An outside caller runs none
    // of its join's work, so the query runs on a worker.
    let onto_a_worker = JobPlan::new(0, 8).with_leaf_shape(LeafShape::PortCompute);
    let (on_worker, ()) = flynnel::join(&onto_a_worker, host_dispatch_profile, || ());
    // Drawn again from outside the pool, which is what the figure
    // describes. This measures rather than reading back: the first draw
    // is the only record in this test's directory, and one draw alone is
    // stored provisional and not served.
    let from_outside = calibrate_host_dispatch();
    println!(
        "FIRST_ON_PARKED_WORKER workers={workers} measured_before={before:?} dispatch={} \
         collapse={} wake={} outside_dispatch={} outside_collapse={} outside_wake={}",
        on_worker.dispatch_cost_ns,
        on_worker.collapse_threshold_ns,
        on_worker.jec_wake_threshold_ns,
        from_outside.dispatch_cost_ns,
        from_outside.collapse_threshold_ns,
        from_outside.jec_wake_threshold_ns
    );

    assert_eq!(before, None, "something queried the profile before the join");
    if workers < 2 {
        println!("FIRST_ON_PARKED_WORKER_UNREACHED the pool has {workers} worker, so no join is handed to another");
        return;
    }
    // A worker's inline pass read 100 ns on a 24-thread Windows host,
    // whose outside draws read 4300 to 4800 in a loaded test process, and
    // 170 to 181 ns on a 16-vCPU Linux guest against 1300 to 2200: a
    // factor of 7 to 48. Idle draws of one host spread by about 15 per
    // cent (1300 to 1500 ns over 48 draws on the Windows host), so half
    // the outside draw separates the two on both hosts with room either
    // side.
    assert!(
        on_worker.dispatch_cost_ns.saturating_mul(2) >= from_outside.dispatch_cost_ns,
        "the first query, on a worker, drew a dispatch cost of {} ns against {} ns drawn \
         from outside the pool, so the worker timed its own inline pass",
        on_worker.dispatch_cost_ns,
        from_outside.dispatch_cost_ns
    );
}
