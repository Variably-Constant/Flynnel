//! The plan's width follows a change to the process affinity mask.
//!
//! `JobPlan::resolved_workers` caps by the CPUs the process may use at
//! that moment, through `sched::host_width::allowed_parallelism`. The
//! arena is spawned once and its threads outlive a change to the mask,
//! so a host that narrows after startup would otherwise leave the plan
//! chunking for threads that cannot reach a core.
//!
//! One test in this binary, deliberately. The affinity mask is
//! process-wide, so a second test running beside this one would be
//! narrowed by it and timed through a mask it did not set.
//!
//! An unsuitable host fails rather than returning early: a test that
//! skips itself reports the same green as one that ran.

use std::time::{Duration, Instant};

use flynnel::sched::host_width::allowed_parallelism;
use flynnel::{CallSiteState, JobPlan, SiteRef};

static SITE: CallSiteState = CallSiteState::new();

/// CPUs the mask is narrowed to.
const NARROW_TO: usize = 2;

#[cfg(target_os = "linux")]
fn set_affinity(cpus: &[usize]) -> bool {
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_ZERO(&mut set);
        for &cpu in cpus {
            libc::CPU_SET(cpu, &mut set);
        }
        libc::sched_setaffinity(0, size_of::<libc::cpu_set_t>(), &set) == 0
    }
}

#[cfg(target_os = "freebsd")]
fn set_affinity(cpus: &[usize]) -> bool {
    unsafe {
        let mut set: libc::cpuset_t = std::mem::zeroed();
        libc::CPU_ZERO(&mut set);
        for &cpu in cpus {
            libc::CPU_SET(cpu, &mut set);
        }
        libc::cpuset_setaffinity(
            libc::CPU_LEVEL_WHICH,
            libc::CPU_WHICH_PID,
            -1,
            size_of::<libc::cpuset_t>(),
            &set,
        ) == 0
    }
}

// Declared here rather than taken from a binding crate: three symbols
// from kernel32 against one more dependency in the tree.
#[cfg(windows)]
unsafe extern "system" {
    fn GetCurrentProcess() -> isize;
    fn SetProcessAffinityMask(process: isize, mask: usize) -> i32;
}

#[cfg(windows)]
fn set_affinity(cpus: &[usize]) -> bool {
    let mut mask: usize = 0;
    for &cpu in cpus {
        mask |= 1usize << cpu;
    }
    unsafe { SetProcessAffinityMask(GetCurrentProcess(), mask) != 0 }
}

#[cfg(not(any(target_os = "linux", target_os = "freebsd", windows)))]
fn set_affinity(_cpus: &[usize]) -> bool {
    false
}

/// The width once it reads `target`, or whatever it reads at the
/// deadline. Polls rather than sleeping the re-read cadence, so this
/// file carries no copy of that interval.
fn width_settling_to(target: usize) -> usize {
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        let seen = allowed_parallelism();
        if seen == target || Instant::now() >= deadline {
            return seen;
        }
        std::thread::sleep(Duration::from_millis(25));
    }
}

#[test]
fn the_plan_follows_a_narrowed_process_mask() {
    // Read through a OnceLock, so it has to be set before the first
    // call that resolves the switch.
    unsafe {
        std::env::set_var("FLYNNEL_LEVER_ALLOWED_WIDTH", "1");
    }

    let full = allowed_parallelism();
    assert!(
        full >= NARROW_TO * 2,
        "this test narrows to {NARROW_TO} cpus and needs a host allowing \
         at least {}; this one allows {full}",
        NARROW_TO * 2
    );

    // Start the arena at full width, so what follows measures a pool
    // that outlived the change rather than one sized after it.
    let plan = JobPlan::new(0, 1 << 16).with_site(SiteRef::new(&SITE));
    let workers_full = plan.resolved_workers();

    let all: Vec<usize> = (0..full).collect();
    let narrow: Vec<usize> = (0..NARROW_TO).collect();

    assert!(
        set_affinity(&narrow),
        "could not narrow the process affinity mask on this platform"
    );
    let width_narrowed = width_settling_to(NARROW_TO);
    let workers_narrowed = plan.resolved_workers();

    // Restored before the assertions, so a failure does not leave the
    // process pinned for whatever runs next.
    let restored = set_affinity(&all);
    let width_widened = width_settling_to(full);

    assert_eq!(
        width_narrowed, NARROW_TO,
        "allowed_parallelism read {width_narrowed} under a {NARROW_TO}-cpu mask"
    );
    assert!(
        workers_narrowed <= NARROW_TO,
        "plan resolved {workers_narrowed} workers under a {NARROW_TO}-cpu mask, \
         having resolved {workers_full} at full width"
    );
    assert!(restored, "could not restore the process affinity mask");
    assert_eq!(
        width_widened, full,
        "allowed_parallelism read {width_widened} after the mask widened \
         back to {full}"
    );
}
