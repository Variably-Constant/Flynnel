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

/// CPUs in the process mask as the platform reports it back, or `None`
/// where the read failed or is not implemented.
///
/// Read after narrowing so a failure says which half broke: a mask that
/// did not take, or a mask that took and a width that did not follow.
/// Without it both arrive as the same assertion.
#[cfg(target_os = "linux")]
fn mask_cpu_count() -> Option<usize> {
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        if libc::sched_getaffinity(0, size_of::<libc::cpu_set_t>(), &mut set) != 0 {
            return None;
        }
        Some(
            (0..libc::CPU_SETSIZE as usize)
                .filter(|&cpu| libc::CPU_ISSET(cpu, &set))
                .count(),
        )
    }
}

#[cfg(target_os = "freebsd")]
fn mask_cpu_count() -> Option<usize> {
    unsafe {
        let mut set: libc::cpuset_t = std::mem::zeroed();
        if libc::cpuset_getaffinity(
            libc::CPU_LEVEL_WHICH,
            libc::CPU_WHICH_PID,
            -1,
            size_of::<libc::cpuset_t>(),
            &mut set,
        ) != 0
        {
            return None;
        }
        Some(
            (0..libc::CPU_SETSIZE as usize)
                .filter(|&cpu| libc::CPU_ISSET(cpu, &set))
                .count(),
        )
    }
}

#[cfg(windows)]
fn mask_cpu_count() -> Option<usize> {
    let mut process_mask: usize = 0;
    let mut system_mask: usize = 0;
    let ok = unsafe {
        GetProcessAffinityMask(GetCurrentProcess(), &mut process_mask, &mut system_mask)
    };
    (ok != 0).then(|| process_mask.count_ones() as usize)
}

#[cfg(not(any(target_os = "linux", target_os = "freebsd", windows)))]
fn mask_cpu_count() -> Option<usize> {
    None
}

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
    fn GetProcessAffinityMask(
        process: isize,
        process_mask: *mut usize,
        system_mask: *mut usize,
    ) -> i32;
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
    let mask_narrowed = mask_cpu_count();
    let width_narrowed = width_settling_to(NARROW_TO);
    let workers_narrowed = plan.resolved_workers();

    // Restored before the assertions, so a failure does not leave the
    // process pinned for whatever runs next.
    let restored = set_affinity(&all);
    let width_widened = width_settling_to(full);

    // The mask first: a width that did not follow a mask that never took
    // is a different defect from one that ignored a mask that did.
    assert_eq!(
        mask_narrowed,
        Some(NARROW_TO),
        "the platform accepted the narrowing call and then reported \
         {mask_narrowed:?} cpus in the mask, so the mask itself did not take"
    );
    assert_eq!(
        width_narrowed, NARROW_TO,
        "the process mask holds {NARROW_TO} cpus and allowed_parallelism \
         read {width_narrowed}, so available_parallelism does not honour \
         the process affinity mask on this platform and the lever cannot \
         see a narrowing here"
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
