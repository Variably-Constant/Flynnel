//! The backend registry and the accelerator-op routing.
//!
//! Every consumer runs with no accelerator registered until it
//! registers one, so the contract that matters most here is graceful
//! degradation: an op with no bound kernel runs its CPU
//! implementation, a request for a backend nobody registered resolves
//! to the CPU one, and a kernel binding against an absent backend is
//! refused rather than accepted and silently never used. A registry
//! that accepted a binding it could not honour would leave the caller
//! believing work was accelerated while the CPU quietly did all of it.

use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use flynnel::sched::adaptive_backend::{
    active_backend_id, migrate_backend, resolve_active_backend,
};
use flynnel::{
    Backend, JobPlan, KernelArg, accel_op_name, accel_target, backend_by_id, backends,
    bind_accel_kernel, cpu_backend, dispatch_accel, register_accel_op,
};

/// The active backend is process-global, so the tests that move it run
/// one at a time and put it back.
static ACTIVE: Mutex<()> = Mutex::new(());

fn active() -> MutexGuard<'static, ()> {
    ACTIVE.lock().unwrap_or_else(|e| e.into_inner())
}

#[test]
fn the_cpu_backend_is_always_present() {
    // Everything else degrades to this one, so its absence would take
    // the whole dispatch path with it.
    let cpu = cpu_backend();
    assert_eq!(cpu.id(), Backend::Cpu, "the always-available backend identifies as such");
    assert!(
        backends().iter().any(|b| b.id() == Backend::Cpu),
        "and it appears in the registry listing"
    );
    assert!(backend_by_id(&Backend::Cpu).is_some(), "and resolves by id");
}

#[test]
fn a_backend_nobody_registered_does_not_resolve() {
    // Asking for one that is not there must answer no rather than
    // hand back something else wearing its name.
    assert!(
        backend_by_id(&Backend::Cuda { device_id: 0 }).is_none(),
        "no CUDA backend is registered in this test binary"
    );
}

#[test]
fn activating_an_absent_backend_falls_back_to_the_cpu() {
    let _g = active();
    let original = active_backend_id();

    migrate_backend(Backend::Cuda { device_id: 0 });
    assert_eq!(
        active_backend_id(),
        Backend::Cuda { device_id: 0 },
        "the request is recorded even when it cannot be honoured"
    );
    let (resolved, fell_back) = resolve_active_backend();
    assert!(fell_back, "and the fallback is reported rather than hidden");
    assert_eq!(
        resolved.id(),
        Backend::Cpu,
        "while the dispatch that follows goes to the CPU rather than failing"
    );

    migrate_backend(original);
    let (restored, fell_back_again) = resolve_active_backend();
    assert!(!fell_back_again, "the CPU backend is registered, so nothing falls back");
    assert_eq!(restored.id(), Backend::Cpu);
    assert_eq!(active_backend_id(), original, "and the knob is restorable");
}

#[test]
fn an_op_with_no_bound_kernel_runs_its_cpu_implementation() {
    let ran = Arc::new(AtomicUsize::new(0));
    let items = Arc::new(AtomicU32::new(0));
    let (r, it) = (Arc::clone(&ran), Arc::clone(&items));
    let op = register_accel_op("test_unbound", 4, move |count, _args| {
        r.fetch_add(1, Ordering::Relaxed);
        it.store(count, Ordering::Relaxed);
    });

    let plan = JobPlan::new(8, 256);
    let report = dispatch_accel(&plan, op, 256, &[KernelArg::U32(7)], &[KernelArg::U32(7)]);

    assert_eq!(ran.load(Ordering::Relaxed), 1, "the CPU implementation ran exactly once");
    assert_eq!(items.load(Ordering::Relaxed), 256, "and was handed the item count");
    assert!(report.cpu_ns.is_some(), "its time was measured");
    assert!(report.backend_ns.is_none(), "no kernel ran, so there is no kernel time");
    assert!(!report.fell_back, "nothing was attempted and failed; there was nothing to attempt");
    assert!(report.fallback_error.is_none());
}

#[test]
fn an_op_has_no_accelerator_target_until_a_kernel_is_bound() {
    let op = register_accel_op("test_no_target", 4, |_count, _args| {});
    let plan = JobPlan::new(8, 4096);
    assert!(
        accel_target(&plan, op).is_none(),
        "an op with no binding resolves to no accelerator"
    );
}

#[test]
fn binding_a_kernel_to_an_absent_backend_is_refused() {
    // The failure mode this prevents: a binding accepted for a backend
    // that does not exist, after which every dispatch quietly runs on
    // the CPU while the caller believes otherwise.
    let op = register_accel_op("test_bind_absent", 4, |_count, _args| {});
    let outcome = bind_accel_kernel(op, Backend::Cuda { device_id: 0 }, "entry", b"not ptx");
    assert!(outcome.is_err(), "a binding against an unregistered backend is refused");
}

#[test]
fn an_op_reports_the_name_it_was_registered_under() {
    let op = register_accel_op("test_named_op", 8, |_count, _args| {});
    assert_eq!(accel_op_name(op), "test_named_op");
    let other = register_accel_op("test_other_op", 8, |_count, _args| {});
    assert_eq!(accel_op_name(other), "test_other_op");
    assert_ne!(op, other, "two registrations are two distinct ids");
}

#[test]
fn dispatching_an_op_repeatedly_keeps_running_it() {
    // The learned model changes placement over calls; with no
    // accelerator bound, every one of those placements must still end
    // in the CPU implementation having run.
    let ran = Arc::new(AtomicUsize::new(0));
    let r = Arc::clone(&ran);
    let op = register_accel_op("test_repeat", 4, move |_count, _args| {
        r.fetch_add(1, Ordering::Relaxed);
    });
    let plan = JobPlan::new(8, 64);
    for _ in 0..32 {
        let report = dispatch_accel(&plan, op, 64, &[], &[]);
        assert!(report.cpu_ns.is_some(), "the CPU side ran on every dispatch");
    }
    assert_eq!(
        ran.load(Ordering::Relaxed),
        32,
        "once per dispatch, not zero and not twice"
    );
}

#[test]
fn a_zero_count_dispatch_still_runs_and_reports() {
    let ran = Arc::new(AtomicUsize::new(0));
    let r = Arc::clone(&ran);
    let op = register_accel_op("test_zero_count", 4, move |count, _args| {
        assert_eq!(count, 0, "the implementation sees the count it was given");
        r.fetch_add(1, Ordering::Relaxed);
    });
    let plan = JobPlan::new(8, 1);
    let report = dispatch_accel(&plan, op, 0, &[], &[]);
    assert_eq!(ran.load(Ordering::Relaxed), 1, "an empty dispatch still runs the op");
    assert!(report.cpu_ns.is_some());
}

#[test]
fn kernel_args_carry_their_payload_to_the_implementation() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let s = Arc::clone(&seen);
    let op = register_accel_op("test_args", 4, move |_count, args| {
        let mut g = s.lock().unwrap_or_else(|e| e.into_inner());
        for a in args {
            match a {
                KernelArg::U32(v) => g.push(format!("u32:{v}")),
                KernelArg::F64(v) => g.push(format!("f64:{v}")),
                KernelArg::HostSlice(b) => g.push(format!("bytes:{}", b.len())),
                _ => g.push("other".to_string()),
            }
        }
    });
    let plan = JobPlan::new(8, 4);
    let payload = [1u8, 2, 3, 4];
    let report = dispatch_accel(
        &plan,
        op,
        4,
        &[KernelArg::U32(9), KernelArg::F64(2.5), KernelArg::HostSlice(&payload)],
        &[],
    );
    assert!(report.cpu_ns.is_some());
    let got = seen.lock().unwrap_or_else(|e| e.into_inner()).clone();
    assert_eq!(got, vec!["u32:9", "f64:2.5", "bytes:4"], "arguments arrive in order and intact");
}
