//! Automatic CPU/accelerator routing for registered ops.
//!
//! A Rust closure cannot execute on a GPU/TPU, so transparent
//! offload of arbitrary closures is impossible; what CAN be
//! automatic is routing an op declared ONCE in both forms: a CPU
//! impl ([`register_accel_op`]) and a per-backend kernel
//! ([`bind_accel_kernel`]). Same id-crosses-not-code pattern as
//! `shared_mem::pass_registry`, applied at the device boundary.
//!
//! [`dispatch_accel`] decides in order:
//! 1. Target: `plan.backend_hint`, else the active-backend tag,
//!    else the first bound-and-registered backend; none -> CPU.
//! 2. Cost gate: with an authoritative per-item cost, skip the
//!    accelerator when est_total < [`LAUNCH_AMORTIZATION_FACTOR`]
//!    x launch_latency + H2D time for `count * bytes_per_item`.
//!    Classifier-default costs never fire the gate.
//! 3. Learned placement: `CallSiteState::choose_placement` EWMAs
//!    per call site per log2-size bucket - race cold, exploit
//!    warm, re-race on the reprobe cadence.
//!
//! `cpu_args` / `kernel_args` are separate views (host vs device
//! pointers); each side touches only its own. Race, reprobe, and
//! launch-failure fallback run BOTH sides sequentially (kernel
//! first), so the two impls must compute the same result and
//! tolerate running twice - the `hybrid_auto` contract. Every
//! failure lands on the CPU impl.

use std::sync::{Arc, OnceLock, RwLock};
use std::time::Instant;

use crate::backend::{Backend, BackendError, KernelArg, KernelHandle, backend_by_id};
use crate::sched::call_site::{Placement, caller_site};
use crate::sched::plan::JobPlan;

/// Multiple of the backend's reported launch latency that the
/// estimated total work must clear before the accelerator is
/// considered. The `>= 4x launch latency` rule of thumb documented
/// on [`crate::sched::hybrid::join_hybrid`], as code.
pub const LAUNCH_AMORTIZATION_FACTOR: u64 = 4;

/// Identifier for a registered accelerator-routable op. Returned by
/// [`register_accel_op`]; stable for the process lifetime.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct AccelOpId(u32);

/// CPU implementation of a registered op: invoked with the dispatch
/// `count` and the caller's `cpu_args` view.
type CpuImpl = Arc<dyn Fn(u32, &[KernelArg<'_>]) + Send + Sync>;

struct AccelOp {
    /// Diagnostic name; also the uniqueness key is NOT enforced,
    /// two registrations with one name are two distinct ops.
    name: String,
    /// Estimated host-to-device traffic per item in bytes, used by
    /// the static cost gate. Zero for device-resident ops.
    bytes_per_item: u32,
    cpu: CpuImpl,
    /// Per-backend kernel bindings in binding order. A `Vec` rather
    /// than a map so "first bound" is deterministic.
    kernels: RwLock<Vec<(Backend, KernelHandle)>>,
}

fn registry() -> &'static RwLock<Vec<Arc<AccelOp>>> {
    static REGISTRY: OnceLock<RwLock<Vec<Arc<AccelOp>>>> = OnceLock::new();
    REGISTRY.get_or_init(|| RwLock::new(Vec::new()))
}

fn op_by_id(op: AccelOpId) -> Arc<AccelOp> {
    registry()
        .read()
        .expect("accel op registry poisoned")
        .get(op.0 as usize)
        .cloned()
        .expect("AccelOpId not issued by register_accel_op")
}

/// Register an accelerator-routable op: a CPU implementation plus a
/// per-item H2D byte estimate for the cost gate. Kernel bindings
/// attach afterwards via [`bind_accel_kernel`] /
/// [`bind_accel_kernel_handle`]; until one is bound, every dispatch
/// of this op runs the CPU implementation.
pub fn register_accel_op<F>(name: &str, bytes_per_item: u32, cpu: F) -> AccelOpId
where
    F: Fn(u32, &[KernelArg<'_>]) + Send + Sync + 'static,
{
    let mut guard = registry().write().expect("accel op registry poisoned");
    let id = AccelOpId(guard.len() as u32);
    guard.push(Arc::new(AccelOp {
        name: name.to_string(),
        bytes_per_item,
        cpu: Arc::new(cpu),
        kernels: RwLock::new(Vec::new()),
    }));
    id
}

/// Compile-and-bind: register `source` (backend-native kernel text,
/// e.g. PTX) under `entry` with the backend registered as `backend`,
/// and bind the resulting handle to `op`. Returns
/// [`BackendError::DeviceUnavailable`] when no such backend is
/// registered.
pub fn bind_accel_kernel(
    op: AccelOpId,
    backend: Backend,
    entry: &str,
    source: &[u8],
) -> Result<(), BackendError> {
    let be = backend_by_id(&backend).ok_or(BackendError::DeviceUnavailable(backend))?;
    let handle = be.register_kernel(entry, source)?;
    bind_accel_kernel_handle(op, backend, handle);
    Ok(())
}

/// Bind an already-registered kernel handle to `op` for `backend`.
/// For kernels the caller registered directly (or a custom backend
/// whose handles come from elsewhere). Rebinding the same backend
/// replaces the previous handle.
pub fn bind_accel_kernel_handle(op: AccelOpId, backend: Backend, handle: KernelHandle) {
    let op = op_by_id(op);
    let mut guard = op.kernels.write().expect("accel op kernel table poisoned");
    if let Some(slot) = guard.iter_mut().find(|(b, _)| *b == backend) {
        slot.1 = handle;
    } else {
        guard.push((backend, handle));
    }
}

/// The accelerator this op would route to right now, or `None` when
/// every dispatch runs on the CPU (no binding, or no bound backend
/// registered). Resolution order matches [`dispatch_accel`]: the
/// plan hint wins, then the process-global active backend, then the
/// first bound-and-registered backend.
pub fn accel_target(plan: &JobPlan, op: AccelOpId) -> Option<(Backend, KernelHandle)> {
    let op = op_by_id(op);
    let guard = op.kernels.read().expect("accel op kernel table poisoned");
    let bound_and_registered = |b: Backend| -> Option<(Backend, KernelHandle)> {
        let handle = guard.iter().find(|(k, _)| *k == b).map(|(_, h)| *h)?;
        backend_by_id(&b)?;
        Some((b, handle))
    };
    if let Some(hint) = plan.backend_hint
        && let Some(found) = bound_and_registered(hint)
    {
        return Some(found);
    }
    let active = crate::sched::adaptive_backend::active_backend_id();
    if active != Backend::Cpu
        && let Some(found) = bound_and_registered(active)
    {
        return Some(found);
    }
    guard
        .iter()
        .find(|(b, _)| backend_by_id(b).is_some())
        .map(|(b, h)| (*b, *h))
}

/// Static cost-gate verdict for one dispatch. `None` when the plan
/// carries no authoritative per-item cost (classifier defaults are
/// hints, not measurements); `Some(false)` when the estimated total
/// work cannot amortize the accelerator's launch latency plus the
/// H2D transfer for `count * bytes_per_item`.
pub(crate) fn cost_gate_pass(
    plan: &JobPlan,
    caps: &crate::backend::BackendCapabilities,
    count: u32,
    bytes_per_item: u32,
) -> Option<bool> {
    if !plan.estimated_per_item_ns_explicit {
        return None;
    }
    let per_item = plan.estimated_per_item_ns? as u64;
    let est_total_ns = per_item.saturating_mul(count as u64);
    let launch_ns =
        LAUNCH_AMORTIZATION_FACTOR.saturating_mul(caps.launch_latency_ns as u64);
    let bytes = (count as u64).saturating_mul(bytes_per_item as u64);
    let h2d_ns = bytes
        .saturating_mul(1_000_000_000)
        .checked_div(caps.h2d_bw_bytes_per_sec)
        .unwrap_or(0);
    Some(est_total_ns >= launch_ns.saturating_add(h2d_ns))
}

/// Outcome of one [`dispatch_accel`] call, for telemetry and tests.
#[derive(Debug, Clone, Copy)]
pub struct AccelReport {
    /// The placement the learned model chose. [`Placement::Race`]
    /// means both sides ran and both samples were recorded.
    pub placement: Placement,
    /// The accelerator that was resolved for this dispatch, whether
    /// or not it ended up running. `None` when every path was CPU.
    pub backend: Option<Backend>,
    /// Wall time of the CPU implementation when it ran.
    pub cpu_ns: Option<u64>,
    /// End-to-end wall time of the kernel dispatch when it ran and
    /// succeeded (queueing + execution via
    /// [`crate::backend::DispatchBackend::dispatch_kernel_sync`]).
    pub backend_ns: Option<u64>,
    /// The static cost gate rejected the accelerator for this
    /// dispatch; the CPU implementation ran without a placement
    /// sample being recorded against the backend.
    pub gate_blocked: bool,
    /// A kernel launch was attempted and failed; the CPU
    /// implementation covered the dispatch.
    pub fell_back: bool,
}

/// Route one dispatch of a registered op to the CPU implementation
/// or a bound accelerator kernel, per the three-step decision in the
/// module docs. `cpu_args` is handed to the CPU implementation;
/// `kernel_args` to [`crate::backend::DispatchBackend::dispatch_kernel_sync`]. The
/// call blocks until whichever side ran has completed.
///
/// The per-call-site learned state keys on the caller's source
/// location (`#[track_caller]`), or on the plan's explicit site when
/// [`JobPlan::with_site`](crate::sched::JobPlan::with_site) attached
/// one.
///
/// # Panics
///
/// Panics on an `op` id that [`register_accel_op`] never issued.
#[track_caller]
pub fn dispatch_accel(
    plan: &JobPlan,
    op: AccelOpId,
    count: u32,
    cpu_args: &[KernelArg<'_>],
    kernel_args: &[KernelArg<'_>],
) -> AccelReport {
    let op_arc = op_by_id(op);
    let site_ref = plan.site.unwrap_or_else(caller_site);
    let site = site_ref.get();

    let run_cpu = |record: bool| -> u64 {
        let t0 = Instant::now();
        (op_arc.cpu)(count, cpu_args);
        let ns = t0.elapsed().as_nanos() as u64;
        if record {
            site.record_placement(count, Some(ns), None);
        }
        ns
    };

    let Some((backend_id, handle)) = accel_target(plan, op) else {
        let cpu_ns = run_cpu(true);
        return AccelReport {
            placement: Placement::Cpu,
            backend: None,
            cpu_ns: Some(cpu_ns),
            backend_ns: None,
            gate_blocked: false,
            fell_back: false,
        };
    };
    let backend = backend_by_id(&backend_id).expect("accel_target checked registration");

    if cost_gate_pass(plan, &backend.capabilities(), count, op_arc.bytes_per_item)
        == Some(false)
    {
        let cpu_ns = run_cpu(true);
        return AccelReport {
            placement: Placement::Cpu,
            backend: Some(backend_id),
            cpu_ns: Some(cpu_ns),
            backend_ns: None,
            gate_blocked: true,
            fell_back: false,
        };
    }

    let run_kernel = || -> Result<u64, BackendError> {
        let t0 = Instant::now();
        backend.dispatch_kernel_sync(handle, count, kernel_args)?;
        Ok(t0.elapsed().as_nanos() as u64)
    };

    match site.choose_placement(count) {
        Placement::Cpu => {
            let cpu_ns = run_cpu(true);
            AccelReport {
                placement: Placement::Cpu,
                backend: Some(backend_id),
                cpu_ns: Some(cpu_ns),
                backend_ns: None,
                gate_blocked: false,
                fell_back: false,
            }
        }
        Placement::Backend => match run_kernel() {
            Ok(ns) => {
                site.record_placement(count, None, Some(ns));
                AccelReport {
                    placement: Placement::Backend,
                    backend: Some(backend_id),
                    cpu_ns: None,
                    backend_ns: Some(ns),
                    gate_blocked: false,
                    fell_back: false,
                }
            }
            Err(_) => {
                let cpu_ns = run_cpu(true);
                AccelReport {
                    placement: Placement::Cpu,
                    backend: Some(backend_id),
                    cpu_ns: Some(cpu_ns),
                    backend_ns: None,
                    gate_blocked: false,
                    fell_back: true,
                }
            }
        },
        Placement::Race => {
            // Sequential on purpose: the two sides may share logical
            // state through their argument views, and sequencing
            // (kernel, then CPU) keeps the race sound under the
            // idempotency contract without demanding disjoint
            // buffers from every op.
            let kernel_outcome = run_kernel();
            let cpu_ns = run_cpu(false);
            match kernel_outcome {
                Ok(dev_ns) => {
                    site.record_placement(count, Some(cpu_ns), Some(dev_ns));
                    AccelReport {
                        placement: Placement::Race,
                        backend: Some(backend_id),
                        cpu_ns: Some(cpu_ns),
                        backend_ns: Some(dev_ns),
                        gate_blocked: false,
                        fell_back: false,
                    }
                }
                Err(_) => {
                    site.record_placement(count, Some(cpu_ns), None);
                    AccelReport {
                        placement: Placement::Cpu,
                        backend: Some(backend_id),
                        cpu_ns: Some(cpu_ns),
                        backend_ns: None,
                        gate_blocked: false,
                        fell_back: true,
                    }
                }
            }
        }
    }
}

/// Diagnostic name of a registered op.
pub fn accel_op_name(op: AccelOpId) -> String {
    op_by_id(op).name.clone()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{BackendCapabilities, DispatchBackend, register_backend};
    use std::sync::atomic::{AtomicU32, Ordering};

    /// Stub accelerator: counts kernel dispatches, optional
    /// simulated failure.
    struct StubAccel {
        id: Backend,
        caps: BackendCapabilities,
        kernel_calls: AtomicU32,
        fail: bool,
    }

    impl DispatchBackend for StubAccel {
        fn id(&self) -> Backend {
            self.id
        }
        fn capabilities(&self) -> BackendCapabilities {
            self.caps
        }
        fn dispatch_parallel_for(&self, _count: u32, _work: &(dyn Fn(u32) + Send + Sync)) {}
        fn dispatch_one(&self, work: Box<dyn FnOnce() + Send>) {
            work();
        }
        fn register_kernel(
            &self,
            _name: &str,
            _source: &[u8],
        ) -> Result<KernelHandle, BackendError> {
            Ok(KernelHandle(0xACCE1))
        }
        fn dispatch_kernel(
            &self,
            _handle: KernelHandle,
            _count: u32,
            _args: &[KernelArg<'_>],
        ) -> Result<(), BackendError> {
            if self.fail {
                return Err(BackendError::Launch("stub failure".into()));
            }
            self.kernel_calls.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
    }

    fn fast_caps() -> BackendCapabilities {
        BackendCapabilities {
            simt_width: 32,
            max_threads_in_flight: 4096,
            launch_latency_ns: 1_000,
            h2d_bw_bytes_per_sec: 10_000_000_000,
        }
    }

    #[test]
    fn unbound_op_runs_cpu() {
        let cpu_calls = Arc::new(AtomicU32::new(0));
        let c = Arc::clone(&cpu_calls);
        let op = register_accel_op("unbound", 4, move |_n, _a| {
            c.fetch_add(1, Ordering::SeqCst);
        });
        let plan = JobPlan::bare(0, 64);
        let report = dispatch_accel(&plan, op, 64, &[], &[]);
        assert_eq!(report.placement, Placement::Cpu);
        assert!(report.backend.is_none());
        assert_eq!(cpu_calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn cold_bucket_races_then_warm_exploits_faster_backend() {
        let stub = Arc::new(StubAccel {
            id: Backend::Custom(0x7001),
            caps: fast_caps(),
            kernel_calls: AtomicU32::new(0),
            fail: false,
        });
        register_backend(Arc::clone(&stub) as _);
        let cpu_calls = Arc::new(AtomicU32::new(0));
        let c = Arc::clone(&cpu_calls);
        let op = register_accel_op("race_then_exploit", 4, move |_n, _a| {
            c.fetch_add(1, Ordering::SeqCst);
            // The slow side: the stub kernel returns immediately, so
            // the EWMA must learn Backend as the winner.
            std::thread::sleep(std::time::Duration::from_millis(3));
        });
        bind_accel_kernel(op, Backend::Custom(0x7001), "k", b"")
            .expect("stub registers any kernel");
        let plan = JobPlan::bare(0, 4096).with_backend(Backend::Custom(0x7001));

        let first = dispatch_accel(&plan, op, 4096, &[], &[]);
        assert_eq!(first.placement, Placement::Race, "cold bucket races");
        assert_eq!(cpu_calls.load(Ordering::SeqCst), 1);
        assert_eq!(stub.kernel_calls.load(Ordering::SeqCst), 1);

        for _ in 0..8 {
            let r = dispatch_accel(&plan, op, 4096, &[], &[]);
            assert_eq!(r.placement, Placement::Backend, "warm bucket exploits");
        }
        assert_eq!(cpu_calls.load(Ordering::SeqCst), 1, "CPU stays cold");
        assert_eq!(stub.kernel_calls.load(Ordering::SeqCst), 9);
    }

    #[test]
    fn gate_blocks_work_below_launch_amortization() {
        let stub = Arc::new(StubAccel {
            id: Backend::Custom(0x7002),
            caps: BackendCapabilities {
                launch_latency_ns: 100_000,
                ..fast_caps()
            },
            kernel_calls: AtomicU32::new(0),
            fail: false,
        });
        register_backend(Arc::clone(&stub) as _);
        let cpu_calls = Arc::new(AtomicU32::new(0));
        let c = Arc::clone(&cpu_calls);
        let op = register_accel_op("gated", 4, move |_n, _a| {
            c.fetch_add(1, Ordering::SeqCst);
        });
        bind_accel_kernel(op, Backend::Custom(0x7002), "k", b"").expect("stub binds");
        // 100 items at an authoritative 10 ns each = 1 us total,
        // against a 400 us launch-amortization floor.
        let plan = JobPlan::bare(0, 100)
            .with_backend(Backend::Custom(0x7002))
            .with_estimated_per_item_ns(10);
        let r = dispatch_accel(&plan, op, 100, &[], &[]);
        assert!(r.gate_blocked, "gate must reject sub-breakeven work");
        assert_eq!(r.placement, Placement::Cpu);
        assert_eq!(stub.kernel_calls.load(Ordering::SeqCst), 0);
        assert_eq!(cpu_calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn active_backend_tag_steers_hintless_dispatch() {
        let stub = Arc::new(StubAccel {
            id: Backend::Custom(0x7004),
            caps: fast_caps(),
            kernel_calls: AtomicU32::new(0),
            fail: false,
        });
        register_backend(Arc::clone(&stub) as _);
        let op = register_accel_op("tag_steered", 4, |_n, _a| {});
        bind_accel_kernel(op, Backend::Custom(0x7004), "k", b"").expect("stub binds");
        let hint_less = JobPlan::bare(0, 4096);

        // The migrate_backend Release-store is the whole steering
        // surface: same tag flip, same visibility contract as the
        // other adaptive axes. Restored to Cpu before the test
        // ends so parallel tests observe the default.
        crate::sched::adaptive_backend::migrate_backend(Backend::Custom(0x7004));
        let steered = accel_target(&hint_less, op);
        crate::sched::adaptive_backend::migrate_backend(Backend::Cpu);
        assert_eq!(
            steered.map(|(b, _)| b),
            Some(Backend::Custom(0x7004)),
            "hint-less resolution must honor the active-backend tag",
        );

        // A hint still wins over the tag, and the first-binding
        // fallback still resolves when the tag names Cpu.
        let hinted = JobPlan::bare(0, 4096).with_backend(Backend::Custom(0x7004));
        assert_eq!(
            accel_target(&hinted, op).map(|(b, _)| b),
            Some(Backend::Custom(0x7004)),
        );
        assert_eq!(
            accel_target(&hint_less, op).map(|(b, _)| b),
            Some(Backend::Custom(0x7004)),
            "first bound-and-registered backend resolves under a Cpu tag",
        );
    }

    #[test]
    fn binding_to_unregistered_backend_stays_cpu() {
        let cpu_calls = Arc::new(AtomicU32::new(0));
        let c = Arc::clone(&cpu_calls);
        let op = register_accel_op("ghost_backend", 4, move |_n, _a| {
            c.fetch_add(1, Ordering::SeqCst);
        });
        bind_accel_kernel_handle(op, Backend::Custom(0x7BAD), KernelHandle(1));
        let plan = JobPlan::bare(0, 64);
        let r = dispatch_accel(&plan, op, 64, &[], &[]);
        assert_eq!(r.placement, Placement::Cpu);
        assert!(r.backend.is_none(), "unregistered binding is no target");
        assert_eq!(cpu_calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn kernel_failure_falls_back_to_cpu() {
        let stub = Arc::new(StubAccel {
            id: Backend::Custom(0x7003),
            caps: fast_caps(),
            kernel_calls: AtomicU32::new(0),
            fail: true,
        });
        register_backend(Arc::clone(&stub) as _);
        let cpu_calls = Arc::new(AtomicU32::new(0));
        let c = Arc::clone(&cpu_calls);
        let op = register_accel_op("flaky", 4, move |_n, _a| {
            c.fetch_add(1, Ordering::SeqCst);
        });
        bind_accel_kernel(op, Backend::Custom(0x7003), "k", b"").expect("stub binds");
        let plan = JobPlan::bare(0, 512).with_backend(Backend::Custom(0x7003));
        let r = dispatch_accel(&plan, op, 512, &[], &[]);
        assert!(r.fell_back, "launch failure must be covered by CPU");
        assert_eq!(r.placement, Placement::Cpu);
        assert_eq!(cpu_calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn cost_gate_math() {
        let plan = JobPlan::bare(0, 1000).with_estimated_per_item_ns(1000);
        let caps = fast_caps();
        // 1000 items * 1000 ns = 1 ms >> 4 us launch + tiny H2D.
        assert_eq!(cost_gate_pass(&plan, &caps, 1000, 4), Some(true));
        // Non-authoritative plan: no verdict.
        let hint_less = JobPlan::bare(0, 1000);
        assert_eq!(cost_gate_pass(&hint_less, &caps, 1000, 4), None);
        // Huge per-item H2D swamps the estimate.
        let heavy_bytes = cost_gate_pass(&plan, &caps, 1000, u32::MAX);
        assert_eq!(heavy_bytes, Some(false));
    }

    #[test]
    fn accel_op_name_round_trips() {
        let op = register_accel_op("named_op", 0, |_n, _a| {});
        assert_eq!(accel_op_name(op), "named_op");
    }
}
