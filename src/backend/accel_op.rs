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
//! launch-failure fallback run both sides sequentially (kernel
//! first), so the two impls must compute the same result and
//! tolerate running twice - the `hybrid_auto` contract. Every
//! failure lands on the CPU impl.

use core::sync::atomic::{AtomicPtr, AtomicU32, Ordering};
use std::sync::Arc;
use std::time::Instant;

use crate::backend::{Backend, BackendError, KernelArg, KernelHandle, backend_by_id};
use crate::sched::hazard::HazardDomain;
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
    ///
    /// Replaced whole rather than edited in place: binding happens at
    /// startup and reading happens on every dispatch, so the cost
    /// belongs on the writer. A reader follows the pointer under a
    /// hazard and the replaced list is freed once no reader holds it.
    kernels: AtomicPtr<Bindings>,
}

type Bindings = Vec<(Backend, KernelHandle)>;

/// Registered ops the table can hold. Ops are registered at startup
/// and never removed, and an [`AccelOpId`] IS the index, so there is
/// no hashing and no probing here.
const MAX_OPS: usize = 1024;

/// Threads that may route a dispatch, and replaced binding lists
/// awaiting a sweep.
const READERS: usize = 256;
const RETIRED: usize = 64;

static OPS: [AtomicPtr<AccelOp>; MAX_OPS] =
    [const { AtomicPtr::new(core::ptr::null_mut()) }; MAX_OPS];
static NEXT_OP: AtomicU32 = AtomicU32::new(0);
static BINDINGS: HazardDomain<Bindings, READERS, RETIRED> = HazardDomain::new();

fn binding_reader() -> usize {
    thread_local! {
        static READER: core::cell::Cell<usize> = const { core::cell::Cell::new(usize::MAX) };
    }
    READER.with(|cell| {
        let held = cell.get();
        if held != usize::MAX {
            return held;
        }
        let fresh = BINDINGS.claim_reader();
        cell.set(fresh);
        fresh
    })
}

fn op_by_id(op: AccelOpId) -> &'static AccelOp {
    let slot = OPS
        .get(op.0 as usize)
        .expect("AccelOpId not issued by register_accel_op");
    let published = slot.load(Ordering::Acquire);
    assert!(
        !published.is_null(),
        "AccelOpId {} was not issued by register_accel_op",
        op.0
    );
    // SAFETY: a published slot holds a leaked AccelOp that is never
    // freed or moved.
    unsafe { &*published }
}

/// Replace `op`'s binding list by applying `edit` to a copy, retrying
/// when a concurrent binding wins the swap.
fn update_bindings(op: &'static AccelOp, edit: impl Fn(&mut Bindings)) {
    loop {
        let current = op.kernels.load(Ordering::Acquire);
        // SAFETY: the list is published before the pointer and is
        // freed only once no reader holds it; the compare-exchange
        // below fails if it has been replaced meanwhile.
        let mut next: Bindings = unsafe { &*current }.clone();
        edit(&mut next);
        let fresh = Box::into_raw(Box::new(next));
        match op
            .kernels
            .compare_exchange(current, fresh, Ordering::AcqRel, Ordering::Acquire)
        {
            Ok(replaced) => {
                // SAFETY: the compare-exchange removed it from the only
                // source a reader can reach, and it is retired once.
                unsafe { BINDINGS.retire(replaced) };
                return;
            }
            Err(winner) => {
                // Another binding landed first, so the copy built here
                // is stale and the next pass rebuilds it from theirs.
                debug_assert!(!winner.is_null(), "a binding list is never null");
                // SAFETY: never published, so nothing else reaches it.
                drop(unsafe { Box::from_raw(fresh) });
            }
        }
    }
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
    let index = NEXT_OP.fetch_add(1, Ordering::AcqRel) as usize;
    assert!(
        index < MAX_OPS,
        "accel op table sized for {MAX_OPS} ops, {name:?} asked to be the {index}th"
    );
    let op: &'static AccelOp = Box::leak(Box::new(AccelOp {
        name: name.to_string(),
        bytes_per_item,
        cpu: Arc::new(cpu),
        kernels: AtomicPtr::new(Box::into_raw(Box::new(Bindings::new()))),
    }));
    // The op is complete before the pointer that publishes it, so a
    // reader that sees a non-null slot sees a whole op.
    OPS[index].store(op as *const AccelOp as *mut AccelOp, Ordering::Release);
    AccelOpId(index as u32)
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
    update_bindings(op, |bindings| {
        match bindings.iter_mut().find(|(b, _)| *b == backend) {
            Some(slot) => slot.1 = handle,
            None => bindings.push((backend, handle)),
        }
    });
}

/// The accelerator this op would route to right now, or `None` when
/// every dispatch runs on the CPU (no binding, or no bound backend
/// registered). Resolution order matches [`dispatch_accel`]: the
/// plan hint wins, then the process-global active backend, then the
/// first bound-and-registered backend.
pub fn accel_target(plan: &JobPlan, op: AccelOpId) -> Option<(Backend, KernelHandle)> {
    let op = op_by_id(op);
    let guard = BINDINGS.protect(binding_reader(), &op.kernels);
    let bound = match guard.get() {
        Some(bindings) => bindings,
        // The list is installed when the op is registered, so this is
        // unreachable rather than an empty binding set.
        None => return None,
    };
    let bound_and_registered = |b: Backend| -> Option<(Backend, KernelHandle)> {
        let handle = bound.iter().find(|(k, _)| *k == b).map(|(_, h)| *h)?;
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
    bound
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
#[derive(Debug, Clone)]
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
    /// The launch error behind `fell_back`, rendered with `Debug`.
    pub fallback_error: Option<String>,
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
            fallback_error: None,
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
            fallback_error: None,
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
                fallback_error: None,
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
                    fallback_error: None,
                }
            }
            Err(e) => {
                let cpu_ns = run_cpu(true);
                AccelReport {
                    placement: Placement::Cpu,
                    backend: Some(backend_id),
                    cpu_ns: Some(cpu_ns),
                    backend_ns: None,
                    gate_blocked: false,
                    fell_back: true,
                    fallback_error: Some(format!("{e:?}")),
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
                        fallback_error: None,
                    }
                }
                Err(e) => {
                    site.record_placement(count, Some(cpu_ns), None);
                    AccelReport {
                        placement: Placement::Cpu,
                        backend: Some(backend_id),
                        cpu_ns: Some(cpu_ns),
                        backend_ns: None,
                        gate_blocked: false,
                        fell_back: true,
                        fallback_error: Some(format!("{e:?}")),
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

/// One registered op, as [`registered_accel_ops`] answers it.
#[derive(Clone, Debug)]
pub struct RegisteredAccelOp {
    /// The id, usable with [`accel_target`] and [`dispatch_accel`].
    pub op: AccelOpId,
    /// The name it registered under.
    pub name: String,
    /// Bytes each item moves, which the gate reads to decide whether
    /// a transfer is worth making.
    pub bytes_per_item: u32,
    /// Every backend this op has a kernel bound on. Empty means every
    /// dispatch of it runs the CPU implementation.
    pub kernels: Vec<Backend>,
}

/// Every op registered in this process, in the order they registered.
///
/// The registry is private and an `AccelOpId` cannot be constructed
/// from a number, so without this a caller holding no id from
/// [`register_accel_op`] has no way to reach an op at all - which is
/// the position anything inspecting the process from outside is in,
/// including a binding.
///
/// It copies as it walks and holds nothing a dispatch might want. Not
/// on any dispatch path: the routing in [`dispatch_accel`] reaches an
/// op by index and never enumerates.
///
/// An op registered while the walk is past its index is not in the
/// result. Ops are never removed, so one that is reported is real.
pub fn registered_accel_ops() -> Vec<RegisteredAccelOp> {
    let reader = binding_reader();
    let mut out = Vec::new();
    for (index, slot) in OPS.iter().enumerate() {
        let published = slot.load(Ordering::Acquire);
        if published.is_null() {
            continue;
        }
        // SAFETY: as in op_by_id.
        let op = unsafe { &*published };
        let guard = BINDINGS.protect(reader, &op.kernels);
        let kernels = match guard.get() {
            Some(bindings) => bindings.iter().map(|(backend, _)| *backend).collect(),
            None => Vec::new(),
        };
        out.push(RegisteredAccelOp {
            op: AccelOpId(index as u32),
            name: op.name.clone(),
            bytes_per_item: op.bytes_per_item,
            kernels,
        });
    }
    out
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

    #[test]
    fn the_registry_lists_an_op_with_its_bytes_and_its_bindings() {
        // The registry is process-wide and every other test in this
        // file registers into it, so this asserts about its OWN op by
        // name rather than about the vector's length or its last
        // entry, either of which would depend on test order.
        let op = register_accel_op("listed_op", 12, |_n, _a| {});
        let listed = registered_accel_ops();

        let mine = listed
            .iter()
            .find(|r| r.name == "listed_op")
            .expect("an op that registered is in the registry");
        assert_eq!(mine.op, op, "the id listed is the id register handed back");
        assert_eq!(mine.bytes_per_item, 12);
        assert!(
            mine.kernels.is_empty(),
            "an op with nothing bound reports no kernels, which is what says every \
             dispatch of it runs the CPU implementation"
        );

        // And a binding shows up, so the column is not always empty.
        let stub = Arc::new(StubAccel {
            id: Backend::Custom(0x7005),
            caps: BackendCapabilities::cpu_defaults(),
            kernel_calls: AtomicU32::new(0),
            fail: false,
        });
        register_backend(Arc::clone(&stub) as _);
        bind_accel_kernel(op, Backend::Custom(0x7005), "k", b"").expect("stub binds");

        let after = registered_accel_ops();
        let mine = after
            .iter()
            .find(|r| r.name == "listed_op")
            .expect("still registered");
        assert_eq!(mine.kernels, vec![Backend::Custom(0x7005)]);
    }

    #[test]
    fn every_listed_id_resolves_to_the_name_it_was_listed_under() {
        // The id is an index into a private vector, so a listing that
        // paired the wrong id with a name would be undetectable from
        // outside and would send every later accel_target and
        // dispatch_accel at the wrong op.
        register_accel_op("paired_a", 1, |_n, _a| {});
        register_accel_op("paired_b", 2, |_n, _a| {});
        for row in registered_accel_ops() {
            assert_eq!(
                accel_op_name(row.op),
                row.name,
                "the id listed for {} resolves to a different op",
                row.name
            );
        }
    }
}
