//! Process-global backend registry. Consumers call
//! [`register_backend`] at startup; routing helpers
//! (`JobPlan::pick_backend`, `sched::join_hybrid`) look up via
//! [`backend_by_id`]; debug / observability paths walk via
//! [`backends`].
//!
//! The registry stores `Arc<dyn DispatchBackend>` keyed by
//! [`Backend`]. Multiple instances of the same class are
//! distinguished by their `device_id` so a multi-GPU host can host
//! several CUDA backends.
//!
//! The CPU backend ([`crate::backend::cpu::CpuBackend`]) is
//! auto-registered on first access; consumers never have to
//! register it manually. [`cpu_backend`] returns the canonical
//! shared `Arc` for it.

use std::collections::HashMap;
use std::sync::{Arc, OnceLock, RwLock};

use crate::backend::{Backend, BackendRef, CpuBackend, DispatchBackend};

fn registry() -> &'static RwLock<HashMap<Backend, BackendRef>> {
    static CACHE: OnceLock<RwLock<HashMap<Backend, BackendRef>>> = OnceLock::new();
    CACHE.get_or_init(|| {
        let mut m: HashMap<Backend, BackendRef> = HashMap::new();
        let cpu: Arc<dyn DispatchBackend> = Arc::new(CpuBackend::new());
        m.insert(Backend::Cpu, cpu);
        RwLock::new(m)
    })
}

/// Register a backend implementation with the process-global
/// registry. If a backend with the same [`Backend`] id is already
/// registered, this call REPLACES it. Replacement is the documented
/// hot-swap path for consumer crates that want to install a more
/// capable backend over the default.
pub fn register_backend(b: BackendRef) {
    let id = b.id();
    if let Ok(mut guard) = registry().write() {
        guard.insert(id, b);
    }
}

/// Look up a backend by id. Returns `None` if no backend with that
/// exact id (including matching `device_id`) is registered.
pub fn backend_by_id(id: &Backend) -> Option<BackendRef> {
    registry().read().ok().and_then(|g| g.get(id).cloned())
}

/// Snapshot every registered backend. Useful for telemetry and
/// for the `JobPlan::pick_backend` fallback path that picks the
/// best-available backend when no explicit hint is set.
pub fn backends() -> Vec<BackendRef> {
    registry()
        .read()
        .map(|g| g.values().cloned().collect())
        .unwrap_or_default()
}

/// Canonical [`Arc`] for the always-available CPU backend.
/// Equivalent to `backend_by_id(&Backend::Cpu).unwrap()`, but
/// infallible: the CPU backend is auto-registered on first access
/// and cannot be removed.
pub fn cpu_backend() -> BackendRef {
    backend_by_id(&Backend::Cpu).expect("CPU backend is auto-registered")
}

/// Forces the registry to initialize so the CPU backend is present.
/// Most callers do not need this - any registry access auto-inits.
/// Exposed for explicit-init callers that want predictable startup
/// timing (e.g. registering several consumer backends at program
/// start and observing the registry state right after).
///
/// Returns true once the CPU backend is observable, which is always
/// after this call.
pub fn ensure_default_registered() -> bool {
    backend_by_id(&Backend::Cpu).is_some()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{BackendCapabilities, KernelArg, KernelHandle};

    struct StubBackend(Backend);
    impl DispatchBackend for StubBackend {
        fn id(&self) -> Backend {
            self.0
        }
        fn capabilities(&self) -> BackendCapabilities {
            BackendCapabilities {
                simt_width: 32,
                max_threads_in_flight: 1024,
                launch_latency_ns: 10_000,
                h2d_bw_bytes_per_sec: 25_000_000_000,
            }
        }
        fn dispatch_parallel_for(&self, _count: u32, _work: &(dyn Fn(u32) + Send + Sync)) {}
        fn dispatch_one(&self, _work: Box<dyn FnOnce() + Send>) {}
    }

    #[test]
    fn cpu_backend_auto_registers() {
        ensure_default_registered();
        let b = backend_by_id(&Backend::Cpu);
        assert!(b.is_some());
    }

    #[test]
    fn cpu_backend_helper_is_infallible() {
        let cpu = cpu_backend();
        assert_eq!(cpu.id(), Backend::Cpu);
    }

    #[test]
    fn register_then_lookup_round_trip() {
        let stub = Arc::new(StubBackend(Backend::Cuda { device_id: 7 }));
        register_backend(stub);
        let found = backend_by_id(&Backend::Cuda { device_id: 7 });
        assert!(found.is_some());
        assert_eq!(
            found.unwrap().id(),
            Backend::Cuda { device_id: 7 }
        );
    }

    #[test]
    fn distinct_device_ids_register_independently() {
        let a = Arc::new(StubBackend(Backend::Cuda { device_id: 100 }));
        let b = Arc::new(StubBackend(Backend::Cuda { device_id: 101 }));
        register_backend(a);
        register_backend(b);
        assert!(backend_by_id(&Backend::Cuda { device_id: 100 }).is_some());
        assert!(backend_by_id(&Backend::Cuda { device_id: 101 }).is_some());
    }

    #[test]
    fn missing_backend_lookup_returns_none() {
        let res = backend_by_id(&Backend::Custom(999_999));
        assert!(res.is_none());
    }

    #[test]
    fn backends_snapshot_includes_cpu() {
        let all = backends();
        assert!(all.iter().any(|b| b.id() == Backend::Cpu));
    }

    #[test]
    fn register_replaces_existing_id() {
        let a = Arc::new(StubBackend(Backend::Custom(7777)));
        register_backend(a);
        let b = Arc::new(StubBackend(Backend::Custom(7777)));
        register_backend(b);
        let found = backend_by_id(&Backend::Custom(7777)).unwrap();
        assert_eq!(found.id(), Backend::Custom(7777));
    }

    #[test]
    fn stub_backend_kernel_methods_default_to_not_supported() {
        let stub = StubBackend(Backend::Cuda { device_id: 0 });
        let result = stub.register_kernel("k", b"");
        assert!(result.is_err());
        let launch = stub.dispatch_kernel(KernelHandle(0), 1, &[KernelArg::I32(0)]);
        assert!(launch.is_err());
    }
}
