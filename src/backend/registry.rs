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

use core::sync::atomic::{AtomicPtr, Ordering};
use std::hash::{Hash, Hasher};
use std::sync::{Arc, OnceLock};

use crate::backend::{Backend, BackendRef, CpuBackend, DispatchBackend};

/// Slot count of the backend table. One slot per distinct [`Backend`]
/// id, which is the CPU plus one per device per accelerator class, so
/// sixty-four leaves a multi-GPU host far from the edge. Registration
/// replaces rather than appends, so the population does not grow with
/// the number of hot-swaps.
const SLOTS: usize = 64;
const MASK: usize = SLOTS - 1;

/// One registered id and whatever is currently installed under it.
///
/// The id is written once, when the entry claims its slot. The
/// implementation behind it is replaceable, so it is reached through
/// a pointer a reader loads and a registrar stores.
struct Entry {
    id: Backend,
    current: AtomicPtr<BackendRef>,
}

/// Open-addressed, insert-once, never-removed table of leaked entries.
/// A non-null slot is a complete entry, so a reader needs one atomic
/// load per probe and no lock.
static TABLE: [AtomicPtr<Entry>; SLOTS] = [const { AtomicPtr::new(core::ptr::null_mut()) }; SLOTS];

/// Gate for the auto-registration of the CPU backend, so first access
/// from several threads installs it once.
static CPU_READY: OnceLock<()> = OnceLock::new();

fn key_of(id: &Backend) -> u64 {
    let mut h = std::hash::DefaultHasher::new();
    id.hash(&mut h);
    h.finish()
}

/// The entry for `id`, with the probe stopping at the first empty
/// slot, which is where an insert for this id would go.
fn find(id: &Backend) -> Option<&'static Entry> {
    let mut idx = (key_of(id) as usize) & MASK;
    for _ in 0..SLOTS {
        let p = TABLE[idx].load(Ordering::Acquire);
        if p.is_null() {
            return None;
        }
        // SAFETY: a non-null slot holds a leaked Entry that is never
        // freed, moved or rehashed.
        let entry = unsafe { &*p };
        if entry.id == *id {
            return Some(entry);
        }
        idx = (idx + 1) & MASK;
    }
    None
}

/// Install `b` under `id`, replacing whatever was there.
///
/// The replaced pointer is not freed. A reader that has loaded it is
/// about to clone through it, and there is no point at which that is
/// known to be finished, so the alternative to leaking is a use after
/// free. What leaks is one pointer-sized box per registration, and
/// registration is a startup act a process performs a handful of
/// times.
fn install(id: Backend, b: BackendRef) {
    let fresh = Box::into_raw(Box::new(b));
    if let Some(entry) = find(&id) {
        entry.current.store(fresh, Ordering::Release);
        return;
    }
    let mut prepared: Option<&'static Entry> = None;
    let mut idx = (key_of(&id) as usize) & MASK;
    for _ in 0..SLOTS {
        let p = TABLE[idx].load(Ordering::Acquire);
        if p.is_null() {
            let entry = *prepared.get_or_insert_with(|| {
                &*Box::leak(Box::new(Entry {
                    id,
                    current: AtomicPtr::new(fresh),
                }))
            });
            let node = entry as *const Entry as *mut Entry;
            match TABLE[idx].compare_exchange(
                core::ptr::null_mut(),
                node,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return,
                Err(taken) => {
                    // SAFETY: as in find.
                    let other = unsafe { &*taken };
                    if other.id == id {
                        other.current.store(fresh, Ordering::Release);
                        return;
                    }
                }
            }
        } else {
            // SAFETY: as in find.
            let other = unsafe { &*p };
            if other.id == id {
                other.current.store(fresh, Ordering::Release);
                return;
            }
        }
        idx = (idx + 1) & MASK;
    }
    panic!("backend table full at {SLOTS} distinct ids; {id:?} could not be registered");
}

fn ensure_cpu() {
    CPU_READY.get_or_init(|| {
        let cpu: Arc<dyn DispatchBackend> = Arc::new(CpuBackend::new());
        install(Backend::Cpu, cpu);
    });
}

/// Register a backend implementation with the process-global
/// registry. If a backend with the same [`Backend`] id is already
/// registered, this call REPLACES it. Replacement is the documented
/// hot-swap path for consumer crates that want to install a more
/// capable backend over the default.
pub fn register_backend(b: BackendRef) {
    ensure_cpu();
    let id = b.id();
    install(id, b);
}

/// Look up a backend by id. Returns `None` if no backend with that
/// exact id (including matching `device_id`) is registered.
pub fn backend_by_id(id: &Backend) -> Option<BackendRef> {
    ensure_cpu();
    let entry = find(id)?;
    let p = entry.current.load(Ordering::Acquire);
    // SAFETY: an entry's current pointer is set before the entry is
    // published and every value ever stored in it is leaked, so it is
    // non-null and its referent outlives this clone.
    Some(unsafe { &*p }.clone())
}

/// Snapshot every registered backend. Useful for telemetry and
/// for the `JobPlan::pick_backend` fallback path that picks the
/// best-available backend when no explicit hint is set.
///
/// Slot order carries no meaning, and a registration landing in an
/// earlier slot after the walk has passed it is not in the snapshot.
/// Nothing is ever removed, so a backend the walk does report is
/// really registered.
pub fn backends() -> Vec<BackendRef> {
    ensure_cpu();
    let mut out = Vec::new();
    for slot in TABLE.iter() {
        let p = slot.load(Ordering::Acquire);
        if p.is_null() {
            continue;
        }
        // SAFETY: as in find, and as in backend_by_id for the value.
        let entry = unsafe { &*p };
        let current = entry.current.load(Ordering::Acquire);
        out.push(unsafe { &*current }.clone());
    }
    out
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
