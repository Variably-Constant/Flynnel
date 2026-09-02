//! Adaptive backend migration: same AtomicU-tag pattern used for
//! K_gating and DispatchProfile, extended to backend selection
//! (CPU / CUDA / ROCm / Metal / TPU / ANE / WASM / SharedMemory
//! Worker / Custom).
//!
//! ## Why backend selection fits the adaptive pattern
//!
//! Backend is consulted ONCE per dispatch (at the entry point that
//! decides which `DispatchBackend` implementation to invoke), NOT
//! per push/pop on the deque hot path. So:
//!
//! - Per-op cost on the active deque: **zero** (no backend check)
//! - Per-dispatch cost: **one AtomicU32 Acquire-load** to read the
//!   active backend tag (~1 ns)
//! - Migration cost: **one AtomicU32 Release-store** (~1 ns)
//!
//! ## Registration vs activation
//!
//! Registration ([`crate::backend::register_backend`]) and
//! activation (this module) are separate concerns:
//!
//! - **Registration**: makes a backend AVAILABLE via the global
//!   registry. CPU auto-registers; CUDA / TPU / WASM are opt-in
//!   via Cargo features (consumer calls `register_backend` at
//!   startup once the runtime is initialized).
//! - **Activation**: marks one of the registered backends as the
//!   ACTIVE one for the next dispatch. Adaptive workload-shift
//!   signals flip the active backend; the dispatcher consumes the
//!   active tag at execute-time.
//!
//! Activation gracefully degrades when the requested backend is
//! not registered: [`resolve_active_backend`] returns the CPU
//! backend (always available) as the fallback.

#![allow(clippy::missing_errors_doc)]

use core::sync::atomic::{AtomicU32, Ordering};

use crate::backend::{Backend, BackendRef, backend_by_id, cpu_backend};

/// Encoded active-backend tag. Backend enum is variable-shaped
/// (some variants carry device_id), so we encode as:
/// - top 8 bits: variant tag (0=Cpu, 1=Cuda, 2=Rocm, 3=Metal,
///   4=Tpu, 5=Ane, 6=Wasm, 7=SharedMemoryWorker, 8=Custom)
/// - bottom 24 bits: device_id / backend_id / custom_id
const fn encode_backend(b: Backend) -> u32 {
    match b {
        Backend::Cpu => 0u32,
        Backend::Cuda { device_id } => (1u32 << 24) | (device_id & 0x00FF_FFFF),
        Backend::Rocm { device_id } => (2u32 << 24) | (device_id & 0x00FF_FFFF),
        Backend::Metal { device_id } => (3u32 << 24) | (device_id & 0x00FF_FFFF),
        Backend::Tpu { device_id } => (4u32 << 24) | (device_id & 0x00FF_FFFF),
        Backend::Ane => 5u32 << 24,
        Backend::Wasm { device_id } => (6u32 << 24) | (device_id & 0x00FF_FFFF),
        Backend::SharedMemoryWorker { backend_id } => {
            (7u32 << 24) | (backend_id & 0x00FF_FFFF)
        }
        Backend::Custom(id) => (8u32 << 24) | (id & 0x00FF_FFFF),
    }
}

fn decode_backend(tag: u32) -> Backend {
    let variant = (tag >> 24) & 0xFF;
    let device_id = tag & 0x00FF_FFFF;
    match variant {
        0 => Backend::Cpu,
        1 => Backend::Cuda { device_id },
        2 => Backend::Rocm { device_id },
        3 => Backend::Metal { device_id },
        4 => Backend::Tpu { device_id },
        5 => Backend::Ane,
        6 => Backend::Wasm { device_id },
        7 => Backend::SharedMemoryWorker { backend_id: device_id },
        _ => Backend::Custom(device_id),
    }
}

/// Global active-backend tag. Initial value: Cpu (always
/// available). Flipped by [`migrate_backend`].
static ACTIVE_BACKEND_TAG: AtomicU32 = AtomicU32::new(0);

/// Linkage confirmation marker. When the binary links this
/// module, `nm <bin> | grep __flynnel_marker` returns this
/// symbol, confirming the adaptive backend routing code path is
/// present in the build.
#[unsafe(no_mangle)]
pub static __flynnel_marker_adaptive_backend: u8 = 0;

/// Read the active backend id via one AtomicU32 Acquire-load.
#[inline]
pub fn active_backend_id() -> Backend {
    decode_backend(ACTIVE_BACKEND_TAG.load(Ordering::Acquire))
}

/// Resolve the active backend to a concrete [`BackendRef`] via
/// the global registry. Gracefully falls back to the CPU backend
/// when the active backend is not registered (the CPU backend is
/// always auto-registered).
///
/// Returns a tuple of `(resolved_backend, fell_back_to_cpu)` so
/// callers that care about the original request can observe the
/// fallback for telemetry.
pub fn resolve_active_backend() -> (BackendRef, bool) {
    let active = active_backend_id();
    match backend_by_id(&active) {
        Some(b) => (b, false),
        None => {
            // Cold path: requested backend not yet registered or
            // de-registered between migrate and dispatch. LLVM
            // lays out the hot Some(b) arm as fall-through.
            core::hint::cold_path();
            (cpu_backend(), true)
        }
    }
}

/// Migrate the global active backend via one AtomicU32 Release-
/// store. Subsequent dispatches that consult
/// [`active_backend_id`] / [`resolve_active_backend`] see the new
/// value. Per-op cost on the deque hot path: zero (backend is
/// consulted at execute-entry, not per push/pop).
///
/// The requested backend does NOT need to be registered at
/// migration time; subsequent dispatches gracefully fall back to
/// CPU via [`resolve_active_backend`] when the target is not
/// available. This lets the application optimistically set the
/// active backend ahead of registration (e.g., during async GPU
/// runtime initialization).
#[inline]
pub fn migrate_backend(backend: Backend) {
    ACTIVE_BACKEND_TAG.store(encode_backend(backend), Ordering::Release);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn restore_default() {
        migrate_backend(Backend::Cpu);
    }

    struct TestGuard;
    impl TestGuard {
        fn new() -> Self {
            restore_default();
            Self
        }
    }
    impl Drop for TestGuard {
        fn drop(&mut self) {
            restore_default();
        }
    }

    #[test]
    fn encode_decode_round_trip_all_variants() {
        let cases = [
            Backend::Cpu,
            Backend::Cuda { device_id: 0 },
            Backend::Cuda { device_id: 7 },
            Backend::Rocm { device_id: 1 },
            Backend::Metal { device_id: 2 },
            Backend::Tpu { device_id: 3 },
            Backend::Ane,
            Backend::Wasm { device_id: 4 },
            Backend::SharedMemoryWorker { backend_id: 5 },
            Backend::Custom(12345),
        ];
        for b in cases {
            let tag = encode_backend(b);
            let back = decode_backend(tag);
            assert_eq!(b, back, "round trip failed for {b:?} (tag={tag:#x})");
        }
    }

    #[test]
    fn default_active_is_cpu() {
        let _g = TestGuard::new();
        assert_eq!(active_backend_id(), Backend::Cpu);
    }

    #[test]
    fn migrate_changes_active() {
        let _g = TestGuard::new();
        migrate_backend(Backend::Cuda { device_id: 0 });
        assert_eq!(active_backend_id(), Backend::Cuda { device_id: 0 });
        migrate_backend(Backend::Tpu { device_id: 3 });
        assert_eq!(active_backend_id(), Backend::Tpu { device_id: 3 });
        migrate_backend(Backend::Cpu);
        assert_eq!(active_backend_id(), Backend::Cpu);
    }

    #[test]
    fn resolve_falls_back_to_cpu_when_target_unregistered() {
        let _g = TestGuard::new();
        // Custom backend with a random id that won't be registered.
        migrate_backend(Backend::Custom(0x00ABCDEF));
        let (resolved, fell_back) = resolve_active_backend();
        assert!(fell_back, "should fall back to CPU when Custom unregistered");
        assert_eq!(resolved.id(), Backend::Cpu);
    }

    #[test]
    fn resolve_returns_active_when_registered() {
        let _g = TestGuard::new();
        // CPU is always registered.
        let (resolved, fell_back) = resolve_active_backend();
        assert!(!fell_back);
        assert_eq!(resolved.id(), Backend::Cpu);
    }
}
