//! Backend dispatch fabric: extends Flynnel from CPU-only
//! work-stealing into a generic dispatcher that can route work to
//! GPUs (CUDA / ROCm / Metal), TPUs (Google / Coral / Apple Neural
//! Engine), or any other accelerator class.
//!
//! Flynnel core owns the fabric: the [`Backend`] taxonomy enum,
//! the [`DispatchBackend`] trait, [`BackendCapabilities`],
//! [`registry`], [`detect`], the always-available [`CpuBackend`],
//! and four feature-gated reference backends (`CudaBackend` /
//! `TpuJaxBackend` / `WasmBackend` /
//! `SharedMemoryChaseLevBackend`). Consumer crates ship concrete
//! kernel runtimes: pre-compiled PTX plus a [`DispatchBackend`]
//! impl attached via [`registry::register_backend`];
//! [`crate::sched::JobPlan::pick_backend`] routes work via
//! [`crate::sched::JobPlan::backend_hint`].
//!
//! This module supplies the SIMT axis
//! ([`DispatchBackend::dispatch_parallel_for`]) and the MIMT axis
//! ([`crate::sched::hybrid::join_hybrid`]) of the extended Flynn
//! table on [`crate`].
//!
//! [`detect`] probes runtimes by dlopen / device-file check, never
//! by linking an SDK: a build on a GPU-less host compiles and runs
//! with the detectors returning `false`. [`detect::detect_all`]
//! lists every backend found (always including [`Backend::Cpu`]).

pub mod accel_op;
pub mod cpu;
pub mod detect;
pub mod registry;

#[cfg(feature = "cuda-reference")]
pub mod cuda;

#[cfg(feature = "tpu-jax-reference")]
pub mod tpu_jax;

#[cfg(feature = "wasm-reference")]
pub mod wasm;

#[cfg(feature = "shared-memory-worker-reference")]
pub mod shared_mem;

use std::sync::Arc;

pub use cpu::CpuBackend;
pub use registry::{
    backend_by_id, backends, cpu_backend, ensure_default_registered, register_backend,
};

/// Taxonomy of dispatch targets. Each variant identifies a class of
/// hardware Flynnel can route work to. `Custom(u32)` is reserved
/// for consumer-defined targets the upstream taxonomy doesn't yet
/// cover (FPGA accelerators, custom AI ASICs, novel research
/// silicon).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Backend {
    /// Always-available default; routed through the in-crate work-
    /// stealing arena.
    Cpu,
    /// NVIDIA SIMT runtime. `device_id` selects which GPU on
    /// multi-GPU hosts (0 for the primary).
    Cuda {
        /// Index of the NVIDIA GPU to dispatch on (0 = primary).
        device_id: u32,
    },
    /// AMD ROCm SIMT runtime (HIP).
    Rocm {
        /// Index of the AMD GPU to dispatch on (0 = primary).
        device_id: u32,
    },
    /// Apple Metal compute runtime.
    Metal {
        /// Index of the Metal device to dispatch on (0 = primary).
        device_id: u32,
    },
    /// Google TPU or Coral Edge TPU runtime.
    Tpu {
        /// Index of the TPU device to dispatch on (0 = primary).
        device_id: u32,
    },
    /// Apple Neural Engine (M-series only).
    Ane,
    /// WebAssembly runtime (wasmtime by default in the reference
    /// impl). Executes registered `.wasm` modules in a sandbox.
    /// Single-threaded scalar execution; NOT SIMT. Used for
    /// portable kernels that need to run inside browser-class
    /// hosts or sandboxed serverless runtimes.
    Wasm {
        /// Index of the WASM engine instance (0 = primary). The
        /// shape mirrors the GPU variants for consistency even
        /// though wasmtime does not have a hardware-device notion.
        device_id: u32,
    },
    /// Peer worker process(es) connected over a memory-mapped
    /// Chase-Lev work-stealing deque + memory-mapped latch arena.
    /// `backend_id` selects which deque/arena pair to attach to
    /// when multiple worker pools coexist on the same host.
    /// Dispatch payload is `(closure_id, args)`; each peer process
    /// pre-registers `closure_id -> handler` in its local
    /// `shared_mem::pass_registry` at startup. Single-threaded per
    /// peer; pool fan-out happens by attaching N peers to the same
    /// deque. NOT SIMT.
    SharedMemoryWorker {
        /// Identifies which deque/arena pair to attach to (0 = primary).
        backend_id: u32,
    },
    /// Consumer-defined custom backend.
    Custom(u32),
}

impl Backend {
    /// Returns a short stable identifier used for logging /
    /// telemetry.
    pub fn name(&self) -> &'static str {
        match self {
            Backend::Cpu => "cpu",
            Backend::Cuda { .. } => "cuda",
            Backend::Rocm { .. } => "rocm",
            Backend::Metal { .. } => "metal",
            Backend::Tpu { .. } => "tpu",
            Backend::Ane => "ane",
            Backend::Wasm { .. } => "wasm",
            Backend::SharedMemoryWorker { .. } => "shared-mem",
            Backend::Custom(_) => "custom",
        }
    }

    /// True if this variant identifies a SIMT (Single Instruction,
    /// Multiple Threads) execution model. CPU, ANE, and WASM are
    /// not SIMT; CUDA / ROCm / Metal / TPU are.
    pub fn is_simt(&self) -> bool {
        matches!(
            self,
            Backend::Cuda { .. }
                | Backend::Rocm { .. }
                | Backend::Metal { .. }
                | Backend::Tpu { .. }
        )
    }
}

/// Descriptor reported by each backend so [`crate::sched::JobPlan`]
/// can make routing decisions (small jobs stay on CPU; large
/// data-parallel jobs go to the GPU). All fields are nominal: a
/// host without a GPU still returns sensible CPU values.
#[derive(Copy, Clone, Debug)]
pub struct BackendCapabilities {
    /// Lanes that execute in lockstep within one launch. 1 for
    /// scalar CPU, 32 for an NVIDIA warp, 64 for an AMD wave, 1024
    /// for a TPU core lane.
    pub simt_width: u32,
    /// Upper bound on threads (CPU) or work-items (GPU) the backend
    /// can hold in flight simultaneously. Used by the breakeven
    /// heuristic to decide whether a job is large enough to amortize
    /// the launch cost.
    pub max_threads_in_flight: u32,
    /// Wall-clock cost of a single empty dispatch in nanoseconds.
    /// CPU = ~100 ns (deque push), CUDA = ~10 us (driver round-
    /// trip), PCIe-class backends ~10-50 us.
    pub launch_latency_ns: u32,
    /// Host-to-device sustained throughput in bytes/second. Zero
    /// for the CPU backend (no copy required). Used by the routing
    /// policy to amortize H2D copy cost for big operand sets.
    pub h2d_bw_bytes_per_sec: u64,
}

impl BackendCapabilities {
    /// Conservative CPU defaults: 1-wide, work-stealing pool sized
    /// to host logical-thread count, ~100 ns launch.
    pub fn cpu_defaults() -> Self {
        let threads = std::thread::available_parallelism()
            .map(std::num::NonZeroUsize::get)
            .unwrap_or(1) as u32;
        Self {
            simt_width: 1,
            max_threads_in_flight: threads,
            launch_latency_ns: 100,
            h2d_bw_bytes_per_sec: 0,
        }
    }
}

/// Opaque handle a backend returns when a consumer registers a
/// pre-built kernel via [`DispatchBackend::register_kernel`]. The
/// consumer keeps the handle and passes it back to
/// [`DispatchBackend::dispatch_kernel`] to launch.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct KernelHandle(pub u64);

/// Single argument passed to a kernel via
/// [`DispatchBackend::dispatch_kernel`]. The backend interprets
/// each variant per its native ABI; backends that do not support a
/// given variant return [`BackendError::NotSupported`].
#[derive(Copy, Clone, Debug)]
pub enum KernelArg<'a> {
    /// 32-bit signed integer scalar (literal `i32`).
    I32(i32),
    /// 64-bit signed integer scalar.
    I64(i64),
    /// 32-bit unsigned scalar.
    U32(u32),
    /// 64-bit unsigned scalar.
    U64(u64),
    /// 32-bit float scalar.
    F32(f32),
    /// 64-bit float scalar.
    F64(f64),
    /// Raw device-side pointer the backend allocated and made
    /// known to the consumer (e.g. via a backend-specific
    /// allocator API).
    DevicePtr(usize),
    /// Host-side byte slice the backend should copy to device
    /// memory as part of the launch (synchronous H2D copy).
    HostSlice(&'a [u8]),
}

/// Error variants returned from [`DispatchBackend`] methods that
/// can fail. Routing helpers ([`crate::sched::JobPlan::pick_backend`])
/// fall back to the CPU backend when a hinted backend returns one
/// of these.
#[derive(Debug, Clone)]
pub enum BackendError {
    /// The backend does not implement the requested operation
    /// (e.g. calling `dispatch_kernel` on the CPU backend).
    NotSupported,
    /// The hinted device is offline or the runtime cannot be loaded.
    DeviceUnavailable(Backend),
    /// Kernel registration failed; payload is a human-readable
    /// compile / parse / link diagnostic.
    KernelCompile(String),
    /// Kernel launch failed at runtime; payload describes the
    /// driver-reported error.
    Launch(String),
    /// Memory operation (alloc, copy, free) failed; payload
    /// describes the cause.
    Memory(String),
}

impl std::fmt::Display for BackendError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BackendError::NotSupported => f.write_str("backend operation not supported"),
            BackendError::DeviceUnavailable(b) => {
                write!(f, "backend device unavailable: {}", b.name())
            }
            BackendError::KernelCompile(msg) => write!(f, "kernel compile failed: {msg}"),
            BackendError::Launch(msg) => write!(f, "kernel launch failed: {msg}"),
            BackendError::Memory(msg) => write!(f, "backend memory error: {msg}"),
        }
    }
}

impl std::error::Error for BackendError {}

/// Object-safe trait every dispatch backend implements. CPU
/// backends typically implement only [`Self::dispatch_parallel_for`]
/// and [`Self::dispatch_one`] (kernel methods return
/// [`BackendError::NotSupported`] by default); GPU / TPU backends
/// implement all four.
///
/// All methods take `&self` so a single [`Arc<dyn DispatchBackend>`]
/// can be shared across worker threads and stored in the global
/// [`registry`].
pub trait DispatchBackend: Send + Sync + 'static {
    /// Identifies this backend instance. Multiple instances of the
    /// same class (e.g. two NVIDIA GPUs) report distinct ids via
    /// the `device_id` field.
    fn id(&self) -> Backend;

    /// Returns this backend's capability descriptor. Cached on the
    /// backend (probed once at construction).
    fn capabilities(&self) -> BackendCapabilities;

    /// SIMT-shaped parallel-for: invoke `work(i)` for `i` in
    /// `0..count`. CPU backends translate to the existing par_iter
    /// arena; GPU backends launch a kernel with `count` work-items
    /// each running the closure (closure shape constrains to
    /// CPU-runnable bodies; for GPU codegen consumers use the
    /// [`Self::dispatch_kernel`] handle path instead).
    fn dispatch_parallel_for(&self, count: u32, work: &(dyn Fn(u32) + Send + Sync));

    /// Single-shot fire-and-forget closure. CPU backends dispatch
    /// via the work-stealing pool; GPU backends typically wrap the
    /// closure in a host-side thread (for backend bookkeeping work)
    /// since GPU kernels cannot run arbitrary Rust closures.
    fn dispatch_one(&self, work: Box<dyn FnOnce() + Send>);

    /// Register a pre-built kernel. `source` is backend-specific
    /// (PTX text for CUDA, SPIR-V bytes for Vulkan / Metal, etc.).
    /// Returns an opaque [`KernelHandle`] the consumer passes to
    /// [`Self::dispatch_kernel`].
    ///
    /// Default impl returns [`BackendError::NotSupported`]; CPU
    /// backends inherit it.
    fn register_kernel(
        &self,
        _name: &str,
        _source: &[u8],
    ) -> Result<KernelHandle, BackendError> {
        Err(BackendError::NotSupported)
    }

    /// Launch a previously-registered kernel with `count` work-items
    /// and the provided arguments. Blocks until the launch is
    /// queued (not until the kernel finishes); concrete backends
    /// expose their own synchronization primitives on their typed
    /// handle for callers that need to wait for completion.
    ///
    /// Default impl returns [`BackendError::NotSupported`]; CPU
    /// backends inherit it.
    fn dispatch_kernel(
        &self,
        _handle: KernelHandle,
        _count: u32,
        _args: &[KernelArg<'_>],
    ) -> Result<(), BackendError> {
        Err(BackendError::NotSupported)
    }

    /// Launch a registered kernel and block until it has COMPLETED,
    /// not merely queued: after `Ok`, the kernel's writes are
    /// host-visible. What [`accel_op::dispatch_accel`] times.
    /// Default forwards to [`Self::dispatch_kernel`] (correct for
    /// inherently synchronous backends); asynchronous-queue
    /// backends (CUDA) override with launch + stream synchronize.
    fn dispatch_kernel_sync(
        &self,
        handle: KernelHandle,
        count: u32,
        args: &[KernelArg<'_>],
    ) -> Result<(), BackendError> {
        self.dispatch_kernel(handle, count, args)
    }
}

/// Convenience alias for the boxed-trait-object form most callers
/// use.
pub type BackendRef = Arc<dyn DispatchBackend>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_names_are_stable() {
        assert_eq!(Backend::Cpu.name(), "cpu");
        assert_eq!(Backend::Cuda { device_id: 0 }.name(), "cuda");
        assert_eq!(Backend::Rocm { device_id: 0 }.name(), "rocm");
        assert_eq!(Backend::Metal { device_id: 0 }.name(), "metal");
        assert_eq!(Backend::Tpu { device_id: 0 }.name(), "tpu");
        assert_eq!(Backend::Ane.name(), "ane");
        assert_eq!(Backend::Wasm { device_id: 0 }.name(), "wasm");
        assert_eq!(
            Backend::SharedMemoryWorker { backend_id: 0 }.name(),
            "shared-mem"
        );
        assert_eq!(Backend::Custom(42).name(), "custom");
    }

    #[test]
    fn simt_classification_matches_taxonomy() {
        assert!(!Backend::Cpu.is_simt());
        assert!(!Backend::Ane.is_simt());
        assert!(Backend::Cuda { device_id: 0 }.is_simt());
        assert!(Backend::Rocm { device_id: 0 }.is_simt());
        assert!(Backend::Metal { device_id: 0 }.is_simt());
        assert!(Backend::Tpu { device_id: 0 }.is_simt());
        assert!(!Backend::Wasm { device_id: 0 }.is_simt());
        assert!(!Backend::SharedMemoryWorker { backend_id: 0 }.is_simt());
        assert!(!Backend::Custom(0).is_simt());
    }

    #[test]
    fn cpu_capabilities_report_host_thread_count() {
        let caps = BackendCapabilities::cpu_defaults();
        assert_eq!(caps.simt_width, 1);
        assert!(caps.max_threads_in_flight >= 1);
        assert_eq!(caps.h2d_bw_bytes_per_sec, 0);
        assert!(caps.launch_latency_ns > 0);
    }

    #[test]
    fn backend_error_display_is_non_empty() {
        let errors = [
            BackendError::NotSupported,
            BackendError::DeviceUnavailable(Backend::Cuda { device_id: 0 }),
            BackendError::KernelCompile("syntax".into()),
            BackendError::Launch("oom".into()),
            BackendError::Memory("invalid ptr".into()),
        ];
        for e in errors {
            assert!(!format!("{e}").is_empty());
        }
    }
}
