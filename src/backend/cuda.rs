//! Reference CUDA backend built on the cudarc crate's dynamic-
//! loading feature. Compiled only when the `cuda-reference`
//! Cargo feature is enabled.
//!
//! ## What this provides
//!
//! - [`CudaBackend::new`] / [`CudaBackend::with_device`]:
//!   constructors that initialize the CUDA driver via cudarc's
//!   dynamic-loading path, claim a device, and create a default
//!   stream.
//! - A full [`crate::backend::DispatchBackend`] implementation
//!   that:
//!   - implements `dispatch_parallel_for` as a synchronous host-
//!     side loop fan-out across worker OS threads (the closure
//!     body is CPU-runnable, not a GPU kernel);
//!   - implements `dispatch_one` by sending the work item to a
//!     single persistent worker thread the constructor spawns
//!     (routed via a flynnel `NotifyHub` MPMC ring); `Drop` shuts
//!     the hub down and joins the worker;
//!   - implements `register_kernel` by parsing PTX source text
//!     via cudarc's safe `CudaContext::load_module` API and
//!     storing the resulting function pointer in an internal
//!     map;
//!   - implements `dispatch_kernel` by reading the registered
//!     function and launching it through cudarc's safe
//!     `launch_builder` API and the corresponding `unsafe`
//!     `launch` call (argument count / type correctness is a
//!     kernel-author contract that cudarc's launch cannot
//!     verify) with the supplied `count` work-items and
//!     `KernelArg` list.
//!
//! ## When to use
//!
//! For consumers that have pre-compiled PTX they want to launch
//! through a uniform Flynnel surface. Consumers that need richer
//! CUDA semantics (per-launch streams, async H2D copies, CUDA
//! graphs) typically ship their own
//! [`crate::backend::DispatchBackend`] impl backed by their
//! preferred CUDA wrapper.

#![allow(clippy::missing_errors_doc)]

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use cudarc::driver::{CudaContext, CudaFunction, CudaModule, CudaStream, DriverError, LaunchConfig, PushKernelArg};

use crate::sched::notify_ring::{NotifyHub, NotifySender};

use crate::backend::{
    Backend, BackendCapabilities, BackendError, DispatchBackend, KernelArg, KernelHandle,
};

/// Boxed closure shape the persistent worker thread consumes from
/// the dispatch_one channel.
type WorkItem = Box<dyn FnOnce() + Send + 'static>;

/// cudarc-backed reference CUDA backend.
pub struct CudaBackend {
    device_id: u32,
    context: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    /// Secondary stream for ping-pong / double-buffered dispatch.
    /// Independent of the default `stream` so two operations on
    /// different streams overlap on the GPU. Consumers that
    /// alternate device buffers across iterations use this stream
    /// for the "pong" iteration while `stream` carries the "ping".
    secondary_stream: Arc<CudaStream>,
    caps: BackendCapabilities,
    next_handle: AtomicU64,
    /// Loaded modules keyed by handle, kept alive for the
    /// lifetime of any function pointers we hand out.
    modules: Mutex<HashMap<u64, Arc<CudaModule>>>,
    /// Function pointers keyed by handle, ready to launch.
    functions: Mutex<HashMap<u64, CudaFunction>>,
    /// Persistent worker thread that processes `dispatch_one`
    /// work items. Routed through a flynnel notify hub
    /// (FlynnelRing + Parker); `Drop` calls `hub.shutdown()` to
    /// signal the worker to exit cleanly.
    worker_hub: NotifyHub<WorkItem>,
    /// Cached sender handle so `dispatch_one` does not pay the
    /// `Arc::clone` per call.
    worker_tx: NotifySender<WorkItem>,
    /// Join handle for the persistent worker; taken in `Drop`.
    worker_handle: Mutex<Option<JoinHandle<()>>>,
}

impl std::fmt::Debug for CudaBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaBackend")
            .field("device_id", &self.device_id)
            .finish()
    }
}

impl CudaBackend {
    /// Initialize the CUDA driver on the primary device (id 0).
    /// Returns [`BackendError::DeviceUnavailable`] when the runtime
    /// is not loadable or the device cannot be opened.
    pub fn new() -> Result<Self, BackendError> {
        Self::with_device(0)
    }

    /// Initialize the CUDA driver on a specific device. `device_id`
    /// indexes into the platform's enumerated GPUs (0 for the
    /// first NVIDIA GPU).
    pub fn with_device(device_id: u32) -> Result<Self, BackendError> {
        let context =
            CudaContext::new(device_id as usize).map_err(|e| map_driver_error(device_id, e))?;
        let stream = context.default_stream();
        // Secondary stream for ping-pong dispatch. cudarc's
        // `new_stream` creates a non-default stream that runs
        // concurrently with the default stream on the device.
        let secondary_stream = context
            .new_stream()
            .map_err(|e| map_driver_error(device_id, e))?;
        let caps = probe_capabilities();

        // Spawn a persistent worker thread that consumes work
        // items from a flynnel notify hub. dispatch_one sends to
        // this hub instead of spawning a fresh OS thread per call.
        const CUDA_WORKER_RING_CAPACITY: usize = 1024;
        let worker_hub = NotifyHub::<WorkItem>::new(CUDA_WORKER_RING_CAPACITY, 1);
        let worker_tx = worker_hub.sender();
        let hub_for_worker = worker_hub.clone();
        let worker_handle = std::thread::Builder::new()
            .name(format!("flynnel-cuda-{device_id}"))
            .spawn(move || {
                let rx = hub_for_worker.register_consumer();
                while let Some(work) = rx.recv() {
                    work();
                }
            })
            .map_err(|e| {
                BackendError::DeviceUnavailable(Backend::Cuda { device_id })
                    .map_io_context(format!("worker thread spawn: {e}"))
            })?;

        Ok(Self {
            device_id,
            context,
            stream,
            secondary_stream,
            caps,
            next_handle: AtomicU64::new(1),
            modules: Mutex::new(HashMap::new()),
            functions: Mutex::new(HashMap::new()),
            worker_hub,
            worker_tx,
            worker_handle: Mutex::new(Some(worker_handle)),
        })
    }

    /// Convenience accessor exposing the underlying cudarc context
    /// for consumers that want to mix Flynnel-routed launches with
    /// direct cudarc usage (e.g. async stream synchronization
    /// outside the trait surface).
    pub fn context(&self) -> &Arc<CudaContext> {
        &self.context
    }

    /// Convenience accessor exposing the default stream.
    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    /// Secondary stream for ping-pong / double-buffered dispatch.
    /// Independent of [`Self::stream`]; operations queued on the
    /// two streams overlap on the GPU. Consumers alternating
    /// device-buffer pairs across iterations use this stream for
    /// the "pong" iteration while [`Self::stream`] carries the
    /// "ping".
    pub fn secondary_stream(&self) -> &Arc<CudaStream> {
        &self.secondary_stream
    }

    /// Pick `stream` or `secondary_stream` by parity of `slot`.
    /// Use this from a pipelined dispatch loop where each
    /// iteration N owns a device-buffer pair indexed by
    /// `N & 1`.
    pub fn stream_for_slot(&self, slot: usize) -> &Arc<CudaStream> {
        if slot & 1 == 0 {
            &self.stream
        } else {
            &self.secondary_stream
        }
    }

    /// Launch a registered kernel on a caller-chosen stream.
    /// Same semantics as [`DispatchBackend::dispatch_kernel`] but
    /// targets `stream` (typically [`Self::stream`] or
    /// [`Self::secondary_stream`]) instead of the default.
    /// Consumers driving a ping-pong pipeline call this with
    /// `stream_for_slot(iter & 1)` so adjacent iterations queue
    /// on independent streams and overlap on the GPU.
    pub fn dispatch_kernel_on_stream(
        &self,
        stream: &Arc<CudaStream>,
        handle: KernelHandle,
        count: u32,
        args: &[KernelArg<'_>],
    ) -> Result<(), BackendError> {
        let function = {
            let guard = self
                .functions
                .lock()
                .map_err(|_| BackendError::Launch("functions mutex poisoned".into()))?;
            guard
                .get(&handle.0)
                .cloned()
                .ok_or_else(|| BackendError::Launch(format!("unknown kernel handle {handle:?}")))?
        };
        let block = 256u32.min(count.max(1));
        let grid = count.div_ceil(block);
        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut builder = stream.launch_builder(&function);
        let mut i32s: Vec<i32> = Vec::new();
        let mut i64s: Vec<i64> = Vec::new();
        let mut u32s: Vec<u32> = Vec::new();
        let mut u64s: Vec<u64> = Vec::new();
        let mut f32s: Vec<f32> = Vec::new();
        let mut f64s: Vec<f64> = Vec::new();
        for arg in args {
            match arg {
                KernelArg::I32(v) => i32s.push(*v),
                KernelArg::I64(v) => i64s.push(*v),
                KernelArg::U32(v) => u32s.push(*v),
                KernelArg::U64(v) => u64s.push(*v),
                KernelArg::F32(v) => f32s.push(*v),
                KernelArg::F64(v) => f64s.push(*v),
                KernelArg::DevicePtr(p) => u64s.push(*p as u64),
                KernelArg::HostSlice(_) => return Err(BackendError::NotSupported),
            }
        }
        let (mut ii32, mut ii64, mut iu32, mut iu64, mut if32, mut if64) =
            (0usize, 0usize, 0usize, 0usize, 0usize, 0usize);
        for arg in args {
            match arg {
                KernelArg::I32(_) => {
                    builder.arg(&i32s[ii32]);
                    ii32 += 1;
                }
                KernelArg::I64(_) => {
                    builder.arg(&i64s[ii64]);
                    ii64 += 1;
                }
                KernelArg::U32(_) => {
                    builder.arg(&u32s[iu32]);
                    iu32 += 1;
                }
                KernelArg::U64(_) => {
                    builder.arg(&u64s[iu64]);
                    iu64 += 1;
                }
                KernelArg::F32(_) => {
                    builder.arg(&f32s[if32]);
                    if32 += 1;
                }
                KernelArg::F64(_) => {
                    builder.arg(&f64s[if64]);
                    if64 += 1;
                }
                KernelArg::DevicePtr(_) => {
                    builder.arg(&u64s[iu64]);
                    iu64 += 1;
                }
                KernelArg::HostSlice(_) => unreachable!("rejected in first pass"),
            }
        }
        // SAFETY: identical to dispatch_kernel - the typed
        // storage Vecs live until this function returns, after
        // the launch returns. Argument count / type correctness
        // is the kernel author's contract.
        unsafe { builder.launch(cfg) }
            .map_err(|e| BackendError::Launch(format!("{e:?}")))?;
        Ok(())
    }
}

impl DispatchBackend for CudaBackend {
    fn id(&self) -> Backend {
        Backend::Cuda {
            device_id: self.device_id,
        }
    }

    fn capabilities(&self) -> BackendCapabilities {
        self.caps
    }

    fn dispatch_parallel_for(&self, count: u32, work: &(dyn Fn(u32) + Send + Sync)) {
        // The closure body is CPU-runnable (an arbitrary Rust
        // closure cannot codegen to PTX). For GPU codegen, callers
        // use the `dispatch_kernel` handle path; this method runs
        // the CPU-shaped body fan-out so a `DispatchBackend` user
        // sees consistent dispatch_parallel_for semantics even on
        // GPU-class backends.
        if count == 0 {
            return;
        }
        std::thread::scope(|scope| {
            let threads = (count as usize).min(
                std::thread::available_parallelism()
                    .map(std::num::NonZeroUsize::get)
                    .unwrap_or(1),
            );
            let chunks = count.div_ceil(threads as u32);
            for t in 0..threads as u32 {
                let lo = t.saturating_mul(chunks);
                let hi = (lo + chunks).min(count);
                if lo >= hi {
                    continue;
                }
                scope.spawn(move || {
                    for i in lo..hi {
                        work(i);
                    }
                });
            }
        });
    }

    fn dispatch_one(&self, work: Box<dyn FnOnce() + Send>) {
        // Send to the persistent worker thread (no per-call OS
        // thread spawn). The notify hub is MPMC and lock-free on
        // the hot path.
        drop(self.worker_tx.send(work));
    }

    fn register_kernel(&self, name: &str, source: &[u8]) -> Result<KernelHandle, BackendError> {
        // The `source` is expected to be PTX text (UTF-8 bytes).
        let ptx_text = std::str::from_utf8(source)
            .map_err(|e| BackendError::KernelCompile(format!("PTX must be UTF-8: {e}")))?;
        let ptx = cudarc::nvrtc::Ptx::from_src(ptx_text);
        let module = self
            .context
            .load_module(ptx)
            .map_err(|e| BackendError::KernelCompile(format!("{e:?}")))?;
        let function = module
            .load_function(name)
            .map_err(|e| BackendError::KernelCompile(format!("function lookup `{name}`: {e:?}")))?;
        let handle_id = self.next_handle.fetch_add(1, Ordering::Relaxed);
        self.modules
            .lock()
            .map_err(|_| BackendError::KernelCompile("modules mutex poisoned".into()))?
            .insert(handle_id, module);
        self.functions
            .lock()
            .map_err(|_| BackendError::KernelCompile("functions mutex poisoned".into()))?
            .insert(handle_id, function);
        Ok(KernelHandle(handle_id))
    }

    fn dispatch_kernel(
        &self,
        handle: KernelHandle,
        count: u32,
        args: &[KernelArg<'_>],
    ) -> Result<(), BackendError> {
        let function = {
            let guard = self
                .functions
                .lock()
                .map_err(|_| BackendError::Launch("functions mutex poisoned".into()))?;
            guard
                .get(&handle.0)
                .cloned()
                .ok_or_else(|| BackendError::Launch(format!("unknown kernel handle {handle:?}")))?
        };
        // Launch geometry: pick a sensible block size (256) and
        // grid size to cover `count` work-items. Consumers that
        // need precise launch configuration ship their own backend
        // impl; the reference impl provides a one-size-fits-most
        // heuristic.
        let block = 256u32.min(count.max(1));
        let grid = count.div_ceil(block);
        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        // Build the launch through cudarc's safe builder. Each
        // KernelArg variant maps to a typed reference push; the
        // builder handles pointer marshalling internally.
        let mut builder = self.stream.launch_builder(&function);
        // Owned storage for the scalar values we push: the builder
        // holds raw pointers into these for the duration of
        // .launch(), so they must outlive the call.
        let mut i32s: Vec<i32> = Vec::new();
        let mut i64s: Vec<i64> = Vec::new();
        let mut u32s: Vec<u32> = Vec::new();
        let mut u64s: Vec<u64> = Vec::new();
        let mut f32s: Vec<f32> = Vec::new();
        let mut f64s: Vec<f64> = Vec::new();
        // First pass: fill the typed storage so backing pointers
        // do not move once we start pushing args.
        for arg in args {
            match arg {
                KernelArg::I32(v) => i32s.push(*v),
                KernelArg::I64(v) => i64s.push(*v),
                KernelArg::U32(v) => u32s.push(*v),
                KernelArg::U64(v) => u64s.push(*v),
                KernelArg::F32(v) => f32s.push(*v),
                KernelArg::F64(v) => f64s.push(*v),
                KernelArg::DevicePtr(p) => u64s.push(*p as u64),
                KernelArg::HostSlice(_) => return Err(BackendError::NotSupported),
            }
        }
        // Second pass: push references into the typed storage in
        // the original caller-supplied order.
        let (mut ii32, mut ii64, mut iu32, mut iu64, mut if32, mut if64) =
            (0usize, 0usize, 0usize, 0usize, 0usize, 0usize);
        for arg in args {
            match arg {
                KernelArg::I32(_) => {
                    builder.arg(&i32s[ii32]);
                    ii32 += 1;
                }
                KernelArg::I64(_) => {
                    builder.arg(&i64s[ii64]);
                    ii64 += 1;
                }
                KernelArg::U32(_) => {
                    builder.arg(&u32s[iu32]);
                    iu32 += 1;
                }
                KernelArg::U64(_) => {
                    builder.arg(&u64s[iu64]);
                    iu64 += 1;
                }
                KernelArg::F32(_) => {
                    builder.arg(&f32s[if32]);
                    if32 += 1;
                }
                KernelArg::F64(_) => {
                    builder.arg(&f64s[if64]);
                    if64 += 1;
                }
                KernelArg::DevicePtr(_) => {
                    builder.arg(&u64s[iu64]);
                    iu64 += 1;
                }
                KernelArg::HostSlice(_) => unreachable!("rejected in first pass"),
            }
        }
        // SAFETY: every arg reference points into one of the typed
        // storage Vecs above; the Vecs live until this function
        // returns, after the launch returns. The kernel function
        // pointer comes from a cudarc-loaded module also kept alive
        // by `self.modules`. Argument count / type correctness is
        // a contract with the kernel author (the safety hole cudarc
        // documents on launch()).
        unsafe { builder.launch(cfg) }
            .map_err(|e| BackendError::Launch(format!("{e:?}")))?;
        Ok(())
    }

    fn dispatch_kernel_sync(
        &self,
        handle: KernelHandle,
        count: u32,
        args: &[KernelArg<'_>],
    ) -> Result<(), BackendError> {
        // CUDA launches queue asynchronously on the stream; the
        // completion contract the auto-routing layer times against
        // needs the launch AND a stream synchronize.
        self.dispatch_kernel(handle, count, args)?;
        self.stream
            .synchronize()
            .map_err(|e| BackendError::Launch(format!("stream synchronize: {e:?}")))
    }
}

/// Capability descriptor for an NVIDIA SIMT backend. Conservative
/// nominals derived from the cudarc 0.19 / CUDA 12.6 ABI surface;
/// consumers that need exact device characteristics query cudarc
/// directly via [`CudaBackend::context`].
fn probe_capabilities() -> BackendCapabilities {
    BackendCapabilities {
        // NVIDIA warp is 32 threads.
        simt_width: 32,
        // Coarse upper bound: most modern NVIDIA GPUs hold 50k-
        // 200k threads in flight. 100k is a safe nominal.
        max_threads_in_flight: 100_000,
        // Driver launch overhead ~10us on PCIe.
        launch_latency_ns: 10_000,
        // PCIe 4.0 x16 sustained ~25 GB/s.
        h2d_bw_bytes_per_sec: 25_000_000_000,
    }
}

fn map_driver_error(device_id: u32, _e: DriverError) -> BackendError {
    BackendError::DeviceUnavailable(Backend::Cuda { device_id })
}

impl Drop for CudaBackend {
    fn drop(&mut self) {
        // Shut down the notify hub: the worker thread's recv()
        // returns None and it exits cleanly.
        self.worker_hub.shutdown();
        if let Ok(mut guard) = self.worker_handle.lock()
            && let Some(handle) = guard.take()
        {
            drop(handle.join());
        }
    }
}

/// Helper used in the constructor to attach a stderr context line
/// to a `DeviceUnavailable` error before returning. The
/// `BackendError::DeviceUnavailable` variant carries no message
/// field; the trait-side context is preserved by writing to stderr.
trait WithIoContext: Sized {
    fn map_io_context(self, msg: String) -> Self;
}

impl WithIoContext for BackendError {
    fn map_io_context(self, msg: String) -> Self {
        match self {
            BackendError::DeviceUnavailable(b) => {
                eprintln!("[flynnel::cuda] {}: {msg}", b.name());
                BackendError::DeviceUnavailable(b)
            }
            other => other,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::detect::cuda_available;

    #[test]
    fn cuda_backend_construction_matches_availability() {
        let res = CudaBackend::new();
        if cuda_available() {
            match res {
                Ok(b) => assert_eq!(b.id(), Backend::Cuda { device_id: 0 }),
                Err(BackendError::DeviceUnavailable(_)) => {}
                Err(e) => panic!("unexpected CUDA construction error: {e}"),
            }
        } else {
            assert!(matches!(res, Err(BackendError::DeviceUnavailable(_))));
        }
    }

    #[test]
    fn capabilities_report_warp_width_32() {
        if let Ok(b) = CudaBackend::new() {
            assert_eq!(b.capabilities().simt_width, 32);
        }
    }

    #[test]
    fn dispatch_parallel_for_invokes_each_index_on_host_fanout() {
        let Ok(backend) = CudaBackend::new() else {
            return;
        };
        use std::sync::atomic::AtomicU32;
        let counters: Vec<AtomicU32> = (0..256).map(|_| AtomicU32::new(0)).collect();
        let cref = &counters;
        backend.dispatch_parallel_for(256, &|i| {
            cref[i as usize].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        });
        for c in &counters {
            assert_eq!(c.load(std::sync::atomic::Ordering::Relaxed), 1);
        }
    }

    /// Smoke-test the register_kernel + dispatch_kernel path with
    /// a trivial PTX kernel (mul3.ptx is a no-op entry that takes
    /// one u32 dummy arg and returns). Validates that cudarc can
    /// parse the PTX, load the module, look up the function by
    /// name, and successfully launch it - all without depending
    /// on any meaningful kernel math. Skips when CUDA is not
    /// available on the host.
    #[test]
    fn register_and_dispatch_trivial_kernel() {
        let Ok(backend) = CudaBackend::new() else {
            return;
        };
        const TRIVIAL_PTX: &str = include_str!("../../kernels/mul3.ptx");
        let handle = backend
            .register_kernel("mul3", TRIVIAL_PTX.as_bytes())
            .expect("registering trivial PTX kernel");
        // Launch with 32 work-items and the one u32 dummy arg the
        // kernel signature declares. The kernel body is just `ret;`
        // so success here means: PTX parsed, module loaded, function
        // resolved, launch geometry accepted, kernel ran to completion.
        backend
            .dispatch_kernel(handle, 32, &[KernelArg::U32(7)])
            .expect("dispatching trivial PTX kernel");
        backend
            .stream()
            .synchronize()
            .expect("sync after trivial kernel dispatch");
    }
}
