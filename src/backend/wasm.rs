//! Reference WebAssembly backend built on wasmtime. Compiled only
//! when the `wasm-reference` Cargo feature is enabled.
//!
//! One wasmtime `Engine` is shared across kernels; each
//! `register_kernel` compiles the `.wasm` bytes, instantiates a
//! fresh per-kernel `Store<()>` with empty host imports, and looks
//! up the named typed export. `dispatch_kernel` translates the
//! `KernelArg` slice to `wasmtime::Val` slots and calls it; the
//! `count` parameter is ignored (WASM kernels are single-threaded
//! scalar). `dispatch_one` uses the same persistent-worker-thread
//! shape as the CUDA backend. `dispatch_parallel_for` is host-side
//! fan-out through [`crate::sched::par_iter::for_each_chunk`] over
//! a Rust closure, not a WASM kernel.
//!
//! Use for sandboxed portable kernels (plugins, user-supplied
//! transforms, browser / WASI targets). Compilation (cranelift
//! JIT) is paid once at `register_kernel`; per-launch cost is a
//! sandboxed function call.
//!
//! Wire types: `I32` / `I64` / `F32` / `F64` pass as scalars,
//! `U32` / `U64` as their signed bit patterns, `DevicePtr` as an
//! unvalidated `i32` linear-memory offset. `HostSlice` returns
//! [`BackendError::NotSupported`]; consumers needing host-to-WASM
//! transfer ship their own backend with an allocator API.

#![allow(clippy::missing_errors_doc)]

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use wasmtime::{Engine, Instance, Module, Store, Val};

use crate::backend::{
    Backend, BackendCapabilities, BackendError, DispatchBackend, KernelArg, KernelHandle,
};
use crate::sched::notify_ring::{NotifyHub, NotifySender};

/// Boxed closure shape the persistent worker thread consumes from
/// the `dispatch_one` channel.
type WorkItem = Box<dyn FnOnce() + Send + 'static>;

/// Stored per registered kernel: the instance keeps the module
/// memory alive while the typed-call wrapper holds the resolved
/// function. Both live in a single store; the store + instance
/// must be locked together (wasmtime APIs require `&mut Store`).
struct KernelEntry {
    store: Store<()>,
    func: wasmtime::Func,
}

/// wasmtime-backed reference WebAssembly backend.
pub struct WasmBackend {
    device_id: u32,
    engine: Engine,
    caps: BackendCapabilities,
    next_handle: AtomicU64,
    /// Registered kernels, keyed by handle. Each entry owns its
    /// own `Store` so concurrent `dispatch_kernel` calls on
    /// different handles do not contend on a single store lock
    /// (wasmtime stores are not Sync; per-handle `Mutex<Store>`
    /// preserves Send across worker threads).
    kernels: Mutex<HashMap<u64, Arc<Mutex<KernelEntry>>>>,
    /// Persistent worker thread for `dispatch_one`. Routed
    /// through a flynnel notify hub (FlynnelRing + Parker).
    worker_hub: NotifyHub<WorkItem>,
    /// Cached sender so `dispatch_one` avoids `Arc::clone` per call.
    worker_tx: NotifySender<WorkItem>,
    worker_handle: Mutex<Option<JoinHandle<()>>>,
}

impl std::fmt::Debug for WasmBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WasmBackend")
            .field("device_id", &self.device_id)
            .finish()
    }
}

impl WasmBackend {
    /// Initialize a new WASM backend on the primary device (id 0).
    /// Pure-Rust path: cannot fail for runtime-resolution reasons
    /// the way CUDA can. Returns `BackendError::DeviceUnavailable`
    /// only if the OS rejects the worker-thread spawn (out of
    /// thread-id resources, etc.).
    pub fn new() -> Result<Self, BackendError> {
        Self::with_device(0)
    }

    /// Initialize a new WASM backend with a specific `device_id`.
    /// The id is purely informational for WASM (no hardware-device
    /// notion in wasmtime); kept for symmetry with the GPU
    /// backends so [`crate::backend::Backend::Wasm`] can carry it
    /// through dispatcher routing.
    pub fn with_device(device_id: u32) -> Result<Self, BackendError> {
        // Default Engine uses the cranelift compiler. wasmtime's
        // pure-Rust engine cannot fail to construct on supported
        // platforms; if a host lacks cranelift support the build
        // would not have linked.
        let engine = Engine::default();

        // Spawn the persistent dispatch_one worker. Same shape as
        // the CUDA backend's worker (notify-hub-based, exits when
        // the hub is shut down).
        const WASM_WORKER_RING_CAPACITY: usize = 1024;
        let worker_hub = NotifyHub::<WorkItem>::new(WASM_WORKER_RING_CAPACITY, 1);
        let worker_tx = worker_hub.sender();
        let hub_for_worker = worker_hub.clone();
        let worker_handle = std::thread::Builder::new()
            .name(format!("flynnel-wasm-{device_id}"))
            .spawn(move || {
                let rx = hub_for_worker.register_consumer();
                while let Some(work) = rx.recv() {
                    work();
                }
            })
            .map_err(|_| BackendError::DeviceUnavailable(Backend::Wasm { device_id }))?;

        Ok(Self {
            device_id,
            engine,
            caps: probe_capabilities(),
            next_handle: AtomicU64::new(1),
            kernels: Mutex::new(HashMap::new()),
            worker_hub,
            worker_tx,
            worker_handle: Mutex::new(Some(worker_handle)),
        })
    }
}

fn probe_capabilities() -> BackendCapabilities {
    // WASM execution is scalar single-threaded. Conservative
    // numbers: 1-wide, host-thread-count for max in-flight (host-
    // side fan-out via dispatch_parallel_for), ~5us per launch
    // (cranelift call setup + sandbox entry).
    let threads = std::thread::available_parallelism()
        .map(std::num::NonZeroUsize::get)
        .unwrap_or(1) as u32;
    BackendCapabilities {
        simt_width: 1,
        max_threads_in_flight: threads,
        launch_latency_ns: 5_000,
        h2d_bw_bytes_per_sec: 0,
    }
}

impl DispatchBackend for WasmBackend {
    fn id(&self) -> Backend {
        Backend::Wasm { device_id: self.device_id }
    }

    fn capabilities(&self) -> BackendCapabilities {
        self.caps
    }

    fn dispatch_parallel_for(
        &self,
        count: u32,
        work: &(dyn Fn(u32) + Send + Sync),
    ) {
        // Host-side fan-out via the global flynnel scheduler arena.
        // The closure body is a CPU-runnable Rust closure (not a
        // WASM kernel); for WASM kernel parallel fan-out, call
        // dispatch_kernel inside a host-side parallel loop.
        let plan = crate::sched::JobPlan::new(0, count);
        let mut indices: Vec<u32> = (0..count).collect();
        crate::sched::par_iter::for_each_chunk(
            &plan,
            indices.as_mut_slice(),
            move |slice: &mut [u32]| {
                for i in slice.iter() {
                    work(*i);
                }
            },
        );
    }

    fn dispatch_one(&self, work: Box<dyn FnOnce() + Send>) {
        // Best-effort send. If the worker has already exited
        // (during Drop) the hub is shut down and `send` returns
        // Closed; drop the work item silently to match the CUDA
        // backend's behavior.
        drop(self.worker_tx.send(work));
    }

    fn register_kernel(
        &self,
        name: &str,
        source: &[u8],
    ) -> Result<KernelHandle, BackendError> {
        // Compile the .wasm module via the engine's cranelift JIT.
        let module = Module::new(&self.engine, source).map_err(|e| {
            BackendError::KernelCompile(format!("wasm module compile failed: {e}"))
        })?;
        // Each kernel owns its own Store. Empty host imports (no
        // host functions exported to the kernel; the kernel is
        // pure compute over its arguments and linear memory).
        let mut store: Store<()> = Store::new(&self.engine, ());
        let instance = Instance::new(&mut store, &module, &[]).map_err(|e| {
            BackendError::KernelCompile(format!(
                "wasm instance create failed for kernel `{name}`: {e}"
            ))
        })?;
        // Look up the named export and confirm it is a function.
        let func = instance
            .get_func(&mut store, name)
            .ok_or_else(|| {
                BackendError::KernelCompile(format!(
                    "wasm export `{name}` not found in module"
                ))
            })?;
        let handle_id = self.next_handle.fetch_add(1, Ordering::SeqCst);
        let entry = KernelEntry { store, func };
        let mut guard = self.kernels.lock().unwrap();
        guard.insert(handle_id, Arc::new(Mutex::new(entry)));
        Ok(KernelHandle(handle_id))
    }

    fn dispatch_kernel(
        &self,
        handle: KernelHandle,
        _count: u32,
        args: &[KernelArg<'_>],
    ) -> Result<(), BackendError> {
        // Convert KernelArg slice into wasmtime Val slots.
        let mut vals: Vec<Val> = Vec::with_capacity(args.len());
        for a in args {
            match a {
                KernelArg::I32(v) => vals.push(Val::I32(*v)),
                KernelArg::I64(v) => vals.push(Val::I64(*v)),
                KernelArg::U32(v) => vals.push(Val::I32(*v as i32)),
                KernelArg::U64(v) => vals.push(Val::I64(*v as i64)),
                KernelArg::F32(v) => vals.push(Val::F32(v.to_bits())),
                KernelArg::F64(v) => vals.push(Val::F64(v.to_bits())),
                KernelArg::DevicePtr(p) => vals.push(Val::I32(*p as i32)),
                KernelArg::HostSlice(_) => {
                    return Err(BackendError::NotSupported);
                }
            }
        }
        // Locate the kernel entry and call its function.
        let entry_arc = {
            let guard = self.kernels.lock().unwrap();
            guard
                .get(&handle.0)
                .cloned()
                .ok_or_else(|| {
                    BackendError::Launch(format!(
                        "wasm kernel handle {} not registered",
                        handle.0
                    ))
                })?
        };
        let mut entry = entry_arc.lock().unwrap();
        // Function arity check: wasmtime will error at call time
        // if the count is wrong, but we surface a clearer message
        // here.
        let func_ty = entry.func.ty(&entry.store);
        let expected_params = func_ty.params().len();
        if expected_params != vals.len() {
            return Err(BackendError::Launch(format!(
                "wasm kernel arg count mismatch: function expects {} \
                 params, caller provided {}",
                expected_params,
                vals.len(),
            )));
        }
        let n_results = func_ty.results().len();
        let mut results: Vec<Val> = (0..n_results).map(|_| Val::I32(0)).collect();
        let KernelEntry { ref mut store, ref func } = *entry;
        func.call(store, &vals, &mut results).map_err(|e| {
            BackendError::Launch(format!("wasm kernel call failed: {e}"))
        })?;
        Ok(())
    }
}

impl Drop for WasmBackend {
    fn drop(&mut self) {
        // Shut down the notify hub so the worker's recv() returns
        // None and it exits cleanly after draining queued work.
        self.worker_hub.shutdown();
        let mut handle_guard = self.worker_handle.lock().unwrap();
        if let Some(handle) = handle_guard.take() {
            // Join failure here only means the worker panicked
            // (best-effort); we cannot meaningfully recover during
            // Drop.
            handle.join().ok();
        }
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Tiny WASM module exporting a function `add` that takes two
    /// i32 parameters and returns their sum. Hand-assembled binary
    /// from the WAT source:
    ///   (module
    ///     (func (export "add") (param i32 i32) (result i32)
    ///       local.get 0
    ///       local.get 1
    ///       i32.add))
    /// Self-contained byte literal so the test does not pull in
    /// `wat` or `wabt` as a dev-dependency.
    const ADD_WASM: &[u8] = &[
        0x00, 0x61, 0x73, 0x6d, // magic
        0x01, 0x00, 0x00, 0x00, // version
        // type section: 1 type, (i32, i32) -> (i32)
        0x01, 0x07, 0x01, 0x60, 0x02, 0x7f, 0x7f, 0x01, 0x7f,
        // function section: 1 function, type 0
        0x03, 0x02, 0x01, 0x00,
        // export section: 1 export, "add" func 0
        0x07, 0x07, 0x01, 0x03, 0x61, 0x64, 0x64, 0x00, 0x00,
        // code section: 1 body
        0x0a, 0x09, 0x01, 0x07, 0x00, 0x20, 0x00, 0x20, 0x01, 0x6a, 0x0b,
    ];

    #[test]
    fn wasm_backend_constructs() {
        WasmBackend::new().expect("wasm backend should init");
    }

    #[test]
    fn wasm_backend_reports_correct_id_and_caps() {
        let b = WasmBackend::with_device(7).expect("init");
        assert_eq!(b.id(), Backend::Wasm { device_id: 7 });
        let caps = b.capabilities();
        assert_eq!(caps.simt_width, 1);
        assert!(caps.max_threads_in_flight >= 1);
    }

    #[test]
    fn wasm_backend_dispatches_kernel_with_add_module() {
        let b = WasmBackend::new().expect("init");
        let handle = b
            .register_kernel("add", ADD_WASM)
            .expect("register `add` export");
        // (i32, i32) -> i32 with values (3, 4). dispatch_kernel
        // does not surface the return value through the trait
        // surface; the call succeeding (no error) is the test
        // contract. The function executes inside the wasmtime
        // sandbox and produces 7, which wasmtime discards.
        b.dispatch_kernel(handle, 1, &[KernelArg::I32(3), KernelArg::I32(4)])
            .expect("dispatch_kernel should succeed on valid args");
    }

    #[test]
    fn wasm_backend_rejects_arity_mismatch() {
        let b = WasmBackend::new().expect("init");
        let handle = b.register_kernel("add", ADD_WASM).expect("register");
        // `add` expects 2 args; pass 1. dispatch_kernel must
        // surface a Launch error rather than panicking inside
        // wasmtime.
        let err = b
            .dispatch_kernel(handle, 1, &[KernelArg::I32(3)])
            .expect_err("arity mismatch must error");
        assert!(matches!(err, BackendError::Launch(_)));
    }

    #[test]
    fn wasm_backend_register_kernel_with_missing_export_errors() {
        let b = WasmBackend::new().expect("init");
        let err = b
            .register_kernel("does_not_exist", ADD_WASM)
            .expect_err("missing export must error");
        assert!(matches!(err, BackendError::KernelCompile(_)));
    }

    #[test]
    fn wasm_backend_register_invalid_module_errors() {
        let b = WasmBackend::new().expect("init");
        let err = b
            .register_kernel("x", b"not a wasm module")
            .expect_err("invalid bytes must error");
        assert!(matches!(err, BackendError::KernelCompile(_)));
    }

    #[test]
    fn wasm_backend_host_slice_arg_returns_not_supported() {
        let b = WasmBackend::new().expect("init");
        let handle = b.register_kernel("add", ADD_WASM).expect("register");
        let buf: [u8; 4] = [1, 2, 3, 4];
        let err = b
            .dispatch_kernel(handle, 1, &[KernelArg::HostSlice(&buf)])
            .expect_err("HostSlice must be unsupported in reference impl");
        assert!(matches!(err, BackendError::NotSupported));
    }
}
