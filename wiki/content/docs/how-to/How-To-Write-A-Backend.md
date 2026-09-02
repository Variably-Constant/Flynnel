---
title: How To Write A Backend
weight: 1
---

A walkthrough for implementing your own [`DispatchBackend`](Backend-System.md#dispatchbackend-trait) and registering it with the Flynnel registry.

## When to write a backend

Write a custom backend when you have an accelerator that doesn't fit the built-in classes, or when the reference CUDA / TPU backends don't expose enough of your runtime's semantics. Examples:

- An FPGA accelerator card with its own DMA + kernel-launch API.
- A custom AI ASIC with a vendor SDK.
- A WebGPU dispatcher running through `wgpu`.
- A ROCm dispatcher (Flynnel ships detection but no reference backend).
- A Metal dispatcher (same).
- A Vulkan compute dispatcher.

## The smallest legal backend

A `DispatchBackend` impl needs four methods: `id`, `capabilities`, `dispatch_parallel_for`, `dispatch_one`. Kernel methods default to `BackendError::NotSupported`.

```rust
use std::sync::Arc;
use flynnel::{
    Backend, BackendCapabilities, DispatchBackend, register_backend,
};

// Reuse Backend::Custom for taxonomy classes Flynnel does not
// name explicitly. Pick a stable u32 your ecosystem agrees on
// (e.g., hash the runtime name). Here we pack ASCII 'wgpu' into
// the high bytes and use the low byte for the device id, so
// device 0 is 0x7767_7075 and device 1 is 0x7767_7076.
const WGPU_BASE_ID: u32 = 0x7767_7075;

struct WebGpuBackend {
    device_id: u32,
}

impl DispatchBackend for WebGpuBackend {
    fn id(&self) -> Backend {
        Backend::Custom(WGPU_BASE_ID + self.device_id)
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            simt_width: 32,                    // WebGPU subgroup
            max_threads_in_flight: 65_536,
            launch_latency_ns: 20_000,         // ~20 us through wgpu
            h2d_bw_bytes_per_sec: 16_000_000_000,
        }
    }

    fn dispatch_parallel_for(&self, count: u32, work: &(dyn Fn(u32) + Send + Sync)) {
        // SIMT body cannot codegen to SPIR-V from a Rust closure.
        // Run it host-side as a fan-out: same shape the reference
        // CUDA / TPU backends use for the parallel-for surface.
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
                if lo >= hi { continue; }
                scope.spawn(move || {
                    for i in lo..hi { work(i); }
                });
            }
        });
    }

    fn dispatch_one(&self, work: Box<dyn FnOnce() + Send>) {
        std::thread::spawn(work);
    }
}

fn main() {
    register_backend(Arc::new(WebGpuBackend { device_id: 0 }));
}
```

That's a working backend. Consumers can now route work to it via:

```rust
let plan = JobPlan::new(8, 1024)
    .with_backend(Backend::Custom(WGPU_BASE_ID));
let backend = plan.pick_backend();
backend.dispatch_parallel_for(1024, &|i| { /* ... */ });
```

## Adding kernel support

Override `register_kernel` and `dispatch_kernel` to support the opaque-kernel-handle path used by GPU codegen.

```rust
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use flynnel::{BackendError, KernelArg, KernelHandle};

struct WebGpuBackend {
    device_id: u32,
    next_handle: AtomicU64,
    pipelines: Mutex<HashMap<u64, ComputePipeline>>,  // your wgpu pipelines
}

impl DispatchBackend for WebGpuBackend {
    // ... id, capabilities, dispatch_parallel_for, dispatch_one as above ...

    fn register_kernel(
        &self,
        name: &str,
        source: &[u8],
    ) -> Result<KernelHandle, BackendError> {
        // Parse SPIR-V or WGSL from `source`; compile a ComputePipeline.
        let pipeline = self
            .compile_pipeline(name, source)
            .map_err(|e| BackendError::KernelCompile(format!("{e}")))?;
        let id = self.next_handle.fetch_add(1, Ordering::Relaxed);
        self.pipelines
            .lock()
            .map_err(|_| BackendError::KernelCompile("mutex poisoned".into()))?
            .insert(id, pipeline);
        Ok(KernelHandle(id))
    }

    fn dispatch_kernel(
        &self,
        handle: KernelHandle,
        count: u32,
        args: &[KernelArg<'_>],
    ) -> Result<(), BackendError> {
        let pipeline = self
            .pipelines
            .lock()
            .map_err(|_| BackendError::Launch("mutex poisoned".into()))?
            .get(&handle.0)
            .cloned()
            .ok_or_else(|| BackendError::Launch(format!("unknown handle {handle:?}")))?;

        // Marshal KernelArg variants into your runtime's bind groups.
        let bindings = args.iter().map(arg_to_binding).collect::<Vec<_>>();
        self.submit_pipeline(&pipeline, count, &bindings)
            .map_err(|e| BackendError::Launch(format!("{e}")))?;
        Ok(())
    }
}
```

## Routing through the registry

Multiple backend instances with different `device_id`s register independently:

```rust
register_backend(Arc::new(WebGpuBackend::new(0)));
register_backend(Arc::new(WebGpuBackend::new(1)));

let plan_a = JobPlan::new(8, 1024).with_backend(Backend::Custom(WGPU_BASE_ID));
let plan_b = JobPlan::new(8, 1024).with_backend(Backend::Custom(WGPU_BASE_ID + 1));

let backend_a = plan_a.pick_backend();   // device 0
let backend_b = plan_b.pick_backend();   // device 1
```

If two registrations share the same `Backend` id, the second replaces the first (this is the documented hot-swap path).

## Graceful degradation

A backend constructor that may fail at runtime (no driver, no device) should return `Result<Self, BackendError>` and the caller registers only on success:

```rust
match WebGpuBackend::new(0) {
    Ok(backend) => {
        register_backend(Arc::new(backend));
        println!("WebGPU backend registered on device 0");
    }
    Err(BackendError::DeviceUnavailable(_)) => {
        // Routing helpers fall back to CPU automatically because
        // the hinted backend id is not in the registry.
        eprintln!("WebGPU not available; falling back to CPU");
    }
    Err(e) => {
        eprintln!("WebGPU backend init failed: {e}");
    }
}

// This runs unchanged in both cases. pick_backend() returns the
// registered WebGPU backend if available, else the CPU fallback.
let plan = JobPlan::new(8, 1024)
    .with_backend(Backend::Custom(WGPU_BASE_ID));
let backend = plan.pick_backend();
backend.dispatch_parallel_for(1024, &|i| { /* ... */ });
```

This is the same contract the reference [`CudaBackend`](Reference-Backends-CUDA-And-TPU.md#cudabackend), [`TpuJaxBackend`](Reference-Backends-CUDA-And-TPU.md#tpujaxbackend), [`WasmBackend`](Reference-Backends-CUDA-And-TPU.md#wasmbackend), and [`SharedMemoryChaseLevBackend`](How-To-Use-The-Shared-Memory-Worker-Backend.md) honor.

## What the trait does NOT require

- **No async**. `DispatchBackend` methods are synchronous from the caller's perspective. If your runtime is async (e.g., wgpu submission queues), block on the completion future inside the trait method.
- **No specific kernel ABI**. The `source` parameter to `register_kernel` is `&[u8]` - interpret it any way your runtime needs (PTX text, SPIR-V bytes, WGSL strings, Python source, a serialized JSON kernel descriptor).
- **No allocation API in the trait**. Device memory allocation, free, host-to-device copies all live on your concrete backend type. The trait only knows about `DevicePtr(usize)` as an opaque pointer the kernel will receive.
- **No completion barriers**. `dispatch_kernel` is "launch was queued"; users that need "launch completed" call your concrete type's sync primitive.

## Testing your backend

Stub a backend that records calls into atomic counters. The existing [`backend_dispatch_demo.rs`](https://github.com/Variably-Constant/Flynnel/blob/main/examples/backend_dispatch_demo.rs) shows the pattern: register the stub, route work, observe the counters tick. This is the recommended test scaffold for new backends.

## Where the existing impls live

| Backend | Lines | File |
|---------|-------|------|
| `CpuBackend` | 152 | [`src/backend/cpu.rs`](https://github.com/Variably-Constant/Flynnel/blob/main/src/backend/cpu.rs) |
| `CudaBackend` (cuda-reference) | 585 | [`src/backend/cuda.rs`](https://github.com/Variably-Constant/Flynnel/blob/main/src/backend/cuda.rs) |
| `TpuJaxBackend` (tpu-jax-reference) | 477 | [`src/backend/tpu_jax.rs`](https://github.com/Variably-Constant/Flynnel/blob/main/src/backend/tpu_jax.rs) |
| `WasmBackend` (wasm-reference) | 429 | [`src/backend/wasm.rs`](https://github.com/Variably-Constant/Flynnel/blob/main/src/backend/wasm.rs) |

The CPU impl is the simplest reference. The CUDA impl shows the dlopen-via-cudarc shape (no SDK at build time). The TPU impl shows the subprocess-bridge shape with JSON wire protocol. The WASM impl shows the pure-Rust wasmtime sandbox shape.
