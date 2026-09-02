---
title: "Reference Backends: CUDA, TPU, and WASM"
weight: 2
---

Flynnel ships three optional reference `DispatchBackend` implementations on this page (`CudaBackend`, `TpuJaxBackend`, `WasmBackend`) and one more on its own page (`SharedMemoryChaseLevBackend`; see [How To Use The Shared-Memory Worker Backend](How-To-Use-The-Shared-Memory-Worker-Backend.md)). Each is gated behind a Cargo feature so the default build pulls in no SDK or runtime dependencies.

## `CudaBackend`

Defined in [`src/backend/cuda.rs`](https://github.com/Variably-Constant/Flynnel/blob/main/src/backend/cuda.rs). Compiled only when the `cuda-reference` feature is enabled.

```toml
flynnel = { version = "0.2", features = ["cuda-reference"] }
```

The feature pulls in [cudarc](https://crates.io/crates/cudarc) with the `dynamic-loading + driver + nvrtc + cuda-12060` features. **No nvcc or CUDA SDK is required at build time** - cudarc dlopens `libcuda` at runtime through the dynamic-loading shim. The CUDA driver must be present at runtime for kernel launches to succeed; without it `CudaBackend::new()` returns `BackendError::DeviceUnavailable`.

### Construction

```rust
use flynnel::backend::cuda::CudaBackend;
use flynnel::register_backend;
use std::sync::Arc;

let backend = CudaBackend::new()?;            // device 0
let backend = CudaBackend::with_device(2)?;   // device 2 on multi-GPU host

register_backend(Arc::new(backend));
```

`with_device(device_id)` indexes into the platform's enumerated NVIDIA GPUs (0 for the first).

### Methods

| Method | Effect |
|--------|--------|
| `id()` | Returns `Backend::Cuda { device_id }`. |
| `capabilities()` | Reports `simt_width = 32` (NVIDIA warp), `max_threads_in_flight = 100_000`, `launch_latency_ns = 10_000`, `h2d_bw_bytes_per_sec = 25_000_000_000` (PCIe 4.0 x16 nominal). |
| `dispatch_parallel_for(count, work)` | Host-side fan-out across worker threads. The Rust closure body cannot codegen to PTX; use `dispatch_kernel` for GPU compute. |
| `dispatch_one(work)` | Spawns an OS thread (same as the CPU backend's fire-and-forget shape). |
| `register_kernel(name, source)` | `source` is PTX text bytes (UTF-8). Loads the module via cudarc's `CudaContext::load_module` and looks up the function by `name`. Returns the per-backend `KernelHandle`. |
| `dispatch_kernel(handle, count, args)` | Launches the registered function through cudarc's safe `launch_builder` + `arg` chain + `launch`. Default launch geometry: block size 256, grid size `count.div_ceil(256)`. |
| `context()` | Exposes the underlying `Arc<CudaContext>` for callers mixing Flynnel routing with direct cudarc ops (async stream synchronization, etc.). |
| `stream()` | Exposes the default `Arc<CudaStream>` ("ping" stream). |
| `secondary_stream()` | Exposes the secondary `Arc<CudaStream>` ("pong" stream) for double-buffered dispatch. Independent of `stream()`; operations queued on the two streams overlap on the GPU. |
| `stream_for_slot(slot)` | Selects `stream()` or `secondary_stream()` by parity of `slot`. Use this from a pipelined dispatch loop where each iteration `N` owns a device-buffer pair indexed by `N & 1`. |
| `dispatch_kernel_on_stream(stream, handle, count, args)` | Same semantics as `dispatch_kernel` but targets a caller-chosen stream. Consumers driving a ping-pong pipeline call this with `stream_for_slot(iter & 1)` so adjacent iterations queue on independent streams and overlap on the GPU. |

### Ping-pong dispatch pattern

Two device buffers + two streams let the GPU process iteration N on stream A while the host preloads iteration N+1's input to a different buffer on stream B. This hides PCIe transfer behind kernel compute. It pays back when kernel time and transfer time are within ~3x of each other; when kernel dominates (compute-bound) or transfer dominates (PCIe-bound) the win shrinks.

```rust
use flynnel::backend::cuda::CudaBackend;
use flynnel::{KernelArg, KernelHandle};

# fn ping_pong_demo(
#     backend: &CudaBackend,
#     handle: KernelHandle,
#     bufs: [&cudarc::driver::CudaSlice<f32>; 2],
#     n_iters: usize,
# ) {
use cudarc::driver::DevicePtr;
for iter in 0..n_iters {
    let stream = backend.stream_for_slot(iter);
    let buf = bufs[iter & 1];
    let (dp, _g) = buf.device_ptr(stream);
    backend.dispatch_kernel_on_stream(
        stream,
        handle,
        1_000_000,
        &[KernelArg::DevicePtr(dp as usize), KernelArg::I32(1_000_000), KernelArg::I32(50)],
    ).unwrap();
    // No synchronize() here - both streams run concurrently on the GPU.
}
// Sync each stream once at the end (or once per iteration if the caller
// needs ordered intermediate results back).
backend.stream().synchronize().unwrap();
backend.secondary_stream().synchronize().unwrap();
# }
```

### `KernelArg` mapping

| Variant | Marshalled as |
|---------|---------------|
| `I32` / `I64` / `U32` / `U64` / `F32` / `F64` | Pushed by value into the cudarc launch builder. |
| `DevicePtr(usize)` | Cast to `u64` and pushed; matches CUDA's pointer-as-u64 calling convention. |
| `HostSlice(&[u8])` | Returns `BackendError::NotSupported`. Consumers do H2D copies through `cudarc::driver::CudaSlice` directly, then pass the resulting `DevicePtr`. |

### Worked example

```rust
use flynnel::{Backend, JobPlan, KernelArg, register_backend};
use flynnel::backend::cuda::CudaBackend;
use std::sync::Arc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // PTX for a saxpy kernel (consumer-supplied; loaded here via
    // include_str! at compile time; alternative paths are loading
    // from disk at runtime, or compiling at consumer build time
    // via nvrtc / nvcc -ptx).
    const SAXPY_PTX: &str = include_str!("../kernels/saxpy.ptx");

    let backend = CudaBackend::new()?;
    let handle = backend.register_kernel("saxpy", SAXPY_PTX.as_bytes())?;
    register_backend(Arc::new(backend));

    let plan = JobPlan::new(8, 1024).with_backend(Backend::Cuda { device_id: 0 });
    let backend = plan.pick_backend();

    // Assume the consumer's setup allocated device pointers x, y, out
    // via cudarc and recorded their raw values.
    let x_ptr = /* cudarc CudaSlice's device pointer */ 0xDEAD_BEEFusize;
    let y_ptr = 0xCAFE_BABEusize;
    let out_ptr = 0xBABE_FACEusize;

    backend.dispatch_kernel(
        handle,
        1024,
        &[
            KernelArg::U32(1024),                // n
            KernelArg::F32(2.5),                 // alpha
            KernelArg::DevicePtr(x_ptr),         // x
            KernelArg::DevicePtr(y_ptr),         // y
            KernelArg::DevicePtr(out_ptr),       // out
        ],
    )?;

    Ok(())
}
```

### Why cudarc + dynamic-loading

Flynnel commits to "no GPU SDK at build time" so the crate builds on any host with a Rust toolchain. cudarc's `dynamic-loading` feature dlopens `libcuda` lazily at runtime; the `cuda-12060` feature selects the ABI shape (the CUDA driver is forward-compatible, so a host with CUDA 12.6 or later satisfies any binary built with this feature).

If your consumer crate needs precise CUDA semantics not exposed through this reference backend (CUDA graphs, multi-stream concurrency, custom allocators, peer-to-peer transfers), ship your own `DispatchBackend` impl backed by your preferred CUDA wrapper. The trait surface is minimal so that drop-in replacement stays a short exercise.

## `TpuJaxBackend`

Defined in [`src/backend/tpu_jax.rs`](https://github.com/Variably-Constant/Flynnel/blob/main/src/backend/tpu_jax.rs). Compiled only when the `tpu-jax-reference` feature is enabled.

```toml
flynnel = { version = "0.2", features = ["tpu-jax-reference"] }
```

The feature pulls in `serde` + `serde_json` for the wire protocol. The host needs `python3` (or `python`) and JAX installed at runtime; without them, `TpuJaxBackend::new()` returns `BackendError::DeviceUnavailable`.

### Architecture

Unlike CUDA, there is no equivalent of `libcuda.so` you can dlopen from Rust for TPU. Google's XLA / TPU runtime lives entirely inside the Python ecosystem; the canonical access shape is "spawn a Python child process and drive it over a wire protocol."

`TpuJaxBackend`:

1. Locates a Python interpreter (tries `python3`, then `python`).
2. Writes the embedded `tpu_jax_bridge.py` script to a temp file under `$TMPDIR`.
3. Spawns the interpreter with the temp script.
4. Exchanges a `ping` handshake to verify Python starts, JAX imports, and `jax.devices()` reports at least one device.
5. On any failure: returns `BackendError::DeviceUnavailable(Backend::Tpu { .. })`.

The bridge script is `include_str!`-baked into the Rust binary at compile time, so the crate ships as a single artifact.

### Construction

```rust
use flynnel::backend::tpu_jax::TpuJaxBackend;
use flynnel::register_backend;
use std::sync::Arc;

let backend = TpuJaxBackend::new()?;            // primary TPU
let backend = TpuJaxBackend::with_device(1)?;   // tag for second device

register_backend(Arc::new(backend));
```

`with_device(device_id)` passes through as the identity tag on the resulting `Backend::Tpu` id; JAX itself manages device selection per its own `jax.devices()` order.

### Methods

| Method | Effect |
|--------|--------|
| `id()` | Returns `Backend::Tpu { device_id }`. |
| `capabilities()` | Reports `simt_width = 128` (TPU MXU lane), `max_threads_in_flight = 200_000`, `launch_latency_ns = 100_000` (Python-JAX dispatch round-trip dominated by JSON encode + subprocess pipe latency), `h2d_bw_bytes_per_sec = 25_000_000_000`. |
| `dispatch_parallel_for(count, work)` | Host-side fan-out (same shape as CUDA backend). |
| `dispatch_one(work)` | Spawns an OS thread. |
| `register_kernel(name, source)` | `source` is UTF-8 Python source defining a function bound to `name`. The bridge `exec`s the source and `jax.jit()`s the function; returns a per-backend handle. |
| `dispatch_kernel(handle, count, args)` | Serializes a JSON request to the bridge with the unpacked args; bridge calls the JIT function and blocks until the result materialises. |
| `devices()` | Devices the JAX runtime reported during the handshake (e.g., `["TpuDevice(id=0, ...)"]`). Useful for telemetry. |

### `KernelArg` mapping

| Variant | Marshalled as JSON |
|---------|-------------------|
| `I32(v)` | `{"i32": v}` |
| `I64(v)` | `{"i64": v}` |
| `U32(v)` | `{"u32": v}` |
| `U64(v)` | `{"u64": v}` |
| `F32(v)` | `{"f32": v}` |
| `F64(v)` | `{"f64": v}` |
| `DevicePtr(p)` | `{"device_ptr": p}` (cast to u64) |
| `HostSlice(_)` | Returns `BackendError::NotSupported`. Use `DevicePtr` after staging via JAX's own allocator. |

### Wire protocol

Line-oriented JSON over the child process's stdin/stdout. One request, one response, in order. Serialized via `serde_json`.

```text
->  {"op":"ping"}
<-  {"ok":true,"devices":["TpuDevice(...)"],"jax_version":"0.4.x"}

->  {"op":"register","name":"my_kernel","source":"def my_kernel(c,*a): return jnp.sum(jnp.arange(c)) + a[0]"}
<-  {"ok":true,"handle":1}

->  {"op":"dispatch","handle":1,"count":4096,"args":[{"i32":7},{"f32":1.5}]}
<-  {"ok":true}

->  {"op":"shutdown"}
<-  {"ok":true,"goodbye":true}
```

### Worked example

```rust
use flynnel::{Backend, JobPlan, KernelArg, register_backend};
use flynnel::backend::tpu_jax::TpuJaxBackend;
use std::sync::Arc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let backend = TpuJaxBackend::new()?;

    let py = r#"
def saxpy(count, alpha):
    arr = jnp.arange(int(count)) * float(alpha)
    return jnp.sum(arr)
"#;
    let handle = backend.register_kernel("saxpy", py.as_bytes())?;
    register_backend(Arc::new(backend));

    let plan = JobPlan::new(8, 4096).with_backend(Backend::Tpu { device_id: 0 });
    let backend = plan.pick_backend();

    backend.dispatch_kernel(
        handle,
        4096,
        &[KernelArg::F32(2.5)],
    )?;

    Ok(())
}
```

### Graceful degradation

```rust
match TpuJaxBackend::new() {
    Ok(backend) => {
        register_backend(Arc::new(backend));
        // TPU dispatches now route correctly.
    }
    Err(_) => {
        // Python missing, JAX missing, no TPU: degrade silently.
        // pick_backend() will fall back to the CPU backend.
    }
}
```

This same code runs unchanged across "no Python", "Python no JAX", "Python + JAX no TPU", and "full TPU host" - failure mode is graceful, not panic.

## `WasmBackend`

Defined in [`src/backend/wasm.rs`](https://github.com/Variably-Constant/Flynnel/blob/main/src/backend/wasm.rs). Compiled only when the `wasm-reference` feature is enabled.

```toml
flynnel = { version = "0.2", features = ["wasm-reference"] }
```

The feature pulls in [wasmtime](https://crates.io/crates/wasmtime) with the `cranelift` + `runtime` features and `default-features = false`. wasmtime ships as a pure-Rust crate, so the build needs no system shared library and the runtime needs no installed engine. The whole WebAssembly compiler and sandbox lives inside the Flynnel binary.

### Construction

```rust
use flynnel::backend::wasm::WasmBackend;
use flynnel::register_backend;
use std::sync::Arc;

let backend = WasmBackend::new()?;           // device 0
let backend = WasmBackend::with_device(2)?;  // engine id 2 (informational only)

register_backend(Arc::new(backend));
```

`device_id` has no hardware meaning here (wasmtime exposes no device notion); it is kept on the constructor for symmetry with the GPU backends so `Backend::Wasm { device_id }` can carry through dispatcher routing if a consumer wants to register several engines.

### Methods

| Method | Effect |
|--------|--------|
| `id()` | Returns `Backend::Wasm { device_id }`. |
| `capabilities()` | Reports `simt_width = 1` (scalar), `max_threads_in_flight = host_thread_count`, `launch_latency_ns = 5_000` (cranelift call setup + sandbox entry), `h2d_bw_bytes_per_sec = 0`. |
| `dispatch_parallel_for(count, work)` | Host-side fan-out through the Flynnel arena. The closure body runs as native Rust, not as a WASM kernel; for WASM-side parallel fan-out call `dispatch_kernel` inside a host-side parallel loop. |
| `dispatch_one(work)` | Sends the boxed closure to the persistent worker thread (same shape as the CUDA backend's `dispatch_one`). |
| `register_kernel(name, source)` | Treats `source` as the bytes of a `.wasm` module, calls `wasmtime::Module::new` + `Instance::new`, and resolves the export named `name`. Returns the per-backend `KernelHandle`. |
| `dispatch_kernel(handle, count, args)` | Translates each `KernelArg` into a `wasmtime::Val` and invokes the function. `count` is ignored (WASM kernels are scalar). |

### `KernelArg` mapping

| Variant | Wire shape into wasmtime |
|---------|--------------------------|
| `I32(v)` / `I64(v)` / `F32(v)` / `F64(v)` | Passed directly as the matching scalar `Val` variant. |
| `U32(v)` / `U64(v)` | Passed as the bit-pattern-equivalent signed integer (`Val::I32` / `Val::I64`). WASM has no unsigned scalar type at the type system level; arithmetic is unsigned-or-signed by opcode choice. |
| `DevicePtr(v)` | Passed as `Val::I32(v as i32)`, the offset into the instance's linear memory. The backend does NOT validate the offset; caller responsibility. |
| `HostSlice(_)` | Returns `BackendError::NotSupported`. The reference impl keeps the surface minimal; consumers who need host-to-WASM byte transfer ship their own backend with a typed allocator API. |

### Worked example

```rust
use flynnel::backend::wasm::WasmBackend;
use flynnel::backend::{Backend, DispatchBackend, KernelArg};

// (module
//   (func (export "add") (param i32 i32) (result i32)
//     local.get 0 local.get 1 i32.add))
const ADD_WASM: &[u8] = &[
    0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
    0x01, 0x07, 0x01, 0x60, 0x02, 0x7f, 0x7f, 0x01, 0x7f,
    0x03, 0x02, 0x01, 0x00,
    0x07, 0x07, 0x01, 0x03, 0x61, 0x64, 0x64, 0x00, 0x00,
    0x0a, 0x09, 0x01, 0x07, 0x00, 0x20, 0x00, 0x20, 0x01, 0x6a, 0x0b,
];

let backend = WasmBackend::new()?;
let handle = backend.register_kernel("add", ADD_WASM)?;
backend.dispatch_kernel(handle, 1, &[KernelArg::I32(3), KernelArg::I32(4)])?;
```

The runnable version is at [`examples/wasm_dispatch_demo.rs`](https://github.com/Variably-Constant/Flynnel/blob/main/examples/wasm_dispatch_demo.rs) (`cargo run --release --example wasm_dispatch_demo --features wasm-reference`).

### When to reach for WASM dispatch

Use this backend when consumers want sandboxed portable kernels: user-supplied transformation closures, runtime-loaded plugins, or kernels that need to run inside browser / WASI environments without recompiling. The host pays a one-time module-compilation cost (cranelift JIT) at `register_kernel`; per-launch cost is comparable to a function call within the wasmtime sandbox (~5 us by the reported capabilities, dominated by sandbox entry and result marshalling).

WASM is NOT the right backend for raw throughput: scalar single-threaded execution per call means a fan-out of host work is faster on the in-process CPU backend. The reach-for-it case is plugin / sandboxing semantics, not speed.

## Side-by-side comparison

| Aspect | `CudaBackend` | `TpuJaxBackend` | `WasmBackend` |
|--------|---------------|-----------------|---------------|
| Runtime access mechanism | dlopen via cudarc dynamic-loading | Python subprocess + JSON wire protocol | In-process wasmtime engine (cranelift JIT) |
| Build-time SDK dep | None | None | None (wasmtime is pure Rust) |
| Runtime requirement | CUDA driver (libcuda) | python3 + jax | None |
| Kernel source format | PTX text | Python function source | `.wasm` module bytes |
| Default launch geometry | 256 threads/block | JAX-managed | Scalar single-threaded per call |
| Per-launch overhead | ~10 us (driver round-trip) | ~100 us (JSON + pipe) | ~5 us (sandbox entry) |
| In-flight throughput | High; saturate the GPU | Bridge-serialized (one request at a time) | One call per host thread |
| Memory model | DevicePtr from cudarc | DevicePtr opaque to bridge | DevicePtr = offset into linear memory |

Pick CUDA when the consumer can codegen PTX and wants per-kernel latency in the tens-of-microseconds range. Pick TPU when the consumer runs JAX-native compute and accepts the bridge's hundred-microsecond launch overhead. Pick WASM when sandboxing or portable plugin loading matters more than raw throughput. For cross-process dispatch (peer worker farms, language interop, license-isolated runtimes) the `SharedMemoryChaseLevBackend` lives at [its own how-to page](How-To-Use-The-Shared-Memory-Worker-Backend.md); per-call cost is 342-881 ns over an MMF Chase-Lev deque + latch arena depending on the coherence tier.

All four are reference implementations. Consumers with richer needs (CUDA graphs, JAX `pmap` across device meshes, WASI sandboxes with imported host functions, peer pools that need transport-level features) ship their own `DispatchBackend` impl that registers alongside.
