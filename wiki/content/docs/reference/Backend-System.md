---
title: Backend System
weight: 4
---

The pluggable dispatch fabric. The [`flynnel::backend`](https://github.com/Variably-Constant/Flynnel/blob/main/src/backend) module owns the `Backend` taxonomy enum, the `DispatchBackend` trait, the runtime registry, and the per-runtime detection probes.

## Mental model

```text
   .------------------.       .------------------.
   |  Backend (enum)  |       |  DispatchBackend |
   |  taxonomy id     |<------+  (trait)         |
   '------------------'       '------------------'
            ^                          ^
            |                          | implements
            |                          |
            |                  .---------------------.
            |  registered      |  CpuBackend         |
            +------------------+  (always available) |
            |                  '---------------------'
            |                          ^
            |                          | implements
            |                  .---------------------.
            +------------------+  CudaBackend        |  (cuda-reference feature)
            |                  '---------------------'
            |                          ^
            |                          | implements
            |                  .---------------------.
            +------------------+  TpuJaxBackend      |  (tpu-jax-reference feature)
            |                  '---------------------'
            |                          ^
            |                          | implements
            |                  .---------------------.
            '------------------+  Consumer-supplied  |
                               '---------------------'
                                       |
                                       v
                              .------------------.
                              |  registry        |
                              |  (Backend -> Arc)|
                              '------------------'
```

`JobPlan::pick_backend()` reads `plan.backend_hint`, looks the id up in the registry, returns the matching `Arc<dyn DispatchBackend>` (or the CPU fallback). Routing helpers like [`join_hybrid`](Sched-Module-Reference.md#join_hybrid) go through `pick_backend` so every call site uses the same resolution path.

## `Backend` enum

```rust
pub enum Backend {
    Cpu,
    Cuda { device_id: u32 },
    Rocm { device_id: u32 },
    Metal { device_id: u32 },
    Tpu { device_id: u32 },
    Ane,
    Wasm { device_id: u32 },
    SharedMemoryWorker { backend_id: u32 },
    Custom(u32),
}
```

`Copy`, `Clone`, `Debug`, `PartialEq`, `Eq`, `Hash`. Methods:

- `backend.name() -> &'static str` returns a stable lowercase tag: `"cpu"`, `"cuda"`, `"rocm"`, `"metal"`, `"tpu"`, `"ane"`, `"wasm"`, `"shared-mem"`, `"custom"`.
- `backend.is_simt() -> bool` returns `true` for `Cuda` / `Rocm` / `Metal` / `Tpu`; `false` for `Cpu` / `Ane` / `Wasm` / `SharedMemoryWorker` / `Custom`.

Two backends with the same class and different `device_id` register as independent entries (a multi-GPU host registers one `CudaBackend` per device).

## `BackendCapabilities`

```rust
pub struct BackendCapabilities {
    pub simt_width: u32,
    pub max_threads_in_flight: u32,
    pub launch_latency_ns: u32,
    pub h2d_bw_bytes_per_sec: u64,
}
```

Reported by each backend so routing helpers can reason about cost (small jobs stay on CPU; large data-parallel jobs amortize GPU launch latency).

| Field | Meaning |
|-------|---------|
| `simt_width` | Lanes that execute in lockstep within one launch. 1 for scalar CPU; 32 for NVIDIA warp; 64 for AMD wave; 128 for TPU MXU lane. |
| `max_threads_in_flight` | Upper bound on threads (CPU) or work-items (GPU) the backend can hold in flight simultaneously. |
| `launch_latency_ns` | Wall-clock cost of a single empty dispatch in nanoseconds. CPU ~100 ns; CUDA ~10 us; TPU JAX ~100 us. |
| `h2d_bw_bytes_per_sec` | Host-to-device sustained throughput in bytes/second. Zero for the CPU backend (no copy required). |

`BackendCapabilities::cpu_defaults()` returns conservative CPU values derived from `std::thread::available_parallelism()`.

## `DispatchBackend` trait

Defined in [`src/backend/mod.rs`](https://github.com/Variably-Constant/Flynnel/blob/main/src/backend/mod.rs). Object-safe so `Arc<dyn DispatchBackend>` works.

```rust
pub trait DispatchBackend: Send + Sync + 'static {
    fn id(&self) -> Backend;
    fn capabilities(&self) -> BackendCapabilities;
    fn dispatch_parallel_for(&self, count: u32, work: &(dyn Fn(u32) + Send + Sync));
    fn dispatch_one(&self, work: Box<dyn FnOnce() + Send>);
    fn register_kernel(&self, name: &str, source: &[u8])
        -> Result<KernelHandle, BackendError>;
    fn dispatch_kernel(&self, handle: KernelHandle, count: u32, args: &[KernelArg<'_>])
        -> Result<(), BackendError>;
}
```

### `id`

Returns the [`Backend`](#backend-enum) variant identifying this instance.

### `capabilities`

Returns [`BackendCapabilities`](#backendcapabilities). Cached on the backend (probed once at construction).

### `dispatch_parallel_for`

SIMT-shaped parallel-for: invoke `work(i)` for `i` in `0..count`. CPU backends fan out via the existing arena; GPU / TPU backends translate to a kernel launch with `count` work-items each running the closure.

The closure body is constrained to CPU-runnable code (an arbitrary Rust closure cannot codegen to PTX). For GPU codegen, use the [`dispatch_kernel`](#dispatch_kernel) handle path with pre-built kernels.

### `dispatch_one`

Single-shot fire-and-forget closure. CPU backends spawn a fresh OS thread; GPU backends typically wrap the closure in a host-side thread for backend bookkeeping work (GPU kernels themselves cannot run arbitrary Rust closures).

### `register_kernel`

Register a pre-built kernel; returns a [`KernelHandle`](#kernelhandle) the consumer passes to `dispatch_kernel`. `source` is backend-specific:

- CUDA backend: PTX text bytes (UTF-8).
- TPU JAX backend: Python source defining a function named `name` that JAX can `jit()` (the function signature is `def name(count, *args)`).
- CPU backend: returns `BackendError::NotSupported` (the trait default).

### `dispatch_kernel`

Launch a previously-registered kernel with `count` work-items and the supplied [`KernelArg`](#kernelarg) list. Blocks until the launch is queued (not necessarily complete; concrete backends expose their own synchronization primitives on the typed handle for callers that need to wait).

CPU backend: returns `BackendError::NotSupported` (the trait default).

## `KernelHandle`

```rust
pub struct KernelHandle(pub u64);
```

Opaque per-backend handle. `Copy`, `Clone`, `Debug`, `Hash`. Handles are *not* portable across backends; the same `u64` can mean different kernels in two different `DispatchBackend` instances.

## `KernelArg`

```rust
pub enum KernelArg<'a> {
    I32(i32),
    I64(i64),
    U32(u32),
    U64(u64),
    F32(f32),
    F64(f64),
    DevicePtr(usize),
    HostSlice(&'a [u8]),
}
```

Single argument passed to a kernel via `dispatch_kernel`. Backends interpret each variant per native ABI:

- Scalar variants (`I32` / `U64` / `F32` / etc.): passed by value through the GPU calling convention.
- `DevicePtr(usize)`: raw device-side pointer the backend allocated previously and made known to the consumer.
- `HostSlice(&[u8])`: host bytes to copy to device memory as part of the launch. The reference CUDA backend returns `NotSupported` for this variant (consumers do H2D copies through cudarc directly); the TPU JAX backend also returns `NotSupported` (use `DevicePtr` after staging via a backend-specific allocator).

## `BackendError`

```rust
pub enum BackendError {
    NotSupported,
    DeviceUnavailable(Backend),
    KernelCompile(String),
    Launch(String),
    Memory(String),
}
```

Implements `Display` and `std::error::Error`. Routing helpers (`JobPlan::pick_backend`) collapse most failures to a CPU fallback; explicit error handling matters at backend construction time (e.g., `CudaBackend::new()` returning `DeviceUnavailable` on a host without a CUDA driver).

## Registry

The process-global registry stores `Arc<dyn DispatchBackend>` keyed by [`Backend`](#backend-enum). Defined in [`src/backend/registry.rs`](https://github.com/Variably-Constant/Flynnel/blob/main/src/backend/registry.rs).

### `register_backend`

```rust
pub fn register_backend(b: BackendRef)
```

Register a backend with the global registry. If a backend with the same id is already registered, **the new one replaces it**. This is the documented hot-swap path for consumers installing a more capable backend over the default.

### `backend_by_id`

```rust
pub fn backend_by_id(id: &Backend) -> Option<BackendRef>
```

Look up a backend by id. Returns `None` if no backend with that exact id (including matching `device_id`) is registered.

### `backends`

```rust
pub fn backends() -> Vec<BackendRef>
```

Snapshot every registered backend. Used for telemetry and fallback selection.

### `cpu_backend`

```rust
pub fn cpu_backend() -> BackendRef
```

Canonical `Arc` for the always-available CPU backend. Infallible: the CPU backend is auto-registered on first registry access and cannot be removed.

### `ensure_default_registered`

```rust
pub fn ensure_default_registered() -> bool
```

Forces the registry to initialize. Most callers do not need this since any registry access auto-inits; exposed for explicit-init callers that want predictable startup timing.

## Runtime backend migration (active-backend tag)

Lives in [`src/sched/adaptive_backend.rs`](https://github.com/Variably-Constant/Flynnel/blob/main/src/sched/adaptive_backend.rs). Process-global `AtomicU32` tag selects the active backend for routing helpers that consult the global (e.g. `AdaptiveDispatcher::execute_indexed` and the dispatcher's `resolve_active_backend()` method). Distinct from `backend_hint` on a per-call `JobPlan`: the hint pins one call; the tag drives the default for every call that does not pin a backend.

### `active_backend_id`

```rust
pub fn active_backend_id() -> Backend
```

Reads the global tag via one `AtomicU32::Acquire-load` and decodes to a `Backend` enum value. Default at process start: `Backend::Cpu`.

### `migrate_backend`

```rust
pub fn migrate_backend(backend: Backend)
```

Stores the encoded tag via one `AtomicU32::Release-store`. Subsequent `resolve_active_backend()` calls anywhere in the process see the new backend on their next read. Cost: one atomic store; zero per-op cost on the dispatch hot path.

### `resolve_active_backend`

```rust
pub fn resolve_active_backend() -> (BackendRef, bool)
```

Looks up the registered backend for the active tag. Returns `(backend, fell_back)`. When the requested backend is registered, returns it with `fell_back = false`. When it is not registered (e.g., `Backend::Cuda { device_id: 0 }` on a host without `cuda-reference` enabled and a CUDA driver loadable), returns the always-available CPU backend with `fell_back = true`. The boolean flag lets the caller observe the fallback explicitly instead of silently dispatching to the wrong target.

Wired through [`AdaptiveDispatcher`](Sched-Module-Reference.md#dispatch) as `dispatcher.active_backend_id()`, `dispatcher.migrate_backend(b)`, `dispatcher.resolve_active_backend()`. End-to-end demonstration in Section [7] of [`examples/adaptive_dispatcher_demo.rs`](https://github.com/Variably-Constant/Flynnel/blob/main/examples/adaptive_dispatcher_demo.rs).

## Automatic CPU/accelerator routing (`accel_op`)

Lives in [`src/backend/accel_op.rs`](https://github.com/Variably-Constant/Flynnel/blob/main/src/backend/accel_op.rs). The layer that turns "a GPU is registered" into "eligible work runs on it" without a per-call-site placement decision. A Rust closure cannot execute on a GPU or TPU, so the surface is a registry of DECLARED equivalences, mirroring `pass_registry`'s closure-id pattern at the device boundary: code does not cross, an id and an argument list do.

```rust
pub fn register_accel_op(name, bytes_per_item, cpu_impl) -> AccelOpId
pub fn bind_accel_kernel(op, backend, entry, source) -> Result<(), BackendError>
pub fn bind_accel_kernel_handle(op, backend, handle)
pub fn accel_target(plan, op) -> Option<(Backend, KernelHandle)>
pub fn dispatch_accel(plan, op, count, cpu_args, kernel_args) -> AccelReport
```

Per dispatch, three decisions in order:

1. **Target resolution**: `plan.backend_hint` when set and registered, else the active-backend tag when non-CPU and bound, else the first bound-and-registered backend. No target: the CPU implementation runs.
2. **Static cost gate**: with an authoritative per-item cost on the plan, the accelerator is skipped when the estimated total cannot clear `LAUNCH_AMORTIZATION_FACTOR` (4) times the backend's `launch_latency_ns` plus the H2D time for `count * bytes_per_item` at its reported bandwidth. Classifier-default costs do not fire the gate.
3. **Learned placement**: the per-call-site, per-log2-size-bucket EWMAs of `CallSiteState::choose_placement` pick the side - Race when the bucket is cold (both sides run, sequentially, and both samples record), exploit when warm, re-race every 32nd call to track drift.

`cpu_args` and `kernel_args` are separate views because the two sides address different memory (host vs device pointers); each side reads and writes only through its own view. The two implementations must compute the same result and tolerate both running for one dispatch (cold race, reprobe, kernel-failure fallback) - the same contract `hybrid_auto` imposes.

The trait grows `DispatchBackend::dispatch_kernel_sync` (launch to COMPLETION, defaulted to `dispatch_kernel` for inherently synchronous backends; `CudaBackend` overrides with launch + stream synchronize), which is what the router times against.

Every failure lands on the CPU implementation: no binding, an unregistered backend, a failed launch. E2E on live devices: [`examples/accel_route_demo.rs`](https://github.com/Variably-Constant/Flynnel/blob/main/examples/accel_route_demo.rs) - the gate keeps a 64-item batch on the CPU, the cold bucket races, and 9/9 warm rounds exploit the GPU on both bench hosts (262k-element Newton sqrt: AVX2 host + RTX 3070, CPU 7.4 ms vs GPU 174 us; AVX-512 host + RTX 5070, CPU 2.7 ms vs GPU 87 us).

## Detection probes

[`src/backend/detect.rs`](https://github.com/Variably-Constant/Flynnel/blob/main/src/backend/detect.rs) probes whether each accelerator runtime is present on the host *without linking the SDK at build time*. Every probe `dlopen`s the platform-specific shared library via `libloading` or checks a known device-file / env-var path.

| Probe | What it checks |
|-------|---------------|
| `cuda_available() -> bool` | Tries `nvcuda.dll` / `libcuda.so.1` / `libcuda.so` / `libcuda.dylib` via `libloading::Library::new`. |
| `rocm_available() -> bool` | Tries `amdhip64.dll` / `libamdhip64.so` / `libamdhip64.so.6` / `libamdhip64.so.5`. |
| `metal_available() -> bool` | macOS only: checks `/System/Library/Frameworks/Metal.framework` exists. |
| `tpu_available() -> bool` | Checks `TPU_NAME` env, then on Linux checks `/dev/accel0` / `/dev/apex_0` / `/dev/vfio/vfio`. |
| `ane_available() -> bool` | Returns `cfg!(all(target_os = "macos", target_arch = "aarch64"))`. |
| `wasm_available() -> bool` | Returns `cfg!(feature = "wasm-reference")`. wasmtime ships pure-Rust so the probe collapses to a build-feature check. |
| `shared_memory_worker_available() -> bool` | Returns `cfg!(feature = "shared-memory-worker-reference")`. Same shape as the WASM probe: memmap2 ships pure-Rust so runtime presence is the build-feature check. Whether a peer is actually listening on a given deque is answered separately by attempting `SharedMemoryChaseLevBackend::open`. |
| `detect_all() -> Vec<Backend>` | Returns `[Cpu, ...]` plus every detected runtime in fixed order. |

Each probe caches its result in a `OnceLock` so repeated calls are O(1).

## Device properties

Availability is one question; how wide the device is, is another. `cuda_sm_count` answers the second for NVIDIA hardware.

```rust
pub fn cuda_sm_count(device_ordinal: usize) -> Option<u32>
```

Returns the streaming-multiprocessor count of the CUDA device at `device_ordinal`. `Some(n)` is always at least 1. `None` is the answer when the host has no loadable CUDA driver, when the ordinal names no device, when the driver refuses the query, or when the build has neither the `cuda-reference` nor the `gpu-peer` feature - the same graceful degradation the probes give, so a caller on a machine without a GPU gets nothing to size by instead of an error.

The read is a `cuDeviceGetAttribute` on `CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT`, gated behind `cuda_available()` so no driver symbol is touched on a host that cannot resolve one. It needs no CUDA context, so it can run before [`GpuPeer::init`](GPU-Peer-Reference.md) and produce the config that call consumes. Unlike the probes above the result is not cached: it is a single driver call, and a caller walking several ordinals wants each one answered.

The launch geometry that covers the device once is this count divided across the lanes in use - see [block teams per lane](GPU-Peer-Reference.md#block-teams-per-lane) for the worked sizing.

## Built-in backends

### `CpuBackend`

Always-available default. Wraps the existing work-stealing arena. `dispatch_parallel_for` routes through [`for_each_chunk_indexed_min_leaf`](Sched-Module-Reference.md#par_iter); `dispatch_one` spawns an OS thread. Kernel methods return `NotSupported`.

### `CudaBackend` (feature `cuda-reference`)

cudarc-backed reference NVIDIA backend. Spawns no subprocess; loads `libcuda` dynamically through cudarc's `dynamic-loading` feature. `register_kernel` accepts PTX text; `dispatch_kernel` launches through cudarc's safe `launch_builder` / `launch` API. See [Reference Backends](Reference-Backends-CUDA-And-TPU.md#cudabackend) for the full walkthrough.

### `TpuJaxBackend` (feature `tpu-jax-reference`)

Python-JAX subprocess bridge. Spawns `python3` (or `python`) running the embedded `tpu_jax_bridge.py` script and talks to it over line-oriented JSON on stdin/stdout. `register_kernel` accepts Python source defining a function; `dispatch_kernel` JIT-compiles and launches via JAX. See [Reference Backends](Reference-Backends-CUDA-And-TPU.md#tpujaxbackend) for the protocol details.

### `WasmBackend` (feature `wasm-reference`)

wasmtime-backed WebAssembly sandbox. Pure-Rust dep (cranelift + runtime); no system shared library at build or run time. `WasmBackend::new()` or `::with_device(id)` constructs an engine and spawns a persistent worker thread for the `dispatch_one` path. `register_kernel(name, source)` interprets `source` as the bytes of a `.wasm` module, instantiates it via `Module::new` / `Instance::new`, and resolves the exported function with the given `name`. `dispatch_kernel(handle, _count, args)` translates each `KernelArg` to a `wasmtime::Val` and calls the function; the `count` parameter is ignored because WASM kernels are single-threaded scalar (parallel fan-out happens host-side via `dispatch_parallel_for`, which routes the closure body through the Flynnel arena). `KernelArg::HostSlice` returns `NotSupported`; the reference impl keeps the surface minimal to mirror the CUDA reference's "pass scalars, caller manages buffers" contract. Capabilities: `simt_width = 1`, `launch_latency_ns = 5000` (cranelift call setup + sandbox entry), `h2d_bw_bytes_per_sec = 0`. See [Reference Backends](Reference-Backends-CUDA-And-TPU.md#wasmbackend) for the full walkthrough.

### `SharedMemoryChaseLevBackend` (feature `shared-memory-worker-reference`)

Dispatch into one or more peer worker *processes* over a memory-mapped Chase-Lev work-stealing deque + memory-mapped latch arena. No SDK or runtime library; the dep is `memmap2`, which is pure Rust. The wire format is `RemoteJobSlot { closure_id, args_inline[48], latch_offset }`; each peer process pre-registers a `closure_id -> handler` mapping at startup via `pass_registry::register`, so the wire never carries closure code (which is unsound across address spaces). `register_kernel(name, _)` derives the wire id deterministically from `hash_name(name)` so peers agree without coordination. Per-call cost is 342-881 ns on Zen+ depending on the coherence tier; faster than `std::sync::mpsc` in every measured pinning tier and 25-60x faster than pipe-based IPC.

`dispatch_one` panics on this backend because Rust closures cannot cross process boundaries; route through `register_kernel` + `dispatch_kernel`. `dispatch_parallel_for` is a no-op for the same reason; cross-process fan-out happens by attaching multiple peer processes to the same deque, not by issuing a single parallel-for. See [Shared-Memory Worker Backend](../explanation/Shared-Memory-Worker-Backend.md) for the architecture and the [how-to](../how-to/How-To-Use-The-Shared-Memory-Worker-Backend.md) for end-to-end setup.

## Worked example

```rust
use std::sync::Arc;
use flynnel::{Backend, DispatchBackend, JobPlan, KernelArg, register_backend, join_hybrid};
use flynnel::backend::detect;

fn main() {
    // 1. What's available?
    println!("Detected: {:?}", detect::detect_all());
    println!("CUDA runtime loadable: {}", detect::cuda_available());
    println!("CUDA device 0 multiprocessors: {:?}", detect::cuda_sm_count(0));

    // 2. Consumer-supplied backend.
    struct MyBackend;
    impl DispatchBackend for MyBackend {
        fn id(&self) -> Backend { Backend::Custom(42) }
        fn capabilities(&self) -> flynnel::BackendCapabilities {
            flynnel::BackendCapabilities::cpu_defaults()
        }
        fn dispatch_parallel_for(&self, count: u32, work: &(dyn Fn(u32) + Send + Sync)) {
            for i in 0..count { work(i); }
        }
        fn dispatch_one(&self, work: Box<dyn FnOnce() + Send>) {
            std::thread::spawn(work);
        }
    }
    register_backend(Arc::new(MyBackend));

    // 3. Route a job to it.
    let plan = JobPlan::new(8, 1024).with_backend(Backend::Custom(42));
    let (cpu_result, gpu_result) = join_hybrid(
        &plan,
        || (0..512).sum::<u64>(),
        || (512..1024).sum::<u64>(),
    );
    assert_eq!(cpu_result + gpu_result, (0..1024).sum::<u64>());
}
```

For more, see [How To Write A Backend](How-To-Write-A-Backend.md).
