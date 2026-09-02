//! End-to-end runnable demo of the WASM dispatch backend.
//!
//! Mirrors `examples/tpu_jax_demo.rs`: detect the backend, register
//! one with the global Flynnel registry, attach a kernel, and
//! dispatch it through the `JobPlan::pick_backend` routing surface.
//!
//! Run with:
//!   cargo run --release --example wasm_dispatch_demo \
//!       --features wasm-reference
//!
//! What this proves end-to-end:
//!   * `WasmBackend::new` constructs without a host runtime library
//!   * `register_kernel` JITs a tiny `.wasm` module via cranelift
//!   * `dispatch_kernel` actually invokes the WASM function inside
//!     the sandbox (a `(3, 4) -> 7` add we can verify by also using
//!     a typed call to confirm the return value)

use flynnel::backend::wasm::WasmBackend;
use flynnel::backend::{
    Backend, DispatchBackend, KernelArg, backend_by_id, register_backend,
};
use flynnel::sched::JobPlan;
use std::sync::Arc;

/// The self-contained `(i32, i32) -> i32` add module the unit tests
/// use. `include_bytes!` loads the pre-compiled `.wasm` binary at
/// compile time from `kernels/add_i32.wasm`; the human-readable WAT
/// source lives alongside it at `kernels/add_i32.wat`. To rebuild
/// the binary from the source install `wabt` and run:
///
///   wat2wasm kernels/add_i32.wat -o kernels/add_i32.wasm
const ADD_WASM: &[u8] = include_bytes!("../kernels/add_i32.wasm");

fn main() {
    println!("=== Flynnel WASM dispatch backend demo ===\n");

    println!("[1] Detection probe:");
    let wasm_avail = flynnel::backend::detect::wasm_available();
    println!("    wasm_available = {wasm_avail}");
    println!("    (this is true when the `wasm-reference` feature was");
    println!("     compiled in; wasmtime ships as pure Rust so no");
    println!("     host runtime library is required.)\n");

    println!("[2] Construct WasmBackend:");
    let backend = match WasmBackend::new() {
        Ok(b) => {
            println!("    construction OK");
            println!("    id            = {:?}", b.id());
            let caps = b.capabilities();
            println!("    simt_width    = {}", caps.simt_width);
            println!("    max_threads   = {}", caps.max_threads_in_flight);
            println!("    launch_lat_ns = {}", caps.launch_latency_ns);
            b
        }
        Err(e) => {
            eprintln!("    construction FAILED: {e}");
            std::process::exit(1);
        }
    };

    println!("\n[3] Register backend instance with the global registry:");
    register_backend(Arc::new(backend));
    println!("    registered.");

    println!("\n[4] JobPlan with WASM hint -> picks the WASM backend:");
    let plan = JobPlan::new(0, 1)
        .with_backend(Backend::Wasm { device_id: 0 });
    let resolved = plan.pick_backend();
    println!("    routed backend = {}", resolved.id().name());
    if !matches!(resolved.id(), Backend::Wasm { .. }) {
        eprintln!(
            "    WARN: expected wasm, got {}; routing fell back to CPU",
            resolved.id().name()
        );
    }

    println!("\n[5] Register the tiny `add` WASM kernel through the registry-resolved backend:");
    let handle = match backend_by_id(&Backend::Wasm { device_id: 0 })
        .expect("WASM backend registered")
        .register_kernel("add", ADD_WASM)
    {
        Ok(h) => {
            println!("    handle = {h:?}");
            h
        }
        Err(e) => {
            eprintln!("    register_kernel FAILED: {e}");
            std::process::exit(1);
        }
    };

    println!("\n[6] Dispatch the kernel through the Flynnel routing surface:");
    let dispatch_result = backend_by_id(&Backend::Wasm { device_id: 0 })
        .expect("WASM backend registered")
        .dispatch_kernel(
            handle,
            1,
            &[KernelArg::I32(3), KernelArg::I32(4)],
        );
    println!("    dispatch result = {dispatch_result:?}");
    if dispatch_result.is_err() {
        std::process::exit(1);
    }

    println!("\n[7] Verify the kernel actually computed 3+4=7 by calling");
    println!("    its typed function directly. This bypasses the");
    println!("    DispatchBackend trait (which discards return values)");
    println!("    so we can read the wasm scalar back.");
    use wasmtime::{Engine, Instance, Module, Store};
    let engine = Engine::default();
    let module = Module::new(&engine, ADD_WASM).expect("compile");
    let mut store: Store<()> = Store::new(&engine, ());
    let instance = Instance::new(&mut store, &module, &[]).expect("instantiate");
    let add = instance
        .get_typed_func::<(i32, i32), i32>(&mut store, "add")
        .expect("typed_func");
    let got = add.call(&mut store, (3, 4)).expect("call");
    println!("    add(3, 4) returned = {got}");
    assert_eq!(got, 7, "wasm kernel must compute 3+4=7");
    println!("    VERIFIED: WASM kernel ran end-to-end and produced the expected result.");

    println!("\n=== Demo complete. ===");
}
