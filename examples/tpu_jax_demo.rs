//! End-to-end demo of the TPU JAX reference backend.
//!
//! Build + run with:
//!
//! ```text
//! cargo run --example tpu_jax_demo --features tpu-jax-reference --release
//! ```
//!
//! On a host with Python + JAX + a TPU device, the demo registers
//! a tiny JAX-jitted kernel and dispatches it through Flynnel's
//! routing fabric. On any other host (no Python, no JAX, no TPU)
//! construction returns `DeviceUnavailable` and the demo prints the
//! diagnosis - which IS the graceful-degradation contract Flynnel
//! advertises.

use flynnel::backend::detect;
use flynnel::backend::tpu_jax::TpuJaxBackend;
use flynnel::{
    Backend, BackendError, DispatchBackend, JobPlan, KernelArg, register_backend,
};
use std::sync::Arc;

fn main() {
    println!("=== Flynnel TPU JAX bridge demo ===\n");

    println!("[1] Detection probe:");
    println!("    tpu_available = {}", detect::tpu_available());
    println!(
        "    (probe checks TPU_NAME env + /dev/accel* device files;"
    );
    println!("     it does NOT verify python/jax are installed)\n");

    println!("[2] Try to construct TpuJaxBackend:");
    let backend = match TpuJaxBackend::new() {
        Ok(b) => {
            println!("    construction OK");
            println!("    id            = {:?}", b.id());
            println!("    devices       = {:?}", b.devices());
            let caps = b.capabilities();
            println!("    simt_width    = {}", caps.simt_width);
            println!("    launch_lat_ns = {}", caps.launch_latency_ns);
            Arc::new(b) as Arc<dyn DispatchBackend>
        }
        Err(BackendError::DeviceUnavailable(b)) => {
            println!(
                "    construction returned DeviceUnavailable({})",
                b.name()
            );
            println!(
                "    (this is the graceful-degradation path: bridge or jax not present)"
            );
            println!(
                "\n[3..5] Skipping JAX-bound steps; falling back to CPU routing.\n"
            );
            println!(
                "[6] JobPlan with TPU hint but no registered backend -> CPU fallback:"
            );
            let plan = JobPlan::new(8, 64)
                .with_backend(Backend::Tpu { device_id: 0 });
            let routed = plan.pick_backend();
            println!(
                "    routed backend = {} (expected: cpu, the fallback)",
                routed.id().name()
            );
            assert_eq!(routed.id(), Backend::Cpu);
            println!("\n=== Graceful degradation verified. ===");
            return;
        }
        Err(other) => {
            eprintln!("Unexpected error constructing TpuJaxBackend: {other}");
            std::process::exit(2);
        }
    };

    println!("\n[3] Register backend instance with the global registry:");
    register_backend(backend);
    println!("    registered.");

    println!("\n[4] JobPlan with TPU hint -> picks the TPU backend:");
    let plan = JobPlan::new(8, 4096).with_backend(Backend::Tpu { device_id: 0 });
    let routed = plan.pick_backend();
    println!("    routed backend = {}", routed.id().name());
    assert_eq!(
        routed.id(),
        Backend::Tpu { device_id: 0 },
        "routing should pick the registered TPU backend"
    );

    println!("\n[5] Register a tiny JAX kernel:");
    // count is bridge-side static (jax.jit static_argnums=(0,)), so it
    // arrives as a concrete Python int suitable for sizing jnp.arange.
    // scalar (and every other KernelArg) is a JAX traced array - use
    // it in arithmetic directly without int() casts, which would raise
    // ConcretizationTypeError at jit-compile time.
    //
    // `include_str!` loads the JAX kernel source at compile time
    // from `kernels/double_then_sum.py`; the file is the human-
    // readable reference companion to the CUDA `.ptx` and WASM
    // `.wat` kernels that also live under `kernels/`.
    let py_source: &str = include_str!("../kernels/double_then_sum.py");
    let handle = routed
        .register_kernel("double_then_sum", py_source.as_bytes())
        .expect("kernel register should succeed when bridge is live");
    println!("    handle = 0x{:X}", handle.0);

    println!("\n[6] Dispatch the kernel through Flynnel routing:");
    let launch = routed.dispatch_kernel(
        handle,
        16, // count
        &[KernelArg::I32(7)],
    );
    println!("    dispatch result = {launch:?}");
    launch.expect("dispatch should succeed");

    println!("\n=== Demo complete. ===");
}
