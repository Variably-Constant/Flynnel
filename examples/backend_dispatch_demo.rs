//! End-to-end demo of Flynnel's backend dispatch fabric.
//!
//! Run with `cargo run --example backend_dispatch_demo`.
//!
//! Exercises:
//! 1. Detection probes (cuda / rocm / metal / tpu / ane availability).
//! 2. Default CPU backend auto-registration + parallel_for.
//! 3. Consumer-supplied stub backend registration + dispatch.
//! 4. Opaque kernel handle path: register a stub kernel, launch it,
//!    observe the recorded launch args.
//! 5. MIMT hybrid join: CPU half + backend half run concurrently.

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use flynnel::backend::detect;
use flynnel::{
    Backend, BackendCapabilities, BackendError, DispatchBackend, JobPlan, KernelArg,
    KernelHandle, backends, cpu_backend, join_hybrid, register_backend,
};

/// Stub backend that records every dispatch call to atomic counters so
/// the demo can observe the dispatch fabric routed work correctly.
struct ObservableStub {
    id: Backend,
    parallel_for_count: Arc<AtomicU32>,
    dispatch_one_count: Arc<AtomicU32>,
    kernel_launches: Arc<AtomicU32>,
    last_kernel_count: Arc<AtomicU32>,
    last_kernel_arg_count: Arc<AtomicU32>,
}

impl DispatchBackend for ObservableStub {
    fn id(&self) -> Backend {
        self.id
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            simt_width: 32,
            max_threads_in_flight: 65536,
            launch_latency_ns: 7500,
            h2d_bw_bytes_per_sec: 20_000_000_000,
        }
    }

    fn dispatch_parallel_for(&self, count: u32, work: &(dyn Fn(u32) + Send + Sync)) {
        self.parallel_for_count.fetch_add(1, Ordering::SeqCst);
        for i in 0..count {
            work(i);
        }
    }

    fn dispatch_one(&self, work: Box<dyn FnOnce() + Send>) {
        self.dispatch_one_count.fetch_add(1, Ordering::SeqCst);
        std::thread::spawn(work);
    }

    fn register_kernel(
        &self,
        _name: &str,
        _source: &[u8],
    ) -> Result<KernelHandle, BackendError> {
        Ok(KernelHandle(0xCAFEBABE))
    }

    fn dispatch_kernel(
        &self,
        handle: KernelHandle,
        count: u32,
        args: &[KernelArg<'_>],
    ) -> Result<(), BackendError> {
        assert_eq!(handle.0, 0xCAFEBABE, "should receive the registered handle");
        self.kernel_launches.fetch_add(1, Ordering::SeqCst);
        self.last_kernel_count.store(count, Ordering::SeqCst);
        self.last_kernel_arg_count
            .store(args.len() as u32, Ordering::SeqCst);
        Ok(())
    }
}

fn main() {
    println!("=== Flynnel backend dispatch demo ===\n");

    println!("[1] Runtime detection probes:");
    for detected in detect::detect_all() {
        println!("    detected: {}", detected.name());
    }
    println!(
        "    cuda_available  = {}",
        detect::cuda_available()
    );
    println!(
        "    rocm_available  = {}",
        detect::rocm_available()
    );
    println!(
        "    metal_available = {}",
        detect::metal_available()
    );
    println!(
        "    tpu_available   = {}",
        detect::tpu_available()
    );
    println!(
        "    ane_available   = {}",
        detect::ane_available()
    );
    println!();

    println!("[2] Default CPU backend auto-registration:");
    let cpu = cpu_backend();
    println!("    cpu id          = {}", cpu.id().name());
    let caps = cpu.capabilities();
    println!(
        "    cpu simt_width  = {} (scalar)",
        caps.simt_width
    );
    println!(
        "    cpu threads     = {}",
        caps.max_threads_in_flight
    );
    println!(
        "    cpu launch_lat  = {} ns",
        caps.launch_latency_ns
    );
    println!();

    println!("[3] Run dispatch_parallel_for on CPU backend (touches every index 0..512):");
    let touches = (0..512).map(|_| AtomicU32::new(0)).collect::<Vec<_>>();
    let tref = &touches;
    cpu.dispatch_parallel_for(512, &|i| {
        tref[i as usize].fetch_add(1, Ordering::Relaxed);
    });
    let touched = touches
        .iter()
        .filter(|c| c.load(Ordering::Relaxed) == 1)
        .count();
    let total_touches: u32 = touches.iter().map(|c| c.load(Ordering::Relaxed)).sum();
    println!(
        "    indices touched exactly once = {touched} / 512 (total touches = {total_touches})"
    );
    assert_eq!(touched, 512, "every index must be touched exactly once");
    println!();

    println!("[4] Register an observable stub backend (id = Custom(0xC0FFEE)):");
    let stub_id = Backend::Custom(0x00C0_FFEE);
    let parallel_for_count = Arc::new(AtomicU32::new(0));
    let dispatch_one_count = Arc::new(AtomicU32::new(0));
    let kernel_launches = Arc::new(AtomicU32::new(0));
    let last_kernel_count = Arc::new(AtomicU32::new(0));
    let last_kernel_arg_count = Arc::new(AtomicU32::new(0));
    let stub = Arc::new(ObservableStub {
        id: stub_id,
        parallel_for_count: Arc::clone(&parallel_for_count),
        dispatch_one_count: Arc::clone(&dispatch_one_count),
        kernel_launches: Arc::clone(&kernel_launches),
        last_kernel_count: Arc::clone(&last_kernel_count),
        last_kernel_arg_count: Arc::clone(&last_kernel_arg_count),
    });
    register_backend(stub);
    println!("    registry now has {} backend(s)", backends().len());
    println!();

    println!("[5] Route a job via JobPlan::with_backend(stub_id):");
    let plan = JobPlan::new(8, 1024).with_backend(stub_id);
    let routed = plan.pick_backend();
    println!("    plan picked backend = {}", routed.id().name());
    routed.dispatch_parallel_for(128, &|_| {});
    println!(
        "    stub.parallel_for_count = {}",
        parallel_for_count.load(Ordering::SeqCst)
    );
    assert_eq!(parallel_for_count.load(Ordering::SeqCst), 1);
    println!();

    println!("[6] Opaque kernel handle path:");
    let handle = routed
        .register_kernel("noop_kernel", b"// fake ptx, stub ignores it")
        .expect("stub accepts register_kernel");
    println!("    registered handle = 0x{:X}", handle.0);
    let launch_result = routed.dispatch_kernel(
        handle,
        4096,
        &[
            KernelArg::U32(0xABCD),
            KernelArg::F32(5.5),
            KernelArg::DevicePtr(0xDEADBEEF),
        ],
    );
    println!(
        "    launch_result   = {:?}",
        launch_result.as_ref().map(|_| "Ok")
    );
    println!(
        "    last_kernel_count    = {}",
        last_kernel_count.load(Ordering::SeqCst)
    );
    println!(
        "    last_kernel_arg_count = {}",
        last_kernel_arg_count.load(Ordering::SeqCst)
    );
    assert_eq!(last_kernel_count.load(Ordering::SeqCst), 4096);
    assert_eq!(last_kernel_arg_count.load(Ordering::SeqCst), 3);
    println!();

    println!("[7] MIMT hybrid join: CPU half + stub-backend half concurrently:");
    let (cpu_sum, gpu_sum) = join_hybrid(
        &plan,
        || (0..512).sum::<u64>(),
        || (512..1024).sum::<u64>(),
    );
    println!("    cpu_sum = {cpu_sum}");
    println!("    gpu_sum = {gpu_sum}");
    println!("    total   = {} (expected {})", cpu_sum + gpu_sum, (0..1024).sum::<u64>());
    assert_eq!(cpu_sum + gpu_sum, (0..1024).sum::<u64>());
    println!(
        "    stub.dispatch_one_count = {}",
        dispatch_one_count.load(Ordering::SeqCst)
    );
    assert_eq!(dispatch_one_count.load(Ordering::SeqCst), 1);
    println!();

    println!("=== All assertions passed; demo complete. ===");
}
