//! E2E demo of the auto accelerator router on a real CUDA device.
//!
//! Registers one op in both forms - a CPU Newton-sqrt loop and the
//! `kernels/newton_sqrt.ptx` kernel - then lets `dispatch_accel`
//! route: the static cost gate keeps sub-breakeven batches on the
//! CPU, the cold bucket races both sides, and the warm bucket
//! exploits whichever side measured faster. Every round verifies
//! the fresh result against the expected sqrt.
//!
//! Exits cleanly with a message when no NVIDIA device is present.
//!
//! Run: cargo run --release --example accel_route_demo --features cuda-reference

use std::sync::Arc;

use cudarc::driver::DevicePtr;
use flynnel::backend::cuda::CudaBackend;
use flynnel::{
    Backend, DispatchBackend, JobPlan, KernelArg, Placement, bind_accel_kernel,
    dispatch_accel, register_accel_op, register_backend,
};

const ITERS: i32 = 12;

/// The CPU form of the op. ABI (cpu_args): [U64 host_ptr, I32 n,
/// I32 iters]; data[i] <- newton_sqrt(data[i]) in place, matching
/// the kernel's per-element iteration.
fn cpu_newton_sqrt(_count: u32, args: &[KernelArg<'_>]) {
    let (ptr, n, iters) = match args {
        [KernelArg::U64(p), KernelArg::I32(n), KernelArg::I32(it)] => {
            (*p as *mut f32, *n as usize, *it)
        }
        other => panic!("cpu_newton_sqrt ABI mismatch: {other:?}"),
    };
    // SAFETY: the demo passes the address and length of a live,
    // exclusively-owned Vec<f32>; dispatch_accel blocks until this
    // implementation returns, so the borrow cannot outlive the Vec.
    let data = unsafe { std::slice::from_raw_parts_mut(ptr, n) };
    for x in data.iter_mut() {
        let target = *x;
        if target <= 0.0 {
            *x = 0.0;
            continue;
        }
        let mut v = target;
        for _ in 0..iters {
            v = 0.5 * (v + target / v);
        }
        *x = v;
    }
}

fn main() {
    let backend = match CudaBackend::new() {
        Ok(b) => Arc::new(b),
        Err(e) => {
            println!("no CUDA device available ({e}); exiting cleanly");
            return;
        }
    };
    register_backend(backend.clone());
    println!("CUDA backend registered: {:?}", backend.id());

    let ptx = include_str!("../kernels/newton_sqrt.ptx");
    let op = register_accel_op("newton_sqrt", 4, cpu_newton_sqrt);
    bind_accel_kernel(op, Backend::Cuda { device_id: 0 }, "newton_sqrt", ptx.as_bytes())
        .expect("PTX registration on the live device");

    // Phase A: sub-breakeven batch with an authoritative cost hint.
    // 64 items at 50 ns cannot amortize a ~10 us launch; the gate
    // must keep it on the CPU without touching the device.
    {
        let n = 64usize;
        let mut host: Vec<f32> = (1..=n).map(|i| (i as f32) * (i as f32)).collect();
        let plan = JobPlan::bare(0, n as u32)
            .with_backend(Backend::Cuda { device_id: 0 })
            .with_estimated_per_item_ns(50);
        let report = dispatch_accel(
            &plan,
            op,
            n as u32,
            &[
                KernelArg::U64(host.as_mut_ptr() as u64),
                KernelArg::I32(n as i32),
                KernelArg::I32(ITERS),
            ],
            &[],
        );
        println!(
            "phase A (n={n}, 50 ns/item hint): placement={:?} gate_blocked={}",
            report.placement, report.gate_blocked
        );
        assert!(report.gate_blocked, "sub-breakeven work must be gated to CPU");
        verify(&host, 42);
    }

    // Phase B: a batch large enough that routing is worth learning.
    // Round 1 races both sides; later rounds exploit the winner.
    let n = 1 << 18;
    let stream = backend.stream().clone();
    let mut placements: Vec<Placement> = Vec::new();
    for round in 0..10 {
        let mut host: Vec<f32> = (1..=n).map(|i| (i as f32) * (i as f32)).collect();
        let device_buf = stream.clone_htod(&host).expect("H2D upload");
        let (dev_ptr, _guard) = device_buf.device_ptr(&stream);

        let plan = JobPlan::bare(0, n as u32).with_backend(Backend::Cuda { device_id: 0 });
        let report = dispatch_accel(
            &plan,
            op,
            n as u32,
            &[
                KernelArg::U64(host.as_mut_ptr() as u64),
                KernelArg::I32(n as i32),
                KernelArg::I32(ITERS),
            ],
            &[
                KernelArg::DevicePtr(dev_ptr as usize),
                KernelArg::I32(n as i32),
                KernelArg::I32(ITERS),
            ],
        );
        drop(_guard);

        // Read back whichever side holds the fresh result.
        let fresh: Vec<f32> = match report.placement {
            Placement::Cpu => host,
            Placement::Backend | Placement::Race => {
                let mut out = vec![0f32; n];
                stream.memcpy_dtoh(&device_buf, &mut out).expect("D2H");
                out
            }
        };
        verify(&fresh, 42);
        verify(&fresh, 511);
        println!(
            "phase B round {round}: placement={:?} cpu={:?}us gpu={:?}us fell_back={}",
            report.placement,
            report.cpu_ns.map(|v| v / 1_000),
            report.backend_ns.map(|v| v / 1_000),
            report.fell_back,
        );
        assert!(!report.fell_back, "kernel must launch on a live device");
        placements.push(report.placement);
    }

    assert_eq!(placements[0], Placement::Race, "cold bucket races");
    let exploited = placements[1..]
        .iter()
        .filter(|p| **p == Placement::Backend)
        .count();
    println!(
        "summary: {}/{} warm rounds exploited the GPU",
        exploited,
        placements.len() - 1
    );
    println!("accel_route_demo: OK");
}

/// data[i] held (i+1)^2; after the op it holds ~(i+1).
fn verify(data: &[f32], idx: usize) {
    let observed = data[idx];
    let expected = (idx + 1) as f32;
    let abs_err = (observed - expected).abs();
    assert!(
        abs_err < 1e-2,
        "data[{idx}] = {observed}, expected ~{expected} (err {abs_err:.3e})"
    );
}
