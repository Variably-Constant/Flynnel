//! Probe whether the CUDA backend constructs successfully on the
//! current host AND can launch a real Newton-sqrt PTX kernel with
//! H2D + kernel + D2H round-trip.
//!
//! Run with:
//!
//! ```text
//! cargo run --example cuda_probe --release --features cuda-reference
//! ```

use cudarc::driver::DevicePtr;
use flynnel::backend::cuda::CudaBackend;
use flynnel::backend::detect;
use flynnel::{BackendError, DispatchBackend, KernelArg};

fn main() {
    println!("=== Flynnel CUDA probe + kernel round-trip ===");
    println!("detect::cuda_available() = {}", detect::cuda_available());
    println!();

    let backend = match CudaBackend::new() {
        Ok(b) => b,
        Err(BackendError::DeviceUnavailable(_)) => {
            println!("CudaBackend::new() = DeviceUnavailable; skipping kernel test.");
            return;
        }
        Err(e) => {
            eprintln!("CudaBackend init failed: {e}");
            std::process::exit(2);
        }
    };
    println!("CudaBackend constructed, device 0.");

    let ptx = include_str!("../kernels/newton_sqrt.ptx");
    let handle = backend
        .register_kernel("newton_sqrt", ptx.as_bytes())
        .unwrap_or_else(|e| {
            eprintln!("register_kernel failed: {e}");
            std::process::exit(2);
        });
    println!("Kernel 'newton_sqrt' registered, handle = 0x{:X}", handle.0);

    const N: usize = 1024;
    const ITERS: i32 = 10;

    let stream = backend.stream();

    let mut host_data: Vec<f32> = (1..=N).map(|i| (i as f32) * (i as f32)).collect();
    let device_buf = stream
        .clone_htod(&host_data)
        .unwrap_or_else(|e| {
            eprintln!("H2D failed: {e:?}");
            std::process::exit(2);
        });

    // Get the raw device pointer to thread through Flynnel's
    // DispatchBackend::dispatch_kernel API. The cudarc DevicePtr
    // trait returns (CUdeviceptr, SyncOnDrop). Hold the SyncOnDrop
    // guard alive across the launch.
    let (dev_ptr, _sync_guard) = device_buf.device_ptr(stream);
    backend
        .dispatch_kernel(
            handle,
            N as u32,
            &[
                KernelArg::DevicePtr(dev_ptr as usize),
                KernelArg::I32(N as i32),
                KernelArg::I32(ITERS),
            ],
        )
        .unwrap_or_else(|e| {
            eprintln!("dispatch_kernel failed: {e}");
            std::process::exit(2);
        });
    stream.synchronize().unwrap_or_else(|e| {
        eprintln!("sync failed: {e:?}");
        std::process::exit(2);
    });
    drop(_sync_guard);

    stream
        .memcpy_dtoh(&device_buf, &mut host_data)
        .unwrap_or_else(|e| {
            eprintln!("D2H failed: {e:?}");
            std::process::exit(2);
        });

    // data[i] = sqrt((i+1)^2) = i+1 after 10 Newton iterations.
    let i_test = 42;
    let observed = host_data[i_test];
    let expected = (i_test + 1) as f32;
    let abs_err = (observed - expected).abs();
    println!(
        "sample: data[42] = {observed:.6}, expected ~{expected:.6}, abs_err = {abs_err:.3e}"
    );
    if abs_err > 1e-3 {
        eprintln!("kernel result diverged from expected sqrt value");
        std::process::exit(2);
    }

    // Time a hot loop to characterize per-launch overhead.
    use std::time::Instant;
    let host2: Vec<f32> = (1..=N).map(|i| (i as f32) * (i as f32)).collect();
    let buf2 = stream
        .clone_htod(&host2)
        .unwrap_or_else(|e| {
            eprintln!("H2D2 failed: {e:?}");
            std::process::exit(2);
        });
    let (dev_ptr2, _sync_guard2) = buf2.device_ptr(stream);
    let t0 = Instant::now();
    for _ in 0..100 {
        backend
            .dispatch_kernel(
                handle,
                N as u32,
                &[
                    KernelArg::DevicePtr(dev_ptr2 as usize),
                    KernelArg::I32(N as i32),
                    KernelArg::I32(ITERS),
                ],
            )
            .unwrap_or_else(|e| {
                eprintln!("hot-loop dispatch failed: {e}");
                std::process::exit(2);
            });
    }
    stream.synchronize().unwrap_or_else(|e| {
        eprintln!("hot-loop sync failed: {e:?}");
        std::process::exit(2);
    });
    let elapsed = t0.elapsed();
    println!(
        "100 launches + sync = {:.2} ms ({:.1} us per launch)",
        elapsed.as_secs_f64() * 1000.0,
        elapsed.as_secs_f64() * 1e6 / 100.0
    );

    drop(_sync_guard2);
    drop(buf2);
    drop(device_buf);
    println!("OK.");
}
