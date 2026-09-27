//! A light kernel over f64 elements, fed and read through device memory
//! and copies, against the same kernel over mapped host memory.
//!
//! One job is what a caller does per batch: write `n` inputs on the
//! host, get them to the kernel, run it, and read the `n` answers back
//! on the host.
//!
//! - `copies`, the baseline: fill a pageable `Vec`, `copy_in` to a
//!   `DeviceBuffer`, launch over it and a second `DeviceBuffer`, then
//!   `copy_out` into a pageable `Vec`, which returns once the answers
//!   are there.
//! - `mapped`: fill an input `MappedBuffer`'s host slice, launch over it
//!   and an output `MappedBuffer`, then read the output's host slice,
//!   which waits for the launch.
//!
//! Both arms keep their buffers across jobs, so neither allocates per
//! job; both sum the answers, so neither read is skipped; and both are
//! checked for the right answers before anything is timed. Each size
//! runs quiet, then beside threads spinning on half and on all of the
//! logical processors.
//!
//!   cargo bench --features cuda-reference --bench cuda_mapped_memory

use std::hint::black_box;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

use flynnel::backend::cuda::{CudaBackend, DeviceBuffer, MappedBuffer};
use flynnel::backend::{BackendError, DispatchBackend, KernelArg, KernelHandle};

const PTX: &str = include_str!("../kernels/twice_plus_one.ptx");

const SIZES: [usize; 3] = [1_000, 100_000, 1_000_000];

/// The inputs a job writes: element `i` is `i` plus the job's `seed`, so
/// consecutive jobs write different values.
fn fill(out: &mut [f64], seed: f64) {
    for (i, v) in out.iter_mut().enumerate() {
        *v = i as f64 + seed;
    }
}

/// Whether `answers` are the kernel's `2 * x + 1` of `fill(_, seed)`.
fn answers_hold(answers: &[f64], seed: f64) -> bool {
    answers
        .iter()
        .enumerate()
        .all(|(i, &v)| v.to_bits() == (2.0 * (i as f64 + seed) + 1.0).to_bits())
}

/// Threads spinning until dropped.
struct Spinners {
    stop: Arc<AtomicBool>,
    threads: Vec<std::thread::JoinHandle<u64>>,
}

impl Spinners {
    fn start(count: usize) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let threads = (0..count)
            .map(|_| {
                let stop = Arc::clone(&stop);
                std::thread::spawn(move || {
                    let mut spins = 0u64;
                    while !stop.load(Ordering::Relaxed) {
                        spins = black_box(spins.wrapping_add(1));
                    }
                    spins
                })
            })
            .collect();
        Self { stop, threads }
    }
}

impl Drop for Spinners {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        for thread in self.threads.drain(..) {
            if let Err(panicked) = thread.join() {
                panic!("a spinner panicked: {panicked:?}");
            }
        }
    }
}

/// The buffers one size's jobs reuse, in both arms.
struct Buffers {
    input: Vec<f64>,
    output: Vec<f64>,
    device_in: DeviceBuffer<f64>,
    device_out: DeviceBuffer<f64>,
    mapped_in: MappedBuffer<f64>,
    mapped_out: MappedBuffer<f64>,
}

impl Buffers {
    fn new(backend: &CudaBackend, n: usize) -> Self {
        Self {
            input: vec![0.0; n],
            output: vec![0.0; n],
            device_in: backend.alloc_zeroed(n).expect("device input"),
            device_out: backend.alloc_zeroed(n).expect("device output"),
            mapped_in: backend.map_host(n).expect("mapped input"),
            mapped_out: backend.map_host(n).expect("mapped output"),
        }
    }

    fn copies(&mut self, backend: &CudaBackend, kernel: KernelHandle, seed: f64) -> f64 {
        let n = self.input.len() as u32;
        fill(&mut self.input, seed);
        backend
            .copy_in(&self.input, &mut self.device_in)
            .expect("copy in");
        backend
            .dispatch_kernel(
                kernel,
                n,
                &[
                    self.device_in.arg(),
                    self.device_out.arg(),
                    KernelArg::U32(n),
                ],
            )
            .expect("launch");
        backend
            .copy_out(&self.device_out, &mut self.output)
            .expect("copy out");
        self.output.iter().sum()
    }

    fn mapped(&mut self, backend: &CudaBackend, kernel: KernelHandle, seed: f64) -> f64 {
        let n = self.mapped_in.len() as u32;
        fill(self.mapped_in.as_mut_slice().expect("input slice"), seed);
        backend
            .dispatch_kernel(
                kernel,
                n,
                &[
                    self.mapped_in.arg(),
                    self.mapped_out.arg(),
                    KernelArg::U32(n),
                ],
            )
            .expect("launch");
        self.mapped_out
            .as_slice()
            .expect("output slice")
            .iter()
            .sum()
    }
}

fn bench_all(c: &mut Criterion) {
    let backend = match CudaBackend::new() {
        Ok(backend) => backend,
        Err(BackendError::DeviceUnavailable(device)) => {
            eprintln!(
                "no usable CUDA device ({}): nothing measured",
                device.name()
            );
            return;
        }
        Err(other) => panic!("the CUDA backend refused for a reason other than no device: {other}"),
    };
    let kernel = backend
        .register_ptx("twice_plus_one", PTX)
        .expect("twice_plus_one loads");
    let device = backend.device_name().expect("device name");
    let logical = std::thread::available_parallelism()
        .map(std::num::NonZeroUsize::get)
        .expect("logical processor count");
    eprintln!("cuda_mapped_memory device={device} logical_processors={logical}");

    let mut sizes: Vec<(usize, Buffers)> = SIZES
        .iter()
        .map(|&n| (n, Buffers::new(&backend, n)))
        .collect();
    for (n, buffers) in &mut sizes {
        buffers.copies(&backend, kernel, 7.0);
        assert!(answers_hold(&buffers.output, 7.0), "copies at {n}");
        buffers.mapped(&backend, kernel, 7.0);
        let answers = buffers.mapped_out.as_slice().expect("output slice");
        assert!(answers_hold(answers, 7.0), "mapped at {n}");
    }

    for (label, spinners) in [("quiet", 0), ("half", logical / 2), ("all", logical)] {
        let load = Spinners::start(spinners);
        let mut group = c.benchmark_group(format!("cuda_memory_{label}"));
        group.warm_up_time(Duration::from_secs(1));
        group.measurement_time(Duration::from_secs(3));
        for (n, buffers) in &mut sizes {
            group.throughput(Throughput::Elements(*n as u64));
            group.bench_function(BenchmarkId::new("copies", *n), |b| {
                let mut seed = 0.0;
                b.iter(|| {
                    seed += 1.0;
                    black_box(buffers.copies(&backend, kernel, seed))
                });
            });
            group.bench_function(BenchmarkId::new("mapped", *n), |b| {
                let mut seed = 0.0;
                b.iter(|| {
                    seed += 1.0;
                    black_box(buffers.mapped(&backend, kernel, seed))
                });
            });
        }
        group.finish();
        drop(load);
    }
}

criterion_group!(benches, bench_all);
criterion_main!(benches);
