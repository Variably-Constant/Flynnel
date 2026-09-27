//! The byte-to-value loops the GPU peer runs over fetched bytes, in the
//! `chunks_exact` form and in the `as_chunks` form, side by side in one
//! process.
//!
//! - `f64`: eight-byte little-endian chunks to `f64`, the loop the peer's
//!   linalg results are read back through.
//! - `i32`: four-byte chunks to `i32`, the loop LU's pivots and info
//!   codes are read back through.
//! - `add1`: four-byte chunks read as `f32`, incremented and written back
//!   in place; the `as_chunks` arm is the crate's own
//!   `gpu_peer::hybrid::add1_f32_cpu`.
//!
//! The `as_chunks` arms of `f64` and `i32` are copies of the crate's
//! private loops. Each loop's two forms are checked for bit-identical
//! answers over the same bytes before anything is timed. Each size runs
//! quiet, then beside threads spinning on half and on all of the logical
//! processors.
//!
//! Every loop's `chunks_exact` form is timed twice, the second time as
//! `<loop>_chunks_exact_control`, so each loop has three arms: the
//! baseline, the `as_chunks` form and the control. The two `chunks_exact`
//! arms run one function over one input, so the gap between them is what
//! an arm's position is worth on the host, and a gap between the two
//! forms inside that one is not a difference between them.
//!
//! A loop's three arms run back to back, starting at the arm
//! `BYTE_CHUNKS_ROTATION` names: 0 (the baseline, and the default), 1 (the
//! `as_chunks` form) or 2 (the control). Over one run at each rotation
//! every arm takes every position once, so the form's ratio to the
//! baseline and the control's are read over the same positions. Each
//! `add1` arm increments a copy of the input made just before it runs.
//!
//!   cargo bench --features gpu-peer --bench byte_chunks
//!   BYTE_CHUNKS_ROTATION=1 cargo bench --features gpu-peer --bench byte_chunks

use std::hint::black_box;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

use flynnel::gpu_peer::hybrid::add1_f32_cpu;

const SIZES: [usize; 3] = [1_000, 100_000, 1_000_000];

/// One of a loop's three arms.
#[derive(Clone, Copy)]
enum Arm {
    Baseline,
    AsChunks,
    Control,
}

impl Arm {
    fn id(self, form: &str, n: usize) -> BenchmarkId {
        let name = match self {
            Arm::Baseline => format!("{form}_chunks_exact"),
            Arm::AsChunks => format!("{form}_as_chunks"),
            Arm::Control => format!("{form}_chunks_exact_control"),
        };
        BenchmarkId::new(name, n)
    }
}

/// A loop's three arms in the order they run, starting at `rotation`.
fn arm_order(rotation: usize) -> [Arm; 3] {
    let arms = [Arm::Baseline, Arm::AsChunks, Arm::Control];
    [
        arms[rotation % 3],
        arms[(rotation + 1) % 3],
        arms[(rotation + 2) % 3],
    ]
}

/// The arm each loop starts at, from `BYTE_CHUNKS_ROTATION`: 0 when unset.
fn rotation() -> usize {
    let raw = match std::env::var("BYTE_CHUNKS_ROTATION") {
        Ok(raw) => raw,
        Err(std::env::VarError::NotPresent) => return 0,
        Err(other) => panic!("BYTE_CHUNKS_ROTATION is not readable: {other}"),
    };
    match raw.trim().parse::<usize>() {
        Ok(rotation) if rotation < 3 => rotation,
        Ok(rotation) => panic!("BYTE_CHUNKS_ROTATION must be 0, 1 or 2, not {rotation}"),
        Err(e) => panic!("BYTE_CHUNKS_ROTATION must be 0, 1 or 2, not {raw:?}: {e}"),
    }
}

/// The `chunks_exact` form of the `f64` loop, the baseline arm.
#[allow(unknown_lints, clippy::chunks_exact_to_as_chunks)]
fn f64_chunks_exact(b: &[u8]) -> Vec<f64> {
    b.chunks_exact(8)
        .map(|c| f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
        .collect()
}

fn f64_as_chunks(b: &[u8]) -> Vec<f64> {
    b.as_chunks::<8>()
        .0
        .iter()
        .map(|c| f64::from_le_bytes(*c))
        .collect()
}

/// The `chunks_exact` form of the `i32` loop, the baseline arm.
#[allow(unknown_lints, clippy::chunks_exact_to_as_chunks)]
fn i32_chunks_exact(b: &[u8]) -> Vec<i32> {
    b.chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn i32_as_chunks(b: &[u8]) -> Vec<i32> {
    b.as_chunks::<4>()
        .0
        .iter()
        .map(|c| i32::from_le_bytes(*c))
        .collect()
}

/// The `chunks_exact` form of the in-place `f32` increment, the baseline
/// arm.
#[allow(unknown_lints, clippy::chunks_exact_to_as_chunks)]
fn add1_chunks_exact(bytes: &mut [u8]) {
    for c in bytes.chunks_exact_mut(4) {
        let v = f32::from_le_bytes([c[0], c[1], c[2], c[3]]) + 1.0;
        c.copy_from_slice(&v.to_le_bytes());
    }
}

/// `n` values' little-endian bytes, finite and all different.
fn f64_bytes(n: usize) -> Vec<u8> {
    (0..n)
        .flat_map(|i| (i as f64 * 0.5 - 7.25).to_le_bytes())
        .collect()
}

fn i32_bytes(n: usize) -> Vec<u8> {
    (0..n)
        .flat_map(|i| (i as i32).wrapping_mul(-7919).to_le_bytes())
        .collect()
}

fn f32_bytes(n: usize) -> Vec<u8> {
    (0..n)
        .flat_map(|i| (i as f32 * 0.25 - 3.0).to_le_bytes())
        .collect()
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

/// One size's input bytes for each loop.
struct Inputs {
    n: usize,
    f64s: Vec<u8>,
    i32s: Vec<u8>,
    f32s: Vec<u8>,
}

/// Every loop's two forms answer the same bits over the same bytes.
fn check_forms(n: usize) {
    let f = f64_bytes(n);
    let exact: Vec<u64> = f64_chunks_exact(&f).iter().map(|v| v.to_bits()).collect();
    let chunked: Vec<u64> = f64_as_chunks(&f).iter().map(|v| v.to_bits()).collect();
    assert_eq!(exact, chunked, "f64 at {n}");
    let i = i32_bytes(n);
    assert_eq!(i32_chunks_exact(&i), i32_as_chunks(&i), "i32 at {n}");
    let mut a = f32_bytes(n);
    let mut b = a.clone();
    add1_chunks_exact(&mut a);
    add1_f32_cpu(&mut b);
    assert_eq!(a, b, "add1 at {n}");
}

fn bench_all(c: &mut Criterion) {
    let logical = std::thread::available_parallelism()
        .map(std::num::NonZeroUsize::get)
        .expect("logical processor count");
    let rotation = rotation();
    eprintln!("byte_chunks logical_processors={logical} rotation={rotation}");
    for &n in &SIZES {
        check_forms(n);
    }
    let inputs: Vec<Inputs> = SIZES
        .iter()
        .map(|&n| Inputs {
            n,
            f64s: f64_bytes(n),
            i32s: i32_bytes(n),
            f32s: f32_bytes(n),
        })
        .collect();

    for (label, spinners) in [("quiet", 0), ("half", logical / 2), ("all", logical)] {
        let load = Spinners::start(spinners);
        let mut group = c.benchmark_group(format!("byte_chunks_{label}"));
        group.warm_up_time(Duration::from_secs(1));
        group.measurement_time(Duration::from_secs(3));
        for input in &inputs {
            let n = input.n;
            group.throughput(Throughput::Elements(n as u64));
            for arm in arm_order(rotation) {
                group.bench_function(arm.id("f64", n), |b| match arm {
                    Arm::AsChunks => b.iter(|| black_box(f64_as_chunks(black_box(&input.f64s)))),
                    Arm::Baseline | Arm::Control => {
                        b.iter(|| black_box(f64_chunks_exact(black_box(&input.f64s))))
                    }
                });
            }
            for arm in arm_order(rotation) {
                group.bench_function(arm.id("i32", n), |b| match arm {
                    Arm::AsChunks => b.iter(|| black_box(i32_as_chunks(black_box(&input.i32s)))),
                    Arm::Baseline | Arm::Control => {
                        b.iter(|| black_box(i32_chunks_exact(black_box(&input.i32s))))
                    }
                });
            }
            for arm in arm_order(rotation) {
                let mut bytes = input.f32s.clone();
                group.bench_function(arm.id("add1", n), |b| match arm {
                    Arm::AsChunks => b.iter(|| add1_f32_cpu(black_box(&mut bytes))),
                    Arm::Baseline | Arm::Control => {
                        b.iter(|| add1_chunks_exact(black_box(&mut bytes)))
                    }
                });
            }
        }
        group.finish();
        drop(load);
    }
}

criterion_group!(benches, bench_all);
criterion_main!(benches);
