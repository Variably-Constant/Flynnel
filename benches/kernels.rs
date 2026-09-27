//! The declared kernels' block work: each kernel driven on the calling
//! thread over one input, quiet and beside threads spinning on every
//! logical processor.
//!
//! Every block runs on the calling thread through `drive_serial`, so a
//! cell times the kernels' own code and none of a pool's dispatch. An
//! in-place kernel works on a fresh copy of its input made outside the
//! clock. Built from two trees, the same cells compare two builds of the
//! kernels: run the base build, the tip build and the base again from a
//! second file as the copy, one process a run, and read each cell's tip
//! against the base beside the copy's.
//!
//!   cargo bench --bench kernels

use std::hint::black_box;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use criterion::{BatchSize, Criterion, Throughput, criterion_group, criterion_main};

use flynnel::kernels::{
    MapOp, MapOperands, ReduceOp, TextTransform, ZipOp, dot_product, drive_serial, histogram, map,
    prefix_sum, reduce, search_text, text_count, update_text, zip,
};

const N: usize = 1_000_000;

/// `n` values with a wide spread and some repeats, the same for a seed.
fn sample(n: usize, seed: u64) -> Vec<f64> {
    let mut state = seed;
    (0..n)
        .map(|i| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let unit = (state >> 11) as f64 / (1u64 << 53) as f64;
            if i % 97 == 0 {
                0.5
            } else {
                (unit - 0.5) * 1.0e6
            }
        })
        .collect()
}

/// About four million bytes of log-like lines.
fn text() -> String {
    let mut s = String::new();
    let mut line = 0u64;
    while s.len() < 4_000_000 {
        s.push_str(&format!("line {line} has ERROR and ERRORERROR\r\n"));
        line += 1;
    }
    s
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

fn bench_all(c: &mut Criterion) {
    let logical = std::thread::available_parallelism()
        .map(std::num::NonZeroUsize::get)
        .expect("logical processor count");
    eprintln!("kernels logical_processors={logical}");
    let x = sample(N, 0x9E37_79B9_7F4A_7C15);
    let y = sample(N, 0xD1B5_4A32_D192_ED03);
    let text = text();
    let operands = MapOperands {
        min: Some(-10.0),
        max: Some(10.0),
        factor: Some(2.5),
        addend: Some(-3.0),
    };

    for (label, spinners) in [("quiet", 0), ("all", logical)] {
        let load = Spinners::start(spinners);
        let mut group = c.benchmark_group(format!("kernels_{label}"));
        group.warm_up_time(Duration::from_millis(500));
        group.measurement_time(Duration::from_secs(2));
        group.throughput(Throughput::Elements(N as u64));
        for op in [MapOp::Sqrt, MapOp::Clamp] {
            group.bench_function(format!("map_{op:?}"), |b| {
                b.iter_batched_ref(
                    || x.clone(),
                    |data| {
                        drive_serial(map(data, op, operands).expect("the operands hold"))
                            .expect("the map finishes")
                    },
                    BatchSize::LargeInput,
                )
            });
        }
        for op in [ZipOp::Multiply, ZipOp::Min, ZipOp::Max] {
            group.bench_function(format!("zip_{op:?}"), |b| {
                b.iter_batched_ref(
                    || x.clone(),
                    |left| {
                        drive_serial(zip(left, &y, op).expect("equal lengths"))
                            .expect("the zip finishes")
                    },
                    BatchSize::LargeInput,
                )
            });
        }
        for op in [ReduceOp::Sum, ReduceOp::Variance, ReduceOp::CountMatching] {
            group.bench_function(format!("reduce_{op:?}"), |b| {
                b.iter(|| {
                    black_box(
                        drive_serial(
                            reduce(black_box(&x), op, Some(-1.0e5), Some(1.0e5)).expect("bounds"),
                        )
                        .expect("the reduction finishes"),
                    )
                })
            });
        }
        group.bench_function("prefix_sum", |b| {
            b.iter_batched_ref(
                || x.clone(),
                |data| drive_serial(prefix_sum(data)).expect("the scan finishes"),
                BatchSize::LargeInput,
            )
        });
        group.bench_function("histogram", |b| {
            b.iter(|| {
                black_box(
                    drive_serial(histogram(black_box(&x), 64, None, None).expect("bins"))
                        .expect("the histogram finishes"),
                )
            })
        });
        group.bench_function("dot_product", |b| {
            b.iter(|| {
                black_box(
                    drive_serial(dot_product(black_box(&x), black_box(&y)).expect("equal lengths"))
                        .expect("the dot product finishes"),
                )
            })
        });
        group.throughput(Throughput::Bytes(text.len() as u64));
        group.bench_function("text_count", |b| {
            b.iter(|| {
                black_box(
                    drive_serial(
                        text_count(black_box(&text), Some("ERRORERROR")).expect("a pattern"),
                    )
                    .expect("the count finishes"),
                )
            })
        });
        group.bench_function("text_search", |b| {
            b.iter(|| {
                black_box(
                    drive_serial(search_text(black_box(&text), "ERROR").expect("a pattern"))
                        .expect("the search finishes"),
                )
            })
        });
        group.bench_function("text_lower", |b| {
            b.iter(|| {
                black_box(
                    drive_serial(
                        update_text(black_box(&text), TextTransform::ToLower, None, None)
                            .expect("no operand"),
                    )
                    .expect("the transform finishes"),
                )
            })
        });
        group.finish();
        drop(load);
    }
}

criterion_group!(benches, bench_all);
criterion_main!(benches);
