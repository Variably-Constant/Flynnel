//! What it costs to root a chain, per chunk and per byte.
//!
//! The chain's fold is ordered: a chunk is deposited in the slot its
//! submission took and the hasher consumes the slots in that order,
//! so a chunk that finishes ahead of its predecessor waits. That
//! ordering is paid for with a second lock on the submit path and a
//! slot table that grows with the chunk count, and both costs are
//! per chunk rather than per byte.
//!
//! So the arms sweep chunk size rather than only chunk count. At 64
//! bytes the hasher does almost nothing and the plumbing is the whole
//! measurement; at 64 KiB the hasher dominates and the plumbing
//! should disappear into it. A ratio that fails to fall across that
//! sweep would say the per-chunk cost is not per chunk.
//!
//! The fan-out arm is the shape the ordering exists for: several
//! producer threads on cloned handles, finishing in whatever order
//! the scheduler gives them.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

use flynnel::sched::verify_chain::VerifyChain;

const CHUNK_COUNT: usize = 1024;
const CHUNK_SIZES: [usize; 3] = [64, 4096, 65536];
const PRODUCERS: usize = 4;

fn trace(chunk_bytes: usize, count: usize) -> Vec<Vec<u8>> {
    (0..count)
        .map(|i| {
            let mut v = vec![0u8; chunk_bytes];
            // A varying first byte keeps every chunk distinct without
            // paying to fill the whole buffer per iteration.
            v[0] = (i % 251) as u8;
            v
        })
        .collect()
}

fn root_sequentially(chunks: &[Vec<u8>]) -> [u8; 32] {
    let chain = VerifyChain::new();
    for c in chunks {
        chain.submit_chunk(c.clone());
    }
    chain.finalize()
}

fn root_from_producers(chunks: &[Vec<u8>], producers: usize) -> [u8; 32] {
    let chain = VerifyChain::new();
    // One shared cursor rather than a contiguous slice each: the
    // point of the arm is that submission order and completion order
    // come apart, and a per-thread contiguous block would keep them
    // nearly together.
    let next = Arc::new(AtomicUsize::new(0));
    let total = chunks.len();
    std::thread::scope(|s| {
        for _ in 0..producers {
            let chain = chain.clone();
            let next = Arc::clone(&next);
            s.spawn(move || {
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= total {
                        break;
                    }
                    chain.submit_chunk(chunks[i].clone());
                }
            });
        }
    });
    chain.finalize()
}

fn bench_sequential(c: &mut Criterion) {
    let mut g = c.benchmark_group("verify_chain_sequential");
    g.sample_size(30);
    for bytes in CHUNK_SIZES {
        let chunks = trace(bytes, CHUNK_COUNT);
        g.throughput(Throughput::Bytes((bytes * CHUNK_COUNT) as u64));
        g.bench_with_input(
            BenchmarkId::from_parameter(bytes),
            &chunks,
            |b, chunks| b.iter(|| std::hint::black_box(root_sequentially(chunks))),
        );
    }
    g.finish();
}

fn bench_fanout(c: &mut Criterion) {
    let mut g = c.benchmark_group("verify_chain_fanout");
    g.sample_size(30);
    for bytes in [64usize, 4096] {
        let chunks = trace(bytes, CHUNK_COUNT);
        g.throughput(Throughput::Bytes((bytes * CHUNK_COUNT) as u64));
        g.bench_with_input(
            BenchmarkId::from_parameter(bytes),
            &chunks,
            |b, chunks| {
                b.iter(|| std::hint::black_box(root_from_producers(chunks, PRODUCERS)))
            },
        );
    }
    g.finish();
}

criterion_group!(verify_chain_fold, bench_sequential, bench_fanout);
criterion_main!(verify_chain_fold);
