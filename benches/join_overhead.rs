//! Direct head-to-head: per-join overhead of flynnel::join vs
//! rayon::join. Uses a tight recursive divide-and-conquer with
//! trivial leaf work so the time is dominated by the join plumbing
//! itself (push, wait, pop, latch).

use criterion::{Criterion, criterion_group, criterion_main};

use flynnel::{JobPlan, join as flynnel_join};

// Recursion depth (each level creates 2 joins). At depth=10 we
// create 2^10 = 1024 leaves and run 2^10 - 1 = 1023 joins.
const DEPTH: u32 = 10;

#[inline(never)]
fn leaf() -> u64 {
    // Smallest non-zero work the compiler won't optimize away.
    std::hint::black_box(42u64)
}

fn rayon_recurse(depth: u32) -> u64 {
    if depth == 0 {
        return leaf();
    }
    let (a, b) = rayon::join(
        || rayon_recurse(depth - 1),
        || rayon_recurse(depth - 1),
    );
    a + b
}

fn flynnel_recurse(plan: &JobPlan, depth: u32) -> u64 {
    if depth == 0 {
        return leaf();
    }
    let (a, b) = flynnel_join(
        plan,
        || flynnel_recurse(plan, depth - 1),
        || flynnel_recurse(plan, depth - 1),
    );
    a + b
}

fn bench_join_overhead(c: &mut Criterion) {
    let plan = JobPlan::new(6, 1024);
    let mut g = c.benchmark_group("join_overhead_d10");
    g.sample_size(50);
    g.bench_function("rayon", |b| b.iter(|| rayon_recurse(DEPTH)));
    g.bench_function("flynnel", |b| b.iter(|| flynnel_recurse(&plan, DEPTH)));
    g.finish();

    // Shallow recursion: just one join.
    let mut g = c.benchmark_group("join_overhead_d1");
    g.sample_size(100);
    g.bench_function("rayon", |b| b.iter(|| rayon_recurse(1)));
    g.bench_function("flynnel", |b| b.iter(|| flynnel_recurse(&plan, 1)));
    g.finish();
}

criterion_group!(join_overhead, bench_join_overhead);
criterion_main!(join_overhead);
