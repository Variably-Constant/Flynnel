//! How deep a worker nests while helping inside a join wait.
//!
//! A worker waiting on its own join calls `find_work`, which probes
//! six tiers including peer steals, and runs whatever it gets on the
//! waiting stack. That job can join, wait, and run another. Nothing
//! about the task tree bounds the chain: it is bounded by how often a
//! worker is pressed into helping, and nothing caps that.
//!
//! Workers are spawned with an 8 MiB stack, which is on the order of
//! fifty thousand frames. That is the number the answer is read
//! against. A maximum in the tens says nested helping is not what
//! exhausts a worker's stack and the overflow seen in soak runs has
//! another cause. A maximum in the thousands says it is, and says how
//! much headroom a bound would have to leave.
//!
//! # What this runs
//!
//! Nested fork-join over a range, splitting to single items, so the
//! task tree is deep and every worker has peers with work to steal.
//! That is the shape that produces nested helping: a worker whose own
//! half is outstanding goes looking, finds a peer's subtree, and runs
//! it inside its own wait.
//!
//! The depth reported is a process-wide maximum over every worker, so
//! one unlucky chain anywhere in the run is what it prints. That is
//! the right reading for a stack overflow, which needs one chain, not
//! a typical one.
//!
//! Run it at several widths, because the chance of a deep chain rises
//! with the number of peers that have work worth taking.

use std::hint::black_box;

use flynnel::sched::plan::JobPlan;
use flynnel::sched::{help_depth_max, join};

/// Leaves in one dispatch. Split to single items, so the tree is
/// `log2(ITEMS)` deep and there is always something to steal.
const ITEMS: usize = 1 << 16;

/// Dispatches per width. A deep chain is a rare interleaving, so the
/// run has to offer many chances at it rather than one.
const ROUNDS: usize = 200;

/// Recursive fork-join over `[lo, hi)`, forking every level.
fn fork(plan: &JobPlan, lo: usize, hi: usize) -> u64 {
    if hi - lo <= 1 {
        // Enough arithmetic that a leaf is not free, so a peer has a
        // reason to still be busy when another goes looking.
        let mut acc = lo as u64;
        for k in 0..64u64 {
            acc = acc.wrapping_mul(6364136223846793005).wrapping_add(k);
        }
        return acc;
    }
    let mid = lo + (hi - lo) / 2;
    let (left, right) = join(plan, || fork(plan, lo, mid), || fork(plan, mid, hi));
    left.wrapping_add(right)
}

fn main() {
    let widths: Vec<usize> = match std::env::args().nth(1) {
        Some(arg) => match arg.parse::<usize>() {
            Ok(n) if n >= 1 => vec![n],
            Ok(n) => panic!("worker count must be at least 1, got {n}"),
            Err(err) => panic!("worker count {arg:?} is not a number: {err}"),
        },
        None => vec![2, 4, 8, 16],
    };

    println!("help_depth items={ITEMS} rounds={ROUNDS}");
    for width in widths {
        let plan = JobPlan::new(ITEMS.trailing_zeros(), ITEMS);
        let before = help_depth_max();
        for _ in 0..ROUNDS {
            black_box(fork(&plan, 0, ITEMS));
        }
        let after = help_depth_max();
        println!(
            "width={width} max_help_depth={after} (rose by {} this round) \
             stack_frames_at_8MiB=~50000",
            after - before
        );
    }
    println!(
        "help_depth_done max_help_depth={} over the whole run",
        help_depth_max()
    );
}
