//! E2E proof: run a streaming byte-scan (grep) through the
//! flynnel scheduler and watch the closing-loop observer migrate
//! the global active WorkloadClass from the startup default
//! (PortBound) to whatever the auto-classifier picks based on
//! observed per-leaf time + cv^2.
//!
//! Run with:
//! ```text
//! cargo run --release --example observe_grep
//! ```
//!
//! Prints the active WorkloadClass before / during / after the
//! scan so the migration is visible in stdout. This is the
//! Hard-Rule-#1 binary-trigger artifact for the adaptive
//! observer work tracked by docs/ADAPTIVE_OBSERVER_TRACKER.md.

use flynnel::sched::adaptive_profile::active_workload_class;
use flynnel::sched::par_iter::for_each_chunk_indexed_min_leaf;
use flynnel::JobPlan;

const HAYSTACK_BYTES: usize = 16 * 1024 * 1024;
const NEEDLE: &[u8] = b"flynnel";
const CHUNK: usize = 1024 * 1024;

fn make_haystack(seed: u64) -> Vec<u8> {
    let mut state = seed.wrapping_add(0xDEAD_BEEF_FEED_1234);
    let mut buf = vec![0u8; HAYSTACK_BYTES];
    for b in buf.iter_mut() {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        *b = b'A' + ((state >> 32) as u32 % 26) as u8;
    }
    let mut i = 0;
    while i + NEEDLE.len() < HAYSTACK_BYTES {
        buf[i..i + NEEDLE.len()].copy_from_slice(NEEDLE);
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        i += 4096 + ((state >> 32) as u32 % 4096) as usize;
    }
    buf
}

fn count_matches(hay: &[u8], needle: &[u8]) -> usize {
    if hay.len() < needle.len() {
        return 0;
    }
    let mut n = 0;
    let last = hay.len() - needle.len();
    let mut i = 0;
    while i <= last {
        if hay[i] == needle[0] && &hay[i..i + needle.len()] == needle {
            n += 1;
        }
        i += 1;
    }
    n
}

fn grep_one_iter(hay: &[u8]) -> usize {
    let overlap = NEEDLE.len();
    let n_chunks = hay.len().div_ceil(CHUNK);
    let mut counts = vec![0usize; n_chunks];
    // Default JobPlan -- no profile hint. The closing-loop observer
    // measures the leaf-time signal during the run and migrates the
    // global active WorkloadClass when it sees a streaming pattern
    // (mean_ns >= 500, cv^2 < 50).
    let plan = JobPlan::new(6, hay.len() as u32);
    for_each_chunk_indexed_min_leaf(&plan, counts.as_mut_slice(), 1, |start_idx, slab| {
        for (i, slot) in slab.iter_mut().enumerate() {
            let ci = start_idx + i;
            let s = ci * CHUNK;
            let e = (s + CHUNK + overlap).min(hay.len());
            *slot = count_matches(&hay[s..e], NEEDLE);
        }
    });
    counts.iter().sum()
}

fn main() {
    println!("== Adaptive Observer E2E proof ==");
    println!("Workload: 16 MiB streaming byte-scan (grep flynnel)");
    println!();

    let hay = make_haystack(0x6E45);
    println!(
        "Active WorkloadClass at startup: {:?}",
        active_workload_class()
    );

    // Warm-up: one iteration so the worker pool spins up.
    println!("Warm-up matches: {}", grep_one_iter(&hay));
    println!(
        "After warm-up (1 iter):           {:?}",
        active_workload_class()
    );

    // Several iterations so the auto-classifier ticks fire and
    // hysteresis is met.
    let mut last = 0usize;
    for i in 1..=10 {
        last = grep_one_iter(&hay);
        println!(
            "After iter {:>2}:                    {:?}",
            i + 1,
            active_workload_class()
        );
    }

    println!();
    println!("Final match count (sanity):       {last}");
    println!();
    println!("If the active class shifted away from PortBound during");
    println!("the run, the closing-loop observer is working: it saw");
    println!("the leaf-time signal of a streaming byte scan and");
    println!("re-routed the global routing decision via the atomic-");
    println!("swap surface in adaptive_profile.rs -- without any");
    println!("application-side classify-or-migrate call.");
}
