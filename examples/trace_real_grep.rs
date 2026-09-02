//! Run a real-workload grep dispatch ONCE with FLYNNEL_TRACE=1 set and
//! dump the per-worker trace buffers to stderr as CSV. Useful for
//! localizing where per-fork wait-loop time goes architecturally.
//!
//! Usage:
//!     FLYNNEL_TRACE=1 cargo run --release --example trace_real_grep 2> trace.csv
//!     python examples/analyze_trace.py trace.csv
//!
//! The expected output covers per-fork (JoinPush -> JoinWaitBegin ->
//! JoinWaitEnd) intervals across all worker threads + per-worker
//! LeafStart/LeafEnd brackets. The wait-loop hot-spin hypothesis
//! predicts long JoinWaitBegin -> JoinWaitEnd intervals on the
//! producer threads while thieves execute stolen right-halves.

use flynnel::JobPlan;
use flynnel::sched::par_iter::for_each_chunk_indexed_min_leaf;
use flynnel::sched::trace;

const GREP_BYTES: usize = 16 * 1024 * 1024;
const GREP_NEEDLE: &[u8] = b"FLYNNEL";

fn make_haystack(seed: u64) -> Vec<u8> {
    let mut state = seed.wrapping_add(0xDEAD_BEEF_FEED_1234);
    let mut out = vec![0u8; GREP_BYTES];
    for b in out.iter_mut() {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *b = ((state >> 32) as u32 % 26) as u8 + b'A';
    }
    let mut i = 100;
    while i + GREP_NEEDLE.len() < GREP_BYTES {
        out[i..i + GREP_NEEDLE.len()].copy_from_slice(GREP_NEEDLE);
        i += 1024;
    }
    out
}

#[inline]
fn count_matches(haystack: &[u8], needle: &[u8]) -> usize {
    if needle.is_empty() || haystack.len() < needle.len() {
        return 0;
    }
    let mut count = 0;
    let last = haystack.len() - needle.len() + 1;
    let first_byte = needle[0];
    let mut i = 0;
    while i < last {
        if haystack[i] == first_byte && &haystack[i..i + needle.len()] == needle {
            count += 1;
        }
        i += 1;
    }
    count
}

fn grep_flynnel(plan: &JobPlan, haystack: &[u8]) -> usize {
    let chunk_size = 1024 * 1024;
    let overlap = GREP_NEEDLE.len();
    let n_chunks = haystack.len().div_ceil(chunk_size);
    let mut chunk_counts: Vec<usize> = vec![0usize; n_chunks];
    for_each_chunk_indexed_min_leaf(plan, chunk_counts.as_mut_slice(), 1, |start_idx, slab| {
        for (i, slot) in slab.iter_mut().enumerate() {
            let chunk_idx = start_idx + i;
            let s = chunk_idx * chunk_size;
            let end = (s + chunk_size + overlap).min(haystack.len());
            *slot = count_matches(&haystack[s..end], GREP_NEEDLE);
        }
    });
    chunk_counts.into_iter().sum()
}

fn main() {
    let trace_on = trace::is_enabled();
    eprintln!("# trace enabled = {trace_on} (set FLYNNEL_TRACE=1 to instrument)");
    let haystack = make_haystack(0x6E45);
    let plan = JobPlan::new(6, GREP_BYTES as u32).with_estimated_per_item_ns(1);

    // Warm the worker pool with a throwaway pre-run so the measured
    // dispatch sees workers in steady state (not cold). Without this,
    // the first dispatch pays pool-init costs that pollute the trace.
    let _warmup = grep_flynnel(&plan, &haystack);

    // -- MEASURED FLYNNEL DISPATCH --
    let t0 = std::time::Instant::now();
    let n_flyn = grep_flynnel(&plan, &haystack);
    let flyn_ns = t0.elapsed().as_nanos() as u64;
    // Trigger all worker threads to flush their trace buffers, then
    // flush the caller thread itself. The worker_loop hook auto-
    // dumps when the global flag is set.
    if trace_on {
        trace::request_worker_flush();
        // Briefly give workers a chance to observe the flag and
        // flush. Small sleep avoids racing the flush with the
        // workers' next loop iter.
        std::thread::sleep(std::time::Duration::from_millis(100));
        trace::flush_with_label("caller");
        trace::clear_worker_flush_request();
    }

    eprintln!(
        "# grep_flynnel: result={n_flyn} elapsed={us:.1}us",
        us = flyn_ns as f64 / 1000.0
    );
}
