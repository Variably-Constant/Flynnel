//! What the trace predicate costs when tracing is off.
//!
//! `trace::is_enabled` is consulted twice per leaf, twice per dispatch
//! and six times per join, so the shape of that predicate is on the
//! shipped path whether or not anyone is tracing. Turning its latch
//! into a settable flag changes the shape, and the standing criterion
//! says a change to the default path is measured rather than reasoned
//! about.
//!
//! # Why this exists rather than a bench arm
//!
//! `pwrs/bench/KernelShapes.ps1` times whole kernels. Two runs of the
//! same build on pc2 put its anchor 3.8 per cent apart, so it cannot
//! resolve one atomic load across the ~1560 consulted loads a dispatch
//! makes. Reading its drift line after this change would report noise
//! either way. The predicate has to be timed directly.
//!
//! # Shape
//!
//! Three cells, all in one process and interleaved so a clock or a
//! frequency change moves all three together:
//!
//!   control  a `black_box` read of a plain `bool`. The floor: what
//!            the loop and the barrier cost with no predicate at all.
//!   latch    `OnceLock<bool>::get_or_init`, which is what shipped.
//!   settable `Once::call_once` plus an `AtomicBool` relaxed load,
//!            which is what replaces it.
//!
//! Each cell is read as its median over `REPEATS`, and the control is
//! subtracted so the answer is the predicate's own cost rather than
//! the harness's. The control is also reported first and last, and a
//! run whose two controls disagree is not readable.
//!
//! Both predicates are written out here rather than called through the
//! crate, so one binary times both shapes and no second build is
//! needed to compare them.

use std::hint::black_box;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

/// Calls per timed cell.
///
/// Five hundred million rather than fifty. At fifty the cell was 31
/// milliseconds and the control moved 2.6 to 12.6 per cent across a
/// run - which is 0.016 to 0.079 ns on a 0.63 ns cell, the same size
/// as the difference being measured. Every run declared itself
/// unreadable, correctly. A longer cell is the fix: the predicate
/// does not get cheaper, the clock and the scheduler get averaged
/// over more of it.
const CALLS: u64 = 500_000_000;

/// Timed cells per shape. The median is taken, so an odd count has a
/// middle.
const REPEATS: usize = 7;

static LATCH: OnceLock<bool> = OnceLock::new();

static SETTABLE: AtomicBool = AtomicBool::new(false);
static SETTABLE_FROM_ENV: std::sync::Once = std::sync::Once::new();

/// The shape that shipped.
#[inline]
fn latch_enabled() -> bool {
    *LATCH.get_or_init(|| {
        std::env::var("FLYNNEL_TRACE_PREDICATE_PROBE")
            .map(|v| v == "1")
            .unwrap_or(false)
    })
}

/// The shape that replaces it.
#[inline]
fn settable_enabled() -> bool {
    SETTABLE_FROM_ENV.call_once(|| {
        let on = std::env::var("FLYNNEL_TRACE_PREDICATE_PROBE")
            .map(|v| v == "1")
            .unwrap_or(false);
        SETTABLE.store(on, Ordering::Relaxed);
    });
    SETTABLE.load(Ordering::Relaxed)
}

/// A predicate that is neither, for the floor.
#[inline]
fn control_enabled() -> bool {
    black_box(false)
}

/// Nanoseconds per call for one cell.
fn cell(f: fn() -> bool) -> f64 {
    let t0 = Instant::now();
    let mut seen = 0u64;
    for _ in 0..CALLS {
        // black_box on the result as well, so the branch cannot be
        // folded away once the compiler proves the predicate constant.
        if black_box(f()) {
            seen += 1;
        }
    }
    let elapsed = t0.elapsed();
    black_box(seen);
    elapsed.as_nanos() as f64 / CALLS as f64
}

fn median(mut xs: Vec<f64>) -> f64 {
    xs.sort_by(|a, b| a.partial_cmp(b).expect("no NaN from a duration"));
    xs[xs.len() / 2]
}

fn main() {
    // Warm both predicates past their one-time initialisation, so no
    // cell pays for it and every cell times the steady state.
    black_box(latch_enabled());
    black_box(settable_enabled());

    let control_first = cell(control_enabled);

    let mut latch = Vec::with_capacity(REPEATS);
    let mut settable = Vec::with_capacity(REPEATS);
    // Interleaved rather than run in blocks: a frequency change or a
    // neighbour arriving partway through would otherwise land on one
    // shape and not the other, and the difference between them is the
    // whole answer.
    for _ in 0..REPEATS {
        latch.push(cell(latch_enabled));
        settable.push(cell(settable_enabled));
    }

    let control_last = cell(control_enabled);

    let latch_ns = median(latch);
    let settable_ns = median(settable);
    let control_ns = (control_first + control_last) / 2.0;
    let drift = (control_last - control_first) / control_first * 100.0;

    println!("calls per cell {CALLS}, repeats {REPEATS}");
    println!("control  {control_first:.4} ns then {control_last:.4} ns, drift {drift:.2}%");
    println!("latch    {latch_ns:.4} ns/call, {:.4} over control", latch_ns - control_ns);
    println!(
        "settable {settable_ns:.4} ns/call, {:.4} over control",
        settable_ns - control_ns
    );
    println!("settable minus latch {:.4} ns/call", settable_ns - latch_ns);
    println!(
        "at 1560 consulted loads a dispatch that is {:.1} ns a dispatch",
        (settable_ns - latch_ns) * 1560.0
    );

    // The bound is the useful answer even when the difference is not
    // resolvable. A dispatch over 200,000 elements costs about 380
    // microseconds, so a per-dispatch figure is read against that.
    const DISPATCH_NS: f64 = 380_000.0;
    let per_dispatch = (settable_ns - latch_ns) * 1560.0;
    println!(
        "which is {:.4}% of a 200,000-element dispatch",
        per_dispatch / DISPATCH_NS * 100.0
    );

    // A run whose control moved across it is measuring the box.
    if drift.abs() > 2.0 {
        println!(
            "UNREADABLE: the control moved {drift:.2}% across this run, so the \
             difference above is not attributable to either shape"
        );
    }
}
