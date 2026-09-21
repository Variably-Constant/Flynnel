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
//! Five cells, all in one process and interleaved so a clock or a
//! frequency change moves all of them together:
//!
//!   control  a `black_box` read of a plain `bool`. The floor: what
//!            the loop and the barrier cost with no predicate at all.
//!   latch    `OnceLock<bool>::get_or_init`, which is what shipped.
//!   settable `Once::call_once` plus an `AtomicBool` relaxed load,
//!            which is what replaces it.
//!   payload  the park path's trace payload with nothing called: a
//!            three-arm match on the wait strategy, or-ed with the
//!            bit marking a park as half of a probe pair.
//!   emit     that same payload handed to the crate's `trace::emit`
//!            with tracing off. Minus the payload cell it is the
//!            guard alone; minus the control it is the whole of what
//!            a park pays to carry a `ParkEnter` row.
//!
//! Each cell is read as its median over `REPEATS`, and the control is
//! subtracted so the answer is the predicate's own cost rather than
//! the harness's. The control is also reported first and last, and a
//! run whose two controls disagree is not readable.
//!
//! Both predicates are written out here rather than called through the
//! crate, so one binary times both shapes and no second build is
//! needed to compare them. The payload match is written out for that
//! reason and because `trace_code` is private to the crate. `emit` is
//! the crate's own, because it is the function whose cost is in
//! question.
//!
//! # Why this rather than an A/B of the park path
//!
//! `benches/parker_wait_strategy.rs` put its overhead arm at -16.5 and
//! +23.9 per cent against the settled arm. A quantity one predicate
//! wide does not survive a spread that size, so an A/B of the park
//! path would report no change whatever the truth was. Timing the
//! expression directly keeps the park's own cost out of the
//! arithmetic entirely.

use std::hint::black_box;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use flynnel::sched::trace::{self, TraceEvent};

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

/// The wait strategies in the encoding a trace row carries.
///
/// Written out rather than taken from `flynnel::sched::sleep`, whose
/// `trace_code` is private to the crate. What is being timed is the
/// shape: a three-arm match on a `Copy` enum yielding a constant.
#[derive(Copy, Clone)]
enum Strategy {
    StdPark,
    Waitpkg,
    Monitorx,
}

impl Strategy {
    #[inline]
    fn code(self) -> u32 {
        match self {
            Self::StdPark => 0,
            Self::Waitpkg => 1,
            Self::Monitorx => 2,
        }
    }
}

/// A strategy from an opaque index, so every variant is constructed
/// somewhere.
///
/// The timed cells park on one variant, which on its own leaves the
/// other two never built and the enum reported as partly dead. Calling
/// this once from the warm-up keeps all three live without putting a
/// second match inside a cell.
#[inline]
fn strategy_from(i: u32) -> Strategy {
    match i {
        0 => Strategy::StdPark,
        1 => Strategy::Waitpkg,
        _ => Strategy::Monitorx,
    }
}

/// The park path's payload expression with nothing called.
///
/// Both inputs go through `black_box` because on the park path both
/// are runtime values: the strategy comes from the controller and the
/// bit from whether this park was chosen for a probe. Leaving them
/// constant would fold the match and time an expression the parker
/// does not have. Returns a `bool` so the harness cell that times the
/// predicates times this one unchanged.
#[inline]
fn payload_only() -> bool {
    let strategy = black_box(Strategy::Monitorx);
    let sampled = black_box(true);
    black_box(strategy.code() | if sampled { 16 } else { 0 });
    false
}

/// The whole emit as it sits on the park path, tracing off.
#[inline]
fn park_emit() -> bool {
    let strategy = black_box(Strategy::Monitorx);
    let sampled = black_box(true);
    trace::emit(
        TraceEvent::ParkEnter,
        strategy.code() | if sampled { 16 } else { 0 },
    );
    false
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
    // The emit cell is only meaningful with tracing off, which is the
    // path it exists to measure. With tracing on it would push CALLS
    // records into a thread-local vector per cell and take the box
    // down long before it reported anything.
    if trace::is_enabled() {
        eprintln!(
            "REFUSING: FLYNNEL_TRACE is on, so the emit cell would record \
             {CALLS} rows a cell. This run measures the off path; unset it."
        );
        std::process::exit(2);
    }

    // Warm both predicates past their one-time initialisation, so no
    // cell pays for it and every cell times the steady state. The emit
    // path has the same seeding inside it.
    black_box(latch_enabled());
    black_box(settable_enabled());
    black_box(park_emit());
    for i in 0..3 {
        black_box(strategy_from(black_box(i)).code());
    }

    let control_first = cell(control_enabled);

    let mut latch = Vec::with_capacity(REPEATS);
    let mut settable = Vec::with_capacity(REPEATS);
    let mut payload = Vec::with_capacity(REPEATS);
    let mut emit = Vec::with_capacity(REPEATS);
    // Interleaved rather than run in blocks: a frequency change or a
    // neighbour arriving partway through would otherwise land on one
    // shape and not the other, and the difference between them is the
    // whole answer.
    for _ in 0..REPEATS {
        latch.push(cell(latch_enabled));
        settable.push(cell(settable_enabled));
        payload.push(cell(payload_only));
        emit.push(cell(park_emit));
    }

    let control_last = cell(control_enabled);

    let latch_ns = median(latch);
    let settable_ns = median(settable);
    let payload_ns = median(payload);
    let emit_ns = median(emit);
    let control_ns = (control_first + control_last) / 2.0;
    let drift = (control_last - control_first) / control_first * 100.0;

    println!("calls per cell {CALLS}, repeats {REPEATS}");
    println!("control  {control_first:.4} ns then {control_last:.4} ns, drift {drift:.2}%");
    println!("latch    {latch_ns:.4} ns/call, {:.4} over control", latch_ns - control_ns);
    println!(
        "settable {settable_ns:.4} ns/call, {:.4} over control",
        settable_ns - control_ns
    );
    println!(
        "payload  {payload_ns:.4} ns/call, {:.4} over control",
        payload_ns - control_ns
    );
    println!(
        "emit     {emit_ns:.4} ns/call, {:.4} over control",
        emit_ns - control_ns
    );
    println!("guard alone, emit minus payload {:.4} ns/call", emit_ns - payload_ns);

    // The settable cell and the guard inside emit are the same `Once`
    // and relaxed load, reached two ways: one through a static this
    // crate can see through, the other behind an inlined call from the
    // library. Reading them apart is the check that a cheap answer is
    // the guard being cheap rather than the compiler having lifted it
    // out of the loop. Where they diverge, the emit figure is the one
    // standing on the shipped path.
    let settable_over = settable_ns - control_ns;
    let guard_over = emit_ns - payload_ns;
    println!("coherence: settable over control {settable_over:.4}, guard inside emit {guard_over:.4}");
    let ratio = if settable_over > 0.0 && guard_over > 0.0 {
        (guard_over / settable_over).max(settable_over / guard_over)
    } else {
        f64::INFINITY
    };
    if ratio > 2.0 {
        println!(
            "DIVERGENT: one guard reads {ratio:.1}x apart through the two paths, so at most \
             one of them is its cost"
        );
    }

    // A park consults this once, so one cell is one park's whole
    // share. Read against 204 ns, the challenger wake cost measured
    // over 119 samples on pc2 - the smallest measured wake figure
    // there is, which makes the percentage an upper bound rather than
    // a typical one.
    const WAKE_NS: f64 = 204.0;
    println!(
        "a park carries {:.4} ns of this, {:.4}% of a 204 ns wake",
        emit_ns - control_ns,
        (emit_ns - control_ns) / WAKE_NS * 100.0
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
