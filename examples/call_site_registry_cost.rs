//! What resolving a call site costs when the per-thread cache misses.
//!
//! `site_for_location` is reached by every `#[track_caller]` dispatch
//! entry. A per-thread one-slot cache answers a caller that keeps
//! hitting the same textual site, so the registry behind it is reached
//! only when a thread alternates between sites, which a driver loop
//! running several kernels does on every call. That miss path held a
//! `RwLock<HashMap>` and now holds an open-addressed table of leaked
//! nodes, and the standing criterion says a change to a shipped path
//! is measured rather than reasoned about.
//!
//! # Shape
//!
//! Four cells, all in one process and interleaved so a clock or a
//! frequency change moves all of them together:
//!
//!   control  a `black_box` read of the key array. The floor: what the
//!            loop and the key rotation cost with no lookup at all.
//!   map      the key looked up in a `RwLock<HashMap<u64, _>>`, which
//!            is what shipped.
//!   table    the same key probed in an open-addressed table of leaked
//!            nodes, which is what replaces it. The probe and the
//!            dependent load through the node pointer are written out
//!            the way the crate does them, so the cell pays for the
//!            node fetch and not only for the slot load.
//!   real     `flynnel::sched::caller_site()` across `SITES` distinct
//!            textual sites, so the per-thread cache misses every call.
//!            This is the shipped path end to end, and it says whether
//!            the `table` cell tracks the code or only resembles it.
//!
//! Both registry shapes are written out here rather than reached
//! through the crate, so one binary times both and the comparison is
//! between two cells of one run instead of between two builds on two
//! clocks. `real` is the crate's own, because it is the function whose
//! cost is in question.
//!
//! # Quiet and loaded
//!
//! Every cell is run twice: alone, and with `threads - 1` siblings
//! running the same cells against the same registries. The loaded row
//! is the one the change is about. A read lock is an atomic
//! read-modify-write on a counter every reader shares, so concurrent
//! readers serialize on that line however read-mostly the map is; a
//! probe is a load of a line nobody writes after startup. A quiet row
//! alone would price the uncontended lock and miss the whole effect.
//!
//! Each cell is read as its median over `REPEATS`, and the control is
//! subtracted so the answer is the lookup's own cost rather than the
//! harness's. The control is interleaved with the other three and its
//! median is what gets subtracted: sampled only at the ends it would
//! sit on the two extremes of the warm-up curve while every other cell
//! averaged over the whole of it.
//!
//! What decides whether a difference holds is the span of the control
//! across its own repeats. A figure smaller than that span was not
//! resolved by the run.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::hint::black_box;
use std::sync::atomic::{AtomicBool, AtomicPtr, Ordering};
use std::sync::{Arc, Barrier, RwLock};
use std::time::Instant;

/// Distinct sites a cell rotates through. Anything above one defeats
/// the per-thread one-slot cache, and sixty-four is the order of a
/// driver loop that runs a handful of kernels over a few shapes.
const SITES: usize = 64;

/// Lookups per timed cell.
const CALLS: u64 = 20_000_000;

/// Timed cells per shape. The median is taken, so an odd count has a
/// middle.
const REPEATS: usize = 7;

/// Slot count of the probe table, matching the crate's.
const SLOTS: usize = 2048;
const MASK: usize = SLOTS - 1;

/// The value a lookup answers for a key the registry does not hold.
/// Both shapes use it, so neither is credited with a shorter miss.
const ABSENT: usize = 0;

/// What the crate stores per site, reduced to the parts a lookup
/// touches: the key it compares and a payload it returns.
struct Node {
    key: u64,
    payload: usize,
}

static TABLE: [AtomicPtr<Node>; SLOTS] = [const { AtomicPtr::new(core::ptr::null_mut()) }; SLOTS];

fn table_insert(key: u64, payload: usize) {
    let node: &'static Node = Box::leak(Box::new(Node { key, payload }));
    let mut idx = (key as usize) & MASK;
    for _ in 0..SLOTS {
        let p = TABLE[idx].load(Ordering::Acquire);
        if p.is_null() {
            let fresh = node as *const Node as *mut Node;
            if TABLE[idx]
                .compare_exchange(
                    core::ptr::null_mut(),
                    fresh,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                )
                .is_ok()
            {
                return;
            }
        }
        idx = (idx + 1) & MASK;
    }
    panic!("probe table full at {SLOTS} slots");
}

#[inline]
fn table_lookup(key: u64) -> usize {
    let mut idx = (key as usize) & MASK;
    for _ in 0..SLOTS {
        let p = TABLE[idx].load(Ordering::Acquire);
        if p.is_null() {
            return ABSENT;
        }
        // SAFETY: a non-null slot holds a leaked Node that is never
        // freed, moved or rehashed.
        let node = unsafe { &*p };
        if node.key == key {
            return node.payload;
        }
        idx = (idx + 1) & MASK;
    }
    ABSENT
}

#[inline]
fn map_lookup(map: &RwLock<HashMap<u64, usize>>, key: u64) -> usize {
    // A panicking reader would poison this, and the map it guards is
    // read-only after setup, so recovering reports what is there
    // rather than abandoning a run over damage the map does not have.
    let guard = match map.read() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    };
    match guard.get(&key) {
        Some(&payload) => payload,
        None => ABSENT,
    }
}

/// The key a location hashes to, formed the way the crate forms it.
fn key_for(file: &str, line: u32, column: u32) -> u64 {
    let mut h = std::hash::DefaultHasher::new();
    file.hash(&mut h);
    line.hash(&mut h);
    column.hash(&mut h);
    h.finish()
}

/// One timed run of a cell, in nanoseconds per call.
fn cell(calls: u64, mut body: impl FnMut(u64) -> usize) -> f64 {
    let start = Instant::now();
    let mut sink = 0usize;
    for i in 0..calls {
        sink = sink.wrapping_add(body(i));
    }
    black_box(sink);
    start.elapsed().as_nanos() as f64 / calls as f64
}

fn median(mut xs: Vec<f64>) -> f64 {
    xs.sort_by(|a, b| a.partial_cmp(b).expect("a timing is never NaN"));
    xs[xs.len() / 2]
}

fn spread_pct(xs: &[f64]) -> f64 {
    let lo = xs.iter().copied().fold(f64::INFINITY, f64::min);
    let hi = xs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if lo <= 0.0 {
        f64::INFINITY
    } else {
        (hi - lo) / lo * 100.0
    }
}

/// Rotate through `SITES` distinct textual call sites so the crate's
/// per-thread one-slot cache misses on every call.
///
/// Written out rather than looped because the identity being exercised
/// is the source location itself: a loop calling one site would hit
/// the cache every time and measure it instead of the registry.
macro_rules! sites {
    ($which:expr, $($n:literal),* $(,)?) => {{
        let w = $which;
        let mut out = ABSENT;
        $(
            if w == $n {
                out = flynnel::sched::caller_site().get() as *const _ as usize;
            }
        )*
        out
    }};
}

#[inline(never)]
fn resolve_one(which: usize) -> usize {
    sites!(
        which, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
        23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
        46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63
    )
}

/// The four cells, interleaved, against the key set already loaded
/// into both registries.
fn measure(keys: &[u64], map: &RwLock<HashMap<u64, usize>>) -> [Vec<f64>; 4] {
    let mut control = Vec::with_capacity(REPEATS);
    let mut map_cell = Vec::with_capacity(REPEATS);
    let mut table_cell = Vec::with_capacity(REPEATS);
    let mut real_cell = Vec::with_capacity(REPEATS);

    for _ in 0..REPEATS {
        control.push(cell(CALLS, |i| {
            black_box(keys[(i as usize) % SITES]) as usize & 1
        }));
        map_cell.push(cell(CALLS, |i| map_lookup(map, keys[(i as usize) % SITES])));
        table_cell.push(cell(CALLS, |i| table_lookup(keys[(i as usize) % SITES])));
        real_cell.push(cell(CALLS / 8, |i| resolve_one((i as usize) % SITES)));
    }
    [control, map_cell, table_cell, real_cell]
}

fn report(label: &str, cells: [Vec<f64>; 4]) {
    let [control, map_cell, table_cell, real_cell] = cells;
    let floor = median(control.clone());
    let control_spread = spread_pct(&control);
    println!(
        "{label} floor={floor:.4} ns control_spread={control_spread:.2}% resolution_floor={:.4} ns",
        floor * control_spread / 100.0
    );
    for (name, xs) in [
        ("map", &map_cell),
        ("table", &table_cell),
        ("real", &real_cell),
    ] {
        let m = median(xs.clone());
        println!(
            "{label} {name}={:.4} ns over_floor={:.4} ns spread={:.2}%",
            m,
            m - floor,
            spread_pct(xs)
        );
    }
    let map_over = median(map_cell) - floor;
    let table_over = median(table_cell) - floor;
    if table_over > 0.0 {
        println!("{label} map_over_table={:.3}x", map_over / table_over);
    } else {
        println!("{label} map_over_table=unreadable, the table cell did not clear the floor");
    }
}

fn thread_count() -> usize {
    match std::env::args().nth(1) {
        Some(arg) => match arg.parse::<usize>() {
            Ok(n) if n >= 1 => n,
            Ok(n) => panic!("thread count must be at least 1, got {n}"),
            Err(err) => panic!("thread count {arg:?} is not a number: {err}"),
        },
        None => std::thread::available_parallelism()
            .expect("the host reports its parallelism")
            .get(),
    }
}

fn main() {
    let threads = thread_count();

    let keys: Vec<u64> = (0..SITES)
        .map(|i| key_for("examples/call_site_registry_cost.rs", 100 + i as u32, 9))
        .collect();

    let map = Arc::new(RwLock::new(HashMap::new()));
    {
        let mut m = map.write().expect("a fresh lock is not poisoned");
        for (i, &k) in keys.iter().enumerate() {
            m.insert(k, i + 1);
            table_insert(k, i + 1);
        }
    }
    // The crate's own registry wants the same sites materialised, so
    // the real cell times a hit and not sixty-four first sights.
    for i in 0..SITES {
        black_box(resolve_one(i));
    }

    println!(
        "call_site_registry_cost threads={threads} sites={SITES} calls={CALLS} repeats={REPEATS}"
    );

    report("quiet", measure(&keys, &map));

    let stop = Arc::new(AtomicBool::new(false));
    let barrier = Arc::new(Barrier::new(threads));
    let mut helpers = Vec::with_capacity(threads.saturating_sub(1));
    for _ in 0..threads.saturating_sub(1) {
        let stop = Arc::clone(&stop);
        let barrier = Arc::clone(&barrier);
        let map = Arc::clone(&map);
        let keys = keys.clone();
        helpers.push(std::thread::spawn(move || {
            barrier.wait();
            let mut sink = 0usize;
            let mut i = 0usize;
            while !stop.load(Ordering::Relaxed) {
                let k = keys[i % SITES];
                sink = sink.wrapping_add(map_lookup(&map, k));
                sink = sink.wrapping_add(table_lookup(k));
                sink = sink.wrapping_add(resolve_one(i % SITES));
                i += 1;
            }
            black_box(sink);
        }));
    }
    if threads > 1 {
        barrier.wait();
    }
    report("loaded", measure(&keys, &map));
    stop.store(true, Ordering::Relaxed);
    for h in helpers {
        h.join().expect("a helper thread panicked");
    }
}
