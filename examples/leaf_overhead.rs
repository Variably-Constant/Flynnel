//! Per-leaf dispatch overhead, read off the trace rather than timed by
//! a window.
//!
//! A throughput window measures dispatches completed in a stretch of
//! wall time, which is the quantity a busy machine destroys: on a host
//! carrying other builds the trial-to-trial spread reached ten times,
//! against differences under test of a fraction of a per cent.
//!
//! The trace stamps every `LeafStart` and `LeafEnd` with a TSC read, so
//! the gap from one leaf ending to the next starting IS the per-leaf
//! dispatch overhead, and one dispatch yields as many samples as it has
//! leaves. Background load can only make a gap longer, never shorter,
//! so the smallest gap over hundreds of leaves is the overhead with the
//! interference removed - the same reason the cheapest calibration draw
//! is the quiet one.
//!
//! One dispatch, because tracing every iteration writes gigabytes.
//!
//! Refuses when `FLYNNEL_TRACE` is off. Tracing disabled makes `emit` a
//! fast return, so the run would produce no rows and an empty trace
//! reads exactly like a dispatch that took no overhead.
//!
//! ```sh
//! FLYNNEL_TRACE=1 leaf_overhead [items] [reps] [entry]
//! ```

use std::env;
use std::hint::black_box;

use flynnel::sched::par_iter::{
    for_each_chunk_indexed_min_leaf, for_each_chunk_min_leaf, for_each_chunk_triple_min_leaf,
};
use flynnel::{CallSiteState, JobPlan, SiteRef};

static SITE: CallSiteState = CallSiteState::new();

const MIN_LEAF: usize = 256;

fn arg<T: std::str::FromStr>(n: usize, default: T) -> T {
    match env::args().nth(n) {
        None => default,
        Some(text) => text.parse().unwrap_or(default),
    }
}

#[derive(Clone, Copy)]
struct Item {
    reps: u32,
    acc: u64,
}

#[inline]
fn grind(slot: &mut Item) {
    let mut acc = slot.acc;
    for _ in 0..slot.reps {
        acc = acc.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
    }
    slot.acc = black_box(acc);
}

fn main() {
    if !flynnel::sched::trace::is_enabled() {
        eprintln!(
            "FLYNNEL_TRACE is not set, so trace::emit returns immediately and this run \
             would record no leaf boundaries; an empty trace reads exactly like a \
             dispatch with no overhead"
        );
        std::process::exit(2);
    }

    let items: usize = arg(1, 1 << 16);
    let reps: u32 = arg(2, 16);
    let entry: String = arg(3, "indexed".to_string());

    let mut buf: Vec<Item> = (0..items).map(|i| Item { reps, acc: i as u64 }).collect();
    let mut aux: Vec<Item> = vec![Item { reps, acc: 0 }; items];

    let plan = JobPlan::new(0, items as u32).with_site(SiteRef::new(&SITE));

    // One untraced dispatch first. The first dispatch of a process pays
    // pool startup and the site's first classifier ticks, and those
    // would land in the gaps as overhead that no later dispatch pays.
    for_each_chunk_min_leaf(&plan, &mut buf, MIN_LEAF, |chunk| {
        for slot in chunk.iter_mut() {
            grind(slot);
        }
    });

    flynnel::sched::trace::reset_current_thread();

    match entry.as_str() {
        "plain" => {
            for_each_chunk_min_leaf(&plan, &mut buf, MIN_LEAF, |chunk| {
                for slot in chunk.iter_mut() {
                    grind(slot);
                }
            });
        }
        "indexed" => {
            for_each_chunk_indexed_min_leaf(&plan, &mut buf, MIN_LEAF, |_start, chunk| {
                for slot in chunk.iter_mut() {
                    grind(slot);
                }
            });
        }
        "triple" => {
            for_each_chunk_triple_min_leaf(
                &plan,
                &mut aux,
                &buf,
                &buf,
                MIN_LEAF,
                |out, a, _b| {
                    for i in 0..out.len() {
                        out[i] = a[i];
                        grind(&mut out[i]);
                    }
                },
            );
        }
        other => {
            eprintln!("entry must be plain, indexed or triple, and was {other:?}");
            std::process::exit(2);
        }
    }

    // The workers hold their own buffers, so a dump of this thread alone
    // would carry only the leaves the caller ran.
    flynnel::sched::trace::request_worker_flush();
    std::thread::sleep(std::time::Duration::from_millis(200));
    flynnel::sched::trace::dump_to_stderr("caller");

    println!(
        "leaf_overhead entry={entry} items={items} reps={reps} leaves={} oncore_items={} \
         per_item_ns={:?}",
        SITE.leaf_count(),
        SITE.oncore_items(),
        SITE.per_item_ns(),
    );

    if SITE.leaf_count() == 0 {
        eprintln!("the site recorded no leaves, so there are no boundaries to read");
        std::process::exit(3);
    }
}
