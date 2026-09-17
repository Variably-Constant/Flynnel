//! Does the on-core bracket reach the indexed and triple entries?
//!
//! One question, asked of the two dispatch entries that most production
//! op sites route through. With `FLYNNEL_LEVER_ONCORE_SPREAD` on, a run
//! through each entry must report a nonzero `oncore_items`; with it off,
//! both must report zero.
//!
//! Why this is not a unit test. The switch is read once per process and
//! cached, so a test binary observes only whichever arm it started in,
//! and the arm that matters is the one a test binary never starts in.
//! An example can be run twice, once per arm.
//!
//! Why the off arm is run at all. A harness that only ever printed a
//! nonzero count could be printing its own bug. The off arm establishes
//! that this binary reports zero when nothing asked for the bracket, so
//! a nonzero count on the other arm is the switch and not the harness.
//!
//! The exit code carries what a reader would otherwise have to spot in
//! the text: 3 when the switch is on and an entry bracketed nothing, 4
//! when it is off and an entry bracketed something, 5 when an entry
//! recorded no leaves at all, which is a broken recorder rather than an
//! answer either way.
//!
//! ```sh
//! FLYNNEL_LEVER_ONCORE_SPREAD=1 oncore_reach
//! ```

use std::hint::black_box;

use flynnel::sched::levers::oncore_spread;
use flynnel::sched::par_iter::{for_each_chunk_indexed_min_leaf, for_each_chunk_triple_min_leaf};
use flynnel::{CallSiteState, JobPlan, SiteRef};

static INDEXED: CallSiteState = CallSiteState::new();
static TRIPLE: CallSiteState = CallSiteState::new();

const ITEMS: usize = 1 << 16;
const MIN_LEAF: usize = 256;

/// Enough dispatches to carry the per-thread leaf buffers past their
/// flush threshold many times over. A count that never left a buffer
/// reads at the site as a count that was never taken.
const ROUNDS: usize = 64;

/// Per-item work the compiler cannot discard, short enough that the run
/// takes seconds. The figure under test is a count, not a duration, so
/// the body only has to be real.
#[inline]
fn grind(seed: u64) -> u64 {
    let mut acc = seed;
    for _ in 0..64 {
        acc = acc.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
    }
    acc
}

fn run_indexed(buf: &mut [u64]) {
    let site = SiteRef::new(&INDEXED);
    for _ in 0..ROUNDS {
        let plan = JobPlan::new(0, ITEMS as u32).with_site(site);
        for_each_chunk_indexed_min_leaf(&plan, buf, MIN_LEAF, |start, chunk| {
            for (i, slot) in chunk.iter_mut().enumerate() {
                *slot = black_box(grind(*slot ^ (start + i) as u64));
            }
        });
    }
}

fn run_triple(out: &mut [u64], a: &[u64], b: &[u64]) {
    let site = SiteRef::new(&TRIPLE);
    for _ in 0..ROUNDS {
        let plan = JobPlan::new(0, ITEMS as u32).with_site(site);
        for_each_chunk_triple_min_leaf(&plan, out, a, b, MIN_LEAF, |out, a, b| {
            for i in 0..out.len() {
                out[i] = black_box(grind(a[i] ^ b[i]));
            }
        });
    }
}

/// The count each entry ended with, and whether it is what the switch
/// asked for.
///
/// Returns the exit code this entry earns, or zero.
fn report(name: &str, site: &CallSiteState, switch: bool) -> i32 {
    let leaves = site.leaf_count();
    let oncore = site.oncore_items();
    let cv2 = site.per_item_oncore_cv2_per_mille();
    println!(
        "reach entry={name} switch={switch} leaves={leaves} \
         oncore_items={oncore} oncore_cv2_per_mille={cv2:?}"
    );
    if leaves == 0 {
        eprintln!("{name}: recorded no leaves, so neither answer is available from this run");
        return 5;
    }
    if switch && oncore == 0 {
        eprintln!(
            "{name}: the switch is on and nothing was bracketed, which is the case that \
             reads exactly like a mechanism that ran and did not help"
        );
        return 3;
    }
    if !switch && oncore != 0 {
        eprintln!("{name}: the switch is off and {oncore} items were bracketed");
        return 4;
    }
    0
}

fn main() {
    let switch = oncore_spread();

    let mut buf: Vec<u64> = (0..ITEMS as u64).collect();
    run_indexed(&mut buf);

    let a: Vec<u64> = (0..ITEMS as u64).collect();
    let b: Vec<u64> = (0..ITEMS as u64).map(|x| x ^ 0x5DEE_CE66).collect();
    let mut out: Vec<u64> = vec![0; ITEMS];
    run_triple(&mut out, &a, &b);

    black_box(&buf);
    black_box(&out);

    let indexed = report("indexed", &INDEXED, switch);
    let triple = report("triple", &TRIPLE, switch);

    // The first nonzero, so a reader who sees only the code knows one
    // entry failed and which reason, and the lines above say which
    // entry it was.
    let code = if indexed != 0 { indexed } else { triple };
    std::process::exit(code);
}
