//! What a calibration drawn on a busy host costs a later process.
//!
//! The three dispatch figures are measured once per host and persisted
//! for every process after. A draw taken while a neighbor held half the
//! machine is slow in all three, and the spread a record already carries
//! cannot see it: uniform contention makes every sample slow by about
//! the same amount, so the samples agree with each other on the wrong
//! number.
//!
//! This times a fixed workload under whatever profile
//! `FLYNNEL_HOST_PROFILE_NS` pins. Two runs, two profiles, arms
//! interleaved by the driver: the difference is what the poisoned draw
//! costs.
//!
//! # Why the pin is the arm rather than a switch inside one process
//!
//! The profile is read once per process and cached with the rest of the
//! host profile, so one process carries one profile for its life. Two
//! processes is the only shape available. The hazard that usually
//! carries - that comparing across processes prices the calibration
//! draw - does not apply here, because the draw is exactly what both
//! arms hold fixed by pinning rather than measuring.
//!
//! # Size the work where the profile decides something
//!
//! A dispatch far above the collapse threshold routes the same way
//! under any profile, and a run there measures nothing however wrong
//! the numbers are. `items` is an argument so a driver can sweep
//! outward from the boundary and find where the profile bites.
//!
//! ```sh
//! FLYNNEL_HOST_PROFILE_NS=1200,60000,40000 \
//!   cargo run --release --example profile_cost -- 4096 200
//! ```

use std::env;
use std::hint::black_box;
use std::time::Instant;

use flynnel::sched::par_iter::for_each_chunk_min_leaf;
use flynnel::{CallSiteState, JobPlan, SiteRef};

static SITE: CallSiteState = CallSiteState::new();

/// The recursion floor every arm passes, so a profile that lowers the
/// effective floor is doing so on its own account.
const MIN_LEAF: usize = 64;

fn arg<T>(n: usize, default: T) -> T
where
    T: std::str::FromStr,
    <T as std::str::FromStr>::Err: std::fmt::Display,
{
    match env::args().nth(n) {
        None => default,
        Some(text) => match text.parse() {
            Ok(value) => value,
            Err(err) => {
                eprintln!("argument {n} is not a valid value: {text:?} ({err})");
                std::process::exit(2);
            }
        },
    }
}

/// Say what this process will actually run under.
///
/// The three cases are different findings and do not share a line. A
/// pin that was never requested is the ordinary measured case. A pin
/// that was requested and could not be read falls back to measuring
/// SILENTLY inside the library, so a row from it looks exactly like a
/// pinned row while being nothing of the kind - which is the failure
/// this whole harness exists to make visible, and it would be absurd to
/// reproduce it here.
fn report_pin() {
    match env::var("FLYNNEL_HOST_PROFILE_NS") {
        Ok(text) => eprintln!("pinned profile requested: {text}"),
        Err(env::VarError::NotPresent) => eprintln!(
            "no FLYNNEL_HOST_PROFILE_NS: this process MEASURES its profile, so this row \
             is not a pinned arm"
        ),
        Err(env::VarError::NotUnicode(raw)) => eprintln!(
            "FLYNNEL_HOST_PROFILE_NS is set to something that is not UTF-8 ({raw:?}); the \
             library cannot read it either and will MEASURE instead, so this row is not \
             the pinned arm it was asked to be"
        ),
    }
}

fn main() {
    let items: usize = arg(1, 4_096);
    let reps: usize = arg(2, 200);
    let warmup: usize = arg(3, 20);

    report_pin();

    let mut buf: Vec<u64> = (0..items as u64).collect();
    let mut timings: Vec<f64> = Vec::with_capacity(reps);

    for i in 0..(warmup + reps) {
        let plan = JobPlan::new(0, items as u32).with_site(SiteRef::new(&SITE));
        let t0 = Instant::now();
        for_each_chunk_min_leaf(&plan, &mut buf, MIN_LEAF, |chunk| {
            for slot in chunk.iter_mut() {
                let mut acc = *slot;
                for _ in 0..8 {
                    acc = acc.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                }
                *slot = black_box(acc);
            }
        });
        let took = t0.elapsed().as_secs_f64() * 1e6;
        // The warmup dispatches are run and discarded on purpose: the
        // first dispatches of a process pay pool startup and the site's
        // first classifier ticks, which belong to neither arm.
        if i >= warmup {
            timings.push(took);
        }
    }

    timings.sort_by(|a, b| a.partial_cmp(b).expect("a timing is never NaN"));
    let median = timings[timings.len() / 2];
    let lo = timings[timings.len() / 20];
    let hi = timings[timings.len() - 1 - timings.len() / 20];

    // Median with the 5th and 95th percentiles beside it, because a
    // median alone cannot say whether two arms that differ are separated
    // or merely overlapping.
    println!("profile_cost {items} {reps} {median:.3} {lo:.3} {hi:.3}");
}
