//! Repeated calibration draws on one host, under a stated load.
//!
//! The trust check refuses a record whose samples spread more than
//! `PROVISIONAL_SPREAD_PER_MILLE` of their median. On every host
//! measured so far no draw clears it, while five independent draws
//! agreed on their medians to within 8 percent - so the range is
//! describing something the medians do not have. Replacing the
//! statistic needs draws taken across known conditions, reporting both
//! candidates, and this is what takes them.
//!
//! # Run it with the sample line on
//!
//! ```sh
//! for d in 1 2 3 4 5; do
//!     FLYNNEL_PROFILE_SAMPLES=1 FLYNNEL_OCCUPANCY=1 \
//!         FLYNNEL_CALIBRATION_DIR="$tmp/draw_$d" \
//!         profile_draws 1 <load_threads>
//! done 2>&1 | tee draws.log
//! ```
//!
//! Each draw prints two `profile sweep:` lines on stderr - one per
//! crossover sweep - carrying the samples, their median, their range
//! and their interquartile range. This prints the installed profile
//! after each draw on stdout. Both streams belong in one file, because
//! a draw is a sweep pair followed by its profile and the order is
//! what pairs them.
//!
//! # How many draws belong in one process
//!
//! `calibrate_host_dispatch` measures only when no stored record clears
//! the trust check. Where one does, the first call measures and
//! publishes and every call after reads that back, so several draws in
//! one process is one measurement and some copies of it.
//!
//! Both kinds of host exist on this fleet: a Zen3 Linux guest writes
//! nothing that passes, and a Ryzen 9 7900X on bare metal does. So the
//! shape that is safe everywhere is a single draw per process against a
//! fresh `FLYNNEL_CALIBRATION_DIR`, and asking for several in one
//! process is a shortcut that works only where records are refused.
//!
//! Taking that shortcut on the wrong host is detected rather than
//! reported as a suspiciously steady one: two measurements never land
//! on identical nanosecond counts, so identical consecutive draws exit
//! 3.
//!
//! # What it deliberately does not do
//!
//! It picks no bound and judges no draw. The figures go to a reader.

use std::env;
use std::hint::black_box;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use flynnel::sched::par_iter::{HostDispatchProfile, calibrate_host_dispatch};

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

/// Burners held for the lifetime of the returned guard.
struct Load {
    stop: Arc<AtomicBool>,
    threads: Vec<std::thread::JoinHandle<()>>,
}

impl Load {
    fn start(n: usize) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let mut threads = Vec::with_capacity(n);
        for _ in 0..n {
            let stop = Arc::clone(&stop);
            threads.push(std::thread::spawn(move || {
                let mut acc = 0u64;
                while !stop.load(Ordering::Relaxed) {
                    for _ in 0..4_096 {
                        acc = acc.wrapping_mul(2_862_933_555_777_941_757).wrapping_add(3);
                    }
                    black_box(acc);
                }
            }));
        }
        Self { stop, threads }
    }

    /// Stop the burners and report any that panicked. A burner that
    /// died early means the draws after it were taken under less load
    /// than the rows claim.
    fn stop(self) {
        self.stop.store(true, Ordering::Relaxed);
        for (t, handle) in self.threads.into_iter().enumerate() {
            if let Err(panic) = handle.join() {
                eprintln!(
                    "load thread {t} panicked, so some draws carried less load than stated: {panic:?}"
                );
            }
        }
    }
}

fn main() {
    let draws: usize = arg(1, 5).max(1);
    let load_threads: usize = arg(2, 0);

    if env::var_os("FLYNNEL_HOST_PROFILE_NS").is_some() {
        eprintln!(
            "FLYNNEL_HOST_PROFILE_NS is set, so calibrate_host_dispatch installs that \
             pin and measures nothing; unset it or there is no draw to take"
        );
        std::process::exit(2);
    }
    if env::var_os("FLYNNEL_PROFILE_SAMPLES").is_none() {
        eprintln!(
            "FLYNNEL_PROFILE_SAMPLES is unset, so the samples behind each draw are not \
             printed and only their medians reach the log; the statistic this run \
             exists to compare cannot be computed from that"
        );
        std::process::exit(2);
    }

    let load = (load_threads > 0).then(|| Load::start(load_threads));
    // Let the burners reach the cores before the first draw, or it is
    // taken under less load than the row claims.
    if load.is_some() {
        std::thread::sleep(std::time::Duration::from_millis(250));
    }

    println!("draws {draws} load {load_threads}");
    let mut previous: Option<HostDispatchProfile> = None;
    let mut repeats = 0usize;
    for i in 1..=draws {
        let p = calibrate_host_dispatch();
        println!(
            "draw {i} {} {} {} load={load_threads}",
            p.dispatch_cost_ns, p.collapse_threshold_ns, p.jec_wake_threshold_ns
        );
        if previous == Some(p) {
            repeats += 1;
        }
        previous = Some(p);
    }

    if let Some(load) = load {
        load.stop();
    }

    // Two measurements never land on the same three nanosecond counts.
    // Draws that do are the stored record being handed back, and a
    // reader cannot tell that from a host that is simply steady.
    println!("repeated_draws {repeats}");
    if repeats > 0 {
        eprintln!(
            "{repeats} of the {draws} draws returned the previous draw's exact figures, \
             which is a stored record answering rather than a measurement. This host \
             writes records that clear the trust check, so the first draw measured and \
             published and the rest read it back. Take one draw per process against a \
             fresh FLYNNEL_CALIBRATION_DIR each, rather than clearing the host's store: \
             that store is what other processes on the machine are reading."
        );
        std::process::exit(3);
    }
}
