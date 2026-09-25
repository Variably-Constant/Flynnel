//! How much stack is left for a body once the native chunk runner's
//! dispatch has nested as far as it goes.
//!
//! flynnel-pwrs's `flynnel_run_chunks_v1` runs a caller's body inside a
//! `join` on a plan that is never routed inline, which puts the whole
//! call on a pool worker, and then through
//! `for_each_chunk_indexed_min_leaf` on a plan carrying the site the
//! caller's key names. This runs that same dispatch with a body that
//! asks the operating system where its own thread's stack ends and
//! records the least room any body was left with. A caller whose body
//! needs more than that can overflow, and an overflow ends the process
//! with nothing able to catch it.
//!
//! The dispatch nests: a bisect recurses once per halving, and a worker
//! waiting on a join runs whatever it steals on the same stack. So one
//! dispatch alone understates the depth. The concurrent rows run several
//! dispatches at once so a waiting worker has another dispatch's leaves
//! to steal, which is the nesting that took reduce_inner past 4 MiB.
//!
//! The figure is room below the body's own frame. The body's frame is
//! small, and the guard page at the low end is not subtracted, so a
//! caller should keep a margin rather than plan to the byte.
//!
//! ```text
//! cargo run --release --example chunk_stack_headroom
//! ```

use std::sync::atomic::{AtomicUsize, Ordering};

use flynnel::sched::par_iter::for_each_chunk_indexed_min_leaf;
use flynnel::{JobPlan, LeafShape, join};

/// Log2 of `n` rounded up and capped at ten, the band the chunk runner
/// sizes its plan with.
fn band_for(n: usize) -> u8 {
    if n <= 1 {
        return 0;
    }
    (usize::BITS - (n - 1).leading_zeros()).min(10) as u8
}

/// The lowest address of this thread's stack.
#[cfg(unix)]
fn stack_low() -> usize {
    // SAFETY: pthread_getattr_np fills an attribute object for the
    // calling thread, which is then read and destroyed here and nowhere
    // else.
    unsafe {
        let mut attr: libc::pthread_attr_t = std::mem::zeroed();
        let got = libc::pthread_getattr_np(libc::pthread_self(), &mut attr);
        assert_eq!(got, 0, "pthread_getattr_np failed with {got}");
        let mut addr: *mut libc::c_void = std::ptr::null_mut();
        let mut size: libc::size_t = 0;
        let read = libc::pthread_attr_getstack(&attr, &mut addr, &mut size);
        assert_eq!(read, 0, "pthread_attr_getstack failed with {read}");
        let freed = libc::pthread_attr_destroy(&mut attr);
        assert_eq!(freed, 0, "pthread_attr_destroy failed with {freed}");
        addr as usize
    }
}

/// The lowest address of this thread's stack.
#[cfg(windows)]
fn stack_low() -> usize {
    unsafe extern "system" {
        fn GetCurrentThreadStackLimits(low: *mut usize, high: *mut usize);
    }
    let (mut low, mut high) = (0usize, 0usize);
    // SAFETY: both pointers are to locals that outlive the call.
    unsafe { GetCurrentThreadStackLimits(&mut low, &mut high) };
    low
}

/// Bytes of stack left below the caller's frame on this thread.
#[inline(never)]
fn stack_left() -> usize {
    let marker = std::hint::black_box(0u8);
    (&marker as *const u8 as usize).saturating_sub(stack_low())
}

/// The key this run's dispatches pass, as a caller of the chunk runner
/// passes one per kind of work.
const SITE_KEY: u64 = 0x57AC_4EAD;

/// Run the chunk runner's dispatch over `n` items and return the least
/// stack any body was left with.
fn one_dispatch(n: usize, min_leaf: usize) -> usize {
    let least = AtomicUsize::new(usize::MAX);
    let onto_a_worker = JobPlan::new(0, 8).with_leaf_shape(LeafShape::PortCompute);
    join(
        &onto_a_worker,
        || {
            let plan = JobPlan::new(band_for(n), n.min(u32::MAX as usize) as u32)
                .with_site(flynnel::site_for_key(SITE_KEY));
            let mut slots = vec![(); n];
            for_each_chunk_indexed_min_leaf(&plan, &mut slots, min_leaf, |_start, _chunk| {
                least.fetch_min(stack_left(), Ordering::Relaxed);
            });
        },
        || (),
    );
    let seen = least.load(Ordering::Relaxed);
    // No body ran if this is still the sentinel, and reporting it would
    // read as unlimited room, which is the worst figure to be wrong about.
    assert_ne!(seen, usize::MAX, "no body ran at n = {n}");
    seen
}

/// Run `callers` dispatches at once, each from its own thread, and
/// return the least stack any body in any of them was left with.
fn concurrent(n: usize, min_leaf: usize, callers: usize) -> usize {
    std::thread::scope(|s| {
        let handles: Vec<_> = (0..callers).map(|_| s.spawn(|| one_dispatch(n, min_leaf))).collect();
        handles
            .into_iter()
            .map(|h| match h.join() {
                Ok(least) => least,
                Err(payload) => std::panic::resume_unwind(payload),
            })
            .min()
            .expect("at least one caller ran")
    })
}

fn main() {
    let workers = flynnel::sched::arena::global_local_arena().local_worker_count();
    println!("workers {workers}; each has an 8 MiB stack");
    println!("{:>10} {:>9} {:>8}  {:>12}", "n", "min_leaf", "callers", "least left");
    for &n in &[1_000usize, 65_536, 1_048_576, 16_777_216] {
        for &min_leaf in &[1usize, 256] {
            for &callers in &[1usize, 4, 16] {
                let least = if callers == 1 {
                    one_dispatch(n, min_leaf)
                } else {
                    concurrent(n, min_leaf, callers)
                };
                println!("{n:>10} {min_leaf:>9} {callers:>8}  {:>9} KiB", least / 1024);
            }
        }
    }
}
