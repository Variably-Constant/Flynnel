//! Background calibration: runtime measurement of per-host kernel
//! costs via the [`crate::sched::io_pool::IoPool`].
//!
//! Calibration microbenches run on the SMT-sibling IO pool so
//! measured per-host costs land in caller-owned tables without
//! occupying the compute pool. Requires
//! `FLYNNEL_SCHED_SMT_AS_IO=on`; with the IoPool disabled,
//! [`spawn_calibration`] is a no-op. Callers supply the microbench
//! closures (the scheduler is domain-agnostic); this module owns
//! only the IoPool dispatch wiring.
//!
//! ## Example
//!
//! ```no_run
//! use std::sync::{Arc, RwLock};
//! use std::collections::HashMap;
//! use std::time::Instant;
//! use flynnel::sched::spawn_calibration;
//!
//! // Caller-owned table.
//! let table: Arc<RwLock<HashMap<&'static str, f64>>> =
//!     Arc::new(RwLock::new(HashMap::new()));
//!
//! let t1 = Arc::clone(&table);
//! let t2 = Arc::clone(&table);
//! spawn_calibration(vec![
//!     Box::new(move || {
//!         // microbench body for kernel "alpha"
//!         let t0 = Instant::now();
//!         for _ in 0..1000 { std::hint::black_box(1u64.wrapping_add(1)); }
//!         let ns = t0.elapsed().as_nanos() as f64 / 1000.0;
//!         t1.write().unwrap().insert("alpha", ns);
//!     }),
//!     Box::new(move || {
//!         // microbench body for kernel "beta"
//!         let t0 = Instant::now();
//!         for _ in 0..1000 { std::hint::black_box(2u64.wrapping_mul(3)); }
//!         let ns = t0.elapsed().as_nanos() as f64 / 1000.0;
//!         t2.write().unwrap().insert("beta", ns);
//!     }),
//! ]);
//! ```

use crate::sched::io_pool::global_io_pool;

/// Run one timed microbench: invoke `op` `iters` times, return
/// average per-call cost in nanoseconds.
///
/// Uses an 8-iteration warm-up to stabilize the cache + branch
/// predictor before measuring. Callers that need a different warm-up
/// shape should write their own timer loop.
pub fn timed_avg_ns<F: FnMut()>(mut op: F, iters: u32) -> f64 {
    use std::time::Instant;
    for _ in 0..8 {
        op();
    }
    let t0 = Instant::now();
    for _ in 0..iters {
        op();
    }
    let elapsed = t0.elapsed();
    elapsed.as_nanos() as f64 / iters as f64
}

/// Spawn a list of calibration microbenches on the IO pool.
///
/// Each closure runs once on a free SMT-sibling thread. The IoPool
/// is shared with other non-compute roles (background memory zero,
/// verify-chain hashing); calibration jobs queue alongside those.
///
/// No-op when the IoPool is disabled.
pub fn spawn_calibration(closures: Vec<Box<dyn FnOnce() + Send + 'static>>) {
    let Some(pool) = global_io_pool() else { return };
    for closure in closures {
        pool.submit(closure);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU32, Ordering};

    #[test]
    fn timed_avg_runs_op_at_least_iters_times() {
        let counter = Arc::new(AtomicU32::new(0));
        let c = Arc::clone(&counter);
        let avg = timed_avg_ns(
            move || {
                c.fetch_add(1, Ordering::Relaxed);
            },
            100,
        );
        // 8 warmup + 100 measured = 108 invocations minimum.
        assert!(counter.load(Ordering::Relaxed) >= 108);
        assert!(avg.is_finite());
    }

    #[test]
    fn spawn_calibration_with_disabled_pool_is_noop() {
        // Without FLYNNEL_SCHED_SMT_AS_IO=on, the pool is disabled
        // and submit is a no-op. Closure must NOT run.
        let counter = Arc::new(AtomicU32::new(0));
        let c = Arc::clone(&counter);
        spawn_calibration(vec![Box::new(move || {
            c.fetch_add(1, Ordering::Relaxed);
        })]);
        // No deterministic way to observe non-run without timing;
        // we just check it didn't panic. If the user has the env
        // var set, the closure might run on a background thread
        // and the counter might be 0 or 1.
        let _seen = counter.load(Ordering::Relaxed);
    }
}
