//! `IoPool`: SMT-sibling thread pool for non-compute roles.
//!
//! With `physical_cores = 8` on Zen+ R7 2700, CPUs 8..15 (SMT siblings
//! of cores 0..7) sit idle when the compute pool runs at the default
//! 8-physical-cores-only count. This module provides a separate
//! thread pool that, when enabled, parks one worker per physical core
//! on the SMT sibling and uses it for **non-compute** roles:
//!
//! - Background calibration runs (microbenchmarks that fill the
//!   dispatch tensor at runtime)
//! - Async prefetch streams for large NTT / Karatsuba operands
//! - Verification hash-chain (BLAKE3 over per-stripe outputs)
//! - GPU event polling
//!
//! SMT siblings share L1d/L2 with their compute partner (hand-off
//! is free) but contest the IMUL / FMA pipes, so they stay off the
//! compute queue and take only async / IO / verify tasks.
//!
//! Off by default; `FLYNNEL_SCHED_SMT_AS_IO=on` (or `=1`, `=true`)
//! enables it. Process-global via [`global_io_pool`]. A separate
//! pool (FIFO channel, own pinning, channel-send submission)
//! keeps IO out of the compute workers' steal loop.
//!

use core::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};
use std::thread::{self, JoinHandle};

use crate::sched::notify_ring::{NotifyHub, NotifySender};

/// A unit of non-compute work to be run on an SMT-sibling thread.
///
/// Boxed closure for simplicity. The boxing cost is amortized by the
/// fact that IO tasks are coarse-grained (microbenchmarks,
/// hash-chains, prefetch sweeps) - not the fine-grained limb-mul
/// fragments the compute pool runs.
pub type IoTask = Box<dyn FnOnce() + Send + 'static>;

/// SMT-sibling thread pool for non-compute roles. Distinct from
/// the compute work-stealing pool ([`crate::sched::arena_local::LocalArena`])
/// to avoid interleaving async IO work with compute jobs on the
/// same threads.
///
/// One worker per physical core when enabled. Each worker pulls
/// `IoTask`s from a shared MPMC notify ring (FlynnelRing + per-
/// consumer Parker); round-robin wake distributes new work.
/// No work-stealing inside the IO pool - the cost model is
/// "submit + run to completion, do not preempt", which fits
/// async/IO/calibration work well.
pub struct IoPool {
    hub: NotifyHub<IoTask>,
    sender: NotifySender<IoTask>,
    workers: Vec<Option<JoinHandle<()>>>,
    shutdown: Arc<AtomicBool>,
}

impl std::fmt::Debug for IoPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IoPool")
            .field("n_workers", &self.workers.len())
            .finish()
    }
}

impl IoPool {
    /// Spawn a new IoPool with `n_workers` background threads.
    /// Pass `n_workers = physical_cores` for the default
    /// "one IO sibling per physical core" pattern.
    ///
    /// The pool is held in an `Arc` so callers can reach it from
    /// any thread; the workers reference-count this Arc.
    pub fn new(n_workers: usize) -> Arc<Self> {
        let n = n_workers.max(1);
        // 1024-slot ring covers typical IO bursts. Each consumer
        // registers its own parker on the worker thread.
        const IO_RING_CAPACITY: usize = 1024;
        let hub = NotifyHub::<IoTask>::new(IO_RING_CAPACITY, n);
        let sender = hub.sender();
        let shutdown = Arc::new(AtomicBool::new(false));
        let mut handles: Vec<Option<JoinHandle<()>>> = Vec::with_capacity(n);

        for idx in 0..n {
            let hub_for_worker = hub.clone();
            let sd = Arc::clone(&shutdown);
            let h = thread::Builder::new()
                .name(format!("flynnel-io-{idx}"))
                .spawn(move || io_worker_loop(idx, hub_for_worker, sd))
                .expect("IO worker thread spawn must succeed");
            handles.push(Some(h));
        }

        Arc::new(Self {
            hub,
            sender,
            workers: handles,
            shutdown,
        })
    }

    /// Submit a closure to run on one of the IO workers. Returns
    /// immediately; the closure is queued and picked up by the
    /// first available IO worker.
    ///
    /// Use this for async / non-compute work:
    /// - Background microbenchmark runs that populate dispatch
    ///   tensors
    /// - BLAKE3 verification of stripe outputs
    /// - Prefetch sweeps that warm L3 ahead of compute
    /// - GPU event polling on the CUDA host-side
    /// - Cross-node MPI/D-UMPA send/recv
    ///
    /// Do NOT submit compute work here - the IO workers are on
    /// SMT siblings of the compute cores and would contest the
    /// IMUL/FMA pipes.
    pub fn submit<F: FnOnce() + Send + 'static>(&self, task: F) {
        // Disregard send errors: the only way send fails is if
        // the hub is shut down, which means the pool is in Drop.
        // Dropping the task is the right behavior then.
        drop(self.sender.send(Box::new(task)));
    }

    /// Worker count.
    pub fn worker_count(&self) -> usize {
        self.workers.len()
    }
}

impl Drop for IoPool {
    fn drop(&mut self) {
        // Signal shutdown.
        self.shutdown.store(true, Ordering::Release);
        // Shut down the notify hub: every parked worker wakes
        // and its NotifyReceiver::recv returns None on the next
        // poll (after draining any in-flight items).
        self.hub.shutdown();
        for slot in &mut self.workers {
            if let Some(h) = slot.take() {
                drop(h.join());
            }
        }
    }
}

fn io_worker_loop(_idx: usize, hub: NotifyHub<IoTask>, shutdown: Arc<AtomicBool>) {
    let rx = hub.register_consumer();
    while !shutdown.load(Ordering::Acquire) {
        match rx.recv() {
            Some(task) => {
                // Run the task. Closure panics propagate up here -
                // we deliberately do NOT catch_unwind because IO
                // tasks should be designed not to panic; if one
                // does, surfacing it loudly during testing is the
                // right behavior.
                task();
            }
            None => {
                // Hub closed - pool dropping. Exit cleanly.
                break;
            }
        }
    }
}

/// Lazily-initialized process-global IO pool. Enabled via
/// `FLYNNEL_SCHED_SMT_AS_IO=on|1|true`. Returns `None` when
/// disabled; in that case callers should run their async work
/// inline on the caller thread (no SMT sibling available).
///
/// Worker count = physical core count from
/// [`crate::cpu_info`]. On Zen+ R7 2700 that's 8 IO
/// workers, sitting on SMT siblings of the 8 compute cores.
pub fn global_io_pool() -> Option<&'static Arc<IoPool>> {
    static POOL: OnceLock<Option<Arc<IoPool>>> = OnceLock::new();
    POOL.get_or_init(|| {
        let enabled = std::env::var("FLYNNEL_SCHED_SMT_AS_IO")
            .map(|v| {
                let v = v.to_ascii_lowercase();
                v == "on" || v == "1" || v == "true"
            })
            .unwrap_or(false);
        if !enabled {
            return None;
        }
        let phys = crate::cpu_info::cpu_info().physical_cores as usize;
        Some(IoPool::new(phys))
    })
    .as_ref()
}

/// Submit `task` to the SMT-sibling IO pool if enabled, otherwise
/// run it inline on the caller thread.
///
/// Convenience wrapper for code that wants to offload work
/// opportunistically: when SMT_AS_IO is enabled, the task runs on
/// an idle SMT sibling; when disabled, the task runs inline so the
/// behavior is correct either way.
///
/// Inline-fallback is sound for tasks that are independent of the
/// caller's subsequent work; for tasks where you NEED async (e.g.,
/// to overlap compute and prefetch), check `global_io_pool()
/// .is_some()` first and pick a different strategy when no IO
/// pool is available.
pub fn submit_io_or_inline<F: FnOnce() + Send + 'static>(task: F) {
    match global_io_pool() {
        Some(pool) => pool.submit(task),
        None => task(),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU32;
    use std::time::{Duration, Instant};

    #[test]
    fn pool_with_one_worker_runs_task() {
        let pool = IoPool::new(1);
        let counter = Arc::new(AtomicU32::new(0));
        let c = Arc::clone(&counter);
        pool.submit(move || {
            c.fetch_add(1, Ordering::SeqCst);
        });
        // Wait up to 5s for the task to run.
        let deadline = Instant::now() + Duration::from_secs(5);
        while counter.load(Ordering::SeqCst) == 0 {
            if Instant::now() > deadline {
                panic!("IO task did not run within 5s");
            }
            thread::yield_now();
        }
        drop(pool);
    }

    #[test]
    fn pool_runs_many_tasks_across_workers() {
        const N: u32 = 128;
        let pool = IoPool::new(4);
        let counter = Arc::new(AtomicU32::new(0));
        for _ in 0..N {
            let c = Arc::clone(&counter);
            pool.submit(move || {
                c.fetch_add(1, Ordering::SeqCst);
            });
        }
        let deadline = Instant::now() + Duration::from_secs(10);
        while counter.load(Ordering::SeqCst) < N {
            if Instant::now() > deadline {
                panic!(
                    "only {}/{N} IO tasks completed within 10s",
                    counter.load(Ordering::SeqCst)
                );
            }
            thread::yield_now();
        }
        assert_eq!(counter.load(Ordering::SeqCst), N);
        drop(pool);
    }

    #[test]
    fn pool_drop_joins_workers_cleanly() {
        let pool = IoPool::new(2);
        let t0 = Instant::now();
        drop(pool);
        let elapsed = t0.elapsed();
        assert!(
            elapsed < Duration::from_secs(2),
            "IoPool drop took {elapsed:?}, expected < 2s"
        );
    }

    #[test]
    fn submit_io_or_inline_runs_when_pool_disabled() {
        // FLYNNEL_SCHED_SMT_AS_IO is not set in the test
        // environment, so global_io_pool() returns None and the
        // task should run inline.
        let counter = Arc::new(AtomicU32::new(0));
        let c = Arc::clone(&counter);
        submit_io_or_inline(move || {
            c.fetch_add(1, Ordering::SeqCst);
        });
        // Inline execution means counter is incremented before
        // submit_io_or_inline returns.
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }
}
