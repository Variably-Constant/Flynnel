//! Default CPU backend implementing [`DispatchBackend`]. The two
//! dispatch methods take different code paths:
//!
//! - [`CpuBackend::dispatch_parallel_for`] routes through Flynnel's
//!   work-stealing arena via [`crate::sched::par_iter`].
//! - [`CpuBackend::dispatch_one`] spawns a fresh OS thread with
//!   `std::thread::spawn`. The arena's `StackJob` lifetime
//!   constraints do not admit an owned trait-object closure, so
//!   thread-spawn is the documented escape hatch for single-shot
//!   owned-closure work.
//!
//! This is the always-available baseline. Every host has it; the
//! [`crate::backend::registry`] auto-registers an instance on first
//! access. Callers that explicitly want CPU dispatch use
//! [`crate::backend::registry::cpu_backend`].

use crate::backend::{Backend, BackendCapabilities, DispatchBackend};
use crate::sched::JobPlan;
use crate::sched::par_iter::for_each_chunk_indexed_min_leaf;

/// Always-available CPU backend. `dispatch_parallel_for` routes
/// to the work-stealing arena via [`crate::sched::par_iter`];
/// `dispatch_one` spawns a fresh OS thread (see module docs for
/// why).
#[derive(Debug)]
pub struct CpuBackend {
    caps: BackendCapabilities,
}

impl Default for CpuBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl CpuBackend {
    /// Construct a new CPU backend with capabilities probed from
    /// the running host's thread count.
    pub fn new() -> Self {
        Self {
            caps: BackendCapabilities::cpu_defaults(),
        }
    }
}

impl DispatchBackend for CpuBackend {
    fn id(&self) -> Backend {
        Backend::Cpu
    }

    fn capabilities(&self) -> BackendCapabilities {
        self.caps
    }

    fn dispatch_parallel_for(&self, count: u32, work: &(dyn Fn(u32) + Send + Sync)) {
        if count == 0 {
            return;
        }
        // Route through the arena's indexed-chunk iterator. We
        // build a u32 index slice and rely on the arena's bisect
        // splitter for parallelization; per-element work runs
        // through the caller's `work` closure.
        let mut idx: Vec<u32> = (0..count).collect();
        let plan = JobPlan::new(0, count);
        for_each_chunk_indexed_min_leaf(&plan, &mut idx, 1, |start, slice| {
            for (offset, &i) in slice.iter().enumerate() {
                // `i` already equals `start + offset` by
                // construction of the index slice, but reading it
                // through the slice keeps the body identical to a
                // pure-functional indexed-for shape.
                debug_assert_eq!(i, start as u32 + offset as u32);
                work(i);
            }
        });
    }

    fn dispatch_one(&self, work: Box<dyn FnOnce() + Send>) {
        // Fire-and-forget on a fresh OS thread. The work-stealing
        // arena requires `StackJob` lifetimes the trait-object
        // shape cannot satisfy here; spawning a thread is the
        // documented escape hatch for owned-closure single-shot
        // work.
        std::thread::spawn(work);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::Arc;

    #[test]
    fn cpu_backend_id_is_cpu() {
        let cpu = CpuBackend::new();
        assert_eq!(cpu.id(), Backend::Cpu);
    }

    #[test]
    fn capabilities_match_cpu_defaults() {
        let cpu = CpuBackend::new();
        let caps = cpu.capabilities();
        let defaults = BackendCapabilities::cpu_defaults();
        assert_eq!(caps.simt_width, defaults.simt_width);
        assert_eq!(caps.max_threads_in_flight, defaults.max_threads_in_flight);
    }

    #[test]
    fn dispatch_parallel_for_invokes_each_index_exactly_once() {
        let cpu = CpuBackend::new();
        let counters: Vec<AtomicU32> = (0..1024).map(|_| AtomicU32::new(0)).collect();
        let counters_ref = &counters;
        cpu.dispatch_parallel_for(1024, &|i| {
            counters_ref[i as usize].fetch_add(1, Ordering::Relaxed);
        });
        for (i, c) in counters.iter().enumerate() {
            assert_eq!(
                c.load(Ordering::Relaxed),
                1,
                "index {i} should be invoked exactly once"
            );
        }
    }

    #[test]
    fn dispatch_parallel_for_zero_count_is_noop() {
        let cpu = CpuBackend::new();
        let touched = Arc::new(AtomicU32::new(0));
        let t = Arc::clone(&touched);
        cpu.dispatch_parallel_for(0, &move |_| {
            t.fetch_add(1, Ordering::Relaxed);
        });
        assert_eq!(touched.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn dispatch_one_executes_closure_on_spawned_thread() {
        let cpu = CpuBackend::new();
        let observed = Arc::new(AtomicU32::new(0));
        let o = Arc::clone(&observed);
        cpu.dispatch_one(Box::new(move || {
            o.store(0xDEAD_BEEF, Ordering::SeqCst);
        }));
        // Spin-wait briefly for the spawned thread.
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
        while observed.load(Ordering::SeqCst) != 0xDEAD_BEEF {
            if std::time::Instant::now() > deadline {
                panic!("dispatch_one closure never ran");
            }
            std::thread::yield_now();
        }
        assert_eq!(observed.load(Ordering::SeqCst), 0xDEAD_BEEF);
    }

    #[test]
    fn register_kernel_returns_not_supported() {
        let cpu = CpuBackend::new();
        let res = cpu.register_kernel("test", b"unused");
        assert!(res.is_err());
    }
}
