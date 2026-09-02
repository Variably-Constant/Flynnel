//! `NumaArena`: per-NUMA-node composition of [`LocalArena`].
//!
//! On single-NUMA hosts (most desktops) this collapses to a
//! single underlying `LocalArena` and the cross-node code paths
//! are dead branches with zero overhead.
//!
//! On multi-NUMA hosts (Colab Genoa, dual-socket Xeon /
//! Threadripper) it creates one sub-arena per NUMA node and
//! routes work to the caller's current-thread node by default.
//! Per ARCAS (arXiv:2503.11460, March 2025) and Olivier-Prins
//! (ROSS '11), this bounds cross-socket cache-coherence traffic
//! to `O(num_nodes)` rather than `O(num_threads)` while letting
//! the scheduler still rebalance via cross-node steal when one
//! node is idle and another is overloaded.
//!
//!
//! ## Per-node sizing
//!
//! Workers per node = `physical_cores_per_node` from the cached
//! `NumaTopology` + `CpuTopology` (computed as
//! `cpus_in_node(n).len() / smt_threads`). The caller-supplied
//! `cpu_set` per sub-arena restricts pinning to the node's own
//! CPUs.
//!
//! ## Submit + try_run_one routing
//!
//! - `submit(job)`: pushes to caller's current-node injector and
//!   burst-wakes that node's workers. Single-NUMA: just the one
//!   node.
//! - `try_run_one(rng)`: tries caller's local node first
//!   (injector + peer-steal); if empty, walks other nodes'
//!   injectors as fallback. Single-NUMA: just the local node.
//!
//! Cross-node stealing is unconditional: any idle worker may
//! probe other nodes.

use core::sync::atomic::AtomicU64;
use std::sync::Arc;

use crate::sched::injector::InjectorSteal as Steal;

use crate::numa_topology::numa_topology;
use crate::sched::arena_local::LocalArena;
use crate::sched::job::JobRef;
use crate::sched::numa_alloc::NUMA_NODE_LOCAL;

/// Per-NUMA arena composition. On single-NUMA hosts collapses
/// to one sub-arena.
pub struct NumaArena {
    /// One sub-arena per NUMA node.
    nodes: Vec<Arc<LocalArena>>,
    /// CPU-to-node lookup, indexed by OS logical-CPU id. Empty
    /// on platforms where the probe failed; defaults to node 0.
    cpu_to_node: Vec<u32>,
    /// Counter for cross-node fallback rotation (so an external
    /// waiter doesn't always probe the same other node).
    cross_node_rotor: AtomicU64,
}

impl std::fmt::Debug for NumaArena {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NumaArena")
            .field("n_nodes", &self.nodes.len())
            .field("workers_per_node",
                &self.nodes.iter().map(|a| a.worker_count()).collect::<Vec<_>>())
            .finish()
    }
}

impl NumaArena {
    /// Diagnostic snapshot of every node's pool state; see
    /// [`LocalArena::debug_snapshot`].
    pub fn debug_snapshot(&self) -> String {
        self.nodes
            .iter()
            .enumerate()
            .map(|(i, a)| format!("node {i}:\n{}", a.debug_snapshot()))
            .collect()
    }
}

impl NumaArena {
    /// Build a NumaArena sized for the current host. Reads
    /// `NumaTopology` + `CpuTopology`, builds one `LocalArena`
    /// per visible NUMA node with workers pinned to that node's
    /// CPUs.
    ///
    /// `workers_per_node` overrides the per-node worker count;
    /// pass `None` for the default (physical cores in the node).
    pub fn new(workers_per_node: Option<usize>) -> Arc<Self> {
        let topo = numa_topology();
        let n_nodes = topo.num_nodes.max(1) as usize;
        let cpu_to_node = topo.node_of_cpu.clone();

        // Materialise all core_ids once; partition into per-node
        // sub-sets by the topology's node_of_cpu map.
        let pinning_enabled = !pin_disabled_env_local();
        let all_core_ids = if pinning_enabled {
            core_affinity::get_core_ids()
        } else {
            None
        };

        let mut nodes: Vec<Arc<LocalArena>> = Vec::with_capacity(n_nodes);
        for node_id in 0..n_nodes as u32 {
            // CPUs in this node, mapped to CoreId values when
            // pinning is enabled.
            let cpus_in_node: Vec<u32> = topo.cpus_in_node(node_id);
            let node_cpu_set: Option<Vec<core_affinity::CoreId>> = all_core_ids.as_ref().map(|ids| {
                cpus_in_node
                    .iter()
                    .filter_map(|&cpu| ids.get(cpu as usize).copied())
                    .collect()
            });

            // Worker count: caller override OR physical cores in
            // node. Physical cores ~= cpus_in_node / smt_threads.
            let cpu_info = crate::cpu_info::cpu_info();
            let smt = (1usize << cpu_info.smt_threads_log2).max(1);
            let phys_in_node = (cpus_in_node.len() / smt).max(1);
            let primary_count = workers_per_node.unwrap_or(phys_in_node);
            // SMT extension: spawn one sibling per primary when SMT
            // is detected on this host AND primaries don't already
            // cover the node's logical-thread count. Siblings start
            // parked and wake when a plan with `use_smt = true` is
            // dispatched. When primaries already equal logical
            // threads (the new SMT-on default), the extension is
            // zero (no duplicated workers).
            let smt_extension = if smt > 1 && primary_count < cpus_in_node.len() {
                primary_count.saturating_mul(smt - 1)
            } else {
                0
            };

            nodes.push(LocalArena::with_smt_extension(
                primary_count,
                smt_extension,
                node_cpu_set,
            ));
        }

        Arc::new(Self {
            nodes,
            cpu_to_node,
            cross_node_rotor: AtomicU64::new(0),
        })
    }

    /// Number of sub-arenas (= number of NUMA nodes). 1 on
    /// single-NUMA hosts.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Migrate the active K_gating across every worker in every
    /// sub-arena. Single Release-store pass per worker per tier
    /// (~30 ns total on a 16-worker arena with 4 tiers). Per-op
    /// cost on subsequent pushes is unchanged.
    pub fn migrate_all_workers_k_gating(&self, gating: super::k_gating::KGating) {
        for arena in &self.nodes {
            arena.migrate_all_workers_k_gating(gating);
        }
    }

    /// Sum the burst-vs-single profile across every worker in
    /// every sub-arena. Returns the global burst ratio in
    /// `[0.0, 1.0]`.
    pub fn global_burst_ratio(&self) -> f32 {
        let mut bursts: u64 = 0;
        let mut singles: u64 = 0;
        for arena in &self.nodes {
            for w in arena.worker_stats() {
                use core::sync::atomic::Ordering;
                bursts = bursts.saturating_add(w.burst_pushes.load(Ordering::Relaxed));
                singles = singles.saturating_add(w.single_pushes.load(Ordering::Relaxed));
            }
        }
        let total = bursts + singles;
        if total == 0 {
            0.5
        } else {
            (bursts as f32) / (total as f32)
        }
    }

    /// Total worker count across all nodes.
    pub fn total_workers(&self) -> usize {
        self.nodes.iter().map(|a| a.worker_count()).sum()
    }

    /// Iterate every per-worker stats handle across every node.
    /// Used by [`crate::sched::split_observer`] to compute pool-
    /// wide steal pressure.
    pub fn iter_worker_stats(&self) -> impl Iterator<Item = &Arc<super::arena_local::WorkerStats>> {
        self.nodes.iter().flat_map(|a| a.worker_stats().iter())
    }

    /// Acquire one SMT request across every sub-arena, returning
    /// a `Vec<SmtGuard>` whose drop releases each per-node
    /// request. Workers on EACH node's LocalArena that are
    /// SMT-siblings will unpark and join the work-stealing loop
    /// while the guards are held.
    ///
    /// Cost: one atomic fetch_add per node + unpark of each
    /// sibling parker on the 0->1 edge per node. Sub-microsecond
    /// for single-NUMA hosts.
    pub fn acquire_smt(&self) -> Vec<super::arena_local::SmtGuard> {
        self.nodes.iter().map(|n| n.acquire_smt()).collect()
    }

    /// Single-NUMA fast path for [`Self::acquire_smt`]. Returns
    /// `Some(guard)` on hosts with exactly one NUMA node (the common
    /// case for desktop/laptop hardware and single-socket servers).
    /// Multi-node hosts return `None` so the caller falls through to
    /// the allocating `Vec<SmtGuard>` path.
    ///
    /// Avoids the per-external-dispatch `Vec::with_capacity` +
    /// `.collect()` allocation on the SMT-on path. The `Vec` allocation
    /// is ~32 bytes per call; for SMT-on workloads under criterion's
    /// 30k iterations per cell that compounds to ~1.5ms of pure
    /// allocator overhead per bench cell. The fast path returns the
    /// `SmtGuard` on the caller's stack instead.
    #[inline]
    pub fn try_acquire_smt_single(&self) -> Option<super::arena_local::SmtGuard> {
        if self.nodes.len() == 1 {
            Some(self.nodes[0].acquire_smt())
        } else {
            None
        }
    }

    /// Whether this arena has exactly one NUMA sub-arena. Single-NUMA
    /// hosts can use the fast paths on accessors like
    /// [`Self::try_acquire_smt_single`].
    #[inline]
    pub fn is_single_numa(&self) -> bool {
        self.nodes.len() == 1
    }

    /// Return an `Arc<LocalArena>` clone for the only NUMA sub-arena
    /// when the host is single-NUMA. Returns `None` on multi-NUMA
    /// hosts. Used by external_dispatch to acquire the per-node
    /// arena handle needed for `with_external_worker_ctx`.
    #[inline]
    pub fn single_node_arc(&self) -> Option<Arc<super::arena_local::LocalArena>> {
        if self.nodes.len() == 1 {
            Some(Arc::clone(&self.nodes[0]))
        } else {
            None
        }
    }

    /// Resolve a `numa_hint` (or the caller's local node) to the
    /// matching `Arc<LocalArena>`. Always returns a valid Arc;
    /// clamps to node 0 on out-of-range hints. Used by
    /// external_dispatch fallback on multi-NUMA hosts.
    #[inline]
    pub fn node_arc(&self, node_hint: Option<u32>) -> Arc<super::arena_local::LocalArena> {
        Arc::clone(&self.nodes[self.resolve_node_idx(node_hint)])
    }

    /// Resolve a node id (or [`NUMA_NODE_LOCAL`]) to a sub-arena
    /// index. Out-of-range / unknown ids clamp to node 0.
    fn resolve_node_idx(&self, node_hint: Option<u32>) -> usize {
        match node_hint {
            Some(n) if n != NUMA_NODE_LOCAL && (n as usize) < self.nodes.len() => n as usize,
            _ => current_thread_node_idx(&self.cpu_to_node, self.nodes.len()),
        }
    }

    /// Submit a [`JobRef`] to the caller's current-node sub-arena.
    /// If `node_hint` is set, routes to that node instead (or
    /// node 0 if out of range).
    ///
    /// # Safety
    ///
    /// Same as [`LocalArena::submit`]: the underlying job's
    /// captured state must remain valid until the job runs.
    pub(crate) unsafe fn submit(&self, job: JobRef, node_hint: Option<u32>) {
        let idx = self.resolve_node_idx(node_hint);
        // SAFETY: this function's `# Safety` clause forwards
        // the captured-state-validity precondition straight to
        // `LocalArena::submit`, which carries the same contract.
        unsafe { self.nodes[idx].submit(job) };
    }

    /// Try to find and execute one job. Prefers the caller's
    /// local node; falls back to cross-node probes rotating
    /// through the remaining nodes.
    pub fn try_run_one(&self, rng_state: &mut u64) -> bool {
        let local_idx = current_thread_node_idx(&self.cpu_to_node, self.nodes.len());
        if self.nodes[local_idx].try_run_one(rng_state) {
            return true;
        }
        // Cross-node fallback: walk remaining nodes starting at
        // a rotor offset so different waiters probe different
        // nodes first.
        if self.nodes.len() <= 1 {
            return false;
        }
        let rotor = self.cross_node_rotor.fetch_add(1, core::sync::atomic::Ordering::Relaxed)
            as usize
            % (self.nodes.len() - 1);
        for offset in 0..self.nodes.len() - 1 {
            let candidate = (local_idx + 1 + (rotor + offset) % (self.nodes.len() - 1))
                % self.nodes.len();
            if candidate == local_idx {
                continue;
            }
            // Try cross-node injector steal directly (avoid
            // recursive try_run_one which would also do peer
            // steals on the remote node - we want only the
            // injector / leader-eligible path here).
            if let Steal::Success(job) = self.nodes[candidate].injector_view().steal() {
                // SAFETY: `JobRef::execute` requires the
                // captured-state pointer to remain valid until
                // the call returns; the producer that pushed
                // the job into the injector upholds that
                // contract via its own `# Safety` clause on
                // `LocalArena::submit`.
                unsafe { job.execute() };
                return true;
            }
        }
        false
    }
}

/// Read `FLYNNEL_SCHED_PIN` once (mirrors the same env probe in
/// `arena_local::pin_disabled_env`). Default is pinning DISABLED;
/// pass `FLYNNEL_SCHED_PIN=on` to opt in. See arena_local for the
/// full rationale (bench-driven).
fn pin_disabled_env_local() -> bool {
    static CACHE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| {
        match std::env::var("FLYNNEL_SCHED_PIN") {
            Ok(v) => {
                let v = v.to_ascii_lowercase();
                if v == "on" || v == "1" || v == "true" {
                    false // pinning enabled
                } else {
                    true
                }
            }
            Err(_) => true, // Default: pinning disabled
        }
    })
}

/// Map the current thread's logical CPU to a node-index, clamped
/// to `n_nodes - 1`.
fn current_thread_node_idx(cpu_to_node: &[u32], n_nodes: usize) -> usize {
    let cpu = current_thread_cpu();
    let n = cpu_to_node.get(cpu).copied().unwrap_or(0) as usize;
    n.min(n_nodes.saturating_sub(1))
}

#[cfg(target_os = "windows")]
fn current_thread_cpu() -> usize {
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetCurrentProcessorNumber() -> u32;
    }
    // SAFETY: `GetCurrentProcessorNumber` takes no arguments,
    // touches no caller memory, and has no preconditions on
    // Windows Vista or newer.
    unsafe { GetCurrentProcessorNumber() as usize }
}

#[cfg(target_os = "linux")]
fn current_thread_cpu() -> usize {
    #[link(name = "c")]
    unsafe extern "C" {
        fn sched_getcpu() -> core::ffi::c_int;
    }
    let cpu = unsafe { sched_getcpu() };
    if cpu < 0 { 0 } else { cpu as usize }
}

#[cfg(not(any(target_os = "windows", target_os = "linux")))]
fn current_thread_cpu() -> usize {
    0
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::time::{Duration, Instant};

    use crate::foundation::Variant;
    use crate::sched::job::{NUMA_HINT_ANY, StackJob};
    use crate::sched::latch::CoreLatch;

    #[test]
    fn numa_arena_collapses_to_single_arena_on_single_numa_host() {
        // On this single-NUMA Zen+ R7 2700 host (numa_topology()
        // reports num_nodes = 1), NumaArena should produce one
        // sub-arena.
        let a = NumaArena::new(Some(4));
        assert_eq!(a.node_count(), numa_topology().num_nodes as usize);
        assert!(a.total_workers() >= 1);
    }

    #[test]
    fn numa_arena_submit_runs_job() {
        let arena = NumaArena::new(Some(2));
        let counter = Arc::new(AtomicU32::new(0));
        let c = Arc::clone(&counter);
        let job = StackJob::new(
            move |_stolen| { c.fetch_add(1, Ordering::SeqCst); },
            CoreLatch::new(),
        );
        unsafe {
            let r = job.as_job_ref(2, NUMA_HINT_ANY, Variant::Faithful);
            arena.submit(r, None);
        }
        let deadline = Instant::now() + Duration::from_secs(5);
        while !job.latch.is_set() {
            if Instant::now() > deadline {
                panic!("NumaArena job did not complete within 5s");
            }
            std::thread::yield_now();
        }
        assert_eq!(counter.load(Ordering::SeqCst), 1);
        drop(arena);
    }

    #[test]
    fn numa_arena_submit_many_completes() {
        const N: u32 = 128;
        let arena = NumaArena::new(Some(2));
        let counter = Arc::new(AtomicU32::new(0));
        let jobs: Vec<_> = (0..N)
            .map(|_| {
                let c = Arc::clone(&counter);
                StackJob::new(
                    move |_stolen| { c.fetch_add(1, Ordering::SeqCst); },
                    CoreLatch::new(),
                )
            })
            .collect();
        for j in &jobs {
            unsafe { arena.submit(j.as_job_ref(2, NUMA_HINT_ANY, Variant::Faithful), None) };
        }
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            let pending = jobs.iter().filter(|j| !j.latch.is_set()).count();
            if pending == 0 { break; }
            if Instant::now() > deadline {
                panic!("only {}/{N} jobs completed within 10s", N as usize - pending);
            }
            std::thread::yield_now();
        }
        assert_eq!(counter.load(Ordering::SeqCst), N);
        drop(arena);
    }

    #[test]
    fn numa_arena_node_hint_oob_clamps_to_local() {
        // Passing a node hint that exceeds num_nodes routes to
        // the caller's current node instead of panicking.
        let arena = NumaArena::new(Some(2));
        let counter = Arc::new(AtomicU32::new(0));
        let c = Arc::clone(&counter);
        let job = StackJob::new(
            move |_stolen| { c.fetch_add(1, Ordering::SeqCst); },
            CoreLatch::new(),
        );
        unsafe {
            arena.submit(
                job.as_job_ref(2, NUMA_HINT_ANY, Variant::Faithful),
                Some(99), // out-of-range
            );
        }
        let deadline = Instant::now() + Duration::from_secs(5);
        while !job.latch.is_set() {
            if Instant::now() > deadline {
                panic!("OOB node-hint job did not complete");
            }
            std::thread::yield_now();
        }
        assert_eq!(counter.load(Ordering::SeqCst), 1);
        drop(arena);
    }

    #[test]
    fn numa_arena_try_run_one_returns_false_when_empty() {
        let arena = NumaArena::new(Some(2));
        let mut rng = 0x9E37_79B9_7F4A_7C15;
        assert!(!arena.try_run_one(&mut rng),
            "try_run_one must return false when no work pending");
        drop(arena);
    }
}
