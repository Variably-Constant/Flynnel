//! `LocalArena`: single-NUMA-node work-stealing thread pool.
//!
//! Each worker owns one adaptive K_inner=3 deque (KHL or Fcl
//! backing; LIFO owner end). Workers pop locally first, then
//! steal from the global [`crate::sched::injector::Injector`]
//! (used by external submitters), then steal randomly from peer
//! workers. When all
//! three fail, a worker parks on its [`crate::sched::sleep::Parker`].
//!
//!
//! ## Construction
//!
//! `LocalArena::new(n_workers)` spawns `n_workers` threads and
//! returns an `Arc<LocalArena>`. The arena owns the workers; when
//! dropped, it signals shutdown and joins.
//!
//! Each worker's [`Parker`] is constructed **inside** the worker
//! thread (so `thread::current()` captures the worker's own
//! handle) and exposed back to the arena via a `OnceLock` so
//! external submitters can call `unpark`.
//!
//! ## CPU pinning
//!
//! Each worker is pinned to a specific logical CPU via
//! `core_affinity::set_for_current()` on startup. Pinning keeps
//! the worker's L1d resident across job executions, which
//! matters for compute-bound multi-precision arithmetic that
//! reuses scratch buffers. Set `FLYNNEL_SCHED_PIN=off` to disable
//! pinning (OS-managed placement).
//!
//! CPU assignment: worker `i` pins to `core_ids[i % core_ids.len()]`
//! from `core_affinity::get_core_ids()`. For typical AMD-style
//! enumerations (CPUs 0..n_phys are first SMT siblings, n_phys..
//! 2*n_phys are second siblings), this spreads workers across
//! physical cores when worker_count == physical_cores.
//!
//! ## Composes
//!
//! [`crate::sched::join`]'s Local tier arm dispatches into the
//! global `LocalArena` when one is installed; without one,
//! Local-tier work runs inline.

use core::cell::Cell;
use core::ptr;
use core::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, OnceLock};
use std::thread::{self, JoinHandle};

use crate::sched::flynnel_ring::{FlynnelRing, PopResult, PushResult};
use crate::sched::injector::{Injector, InjectorSteal as Steal};

use crate::sched::adaptive_worker::{
    AdaptiveStash, AdaptiveStealer, AdaptiveSteal2, AdaptiveWorker,
    new_adaptive, steal_via_stash,
};
use crate::sched::deque_tier::{DequeTier, N_TIERS, peer_distance};
use crate::sched::job::JobRef;
use crate::sched::k_gating::KGating;
use crate::sched::sleep::Parker;

/// Per-worker mailbox capacity. Bounded because mailbox is a
/// LOCALITY HINT primitive, not a deque substitute - if the
/// mailbox fills, push_to_mailbox returns Err and the caller falls
/// back to a regular tiered push. 16 slots covers small recursive-
/// split bursts targeting the same SMT sibling.
const MAILBOX_CAPACITY: usize = 16;

// ---------------------------------------------------------------------------
// WorkerThread context: thread-local fast path for in-worker `join`.
//
// Rayon's central perf trick (rayon-core/src/registry.rs:411-423 +
// join/mod.rs:132-172): when `join(a, b)` is called from inside a
// worker thread, push the right-half job onto THAT worker's own local
// Chase-Lev deque, NOT into the shared Injector. The local deque is
// single-owner-writer LIFO; push/pop costs are a handful of nanoseconds
// because the only atomic involved is the owner-side index. The
// Injector is a global MPMC queue with retry loops; its push costs
// microseconds because of contention with thieves.
//
// We expose a `*const WorkerCtx` in a `thread_local!` so any code path
// reaching `arena::join` can answer "am I currently on a Flynnel
// scheduler worker?" without touching atomic state. The pointer is
// valid only on the owner thread; we register it during worker startup
// and clear it during worker shutdown.
// ---------------------------------------------------------------------------

/// Per-worker context exposed through a thread-local pointer. Owned by
/// the worker thread; lives on its stack inside `worker_loop`.
///
/// `worker` (the Chase-Lev owner half) is `!Send + !Sync`. We only
/// ever access fields on the owner thread (the same thread that
/// created the WorkerCtx), so the raw-pointer thread-local hand-off is
/// sound: every dereference happens from the thread that initialized
/// it.
pub(crate) struct WorkerCtx {
    /// Owner-side handles to this worker's per-coherence-tier
    /// adaptive K_inner=3 deques. Index 0 = [`DequeTier::SmtLocal`]
    /// (tightest cache), index 3 = [`DequeTier::Public`] (any peer
    /// may steal). Only the owner thread writes to these; thieves
    /// read via the corresponding `AdaptiveStealer` in
    /// [`Self::stealers`].
    ///
    /// Each [`AdaptiveWorker`] carries BOTH K_gating backings
    /// (KHL PerSlot + Fcl CounterOnly) and routes per-push via an
    /// AtomicU32 active tag - per-op overhead measured at 0 ns
    /// (the AtomicU32 Acquire load on x86 lowers to a plain MOV;
    /// the match-arm is statically known so LLVM lowers it to
    /// cmp+branch with the common arm as fall-through).
    /// [`AdaptiveWorker::migrate_to`] flips the active tag with a
    /// single Release-store, enabling runtime workload-shift
    /// adaptation without losing the kernel-bypass / shared-
    /// memory-atomic-only properties.
    pub(crate) workers: [AdaptiveWorker; N_TIERS],
    /// Worker index within its arena. Stable for the lifetime of the
    /// worker.
    #[allow(dead_code)]
    pub(crate) index: usize,
    /// Count of burst pushes since the last `flush_all` call. The
    /// burst path (`push_tier_burst`) skips per-push JEC wake; the
    /// flush emits ONE batched wake covering this count. Reset to
    /// 0 inside `flush_all`. Owner-private (Cell).
    pub(crate) burst_pushed: Cell<u32>,
    /// Per-peer, per-tier stealer matrix. Each entry is an
    /// `AdaptiveStealer` that observes the peer's active K_gating
    /// via a shared Arc<AtomicU32> tag and routes the steal to
    /// the matching backing.
    pub(crate) stealers: Vec<[AdaptiveStealer; N_TIERS]>,
    /// Thief-side adaptive batch stash. Holds K_inner=3 batch
    /// leftovers from EITHER KHL or Fcl backing (one stash slot
    /// per backing inside [`AdaptiveStash`]).
    pub(crate) steal_stash: core::cell::UnsafeCell<AdaptiveStash>,
    /// Cluster size (logical CPUs per CCX) used by
    /// [`crate::sched::deque_tier::peer_distance`] to label per-peer
    /// distances. Captured at WorkerCtx construction from
    /// [`crate::numa_topology`]'s `cluster_size_log2`.
    pub(crate) ccx_size: usize,
    /// Per-worker mailbox for owner-directed work hand-off (URD-style
    /// SIMC/MIMC routing). The owner reads on every `find_work`
    /// pass BEFORE the deque tiers - mailbox jobs are the most-
    /// locality-warm work this worker has been given.
    ///
    /// Any peer may write via [`Self::push_to_mailbox`] when the
    /// splitter knows this worker is cache-warm for similar work
    /// (e.g. a recursive-split parent handing the right-half to its
    /// SMT sibling). Bounded at [`MAILBOX_CAPACITY`]; on overflow
    /// the caller falls back to a regular tiered push.
    pub(crate) mailbox: Arc<FlynnelRing<JobRef>>,
    /// Mailbox handles for every peer worker (including self at
    /// `peer_mailboxes[index]`). Lets ANY worker push directly to
    /// any OTHER worker's mailbox without going through the deque
    /// substrate. Indexed by peer worker idx.
    pub(crate) peer_mailboxes: Vec<Arc<FlynnelRing<JobRef>>>,
    /// Shared injector for cross-arena / external submissions.
    pub(crate) injector: Arc<Injector<JobRef>>,
    /// Per-worker pseudo-random victim-selection state. `Cell` because
    /// only the owner thread mutates it.
    pub(crate) rng: Cell<u64>,
    /// Parker handles for every worker in this arena. When this
    /// worker's local deque transitions from empty to non-empty,
    /// `push` unparks one rotated peer so a truly-parked worker has
    /// a chance to wake up and steal. Without this, parked peers
    /// stay parked through the burst of recursive `join_in_worker`
    /// pushes (which never call `submit` and thus never touch the
    /// burst-wake path).
    pub(crate) parkers: Arc<Vec<Arc<OnceLock<Arc<Parker>>>>>,
    /// Rotated index for picking the next peer to unpark on a
    /// transition-to-non-empty push. `Cell` because only owner reads/writes.
    pub(crate) wake_rotor: Cell<usize>,
    /// Shared per-worker stat counters. Owner increments via
    /// `Relaxed` adds (cheap on x86); observer threads read via
    /// `Relaxed` loads. Read [`WorkerStats`] for the contract.
    pub(crate) stats: Arc<WorkerStats>,
    /// Parallel array of EVERY worker's stats (including self at
    /// `peer_stats[index]`). Lets the thief code path increment the
    /// VICTIM's `times_stolen_from` counter (`peer_stats[victim_idx]
    /// .times_stolen_from`). Cheap to carry: one Arc clone per worker
    /// at arena construction.
    pub(crate) peer_stats: Vec<Arc<WorkerStats>>,
    /// Shared JEC sleep coordinator (one instance per LocalArena).
    /// Used by push()'s wake path AND by the idle phase of the
    /// worker_loop.
    pub(crate) sleep: Arc<crate::sched::jec_sleep::Sleep>,
    /// Index of the LAST victim from which we successfully stole a
    /// job. `usize::MAX` sentinel = no previous steal. Adaptive
    /// victim selection probes this victim FIRST before falling
    /// back to the xorshift-random pick. Rationale: victims that
    /// recently had jobs to steal are more likely to have more
    /// (work tends to come in bursts from a single producer like a
    /// recursive split), AND the victim's deque buffer is already
    /// warm in our L2/L3 from the prior successful steal - the
    /// steal CAS lands without an L3 miss on the deque head index.
    /// Source: arxiv 2401.04494 (Adaptive Asynchronous Work-
    /// Stealing) - measures ~10% gain over uniform-random victim
    /// pick.
    pub(crate) last_victim: Cell<usize>,
    /// True if this WorkerCtx belongs to a pre-allocated external
    /// slot (one of `EXTERNAL_SLOT_COUNT` per LocalArena), false
    /// if it is a real spawned-thread worker.
    ///
    /// Critical for the join_in_worker push routing: when an
    /// EXTERNAL caller (sitting in a slot ctx) pushes a right-half
    /// during a join, the push goes to the INJECTOR instead of the
    /// slot's own deque. Reason: primaries' find_work random victim
    /// pick has very low probability of hitting any one slot (1 of
    /// n_workers + EXTERNAL_SLOT_COUNT = 1/36 typical), so right-
    /// halves on slot deques sit unstolen while the external caller
    /// races to pop its own deque after running left -- causing
    /// effective serialization. The injector is checked BY EVERY
    /// primary find_work call BEFORE the random pick (step 5 in
    /// find_work), guaranteeing pickup within one find_work round.
    ///
    /// Real workers always push to their OWN deque (LIFO, ~5ns,
    /// no atomic contention with peers). Only external slots
    /// re-route to the injector.
    pub(crate) is_external_slot: bool,
}

// NOTE: WorkerCtx deliberately carries no `PrivateLifoDeque`
// per-worker fast cache. Bench result on Zen+ 2026-05-17 with that
// cache wired in (`FLYNNEL_SCHED_PRIVATE_DEQUE=on`):
//   spmv_parallel/sched_1000  = 1.63 ms  (vs 290 µs Chase-Lev: 5.6x slower)
//   spmv_parallel/sched_10000 = 14.89 ms (vs 2.37 ms Chase-Lev: 6.3x slower)
// Root cause: the Mutex-based `PrivateLifoDeque` design pays a full
// lock+unlock per push and per pop. Chase-Lev pays one
// Acquire/Release atomic pair (~30ns) per operation. The Mutex path
// is ~200ns minimum - well above the entire push/pop budget.
// The PrivateLifoDeque type and tests live in
// src/sched/private_deque.rs as a documented negative result; a
// lock-free SPSC redesign is the path forward if the Chase-Lev
// atomic overhead ever becomes the bottleneck.

/// Per-worker scheduler statistics. Owned by the LocalArena AS
/// `Vec<Arc<WorkerStats>>` and shared with each worker's
/// `WorkerCtx::stats`. The observer thread
/// ([`crate::sched::split_observer`]) reads all the WorkerStats
/// every sampling interval, computes the steal rate, and updates
/// the split-budget multiplier accordingly.
///
/// Counters use [`core::sync::atomic::Ordering::Relaxed`] because
/// exact precision is unnecessary - the observer only needs trends
/// (steal rate going up or down). Cheap on x86 (single uop add).
///
/// `#[repr(align(128))]` pads every WorkerStats to a 128-byte
/// boundary - Intel L1 hardware prefetcher fetches in PAIRS of
/// 64-byte lines (so the effective false-sharing unit is 128, not
/// 64). Aligning to 128 guarantees that two adjacent workers'
/// counters never share a 128-byte block, eliminating false-sharing
/// invalidation traffic when many workers increment their stats
/// in parallel. Source: MICRO 2024 false-sharing detect/repair
///.
#[derive(Debug, Default)]
#[repr(align(128))]
pub struct WorkerStats {
    /// Jobs popped from the worker's own local Chase-Lev deque.
    pub local_pops: core::sync::atomic::AtomicU64,
    /// Jobs successfully stolen from a peer worker's deque.
    pub peer_steal_hits: core::sync::atomic::AtomicU64,
    /// Peer-probe rounds (in worker_loop step 3) that returned no
    /// work from any probed peer. High value -> low contention;
    /// workers are sitting idle. Low value -> high contention.
    pub peer_steal_misses: core::sync::atomic::AtomicU64,
    /// Times THIS worker's deque was stolen from by a peer. Incremented
    /// by the thief side at the steal site. Used by the default
    /// continuation-steal-driven lazy bisect in
    /// [`crate::sched::par_iter::for_each_chunk`] to detect actual
    /// steal pressure on its own deque per-call instead of pushing
    /// eagerly at every split.
    pub times_stolen_from: core::sync::atomic::AtomicU64,
    /// Single-push count (push that auto-flushes; join right-half
    /// pattern). Useful for distinguishing burst vs single workload
    /// shape via the [`Self::burst_ratio`] helper. Incremented by
    /// `WorkerCtx::try_push_tier` on an accepted push.
    pub single_pushes: core::sync::atomic::AtomicU64,
    /// Burst-push count (`push_tier_burst`). Counts each call,
    /// regardless of whether the underlying accumulator auto-
    /// flushed at n_items=3.
    pub burst_pushes: core::sync::atomic::AtomicU64,
    /// Pushes refused by a full deque (`try_push_tier`); the caller
    /// ran the job inline instead of waiting for a thief.
    pub push_refusals: core::sync::atomic::AtomicU64,
}

impl WorkerStats {
    /// Burst-vs-single ratio in `[0.0, 1.0]`. 1.0 means every push
    /// went through the burst path (cooperative_join_n_flat
    /// pattern); 0.0 means every push went through the auto-flush
    /// path (sched::join pattern). Returns 0.5 when no pushes
    /// have happened yet.
    pub fn burst_ratio(&self) -> f32 {
        use core::sync::atomic::Ordering;
        let b = self.burst_pushes.load(Ordering::Relaxed) as f32;
        let s = self.single_pushes.load(Ordering::Relaxed) as f32;
        let total = b + s;
        if total == 0.0 {
            0.5
        } else {
            b / total
        }
    }
}

impl WorkerCtx {
    /// Owner-directed mailbox push (URD-style SIMC/MIMC hand-off).
    /// ANY worker may call this against ANY peer's mailbox - the
    /// caller's role is "producer who knows the target worker is
    /// cache-warm for similar work." The target worker drains its
    /// mailbox FIRST in `find_work`, before its own deque tiers,
    /// because mailbox jobs are the most-locality-warm work it has
    /// been given.
    ///
    /// Returns `Ok(())` on successful push; returns `Err(job)` if
    /// the target's mailbox is full (caller falls back to a
    /// regular tiered push).
    ///
    /// Issues a wake notification through the JEC coordinator so
    /// a parked target worker has the chance to find the new work.
    ///
    /// Sets the process-global [`MAILBOX_EVER_USED`] flag on first
    /// successful push so subsequent `find_work` calls know to
    /// consult the mailbox. Without any writer, `find_work` skips
    /// the `mailbox.pop()` call entirely (one AtomicBool Acquire-
    /// load replaces a FlynnelRing pop per find_work poll).
    pub(crate) fn push_to_mailbox(
        &self,
        target_idx: usize,
        job: JobRef,
    ) -> Result<(), JobRef> {
        if target_idx >= self.peer_mailboxes.len() {
            return Err(job);
        }
        let was_empty = self.peer_mailboxes[target_idx].is_empty();
        match self.peer_mailboxes[target_idx].push(job) {
            PushResult::Ok => {}
            PushResult::Full(j) => return Err(j),
        }
        // First successful mailbox writer in the process arms the
        // global flag. Subsequent calls re-store true (cheap on the
        // owner-uncontended line). Workers' find_work then knows to
        // consult the mailbox; without this flag the >99% case where
        // no one uses mailbox routing pays a wasted FlynnelRing pop
        // per find_work poll.
        MAILBOX_EVER_USED.store(true, Ordering::Release);
        // Same wake protocol as a deque push: notify JEC so a
        // parked target sees the work on its next loop iter.
        if DISPATCH_USE_JEC_WAKE.with(|c| c.get()) {
            self.sleep.new_internal_jobs(1, was_empty);
        }
        Ok(())
    }

    /// Burst-mode push: buffer this job into the owner accumulator
    /// without auto-flushing. Caller MUST follow up with [`Self::flush_all`]
    /// before any wait-loop that expects thieves to see the pushed
    /// jobs. Routes to the default tier. Use this in producer-fast
    /// burst sites (cooperative_join_n_flat fan-out, for_each_chunk
    /// slicing) to unlock the K_inner=3 amortization - 3 burst
    /// pushes pack into one cache-line slot, delivering 3 jobs per
    /// coherence transfer when a thief steals.
    #[inline]
    pub(crate) fn push_burst(&self, job: JobRef) {
        self.push_tier_burst(job, DequeTier::default());
    }

    /// Tiered burst push. Caller MUST be the owner thread; MUST
    /// follow up with [`Self::flush_all`] before wait-loop entry.
    /// Increments the burst counter so flush_all can fire ONE JEC
    /// wake covering the entire burst (per-push wake is skipped).
    /// Also increments the WorkerStats burst-vs-single profile
    /// counter for observability / shape-aware downstream
    /// decisions.
    #[inline]
    pub(crate) fn push_tier_burst(&self, job: JobRef, tier: DequeTier) {
        self.workers[tier.idx()].push_burst(job);
        self.burst_pushed.set(self.burst_pushed.get().saturating_add(1));
        self.stats
            .burst_pushes
            .fetch_add(1, core::sync::atomic::Ordering::Relaxed);
    }

    /// Flush every tier's accumulator so any buffered burst jobs
    /// become visible to thieves. Issues a single batched JEC
    /// wake notification covering ALL burst pushes since the last
    /// flush (including those that auto-flushed inside the
    /// accumulator at n_items=3). Call after a producer-fast
    /// burst loop, BEFORE entering the wait loop.
    #[inline]
    pub(crate) fn flush_all(&self) {
        for tier in DequeTier::all() {
            self.workers[tier.idx()].flush();
        }
        let n = self.burst_pushed.get();
        self.burst_pushed.set(0);
        if n > 0 && DISPATCH_USE_JEC_WAKE.with(|c| c.get()) {
            // Broadcast on the assumption that the worker pool was
            // empty when this burst started (typical for the
            // cooperative_join_n_flat dispatch pattern; siblings
            // wake on the transition).
            self.sleep.new_internal_jobs(n, true);
        }
    }

    /// Single push to the default ([`DequeTier::Public`]) deque;
    /// see [`Self::try_push_tier`].
    #[inline]
    pub(crate) fn try_push(&self, job: JobRef) -> Result<(), JobRef> {
        self.try_push_tier(job, DequeTier::default())
    }

    /// Tiered single push. Caller MUST be the owner thread. Routes
    /// the job to the named [`DequeTier`]'s deque; peers stealing
    /// at distance `d` may take it only if `tier >= d` per the
    /// asymmetric steal discipline in
    /// [`crate::sched::deque_tier::thief_may_steal`].
    ///
    /// Refuses with `Err(job)` instead of waiting when the deque is
    /// full: 256 slots of this worker's own right-halves are already
    /// waiting for thieves, and waiting for one while every thief
    /// may be an owner in the same state is the circular wait behind
    /// the 65,536-item hang. The caller runs a refused job inline.
    #[inline]
    pub(crate) fn try_push_tier(&self, job: JobRef, tier: DequeTier) -> Result<(), JobRef> {
        let any_was_empty = self.workers.iter().all(|w| w.is_empty());
        match self.workers[tier.idx()].try_push(job) {
            Ok(()) => {}
            Err(job) => {
                self.stats
                    .push_refusals
                    .fetch_add(1, core::sync::atomic::Ordering::Relaxed);
                return Err(job);
            }
        }
        self.stats
            .single_pushes
            .fetch_add(1, core::sync::atomic::Ordering::Relaxed);
        // Hybrid JEC wake. The per-dispatch flag
        // `DISPATCH_USE_JEC_WAKE` is set by `for_each_chunk` based
        // on estimated total work: large dispatches (>= 200us) get
        // the full JEC wake-cascade benefit; small dispatches skip
        // the JEC counter dance and rely on workers' spin-loop
        // polling. Safe because `ROUNDS_UNTIL_SLEEPING` is set
        // high enough in `jec_sleep.rs` that workers stay in spin
        // phase across typical inter-dispatch gaps.
        if DISPATCH_USE_JEC_WAKE.with(|c| c.get()) {
            self.sleep.new_internal_jobs(1, any_was_empty);
        }
        Ok(())
    }

    /// Broadcast unpark to every peer parker. Caller MUST be the
    /// owner thread. The maximal-fanout counterpart of the
    /// single-peer rotor wake (`wake_one_peer` below): a
    /// single-wake cascade measures 5 of 16 workers dormant
    /// through a Heavy/100k dispatch on Zen+ (trace data,
    /// 2026-06-04), with the 11 that wake spreading over a
    /// 25us..2238us window (89x spread); broadcast on the
    /// empty->non-empty transition closes that gap by issuing
    /// every unpark in parallel from the producer side.
    ///
    /// Cost: N-1 unpark syscalls (Windows: SetEvent ~1us each,
    /// Linux: futex wake ~0.5us). For n=16 that is ~15us of
    /// producer cost on a single transition.
    ///
    /// Already-running peers ignore the unpark (Parker semantics
    /// drop spurious unparks once consumed); only truly-parked
    /// peers actually wake up and pay any cost.
    ///
    /// Not wired into `push` (that path wakes through the JEC
    /// coordinator; the broadcast measured no improvement on
    /// Heavy/100k Zen+ there). Kept as a documented alternative
    /// wake primitive for sites that want maximal fanout, e.g.
    /// cross-arena leader-driven balancing.
    #[inline]
    #[allow(dead_code)]
    fn wake_all_peers(&self) {
        let n = self.parkers.len();
        if n <= 1 {
            return;
        }
        for (i, slot) in self.parkers.iter().enumerate() {
            if i == self.index {
                continue;
            }
            if let Some(p) = slot.get() {
                p.unpark();
            }
        }
    }

    /// Pick one peer parker by rotor and call unpark. Single-peer
    /// cascade pattern. Retained as a documented alternative wake
    /// primitive; NOT wired into [`Self::push_tier`] (that path
    /// uses `self.sleep.new_internal_jobs(...)` through the JEC
    /// coordinator instead of touching parkers directly).
    #[inline]
    #[allow(dead_code)]
    fn wake_one_peer(&self) {
        let n = self.parkers.len();
        if n <= 1 {
            return;
        }
        let mut r = self.wake_rotor.get();
        r = (r + 1) % n;
        if r == self.index {
            r = (r + 1) % n;
        }
        self.wake_rotor.set(r);
        if let Some(p) = self.parkers[r].get() {
            p.unpark();
        }
    }

    /// Find one job. Probe order (each stage returns immediately on
    /// success):
    ///
    /// 1. Public-tier LIFO pop (the default tier for unhinted pushes,
    ///    checked first as a fast path so the common case skips the
    ///    empty-tier walk below).
    /// 2. Owner-private steal_stash drain (K_inner=3 batch leftovers
    ///    from a prior successful peer steal - locality-warm).
    /// 3. Mailbox pop (gated on the process-global
    ///    `MAILBOX_EVER_USED` flag - one Acquire-load skips the
    ///    FlynnelRing::pop entirely when no caller has ever opted
    ///    into mailbox routing).
    /// 4. Narrower-tier LIFO pops (SmtLocal -> IntraCcx -> CrossCcx)
    ///    for work that producers routed via `push_tier` with a
    ///    locality hint.
    /// 5. Injector steal (external / cross-arena submissions).
    /// 6. Adaptive victim probe: try `self.last_victim` first across
    ///    every tier the steal discipline allows for that distance
    ///    (arxiv 2401.04494; ~10% gain vs uniform-random).
    /// 7. xorshift-random peer steal, walked across the allowed tiers.
    ///
    /// Returns `None` when all paths are empty.
    ///
    /// On a successful steal (injector or peer), the job's captured-
    /// state pointer is prefetched into L2 before returning. The
    /// caller's next access (`job.execute()` -> `(execute_fn)(pointer)`)
    /// touches that pointer, so warming it cuts the steal-to-execute
    /// latency by hiding the L3/RAM fetch. Source: arxiv 2009.00202
    /// (Helper Without Threads).
    pub(crate) fn find_work(&self) -> Option<JobRef> {
        // (FAST PATH) Public tier first. `DequeTier::default()` is
        // Public, so raw `join` and par-iter helpers without a
        // deque_tier_hint route every push to the Public deque. A
        // distance-ordered walk (SmtLocal -> IntraCcx -> CrossCcx
        // before Public) pays 3 wasted Chase-Lev pops (~60 ns) per
        // call on that common case before hitting the deque that
        // has work; for a blur worker running ~1000 find_work
        // calls per iter that is ~60us of wasted atomic loads on a
        // 700us workload, most of the steady-state plumbing gap vs
        // rayon's single-pop hot path.
        //
        // The Public-first check is purely additive: SMT-tier-
        // pushed work still routes correctly because the tier walk
        // below still checks SmtLocal / IntraCcx / CrossCcx. The
        // only cost paid by SMT-pushed workloads is one extra
        // empty-Public check, ~15 ns.
        if let Some(job) = self.workers[DequeTier::Public.idx()].pop() {
            return Some(job);
        }
        // (-1) Steal-stash drain: a recent successful peer steal
        //      may have left 1-2 extra items here. Drain them
        //      first - they are locality-warm (just stolen) and
        //      paid the coherence transfer already.
        //
        // SAFETY: owner-private steal_stash; single-threaded read.
        {
            let stash = unsafe { &mut *self.steal_stash.get() };
            if let Some(job) = stash.drain_one() {
                return Some(job);
            }
        }
        // (0) Mailbox: owner-directed hand-offs from peers that
        //     knew this worker was cache-warm for the work.
        //     Gated on the process-global MAILBOX_EVER_USED flag:
        //     mailbox routing is opt-in via plan.use_mailbox_routing
        //     (defaults false per the empirical regression noted in
        //     plan.rs). When no caller has ever opted in, skip the
        //     FlynnelRing::pop call entirely via one Acquire-load.
        //     Saves ~0.5% SELF cycles measured on VM Zen3 v5 flame.
        if MAILBOX_EVER_USED.load(Ordering::Acquire)
            && let PopResult::Ok(job) = self.mailbox.pop()
        {
            return Some(job);
        }
        // (1) Tighter tiers (SmtLocal -> IntraCcx -> CrossCcx) for
        //     work that producers deliberately routed via
        //     push_tier() with a narrower locality hint. Public was
        //     already checked above.
        for tier in [DequeTier::SmtLocal, DequeTier::IntraCcx, DequeTier::CrossCcx] {
            if let Some(job) = self.workers[tier.idx()].pop() {
                return Some(job);
            }
        }
        if let Steal::Success(job) = self.injector.steal() {
            // SAFETY: best-effort prefetch over a 64-byte window
            // around the captured-state pointer; the hint never
            // traps even on a dangling pointer.
            crate::sched::prefetch::prefetch_into_l2_inline(unsafe {
                core::slice::from_raw_parts(job.data_ptr() as *const u8, 64)
            });
            return Some(job);
        }
        let n = self.stealers.len();
        if n < 2 {
            return None;
        }

        // (2) Adaptive victim probe: try the last successful victim
        // first across all allowed tiers. The deque's head index is
        // already warm in our L2 from the previous steal, and
        // recursive-split producers tend to keep emitting work to the
        // same deque for several bursts in a row.
        let last = self.last_victim.get();
        if last != usize::MAX
            && last != self.index
            && last < n
            && let Some(job) = self.steal_from_peer_tiered(last)
        {
            return Some(job);
        }

        // (3) Random victim, walked across allowed tiers.
        let mut x = self.rng.get();
        if x == 0 {
            x = 0x9E37_79B9_7F4A_7C15;
        }
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.rng.set(x);
        let mut victim = (x as usize) % n;
        if victim == self.index {
            victim = (victim + 1) % n;
        }
        self.steal_from_peer_tiered(victim)
    }

    /// Prefetch the Stealer cache line for the most-recent
    /// successful victim, so the NEXT `find_work` call's last-
    /// victim probe hits a warm line. Called from the successful-
    /// steal path AFTER the captured-state prefetch but BEFORE the
    /// caller's `execute()`. The execute body's runtime overlaps
    /// the prefetch's coherence fill; by the time control returns
    /// to find_work, the Stealer line is already in L1d/L2.
    ///
    /// Targets the SmtLocal tier of the last victim - that is the
    /// tier the next find_work pass probes first under the
    /// asymmetric steal discipline.
    ///
    /// No-op when `last_victim` is unset (first call) or on
    /// non-x86_64 targets (where `_mm_prefetch` is not exposed
    /// through stable intrinsics).
    #[inline]
    pub(crate) fn prefetch_last_victim_stealer(&self) {
        let last = self.last_victim.get();
        if last == usize::MAX || last >= self.stealers.len() {
            return;
        }
        let stealer = &self.stealers[last][DequeTier::SmtLocal.idx()];
        let addr = stealer as *const _ as *const u8;
        #[cfg(target_arch = "x86_64")]
        {
            // SAFETY: _mm_prefetch is a no-side-effect hint that
            // accepts any pointer value without fault. The stealer
            // is held by `self` (a stack reference inside the
            // worker thread) so the cache line is at minimum a
            // valid stack-resident address.
            unsafe {
                std::arch::x86_64::_mm_prefetch(
                    addr as *const i8,
                    std::arch::x86_64::_MM_HINT_T0,
                );
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            // Reference the pointer so the optimizer does not
            // dead-code-eliminate the address computation; the
            // hint itself has no stable cross-platform intrinsic.
            std::hint::black_box(addr);
        }
    }

    /// Probe peer `peer` across all tiers the steal discipline
    /// allows for our distance to them: tier_idx >= peer_distance.
    /// Returns the first successful steal; prefetches captured state;
    /// updates `last_victim` + peer's `times_stolen_from` counter.
    fn steal_from_peer_tiered(&self, peer: usize) -> Option<JobRef> {
        if peer == usize::MAX || peer == self.index || peer >= self.stealers.len() {
            return None;
        }
        // SAFETY: owner-private steal_stash; the find_work caller
        // drains the stash before reaching peer steal, so the
        // stash is guaranteed empty here (the debug_assert inside
        // steal_via_stash enforces this).
        let stash = unsafe { &mut *self.steal_stash.get() };
        let distance = peer_distance(self.index, peer, self.ccx_size);
        for tier_idx in distance.idx()..N_TIERS {
            match steal_via_stash(&self.stealers[peer][tier_idx], stash) {
                AdaptiveSteal2::Success(job) => {
                    // SAFETY: prefetch hint is no-side-effect even
                    // if the 64-byte read window extends past the
                    // actual captured-state size.
                    crate::sched::prefetch::prefetch_into_l2_inline(unsafe {
                        core::slice::from_raw_parts(job.data_ptr() as *const u8, 64)
                    });
                    if peer < self.peer_stats.len() {
                        self.peer_stats[peer]
                            .times_stolen_from
                            .fetch_add(1, core::sync::atomic::Ordering::Relaxed);
                    }
                    self.last_victim.set(peer);
                    // Prefetch the same victim's Stealer for the
                    // next find_work pass.
                    self.prefetch_last_victim_stealer();
                    return Some(job);
                }
                AdaptiveSteal2::Empty | AdaptiveSteal2::Retry => continue,
            }
        }
        None
    }
}

thread_local! {
    /// `*const WorkerCtx` pointing at the current worker's stack-
    /// resident context. Null when the running thread is not a Flynnel
    /// worker (most external callers, including Criterion's
    /// main thread, criterion's bench harness threads, application
    /// code, etc.).
    static WORKER_CTX: Cell<*const WorkerCtx> = const { Cell::new(ptr::null()) };
}

/// Return the current thread's `WorkerCtx` pointer, or null if the
/// thread is not a Flynnel worker. Reading the thread-local costs ~1
/// ns (one mov).
#[inline]
pub(crate) fn current_worker_ctx() -> *const WorkerCtx {
    WORKER_CTX.with(|c| c.get())
}

/// Register `ctx` as the current thread's worker context. Called once
/// during `worker_loop` startup. The pointer remains valid as long as
/// `worker_loop` does not return.
///
/// # Safety
///
/// Caller must ensure `ctx` outlives every observable use of
/// `current_worker_ctx()` on this thread. In practice that means `ctx`
/// is a `&WorkerCtx` borrowed from a stack frame that does not return
/// until `clear_current_worker_ctx()` is called.
unsafe fn set_current_worker_ctx(ctx: *const WorkerCtx) {
    WORKER_CTX.with(|c| {
        debug_assert!(c.get().is_null(),
            "worker ctx already set on this thread; double-registration");
        c.set(ctx);
    });
}

/// Clear the current thread's worker context. Called from
/// `worker_loop` before its stack frame returns so a later spurious
/// `current_worker_ctx()` read returns null instead of dangling.
fn clear_current_worker_ctx() {
    WORKER_CTX.with(|c| c.set(ptr::null()));
}

/// `true` when CPU pinning should be SKIPPED. Default is `true`
/// (no pinning) because the rayon-crossover benches showed pinning
/// hurts wall-clock perf when the bench machine has any
/// concurrent system load: pinned workers cannot migrate to an
/// idle CPU and must wait for their assigned core to become free.
/// Unpinned workers let the OS scheduler place them on whichever
/// CPU has slack at the moment - on Zen+'s 8c/16t this means SMT
/// siblings of busy cores get used productively rather than
/// sitting idle. The cache-locality argument for pinning (worker
/// L1d stays resident across job executions) holds on dedicated
/// hardware with zero competing load; on any real machine it
/// loses.
///
/// Override via `FLYNNEL_SCHED_PIN=on` (or `1`, `true`) to force
/// pinning back on - useful for dedicated bench rigs or
/// strict-NUMA experiments. `FLYNNEL_SCHED_PIN=off` is also
/// recognized for explicit off (idempotent with the default).
fn pin_disabled_env() -> bool {
    static CACHE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| {
        match std::env::var("FLYNNEL_SCHED_PIN") {
            Ok(v) => {
                let v = v.to_ascii_lowercase();
                if v == "on" || v == "1" || v == "true" {
                    false // pinning enabled
                } else {
                    true // any other value disables pinning
                }
            }
            // Default: pinning DISABLED. See doc comment.
            Err(_) => true,
        }
    })
}

/// Number of `thread::yield_now()` spin rounds a Local-tier worker
/// runs before parking: the per-worker spin floor before actually
/// entering `thread::park()`.
///
/// Tuning evidence (rayon crossover bench + trace data): 64 gives
/// a ~64us hot window. On Heavy/100k Zen+ the wake spread measures
/// 25..2238us across 16 workers; 1024 rounds tightens cold-start
/// to 33-42us but regresses criterion wall-clock 5.82 -> 6.66ms
/// because spinning workers compete with productive workers for
/// CPU cycles on a 16-thread host. 8 rounds is below the park-
/// amortization floor. The hot-window vs CPU-share trade-off
/// optimum sits between those bounds on this silicon class.
pub(crate) const LOCAL_SPIN_ROUNDS: u32 = 64;

/// Adaptive peer-probe cap. For small pools (n <= PROBE_FULL_CUTOFF)
/// walk all n-1 peers per loop iteration: cache traffic is bounded
/// (7 atomic reads * 8 workers = 56 reads/cycle), and missing a
/// busy peer means losing a leaf's worth of work which dominates
/// the savings from probing fewer peers. For large pools
/// (n > PROBE_FULL_CUTOFF) clamp probes to PROBE_LARGE: at
/// n=16, 15 atomic reads * 16 workers = ~240 reads/cycle of
/// cache-line ping-pong that swamps productive work, so we accept
/// the chance of missing a busy peer (wake-on-push catches it via
/// explicit unparks) in exchange for bounded probe cost.
///
/// Threshold of 8 is the Zen+ physical-core count -- that's the
/// typical compute pool size; above that we're in SMT territory
/// where peer contention starts to dominate.
const PROBE_FULL_CUTOFF: usize = 8;
const PROBE_LARGE: usize = 4;

/// Work-stealing arena for the Local tier. Spawns N worker
/// threads that loop over local-pop / injector-steal / random-
/// peer-steal / park.
///
/// External waiters (threads waiting on a child latch via
/// `local_join`) participate in work-stealing through
/// [`Self::try_run_one`] - the same loop body the workers use
/// minus the local-pop step. This means the parent of every
/// fork-join is doing real work during the wait, not idling.
pub struct LocalArena {
    /// Global queue for external submissions. Workers steal from
    /// this when their own deque is empty.
    injector: Arc<Injector<JobRef>>,
    /// Per-worker, per-tier stealer matrix. Outer Vec indexed by
    /// worker idx; inner array indexed by [`DequeTier`]. External
    /// waiters use only [`DequeTier::Public`] (widest tier).
    ///
    /// Stealers are adaptive (KHL + Fcl backings; AtomicU32 active
    /// tag); external callers that need a single job per steal
    /// keep an `AdaptiveStash` and call `steal_via_stash` from
    /// [`crate::sched::adaptive_worker`].
    stealers: Vec<[AdaptiveStealer; N_TIERS]>,
    /// Per-worker park handles. Filled by each worker thread on
    /// startup via OnceLock; external submitters unpark via these.
    parkers: Vec<Arc<OnceLock<Arc<Parker>>>>,
    /// Per-worker stat counters. Shared with each worker via
    /// `Arc<WorkerStats>`. The observer thread reads these to
    /// compute the steal rate.
    stats: Vec<Arc<WorkerStats>>,
    /// Worker thread join handles. Wrapped in `Option` so `Drop`
    /// can take them.
    workers: Vec<Option<JoinHandle<()>>>,
    /// Number of "primary" workers - the always-active head of
    /// the worker slice. Workers `[0..primary_count)` run unconditionally;
    /// workers `[primary_count..workers.len())` are SMT siblings
    /// that park unless `smt_requests > 0`.
    primary_count: usize,
    /// SMT request counter. Workers with idx >= primary_count
    /// park whenever this is 0 and join the work-stealing loop
    /// when this is > 0. Per-call `with_smt` raises it before
    /// submit and lowers it after the latch is set. Reference-
    /// counted so nested `with_smt` calls compose.
    smt_requests: Arc<AtomicU32>,
    /// Shared shutdown flag readable BY EVERY WORKER regardless
    /// of whether its individual `Parker` has been initialized
    /// yet. Set by `Drop` to ensure workers caught between spawn
    /// and parker-slot-set still observe the shutdown signal and
    /// exit cleanly.
    shutdown_flag: Arc<AtomicBool>,
    /// JEC (Jobs Event Counter) sleep coordinator. Drives the
    /// idle-search wake protocol: producers (`push` and `submit`)
    /// call `Sleep::new_internal_jobs` to wake sleepers; workers
    /// transition yield -> sleepy -> sleeping via
    /// `Sleep::no_work_found`. Awake-but-idle accounting lets the
    /// producer skip the wake syscall when enough workers are
    /// already spinning.
    pub(crate) sleep: Arc<crate::sched::jec_sleep::Sleep>,
    /// Per-worker per-tier active-K_gating `Arc<AtomicU32>` tags,
    /// shared with each worker's AdaptiveWorker via Arc::clone.
    /// Indexed [worker_idx][tier_idx]. Used by
    /// [`Self::migrate_all_workers_k_gating`] to flip every
    /// AdaptiveWorker's active backing with a single Release-store
    /// pass.
    pub(crate) k_gating_tags: Vec<[Arc<core::sync::atomic::AtomicU32>; N_TIERS]>,
    /// External-worker slot pool. Pre-allocated at arena
    /// construction; external callers claim a slot, become a
    /// temporary worker for the duration of one external_dispatch,
    /// and release the slot on Drop. Each slot's stealers are
    /// registered in `self.stealers` so peers can steal external-
    /// pushed work; without that registration, external-pushed
    /// work is invisible to the pool and the dispatch deadlocks.
    pub(crate) external_slots: Vec<Arc<ExternalSlot>>,
}


thread_local! {
    /// Per-dispatch hint controlled by `for_each_chunk`: when
    /// `true`, `WorkerCtx::push` issues the full JEC wake
    /// notification (`sleep.new_internal_jobs`); when `false`,
    /// the push relies on workers finding the new job via their
    /// spin-loop polling of own deque + injector (cheap, but
    /// only safe when workers are guaranteed not condvar-sleeping
    /// - which is guaranteed by the bumped
    /// `ROUNDS_UNTIL_SLEEPING` in `jec_sleep.rs`). Default `true`
    /// so any caller that does not set the scope guard pays
    /// the full JEC cost (correctness over performance).
    pub(crate) static DISPATCH_USE_JEC_WAKE: core::cell::Cell<bool>
        = const { core::cell::Cell::new(true) };
}

/// Process-global flag set to `true` the first time any caller
/// successfully posts to a worker's mailbox via
/// [`WorkerCtx::push_to_mailbox`]. When `false`, every worker's
/// [`WorkerCtx::find_work`] skips the `self.mailbox.pop()` call --
/// the FlynnelRing pop has a fast-empty path but still costs a
/// function call + ring-state check (~0.5% SELF measured on VM
/// Zen3 v5 flame). Replacing it with one Acquire-load on the
/// hot path saves those cycles on the >99% of dispatches that
/// never opt into mailbox routing (`plan.use_mailbox_routing`
/// defaults `false` per the empirical regression noted in
/// `dispatch_profile.rs::use_mailbox_routing`).
///
/// One-way flag: once set, stays set until process exit. Callers
/// that mix mailbox-routed and non-mailbox dispatches still get
/// correct mailbox draining for the duration of the process.
pub(crate) static MAILBOX_EVER_USED: AtomicBool = AtomicBool::new(false);

/// RAII guard that sets `DISPATCH_USE_JEC_WAKE` on construction
/// and restores the prior value on drop. Used by `for_each_chunk`
/// to scope the hybrid JEC-vs-skip decision to one dispatch
/// without leaking the flag to subsequent calls.
pub(crate) struct DispatchScope {
    prev: bool,
}

impl DispatchScope {
    /// Construct a DispatchScope that unconditionally sets the per-
    /// thread `DISPATCH_USE_JEC_WAKE` to `use_jec_wake`. Use this
    /// when you KNOW the value differs from the current cell value
    /// (rare). Most callers should prefer [`Self::new_if_change`].
    ///
    /// The unconditional-set variant: for callers that explicitly
    /// want set semantics without the read-and-compare
    /// optimization. No production site uses it.
    #[inline]
    #[allow(dead_code)]
    pub(crate) fn new(use_jec_wake: bool) -> Self {
        let prev = DISPATCH_USE_JEC_WAKE.with(|c| {
            let p = c.get();
            c.set(use_jec_wake);
            p
        });
        Self { prev }
    }

    /// Construct a DispatchScope only if `use_jec_wake` differs from
    /// the current cell value; otherwise return `None`. Saves the
    /// per-call TLS write on the common path AND the Drop-side
    /// TLS write+read on scope exit.
    ///
    /// The DISPATCH_USE_JEC_WAKE cell defaults to `true`. The
    /// production caller (`par_iter::for_each_chunk`) sets the flag
    /// based on estimated workload size: `true` for >= 200us (the
    /// majority of bench cells), `false` for smaller. When the
    /// cell already matches the caller's desired value (the common
    /// case where consecutive calls have the same workload class),
    /// the scope becomes a no-op.
    ///
    /// Per-call cost: 1 TLS read + 1 compare. If no change needed,
    /// returns `None` immediately. Saves 1 TLS write (new) +
    /// 1 TLS read + 1 TLS write (drop) vs the unconditional `new`.
    #[inline]
    pub(crate) fn new_if_change(use_jec_wake: bool) -> Option<Self> {
        DISPATCH_USE_JEC_WAKE.with(|c| {
            let prev = c.get();
            if prev == use_jec_wake {
                None
            } else {
                c.set(use_jec_wake);
                Some(Self { prev })
            }
        })
    }
}

impl Drop for DispatchScope {
    #[inline]
    fn drop(&mut self) {
        DISPATCH_USE_JEC_WAKE.with(|c| c.set(self.prev));
    }
}

impl std::fmt::Debug for LocalArena {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LocalArena")
            .field("n_workers", &self.workers.len())
            .field("injector_len", &self.injector.len())
            .finish()
    }
}

impl LocalArena {
    /// Spawn an arena with `n_workers` worker threads pinned to
    /// distinct logical CPUs from `core_affinity::get_core_ids()`.
    /// `n_workers` is clamped to >= 1.
    ///
    /// Pinning: each worker calls `core_affinity::set_for_current`
    /// to bind to `core_ids[idx % core_ids.len()]`. Set
    /// `FLYNNEL_SCHED_PIN=off` to disable.
    pub fn new(n_workers: usize) -> Arc<Self> {
        let core_ids: Option<Vec<core_affinity::CoreId>> = if pin_disabled_env() {
            None
        } else {
            core_affinity::get_core_ids()
        };
        Self::with_cpu_set(n_workers, core_ids)
    }

    /// Spawn an arena with `n_workers` primary worker threads
    /// (always active) and no SMT-sibling extension. Equivalent
    /// to `with_smt_extension(n_workers, 0, cpu_set)`. Used by
    /// callers that want exactly `n_workers` threads.
    pub fn with_cpu_set(
        n_workers: usize,
        cpu_set: Option<Vec<core_affinity::CoreId>>,
    ) -> Arc<Self> {
        Self::with_smt_extension(n_workers, 0, cpu_set)
    }

    /// Spawn an arena with `primary_count` always-active worker
    /// threads + `smt_extension` SMT-sibling worker threads. The
    /// first `primary_count` workers run unconditionally; the
    /// remaining `smt_extension` workers park whenever the
    /// arena's SMT request counter is 0, and join the work-
    /// stealing loop while it is > 0.
    ///
    /// Pattern for Zen+ R7 2700 (8c/16t): `primary_count = 8`,
    /// `smt_extension = 8`. Default workload runs on 8 physical
    /// workers; `JobPlan::with_smt()` raises the counter and the
    /// 8 SMT siblings join.
    ///
    /// `cpu_set` covers BOTH primaries and siblings. With pinning
    /// enabled, the first `primary_count` CoreIds pin primaries
    /// to physical cores and the remaining CoreIds pin siblings
    /// to SMT siblings. With pinning disabled (the default), the
    /// OS scheduler distributes both groups.
    pub fn with_smt_extension(
        primary_count: usize,
        smt_extension: usize,
        cpu_set: Option<Vec<core_affinity::CoreId>>,
    ) -> Arc<Self> {
        let primary = primary_count.max(1);
        let smt = smt_extension;
        let n = primary + smt;
        let injector: Arc<Injector<JobRef>> = Arc::new(Injector::new());

        // Build per-worker per-tier (KhlWorker, KhlStealer) pairs.
        // Each worker has [N_TIERS] KHL-backed deques; each peer
        // in the arena holds [N_TIERS] stealers per worker.
        //
        // KHL_SLOT_CAPACITY sets each tier's ring size (in SLOTS,
        // each holding up to 3 jobs). Sized large enough to absorb
        // producer bursts without triggering the spin-on-publish
        // back-pressure path; small enough to keep the buffer's
        // cold-page footprint bounded. 256 slots = 768 jobs per
        // tier per worker, ~16KB per tier per worker.
        const ADAPTIVE_SLOT_CAPACITY: usize = 256;
        // Initial gating: KGating::Auto resolves to the host-
        // calibrated winner. On Zen+ R7 2700 (the bench host),
        // calibration picks PerSlot (KHL active).
        let initial_gating = KGating::Auto;
        let mut workers_per_idx: Vec<[Option<AdaptiveWorker>; N_TIERS]> =
            (0..n).map(|_| [const { None }; N_TIERS]).collect();
        let mut stealers: Vec<[AdaptiveStealer; N_TIERS]> = Vec::with_capacity(n);
        let mut k_gating_tags: Vec<[Arc<core::sync::atomic::AtomicU32>; N_TIERS]>
            = Vec::with_capacity(n);
        for slot in workers_per_idx.iter_mut() {
            let (w0, s0) = new_adaptive(ADAPTIVE_SLOT_CAPACITY, initial_gating);
            let (w1, s1) = new_adaptive(ADAPTIVE_SLOT_CAPACITY, initial_gating);
            let (w2, s2) = new_adaptive(ADAPTIVE_SLOT_CAPACITY, initial_gating);
            let (w3, s3) = new_adaptive(ADAPTIVE_SLOT_CAPACITY, initial_gating);
            let tags = [w0.active_tag(), w1.active_tag(), w2.active_tag(), w3.active_tag()];
            slot[0] = Some(w0);
            slot[1] = Some(w1);
            slot[2] = Some(w2);
            slot[3] = Some(w3);
            stealers.push([s0, s1, s2, s3]);
            k_gating_tags.push(tags);
        }

        // ------- External-slot pool construction -------
        // Pre-allocate EXTERNAL_SLOT_COUNT external slots. Each slot
        // owns 4 AdaptiveWorker deques (so an external caller can
        // push right-halves into its own deque) AND has its stealers
        // registered in arena.stealers (so peers can see + steal
        // external-pushed work). This is the slot-pool design the
        // rayon Registry::in_worker pattern is built on.
        //
        // Slots are indexed [n .. n+EXTERNAL_SLOT_COUNT) in
        // arena.stealers / parkers / stats / mailboxes. Workers
        // iterate over the full extended vec for victim selection;
        // slot indices appear as just another peer.
        let mut slot_workers_per_idx: Vec<[Option<AdaptiveWorker>; N_TIERS]> =
            (0..EXTERNAL_SLOT_COUNT)
                .map(|_| [const { None }; N_TIERS]).collect();
        for slot in slot_workers_per_idx.iter_mut() {
            let (w0, s0) = new_adaptive(EXTERNAL_SLOT_DEQUE_CAPACITY, initial_gating);
            let (w1, s1) = new_adaptive(EXTERNAL_SLOT_DEQUE_CAPACITY, initial_gating);
            let (w2, s2) = new_adaptive(EXTERNAL_SLOT_DEQUE_CAPACITY, initial_gating);
            let (w3, s3) = new_adaptive(EXTERNAL_SLOT_DEQUE_CAPACITY, initial_gating);
            let tags = [w0.active_tag(), w1.active_tag(), w2.active_tag(), w3.active_tag()];
            slot[0] = Some(w0);
            slot[1] = Some(w1);
            slot[2] = Some(w2);
            slot[3] = Some(w3);
            stealers.push([s0, s1, s2, s3]);
            k_gating_tags.push(tags);
        }
        // Total indices addressable in arena.stealers et al.
        let total_n = n + EXTERNAL_SLOT_COUNT;

        // Parker slots: one per real worker + one dummy per external
        // slot (slots never park; we put OnceLock<Parker> placeholders
        // so the index space stays uniform). External-slot parkers
        // remain unset for the lifetime of the arena; any code that
        // tries to .get() on them gets None and skips the unpark.
        let parkers_vec: Vec<Arc<OnceLock<Arc<Parker>>>> =
            (0..total_n).map(|_| Arc::new(OnceLock::new())).collect();
        // Wrap in an Arc<Vec<...>> so worker_loop's WorkerCtx can
        // share the parker handles for the wake-on-push fast path
        // without cloning N Arcs per spawn.
        let parkers_handle: Arc<Vec<Arc<OnceLock<Arc<Parker>>>>> =
            Arc::new(parkers_vec.clone());

        // Per-worker + per-slot stat counters. Slot stats track
        // external-caller activity uniformly with worker stats.
        let stats_vec: Vec<Arc<WorkerStats>> =
            (0..total_n).map(|_| Arc::new(WorkerStats::default())).collect();

        // Use the caller-supplied cpu set. None means skip pinning.
        let core_ids = cpu_set;

        let shutdown_flag = Arc::new(AtomicBool::new(false));
        let smt_requests: Arc<AtomicU32> = Arc::new(AtomicU32::new(0));
        let sleep_arc = Arc::new(crate::sched::jec_sleep::Sleep::new(n));
        // Cluster size for the peer-distance helper. Captured
        // once at arena construction from numa_topology; passed
        // into every WorkerCtx so steal_from_peer_tiered can
        // label each victim's distance.
        let ccx_size_for_workers = {
            let topo = crate::numa_topology();
            1usize << topo.cluster_size_log2 as usize
        };
        // Per-worker + per-slot mailboxes. Indices [0..n) are real
        // workers; [n..total_n) are external slots. Mailbox routing
        // is opt-in via dispatch_profile.use_mailbox_routing (off by
        // default), so slot mailboxes stay empty in practice.
        let mailboxes: Vec<Arc<FlynnelRing<JobRef>>> =
            (0..total_n).map(|_| Arc::new(FlynnelRing::new(MAILBOX_CAPACITY))).collect();

        let mut handles: Vec<Option<JoinHandle<()>>> = Vec::with_capacity(n);
        for idx in 0..n {
            // Take this worker's per-tier AdaptiveWorker primitives out of the slot.
            let my_tier_deques: [AdaptiveWorker; N_TIERS] = {
                let slot = &mut workers_per_idx[idx];
                [
                    slot[0].take().expect("tier 0 deque present"),
                    slot[1].take().expect("tier 1 deque present"),
                    slot[2].take().expect("tier 2 deque present"),
                    slot[3].take().expect("tier 3 deque present"),
                ]
            };
            let peer_stealers = stealers.clone();
            let inj = Arc::clone(&injector);
            let park_slot = Arc::clone(&parkers_vec[idx]);
            let shutdown = Arc::clone(&shutdown_flag);
            let parkers_for_worker = Arc::clone(&parkers_handle);
            let stats_for_worker = Arc::clone(&stats_vec[idx]);
            // This worker's own mailbox + the full peer mailbox vec
            // so push_to_mailbox can target any peer.
            let my_mailbox = Arc::clone(&mailboxes[idx]);
            let peer_mailboxes_for_worker: Vec<Arc<FlynnelRing<JobRef>>> =
                mailboxes.iter().map(Arc::clone).collect();
            // Each worker carries a parallel array of every worker's
            // stats so the thief code can increment the VICTIM's
            // `times_stolen_from` counter at the steal site.
            let peer_stats_for_worker: Vec<Arc<WorkerStats>> =
                stats_vec.iter().map(Arc::clone).collect();
            let smt_req_for_worker = Arc::clone(&smt_requests);
            let sleep_for_worker = Arc::clone(&sleep_arc);
            let is_primary = idx < primary;
            // Round-robin CPU assignment so workers spread across
            // available cores. When n_workers == physical_cores
            // and the OS enumerates first-SMT-siblings as the
            // low-numbered CPUs (typical on AMD Zen / modern
            // Intel), workers 0..n-1 each land on a distinct
            // physical core.
            let assigned_core = core_ids.as_ref().map(|ids| ids[idx % ids.len()]);
            let h = thread::Builder::new()
                .name(format!("flynnel-sched-{idx}"))
                // 8 MiB stack: the default (1 MiB on Windows,
                // 8 MiB on Linux pthread) is not portable for the
                // deep reduce_inner bisect path. reduce_chunks
                // over 16M items with MIN_LEAF_ITEMS = 256 gives
                // ~16 levels of recursive join_context; each
                // level retains both half-result slots on the
                // stack plus the StackJob captured closure state,
                // and large result types like [u64; 256] for
                // histogram workloads multiply the per-level cost.
                // The worker wait-loop additionally executes
                // stolen StackJobs whose bodies themselves
                // re-enter reduce_inner, so the effective
                // nesting depth is the bisect depth * the
                // wait-loop recursive-steal depth. Empirically
                // verified 2026-06-16 via FLYNNEL_RC_DEBUG stack-
                // pointer trace: 4 MiB overflowed on Windows at
                // 16M items / max_budget=32; 8 MiB does not.
                .stack_size(8 * 1024 * 1024)
                .spawn(move || {
                    // Pre-init shutdown check: if the arena was
                    // dropped between spawn and worker startup,
                    // exit without ever entering the loop.
                    if shutdown.load(Ordering::Acquire) {
                        return;
                    }
                    // Pin this worker BEFORE constructing the
                    // Parker so subsequent thread::current() calls
                    // and any cache pre-touches happen on the
                    // assigned core.
                    if let Some(core) = assigned_core {
                        let _ = core_affinity::set_for_current(core);
                    }
                    // Construct the Parker inside the worker so it
                    // captures THIS thread's handle.
                    let parker = Arc::new(Parker::new(LOCAL_SPIN_ROUNDS));
                    park_slot
                        .set(Arc::clone(&parker))
                        .expect("park_slot must only be set once per worker");
                    // Re-check shutdown after parker init: it is
                    // possible (race) for Drop to fire between the
                    // pre-check and parker-slot-set. The shared
                    // flag is independent of Parker so this is the
                    // canonical exit point in either case.
                    if shutdown.load(Ordering::Acquire) {
                        return;
                    }
                    worker_loop(
                        idx,
                        my_tier_deques,
                        peer_stealers,
                        ccx_size_for_workers,
                        my_mailbox,
                        peer_mailboxes_for_worker,
                        inj,
                        parker,
                        shutdown,
                        parkers_for_worker,
                        stats_for_worker,
                        peer_stats_for_worker,
                        sleep_for_worker,
                        is_primary,
                        smt_req_for_worker,
                    );
                })
                .expect("worker thread spawn must succeed");
            handles.push(Some(h));
        }

        // ------- Build external slot WorkerCtx + Arc<ExternalSlot> -------
        // Each slot's WorkerCtx has the FULL extended stealers /
        // mailboxes / stats / parkers so external callers can:
        //  - see all workers + other slots as victims for stealing
        //  - have peers steal from THEIR pushed work (since the
        //    slot's stealers are at arena.stealers[n+slot_id])
        let mut external_slots: Vec<Arc<ExternalSlot>> =
            Vec::with_capacity(EXTERNAL_SLOT_COUNT);
        for (slot_id, slot_entry) in slot_workers_per_idx.iter_mut().enumerate() {
            let arena_index = n + slot_id;
            let deques: [AdaptiveWorker; N_TIERS] = [
                slot_entry[0].take().expect("slot tier 0 present"),
                slot_entry[1].take().expect("slot tier 1 present"),
                slot_entry[2].take().expect("slot tier 2 present"),
                slot_entry[3].take().expect("slot tier 3 present"),
            ];
            let slot_mailbox = Arc::clone(&mailboxes[arena_index]);
            let slot_peer_mailboxes: Vec<Arc<FlynnelRing<JobRef>>> =
                mailboxes.iter().map(Arc::clone).collect();
            let slot_stats = Arc::clone(&stats_vec[arena_index]);
            let slot_peer_stats: Vec<Arc<WorkerStats>> =
                stats_vec.iter().map(Arc::clone).collect();
            let slot_stealers = stealers.clone();
            let ctx = WorkerCtx {
                workers: deques,
                index: arena_index,
                burst_pushed: Cell::new(0),
                stealers: slot_stealers,
                steal_stash: core::cell::UnsafeCell::new(AdaptiveStash::empty()),
                ccx_size: ccx_size_for_workers,
                mailbox: slot_mailbox,
                peer_mailboxes: slot_peer_mailboxes,
                injector: Arc::clone(&injector),
                rng: Cell::new(0x9E37_79B9_7F4A_7C15u64
                    .wrapping_mul(arena_index as u64 + 1)),
                parkers: Arc::clone(&parkers_handle),
                wake_rotor: Cell::new(arena_index),
                stats: slot_stats,
                peer_stats: slot_peer_stats,
                sleep: Arc::clone(&sleep_arc),
                last_victim: Cell::new(usize::MAX),
                is_external_slot: true,
            };
            external_slots.push(Arc::new(ExternalSlot {
                claimed: AtomicBool::new(false),
                ctx: core::cell::UnsafeCell::new(ctx),
                index_in_arena: arena_index,
            }));
        }

        Arc::new(Self {
            injector,
            stealers,
            parkers: parkers_vec,
            stats: stats_vec,
            workers: handles,
            primary_count: primary,
            smt_requests,
            shutdown_flag,
            sleep: sleep_arc,
            k_gating_tags,
            external_slots,
        })
    }

    /// Try to claim an unused external slot. Returns `Some(guard)`
    /// holding the slot and an installed TLS WorkerCtx pointer so
    /// the calling thread can run `join_in_worker` directly on
    /// the slot's ctx. Returns `None` if all slots are currently
    /// in use (rare under normal load -- 32 slots is plenty for
    /// concurrent external dispatch).
    ///
    /// The returned `ExternalSlotGuard` releases the claim on Drop
    /// AND restores any prior TLS ctx pointer.
    pub(crate) fn try_claim_external_slot(self: &Arc<Self>)
        -> Option<ExternalSlotGuard>
    {
        for slot in self.external_slots.iter() {
            // Fast-path read before the CAS: an Acquire load is
            // cheaper than a failing CAS (no cache-line invalidation
            // for the writer). Skip the CAS attempt entirely if
            // the slot is observably claimed.
            if slot.is_claimed() {
                continue;
            }
            if slot.claimed
                .compare_exchange(false, true, Ordering::AcqRel, Ordering::Relaxed)
                .is_ok()
            {
                debug_assert_eq!(
                    slot.index(),
                    unsafe { (*slot.ctx.get()).index },
                    "slot index_in_arena must match ctx.index",
                );
                // Won the claim. Save prior TLS ctx (usually null),
                // install slot's ctx pointer.
                let prev = current_worker_ctx();
                let ctx_ptr: *const WorkerCtx = slot.ctx.get();
                // Reset Cell state for the new claimer so previous
                // claimer's RNG / last_victim / counters don't
                // leak across uses.
                // SAFETY: claim CAS guarantees exclusive access;
                // ctx pointer is stable and valid.
                unsafe {
                    let ctx = &*ctx_ptr;
                    ctx.rng.set(0x9E37_79B9_7F4A_7C15u64
                        .wrapping_mul(slot.index_in_arena as u64 + 1));
                    ctx.last_victim.set(usize::MAX);
                    ctx.burst_pushed.set(0);
                    ctx.wake_rotor.set(slot.index_in_arena);
                }
                // Clear any existing TLS ctx first (debug_assert
                // in set_current_worker_ctx requires a null
                // starting state; we save prev above so Drop
                // restores it). This guards against the rare
                // case where the caller already has a worker
                // ctx (e.g., a test that pre-set TLS, or a
                // recursive external_dispatch path bypassing
                // local_join_context's null-check fast path).
                clear_current_worker_ctx();
                unsafe { set_current_worker_ctx(ctx_ptr) };
                return Some(ExternalSlotGuard {
                    slot: Arc::clone(slot),
                    prev_ctx: prev,
                });
            }
        }
        None
    }


    /// Flip the active K_gating across EVERY worker and tier in
    /// this arena. Single Release-store pass over all
    /// `k_gating_tags`; new pushes route to the new backing
    /// starting immediately on each worker. Existing items in
    /// the old (now-dormant) backing drain naturally via each
    /// worker's `pop()` (which walks both backings active-first).
    ///
    /// The migration cost is essentially zero - one atomic store
    /// per (worker, tier) tuple. On a 16-worker arena with 4
    /// tiers that's 64 atomic stores, ~30 ns total. Per-op cost
    /// on subsequent pushes is unchanged (AtomicU32 Acquire load
    /// adds 0.02 ns over direct dispatch, measured on Zen+ R7
    /// 2700 2026-06-06).
    pub fn migrate_all_workers_k_gating(&self, gating: KGating) {
        use core::sync::atomic::Ordering;
        let target = match gating.resolved() {
            KGating::CounterOnly => 1u32,
            _ => 0u32, // PerSlot / Auto -> KHL
        };
        for tier_arr in &self.k_gating_tags {
            for tag in tier_arr.iter() {
                tag.store(target, Ordering::Release);
            }
        }
    }

    /// Sum the burst-vs-single profile across all workers. Returns
    /// the global burst ratio in [0.0, 1.0] - 1.0 means every push
    /// went through `push_burst` (cooperative fan-out pattern);
    /// 0.0 means every push went through `push` (join right-half
    /// pattern). Used by an application's workload-shift detector
    /// to decide when to call [`Self::migrate_all_workers_k_gating`].
    pub fn global_burst_ratio(&self) -> f32 {
        use core::sync::atomic::Ordering;
        let mut bursts: u64 = 0;
        let mut singles: u64 = 0;
        for w in &self.stats {
            bursts = bursts.saturating_add(w.burst_pushes.load(Ordering::Relaxed));
            singles = singles.saturating_add(w.single_pushes.load(Ordering::Relaxed));
        }
        let total = bursts + singles;
        if total == 0 {
            0.5
        } else {
            (bursts as f32) / (total as f32)
        }
    }

    /// Borrow the per-worker stat counters. Used by the split
    /// observer to compute steal pressure across the pool.
    pub fn worker_stats(&self) -> &[Arc<WorkerStats>] {
        &self.stats
    }

    /// Number of always-active primary workers. Workers beyond
    /// this index are SMT siblings gated on the SMT request
    /// counter.
    pub fn primary_count(&self) -> usize {
        self.primary_count
    }

    /// Number of SMT-sibling workers (parked unless an active
    /// SMT request keeps them awake). May be 0 if the arena was
    /// built with [`Self::with_cpu_set`] (no SMT extension).
    pub fn smt_extension_count(&self) -> usize {
        self.workers.len().saturating_sub(self.primary_count)
    }

    /// Raise the SMT request counter and unpark all SMT-sibling
    /// workers so they join the work-stealing loop. Returns an
    /// RAII guard that lowers the counter when dropped; when the
    /// counter returns to 0, siblings re-park at the next loop
    /// iteration.
    ///
    /// Nested calls compose via the counter - the last guard to
    /// drop is the one that returns the pool to default state.
    pub fn acquire_smt(self: &Arc<Self>) -> SmtGuard {
        let prev = self.smt_requests.fetch_add(1, Ordering::AcqRel);
        if prev == 0 {
            // Edge from "no SMT requests" to "1 request" - wake
            // every sibling parker so they observe the change.
            for slot in self.parkers.iter().skip(self.primary_count) {
                if let Some(p) = slot.get() {
                    p.unpark();
                }
            }
        }
        SmtGuard {
            arena: Arc::clone(self),
        }
    }

    /// Diagnostic snapshot for hang reports: sleep counters, each
    /// worker's per-tier deque occupancy (`.` = empty, else the
    /// unclaimed KHL body count; tiers SmtLocal/IntraCcx/CrossCcx/
    /// Public) plus block state and counters, and every external
    /// slot that is claimed or holds work.
    pub fn debug_snapshot(&self) -> String {
        use core::sync::atomic::Ordering::Relaxed;
        let n = self.workers.len();
        let sleep = self.sleep.debug_state();
        let occupancy = |idx: usize| -> String {
            self.stealers[idx]
                .iter()
                .map(|st| {
                    if st.is_empty() {
                        ".".to_string()
                    } else {
                        format!("{}", st.khl_len().max(1))
                    }
                })
                .collect::<Vec<_>>()
                .join(",")
        };
        let mut s = format!(
            "workers={n} primary={} smt_requests={} injector_empty={} \
             sleeping={} inactive={} jec={}\n",
            self.primary_count,
            self.smt_requests.load(Ordering::Acquire),
            self.injector.is_empty(),
            sleep.sleeping,
            sleep.inactive,
            sleep.jec,
        );
        for i in 0..n {
            let st = &self.stats[i];
            let state = match sleep.blocked.get(i) {
                Some(Some(true)) => "blocked",
                Some(Some(false)) => "awake",
                _ => "mutex-held",
            };
            s.push_str(&format!(
                "  w{i:02} deques[{}] {state} pops={} steals={} stolen_from={} refused={}\n",
                occupancy(i),
                st.local_pops.load(Relaxed),
                st.peer_steal_hits.load(Relaxed),
                st.times_stolen_from.load(Relaxed),
                st.push_refusals.load(Relaxed),
            ));
        }
        for (k, slot) in self.external_slots.iter().enumerate() {
            let occ = occupancy(n + k);
            if slot.is_claimed() || occ.chars().any(|c| c.is_ascii_digit()) {
                s.push_str(&format!(
                    "  slot{k:02} claimed={} deques[{occ}]\n",
                    slot.is_claimed()
                ));
            }
        }
        s
    }
}

// ---------------------------------------------------------------------------
// External slot pool: pre-allocated WorkerCtx slots that external
// caller threads can claim/release on each external dispatch, so they
// can be temporary workers without per-call WorkerCtx allocation AND
// have their stealers registered in arena.stealers so peers can
// steal external-pushed work.
// ---------------------------------------------------------------------------

/// Number of pre-allocated external slots per LocalArena. Sized so
/// concurrent external callers (multi-threaded tests, criterion
/// parallel benches, application threads each calling
/// `sched::join` from outside the pool) get a slot without CAS
/// contention. 32 covers typical concurrent-external loads on
/// hosts up to ~32 cores; oversubscription beyond this falls back
/// to the wrapper-job path.
pub(crate) const EXTERNAL_SLOT_COUNT: usize = 32;

/// Capacity per AdaptiveWorker ring inside an external slot. External
/// callers push at most O(log N) right-halves into their own deque
/// before peers steal them or the caller pops them back. 64 slots
/// (~192 jobs per tier) is plenty.
const EXTERNAL_SLOT_DEQUE_CAPACITY: usize = 64;

/// One pre-allocated external-worker slot. The deques + ctx live on
/// the heap for the lifetime of the arena. Peers can steal from
/// this slot's deques via their registered stealer in
/// arena.stealers (at index `slot.index_in_arena`).
///
/// The `claimed` AtomicBool acts as a CAS-based exclusion: at most
/// one external caller holds a slot at a time. CAS failure falls
/// through to the next slot or eventually to the wrapper-job
/// fallback path in external_dispatch.
pub(crate) struct ExternalSlot {
    /// Set true when an external caller has claimed this slot;
    /// release on RAII guard Drop.
    claimed: AtomicBool,
    /// Stable WorkerCtx for this slot. Boxed so its address is
    /// stable across claims; the TLS pointer published by
    /// `claim` points here.
    ///
    /// SAFETY: WorkerCtx contains `Cell` fields (rng, last_victim,
    /// burst_pushed, wake_rotor) and an `UnsafeCell<AdaptiveStash>`.
    /// These are touched ONLY by the claiming thread per the
    /// `claimed` AtomicBool exclusion. Between claims the Cell
    /// values may be stale (from prior claimer), but they only
    /// affect heuristics (RNG seed, last-victim, etc.) and never
    /// correctness.
    ctx: core::cell::UnsafeCell<WorkerCtx>,
    /// Index in arena.stealers (and parkers / stats / mailboxes)
    /// where THIS slot's stealer/mailbox/stats live. Workers
    /// stealing from this slot use index = self.index_in_arena.
    index_in_arena: usize,
}

// SAFETY: ExternalSlot is Sync because the `claimed` CAS ensures
// only one thread accesses ctx at a time. WorkerCtx's !Send / !Sync
// bound is upheld by the same exclusion. Between claims no thread
// holds a reference into ctx.
unsafe impl Sync for ExternalSlot {}

impl ExternalSlot {
    /// True if this slot is currently held by an external caller.
    #[inline]
    pub(crate) fn is_claimed(&self) -> bool {
        self.claimed.load(Ordering::Acquire)
    }

    /// Slot's position in arena.stealers / parkers / stats /
    /// mailboxes. Workers' WorkerCtx.stealers[index_in_arena]
    /// is the stealer for this slot's deques.
    #[inline]
    pub(crate) fn index(&self) -> usize {
        self.index_in_arena
    }
}

/// RAII guard returned by [`LocalArena::try_claim_external_slot`].
/// While held, the TLS `current_worker_ctx` points to the slot's
/// WorkerCtx so nested join calls take the in-worker fast path. On
/// Drop, the guard clears the TLS pointer and releases the slot's
/// `claimed` flag.
pub(crate) struct ExternalSlotGuard {
    slot: Arc<ExternalSlot>,
    prev_ctx: *const WorkerCtx,
}

impl ExternalSlotGuard {
    /// Pointer to the WorkerCtx this guard installed in TLS. Used by
    /// external_dispatch to call `join_in_worker(ctx, ...)` without
    /// re-fetching from TLS.
    #[inline]
    pub(crate) fn ctx_ref(&self) -> &WorkerCtx {
        // SAFETY: ctx_box lives in the slot for the arena's
        // lifetime (Arc<ExternalSlot> held in arena.external_slots).
        // The claim CAS prevents other threads from accessing the
        // same ctx concurrently. The reference is valid for the
        // lifetime of this guard.
        unsafe { &*self.slot.ctx.get() }
    }
}

impl Drop for ExternalSlotGuard {
    fn drop(&mut self) {
        // Restore prior TLS pointer (usually null). Must clear
        // first because set_current_worker_ctx debug-asserts
        // that the slot is null when called -- our TLS currently
        // holds the slot's ctx pointer that we installed at
        // claim time.
        clear_current_worker_ctx();
        if !self.prev_ctx.is_null() {
            unsafe { set_current_worker_ctx(self.prev_ctx) };
        }
        // Release the claim so another external caller can use
        // this slot.
        self.slot.claimed.store(false, Ordering::Release);
    }
}

/// RAII guard for an SMT request held against a [`LocalArena`].
/// Dropping the guard decrements the request counter. When the
/// counter returns to 0, SMT-sibling workers park at the next
/// loop iteration and the pool reverts to primary-only.
#[derive(Debug)]
pub struct SmtGuard {
    arena: Arc<LocalArena>,
}

impl Drop for SmtGuard {
    fn drop(&mut self) {
        // Decrement; no wake needed - siblings will observe the
        // 0 value at the top of their next loop iteration and
        // re-park. The work they're currently running completes
        // normally; only the next round of work-search gates them.
        self.arena.smt_requests.fetch_sub(1, Ordering::AcqRel);
    }
}

impl LocalArena {
    /// Submit a `JobRef` to the global injector + unpark workers.
    ///
    /// Wake policy (tuned on the rayon crossover bench): if the
    /// injector was empty BEFORE this push, the
    /// pool was either at cold-start or all workers had drained
    /// to their parks; broadcast unpark fills the pool. If non-
    /// empty, a single rotated unpark is enough because workers
    /// keep stealing from the injector while it has items.
    ///
    /// Combined with the [`LOCAL_SPIN_ROUNDS`] spin floor (64; see
    /// its doc for the tuning evidence), most repeated-dispatch
    /// loops never actually park their workers; the burst-wake
    /// fires only on genuine cold-start.
    ///
    /// # Safety
    ///
    /// Same as [`JobRef::execute`]: the underlying job's captured
    /// state must remain valid until the job runs.
    pub(crate) unsafe fn submit(&self, job: JobRef) {
        let was_empty = self.injector.is_empty();
        self.injector.push(job);
        // Route notification through Sleep so workers parked on
        // the JEC condvar get woken via `wake_specific_thread`.
        self.sleep.new_internal_jobs(1, was_empty);
    }

    /// Try to find and execute exactly one job from the pool.
    /// Returns `true` if a job was found and ran, `false` if no
    /// work is currently available.
    ///
    /// Threads waiting on a child latch (e.g., the parent half of
    /// `local_join`) call this in their wait loop to participate
    /// in work-stealing instead of busy-yielding. Each call
    /// either drives the pool forward by one job or returns
    /// quickly so the caller can re-check its latch.
    ///
    /// `rng_state` is a caller-owned xorshift state used to pick
    /// the victim. Each waiter should pass its own state so
    /// different waiters don't all converge on the same victim.
    pub fn try_run_one(&self, rng_state: &mut u64) -> bool {
        // Try the global injector first - this is where external
        // submissions land and where most fork-half right-jobs
        // live.
        if let Steal::Success(job) = self.injector.steal() {
            // Prefetch the job's captured state into L2 before
            // `execute()` dispatches through the vtable.
            //
            // SAFETY: the prefetch hint is no-side-effect even
            // if the 64-byte window extends past the captured
            // state (see `prefetch::prefetch_hint_l2`).
            crate::sched::prefetch::prefetch_into_l2_inline(unsafe {
                core::slice::from_raw_parts(job.data_ptr() as *const u8, 64)
            });
            // SAFETY: `JobRef::execute` requires single-call and
            // data-validity; we got `job` from the injector
            // which was produced by a matching `submit` call
            // that owns the validity contract.
            unsafe { job.execute() };
            return true;
        }
        // Random peer steal. External waiters have an unknown
        // topology distance to any specific worker (they may be on
        // a thread the OS scheduler placed anywhere), so they only
        // probe the Public tier - the widest, no-distance-required
        // tier of each peer.
        let n = self.stealers.len();
        if n == 0 {
            return false;
        }
        // Xorshift step.
        let mut x = *rng_state;
        if x == 0 {
            x = 0x9E37_79B9_7F4A_7C15;
        }
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        *rng_state = x;
        let idx = (x as usize) % n;
        // External waiters use a per-thread stash via thread_local
        // so K_inner=3 batches deliver up to 3 jobs per coherence
        // transfer; the waiter consumes them all inline before
        // returning to its latch poll loop.
        thread_local! {
            static EXTERNAL_STEAL_STASH: core::cell::RefCell<AdaptiveStash>
                = core::cell::RefCell::new(AdaptiveStash::empty());
        }
        // Drain any prior batch's leftovers first.
        let drained = EXTERNAL_STEAL_STASH.with(|s| s.borrow_mut().drain_one());
        if let Some(job) = drained {
            crate::sched::prefetch::prefetch_into_l2_inline(unsafe {
                core::slice::from_raw_parts(job.data_ptr() as *const u8, 64)
            });
            // SAFETY: the drained item came from a completed
            // successful steal that absorbed its captured-state
            // contract.
            unsafe { job.execute() };
            return true;
        }
        // Empty stash: steal a fresh batch.
        let steal_result = EXTERNAL_STEAL_STASH.with(|s| {
            steal_via_stash(&self.stealers[idx][DequeTier::Public.idx()], &mut s.borrow_mut())
        });
        if let AdaptiveSteal2::Success(job) = steal_result {
            crate::sched::prefetch::prefetch_into_l2_inline(unsafe {
                core::slice::from_raw_parts(job.data_ptr() as *const u8, 64)
            });
            // SAFETY: same data-validity contract as the in-pool
            // peer steal path.
            unsafe { job.execute() };
            return true;
        }
        false
    }

    /// Number of worker threads in this arena.
    pub fn worker_count(&self) -> usize {
        self.workers.len()
    }

    /// Test helper: pending injector length.
    #[cfg(test)]
    #[allow(dead_code)]
    pub(crate) fn injector_len(&self) -> usize {
        self.injector.len()
    }

    /// Expose the injector for cross-arena probes by
    /// [`crate::sched::arena_numa::NumaArena`]. Used only inside
    /// the crate; external callers should go through
    /// `LocalArena::submit` / `NumaArena::submit`.
    #[allow(dead_code)]
    pub(crate) fn injector_view(&self) -> &Injector<JobRef> {
        &self.injector
    }

    /// Block until the injector is empty. Used by tests that need
    /// to ensure all submitted jobs have been picked up. Note:
    /// "picked up" does not mean "completed" because per-worker
    /// deques are not visible from here; the caller must own its
    /// own completion signal (typically a latch in each StackJob).
    pub fn wait_injector_drained(&self) {
        while !self.injector.is_empty() {
            thread::yield_now();
        }
    }
}

impl Drop for LocalArena {
    fn drop(&mut self) {
        // Step 1: signal the shared shutdown flag. Every worker
        // observes this irrespective of whether its individual
        // Parker has been initialized yet, so workers caught
        // mid-init exit cleanly via the pre/post-init checks in
        // their spawn closure.
        self.shutdown_flag.store(true, Ordering::Release);
        // Step 2: also shutdown each Parker (sets per-Parker flag
        // + unparks). This wakes workers that have already
        // entered `worker_loop` and may be parked.
        for p in &self.parkers {
            if let Some(parker) = p.get() {
                parker.shutdown();
            }
        }
        // Step 2b: JEC sleep coordinator broadcast-wake. Workers
        // sitting on the JEC condvar do not observe
        // Parker.shutdown(); they need the condvar notify_one in
        // wake_all_for_shutdown.
        self.sleep.wake_all_for_shutdown();
        // Step 3: join all worker threads. With both signals
        // delivered, every worker exits either via the pre-init
        // shutdown check (uninitialized case) or via the
        // `worker_loop` shutdown branch (running case).
        for slot in &mut self.workers {
            if let Some(h) = slot.take()
                && h.join().is_err()
            {
                eprintln!("flynnel: worker thread panicked before shutdown join");
            }
        }
    }
}

/// Worker thread body. Each iteration probes in order:
///   0. SMT-sibling gate: siblings park while `smt_requests == 0`.
///   1. Steal-stash drain (K_inner=3 leftovers from a prior steal).
///   2. Mailbox pop (owner-directed hand-offs from peers).
///   3. Local deque tier walk (SmtLocal -> IntraCcx -> CrossCcx
///      -> Public), LIFO.
///   4. Injector steal (externally-submitted work).
///   5. Peer steal: up to `PROBE_LARGE` (or `n-1` on small pools)
///      random peers, each walked across the tiers the asymmetric
///      steal discipline allows from our distance.
///   6. Park via the JEC coordinator (`Sleep::no_work_found`) with
///      a predicate that re-checks own local + injector; the
///      producer's `Sleep::new_internal_jobs` wakes us on push.
///
/// On entry: stack-allocates a [`WorkerCtx`] holding (my_deque,
/// stealers, injector, per-worker rng) and registers a `*const
/// WorkerCtx` in this thread's `WORKER_CTX` thread-local. That
/// pointer is consumed by `arena::join`'s fast path: when `join` is
/// called from inside a worker, it pushes the right-half job to this
/// worker's local deque (5-10 ns) instead of the global injector
/// (5-30 µs). The local-deque path is rayon's central perf win;
/// without it sched cannot match rayon's wallclock at small N. See
/// rayon-core/src/registry.rs::WorkerThread for the lineage.
///
/// Exits as soon as either `parker.is_shutdown()` (individual
/// signal, used by ordinary shutdown) or `arena_shutdown.load`
/// (shared flag, used by `Drop` to reach workers that may not
/// yet be parked) returns true. The thread-local pointer is
/// cleared before this function returns.
#[allow(clippy::too_many_arguments)]
fn worker_loop(
    idx: usize,
    my_tier_deques: [AdaptiveWorker; N_TIERS],
    stealers: Vec<[AdaptiveStealer; N_TIERS]>,
    ccx_size: usize,
    mailbox: Arc<FlynnelRing<JobRef>>,
    peer_mailboxes: Vec<Arc<FlynnelRing<JobRef>>>,
    injector: Arc<Injector<JobRef>>,
    parker: Arc<Parker>,
    arena_shutdown: Arc<AtomicBool>,
    parkers: Arc<Vec<Arc<OnceLock<Arc<Parker>>>>>,
    stats: Arc<WorkerStats>,
    peer_stats: Vec<Arc<WorkerStats>>,
    sleep: Arc<crate::sched::jec_sleep::Sleep>,
    is_primary: bool,
    smt_requests: Arc<AtomicU32>,
) {
    // Build the stack-resident WorkerCtx. Lives until this function
    // returns; the thread-local pointer never outlives it.
    let ctx = WorkerCtx {
        workers: my_tier_deques,
        index: idx,
        burst_pushed: Cell::new(0),
        stealers,
        steal_stash: core::cell::UnsafeCell::new(AdaptiveStash::empty()),
        ccx_size,
        mailbox,
        peer_mailboxes,
        injector,
        rng: Cell::new(0x9E37_79B9_7F4A_7C15u64.wrapping_mul(idx as u64 + 1)),
        parkers,
        wake_rotor: Cell::new(idx),
        stats,
        peer_stats,
        sleep,
        last_victim: Cell::new(usize::MAX),
        is_external_slot: false,
    };
    // SAFETY: ctx lives on this stack frame until clear_current_worker_ctx()
    // runs at the bottom of the function. No other thread reads our
    // thread-local, so the *const WorkerCtx is only dereferenced from
    // this owner thread.
    unsafe { set_current_worker_ctx(&ctx as *const WorkerCtx) };
    // Use a defer-guard for the clear so a panic in any code below
    // still unregisters the thread-local before the stack unwinds
    // past ctx (defence-in-depth; worker bodies should not panic).
    struct ClearOnDrop;
    impl Drop for ClearOnDrop {
        fn drop(&mut self) {
            clear_current_worker_ctx();
        }
    }
    let _clear_guard = ClearOnDrop;

    let n = ctx.stealers.len();
    // Per-worker pseudo-random state for victim selection.
    let mut rng_state: u64 = 0x9E37_79B9_7F4A_7C15u64.wrapping_mul(idx as u64 + 1);
    let trace_label = format!("flynnel-worker-{idx}");
    // JEC idle state. None when actively executing; Some when
    // searching for work (between job execution). Created on the
    // first miss after a find; destroyed when a find succeeds.
    let mut jec_idle: Option<crate::sched::jec_sleep::IdleState> = None;
    // Helper macro: account a successful find transitioning us
    // from idle->active. No-op when the worker is not in the
    // idle search loop (jec_idle is None).
    macro_rules! jec_account_work_found {
        () => {{
            if let Some(idle) = jec_idle.take() {
                ctx.sleep.work_found(&idle);
            }
        }};
    }

    while !parker.is_shutdown() && !arena_shutdown.load(Ordering::Acquire) {
        // Trace flush hook. When a debug binary calls
        // `crate::sched::trace::request_worker_flush()`, each worker
        // sees the flag here on its next loop iteration and dumps
        // its thread-local trace buffer to stderr. Bare expression
        // statement so the informational bool return is dropped
        // without hitting the no-let-underscore hook (it is not a
        // Result). Branch-cheap when tracing is disabled.
        crate::sched::trace::worker_loop_maybe_flush(&trace_label);

        // SMT sibling gate. Siblings (is_primary=false) participate
        // only while smt_requests > 0. When the counter is 0 they
        // wait via a two-step yield-then-sleep window before
        // truly parking. The gate is load-bearing: siblings
        // running unconditionally measure a 1.88x regression on
        // 10ms-sqrt-chain workloads because they contest the
        // physical core's FP pipe. The OS schedules both SMT
        // threads concurrently; only parking the sibling frees
        // the FP pipe for the primary's dependency chain.
        //
        // Two-step wait for cheap re-acquire on slow iter cadences
        // (criterion 10ms x N benches): yield-spin then micro-sleep
        // before truly parking. Avoids parker.unpark syscall on
        // the common rapid-fire case while still allowing the OS
        // to deschedule the sibling for FP-contention-sensitive
        // workloads.
        if !is_primary && smt_requests.load(Ordering::Acquire) == 0 {
            const SMT_SIBLING_SPIN_ROUNDS: u32 = 1000;
            let mut spun = 0u32;
            let mut acquired_again = false;
            while spun < SMT_SIBLING_SPIN_ROUNDS {
                if smt_requests.load(Ordering::Acquire) > 0 {
                    acquired_again = true;
                    break;
                }
                if parker.is_shutdown() || arena_shutdown.load(Ordering::Acquire) {
                    return;
                }
                std::thread::yield_now();
                spun += 1;
            }
            if acquired_again {
                continue;
            }
            const SMT_SIBLING_SLEEP_ROUNDS: u32 = 100;
            const SMT_SIBLING_SLEEP_US: u64 = 100;
            let mut slept = 0u32;
            while slept < SMT_SIBLING_SLEEP_ROUNDS {
                if smt_requests.load(Ordering::Acquire) > 0 {
                    acquired_again = true;
                    break;
                }
                if parker.is_shutdown() || arena_shutdown.load(Ordering::Acquire) {
                    return;
                }
                std::thread::sleep(std::time::Duration::from_micros(SMT_SIBLING_SLEEP_US));
                slept += 1;
            }
            if acquired_again {
                continue;
            }
            let smt_ref = &*smt_requests;
            let parker_ref = &*parker;
            let arena_shutdown_ref = &*arena_shutdown;
            let _wait_rc = parker.park_until(|| {
                smt_ref.load(Ordering::Acquire) > 0
                    || parker_ref.is_shutdown()
                    || arena_shutdown_ref.load(Ordering::Acquire)
            });
            continue;
        }
        // (-1) Steal-stash drain: a recent successful peer steal
        //      left 1-2 extra items in the WorkerCtx steal_stash.
        //      Drain them first - they are locality-warm and
        //      already paid the coherence transfer.
        //
        // SAFETY: owner-private steal_stash; single-threaded read.
        {
            let stash = unsafe { &mut *ctx.steal_stash.get() };
            if let Some(job) = stash.drain_one() {
                jec_account_work_found!();
                // SAFETY: drained job came from a successful peer
                // steal that absorbed its captured-state contract.
                unsafe { job.execute() };
                continue;
            }
        }
        // (0) Mailbox first - owner-directed hand-offs from peers.
        //     Mailbox work is the most-locality-warm thing the
        //     worker has been given; drain it before any deque.
        if let PopResult::Ok(job) = ctx.mailbox.pop() {
            ctx.stats
                .local_pops
                .fetch_add(1, core::sync::atomic::Ordering::Relaxed);
            jec_account_work_found!();
            // SAFETY: `job` was placed in our mailbox by some
            // peer's `push_to_mailbox` call, which upholds the
            // same data-validity contract as `push_tier`.
            unsafe { job.execute() };
            continue;
        }
        // (1) Local deques - walk tiers in distance order (SmtLocal
        //     first; closest = hottest cache).
        let mut popped_local = false;
        for tier in DequeTier::all() {
            if let Some(job) = ctx.workers[tier.idx()].pop() {
                ctx.stats
                    .local_pops
                    .fetch_add(1, core::sync::atomic::Ordering::Relaxed);
                jec_account_work_found!();
                // SAFETY: `job` came from one of our own per-tier
                // deques via `WorkerCtx::push_tier`, whose caller
                // upholds the data-validity contract; a successful
                // pop gives us sole ownership of the single execute
                // call.
                unsafe { job.execute() };
                popped_local = true;
                break;
            }
        }
        if popped_local {
            continue;
        }
        // (2) Global injector for externally submitted work.
        match ctx.injector.steal() {
            Steal::Success(job) => {
                jec_account_work_found!();
                // SAFETY: `job` was placed in the injector by
                // `LocalArena::submit`, whose `# Safety` clause
                // requires captured-state validity to outlast
                // the eventual `execute`.
                unsafe { job.execute() };
                continue;
            }
            Steal::Empty | Steal::Retry => {}
        }
        // (3) Peer steal: probe up to PROBE_LARGE random peers in
        //     rotated order, each at all tiers the steal discipline
        //     allows from our distance. Cilk's THE protocol +
        //     KHPD-style per-tier filtering.
        if n >= 2 {
            let probe_count = if n <= PROBE_FULL_CUTOFF {
                n - 1
            } else {
                PROBE_LARGE
            };
            let mut found = false;
            let start = pick_victim(&mut rng_state, idx, n);
            let mut probed = 0usize;
            let mut offset = 0usize;
            while probed < probe_count && offset < n {
                let victim = (start + offset) % n;
                offset += 1;
                if victim == idx {
                    continue;
                }
                probed += 1;
                // Walk allowed tiers for this peer: tier >= our
                // distance to them. SmtLocal-distance thieves get
                // all 4 tiers; cross-CCX thieves only Public.
                let distance = peer_distance(idx, victim, ccx_size);
                for tier_idx in distance.idx()..N_TIERS {
                    // Drain via the worker's steal stash so the
                    // K_inner=3 batch returns 3 logical jobs per
                    // coherence transfer.
                    //
                    // SAFETY: owner-private steal_stash; the outer
                    // loop drains the stash via find_work before
                    // returning to peer-steal, so the stash is
                    // empty here (debug_assert in steal_via_stash
                    // catches any violation).
                    let stash = unsafe { &mut *ctx.steal_stash.get() };
                    let job_opt = match steal_via_stash(&ctx.stealers[victim][tier_idx], stash) {
                        AdaptiveSteal2::Success(j) => Some(j),
                        AdaptiveSteal2::Empty => None,
                        AdaptiveSteal2::Retry => {
                            // Single retry then move on.
                            match steal_via_stash(&ctx.stealers[victim][tier_idx], stash) {
                                AdaptiveSteal2::Success(j) => Some(j),
                                _ => None,
                            }
                        }
                    };
                    if let Some(job) = job_opt {
                        ctx.stats
                            .peer_steal_hits
                            .fetch_add(1, core::sync::atomic::Ordering::Relaxed);
                        if victim < ctx.peer_stats.len() {
                            ctx.peer_stats[victim]
                                .times_stolen_from
                                .fetch_add(1, core::sync::atomic::Ordering::Relaxed);
                        }
                        // Record victim for next find_work + warm
                        // its Stealer line.
                        ctx.last_victim.set(victim);
                        ctx.prefetch_last_victim_stealer();
                        jec_account_work_found!();
                        // SAFETY: same data-validity contract.
                        unsafe { job.execute() };
                        found = true;
                        break;
                    }
                }
                if found {
                    break;
                }
            }
            if found {
                continue;
            }
            // All probes empty; count as a miss for observer.
            ctx.stats
                .peer_steal_misses
                .fetch_add(1, core::sync::atomic::Ordering::Relaxed);
        }
        // (4) Park until unparked or shutdown. The predicate
        // checks only own local + injector + shutdown, never peer
        // stealers: a peer-walking predicate costs 448 atomic
        // reads per spin window per parked worker and contests
        // the line the productive worker writes on every push.
        // Wakes come from the producer side instead:
        // `WorkerCtx::push_tier` calls
        // `sleep.new_internal_jobs(1, was_empty)` and the JEC
        // coordinator wakes one condvar-parked worker as needed.
        // The idle path advances yield -> sleepy -> sleeping per
        // call; `has_injected_jobs` is consulted in the sleep()
        // race-recovery path for a job that landed mid-transition.
        let idle = jec_idle.get_or_insert_with(|| ctx.sleep.start_looking(idx));
        let inj_ref = &ctx.injector;
        ctx.sleep.no_work_found(idle, || !inj_ref.is_empty());
    }
    // Final accounting: if we exit the loop while still idle,
    // balance the JEC inactive counter so the arena's Drop
    // shutdown can shut down cleanly without a stuck counter.
    if let Some(idle) = jec_idle.take() {
        ctx.sleep.work_found(&idle);
    }
    // _clear_guard drops here, clearing the thread-local.
}

/// Xorshift-style victim selector that skips the worker's own
/// index. Cheap pseudo-random; no fairness guarantee required
/// (just need to avoid degenerate "always steal from worker 1"
/// patterns).
fn pick_victim(state: &mut u64, self_idx: usize, n: usize) -> usize {
    // Xorshift64 step.
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    let mut v = (x as usize) % n;
    if v == self_idx {
        v = (v + 1) % n;
    }
    v
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU32;
    use std::time::{Duration, Instant};

    use crate::foundation::Variant;
    use crate::sched::job::{NUMA_HINT_ANY, StackJob};
    use crate::sched::latch::CoreLatch;

    /// Smoke test: external slots pre-allocated at arena init,
    /// each with a stable index inside arena.stealers so peers
    /// can steal external-pushed work.
    #[test]
    fn external_slots_preallocated_and_indexed() {
        let arena = LocalArena::with_smt_extension(2, 0, None);
        assert_eq!(arena.external_slots.len(), EXTERNAL_SLOT_COUNT);
        for (i, slot) in arena.external_slots.iter().enumerate() {
            assert!(!slot.is_claimed(), "slot {i} unclaimed at init");
            assert_eq!(slot.index(), 2 + i,
                "slot {i} indexed at workers.len() + i");
        }
        assert_eq!(arena.stealers.len(),
            arena.workers.len() + EXTERNAL_SLOT_COUNT);
    }

    #[test]
    fn external_slot_claim_release_lifecycle() {
        let arena = LocalArena::with_smt_extension(1, 0, None);
        let guard = arena.try_claim_external_slot()
            .expect("first claim must succeed");
        let slot_idx = guard.ctx_ref().index;
        assert_eq!(slot_idx, 1, "first claim takes slot 0 at index n=1");
        assert!(arena.external_slots[0].is_claimed(),
            "slot 0 marked claimed");
        drop(guard);
        assert!(!arena.external_slots[0].is_claimed(),
            "slot 0 released on guard drop");
    }

    #[test]
    fn new_arena_spawns_workers() {
        let a = LocalArena::new(4);
        assert_eq!(a.worker_count(), 4);
        // Drop releases workers.
        drop(a);
    }

    #[test]
    fn new_arena_with_zero_workers_clamps_to_one() {
        let a = LocalArena::new(0);
        assert_eq!(a.worker_count(), 1);
    }

    #[test]
    fn submit_one_job_runs_to_completion() {
        let arena = LocalArena::new(2);
        let counter = Arc::new(AtomicU32::new(0));
        let c = Arc::clone(&counter);
        let job = StackJob::new(
            move |_stolen| {
                c.fetch_add(1, Ordering::SeqCst);
            },
            CoreLatch::new(),
        );
        unsafe {
            let r = job.as_job_ref(2, NUMA_HINT_ANY, Variant::Faithful);
            arena.submit(r);
        }
        // Wait for the latch to be set (job completed on a worker).
        let deadline = Instant::now() + Duration::from_secs(5);
        while !job.latch.is_set() {
            if Instant::now() > deadline {
                panic!("submitted job did not complete within 5s");
            }
            thread::yield_now();
        }
        assert_eq!(counter.load(Ordering::SeqCst), 1);
        drop(arena);
    }

    #[test]
    fn submit_many_jobs_all_run_exactly_once() {
        const N: u32 = 256;
        let arena = LocalArena::new(4);
        let counter = Arc::new(AtomicU32::new(0));

        let jobs: Vec<_> = (0..N)
            .map(|_| {
                let c = Arc::clone(&counter);
                StackJob::new(
                    move |_stolen| {
                        c.fetch_add(1, Ordering::SeqCst);
                    },
                    CoreLatch::new(),
                )
            })
            .collect();

        // Submit all jobs in a burst.
        for j in &jobs {
            unsafe {
                arena.submit(j.as_job_ref(2, NUMA_HINT_ANY, Variant::Faithful));
            }
        }

        // Wait for all latches.
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            let pending = jobs.iter().filter(|j| !j.latch.is_set()).count();
            if pending == 0 {
                break;
            }
            if Instant::now() > deadline {
                panic!("only {}/{N} jobs completed within 10s",
                    N as usize - pending);
            }
            thread::yield_now();
        }
        assert_eq!(counter.load(Ordering::SeqCst), N);
        drop(arena);
    }

    #[test]
    fn pick_victim_never_returns_self() {
        let mut state: u64 = 0x1234_5678_9ABC_DEF0;
        for _ in 0..1000 {
            let v = pick_victim(&mut state, 3, 8);
            assert_ne!(v, 3, "victim must not be self");
            assert!(v < 8, "victim must be in range");
        }
    }

    #[test]
    fn pick_victim_distributes_across_n_minus_one_peers() {
        let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut hits = [0u32; 8];
        let self_idx = 0;
        for _ in 0..1000 {
            let v = pick_victim(&mut state, self_idx, 8);
            hits[v] += 1;
        }
        assert_eq!(hits[self_idx], 0, "self never picked");
        // Other 7 buckets should all have non-trivial counts.
        for (i, &h) in hits.iter().enumerate() {
            if i != self_idx {
                assert!(h > 50, "peer {i} got only {h} hits / 1000");
            }
        }
    }

    #[test]
    fn mailbox_push_from_worker_routes_to_target_worker() {
        // Real mailbox-routing E2E: a parent job dispatched onto
        // an arena calls push_to_mailbox(target_idx, child) for N
        // children. The target worker drains all N from its mailbox
        // (find_work pops mailbox before deques).
        //
        // The parent uses a fixed_target shared atomic so child
        // pointers are computed inside the closure from the boxed
        // children Vec - whose addresses are stable for the test's
        // lifetime.
        let arena = LocalArena::new(2);
        const N: u32 = 8;
        let counter = Arc::new(AtomicU32::new(0));

        // Boxed children so their addresses are stable across the
        // closure execution. StackJob<L, F, R> = <CoreLatch, F, ()>.
        // Box<dyn FnOnce(bool)> implements FnOnce(bool) -> () so it
        // satisfies the F bound.
        type ChildBody = Box<dyn FnOnce(bool) + Send + 'static>;
        type ChildJob = StackJob<CoreLatch, ChildBody, ()>;
        let children: Vec<Box<ChildJob>> = (0..N)
            .map(|_| {
                let c = Arc::clone(&counter);
                let body: ChildBody = Box::new(move |_| {
                    c.fetch_add(1, Ordering::SeqCst);
                });
                Box::new(StackJob::new(body, CoreLatch::new()))
            })
            .collect();
        // Send-safe wrapper for raw pointer addresses we pass into
        // the parent closure.
        struct SendAddrs(Vec<usize>);
        // SAFETY: usize is Send; we cast back to *const ChildJob
        // inside the worker thread + dereference. The Box<ChildJob>
        // vec outlives the parent execution because the test's outer
        // scope holds the Vec until the assertion loop returns.
        unsafe impl Send for SendAddrs {}
        let addrs = SendAddrs(
            children.iter().map(|b| Box::as_ref(b) as *const _ as usize).collect(),
        );

        let parent_done = Arc::new(AtomicU32::new(0));
        let parent_done_clone = Arc::clone(&parent_done);
        let parent_body: Box<dyn FnOnce(bool) + Send + 'static> = Box::new(move |_| {
            let ctx_ptr = current_worker_ctx();
            assert!(!ctx_ptr.is_null(), "parent must run on a worker thread");
            // SAFETY: ctx is valid for the duration of this closure -
            // the worker_loop holds the WorkerCtx on its stack frame
            // and only clears the thread-local after job execution.
            let ctx = unsafe { &*ctx_ptr };
            // Always push to the OTHER worker's mailbox.
            let target = if ctx.index == 0 { 1 } else { 0 };
            for addr in &addrs.0 {
                // SAFETY: addr originated from Box::as_ref above.
                // The Box still lives in the outer Vec for the
                // test's duration.
                let sj_ptr = *addr as *const ChildJob;
                let job = unsafe {
                    (*sj_ptr).as_job_ref(2, NUMA_HINT_ANY, Variant::Faithful)
                };
                if let Err(j) = ctx.push_to_mailbox(target, job) {
                    // Mailbox full - fall back to a regular push
                    // so no child is lost.
                    assert!(ctx.try_push(j).is_ok(), "test deque has room");
                }
            }
            parent_done_clone.store(1, Ordering::Release);
        });
        let parent = StackJob::new(parent_body, CoreLatch::new());

        // SAFETY: parent + children remain alive on this stack
        // until the wait loop below joins all latches; submit's
        // JobRef construction respects that lifetime.
        unsafe {
            arena.submit(parent.as_job_ref(2, NUMA_HINT_ANY, Variant::Faithful));
        }

        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            let pending = children.iter().filter(|j| !j.latch.is_set()).count();
            let parent_set = parent.latch.is_set();
            if pending == 0 && parent_set && parent_done.load(Ordering::Acquire) == 1 {
                break;
            }
            if Instant::now() > deadline {
                panic!(
                    "mailbox routing failed: {}/{N} children completed; \
                     parent_done={}; parent_latch={}",
                    N as usize - pending,
                    parent_done.load(Ordering::Acquire),
                    parent_set,
                );
            }
            thread::yield_now();
        }
        assert_eq!(counter.load(Ordering::SeqCst), N);
        drop(arena);
    }

    #[test]
    fn push_tier_smt_local_is_drained_by_owner_or_sibling() {
        // Inside a worker, push N children to SmtLocal tier. The
        // SmtLocal-tier deque is drainable by the owner (top of
        // find_work) and by the SMT-sibling thief (peer_distance==
        // SmtLocal allows steal). All N must complete.
        let arena = LocalArena::new(2);
        const N: u32 = 8;
        let counter = Arc::new(AtomicU32::new(0));
        type ChildBody = Box<dyn FnOnce(bool) + Send + 'static>;
        type ChildJob = StackJob<CoreLatch, ChildBody, ()>;
        let children: Vec<Box<ChildJob>> = (0..N)
            .map(|_| {
                let c = Arc::clone(&counter);
                let body: ChildBody = Box::new(move |_| {
                    c.fetch_add(1, Ordering::SeqCst);
                });
                Box::new(StackJob::new(body, CoreLatch::new()))
            })
            .collect();
        struct SendAddrs(Vec<usize>);
        // SAFETY: same justification as the mailbox test's SendAddrs.
        // The Box<ChildJob> vec outlives the parent closure; the
        // raw addresses we cast back to *const ChildJob remain valid.
        unsafe impl Send for SendAddrs {}
        let addrs = SendAddrs(
            children.iter().map(|b| Box::as_ref(b) as *const _ as usize).collect(),
        );
        let parent_done = Arc::new(AtomicU32::new(0));
        let parent_done_clone = Arc::clone(&parent_done);
        let parent_body: Box<dyn FnOnce(bool) + Send + 'static> = Box::new(move |_| {
            let ctx_ptr = current_worker_ctx();
            assert!(!ctx_ptr.is_null());
            // SAFETY: WorkerCtx pointer is live for the duration
            // of this job execution.
            let ctx = unsafe { &*ctx_ptr };
            for addr in &addrs.0 {
                let sj_ptr = *addr as *const ChildJob;
                // SAFETY: sj_ptr originated from Box::as_ref on
                // the children Vec.
                let job = unsafe {
                    (*sj_ptr).as_job_ref(2, NUMA_HINT_ANY, Variant::Faithful)
                };
                assert!(ctx.try_push_tier(job, DequeTier::SmtLocal).is_ok(), "test deque has room");
            }
            parent_done_clone.store(1, Ordering::Release);
        });
        let parent = StackJob::new(parent_body, CoreLatch::new());

        // SAFETY: parent + children live on this stack until the
        // wait loop drains them.
        unsafe {
            arena.submit(parent.as_job_ref(2, NUMA_HINT_ANY, Variant::Faithful));
        }
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            let pending = children.iter().filter(|j| !j.latch.is_set()).count();
            if pending == 0 && parent_done.load(Ordering::Acquire) == 1 {
                break;
            }
            if Instant::now() > deadline {
                panic!(
                    "push_tier SmtLocal: {}/{N} children pending; parent_done={}",
                    pending,
                    parent_done.load(Ordering::Acquire)
                );
            }
            thread::yield_now();
        }
        assert_eq!(counter.load(Ordering::SeqCst), N);
        drop(arena);
    }

    #[test]
    fn arena_drop_joins_workers_cleanly() {
        // Spawn an arena, submit a few jobs, drop it. The Drop
        // impl should signal shutdown + join all worker threads
        // without hanging.
        let arena = LocalArena::new(4);
        let counter = Arc::new(AtomicU32::new(0));

        for _ in 0..16 {
            let c = Arc::clone(&counter);
            // Use a HEAP-allocated job since StackJob lifetime
            // requires the parent stack frame to outlive the job.
            // For this drop-test we want the arena to be able to
            // drop without waiting on every job; the JobRef
            // protocol allows leaked jobs (closures simply don't
            // run if the arena drops before they're stolen).
            //
            // Skip submission to keep the test focused on drop
            // semantics. The previous test exercises submit.
            let _ = c;
        }

        let t0 = Instant::now();
        drop(arena);
        let elapsed = t0.elapsed();
        // Drop should complete within ~2s (workers parked
        // most of the time; shutdown unparks each).
        assert!(elapsed < Duration::from_secs(2),
            "arena drop took {elapsed:?}, expected < 2s");
    }
}
