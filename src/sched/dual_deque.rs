//! Dual-deque worker: one in-heap Chase-Lev deque for environment-
//! capturing closures (`!Marshal` jobs) plus one MMF-backed Chase-Lev
//! deque for process-portable Marshal jobs.
//!
//! A `JobRef` carries heap pointers a cross-process thief cannot
//! dereference, so jobs split into two classes: `!Marshal`
//! closures stay in
//! [`crate::sched::chase_lev_local::Worker<JobRef>`] (cross-thread
//! only), [`crate::sched::Marshal`] jobs reduce to `(closure_id,
//! args)` in
//! [`crate::backend::shared_mem::chase_lev_mmf::MmfChaseLevDeque`]
//! (cross-thread and cross-process).
//!
//! Owner pop and [`DualDeque::steal_any`] prefer MMF first so the
//! cross-process pipe stays primed; without that preference the
//! in-heap deque acts as a sink and cross-process thieves starve.
//! Pushes are explicit: [`DualDeque::push_local`] vs
//! [`DualDeque::push_marshal`]. Cross-process thieves open the MMF
//! file path directly, never this struct.
//!
//! This is a per-worker storage substrate; the production arena
//! ([`crate::sched::arena_local::WorkerCtx`]) uses the adaptive
//! K_inner=3 stack instead. `examples/chase_lev_mmf_steal.rs`
//! exercises this struct directly.

#![allow(clippy::missing_errors_doc)]

use std::path::PathBuf;

use crate::sched::chase_lev_local::{
    Steal as CbSteal, Stealer, Worker, new_chase_lev,
};

use crate::backend::shared_mem::chase_lev_mmf::{
    self, MmfChaseLevDeque, RemoteJobSlot, Steal as MmfSteal,
};
use crate::backend::shared_mem::latch_mmf::MmfLatchArena;
use crate::sched::job::JobRef;

/// One worker's deque pair plus the latch arena both halves publish
/// into.
pub struct DualDeque {
    /// In-heap Chase-Lev for `!Marshal` jobs (closure / vtable
    /// pointer + captured state from this process's heap).
    in_heap: Worker<JobRef>,
    /// MMF-backed Chase-Lev for Marshal jobs (cross-process
    /// portable `(closure_id, args)` slots).
    mmf: MmfChaseLevDeque,
    /// MMF latch arena where MMF-side job results are published.
    latches: MmfLatchArena,
    /// Cached path of the MMF deque file so peers can be told
    /// (e.g., logged or written to a manifest).
    mmf_deque_path: PathBuf,
    /// Cached path of the MMF latch arena file.
    mmf_latches_path: PathBuf,
}

/// Outcome of pushing a Marshal-shaped slot into the MMF half.
pub use crate::backend::shared_mem::chase_lev_mmf::PushError as PushMarshalError;

/// Outcome of an owner-side pop (LIFO from either deque).
///
/// Crate-internal because the `InHeap` variant carries a `JobRef`
/// whose visibility is also `pub(crate)`. External consumers use
/// the Marshal-only [`DualDeque::pop_marshal`] surface.
#[allow(
    dead_code,
    reason = "Exercised by in-module tests + reserved as the canonical pop-shape for the production worker-loop integration that picks between in-heap and MMF work. The MMF half ships now; the in-heap half is wired pending the arena_local::find_work hookup."
)]
pub(crate) enum LocalPop {
    /// Got a `!Marshal` JobRef out of the in-heap deque.
    InHeap(JobRef),
    /// Got a Marshal slot out of the MMF deque.
    Mmf(RemoteJobSlot),
    /// Both deques are empty.
    Empty,
}

impl std::fmt::Debug for LocalPop {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // JobRef does not implement Debug because its captured-state
        // pointer is type-erased; render the closure id instead so
        // diagnostics still distinguish the branches.
        match self {
            LocalPop::InHeap(_) => f.write_str("LocalPop::InHeap(<JobRef>)"),
            LocalPop::Mmf(slot) => {
                write!(f, "LocalPop::Mmf(closure_id={:#x})", slot.closure_id)
            }
            LocalPop::Empty => f.write_str("LocalPop::Empty"),
        }
    }
}

/// Outcome of a thief-side steal (FIFO; tries both deques).
///
/// Crate-internal because the `InHeap` variant carries a `JobRef`
/// whose visibility is also `pub(crate)`. External consumers use
/// the Marshal-only [`DualDeque::steal_marshal`] surface.
#[allow(
    dead_code,
    reason = "Exercised by in-module tests + reserved as the canonical steal-shape for cross-thread thieves once the production worker-loop integration is wired. The MMF half ships now; the in-heap half is wired pending the arena_local::find_work hookup."
)]
pub(crate) enum StealAny {
    /// Got a `!Marshal` JobRef from the in-heap deque.
    InHeap(JobRef),
    /// Got a Marshal slot from the MMF deque.
    Mmf(RemoteJobSlot),
    /// Both deques were empty.
    Empty,
    /// Lost a CAS race; caller should retry.
    Retry,
}

impl std::fmt::Debug for StealAny {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StealAny::InHeap(_) => f.write_str("StealAny::InHeap(<JobRef>)"),
            StealAny::Mmf(slot) => {
                write!(f, "StealAny::Mmf(closure_id={:#x})", slot.closure_id)
            }
            StealAny::Empty => f.write_str("StealAny::Empty"),
            StealAny::Retry => f.write_str("StealAny::Retry"),
        }
    }
}

impl DualDeque {
    /// Create a new dual-deque with a fresh MMF deque + MMF latch
    /// arena at the given paths. The current process is recorded as
    /// owner of the MMF deque; the in-heap deque is local-only by
    /// construction.
    pub fn create(
        mmf_deque_path: impl Into<PathBuf>,
        mmf_latches_path: impl Into<PathBuf>,
        mmf_capacity: usize,
        latch_capacity: usize,
    ) -> std::io::Result<Self> {
        let mmf_deque_path = mmf_deque_path.into();
        let mmf_latches_path = mmf_latches_path.into();
        // chase_lev_local is bounded; size generously to
        // absorb typical bursts. 1024 slots covers all observed
        // workloads.
        const IN_HEAP_CAPACITY: usize = 1024;
        let (in_heap, _in_heap_stealer) = new_chase_lev::<JobRef>(IN_HEAP_CAPACITY);
        let mmf = MmfChaseLevDeque::create(&mmf_deque_path, mmf_capacity)?;
        let latches = MmfLatchArena::create(&mmf_latches_path, latch_capacity)?;
        Ok(Self {
            in_heap,
            mmf,
            latches,
            mmf_deque_path,
            mmf_latches_path,
        })
    }

    /// Path of the MMF deque file. Peer processes pass this to
    /// [`MmfChaseLevDeque::open`] to attach as thieves.
    pub fn mmf_deque_path(&self) -> &std::path::Path {
        &self.mmf_deque_path
    }

    /// Path of the MMF latch arena file. Peer processes pass this
    /// to [`MmfLatchArena::open`] to publish results.
    pub fn mmf_latches_path(&self) -> &std::path::Path {
        &self.mmf_latches_path
    }

    /// Return a [`Stealer`] that other in-process threads can use
    /// to steal from this worker's in-heap deque. (Crossbeam's
    /// idiomatic stealer-handle pattern.) Crate-internal because
    /// `JobRef` is itself `pub(crate)`.
    #[allow(
        dead_code,
        reason = "Exercised by in-module tests + reserved for the production worker-loop integration where sibling threads need a Stealer<JobRef> to peer-steal in-heap jobs."
    )]
    pub(crate) fn in_heap_stealer(&self) -> Stealer<JobRef> {
        self.in_heap.stealer()
    }

    /// Borrow the MMF deque so peer in-process threads can call
    /// [`MmfChaseLevDeque::steal`] on it. Cross-process thieves do
    /// NOT use this method - they open the deque file by path.
    pub fn mmf_deque(&self) -> &MmfChaseLevDeque {
        &self.mmf
    }

    /// Borrow the MMF latch arena, e.g. so external originators can
    /// allocate cells for in-flight dispatches.
    pub fn mmf_latches(&self) -> &MmfLatchArena {
        &self.latches
    }

    /// Owner-side push for `!Marshal` work. Goes to the in-heap
    /// Chase-Lev. Same shape as crate-local
    /// [`crate::sched::chase_lev_local::Worker::push`] but
    /// infallible at this level (overflow drops the job).
    /// Crate-internal because `JobRef` is itself `pub(crate)`.
    #[allow(
        dead_code,
        reason = "Exercised by in-module tests + reserved for the production worker-loop integration where the in-heap deque receives !Marshal jobs from the join entry point."
    )]
    pub(crate) fn push_local(&self, job: JobRef) {
        // chase_lev_local::Worker::push is bounded. On overflow
        // drop the job to keep the push-local API infallible -
        // any production wiring must size the deque to cover the
        // worst-case burst from its dispatcher (1024 slots ample).
        if self.in_heap.push(job).is_err() {
            // Best-effort: signal via stderr but don't panic.
            eprintln!("flynnel::sched::dual_deque: in-heap deque overflow; job dropped");
        }
    }

    /// Owner-side push for Marshal-shaped work. Allocates a latch
    /// cell, builds the slot, pushes the slot onto the MMF deque,
    /// returns the latch offset for the caller to wait on later.
    pub fn push_marshal(
        &self,
        closure_id: u32,
        args: &[u8],
    ) -> Result<u32, PushMarshalError> {
        if args.len() > chase_lev_mmf::ARGS_INLINE_BYTES {
            return Err(PushMarshalError::PayloadTooLarge);
        }
        let latch_offset = self.latches.alloc();
        let slot = RemoteJobSlot::new(closure_id, latch_offset, args)?;
        self.mmf.push(slot)?;
        Ok(latch_offset)
    }

    /// Owner-side pop. Tries MMF first to keep cross-process
    /// consumers fed, then falls back to in-heap LIFO.
    /// Crate-internal because `LocalPop` carries a `JobRef`.
    #[allow(
        dead_code,
        reason = "Exercised by in-module tests + reserved for the production worker-loop integration; this is the canonical owner-side pop the worker loop will call."
    )]
    pub(crate) fn pop_local(&self) -> LocalPop {
        match self.mmf.pop() {
            MmfSteal::Success(slot) => return LocalPop::Mmf(slot),
            MmfSteal::Empty | MmfSteal::Retry => {}
        }
        match self.in_heap.pop() {
            CbSteal::Success(job) => LocalPop::InHeap(job),
            CbSteal::Empty | CbSteal::Retry => LocalPop::Empty,
        }
    }

    /// Owner-side Marshal-only pop. Returns `Some(slot)` if the
    /// MMF deque had work, `None` otherwise. Public surface for
    /// callers that only deal in Marshal jobs (the in-heap deque
    /// is ignored).
    pub fn pop_marshal(&self) -> Option<RemoteJobSlot> {
        match self.mmf.pop() {
            MmfSteal::Success(slot) => Some(slot),
            MmfSteal::Empty | MmfSteal::Retry => None,
        }
    }

    /// Thief-side Marshal-only steal. Returns `Some(slot)` if a
    /// Marshal slot was stolen, `None` on empty or CAS-loss (the
    /// caller's outer steal loop should re-call on Retry). Public
    /// surface for callers that only deal in Marshal jobs.
    pub fn steal_marshal(&self) -> Option<RemoteJobSlot> {
        match self.mmf.steal() {
            MmfSteal::Success(slot) => Some(slot),
            MmfSteal::Empty | MmfSteal::Retry => None,
        }
    }

    /// Thief-side steal. Tries the MMF deque first, then the
    /// in-heap deque via the caller-supplied stealer handle
    /// (which the caller obtained from `self.in_heap_stealer()`). Crate-internal because
    /// `StealAny` carries a `JobRef`.
    #[allow(
        dead_code,
        reason = "Exercised by in-module tests + reserved for the production worker-loop integration where sibling threads call this on a peer's dual deque to acquire either MMF or in-heap work."
    )]
    pub(crate) fn steal_any(&self, in_heap_stealer: &Stealer<JobRef>) -> StealAny {
        match self.mmf.steal() {
            MmfSteal::Success(slot) => return StealAny::Mmf(slot),
            MmfSteal::Retry => return StealAny::Retry,
            MmfSteal::Empty => {}
        }
        match in_heap_stealer.steal() {
            CbSteal::Success(job) => StealAny::InHeap(job),
            CbSteal::Empty => StealAny::Empty,
            CbSteal::Retry => StealAny::Retry,
        }
    }

    /// Convenience: total observed size across both deques. For
    /// debug / scaling decisions; not safe to use for scheduling
    /// invariants (each load is independent so the pair is not a
    /// linearizable snapshot).
    pub fn total_observed_size(&self) -> usize {
        let in_heap_len = self.in_heap.len();
        let (_, _, mmf_size) = self.mmf.snapshot_size();
        in_heap_len + mmf_size.max(0) as usize
    }
}

impl Drop for DualDeque {
    fn drop(&mut self) {
        // Advance the MMF deque's epoch + zero the owner pid so any
        // peer process still attached can observe abandonment.
        self.mmf.close_owner();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::shared_mem::pass_registry::{hash_name, register, unregister};
    use crate::foundation::Variant;
    use crate::sched::job::{NUMA_HINT_ANY, StackJob};
    use crate::sched::latch::CoreLatch;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU32, Ordering};

    fn temp_paths(label: &str) -> (PathBuf, PathBuf) {
        let mut d = std::env::temp_dir();
        let mut l = std::env::temp_dir();
        let pid = std::process::id();
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        d.push(format!("flynnel_dual_d_{pid}_{nonce}_{label}.bin"));
        l.push(format!("flynnel_dual_l_{pid}_{nonce}_{label}.bin"));
        (d, l)
    }

    #[test]
    fn push_local_pops_in_heap_via_pop_local() {
        let (d, l) = temp_paths("push_local");
        let dq = DualDeque::create(&d, &l, 4, 4).expect("create");

        let counter = Arc::new(AtomicU32::new(0));
        let c = Arc::clone(&counter);
        let job = StackJob::new(
            move |_stolen| { c.fetch_add(42, Ordering::Relaxed); },
            CoreLatch::new(),
        );
        // SAFETY: StackJob lives until end of test scope; the
        // JobRef we make is consumed via execute() before the
        // StackJob drops.
        unsafe {
            let jr = job.as_job_ref(0, NUMA_HINT_ANY, Variant::Faithful);
            dq.push_local(jr);
        }

        // MMF is empty so pop_local should return the in-heap job.
        match dq.pop_local() {
            LocalPop::InHeap(jr) => {
                // SAFETY: the jr we just popped was the one we pushed.
                unsafe { jr.execute(); }
                assert_eq!(counter.load(Ordering::Relaxed), 42);
            }
            other => panic!("expected InHeap, got {other:?}"),
        }
        assert!(matches!(dq.pop_local(), LocalPop::Empty));
        std::fs::remove_file(&d).ok();
        std::fs::remove_file(&l).ok();
    }

    #[test]
    fn push_marshal_pops_mmf_via_pop_local() {
        let (d, l) = temp_paths("push_marshal");
        let dq = DualDeque::create(&d, &l, 4, 4).expect("create");

        let payload = b"hello";
        let latch_off = dq.push_marshal(0xDEAD_BEEF, payload).expect("push_marshal");

        // MMF preferred: pop_local returns the marshal slot first.
        match dq.pop_local() {
            LocalPop::Mmf(slot) => {
                assert_eq!(slot.closure_id, 0xDEAD_BEEF);
                assert_eq!(slot.latch_offset, latch_off);
                assert_eq!(slot.args(), payload);
            }
            other => panic!("expected Mmf, got {other:?}"),
        }
        assert!(matches!(dq.pop_local(), LocalPop::Empty));
        std::fs::remove_file(&d).ok();
        std::fs::remove_file(&l).ok();
    }

    #[test]
    fn pop_local_prefers_mmf_when_both_have_work() {
        let (d, l) = temp_paths("mmf_first");
        let dq = DualDeque::create(&d, &l, 4, 4).expect("create");

        let counter = Arc::new(AtomicU32::new(0));
        let c = Arc::clone(&counter);
        let job = StackJob::new(
            move |_| { c.fetch_add(1, Ordering::Relaxed); },
            CoreLatch::new(),
        );
        // SAFETY: StackJob outlives the JobRef; the JobRef is
        // executed in this test body before the StackJob drops.
        unsafe {
            dq.push_local(job.as_job_ref(0, NUMA_HINT_ANY, Variant::Faithful));
        }
        dq.push_marshal(0xCAFE, b"mmf-first").expect("push_marshal");

        // MMF should drain first.
        match dq.pop_local() {
            LocalPop::Mmf(slot) => assert_eq!(slot.closure_id, 0xCAFE),
            other => panic!("expected Mmf first, got {other:?}"),
        }
        // Then in-heap.
        match dq.pop_local() {
            LocalPop::InHeap(jr) => {
                // SAFETY: the jr we just popped was the one we pushed
                // immediately above (one job in the in-heap deque).
                unsafe { jr.execute(); }
                assert_eq!(counter.load(Ordering::Relaxed), 1);
            }
            other => panic!("expected InHeap second, got {other:?}"),
        }
        std::fs::remove_file(&d).ok();
        std::fs::remove_file(&l).ok();
    }

    #[test]
    fn steal_any_prefers_mmf_when_both_have_work() {
        let (d, l) = temp_paths("steal_mmf_first");
        let dq = DualDeque::create(&d, &l, 4, 4).expect("create");
        let stealer = dq.in_heap_stealer();

        let counter = Arc::new(AtomicU32::new(0));
        let c = Arc::clone(&counter);
        let job = StackJob::new(
            move |_| { c.fetch_add(1, Ordering::Relaxed); },
            CoreLatch::new(),
        );
        // SAFETY: StackJob outlives the JobRef; we either execute
        // the JobRef or the test panics. Both paths complete inside
        // this test scope.
        unsafe {
            dq.push_local(job.as_job_ref(0, NUMA_HINT_ANY, Variant::Faithful));
        }
        dq.push_marshal(0xFAFA, b"mmf-first-steal").expect("push_marshal");

        match dq.steal_any(&stealer) {
            StealAny::Mmf(slot) => assert_eq!(slot.closure_id, 0xFAFA),
            other => panic!("expected Mmf first via steal, got {other:?}"),
        }
        // Drain the in-heap side via successive steal/empty until
        // we either get the JobRef or hit Empty (crossbeam can
        // return Retry on race; loop bounded).
        let mut got_in_heap = false;
        for _ in 0..16 {
            match dq.steal_any(&stealer) {
                StealAny::InHeap(jr) => {
                    // SAFETY: jr was popped from the in-heap deque,
                    // which we know holds exactly the one job we
                    // pushed above. Execute and stop the loop.
                    unsafe { jr.execute(); }
                    got_in_heap = true;
                    break;
                }
                StealAny::Empty => break,
                StealAny::Retry => continue,
                other => panic!("unexpected: {other:?}"),
            }
        }
        assert!(got_in_heap, "should have stolen the in-heap job after the mmf job");
        assert_eq!(counter.load(Ordering::Relaxed), 1);

        std::fs::remove_file(&d).ok();
        std::fs::remove_file(&l).ok();
    }

    #[test]
    fn push_marshal_oversize_rejected() {
        let (d, l) = temp_paths("oversize_marshal");
        let dq = DualDeque::create(&d, &l, 2, 2).expect("create");
        let big = vec![0u8; chase_lev_mmf::ARGS_INLINE_BYTES + 1];
        let err = dq.push_marshal(0, &big).expect_err("expected oversize");
        assert_eq!(err, PushMarshalError::PayloadTooLarge);
        std::fs::remove_file(&d).ok();
        std::fs::remove_file(&l).ok();
    }

    #[test]
    fn marshal_latch_round_trips_via_dual_deque() {
        // Full owner-side cycle: push_marshal, pop_local (Mmf branch),
        // run the registered handler manually, publish result into
        // the latch, then verify the latch sees SET.
        let (d, l) = temp_paths("marshal_latch_round_trip");
        let dq = DualDeque::create(&d, &l, 4, 4).expect("create");

        let id = hash_name("flynnel.test.dual_deque.echo");
        register(id, |args| Ok(args.to_vec()));

        let latch_off = dq.push_marshal(id, b"echo-me").expect("push_marshal");
        match dq.pop_local() {
            LocalPop::Mmf(slot) => {
                let pass = crate::backend::shared_mem::pass_registry::Pass {
                    closure_id: slot.closure_id,
                    args: slot.args().to_vec(),
                };
                let reply = crate::backend::shared_mem::pass_registry::execute(&pass).expect("exec");
                dq.mmf_latches().publish(latch_off, &reply).expect("publish");
            }
            other => panic!("expected Mmf, got {other:?}"),
        }
        // Originator side: latch should be set; read it back.
        assert!(dq.mmf_latches().is_set(latch_off).expect("is_set"));
        let mut buf = Vec::new();
        let state = dq.mmf_latches().read_result(latch_off, &mut buf).expect("read");
        assert_eq!(state, crate::backend::shared_mem::latch_mmf::SET);
        assert_eq!(&buf[..], b"echo-me");

        unregister(id);
        std::fs::remove_file(&d).ok();
        std::fs::remove_file(&l).ok();
    }
}
