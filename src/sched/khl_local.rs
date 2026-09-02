//! KHL: K_inner=3 deque with per-slot Vyukov publication.
//!
//! Per-slot K_gating variant of the K_inner=3 family. Where Fcl
//! ([`crate::sched::fcl_local`]) uses a single `bottom` counter for
//! both ordering and publication (Chase-Lev family), KHL distributes
//! publication contention across an array of per-slot `seq` atomics:
//!
//! - **Counter-only gating (Fcl, Chase-Lev)**: every thief loads the
//!   same `bottom` cache line to learn what is publishable. All
//!   coherence traffic concentrates on one line.
//! - **Per-slot gating (KHL)**: thieves load `slot[t].seq` (a
//!   different line per slot index) to learn that slot's
//!   publication state. Coherence traffic spreads across many lines,
//!   exploiting store-buffer parallelism on the producer side.
//!
//! On store-buffer-rich cores (Zen+, Sapphire Rapids) the per-slot
//! pattern wins on producer-fast K=64 (measured 3.0x faster than
//! Chase-Lev K=1). On smaller-store-buffer cores (in-order ARM,
//! embedded) counter-only typically wins.
//!
//! ## Protocol
//!
//! Per-slot Vyukov MPMC discipline (producer single, consumer race
//! via head CAS):
//!
//! ```text
//! Owner publish (b = bottom counter; owner-private):
//!   slot = buffer[b mod cap]
//!   spin until slot.seq.load(Acquire) == b      // last round released
//!   write slot.body                              // items + metadata
//!   slot.seq.store(b + 1, Release)               // PUBLISH
//!   bottom.store(b + 1, Relaxed)                 // emptiness hint
//!
//! Thief steal:
//!   t = head.load(Acquire)
//!   b = bottom.load(Acquire)
//!   if t >= b: return Empty                      // emptiness hint
//!   if head.cas(t, t+1, Acquire, Relaxed) fails: return Retry
//!   spin until slot[t].seq.load(Acquire) == t + 1   // wait for publish
//!   items = read slot[t].body
//!   slot[t].seq.store(t + capacity, Release)     // RELEASE for next round
//!   return Success(items)
//! ```
//!
//! The `bottom` Acquire-load on the thief side is a hint, not a
//! synchronization point - the per-slot `seq` is the true
//! publication signal. The `bottom` load lets thieves bail out
//! cheaply when the deque is empty without paying the head CAS.
//!
//! ## Slot layout (exactly 64 bytes)
//!
//! ```text
//! +----------------+--+--+--+--+--+--+--+--+----------------+
//! | seq (8 bytes)  |  n_items + k_outer + numa_hint +       |
//! |                |  variant + 4-byte pad (8 total)         |
//! |                |  + items: [CompactJobRef; 3] (48 bytes) |
//! +----------------+-----------------------------------------+
//! ```

#![allow(clippy::missing_errors_doc)]

use core::cell::UnsafeCell;
use core::sync::atomic::{AtomicI64, AtomicU64, Ordering};
use std::sync::Arc;

use crate::sched::job::{CompactJobRef, JobRef};

/// Items per KHL slot. 3 matches Fcl's cache-line-fit ratio.
pub const KHL_LINE_ITEMS: usize = 3;

/// Slot body: shared metadata + 3 [`CompactJobRef`] payloads. 56
/// bytes. Wrapped inside [`KhlSlot`] under an [`UnsafeCell`]; the
/// `seq` protocol gates access (only the owner writes when
/// `seq == owner_index`; only the claiming thief reads when
/// `seq == consumer_index + 1`).
#[repr(C, align(8))]
pub struct KhlBody {
    /// Filled count for this batch (1..=3). Slots beyond this index
    /// are unspecified.
    pub n_items: u8,
    /// Shared `K_outer` metadata.
    pub k_outer: u8,
    /// Shared NUMA hint.
    pub numa_hint: u8,
    /// Shared variant tag.
    pub variant: u8,
    pad: [u8; 4],
    /// Compact-ref payload. `items[..n_items]` are valid.
    items: [CompactJobRef; KHL_LINE_ITEMS],
}

impl KhlBody {
    /// Construct an empty body with the named shared metadata.
    #[inline]
    pub fn empty(k_outer: u8, numa_hint: u8, variant: u8) -> Self {
        Self {
            n_items: 0,
            k_outer,
            numa_hint,
            variant,
            pad: [0u8; 4],
            items: [
                CompactJobRef::null(),
                CompactJobRef::null(),
                CompactJobRef::null(),
            ],
        }
    }

    /// Crate-internal accessor for the underlying compact-ref
    /// array. Used by the [`super::khl_worker::KhlStash`]
    /// adapter to drain stash items without going through the
    /// LIFO-execute path. Returns the FULL backing array including
    /// any padding slots beyond `n_items`; callers must respect
    /// `n_items` to avoid touching null padding slots.
    #[inline]
    pub(crate) fn items_for_adapter(&self) -> &[CompactJobRef; KHL_LINE_ITEMS] {
        &self.items
    }

    /// Execute every item in LIFO order. Caller must honor the
    /// once-only contract; the body must come from a successful
    /// `pop` or `steal`.
    ///
    /// # Safety
    ///
    /// Each item's captured-state pointer must still be valid;
    /// the body must not have been executed already.
    #[inline]
    pub unsafe fn execute_all_lifo(self) {
        for i in (0..self.n_items as usize).rev() {
            // SAFETY: items[i] is in the active prefix; the
            // once-only contract on KhlBody says execute_all_lifo
            // is the single execution path.
            unsafe { self.items[i].execute() }
        }
    }
}

impl core::fmt::Debug for KhlBody {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("KhlBody")
            .field("n_items", &self.n_items)
            .field("k_outer", &self.k_outer)
            .finish_non_exhaustive()
    }
}

/// One KHL slot. 64 bytes: 8B seq atomic + 56B body (UnsafeCell).
#[repr(C, align(64))]
struct KhlSlot {
    /// Vyukov publication sequence. Even slot value `b`: ready for
    /// owner to fill (last round released). Value `b + 1`: ready
    /// for thief to consume. Value `b + capacity`: released by
    /// thief, ready for next round (`b + capacity`).
    seq: AtomicU64,
    body: UnsafeCell<KhlBody>,
}

// SAFETY: the seq protocol gates body access (only one party at a
// time). Send + Sync follow from CompactJobRef's Send + Sync and
// the disciplined synchronization through seq.
unsafe impl Send for KhlSlot {}
unsafe impl Sync for KhlSlot {}

/// Header + slot buffer. Owner-private `bottom`; shared `head` that
/// thieves CAS to claim slot indices.
#[repr(C, align(64))]
struct KhlHeader {
    /// Owner-private write counter. Owner stores Relaxed; thieves
    /// Acquire-load as an emptiness hint.
    bottom: AtomicI64,
    _pad_bottom: [u8; 56],
    /// Thief-side claim counter. Thieves CAS this to take a slot.
    head: AtomicI64,
    _pad_head: [u8; 56],
    capacity: usize,
    capacity_mask: i64,
    buffer: Box<[KhlSlot]>,
}

// SAFETY: the per-atomic ordering discipline gates concurrent
// access. Send + Sync because all fields are individually safe.
unsafe impl Send for KhlHeader {}
unsafe impl Sync for KhlHeader {}

/// KHL deque: owner half + accumulator. Single-owner-writer.
pub struct SchedKhlDeque {
    inner: Arc<KhlHeader>,
    /// Owner-side accumulator buffer (1..=3 items before flush).
    /// Single-threaded access; UnsafeCell sufficient.
    accumulator: UnsafeCell<KhlBody>,
}

// SAFETY: only the owner thread touches `accumulator`; the inner
// Arc<KhlHeader> handles its own thread-safety via the seq protocol.
unsafe impl Send for SchedKhlDeque {}

/// Thief handle for [`SchedKhlDeque`]. Clonable.
pub struct KhlStealer {
    inner: Arc<KhlHeader>,
}

impl Clone for KhlStealer {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

/// Outcome of [`SchedKhlDeque::pop`] / [`KhlStealer::steal`].
#[derive(Debug)]
pub enum KhlSteal {
    /// Got a batch.
    Success(KhlBody),
    /// Deque was empty (no race).
    Empty,
    /// Lost head CAS (steal only) or seq race (rare).
    Retry,
}

/// Construct a new KHL deque with `capacity` slots (rounded up to
/// next power of two; minimum 2). Returns owner + one thief handle.
pub fn new_khl(capacity: usize) -> (SchedKhlDeque, KhlStealer) {
    let capacity = capacity.max(2).next_power_of_two();
    let mut buf = Vec::with_capacity(capacity);
    for i in 0..capacity {
        buf.push(KhlSlot {
            // Initial seq = i: producer at index i will see
            // seq == i (matches `b == i` when b == i) and publish.
            // This bootstraps the protocol so the first round can
            // proceed without prior consumer release.
            seq: AtomicU64::new(i as u64),
            body: UnsafeCell::new(KhlBody::empty(0, 0, 0)),
        });
    }
    let inner = Arc::new(KhlHeader {
        bottom: AtomicI64::new(0),
        _pad_bottom: [0u8; 56],
        head: AtomicI64::new(0),
        _pad_head: [0u8; 56],
        capacity,
        capacity_mask: (capacity as i64) - 1,
        buffer: buf.into_boxed_slice(),
    });
    let owner = SchedKhlDeque {
        inner: Arc::clone(&inner),
        accumulator: UnsafeCell::new(KhlBody::empty(0, 0, 0)),
    };
    let stealer = KhlStealer { inner };
    (owner, stealer)
}

impl SchedKhlDeque {
    /// Buffer a `JobRef` into the owner accumulator. When the
    /// accumulator fills (`n_items == 3`) the body is flushed to
    /// the next slot. Spins briefly if the next slot has not yet
    /// been released by its previous consumer.
    #[inline(always)]
    pub fn push(&self, job: JobRef) {
        // SAFETY: owner-private accumulator; single-threaded access.
        let acc = unsafe { &mut *self.accumulator.get() };
        if acc.n_items == 0 {
            acc.k_outer = job.k_outer;
            acc.numa_hint = job.numa_hint;
            acc.variant = job.variant;
        }
        let idx = acc.n_items as usize;
        // SAFETY: idx < n_items + 1, and n_items < KHL_LINE_ITEMS
        // by the branch above (this code only runs when n_items < 3
        // pre-increment, since the auto-flush below resets to 0).
        unsafe { *acc.items.get_unchecked_mut(idx) = job.compact(); }
        acc.n_items += 1;
        if acc.n_items == KHL_LINE_ITEMS as u8 {
            let body = core::mem::replace(acc, KhlBody::empty(0, 0, 0));
            self.publish(body);
        }
    }

    /// Single-push fast path: publish a 1-item slot directly,
    /// bypassing the accumulator when it is empty. Saves one
    /// 56-byte body copy (accumulator -> slot via flush) for the
    /// hot single-push case (`sched::join` right-half push); the
    /// K_inner=3 burst path stays correct because non-empty
    /// accumulator falls through to the standard push+flush.
    ///
    /// Closes the Heavy/100k gap vs crossbeam where the bisection
    /// pattern is all single-push and the accumulator-then-flush
    /// double-write is pure overhead.
    #[inline(always)]
    pub fn push_one(&self, job: JobRef) {
        // SAFETY: owner-private accumulator; single-threaded.
        let acc_n = unsafe { (*self.accumulator.get()).n_items };
        if acc_n == 0 {
            // Direct publish: construct a 1-item body and skip
            // the accumulator entirely.
            let body = KhlBody {
                n_items: 1,
                k_outer: job.k_outer,
                numa_hint: job.numa_hint,
                variant: job.variant,
                pad: [0u8; 4],
                items: [job.compact(), CompactJobRef::null(), CompactJobRef::null()],
            };
            self.publish(body);
        } else {
            // Burst-mode in progress; append + flush.
            self.push(job);
            self.flush();
        }
    }

    /// Force-flush any buffered items to the underlying KHL.
    #[inline]
    pub fn flush(&self) {
        // SAFETY: owner-private accumulator.
        let acc = unsafe { &mut *self.accumulator.get() };
        if acc.n_items == 0 {
            return;
        }
        let body = core::mem::replace(acc, KhlBody::empty(0, 0, 0));
        self.publish(body);
    }

    /// Internal: publish a body to the next slot. Spins if the
    /// slot is still held by its previous consumer.
    #[inline]
    fn publish(&self, body: KhlBody) {
        let h = &*self.inner;
        let b = h.bottom.load(Ordering::Relaxed);
        // SAFETY: (b & capacity_mask) is always in [0, capacity).
        let slot = unsafe { h.buffer.get_unchecked((b & h.capacity_mask) as usize) };
        // PREFETCHW signals write-intent so the cache line is
        // requested in M state (modified) up front instead of
        // M-after-S (shared->modified upgrade) when we write below.
        // Saves the read-for-ownership round-trip when a thief's
        // recent body-read left the line in S state on this core.
        #[cfg(target_arch = "x86_64")]
        unsafe {
            core::arch::asm!(
                "prefetchw [{addr}]",
                addr = in(reg) slot.body.get(),
                options(nostack, preserves_flags),
            );
        }
        // Wait for slot to be released by its previous consumer.
        // For the first cap rounds (b < cap), seq == b initially
        // by construction so this loop exits immediately.
        //
        // Back-pressure mitigation: bounded spin, then yield. A
        // pure spin-loop monopolises this CPU; yielding gives the
        // OS a chance to schedule thieves on this core if they
        // are runnable but not yet on-cpu. Threshold 64 chosen
        // empirically: a thief's release sequence is typically
        // dozens of cycles after the prior consumer's body read,
        // so anything beyond 64 spin iterations is contention.
        let mut spins: u32 = 0;
        while slot.seq.load(Ordering::Acquire) != b as u64 {
            spins = spins.wrapping_add(1);
            if spins & 63 == 0 {
                // Yield without preempting if no other thread is
                // runnable on this core; otherwise hand off a
                // time slice to a thief.
                std::thread::yield_now();
            } else {
                std::hint::spin_loop();
            }
        }
        // SAFETY: seq invariant says we own the body for this round
        // (no consumer reads it until our Release-store of
        // seq = b + 1 below).
        unsafe {
            core::ptr::write(slot.body.get(), body);
        }
        slot.seq.store((b as u64) + 1, Ordering::Release);
        // Bottom advances (Relaxed - seq is the true publication
        // signal; bottom is just an emptiness hint for thieves).
        h.bottom.store(b + 1, Ordering::Relaxed);
    }

    /// Owner-side pop. Drains the accumulator first (LIFO across
    /// slots: most-recent push first), then falls through to the
    /// underlying inner deque via head CAS.
    pub fn pop(&self) -> KhlSteal {
        // SAFETY: owner-private accumulator.
        let acc = unsafe { &mut *self.accumulator.get() };
        if acc.n_items > 0 {
            let body = core::mem::replace(acc, KhlBody::empty(0, 0, 0));
            return KhlSteal::Success(body);
        }
        // Drain from inner. The owner uses the thief steal pattern
        // (head CAS) for inner consumption; works whether the owner
        // pops LIFO (b - 1) or FIFO (head). KHL is a Vyukov ring
        // so owner-FIFO via head is the natural fit.
        self.steal_inner()
    }

    #[inline]
    fn steal_inner(&self) -> KhlSteal {
        let h = &*self.inner;
        let t = h.head.load(Ordering::Acquire);
        let b = h.bottom.load(Ordering::Acquire);
        if t >= b {
            return KhlSteal::Empty;
        }
        if h.head
            .compare_exchange(t, t + 1, Ordering::Acquire, Ordering::Relaxed)
            .is_err()
        {
            return KhlSteal::Retry;
        }
        // SAFETY: (t & capacity_mask) is always in [0, capacity).
        let slot = unsafe { h.buffer.get_unchecked((t & h.capacity_mask) as usize) };
        // Wait for producer publish at this slot (seq == t + 1).
        // Same back-pressure mitigation as publish().
        let mut spins: u32 = 0;
        while slot.seq.load(Ordering::Acquire) != (t as u64) + 1 {
            spins = spins.wrapping_add(1);
            if spins & 63 == 0 {
                std::thread::yield_now();
            } else {
                std::hint::spin_loop();
            }
        }
        // SAFETY: we've claimed slot t; the seq invariant means
        // we are the sole reader of the body until our release
        // store below.
        let body = unsafe { core::ptr::read(slot.body.get()) };
        slot.seq.store(
            (t as u64) + (h.capacity as u64),
            Ordering::Release,
        );
        KhlSteal::Success(body)
    }

    /// Clone a fresh thief handle.
    #[inline]
    pub fn stealer(&self) -> KhlStealer {
        KhlStealer {
            inner: Arc::clone(&self.inner),
        }
    }

    /// Snapshot whether the inner ring is empty (head >= bottom).
    /// Owner-only read using Acquire loads. Hint only - concurrent
    /// thief CAS may invalidate immediately after return.
    #[inline]
    pub fn is_inner_empty(&self) -> bool {
        let h = &*self.inner;
        let t = h.head.load(Ordering::Acquire);
        let b = h.bottom.load(Ordering::Acquire);
        t >= b
    }

    /// Owner-private accumulator's current item count (0..=3).
    /// Caller must be the owner thread.
    #[inline]
    pub fn accumulator_n_items(&self) -> u8 {
        // SAFETY: owner-private accumulator; single-threaded read.
        unsafe { (*self.accumulator.get()).n_items }
    }
}

impl KhlStealer {
    /// Steal one slot's worth of work. Returns `Retry` on head CAS
    /// loss (another thief beat us).
    #[inline]
    pub fn steal(&self) -> KhlSteal {
        let h = &*self.inner;
        let t = h.head.load(Ordering::Acquire);
        let b = h.bottom.load(Ordering::Acquire);
        if t >= b {
            return KhlSteal::Empty;
        }
        if h.head
            .compare_exchange(t, t + 1, Ordering::Acquire, Ordering::Relaxed)
            .is_err()
        {
            return KhlSteal::Retry;
        }
        // SAFETY: (t & capacity_mask) always in [0, capacity).
        let slot = unsafe { h.buffer.get_unchecked((t & h.capacity_mask) as usize) };
        // Wait for producer publish. Bounded spin then yield to
        // cooperate with concurrent thieves and producers.
        let mut spins: u32 = 0;
        while slot.seq.load(Ordering::Acquire) != (t as u64) + 1 {
            spins = spins.wrapping_add(1);
            if spins & 63 == 0 {
                std::thread::yield_now();
            } else {
                std::hint::spin_loop();
            }
        }
        // SAFETY: same as steal_inner; seq invariant gates body
        // access.
        let body = unsafe { core::ptr::read(slot.body.get()) };
        slot.seq.store(
            (t as u64) + (h.capacity as u64),
            Ordering::Release,
        );
        KhlSteal::Success(body)
    }

    /// Architectural prefetch hint for the next steal target.
    /// Touches the next slot's body line so the next steal's body
    /// read is L1d-warm.
    #[inline]
    pub fn prefetch_for_steal(&self) {
        let h = &*self.inner;
        let t = h.head.load(Ordering::Relaxed);
        let slot = &h.buffer[(t & h.capacity_mask) as usize];
        let body_ptr = slot.body.get() as *const u8;
        #[cfg(target_arch = "x86_64")]
        {
            // SAFETY: _mm_prefetch is a no-side-effect hint that
            // accepts any pointer without fault.
            unsafe {
                std::arch::x86_64::_mm_prefetch(
                    body_ptr as *const i8,
                    std::arch::x86_64::_MM_HINT_T0,
                );
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            std::hint::black_box(body_ptr);
        }
    }

    /// Approximate is-empty snapshot. Hint only; concurrent
    /// activity may invalidate immediately.
    #[inline]
    pub fn is_empty(&self) -> bool {
        let h = &*self.inner;
        let t = h.head.load(Ordering::Acquire);
        let b = h.bottom.load(Ordering::Acquire);
        t >= b
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::foundation::Variant;
    use crate::sched::job::StackJob;
    use crate::sched::latch::CoreLatch;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering as O};
    use std::thread;

    #[test]
    fn slot_is_exactly_one_cache_line() {
        assert_eq!(core::mem::size_of::<KhlSlot>(), 64,
            "KhlSlot must be exactly 64 bytes (one cache line)");
        assert_eq!(core::mem::align_of::<KhlSlot>(), 64);
    }

    #[test]
    fn body_round_trip_single_batch() {
        let (deque, stealer) = new_khl(4);
        let j = StackJob::new(|_| 42u32, CoreLatch::new());
        let r = unsafe { j.as_job_ref(8, 0, Variant::Faithful) };
        deque.push(r);
        // Before flush, steal sees empty.
        assert!(matches!(stealer.steal(), KhlSteal::Empty));
        deque.flush();
        match stealer.steal() {
            KhlSteal::Success(b) => {
                assert_eq!(b.n_items, 1);
                unsafe { b.execute_all_lifo(); }
                assert!(j.latch.is_set());
                assert_eq!(unsafe { j.into_result() }, 42);
            }
            other => panic!("expected Success, got {other:?}"),
        }
    }

    #[test]
    fn batched_three_then_steal() {
        let (deque, stealer) = new_khl(4);
        let j1 = StackJob::new(|_| 1u32, CoreLatch::new());
        let j2 = StackJob::new(|_| 2u32, CoreLatch::new());
        let j3 = StackJob::new(|_| 3u32, CoreLatch::new());
        let r1 = unsafe { j1.as_job_ref(4, 0, Variant::Fast) };
        let r2 = unsafe { j2.as_job_ref(4, 0, Variant::Fast) };
        let r3 = unsafe { j3.as_job_ref(4, 0, Variant::Fast) };
        deque.push(r1);
        deque.push(r2);
        deque.push(r3);
        // Auto-flush at 3.
        match stealer.steal() {
            KhlSteal::Success(b) => {
                assert_eq!(b.n_items, 3);
                unsafe { b.execute_all_lifo(); }
                assert!(j1.latch.is_set() && j2.latch.is_set() && j3.latch.is_set());
            }
            other => panic!("expected Success, got {other:?}"),
        }
    }

    #[test]
    fn owner_pop_drains_accumulator() {
        let (deque, _stealer) = new_khl(4);
        let j1 = StackJob::new(|_| 10u32, CoreLatch::new());
        let r1 = unsafe { j1.as_job_ref(4, 0, Variant::Fast) };
        deque.push(r1);
        match deque.pop() {
            KhlSteal::Success(b) => {
                assert_eq!(b.n_items, 1);
                unsafe { b.execute_all_lifo(); }
                assert!(j1.latch.is_set());
            }
            other => panic!("expected Success, got {other:?}"),
        }
        assert!(matches!(deque.pop(), KhlSteal::Empty));
    }

    #[test]
    fn empty_steal_returns_empty() {
        let (_deque, stealer) = new_khl(4);
        assert!(matches!(stealer.steal(), KhlSteal::Empty));
    }

    #[test]
    fn concurrent_owner_publish_and_thieves_no_double_take() {
        // Stress: producer publishes N batches; multiple thieves
        // race to steal. Sum invariant verifies each item is
        // consumed exactly once.
        use std::sync::atomic::AtomicU32;

        // Each Job is a tiny captured-state struct: pointer to an
        // AtomicU32 counter; execute_fn fetch_adds 1.
        struct Counter {
            counter: Arc<AtomicU32>,
            value: u32,
        }
        unsafe fn exec_counter(p: *const ()) {
            // SAFETY: pointer was produced by Box::into_raw.
            let b = unsafe { Box::from_raw(p as *mut Counter) };
            b.counter.fetch_add(b.value, O::Relaxed);
        }

        let n = 600u32;  // multiple of 3 for clean batching
        let counter = Arc::new(AtomicU32::new(0));
        let (deque, s_proto) = new_khl(64);

        let consumed = Arc::new(AtomicUsize::new(0));
        let mut thieves = Vec::new();
        for _ in 0..4 {
            let s = s_proto.clone();
            let consumed = Arc::clone(&consumed);
            thieves.push(thread::spawn(move || {
                while consumed.load(O::Relaxed) < n as usize {
                    match s.steal() {
                        KhlSteal::Success(b) => {
                            let n_items = b.n_items as usize;
                            unsafe { b.execute_all_lifo(); }
                            consumed.fetch_add(n_items, O::Relaxed);
                        }
                        KhlSteal::Empty | KhlSteal::Retry => std::thread::yield_now(),
                    }
                }
            }));
        }

        let counter_p = Arc::clone(&counter);
        // Owner moves into producer thread - single-owner invariant.
        let producer = thread::spawn(move || {
            for i in 0..n {
                let boxed = Box::new(Counter { counter: Arc::clone(&counter_p), value: i });
                let p = Box::into_raw(boxed) as *const ();
                // SAFETY: pointer was produced by Box::into_raw with
                // matching exec_counter; the JobRef will execute
                // exactly once.
                let jref = unsafe { JobRef::from_raw_parts(p, exec_counter, 0, 0, 0) };
                deque.push(jref);
            }
            // Flush any partial accumulator.
            deque.flush();
        });

        producer.join().expect("producer joined");
        for h in thieves {
            h.join().expect("thief joined");
        }

        let expected: u32 = (0..n).sum();
        assert_eq!(counter.load(O::Relaxed), expected,
            "sum invariant: each job's value must be added exactly once");
        assert_eq!(consumed.load(O::Relaxed), n as usize);
    }
}
