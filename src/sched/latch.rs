//! 4-state core latch + `Latch` trait with self-invalidation discipline.
//!
//! Direct adaptation of rayon-core 1.13's `latch.rs` shape, reshaped
//! for Flynnel. The state machine and the `Latch::set(*const Self)`
//! contract are borrowed verbatim because the design has been
//! battle-hardened in rayon across a decade of edge cases.
//!
//! ## State machine
//!
//! ```text
//!   UNSET ----get_sleepy()---> SLEEPY ----fall_asleep()---> SLEEPING
//!     |                          |                              |
//!     |                          |                              |
//!     v                          v                              v
//!    SET (Latch::set called; publisher observed prior state)
//! ```
//!
//! Transitions are all CAS with SeqCst on success, Relaxed on failure.
//! `Latch::set` is a `swap(SET, AcqRel)` that returns the prior state;
//! the publisher uses the return value to decide whether to wake a
//! parked worker (only needed when prior was `SLEEPING`).
//!
//! ## The self-invalidation contract
//!
//! `Latch::set` takes `*const Self` rather than `&self` because the
//! publishing CAS may wake a thread that immediately deallocates the
//! latch (e.g., when a `StackJob` finishes and its parent frame
//! returns). Implementations MUST read every field they need BEFORE
//! the publishing store; `SpinLatch::set` does this by copying
//! `target_worker_index` to a local first, and `CountLatch` /
//! `LockLatch` follow the same discipline.
//!

use core::sync::atomic::{AtomicU8, Ordering};

/// Latch not set; owning thread is awake.
const UNSET: u8 = 0;
/// Latch not set; owning thread is preparing to sleep but has not
/// committed yet. The publisher can still observe this state and
/// abort the park.
const SLEEPY: u8 = 1;
/// Latch not set; owning thread is parked on a condvar elsewhere
/// (the latch only owns the lifecycle marker). The publisher MUST
/// wake the parked thread after setting.
const SLEEPING: u8 = 2;
/// Latch is set. Terminal state.
const SET: u8 = 3;

/// One-time signalling primitive. Starts at [`UNSET`]; transitions
/// monotonically to [`SET`].
///
/// See module-level docs for the state machine and the
/// `set(*const Self)` self-invalidation contract.
#[derive(Debug)]
pub struct CoreLatch {
    state: AtomicU8,
}

impl Default for CoreLatch {
    fn default() -> Self {
        Self::new()
    }
}

impl CoreLatch {
    /// Construct a fresh latch in the `UNSET` state.
    #[inline]
    pub const fn new() -> Self {
        Self {
            state: AtomicU8::new(UNSET),
        }
    }

    /// First phase of the sleep handshake. The owning thread calls
    /// this to declare intent to park. Returns `true` if the
    /// transition `UNSET -> SLEEPY` succeeded and the caller may
    /// proceed to [`Self::fall_asleep`]; returns `false` if the latch
    /// was already set in the meantime (caller should NOT park).
    #[inline]
    pub fn get_sleepy(&self) -> bool {
        self.state
            .compare_exchange(UNSET, SLEEPY, Ordering::SeqCst, Ordering::Relaxed)
            .is_ok()
    }

    /// Second phase of the sleep handshake. The owning thread calls
    /// this immediately before parking on its condvar. Returns `true`
    /// if the transition `SLEEPY -> SLEEPING` succeeded (caller may
    /// park); returns `false` if the latch was set after
    /// `get_sleepy` (caller must NOT park).
    #[inline]
    pub fn fall_asleep(&self) -> bool {
        self.state
            .compare_exchange(SLEEPY, SLEEPING, Ordering::SeqCst, Ordering::Relaxed)
            .is_ok()
    }

    /// Called by the owning thread after a wakeup that did NOT
    /// observe `SET` (e.g., a spurious condvar wake or the JEC-
    /// observed-injected-job path). Reverts `SLEEPING -> UNSET` so
    /// the thread can re-enter the work-search loop. No-op if the
    /// latch is already `SET`.
    #[inline]
    pub fn wake_up(&self) {
        if !self.is_set() {
            let _ = self.state.compare_exchange(
                SLEEPING,
                UNSET,
                Ordering::SeqCst,
                Ordering::Relaxed,
            );
        }
    }

    /// Test whether the latch has been set. Acquire-ordered so any
    /// effects published before [`Latch::set`] are visible.
    #[inline]
    pub fn is_set(&self) -> bool {
        self.state.load(Ordering::Acquire) == SET
    }

    /// Direct setter used by [`Latch::set`] impls. Returns `true` if
    /// the previous state was `SLEEPING` (publisher must wake the
    /// parked thread); `false` otherwise.
    ///
    /// # Safety
    ///
    /// Caller must ensure `this` points to a valid `CoreLatch` for
    /// the duration of the swap. After the swap returns, callers
    /// MUST NOT touch any field of `*this` except those captured by
    /// value BEFORE this call: a parked thread may wake and
    /// deallocate the latch as soon as it observes `SET`.
    #[inline]
    pub(crate) unsafe fn set(this: *const Self) -> bool {
        // SAFETY: the `# Safety` clause on this function shifts
        // the validity-of-`this` precondition onto the caller;
        // here we may deref it once for the publishing atomic
        // swap. We must NOT touch `*this` after the swap because
        // a parked thread observing SLEEPING -> SET is free to
        // deallocate the latch.
        let old = unsafe { (*this).state.swap(SET, Ordering::AcqRel) };
        old == SLEEPING
    }
}

/// Latches share the contract that exactly one [`Latch::set`] call
/// transitions the latch from "not set" to "set", at which point
/// every [`CoreLatch::is_set`] returns `true` (with happens-before
/// ordering of all effects sequenced before the set).
///
/// `set` takes `*const Self` rather than `&self` to allow the
/// pointed-to memory to become invalidated DURING the call (a
/// parked thread may wake and free the latch). See the module
/// docs for the discipline this imposes on implementations.
pub trait Latch {
    /// Set the latch. Implementations must read every field of
    /// `*this` they need BEFORE the publishing store; after that
    /// store, `*this` may be invalidated.
    ///
    /// # Safety
    ///
    /// The caller asserts that `this` points to a valid `Self`
    /// upon entry and that no other code path will invalidate it
    /// during this call except for actions triggered by `set`
    /// itself (i.e., a parked thread waking and deallocating).
    unsafe fn set(this: *const Self);
}

impl Latch for CoreLatch {
    #[inline]
    unsafe fn set(this: *const Self) {
        // Discard the "was sleeping" bit because plain CoreLatch has
        // no integrated wakeup mechanism. Wrappers (SpinLatch /
        // CountLatch / LockLatch) override `Latch::set` to wake
        // their associated worker.
        //
        // SAFETY: the trait method's `# Safety` clause forwards
        // the validity-of-`this` precondition to the caller;
        // inner `CoreLatch::set` honors the same contract.
        // @hook-allow:no-let-underscore
        let _ = unsafe { CoreLatch::set(this) };
    }
}

/// Wake-capable latch wrapper: holds a `CoreLatch` plus an
/// `Arc<Parker>` for the worker that is waiting on this latch. When
/// `Latch::set` fires AND the prior state was `SLEEPING`, the
/// publisher unparks the parker, releasing the parked worker.
///
/// This is the missing piece that lets the join wait loop in
/// `arena::join_in_worker` actually sleep on the latch instead of
/// hot-spinning on `yield_now()`. Without a wake-capable latch, the
/// wait-loop worker burns CPU continuously while the thief executes
/// its stolen right-half, starving productive CPU from peers and
/// the main thread.
///
/// Lifetime + invalidation: per the `Latch::set` `# Safety` clause,
/// `set` reads every field BEFORE the publishing CoreLatch store so
/// the latch may be deallocated by the awakened thread the moment
/// `is_set` returns true. The `parker.clone()` bumps the parker's
/// Arc refcount via a heap-allocated control block whose lifetime
/// is independent of the latch's stack frame.
pub struct SpinLatch {
    /// State machine underlying this wake-capable wrapper. Exposed
    /// so callers can route the existing `Latch::set` /
    /// `is_set` API through the inner core without duplicating
    /// every method.
    pub core: CoreLatch,
    /// Parker for the worker that is sleeping (or is about to
    /// sleep) on this latch. `Latch::set` clones this Arc BEFORE
    /// the publishing CoreLatch store so the parker remains valid
    /// even after the awakened worker deallocates the latch's
    /// stack frame.
    pub parker: std::sync::Arc<crate::sched::sleep::Parker>,
}

impl SpinLatch {
    /// Construct a fresh wake-capable latch attached to `parker`.
    #[inline]
    pub fn new(parker: std::sync::Arc<crate::sched::sleep::Parker>) -> Self {
        Self {
            core: CoreLatch::new(),
            parker,
        }
    }

    /// Forwarded test for is_set on the underlying CoreLatch.
    #[inline]
    pub fn is_set(&self) -> bool {
        self.core.is_set()
    }

    /// Forwarded sleep handshake: first transition (`UNSET ->
    /// SLEEPY`).
    #[inline]
    pub fn get_sleepy(&self) -> bool {
        self.core.get_sleepy()
    }

    /// Forwarded sleep handshake: second transition (`SLEEPY ->
    /// SLEEPING`).
    #[inline]
    pub fn fall_asleep(&self) -> bool {
        self.core.fall_asleep()
    }

    /// Forwarded post-wake reset (`SLEEPING -> UNSET`).
    #[inline]
    pub fn wake_up(&self) {
        self.core.wake_up();
    }
}

impl Latch for SpinLatch {
    #[inline]
    unsafe fn set(this: *const Self) {
        // SAFETY: read the parker Arc BEFORE the publishing
        // CoreLatch::set store. The parker Arc lives in a heap-
        // allocated control block whose lifetime is independent
        // of the latch's owning stack frame, so the cloned Arc
        // remains valid even after the parked worker observes
        // `SET` and deallocates `*this`.
        let parker = unsafe { (*this).parker.clone() };
        // SAFETY: trait method's `# Safety` clause forwards the
        // validity-of-`this` precondition; inner CoreLatch::set
        // honors the publish-then-invalidate contract.
        let was_sleeping = unsafe { CoreLatch::set(&(*this).core) };
        if was_sleeping {
            // Worker was parked on its Parker waiting for this
            // latch; wake it so the wait loop's predicate observes
            // `is_set == true` and returns from the join.
            parker.unpark();
        }
    }
}

/// N-participant wake-capable latch. The inner `CoreLatch` only
/// transitions to `SET` when a counter reaches zero; each
/// `Latch::set` call decrements the counter. The publisher whose
/// decrement reaches zero (and only that publisher) is responsible
/// for unparking the parked waiter.
///
/// Designed for the SIMC `cooperative_join_n` cooperative-vector
/// dispatch: N participating workers each call `Latch::set` when
/// they complete their slice; the owner that submitted the
/// cooperative dispatch is the parked waiter. Without an integrated
/// count, callers would track the N-completion gate manually with
/// an external AtomicUsize and a separate condvar pair, which the
/// existing cooperative.rs path does. This struct centralizes the
/// pattern so the existing sites can adopt it.
///
/// Lifetime + invalidation: the parker Arc is cloned BEFORE the
/// CoreLatch publishing store on the final decrement, same
/// discipline as SpinLatch.
pub struct CountLatch {
    /// Outstanding count of `Latch::set` calls remaining before
    /// the inner CoreLatch transitions to SET. Decremented by each
    /// publisher; the decrementer that observes the new value 0
    /// performs the CoreLatch swap + unpark.
    count: std::sync::atomic::AtomicUsize,
    /// State machine underlying this counted wrapper.
    core: CoreLatch,
    /// Parker for the worker awaiting the all-set transition.
    /// Cloned before publish, same as SpinLatch.
    parker: std::sync::Arc<crate::sched::sleep::Parker>,
}

impl CountLatch {
    /// Construct a fresh count latch for `n` participants.
    /// Panics if `n == 0` (a latch that's already-set from
    /// construction has no callers).
    #[inline]
    pub fn new(n: usize, parker: std::sync::Arc<crate::sched::sleep::Parker>) -> Self {
        assert!(n > 0, "CountLatch requires at least 1 participant");
        Self {
            count: std::sync::atomic::AtomicUsize::new(n),
            core: CoreLatch::new(),
            parker,
        }
    }

    /// Test whether the latch has transitioned to SET (all N
    /// participants have called Latch::set).
    #[inline]
    pub fn is_set(&self) -> bool {
        self.core.is_set()
    }

    /// Sleep handshake first transition (UNSET -> SLEEPY).
    /// Mirrors SpinLatch; forwarded to the inner CoreLatch.
    #[inline]
    pub fn get_sleepy(&self) -> bool {
        self.core.get_sleepy()
    }

    /// Sleep handshake second transition (SLEEPY -> SLEEPING).
    /// Mirrors SpinLatch; forwarded to the inner CoreLatch.
    #[inline]
    pub fn fall_asleep(&self) -> bool {
        self.core.fall_asleep()
    }

    /// Post-wake reset (SLEEPING -> UNSET). Mirrors SpinLatch;
    /// forwarded to the inner CoreLatch.
    #[inline]
    pub fn wake_up(&self) {
        self.core.wake_up();
    }

    /// Read the outstanding count (debug / observability).
    #[inline]
    pub fn outstanding(&self) -> usize {
        self.count.load(std::sync::atomic::Ordering::Acquire)
    }
}

impl Latch for CountLatch {
    #[inline]
    unsafe fn set(this: *const Self) {
        // SAFETY: read both the count cell and the parker BEFORE
        // the final-decrement publishing CoreLatch store. After
        // that store, *this is allowed to be deallocated by the
        // awakened waiter.
        let prior = unsafe {
            (*this).count.fetch_sub(1, std::sync::atomic::Ordering::AcqRel)
        };
        debug_assert!(prior > 0, "CountLatch::set called more times than the construction count");
        if prior > 1 {
            // Not the last decrementer; nothing more to publish.
            return;
        }
        // We are the final decrementer. Clone the parker BEFORE
        // the CoreLatch swap so we still have a valid reference
        // even after the awakened waiter deallocates *this.
        let parker = unsafe { (*this).parker.clone() };
        // SAFETY: forwarded by the trait method's contract; inner
        // CoreLatch::set honors the publish-then-invalidate
        // discipline.
        let was_sleeping = unsafe { CoreLatch::set(&(*this).core) };
        if was_sleeping {
            parker.unpark();
        }
    }
}

/// Cross-thread mutex-free wake-capable latch. Used when the
/// waiter is OUTSIDE the worker pool (the main thread calling
/// `arena::join` from outside, or a foreign-language caller that
/// doesn't own a Parker). The waiter calls `wait()` which uses
/// `thread::park` to block on the kernel-native futex without
/// going through Mutex+Condvar; the setter calls `unpark` on the
/// stored Thread handle after publishing the AtomicBool flag.
///
/// Avoids a Mutex+Condvar pairing: each Mutex lock/unlock on
/// Linux costs an uncontended atomic CAS (fast) plus, on
/// contention, a futex syscall, and each Condvar wait/notify is a
/// futex syscall. The mutex-free design uses just one atomic
/// store + one unpark syscall on the setter side and one atomic
/// load + one park syscall on the waiter side - half the syscalls
/// per side compared to the mutex+condvar pattern. Mirrors
/// rayon's `LockLatch` lock-free shape.
pub struct LockLatch {
    /// Fast-path state. The waiter checks this before parking;
    /// the setter publishes Release-true BEFORE the unpark so a
    /// post-publish waiter observes true on its is_set check and
    /// skips the park syscall entirely.
    flag: std::sync::atomic::AtomicBool,
    /// Handle of the parked waiter thread, published exactly
    /// once by `wait` BEFORE parking. The setter clones it out
    /// before the publishing store on `flag` so the parker
    /// remains valid even if the awakened waiter deallocates
    /// `*this`.
    waiter: std::sync::OnceLock<std::thread::Thread>,
}

impl Default for LockLatch {
    fn default() -> Self {
        Self::new()
    }
}

impl LockLatch {
    /// Construct a fresh cross-thread latch in the UNSET state.
    #[inline]
    pub fn new() -> Self {
        Self {
            flag: std::sync::atomic::AtomicBool::new(false),
            waiter: std::sync::OnceLock::new(),
        }
    }

    /// Block the calling thread until `Latch::set` is called.
    /// Idempotent and re-entrant: returns immediately if already
    /// set when called.
    pub fn wait(&self) {
        if locklatch_diagnose_enabled() {
            eprintln!("[locklatch] wait enter tid={:?}", std::thread::current().id());
        }
        // Fast path: if the latch is already set, return without
        // touching the OnceLock or any park syscalls.
        if self.flag.load(std::sync::atomic::Ordering::Acquire) {
            if locklatch_diagnose_enabled() {
                eprintln!("[locklatch] wait exit (fast)  tid={:?}", std::thread::current().id());
            }
            return;
        }
        // Slow path: publish our Thread handle so the setter
        // can unpark us. The OnceLock can only be set once per
        // LockLatch lifetime, which matches the latch's one-
        // waiter contract. The error path on a duplicate set
        // means a second wait() call concurrent with the first -
        // not supported; LockLatch is a single-waiter primitive.
        // Discard returns Result<(), Thread> on duplicate set;
        // any wait() call after the first wait() of the same
        // LockLatch is a contract violation, but we tolerate it
        // by falling through to the park-retry loop below where
        // the flag re-check on every wake still produces correct
        // termination.
        let our_handle = std::thread::current();
        drop(self.waiter.set(our_handle));
        // Race-recovery re-check: the setter might have fired
        // between our fast-path check and our handle publish.
        // If so, the setter would observe no waiter handle and
        // not call unpark; we must observe the flag here so we
        // do not park forever.
        if self.flag.load(std::sync::atomic::Ordering::Acquire) {
            if locklatch_diagnose_enabled() {
                eprintln!("[locklatch] wait exit (race)  tid={:?}", std::thread::current().id());
            }
            return;
        }
        // Park-retry loop: thread::park can return spuriously, so
        // we re-check the flag on every wake. The setter's unpark
        // is delivered as a permit to the saved Thread handle, so
        // any number of pre-park unparks (e.g. from a race where
        // setter unparks before we reach park) are absorbed and
        // make the first park return immediately.
        while !self.flag.load(std::sync::atomic::Ordering::Acquire) {
            std::thread::park();
        }
        if locklatch_diagnose_enabled() {
            eprintln!("[locklatch] wait exit  tid={:?}", std::thread::current().id());
        }
    }

    /// Test whether the latch is SET without blocking. Single
    /// Acquire load on the fast-path AtomicBool; does NOT take
    /// the mutex. Callers spinning on `is_set` before falling
    /// through to `wait` pay no syscall per spin iteration.
    pub fn is_set(&self) -> bool {
        self.flag.load(std::sync::atomic::Ordering::Acquire)
    }
}

/// Diagnostic gate for LockLatch.wait() entry/exit logging. Reads
/// the env var once via OnceLock so the hot wait path pays just a
/// cached Relaxed load per call. Set FLYNNEL_LOCKLATCH_DIAGNOSE=1
/// to enable; default off.
fn locklatch_diagnose_enabled() -> bool {
    static CACHE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| {
        std::env::var("FLYNNEL_LOCKLATCH_DIAGNOSE")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    })
}

impl Latch for LockLatch {
    #[inline]
    unsafe fn set(this: *const Self) {
        // SAFETY: read the waiter handle BEFORE the publishing
        // flag.store. After the store, the waiter may observe
        // is_set=true (in its fast-path or race-recovery check)
        // and deallocate *this. The Thread handle is itself an
        // Arc<Inner> heap-allocated cell; cloning it bumps the
        // refcount on the inner before any race window opens, so
        // the handle remains valid for the post-store unpark even
        // after *this is invalidated.
        let waiter_opt = unsafe { (*this).waiter.get().cloned() };
        // Publishing store. Release ordering pairs with the
        // waiter's Acquire load on the flag.
        unsafe { (*this).flag.store(true, std::sync::atomic::Ordering::Release) };
        // *this may now be invalidated; use waiter_opt only.
        if let Some(handle) = waiter_opt {
            handle.unpark();
        }
        // If waiter_opt is None, the waiter hasn't yet published
        // its handle (race: waiter passed its fast-path check but
        // hasn't set the OnceLock yet). The waiter's post-publish
        // re-check of the flag will observe true and return
        // immediately without parking.
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::AtomicUsize;
    use std::thread;
    use std::time::{Duration, Instant};

    #[test]
    fn new_latch_is_unset() {
        let l = CoreLatch::new();
        assert!(!l.is_set());
    }

    #[test]
    fn set_marks_set() {
        let l = CoreLatch::new();
        unsafe { Latch::set(&l) };
        assert!(l.is_set());
    }

    #[test]
    fn spin_latch_wakes_parked_thread() {
        // E2E test of the SpinLatch wake-capable wrapper. The Parker
        // is constructed on the WAITER thread (captures
        // thread::current() at construction), then handed to the
        // SETTER thread via a channel along with the latch. This
        // mirrors the real flynnel usage where each worker's parker
        // is owned by that worker, and a thief on a different
        // thread calls Latch::set.
        //
        // Without the parker.unpark() in SpinLatch::set, the parked
        // waiter would hang until a spurious wake or shutdown.
        use std::sync::mpsc;
        let (tx, rx) = mpsc::channel::<Arc<SpinLatch>>();

        let waiter = thread::spawn(move || {
            // Waiter owns the Parker; capture its own thread handle
            // so SpinLatch::set's unpark targets THIS thread.
            let parker = Arc::new(crate::sched::sleep::Parker::new(0));
            let latch = Arc::new(SpinLatch::new(parker));
            // Hand a clone of the latch to the setter thread.
            tx.send(latch.clone()).unwrap();
            // Sleep handshake on this thread (the parker's owner).
            assert!(latch.get_sleepy(), "fresh latch must accept get_sleepy");
            assert!(!latch.is_set(), "no setter has run yet");
            assert!(
                latch.fall_asleep(),
                "no setter raced us between get_sleepy and fall_asleep"
            );
            // Park until latch set; predicate re-checks on wake.
            let _unparked: bool = latch.parker.park_until(|| latch.is_set());
            latch.wake_up();
            assert!(latch.is_set(), "must observe SET after wake");
        });

        // Setter on the test main thread: receive the latch and
        // publish SET after a delay long enough for the waiter to
        // reach the actual park syscall.
        let latch = rx.recv().expect("waiter must send latch");
        thread::sleep(Duration::from_millis(50));
        unsafe { Latch::set(&*latch) };

        // Waiter must complete within a generous bound; if it hangs
        // here the unpark from Latch::set never reached the waiter.
        let t0 = Instant::now();
        loop {
            if waiter.is_finished() {
                waiter.join().unwrap();
                break;
            }
            if t0.elapsed() > Duration::from_secs(5) {
                panic!("waiter did not wake within 5s after Latch::set");
            }
            thread::sleep(Duration::from_millis(10));
        }
    }

    #[test]
    fn count_latch_only_sets_on_final_decrement() {
        // Build a 3-participant count latch on a dummy parker.
        // First two decrements MUST leave the inner state unset;
        // third decrement transitions to SET.
        let parker = Arc::new(crate::sched::sleep::Parker::new(0));
        let latch = CountLatch::new(3, parker);
        assert_eq!(latch.outstanding(), 3);
        assert!(!latch.is_set());
        unsafe { Latch::set(&latch) };
        assert_eq!(latch.outstanding(), 2);
        assert!(!latch.is_set(), "must not set until final decrement");
        unsafe { Latch::set(&latch) };
        assert_eq!(latch.outstanding(), 1);
        assert!(!latch.is_set(), "must not set until final decrement");
        unsafe { Latch::set(&latch) };
        assert_eq!(latch.outstanding(), 0);
        assert!(latch.is_set(), "final decrement must transition to SET");
    }

    #[test]
    fn count_latch_wakes_parked_thread_on_final_set() {
        // E2E: N setters across multiple threads + one waiter on
        // its own Parker. Only the final decrement unparks.
        use std::sync::mpsc;
        let (tx, rx) = mpsc::channel::<Arc<CountLatch>>();
        let n = 4usize;

        let waiter = thread::spawn(move || {
            let parker = Arc::new(crate::sched::sleep::Parker::new(0));
            let latch = Arc::new(CountLatch::new(n, parker));
            tx.send(latch.clone()).unwrap();
            assert!(latch.get_sleepy());
            assert!(latch.fall_asleep());
            let _unparked: bool = latch.parker.park_until(|| latch.is_set());
            latch.wake_up();
            assert!(latch.is_set(), "must observe SET after final decrement");
        });

        let latch = rx.recv().expect("waiter sent latch");
        thread::sleep(Duration::from_millis(20));
        let mut setters = Vec::new();
        for _ in 0..n {
            let l2 = latch.clone();
            setters.push(thread::spawn(move || unsafe { Latch::set(&*l2) }));
        }
        for s in setters {
            s.join().unwrap();
        }

        let t0 = Instant::now();
        loop {
            if waiter.is_finished() {
                waiter.join().unwrap();
                break;
            }
            if t0.elapsed() > Duration::from_secs(5) {
                panic!("waiter did not wake within 5s after N decrements");
            }
            thread::sleep(Duration::from_millis(10));
        }
    }

    #[test]
    fn lock_latch_wakes_cross_thread_waiter() {
        // External-thread pattern: main thread waits via wait();
        // worker thread calls Latch::set; main thread unblocks.
        let latch = Arc::new(LockLatch::new());
        assert!(!latch.is_set());

        let l2 = latch.clone();
        let setter = thread::spawn(move || {
            thread::sleep(Duration::from_millis(30));
            unsafe { Latch::set(&*l2) };
        });

        let t0 = Instant::now();
        latch.wait();
        let elapsed = t0.elapsed();
        assert!(latch.is_set());
        assert!(
            elapsed >= Duration::from_millis(20),
            "wait must have actually blocked (got {elapsed:?})"
        );
        assert!(
            elapsed < Duration::from_secs(5),
            "wait must wake within 5s (got {elapsed:?})"
        );
        setter.join().unwrap();
    }

    #[test]
    fn lock_latch_wait_returns_immediately_when_already_set() {
        // Idempotency: a wait() on an already-set LockLatch
        // returns without blocking.
        let latch = LockLatch::new();
        unsafe { Latch::set(&latch) };
        let t0 = Instant::now();
        latch.wait();
        assert!(
            t0.elapsed() < Duration::from_millis(50),
            "wait on already-set latch must return immediately"
        );
        assert!(latch.is_set());
    }

    #[test]
    fn spin_latch_already_set_does_not_park() {
        // Edge case: latch is already SET when waiter attempts the
        // handshake. get_sleepy must fail because the state is SET,
        // not UNSET. The waiter must NOT park.
        let parker = Arc::new(crate::sched::sleep::Parker::new(0));
        let latch = SpinLatch::new(parker);
        unsafe { Latch::set(&latch) };
        assert!(latch.is_set());
        // get_sleepy CAS UNSET->SLEEPY fails when state is SET.
        assert!(!latch.get_sleepy(), "get_sleepy must fail on already-set latch");
    }

    #[test]
    fn get_sleepy_succeeds_then_fails_after_set() {
        let l = CoreLatch::new();
        assert!(l.get_sleepy(), "UNSET -> SLEEPY should succeed");
        // Second call returns false because state is no longer UNSET.
        assert!(!l.get_sleepy(), "SLEEPY -> SLEEPY should fail");
        unsafe { Latch::set(&l) };
        // Now the state is SET, so get_sleepy should still fail.
        assert!(!l.get_sleepy(), "SET -> SLEEPY should fail");
    }

    #[test]
    fn fall_asleep_requires_sleepy_first() {
        let l = CoreLatch::new();
        // UNSET -> SLEEPING is NOT a valid direct transition.
        assert!(!l.fall_asleep(), "UNSET -> SLEEPING must fail");
        assert!(l.get_sleepy());
        assert!(l.fall_asleep(), "SLEEPY -> SLEEPING should succeed");
        // Second fall_asleep no-ops because state is no longer SLEEPY.
        assert!(!l.fall_asleep());
    }

    #[test]
    fn set_returns_true_when_prior_was_sleeping() {
        let l = CoreLatch::new();
        assert!(l.get_sleepy());
        assert!(l.fall_asleep());
        // Direct call to the unsafe setter so we can observe the
        // return value (the Latch::set trait impl discards it).
        let was_sleeping = unsafe { CoreLatch::set(&l) };
        assert!(was_sleeping, "set must return true when prior was SLEEPING");
    }

    #[test]
    fn set_returns_false_when_prior_was_unset() {
        let l = CoreLatch::new();
        let was_sleeping = unsafe { CoreLatch::set(&l) };
        assert!(!was_sleeping, "set must return false when prior was UNSET");
    }

    #[test]
    fn set_returns_false_when_prior_was_sleepy() {
        let l = CoreLatch::new();
        assert!(l.get_sleepy());
        let was_sleeping = unsafe { CoreLatch::set(&l) };
        assert!(!was_sleeping, "set must return false when prior was SLEEPY");
        // Publisher dodged the park: caller never reached fall_asleep.
    }

    #[test]
    fn wake_up_reverts_sleeping_to_unset() {
        let l = CoreLatch::new();
        assert!(l.get_sleepy());
        assert!(l.fall_asleep());
        // Simulate a spurious wakeup: thread observed work without
        // the latch being set. wake_up reverts SLEEPING -> UNSET so
        // it can re-enter the work search.
        l.wake_up();
        assert!(!l.is_set());
        // We can now get_sleepy again because state is UNSET.
        assert!(l.get_sleepy());
    }

    #[test]
    fn wake_up_after_set_is_noop() {
        let l = CoreLatch::new();
        assert!(l.get_sleepy());
        assert!(l.fall_asleep());
        unsafe { Latch::set(&l) };
        // wake_up sees is_set() == true and exits early.
        l.wake_up();
        assert!(l.is_set(), "set state must not be reverted by wake_up");
    }

    #[test]
    fn concurrent_get_sleepy_serializes_to_one_winner() {
        // Two threads racing on get_sleepy: exactly one should win.
        let l = Arc::new(CoreLatch::new());
        let wins = Arc::new(AtomicUsize::new(0));
        let mut handles = Vec::new();
        for _ in 0..8 {
            let l = Arc::clone(&l);
            let wins = Arc::clone(&wins);
            handles.push(thread::spawn(move || {
                if l.get_sleepy() {
                    wins.fetch_add(1, Ordering::SeqCst);
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        assert_eq!(wins.load(Ordering::SeqCst), 1,
            "exactly one thread should win the UNSET -> SLEEPY CAS");
    }

    #[test]
    fn set_synchronizes_publisher_and_observer() {
        // Acquire-Release sanity check: data written by the
        // publisher BEFORE set must be visible to the observer
        // AFTER is_set returns true.
        let l = Arc::new(CoreLatch::new());
        let payload: Arc<AtomicUsize> = Arc::new(AtomicUsize::new(0));

        let l_pub = Arc::clone(&l);
        let payload_pub = Arc::clone(&payload);
        let pub_handle = thread::spawn(move || {
            payload_pub.store(0xDEAD_BEEF, Ordering::Relaxed);
            unsafe { Latch::set(&*l_pub) };
        });

        let l_obs = Arc::clone(&l);
        let payload_obs = Arc::clone(&payload);
        let obs_handle = thread::spawn(move || {
            let deadline = Instant::now() + Duration::from_secs(2);
            while !l_obs.is_set() {
                if Instant::now() > deadline {
                    panic!("observer never saw set within 2s");
                }
                std::thread::yield_now();
            }
            // After is_set returns true, the payload store must be
            // visible. Acquire ordering on is_set + Relaxed store on
            // the publisher works because the AcqRel swap in set
            // provides the release fence.
            payload_obs.load(Ordering::Relaxed)
        });

        pub_handle.join().unwrap();
        let observed = obs_handle.join().unwrap();
        assert_eq!(observed, 0xDEAD_BEEF,
            "observer must see publisher's writes after is_set");
    }
}
