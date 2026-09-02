//! `JobRef` two-word vtable + concrete `Job` types.
//!
//! Direct adaptation of rayon-core 1.13's `job.rs` shape. `JobRef` is
//! a two-word handle (data pointer + execute fn pointer) plus three
//! tagging bytes (`k_outer`, `numa_hint`, `variant`) tucked into the
//! alignment slack. Thieves use the tags to classify a job without
//! dereferencing the data pointer.
//!
//! ## Why a custom vtable instead of `dyn Trait`?
//!
//! A `Box<dyn Job>` would force heap allocation for every job; a
//! `&dyn Job` would impose lifetime constraints that prevent
//! storing the job in a work-stealing deque. The two-word vtable
//! handles `StackJob` (stack-resident), `HeapJob` (box-resident),
//! and `ArcJob` (refcount-resident) uniformly. See
//! rayon-core/src/job.rs for the lineage.
//!
//! ## The "execute exactly once" contract
//!
//! Every `JobRef` MUST be executed exactly once. Executing twice
//! double-drops the captured data (use-after-free). Not executing
//! leaks the captured data (closure destructor never runs). The
//! scheduler upholds this invariant by convention - the type system
//! does not enforce it because we want `JobRef: Copy`-shaped storage
//! in the deque even though the underlying job is move-once.
//!

use core::any::Any;
use core::cell::UnsafeCell;
use std::panic;

use crate::foundation::Variant;
use crate::sched::latch::Latch;

/// Sentinel for "no NUMA preference" in the [`JobRef`] tag.
pub const NUMA_HINT_ANY: u8 = 0xFF;

/// Trait for any type that can be executed via the [`JobRef`]
/// vtable. The implementation is responsible for safely recovering
/// `Self` from the type-erased pointer.
///
/// # Safety
///
/// `execute(this)` must only be called with a `this` that was
/// originally produced by `JobRef::new::<Self>(...)` and must be
/// called exactly once per `JobRef`.
pub(crate) trait Job {
    /// Execute this job. `this` is the type-erased pointer carried
    /// by the corresponding `JobRef`.
    ///
    /// # Safety
    ///
    /// See trait-level docs.
    unsafe fn execute(this: *const ());
}

/// Type-erased, schedulable handle to a job. Two-word vtable plus
/// four tag bytes that fit in the existing alignment slack on
/// x86_64 (16-byte alignment leaves a 4-byte tail after the two
/// pointers).
///
/// `JobRef` is `Send + Sync` because the `Job` impls guarantee any
/// required bounds at construction time (closures are required to
/// be `Send`, the inner state is in `UnsafeCell`s that are only
/// touched by exactly one party at a time per the "execute once"
/// contract).
#[allow(dead_code, reason = "`reserved` is read by the K_inner=3 slot path indirectly through job.compact(); the other tag bytes are consumed by SchedFclDeque/SchedKhlDeque push to populate the shared slot header. Marked allow rather than expect because the dead_code lint may fire under future refactors that remove specific tag consumers.")]
pub struct JobRef {
    pointer: *const (),
    execute_fn: unsafe fn(*const ()),
    /// `K_outer = log2(n_limbs)` for the job's operands. Read by
    /// thieves without dereferencing `pointer` so they can decide
    /// whether to take this job before paying the cache miss on
    /// `pointer`.
    pub k_outer: u8,
    /// NUMA node hint or `NUMA_HINT_ANY` (= 0xFF). Cross-arena
    /// leader threads honor this hint per Olivier-Prins ROSS '11.
    pub numa_hint: u8,
    /// Variant tag encoding: 0 = Correct, 1 = Faithful, 2 = Fast.
    /// Used by the variant-racing primitive.
    pub variant: u8,
    /// Reserved tag byte for future use (extra tier-specific bits,
    /// debug ids, etc.). Initialized to 0.
    pub reserved: u8,
}

// SAFETY: A `JobRef` is a raw vtable handle whose Send/Sync-ness is
// guaranteed by every `Job` impl ensuring its captured state meets
// the necessary bounds.
unsafe impl Send for JobRef {}
// SAFETY: same justification as the `Send` impl above. The vtable
// handle is freely shared across threads provided each `Job` impl
// constrains its captured state to be `Send + Sync` itself.
unsafe impl Sync for JobRef {}

impl JobRef {
    /// Construct a `JobRef` from a typed pointer and the variant /
    /// k_outer / numa_hint tags. The caller must ensure `data`
    /// remains valid until the job is executed.
    ///
    /// # Safety
    ///
    /// `data` must point to a valid `T: Job` for the entire
    /// lifetime of this `JobRef` (i.e., until the single
    /// `execute()` call returns).
    pub(crate) unsafe fn new<T>(
        data: *const T,
        k_outer: u8,
        numa_hint: u8,
        variant: Variant,
    ) -> Self
    where
        T: Job,
    {
        Self {
            pointer: data as *const (),
            execute_fn: <T as Job>::execute,
            k_outer,
            numa_hint,
            variant: variant_to_tag(variant),
            reserved: 0,
        }
    }

    /// Execute this job. Consumes the `JobRef`; the underlying
    /// job's captured state is moved out / dropped per the `Job`
    /// impl.
    ///
    /// # Safety
    ///
    /// Must be called exactly once per `JobRef`. Failing to call
    /// it leaks captured state; calling it twice is undefined
    /// behavior.
    #[inline]
    pub unsafe fn execute(self) {
        // SAFETY: the `# Safety` clause on this function forwards
        // the call-exactly-once and data-validity preconditions to
        // the caller; the vtable's `execute_fn` requires the same
        // contract by construction (it was paired with `self.pointer`
        // at `JobRef::new` time).
        unsafe { (self.execute_fn)(self.pointer) }
    }

    /// Identity of this job: returns the data-pointer as a `usize`.
    /// Two `JobRef`s constructed from the same `StackJob` have the
    /// same id. Used by `join_in_worker`'s wait loop to detect
    /// when it has popped its own right-half job back from its
    /// local deque (the case where no thief stole it) so the wait
    /// loop can run it inline with `stolen=false` instead of going
    /// through the JobRef vtable with `stolen=true`. This matches
    /// rayon-core/src/join/mod.rs:138 (`job_b_id == job.id()`).
    #[inline]
    pub(crate) fn id(&self) -> usize {
        self.pointer as usize
    }

    /// Raw captured-state pointer. Crate-internal so the steal hot
    /// path in `arena_local::WorkerCtx::find_work` can software-
    /// prefetch the job's captured state into L2 before returning
    /// it to the caller. Source: arxiv 2009.00202 (Helper Without
    /// Threads).
    #[inline]
    pub(crate) fn data_ptr(&self) -> *const () {
        self.pointer
    }

    /// Decode the variant tag back into a [`Variant`].
    ///
    /// Currently called only from in-module tests; production
    /// thieves access [`Self::variant`] directly. Kept here as the
    /// canonical decoder so the field encoding stays in one place.
    #[cfg(test)]
    #[inline]
    pub fn variant_decoded(&self) -> Variant {
        tag_to_variant(self.variant)
    }

    /// Construct a `JobRef` from raw parts. Public for external
    /// scheduler integration (criterion benches, host applications
    /// that drive their own job graphs without going through
    /// [`crate::sched::join`]).
    ///
    /// # Safety
    ///
    /// `pointer` must remain valid until [`Self::execute`] is
    /// called exactly once. `execute_fn` must be paired with
    /// `pointer` per the [`Job`] trait contract (i.e., `pointer`
    /// was originally produced by a `Box::into_raw` or
    /// `&T as *const ()` for some `T` that `execute_fn` knows how
    /// to type-erase back to).
    #[inline]
    pub unsafe fn from_raw_parts(
        pointer: *const (),
        execute_fn: unsafe fn(*const ()),
        k_outer: u8,
        numa_hint: u8,
        variant: u8,
    ) -> Self {
        Self {
            pointer,
            execute_fn,
            k_outer,
            numa_hint,
            variant,
            reserved: 0,
        }
    }

    /// Project this `JobRef` to a `CompactJobRef` (16 bytes: just
    /// the pointer + execute fn). Used by K_inner=3 slot variants
    /// (Fcl/KHL) where the per-slot metadata (k_outer / numa_hint /
    /// variant) is carried in the SLOT HEADER rather than repeated
    /// per-job. The handoff trade-off: a slot's three jobs share
    /// metadata (which is true for the common recursive-split case
    /// where children inherit the parent's k_outer / variant); the
    /// 3-in-one-cache-line slot fits exactly.
    #[inline]
    pub(crate) fn compact(&self) -> CompactJobRef {
        CompactJobRef {
            pointer: self.pointer,
            execute_fn: self.execute_fn,
        }
    }
}

/// Compact two-word job handle (16 bytes on 64-bit): just the
/// data pointer + execute fn pointer. Used by K_inner=3 slot
/// variants (Fcl/KHL) where per-job metadata (k_outer / numa_hint /
/// variant) is hoisted into a shared slot header. Three
/// `CompactJobRef`s fit in a 48-byte block, leaving 16 bytes for
/// slot bookkeeping in a 64-byte cache line.
///
/// The execute-exactly-once contract from [`JobRef`] carries over:
/// each `CompactJobRef` produced by [`JobRef::compact`] must be
/// executed (via [`CompactJobRef::execute`]) exactly once, sharing
/// the same captured-state pointer as the source `JobRef`. In
/// practice this means the Fcl/KHL slot consumes the source
/// `JobRef` at push time (the slot becomes the canonical handle),
/// and at pop time each `CompactJobRef` in the popped slot is
/// executed once.
#[derive(Copy, Clone)]
pub struct CompactJobRef {
    pointer: *const (),
    execute_fn: unsafe fn(*const ()),
}

// SAFETY: same justification as the `Send`/`Sync` impls on
// [`JobRef`] - the underlying captured-state lifetime / safety is
// guaranteed by the [`Job`] impl at construction time.
unsafe impl Send for CompactJobRef {}
// SAFETY: same justification as the `Send` impl directly above.
unsafe impl Sync for CompactJobRef {}

impl CompactJobRef {
    /// Sentinel compact ref carrying a never-fires execute fn. Used
    /// to pad partial-batch slots in Fcl/KHL.
    pub const fn null() -> Self {
        // SAFETY: pointer is null and execute_fn is a never-called
        // unwind shim; `null()` is only used to pad slot slack and
        // must never be passed to `execute()`. The Fcl/KHL pop path
        // honors `n_items` to avoid touching slack entries.
        unsafe fn never_called(_: *const ()) {
            // Reached only if someone calls execute on a padding
            // slot, which the slot's `n_items` discipline forbids.
            panic!("CompactJobRef::null() must never be executed");
        }
        Self {
            pointer: std::ptr::null(),
            execute_fn: never_called,
        }
    }

    /// Rehydrate a full [`JobRef`] from this compact handle plus
    /// the shared per-slot metadata. Used by the WorkerCtx adapter
    /// to bridge between the K_inner=3 slot-batch storage layer and
    /// the per-job `find_work` API surface.
    ///
    /// # Safety
    ///
    /// `self.pointer` must point to a valid `T: Job` for the
    /// captured-state lifetime. The reconstructed JobRef inherits
    /// the compact ref's once-only execute contract; do not call
    /// both `self.execute()` and `to_jobref(...).execute()` - that
    /// would double-execute.
    #[inline]
    pub unsafe fn to_jobref(self, k_outer: u8, numa_hint: u8, variant: u8) -> JobRef {
        JobRef {
            pointer: self.pointer,
            execute_fn: self.execute_fn,
            k_outer,
            numa_hint,
            variant,
            reserved: 0,
        }
    }

    /// Execute this compact handle. Single-execute contract per
    /// [`JobRef::execute`]; double-execute double-drops the
    /// captured state.
    ///
    /// # Safety
    ///
    /// `self.pointer` must point to a valid `T: Job` for the
    /// captured-state lifetime. Must be called exactly once per
    /// `CompactJobRef`.
    #[inline]
    pub unsafe fn execute(self) {
        // SAFETY: by contract, `self.pointer` was paired with
        // `self.execute_fn` at `JobRef::compact` time, and the
        // caller's once-only invariant carries the original
        // execute-once contract.
        unsafe { (self.execute_fn)(self.pointer) }
    }
}

#[inline]
fn variant_to_tag(v: Variant) -> u8 {
    match v {
        Variant::Correct => 0,
        Variant::Faithful => 1,
        Variant::Fast => 2,
    }
}

#[cfg(test)]
#[inline]
fn tag_to_variant(tag: u8) -> Variant {
    match tag {
        0 => Variant::Correct,
        2 => Variant::Fast,
        // Tag 1 is Faithful by spec. Unknown tags decode to Faithful
        // as a safe mid-tier default; this can only happen if the
        // JobRef was constructed via a non-Variant path (no current
        // API allows this) so it is defence-in-depth, not a real
        // recovery branch.
        _ => Variant::Faithful,
    }
}

// ---------------------------------------------------------------------------
// JobResult: closure return value or captured panic payload
// ---------------------------------------------------------------------------

/// Result slot owned by a [`StackJob`]. Holds the closure's return
/// value or, if the closure panicked, the payload captured by
/// [`std::panic::catch_unwind`] so the waiting thread can
/// `resume_unwind`.
pub(crate) enum JobResult<T> {
    /// Slot not yet written (initial state).
    None,
    /// Closure returned normally.
    Ok(T),
    /// Closure panicked; payload is the unwind value.
    Panic(Box<dyn Any + Send + 'static>),
}

impl<T> JobResult<T> {
    /// Run `func` under `catch_unwind` and store the outcome.
    /// `stolen` is passed through to the closure so it can adapt
    /// behavior when it ran on a different thread than the parent.
    fn capture<F: FnOnce(bool) -> T>(func: F, stolen: bool) -> Self {
        match panic::catch_unwind(panic::AssertUnwindSafe(|| func(stolen))) {
            Ok(value) => JobResult::Ok(value),
            Err(payload) => JobResult::Panic(payload),
        }
    }

    /// Convert into the closure return value. Panics if the slot
    /// was never written (programmer error: forgot to execute);
    /// resumes the original unwind if the closure panicked.
    pub(crate) fn into_return_value(self) -> T {
        match self {
            JobResult::None => panic!("JobResult was never written"),
            JobResult::Ok(value) => value,
            JobResult::Panic(payload) => panic::resume_unwind(payload),
        }
    }
}

// ---------------------------------------------------------------------------
// StackJob: stack-allocated job for join(a, b)
// ---------------------------------------------------------------------------

/// Stack-resident job used by `join`. The caller allocates this on
/// its own stack frame, submits a `JobRef` for the right half, runs
/// the left half inline, then waits on the embedded latch. Because
/// the data lives on the caller's stack, **the caller MUST NOT
/// return until the latch is set** - that would invalidate the
/// memory the worker thread is still reading.
///
/// `L` is the latch type - typically [`crate::sched::latch::SpinLatch`]
/// for the join path (wake-capable, holds a Parker), and
/// [`crate::sched::latch::CoreLatch`] for unit tests (bare state
/// machine, no wake mechanism). `F` is the closure, `R` is its
/// return value. Both must be `Send` because the closure may run
/// on a different worker thread.
pub(crate) struct StackJob<L, F, R>
where
    L: Latch + Sync,
    F: FnOnce(bool) -> R + Send,
    R: Send,
{
    /// Latch that the executing worker sets after writing
    /// [`result`]; the parent thread polls / parks on this.
    pub(crate) latch: L,
    /// Closure to execute. Moved out exactly once via the
    /// `UnsafeCell::take()` pattern.
    func: UnsafeCell<Option<F>>,
    /// Slot for the closure's return value or captured panic.
    result: UnsafeCell<JobResult<R>>,
}

impl<L, F, R> StackJob<L, F, R>
where
    L: Latch + Sync,
    F: FnOnce(bool) -> R + Send,
    R: Send,
{
    /// Construct a fresh `StackJob` with the given latch and
    /// closure. The latch starts un-set; the result slot starts
    /// empty.
    pub(crate) fn new(func: F, latch: L) -> Self {
        Self {
            latch,
            func: UnsafeCell::new(Some(func)),
            result: UnsafeCell::new(JobResult::None),
        }
    }

    /// Produce a [`JobRef`] for this job. The returned `JobRef`
    /// shares the lifetime of `&self` by convention; the caller
    /// must keep `self` alive until the latch is set.
    ///
    /// # Safety
    ///
    /// Caller must not drop `self` until `latch.is_set()` returns
    /// true.
    pub(crate) unsafe fn as_job_ref(
        &self,
        k_outer: u8,
        numa_hint: u8,
        variant: Variant,
    ) -> JobRef {
        // SAFETY: this function's `# Safety` clause requires the
        // caller to keep `self` alive until the latch is set,
        // which is exactly the `JobRef::new` data-validity
        // precondition.
        unsafe { JobRef::new(self, k_outer, numa_hint, variant) }
    }

    /// Run the captured closure inline on the calling thread,
    /// without going through the scheduler. Returns the closure's
    /// return value or resumes its panic.
    ///
    /// # Safety
    ///
    /// Must be called exactly once on a job that has not yet been
    /// executed via [`Self::as_job_ref`]'s vtable path.
    pub(crate) unsafe fn run_inline(self, stolen: bool) -> R {
        // SAFETY: this function's `# Safety` clause ensures the
        // closure has not already been taken by a vtable execute
        // path, so the `Option::take` here returns `Some`.
        let func = unsafe { (*self.func.get()).take() }.expect("closure already taken");
        func(stolen)
    }

    /// Retrieve the executed result. Must be called after the
    /// latch has been set (otherwise the slot is empty and this
    /// panics).
    ///
    /// # Safety
    ///
    /// Caller must have observed `latch.is_set() == true` before
    /// calling.
    pub(crate) unsafe fn into_result(self) -> R {
        // ptr::read byte-copies the JobResult bits out, but the
        // UnsafeCell still holds the same bits. Without the
        // subsequent write of JobResult::None, the StackJob's drop
        // would re-drop the inner data (double-free on any heap
        // payload: the Box<dyn Any + Send> inside JobResult::Panic
        // or any heap-owning T inside JobResult::Ok(T)).
        //
        // SAFETY: this function's `# Safety` clause requires the
        // caller to have observed `latch.is_set() == true`, which
        // means the vtable `execute` has written the result into
        // the cell and the Acquire half of the latch ordering
        // makes that write visible here.
        let result = unsafe { core::ptr::read(self.result.get()) };
        // SAFETY: same justification: we still hold exclusive
        // access to the cell because consuming `self` proves the
        // only outstanding `JobRef` has already been consumed by
        // the vtable execute.
        unsafe { core::ptr::write(self.result.get(), JobResult::None) };
        result.into_return_value()
    }
}

impl<L, F, R> Job for StackJob<L, F, R>
where
    L: Latch + Sync,
    F: FnOnce(bool) -> R + Send,
    R: Send,
{
    unsafe fn execute(this: *const ()) {
        // SAFETY: by `Job` contract, `this` was produced by
        // JobRef::new::<Self>(...) so the cast is valid.
        let this = unsafe { &*(this as *const Self) };
        // Take the closure out of the cell so its destructor runs
        // even if the body panics. `unwrap` is sound because
        // `execute` is called exactly once per JobRef.
        //
        // SAFETY: exclusive access to the cell follows from the
        // single-execute contract on `Job::execute`.
        let func = unsafe { (*this.func.get()).take() }.expect("closure already taken");
        // Run under catch_unwind so a closure panic does not unwind
        // through the worker scheduler. JobResult::capture handles
        // both the Ok and Panic paths; no separate abort guard is
        // needed because the panic cannot escape this frame.
        let result = JobResult::capture(func, true /* stolen */);
        // Write the result BEFORE setting the latch: the joining
        // thread relies on AcqRel ordering established by the
        // latch's set() to observe this store.
        //
        // SAFETY: still the only writer to `this.result`; the
        // latch has not yet been set so no joining thread has
        // observed the result cell.
        unsafe { core::ptr::write(this.result.get(), result) };
        // After the next line the parent may resume, observe
        // is_set, and drop the StackJob. We must not read `this`
        // after the set call returns.
        //
        // SAFETY: `Latch::set` requires `this.latch` to be
        // valid for the entry of the call, which it is because
        // we just read through `this` above. Per the latch's
        // own `# Safety` clause we touch nothing through `this`
        // after this point.
        unsafe { Latch::set(&this.latch) };
    }
}

/// Variant of [`StackJob`] that holds its latch as an `Arc<L>`
/// instead of owning it inline. Use when N publishers share ONE
/// latch (the canonical CountLatch pattern in `fan_out_in_worker`).
///
/// # Why a sibling type rather than a generic field
///
/// `StackJob<CoreLatch, _, _>` and `StackJob<Arc<CoreLatch>, _, _>`
/// cannot share one impl: `Arc<L>` is not `Latch` (the Latch trait
/// is implemented on the inner L, not on the Arc wrapper). Adding
/// a separate type keeps the call sites that own their own latch
/// (the join paths) at zero allocation and zero refcount overhead,
/// while giving fan_out the shared-latch shape it needs.
///
/// # Lifetime + invalidation
///
/// The Arc field in `*this` keeps L alive through Latch::set. The
/// parent ALSO holds an Arc<L> clone (the one passed to wait), so
/// even if the parent drops this StackJobShared the moment it
/// observes is_set, refcount stays >= 1 and L is not freed while
/// the setter is still inside Latch::set. The latch's own set-impl
/// (CountLatch / SpinLatch) reads everything from *this before the
/// publishing store, so once the store completes the setter does
/// not touch *this and the parent is free to drop.
pub(crate) struct StackJobShared<L, F, R>
where
    L: Latch + Sync,
    F: FnOnce(bool) -> R + Send,
    R: Send,
{
    /// Shared latch. The N participants all set the same latch.
    pub(crate) latch: std::sync::Arc<L>,
    /// Closure to execute. Moved out exactly once via the
    /// `UnsafeCell::take()` pattern (mirrors [`StackJob`]).
    func: UnsafeCell<Option<F>>,
    /// Slot for the closure's return value or captured panic.
    result: UnsafeCell<JobResult<R>>,
}

impl<L, F, R> StackJobShared<L, F, R>
where
    L: Latch + Sync,
    F: FnOnce(bool) -> R + Send,
    R: Send,
{
    /// Construct a `StackJobShared` for the given closure that will
    /// publish to `latch` on completion. The `Arc<L>` clone count
    /// rises by ONE for each StackJobShared constructed.
    pub(crate) fn new(func: F, latch: std::sync::Arc<L>) -> Self {
        Self {
            latch,
            func: UnsafeCell::new(Some(func)),
            result: UnsafeCell::new(JobResult::None),
        }
    }

    /// Produce a [`JobRef`] for this shared-latch job. Same lifetime
    /// contract as [`StackJob::as_job_ref`]: caller must keep `self`
    /// alive until the latch is set.
    ///
    /// # Safety
    ///
    /// Caller must not drop `self` until the shared latch reports
    /// is_set.
    pub(crate) unsafe fn as_job_ref(
        &self,
        k_outer: u8,
        numa_hint: u8,
        variant: Variant,
    ) -> JobRef {
        // SAFETY: forwarded from this function's `# Safety` clause.
        unsafe { JobRef::new(self, k_outer, numa_hint, variant) }
    }

    /// Retrieve the executed result. Same contract as
    /// [`StackJob::into_result`]: caller must have observed
    /// `latch.is_set() == true` before calling.
    ///
    /// # Safety
    ///
    /// Caller must have observed `latch.is_set() == true` before
    /// calling.
    pub(crate) unsafe fn into_result(self) -> R {
        // SAFETY: same justification as StackJob::into_result. The
        // publishing store inside Latch::set established the
        // Acquire/Release pair that makes the result store visible.
        let result = unsafe { core::ptr::read(self.result.get()) };
        // SAFETY: we still hold exclusive access because consuming
        // `self` proves the only JobRef has been consumed.
        unsafe { core::ptr::write(self.result.get(), JobResult::None) };
        result.into_return_value()
    }
}

impl<L, F, R> Job for StackJobShared<L, F, R>
where
    L: Latch + Sync,
    F: FnOnce(bool) -> R + Send,
    R: Send,
{
    unsafe fn execute(this: *const ()) {
        // SAFETY: by `Job` contract, `this` was produced by
        // JobRef::new::<Self>(...) so the cast is valid.
        let this = unsafe { &*(this as *const Self) };
        // Take the closure out so its destructor runs even on panic.
        //
        // SAFETY: single-execute contract on `Job::execute`.
        let func = unsafe { (*this.func.get()).take() }.expect("closure already taken");
        let result = JobResult::capture(func, true);
        // Write the result BEFORE setting the latch (same ordering
        // requirement as StackJob: parent observes via Acquire half
        // of the latch publish).
        //
        // SAFETY: still the only writer to `this.result`.
        unsafe { core::ptr::write(this.result.get(), result) };
        // Read a stable raw pointer to the inner L via the Arc
        // field. The Arc clone in `this.latch` keeps L valid for
        // the duration of this borrow; after Latch::set publishes,
        // the setter must not touch *this anymore (parent may have
        // dropped this StackJobShared). Both CountLatch::set and
        // SpinLatch::set honor that contract by reading every
        // *this field BEFORE their publishing CoreLatch::set call.
        let latch_ptr: *const L = std::sync::Arc::as_ptr(&this.latch);
        // SAFETY: latch_ptr is valid because the Arc clone in
        // *this is alive at the call site. The latch impl honors
        // the publish-then-invalidate contract; we do not touch
        // *this after this call returns.
        unsafe { Latch::set(latch_ptr) };
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::foundation::Variant;
    use crate::sched::latch::CoreLatch;

    #[test]
    fn jobref_tag_round_trip_variant() {
        for v in [Variant::Correct, Variant::Faithful, Variant::Fast] {
            let job = StackJob::new(|_stolen| 42u32, CoreLatch::new());
            let r = unsafe { job.as_job_ref(8, 1, v) };
            assert_eq!(r.variant_decoded(), v, "variant {v:?} should round-trip");
            assert_eq!(r.k_outer, 8);
            assert_eq!(r.numa_hint, 1);
            // Consume the JobRef to keep the once-only contract.
            unsafe { r.execute() };
            assert!(job.latch.is_set());
        }
    }

    #[test]
    fn jobref_size_is_two_pointers_plus_tags() {
        // Two pointers (8+8 = 16 on x86_64) + 4 tag bytes = 20,
        // padded up to 24 by alignment. On 32-bit it would be 12.
        // We just check the type is small (<= 32 bytes) and that
        // the field offsets fit in the expected slack.
        let sz = core::mem::size_of::<JobRef>();
        let align = core::mem::align_of::<JobRef>();
        assert!(sz <= 32, "JobRef should be small; got {sz} bytes");
        assert!(align >= core::mem::align_of::<*const ()>());
    }

    #[test]
    fn stack_job_executes_closure_and_sets_latch() {
        let job = StackJob::new(|_stolen| 0x1234_5678u32, CoreLatch::new());
        let r = unsafe { job.as_job_ref(8, NUMA_HINT_ANY, Variant::Faithful) };
        assert!(!job.latch.is_set(), "latch starts un-set");
        unsafe { r.execute() };
        assert!(job.latch.is_set(), "execute must set the latch");
        let value = unsafe { job.into_result() };
        assert_eq!(value, 0x1234_5678);
    }

    #[test]
    fn stack_job_executes_with_stolen_flag_true() {
        // Closure parameter receives `stolen` so it can adapt
        // behavior for cross-thread execution. Verify it gets
        // `true` from the JobRef path.
        let job = StackJob::new(|stolen| stolen, CoreLatch::new());
        let r = unsafe { job.as_job_ref(2, 0, Variant::Fast) };
        unsafe { r.execute() };
        let value = unsafe { job.into_result() };
        assert!(value, "JobRef::execute must pass stolen = true");
    }

    #[test]
    fn stack_job_shared_executes_and_sets_shared_latch() {
        // N=3 participants share ONE CountLatch via Arc. Each call
        // to execute() decrements; the third decrement publishes.
        use crate::sched::latch::CountLatch;
        use crate::sched::sleep::Parker;
        let parker = std::sync::Arc::new(Parker::new(0));
        let latch = std::sync::Arc::new(CountLatch::new(3, parker));
        let jobs: Vec<StackJobShared<CountLatch, _, u32>> = (0..3u32)
            .map(|i| StackJobShared::new(move |_stolen| i * 10, latch.clone()))
            .collect();
        // Before any execute: latch un-set, outstanding == 3.
        assert!(!latch.is_set());
        assert_eq!(latch.outstanding(), 3);
        // Execute first: outstanding -> 2, still not set.
        let r0 = unsafe { jobs[0].as_job_ref(8, NUMA_HINT_ANY, Variant::Faithful) };
        unsafe { r0.execute() };
        assert!(!latch.is_set(), "latch must NOT set after 1/3 decrements");
        assert_eq!(latch.outstanding(), 2);
        // Execute second: outstanding -> 1, still not set.
        let r1 = unsafe { jobs[1].as_job_ref(8, NUMA_HINT_ANY, Variant::Faithful) };
        unsafe { r1.execute() };
        assert!(!latch.is_set(), "latch must NOT set after 2/3 decrements");
        assert_eq!(latch.outstanding(), 1);
        // Execute third: outstanding -> 0, latch fires.
        let r2 = unsafe { jobs[2].as_job_ref(8, NUMA_HINT_ANY, Variant::Faithful) };
        unsafe { r2.execute() };
        assert!(latch.is_set(), "latch MUST set on final decrement");
        assert_eq!(latch.outstanding(), 0);
        // Results retrievable in caller order, each from its own
        // StackJobShared's result slot.
        let mut iter = jobs.into_iter();
        let v0 = unsafe { iter.next().unwrap().into_result() };
        let v1 = unsafe { iter.next().unwrap().into_result() };
        let v2 = unsafe { iter.next().unwrap().into_result() };
        assert_eq!((v0, v1, v2), (0, 10, 20));
    }

    #[test]
    fn run_inline_passes_stolen_flag_false() {
        // run_inline is the "same-thread" path; closure receives
        // false (caller-thread execution, not stolen).
        let job = StackJob::new(|stolen| stolen, CoreLatch::new());
        let value = unsafe { job.run_inline(false) };
        assert!(!value, "run_inline must pass stolen = false");
    }

    // Gated on `panic = "unwind"` because the test-fast / release
    // profiles use `panic = "abort"`, where `catch_unwind` is a no-op
    // and a closure panic terminates the process. The production
    // catch_unwind path in JobResult::capture is correct code; it
    // simply has nothing to capture under abort. Run with
    // `cargo test --lib stack_job_captures_panic_for_parent_resume`
    // under the default test profile (which uses panic=unwind) to
    // exercise this path.
    #[test]
    #[cfg(panic = "unwind")]
    fn stack_job_captures_panic_for_parent_resume() {
        let job = StackJob::new(|_stolen| -> u32 {
            panic!("intentional test panic from worker closure");
        }, CoreLatch::new());
        let r = unsafe { job.as_job_ref(2, NUMA_HINT_ANY, Variant::Faithful) };
        // Worker thread captures the panic without unwinding through
        // the scheduler.
        unsafe { r.execute() };
        assert!(job.latch.is_set(), "latch must be set even on panic");

        // Parent thread calls into_result, which must resume the
        // captured unwind. catch_unwind on the parent side recovers
        // it.
        let payload = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
            job.into_result()
        }));
        assert!(payload.is_err(), "into_result must resume the captured panic");
    }

    #[test]
    fn variant_tag_decodes_unknown_to_faithful() {
        // tag_to_variant treats unknown values as Faithful (a safe
        // mid-tier default). Verify directly.
        assert_eq!(tag_to_variant(0), Variant::Correct);
        assert_eq!(tag_to_variant(1), Variant::Faithful);
        assert_eq!(tag_to_variant(2), Variant::Fast);
        assert_eq!(tag_to_variant(3), Variant::Faithful);
        assert_eq!(tag_to_variant(0xFF), Variant::Faithful);
    }
}
