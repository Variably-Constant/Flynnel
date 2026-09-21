//! Hazard-pointer reclamation: free a pointer a writer has replaced,
//! once no reader can still be about to dereference it.
//!
//! # The problem this exists for
//!
//! A lock-free slot hands a reader a raw pointer, and the reader then
//! dereferences it. A writer that replaces the slot cannot free what
//! it removed, because a reader may have loaded that pointer one
//! instruction earlier and be about to follow it. Leaking instead is
//! the usual answer and is correct, but it only works where the
//! population is bounded: a registry whose ids churn retires a pointer
//! per call and would grow without end.
//!
//! # The protocol
//!
//! A reader publishes the pointer it intends to follow into a slot of
//! its own, then re-reads the source and starts again if the source
//! has moved on. A writer unlinks first and retires second. A sweep
//! frees a retired pointer only when no published slot holds it.
//!
//! The ordering argument is the whole of the correctness, so it is
//! written out rather than asserted. A reader only dereferences a
//! pointer whose re-read confirmed it was still in the source. A
//! writer only retires a pointer it has already removed from the
//! source. So a reader that publishes a retired pointer must fail its
//! re-read, because the source no longer holds it, and it returns
//! without following it. A published-but-doomed pointer therefore
//! delays a free rather than permitting a use after free, and the
//! sweep that skips it is being conservative, not correct by luck.
//!
//! Every one of the four operations is `SeqCst`, and the count is the
//! point. The reader publishes then re-reads the source; the writer
//! unlinks the source then reads what is published. This is the
//! store-buffer shape, where both threads read after writing, and only
//! a total order rules out both of them missing the other. Three out
//! of four does not do it: whichever operation is left out can be
//! reordered past the other side's, and the pair that was supposed to
//! catch each other both come back empty.
//!
//! The reader's two are in this module, and so is the writer's read of
//! the published slots. The writer's unlink is not. It belongs to
//! whoever owns the source, so the contract on
//! [`HazardDomain::retire`] states it, and a caller that unlinks with
//! `AcqRel` breaks the protocol without touching this file.
//!
//! # What this is not
//!
//! It is not epoch reclamation. Epochs free in batches behind a global
//! counter and need every participant to pin and unpin; hazards cost a
//! store and a re-read per read and free one pointer at a time. For a
//! handful of live pointers read far more often than they are
//! replaced, which is what a handler registry is, the hazard is the
//! cheaper and much smaller mechanism. It is also bounded by
//! construction, which an epoch scheme is not: a reader that never
//! unpins stalls an epoch domain forever, where here it stalls only
//! the one pointer it published.
//!
//! Reader slots are claimed per thread by the caller and never
//! returned, so a domain serves at most `READERS` distinct threads
//! that have ever read from it.

use core::sync::atomic::{AtomicPtr, AtomicU64, AtomicUsize, Ordering};

/// A set of pointers of one type, the readers currently following
/// them, and the ones waiting to be freed.
pub struct HazardDomain<T, const READERS: usize, const RETIRED: usize> {
    /// What each reader thread has published. Null when that reader is
    /// not inside a read.
    published: [AtomicPtr<T>; READERS],
    /// Unlinked pointers not yet known to be unreachable.
    retired: [AtomicPtr<T>; RETIRED],
    /// Next reader slot to hand out.
    next_reader: AtomicUsize,
    /// Pointers the retire array had no room for. Non-zero means the
    /// domain is leaking and `RETIRED` is too small for the churn.
    spilled: AtomicU64,
}

impl<T, const READERS: usize, const RETIRED: usize> HazardDomain<T, READERS, RETIRED> {
    /// An empty domain, with no reader slots handed out and nothing
    /// retired. Const so a domain can be a `static` beside the table
    /// it protects.
    pub const fn new() -> Self {
        Self {
            published: [const { AtomicPtr::new(core::ptr::null_mut()) }; READERS],
            retired: [const { AtomicPtr::new(core::ptr::null_mut()) }; RETIRED],
            next_reader: AtomicUsize::new(0),
            spilled: AtomicU64::new(0),
        }
    }

    /// Claim a reader slot for the calling thread. Callers hold the
    /// index in a thread-local and claim once.
    ///
    /// # Panics
    ///
    /// When more than `READERS` threads have ever read from this
    /// domain. Growing silently is not available: an unclaimed reader
    /// would publish nothing and a sweep would then free a pointer it
    /// is following.
    pub fn claim_reader(&self) -> usize {
        let slot = self.next_reader.fetch_add(1, Ordering::AcqRel);
        assert!(
            slot < READERS,
            "hazard domain sized for {READERS} readers, thread {slot} asked for a slot"
        );
        slot
    }

    /// Follow `source` safely: publish what it holds, confirm it is
    /// still there, and keep it published until the guard drops.
    pub fn protect<'d>(
        &'d self,
        reader: usize,
        source: &AtomicPtr<T>,
    ) -> HazardGuard<'d, T, READERS, RETIRED> {
        loop {
            let candidate = source.load(Ordering::Acquire);
            self.published[reader].store(candidate, Ordering::SeqCst);
            if source.load(Ordering::SeqCst) == candidate {
                return HazardGuard {
                    domain: self,
                    reader,
                    ptr: candidate,
                };
            }
        }
    }

    /// Hand `ptr` over to be freed once no reader holds it.
    ///
    /// The caller must have removed `ptr` from every source a reader
    /// can reach before calling this. Retiring something still
    /// reachable is what the protocol above rules out.
    ///
    /// The unlink has to be `SeqCst`, and that requirement lives at the
    /// call site rather than in this module, which is the reason it is
    /// spelled out here. The protocol is a Dekker pair and all four of
    /// its operations have to sit in the one total order:
    ///
    ///   reader   store(published, p, SeqCst) then load(source, SeqCst)
    ///   writer   unlink(source, SeqCst)      then load(published, SeqCst)
    ///
    /// Three of those are in this module. The fourth is the caller's
    /// swap or compare-exchange, and an `AcqRel` there is not in the
    /// total order, which makes this legal:
    ///
    ///   writer   unlinks, not yet visible
    ///   writer   sweeps, reads the slot as empty
    ///   reader   publishes p
    ///   reader   re-reads the source, sees p still linked, so its
    ///            check passes
    ///   writer   frees p, and the reader follows it
    ///
    /// A use after free, reached without either side doing anything
    /// wrong locally. x86 hides it because a locked read-modify-write
    /// drains the store buffer; aarch64 does not.
    ///
    /// # Safety
    ///
    /// `ptr` came from `Box::into_raw`, was unlinked with `SeqCst`,
    /// and is retired exactly once.
    pub unsafe fn retire(&self, ptr: *mut T) {
        if ptr.is_null() {
            return;
        }
        self.sweep();
        for slot in self.retired.iter() {
            if slot
                .compare_exchange(
                    core::ptr::null_mut(),
                    ptr,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                )
                .is_ok()
            {
                self.sweep();
                return;
            }
        }
        // Every retire slot is occupied by something a reader is still
        // holding. Dropping this on the floor keeps the domain sound
        // and counts itself, which is the only honest option left:
        // freeing it here is the use after free the whole module
        // exists to prevent.
        self.spilled.fetch_add(1, Ordering::Relaxed);
    }

    /// Free every retired pointer no reader has published.
    pub fn sweep(&self) {
        for slot in self.retired.iter() {
            let candidate = slot.load(Ordering::Acquire);
            if candidate.is_null() || self.is_published(candidate) {
                continue;
            }
            if slot
                .compare_exchange(
                    candidate,
                    core::ptr::null_mut(),
                    Ordering::AcqRel,
                    Ordering::Acquire,
                )
                .is_ok()
            {
                // SAFETY: the compare-exchange took this pointer out of
                // the retire array, so this call owns it. It was
                // unlinked before being retired and no reader has it
                // published, so by the argument in the module header no
                // reader can go on to dereference it.
                drop(unsafe { Box::from_raw(candidate) });
            }
        }
    }

    fn is_published(&self, ptr: *mut T) -> bool {
        self.published
            .iter()
            .any(|p| p.load(Ordering::SeqCst) == ptr)
    }

    /// Pointers this domain could not take and did not free. Non-zero
    /// means `RETIRED` is too small for the churn it is seeing.
    pub fn spilled(&self) -> u64 {
        self.spilled.load(Ordering::Relaxed)
    }
}

impl<T, const READERS: usize, const RETIRED: usize> Default
    for HazardDomain<T, READERS, RETIRED>
{
    fn default() -> Self {
        Self::new()
    }
}

/// A pointer held published for as long as this lives.
pub struct HazardGuard<'d, T, const READERS: usize, const RETIRED: usize> {
    domain: &'d HazardDomain<T, READERS, RETIRED>,
    reader: usize,
    ptr: *mut T,
}

impl<T, const READERS: usize, const RETIRED: usize> HazardGuard<'_, T, READERS, RETIRED> {
    /// What the source held, or `None` when it held nothing.
    pub fn get(&self) -> Option<&T> {
        if self.ptr.is_null() {
            return None;
        }
        // SAFETY: the pointer was confirmed present in the source
        // after being published, so no sweep can have freed it and
        // none can free it while this guard keeps it published.
        Some(unsafe { &*self.ptr })
    }
}

impl<T, const READERS: usize, const RETIRED: usize> Drop for HazardGuard<'_, T, READERS, RETIRED> {
    fn drop(&mut self) {
        self.domain.published[self.reader].store(core::ptr::null_mut(), Ordering::SeqCst);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::AtomicBool;

    type Domain = HazardDomain<u64, 8, 16>;

    #[test]
    fn a_guard_reads_what_the_source_holds() {
        let domain = Domain::new();
        let reader = domain.claim_reader();
        let source = AtomicPtr::new(Box::into_raw(Box::new(7u64)));
        let guard = domain.protect(reader, &source);
        assert_eq!(guard.get().copied(), Some(7));
        drop(guard);
        // SAFETY: nothing else reaches this pointer in this test.
        unsafe { domain.retire(source.swap(core::ptr::null_mut(), Ordering::SeqCst)) };
    }

    #[test]
    fn an_empty_source_reads_as_nothing() {
        let domain = Domain::new();
        let reader = domain.claim_reader();
        let source: AtomicPtr<u64> = AtomicPtr::new(core::ptr::null_mut());
        let guard = domain.protect(reader, &source);
        assert!(guard.get().is_none());
    }

    #[test]
    fn a_retired_pointer_is_held_while_a_reader_publishes_it() {
        let domain = Domain::new();
        let reader = domain.claim_reader();
        let source = AtomicPtr::new(Box::into_raw(Box::new(11u64)));
        let guard = domain.protect(reader, &source);
        let removed = source.swap(core::ptr::null_mut(), Ordering::SeqCst);
        // SAFETY: removed from the source above, retired once.
        unsafe { domain.retire(removed) };
        // The sweep inside retire must have skipped it, so the guard
        // still reads it. Freeing here is exactly the defect the
        // module prevents, and reading a freed value would be
        // undefined rather than merely wrong, so this asserts the
        // value is intact.
        assert_eq!(guard.get().copied(), Some(11));
        drop(guard);
        domain.sweep();
        assert_eq!(domain.spilled(), 0, "the retire array had room");
    }

    #[test]
    fn replacing_under_readers_frees_every_generation() {
        // Each round retires the previous pointer while readers are
        // following the slot, so a sweep that freed too early would
        // hand a reader a dangling pointer and one that never freed
        // would spill.
        let domain: Arc<HazardDomain<u64, 8, 16>> = Arc::new(HazardDomain::new());
        let source = Arc::new(AtomicPtr::new(Box::into_raw(Box::new(0u64))));
        let stop = Arc::new(AtomicBool::new(false));

        let mut readers = Vec::new();
        for _ in 0..4 {
            let domain = Arc::clone(&domain);
            let source = Arc::clone(&source);
            let stop = Arc::clone(&stop);
            readers.push(std::thread::spawn(move || {
                let slot = domain.claim_reader();
                let mut seen = 0u64;
                while !stop.load(Ordering::Relaxed) {
                    let guard = domain.protect(slot, &source);
                    if let Some(v) = guard.get() {
                        // Reading it at all is the test: a freed
                        // generation would be undefined here.
                        seen = seen.wrapping_add(*v);
                    }
                }
                seen
            }));
        }

        for round in 1..=200u64 {
            let fresh = Box::into_raw(Box::new(round));
            // SeqCst, because a test that unlinks more weakly than the
            // contract asks for is not exercising the contract.
            let previous = source.swap(fresh, Ordering::SeqCst);
            // SAFETY: swapped out of the source above, retired once.
            unsafe { domain.retire(previous) };
        }
        stop.store(true, Ordering::Relaxed);
        for r in readers {
            r.join().expect("reader thread panicked");
        }

        let last = source.swap(core::ptr::null_mut(), Ordering::SeqCst);
        // SAFETY: swapped out above, retired once.
        unsafe { domain.retire(last) };
        domain.sweep();
        assert_eq!(
            domain.spilled(),
            0,
            "16 retire slots served 4 readers over 200 replacements"
        );
    }

    #[test]
    fn a_domain_refuses_more_readers_than_it_is_sized_for() {
        let domain: HazardDomain<u64, 2, 4> = HazardDomain::new();
        assert_eq!(domain.claim_reader(), 0);
        assert_eq!(domain.claim_reader(), 1);
        let over = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            domain.claim_reader();
        }));
        assert!(over.is_err(), "a third reader must not get a silent slot");
    }
}
