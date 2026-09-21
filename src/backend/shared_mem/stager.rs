//! Per-thread staging slots for the shared-memory backends.
//!
//! # Why a staging buffer is not shared
//!
//! `khpd`, `urd_backend` and `lcrq_lifo` each accumulate items and
//! flush them into the ring in batches. Each documented its buffer as
//! owner-side ("Only the owner process may stage") and then guarded it
//! with a mutex anyway, because `stage(&self)` has to mutate a `Vec`
//! and a mutex is the easiest thing that lets it.
//!
//! That lock does one of two things and neither is useful. If the
//! single-writer rule holds it excludes nothing and pays an atomic
//! read-modify-write per item to enforce a rule already guaranteed. If
//! the rule does not hold, and `stage(&self)` reached through an
//! `Arc<dyn DispatchBackend>` makes that reachable, it absorbs the
//! violation silently: a broken invariant surfaces as contention
//! rather than as an error.
//!
//! Measured on zen3 at 16 threads, quiet, over a control floor, in
//! nanoseconds per push: a thread-local vector 2.55, the mutex 5.03, a
//! compare-exchange stack 13.96. The lock-free staging buffer is the
//! slowest of the three, because pushing a node allocates where the
//! vector does not. Giving each thread its own buffer is both the
//! fastest arm and the one that needs no exclusion at all, which is
//! why this module exists rather than a lock-free queue.
//!
//! # The slot index
//!
//! A thread claims one index here and uses that same index in every
//! staging table in the process. One counter rather than one per
//! table: a thread that stages into two backends holds one identity,
//! and a table is then a plain array indexed by it.
//!
//! Indexes are never returned. A process that spawns and retires more
//! than [`MAX_STAGERS`] distinct staging threads over its life will
//! exhaust them, which is a panic naming the limit rather than a
//! silently shared slot, because two threads on one slot is the data
//! race this module exists to remove.

use core::cell::{Cell, UnsafeCell};
use core::sync::atomic::{AtomicUsize, Ordering};

/// Threads that may stage into a shared-memory backend over the life
/// of the process.
pub const MAX_STAGERS: usize = 256;

static NEXT_STAGER: AtomicUsize = AtomicUsize::new(0);

/// The calling thread's staging index, claimed on first use.
///
/// # Panics
///
/// When more than [`MAX_STAGERS`] threads have ever staged.
pub fn stager_index() -> usize {
    thread_local! {
        static INDEX: Cell<usize> = const { Cell::new(usize::MAX) };
    }
    INDEX.with(|cell| {
        let held = cell.get();
        if held != usize::MAX {
            return held;
        }
        let fresh = NEXT_STAGER.fetch_add(1, Ordering::AcqRel);
        assert!(
            fresh < MAX_STAGERS,
            "shared-memory staging is sized for {MAX_STAGERS} threads and thread {fresh} asked \
             for a slot; sharing one would be a data race"
        );
        cell.set(fresh);
        fresh
    })
}

/// One thread's staging buffer, and its length for anyone counting.
struct Slot<T> {
    /// Written only by the thread holding this slot's index.
    items: UnsafeCell<Vec<T>>,
    /// Published by that same thread after each change, so a
    /// diagnostic can total the table without reading a buffer it does
    /// not own. Reading `items` from another thread would be a data
    /// race; reading this is not.
    len: AtomicUsize,
}

/// A staging table: one buffer per staging thread, no sharing and so
/// nothing to exclude.
pub struct StagingTable<T> {
    slots: Box<[Slot<T>]>,
}

// SAFETY: a slot's buffer is reached only through `with_mine`, which
// indexes by the caller's own `stager_index`, and an index belongs to
// exactly one thread for the life of the process. No other method
// dereferences `items`.
unsafe impl<T: Send> Sync for StagingTable<T> {}

impl<T> StagingTable<T> {
    /// A table with a buffer per possible staging thread. The buffers
    /// are empty until a thread first stages into one.
    pub fn new() -> Self {
        let mut slots = Vec::with_capacity(MAX_STAGERS);
        for _ in 0..MAX_STAGERS {
            slots.push(Slot {
                items: UnsafeCell::new(Vec::new()),
                len: AtomicUsize::new(0),
            });
        }
        Self {
            slots: slots.into_boxed_slice(),
        }
    }

    /// Run `act` against the calling thread's own buffer.
    ///
    /// The buffer belongs to this thread alone, so `act` gets it
    /// exclusively without excluding anyone.
    pub fn with_mine<R>(&self, act: impl FnOnce(&mut Vec<T>) -> R) -> R {
        let slot = &self.slots[stager_index()];
        // SAFETY: this index belongs to the calling thread alone and
        // nothing else dereferences `items`, so this is the only
        // reference to the buffer in existence.
        let buffer = unsafe { &mut *slot.items.get() };
        let out = act(buffer);
        slot.len.store(buffer.len(), Ordering::Release);
        out
    }

    /// How many items the calling thread has staged.
    pub fn mine_len(&self) -> usize {
        self.slots[stager_index()].len.load(Ordering::Acquire)
    }

    /// How many items are staged across every thread.
    ///
    /// A total rather than an instant: a thread staging while this
    /// walks is counted at whatever length it had when its slot was
    /// read. For reporting, not for deciding when to flush.
    pub fn staged_total(&self) -> usize {
        self.slots
            .iter()
            .map(|slot| slot.len.load(Ordering::Acquire))
            .sum()
    }
}

impl<T> Default for StagingTable<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> std::fmt::Debug for StagingTable<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StagingTable")
            .field("staged_total", &self.staged_total())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn a_thread_keeps_one_index() {
        let first = stager_index();
        assert_eq!(first, stager_index(), "an index is claimed once");
    }

    #[test]
    fn staging_is_private_to_the_staging_thread() {
        let table: Arc<StagingTable<u64>> = Arc::new(StagingTable::new());
        let mut writers = Vec::new();
        for who in 0..4u64 {
            let table = Arc::clone(&table);
            writers.push(std::thread::spawn(move || {
                for i in 0..100u64 {
                    table.with_mine(|buffer| buffer.push(who * 1000 + i));
                }
                // Each thread sees only what it staged, which is what
                // makes the flush count a local decision.
                assert_eq!(table.mine_len(), 100);
                table.with_mine(|buffer| {
                    assert!(
                        buffer.iter().all(|v| v / 1000 == who),
                        "a buffer holds only its own thread's items"
                    );
                    buffer.clear();
                });
                assert_eq!(table.mine_len(), 0);
            }));
        }
        for w in writers {
            w.join().expect("a staging thread panicked");
        }
        assert_eq!(table.staged_total(), 0, "every thread drained its own");
    }

    #[test]
    fn the_total_sees_every_thread() {
        let table: Arc<StagingTable<u64>> = Arc::new(StagingTable::new());
        let mut writers = Vec::new();
        for who in 0..4u64 {
            let table = Arc::clone(&table);
            writers.push(std::thread::spawn(move || {
                for i in 0..25u64 {
                    table.with_mine(|buffer| buffer.push(who * 1000 + i));
                }
            }));
        }
        for w in writers {
            w.join().expect("a staging thread panicked");
        }
        assert_eq!(
            table.staged_total(),
            100,
            "four threads staged 25 each and none drained"
        );
    }
}
