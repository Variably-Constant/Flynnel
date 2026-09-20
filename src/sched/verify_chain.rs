//! Verification hash-chain on the SMT-sibling IoPool.
//!
//! D-UMPA's verify-trace + zk-export features (and any future
//! per-stripe verification pass) need a running hash over per-stripe
//! outputs. Doing this inline on the compute workers stalls them
//! between stripes; this module offloads the hash work to the IO
//! pool so compute proceeds without latency penalty.
//!
//! ## API shape
//!
//! [`VerifyChain`] owns an `Arc<Mutex<HashState>>`. Submitting a
//! chunk via [`VerifyChain::submit_chunk`] schedules a hash-update
//! task onto the IoPool; the task takes the mutex briefly to update
//! the running state. [`VerifyChain::finalize`] blocks until all
//! submitted chunks have been processed and returns the root.
//!
//! ## Hashing back-end
//!
//! When the `verify-chain` Cargo feature is enabled (the only build
//! that depends on `blake3`), the chain uses BLAKE3. Otherwise it
//! uses an internal FxHash-style 64-bit accumulator zero-padded to
//! 32 bytes; this lets the module compile and unit-test on any
//! build but is NOT cryptographically sound. Production attestation
//! MUST use the BLAKE3 path (enable `verify-chain`).
//!

use std::sync::{Arc, Condvar, Mutex};
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::sched::io_pool::global_io_pool;

/// Trait abstracting the hash back-end. Implementations must accept
/// `&[u8]` chunks via `update` and produce a 32-byte root via
/// `finalize`. Implementations are `Send` because the verify worker
/// may run on any IoPool thread.
pub trait VerifyHasher: Send + 'static {
    /// Absorb a chunk of bytes into the running state.
    fn update(&mut self, chunk: &[u8]);
    /// Produce the final 32-byte root, consuming the state.
    /// Takes `Box<Self>` so the trait stays object-safe (move-by-
    /// value out of a `dyn VerifyHasher` would not compile; the
    /// boxed-self form lets the implementation take ownership of
    /// the inner state through the Box).
    fn finalize(self: Box<Self>) -> [u8; 32];
}

#[cfg(feature = "verify-chain")]
mod blake3_impl {
    use super::VerifyHasher;
    /// BLAKE3-rooted [`VerifyHasher`] used when the `verify-chain`
    /// feature is enabled.
    pub struct Blake3Hasher(pub blake3::Hasher);
    impl Blake3Hasher {
        /// New empty BLAKE3 hasher state.
        pub fn new() -> Self {
            Self(blake3::Hasher::new())
        }
    }
    impl Default for Blake3Hasher {
        fn default() -> Self {
            Self::new()
        }
    }
    impl VerifyHasher for Blake3Hasher {
        fn update(&mut self, chunk: &[u8]) {
            self.0.update(chunk);
        }
        fn finalize(self: Box<Self>) -> [u8; 32] {
            *self.0.finalize().as_bytes()
        }
    }
}
#[cfg(feature = "verify-chain")]
pub use blake3_impl::Blake3Hasher;

/// FxHash-style fallback hasher. Fast u64 multiplicative chain
/// embedded into a 32-byte root by repeated mixing. NOT
/// cryptographic; the cycle structure of a u64 multiply lets
/// crafted inputs produce desired output. Provided so this module
/// compiles + unit-tests on builds without the `dumpa-experimental`
/// feature.
///
/// Production verify-trace must enable the `verify-chain` feature
/// (which pulls in the BLAKE3 dep) to use the BLAKE3-rooted hasher.
pub struct FxFallbackHasher {
    state: u64,
}

impl FxFallbackHasher {
    /// New hasher initialized with the FxHash seed.
    pub fn new() -> Self {
        Self { state: 0xCBF2_9CE4_8422_2325 }
    }
}

impl Default for FxFallbackHasher {
    fn default() -> Self {
        Self::new()
    }
}

impl VerifyHasher for FxFallbackHasher {
    fn update(&mut self, chunk: &[u8]) {
        const PRIME: u64 = 0x100_0000_01B3;
        for &b in chunk {
            self.state = self.state.wrapping_mul(PRIME) ^ (b as u64);
        }
    }
    fn finalize(self: Box<Self>) -> [u8; 32] {
        let mut out = [0u8; 32];
        // Tile the 8-byte state into the 32-byte buffer with a
        // mixing constant per quarter so identical zero-padded
        // inputs don't produce zero outputs.
        let mut acc = self.state;
        for i in 0..4 {
            let mix = acc.wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
            out[i * 8..i * 8 + 8].copy_from_slice(&mix.to_le_bytes());
            acc = mix.rotate_left(17);
        }
        out
    }
}

/// Build the recommended default hasher: BLAKE3 when the
/// `verify-chain` feature is enabled, else the FxFallback hasher.
#[cfg(feature = "verify-chain")]
pub fn default_hasher() -> Box<dyn VerifyHasher> {
    Box::new(Blake3Hasher::new())
}

/// FxFallback default hasher used when the `verify-chain` feature
/// (and its BLAKE3 dep) is not enabled.
#[cfg(not(feature = "verify-chain"))]
pub fn default_hasher() -> Box<dyn VerifyHasher> {
    Box::new(FxFallbackHasher::new())
}

/// Internal shared state for a chain: the running hasher, a
/// pending-chunks counter for the finalize barrier, and a
/// condition variable that submit/finalize use to coordinate.
struct ChainShared {
    hasher: Mutex<Option<Box<dyn VerifyHasher>>>,
    pending: AtomicUsize,
    /// (signalled flag, condvar) for finalize to wait on. Workers
    /// pulse the condvar when they decrement pending to zero.
    notify: (Mutex<()>, Condvar),
    /// Chunks that have arrived and not yet been folded in, by the
    /// index they were submitted at, and how far the fold has got.
    ///
    /// A hash chain is ordered: `update(a)` then `update(b)` is not
    /// `update(b)` then `update(a)`. Submitting to the IO pool means
    /// tasks finish in whatever order the pool runs them, so folding
    /// each chunk in as its own task completed made the root depend
    /// on completion order. Two runs over the same chunks could root
    /// differently, and a chain whose purpose is deciding whether two
    /// traces are bit-exact would report a mismatch between identical
    /// ones.
    ///
    /// So a task deposits its bytes at its own index and then folds
    /// in whatever unbroken prefix is ready, under the hasher lock.
    /// The hashing still leaves the caller's thread; only the order
    /// is pinned.
    arrivals: Mutex<Arrivals>,
}

/// Chunks waiting to be folded in, and how far the fold has got.
#[derive(Default)]
struct Arrivals {
    /// One slot per submitted chunk, taken as soon as it is folded
    /// in so the bytes are dropped the moment they are spent.
    slots: Vec<Option<Vec<u8>>>,
    /// The next index the hasher wants. Everything below it is in.
    next: usize,
}

/// The arrival table, recovering a lock a panicking fold poisoned.
///
/// Refusing here would strand every later chunk, and the state behind
/// the lock is a vector of byte slots with an index into it: a panic
/// leaves it consistent, because nothing is half-written across the
/// two fields except inside this lock.
fn lock_arrivals(inner: &ChainShared) -> std::sync::MutexGuard<'_, Arrivals> {
    match inner.arrivals.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

/// The hasher, recovering a lock a panicking fold poisoned.
///
/// Dropping the update on a poisoned lock is what the first version
/// of this function did, and it is the same failure this whole
/// rewrite is about: a chunk that never enters the chain gives a root
/// that is wrong and looks fine. Recovering means a panic during one
/// fold costs the panicking chunk and not every chunk after it.
fn lock_hasher(inner: &ChainShared) -> std::sync::MutexGuard<'_, Option<Box<dyn VerifyHasher>>> {
    match inner.hasher.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

/// Put a chunk in its slot and fold in whatever unbroken prefix is
/// now ready.
///
/// Separate from the task that calls it so a test can drive arrivals
/// in any order it likes. The global IO pool is behind a `OnceLock`
/// seeded from an environment variable, so a test cannot install one
/// without fixing it for every other test in the binary; the property
/// that matters is that the fold order does not depend on arrival
/// order, and that is testable here directly.
fn deposit_and_fold(inner: &ChainShared, index: usize, chunk: Vec<u8>) {
    {
        let mut arrivals = lock_arrivals(inner);
        arrivals.slots[index] = Some(chunk);
    }
    fold_ready_prefix(inner);
}

/// Fold in every chunk from the fold cursor up to the first gap.
///
/// Whichever caller finds the prefix ready does the work, so none
/// waits on a particular peer and the hasher still sees submission
/// order.
fn fold_ready_prefix(inner: &ChainShared) {
    let mut arrivals = lock_arrivals(inner);
    let mut hasher = lock_hasher(inner);
    let Some(h) = hasher.as_mut() else { return };
    let mut next = arrivals.next;
    while let Some(slot) = arrivals.slots.get_mut(next) {
        let Some(bytes) = slot.take() else { break };
        h.update(&bytes);
        next += 1;
    }
    arrivals.next = next;
}

/// Running hash-chain over a sequence of stripe outputs. Submit
/// chunks as they become available from compute; call `finalize`
/// when the producer is done to retrieve the 32-byte root.
///
/// Cloning a `VerifyChain` produces another handle that shares the
/// same internal state. Producers that fan out across multiple
/// compute threads can clone a handle per producer.
#[derive(Clone)]
pub struct VerifyChain {
    inner: Arc<ChainShared>,
}

impl VerifyChain {
    /// Build a new chain with the default hasher (BLAKE3 if
    /// `dumpa-experimental` is on; FxFallback otherwise).
    pub fn new() -> Self {
        Self::with_hasher(default_hasher())
    }

    /// Build a new chain with a caller-supplied hasher. The hasher
    /// is consumed when [`Self::finalize`] runs.
    pub fn with_hasher(hasher: Box<dyn VerifyHasher>) -> Self {
        Self {
            inner: Arc::new(ChainShared {
                hasher: Mutex::new(Some(hasher)),
                pending: AtomicUsize::new(0),
                notify: (Mutex::new(()), Condvar::new()),
                arrivals: Mutex::new(Arrivals::default()),
            }),
        }
    }

    /// Submit a chunk of bytes for hashing. If the global IoPool is
    /// enabled, runs asynchronously on an SMT-sibling thread;
    /// otherwise runs inline on the caller thread.
    pub fn submit_chunk(&self, chunk: Vec<u8>) {
        self.inner.pending.fetch_add(1, Ordering::AcqRel);
        // The index is taken here, on the submitting thread, because
        // submission order is the order the caller means and the
        // order the pool happens to run the tasks in is not.
        let index = {
            let mut arrivals = lock_arrivals(&self.inner);
            arrivals.slots.push(None);
            arrivals.slots.len() - 1
        };
        let inner = Arc::clone(&self.inner);
        let task = move || {
            deposit_and_fold(&inner, index, chunk);
            // Decrement pending; if we hit zero, notify any
            // finalize waiter.
            let prev = inner.pending.fetch_sub(1, Ordering::AcqRel);
            if prev == 1 {
                let _g = inner.notify.0.lock();
                inner.notify.1.notify_all();
            }
        };
        match global_io_pool() {
            Some(pool) => pool.submit(task),
            None => task(),
        }
    }

    /// Block until every submitted chunk has been processed, then
    /// consume the hasher and return the 32-byte root.
    ///
    /// # Errors
    ///
    /// Returns `[0u8; 32]` if the hasher has already been finalized
    /// (calling finalize twice on the same chain).
    pub fn finalize(self) -> [u8; 32] {
        // Wait for pending to drop to zero.
        loop {
            if self.inner.pending.load(Ordering::Acquire) == 0 {
                break;
            }
            let mut g = self.inner.notify.0.lock().unwrap();
            // Re-check inside the lock to avoid lost-wakeup.
            if self.inner.pending.load(Ordering::Acquire) == 0 {
                break;
            }
            // park on the condvar; workers signal when pending = 0
            g = self.inner.notify.1.wait(g).unwrap();
            drop(g);
        }
        // Every task has run, so every slot is filled; fold in any
        // prefix a task left behind because its own predecessor had
        // not arrived when it held the lock. Without this, a chain
        // whose last task finished before an earlier one would root
        // over a short prefix and say nothing about it.
        fold_ready_prefix(&self.inner);
        let mut guard = lock_hasher(&self.inner);
        match guard.take() {
            Some(hasher) => hasher.finalize(),
            None => [0u8; 32],
        }
    }

    /// Diagnostic: pending chunk count. Useful for status reporting
    /// or backpressure decisions in the producer.
    pub fn pending_count(&self) -> usize {
        self.inner.pending.load(Ordering::Acquire)
    }
}

impl Default for VerifyChain {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for VerifyChain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VerifyChain")
            .field("pending", &self.pending_count())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fx_fallback_finalize_with_no_chunks_returns_seeded_root() {
        let h: Box<FxFallbackHasher> = Box::default();
        let root = h.finalize();
        // Seeded constant should produce a stable non-zero output.
        assert!(root.iter().any(|&b| b != 0), "expected non-zero root for seeded empty input");
    }

    #[test]
    fn chain_inline_single_chunk_produces_stable_root() {
        let chain = VerifyChain::new();
        chain.submit_chunk(b"hello world".to_vec());
        let root1 = chain.finalize();

        let chain2 = VerifyChain::new();
        chain2.submit_chunk(b"hello world".to_vec());
        let root2 = chain2.finalize();

        assert_eq!(root1, root2, "same input should produce same root");
    }

    #[test]
    fn chain_inline_different_chunks_produce_different_roots() {
        let c1 = VerifyChain::new();
        c1.submit_chunk(b"input A".to_vec());
        let r1 = c1.finalize();

        let c2 = VerifyChain::new();
        c2.submit_chunk(b"input B".to_vec());
        let r2 = c2.finalize();

        assert_ne!(r1, r2, "different inputs should produce different roots");
    }

    #[test]
    fn chain_inline_many_chunks_finalize_blocks_until_done() {
        const N: usize = 32;
        let chain = VerifyChain::new();
        for i in 0..N {
            let chunk = format!("stripe-{:08}", i).into_bytes();
            chain.submit_chunk(chunk);
        }
        // Since IoPool is disabled in tests, submissions run
        // inline; pending should be 0 by the time we reach
        // finalize.
        assert_eq!(chain.pending_count(), 0);
        let root = chain.finalize();
        assert!(root.iter().any(|&b| b != 0));
    }

    #[test]
    fn the_root_does_not_depend_on_the_order_the_chunks_finish() {
        // The property the whole family rests on, and the one no test
        // reached: a chain decides whether two traces are bit-exact,
        // so a root that moves with completion order would report a
        // mismatch between identical traces.
        //
        // Driven through deposit_and_fold rather than through the IO
        // pool because the pool is behind a OnceLock seeded from an
        // environment variable: installing one in a test fixes it for
        // every other test in the binary. What matters is that the
        // fold order is submission order whatever order the arrivals
        // come in, and that is exactly what this drives.
        let chunks: Vec<Vec<u8>> = vec![
            b"first".to_vec(),
            b"second".to_vec(),
            b"third".to_vec(),
            b"fourth".to_vec(),
        ];

        let in_order = VerifyChain::new();
        for c in &chunks {
            in_order.submit_chunk(c.clone());
        }
        let want = in_order.finalize();

        // The same four submitted in the same order, but completing
        // last-to-first, which is what a pool is free to do.
        for arrival in [[3usize, 1, 0, 2], [3, 2, 1, 0], [1, 3, 2, 0]] {
            let chain = VerifyChain::new();
            {
                let mut arrivals = lock_arrivals(&chain.inner);
                arrivals.slots.resize_with(chunks.len(), || None);
            }
            for &i in &arrival {
                deposit_and_fold(&chain.inner, i, chunks[i].clone());
            }
            assert_eq!(
                chain.finalize(),
                want,
                "arrivals {arrival:?} rooted differently from submission order"
            );
        }
    }

    #[test]
    fn a_chunk_still_outstanding_is_folded_in_by_finalize() {
        // A task can leave its chunk in its slot when an earlier one
        // has not arrived. If finalize did not sweep the remainder,
        // the root would cover a prefix and say nothing about it.
        let chunks: Vec<Vec<u8>> = vec![b"a".to_vec(), b"b".to_vec()];

        let in_order = VerifyChain::new();
        for c in &chunks {
            in_order.submit_chunk(c.clone());
        }
        let want = in_order.finalize();

        let chain = VerifyChain::new();
        {
            let mut arrivals = lock_arrivals(&chain.inner);
            arrivals.slots.resize_with(2, || None);
        }
        // Only the second arrives, so nothing can fold yet.
        deposit_and_fold(&chain.inner, 1, chunks[1].clone());
        {
            let arrivals = lock_arrivals(&chain.inner);
            assert_eq!(arrivals.next, 0, "nothing folds while index 0 is missing");
        }
        // The first arrives without anyone folding after it.
        {
            let mut arrivals = lock_arrivals(&chain.inner);
            arrivals.slots[0] = Some(chunks[0].clone());
        }
        assert_eq!(chain.finalize(), want, "finalize must sweep what is left");
    }

    #[test]
    fn chain_finalize_after_finalize_returns_zero_root() {
        let chain = VerifyChain::new();
        chain.submit_chunk(vec![1, 2, 3]);
        let first = chain.clone().finalize();
        let second = chain.finalize();
        assert!(first.iter().any(|&b| b != 0));
        assert_eq!(second, [0u8; 32]);
    }
}
