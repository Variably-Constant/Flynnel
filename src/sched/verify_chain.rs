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
//! [`VerifyChain`] owns shared fold state behind an `Arc`. Submitting
//! a chunk via [`VerifyChain::submit_chunk`] takes its index and
//! schedules a task onto the IoPool; the task deposits its bytes and
//! then folds in whatever unbroken prefix is ready, if it wins the
//! fold token. [`VerifyChain::finalize`] blocks until all submitted
//! chunks have been processed and returns the root.
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

use core::cell::UnsafeCell;
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};

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

/// Internal shared state for a chain: the ordered fold state, a
/// pending-chunks counter for the finalize barrier, and the handle
/// `root` parks on.
struct ChainShared {
    pending: AtomicUsize,
    /// The index the next submission takes. Taken on the submitting
    /// thread, because submission order is the order the caller means
    /// and the order the pool runs the tasks in is not.
    submitted: AtomicUsize,
    /// The thread blocked in `root`, stored once when it starts
    /// waiting. The task that drives `pending` to zero unparks it.
    ///
    /// One waiter: `root` consumes the chain. A second caller would
    /// find the slot taken and never be woken.
    waiter: OnceLock<std::thread::Thread>,
    /// Chunks that have arrived and not yet been taken into the fold,
    /// newest first. A task pushes its own and then tries for the
    /// fold token; pushing is a compare-exchange and never waits.
    arrived: AtomicPtr<Arrival>,
    /// Held by whichever task is folding. A task that does not get it
    /// returns rather than waiting: what it deposited is already on
    /// `arrived`, so the holder will take it, and the holder rechecks
    /// after releasing so a deposit cannot be stranded.
    folding: AtomicBool,
    /// The hasher and the out-of-order chunks waiting on a
    /// predecessor. Touched only by the holder of `folding`.
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
    /// So a task deposits its bytes under its own index and whoever
    /// holds the token folds in whatever unbroken prefix is ready.
    /// The hashing still leaves the caller's thread; only the order
    /// is pinned. The token is exclusion, and it is not pretending
    /// otherwise: an ordered hash cannot be folded by two threads at
    /// once. What it buys over a lock is that a producer never waits
    /// on it.
    fold: UnsafeCell<FoldState>,
}

// SAFETY: every field is Send, and `fold` is reached only by the
// thread that has won `folding`, which exactly one thread holds at a
// time. `arrived` is a compare-exchange stack of owned boxes.
unsafe impl Sync for ChainShared {}

/// One deposited chunk and the index it was submitted at.
struct Arrival {
    index: usize,
    bytes: Vec<u8>,
    next: *mut Arrival,
}

/// The hasher and the chunks still waiting on a predecessor.
#[derive(Default)]
struct FoldState {
    hasher: Option<Box<dyn VerifyHasher>>,
    /// One slot per submitted chunk, taken as soon as it is folded
    /// in so the bytes are dropped the moment they are spent.
    waiting: Vec<Option<Vec<u8>>>,
    /// The next index the hasher wants. Everything below it is in.
    next: usize,
}

/// Run `act` holding the fold token, or answer `None` when another
/// thread holds it.
///
/// The token is the exclusion an ordered hash needs. What it is not is
/// a wait: a caller that does not get it returns, having already put
/// its chunk somewhere the holder will find.
fn with_fold<R>(inner: &ChainShared, act: impl FnOnce(&mut FoldState) -> R) -> Option<R> {
    if inner
        .folding
        .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
        .is_err()
    {
        return None;
    }
    // SAFETY: the compare-exchange above makes this the only thread
    // reaching the fold state until the store below releases it.
    let state = unsafe { &mut *inner.fold.get() };
    let out = act(state);
    inner.folding.store(false, Ordering::Release);
    Some(out)
}

/// Put a chunk where the fold will find it and fold whatever unbroken
/// prefix is now ready.
///
/// Separate from the task that calls it so a test can drive arrivals
/// in any order it likes. The global IO pool is behind a `OnceLock`
/// seeded from an environment variable, so a test cannot install one
/// without fixing it for every other test in the binary; the property
/// that matters is that the fold order does not depend on arrival
/// order, and that is testable here directly.
fn deposit_and_fold(inner: &ChainShared, index: usize, chunk: Vec<u8>) {
    let node = Box::into_raw(Box::new(Arrival {
        index,
        bytes: chunk,
        next: core::ptr::null_mut(),
    }));
    loop {
        let head = inner.arrived.load(Ordering::Acquire);
        // SAFETY: this box is not published until the compare-exchange
        // below succeeds, so nothing else reaches it.
        unsafe { (*node).next = head };
        if inner
            .arrived
            .compare_exchange(head, node, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
        {
            break;
        }
    }
    fold_ready_prefix(inner);
}

/// Fold in every chunk from the fold cursor up to the first gap.
///
/// Whichever caller wins the token does the work, so none waits on a
/// particular peer and the hasher still sees submission order.
fn fold_ready_prefix(inner: &ChainShared) {
    loop {
        let folded = with_fold(inner, |state| {
            // Take the whole deposit stack in one swap, then order it
            // by the index each chunk was submitted at.
            let mut chain = inner.arrived.swap(core::ptr::null_mut(), Ordering::AcqRel);
            while !chain.is_null() {
                // SAFETY: a node on the stack was built by
                // deposit_and_fold and is taken exactly once, by the
                // swap above.
                let Arrival { index, bytes, next } = *unsafe { Box::from_raw(chain) };
                chain = next;
                if state.waiting.len() <= index {
                    state.waiting.resize_with(index + 1, || None);
                }
                state.waiting[index] = Some(bytes);
            }
            if let Some(h) = state.hasher.as_mut() {
                let mut next = state.next;
                while let Some(slot) = state.waiting.get_mut(next) {
                    let Some(bytes) = slot.take() else { break };
                    h.update(&bytes);
                    next += 1;
                }
                state.next = next;
            }
        });
        if folded.is_none() {
            // Someone else holds the token. What this caller deposited
            // is on the stack, and the holder rechecks before leaving.
            return;
        }
        // A deposit that landed while the token was held, whose
        // depositor found the token taken and returned, would sit here
        // until the next submission. Take it now.
        if inner.arrived.load(Ordering::Acquire).is_null() {
            return;
        }
    }
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
                pending: AtomicUsize::new(0),
                submitted: AtomicUsize::new(0),
                waiter: OnceLock::new(),
                arrived: AtomicPtr::new(core::ptr::null_mut()),
                folding: AtomicBool::new(false),
                fold: UnsafeCell::new(FoldState {
                    hasher: Some(hasher),
                    waiting: Vec::new(),
                    next: 0,
                }),
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
        let index = self.inner.submitted.fetch_add(1, Ordering::AcqRel);
        let inner = Arc::clone(&self.inner);
        let task = move || {
            deposit_and_fold(&inner, index, chunk);
            // Decrement pending; if we hit zero, notify any
            // finalize waiter.
            let prev = inner.pending.fetch_sub(1, Ordering::AcqRel);
            if prev == 1
                && let Some(waiter) = inner.waiter.get()
            {
                waiter.unpark();
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
            // Published before the count is re-read. A task that
            // finishes after this finds the handle and unparks; one
            // that finished before it left pending at zero, which the
            // loop condition reads next. park keeps a permit either
            // way, so a wake between the two is not lost.
            self.inner.waiter.get_or_init(std::thread::current);
            if self.inner.pending.load(Ordering::Acquire) == 0 {
                break;
            }
            std::thread::park();
        }
        // Every task has run, so every chunk has been deposited; fold
        // in any prefix a task left behind because its own predecessor
        // had not arrived when it held the token. Without this, a chain
        // whose last task finished before an earlier one would root
        // over a short prefix and say nothing about it.
        fold_ready_prefix(&self.inner);
        // Every task has returned, so nothing else wants the token.
        // Taking it rather than assuming that is what makes the read of
        // the hasher sound on its own terms.
        loop {
            if let Some(root) = with_fold(&self.inner, |state| match state.hasher.take() {
                Some(hasher) => hasher.finalize(),
                None => [0u8; 32],
            }) {
                return root;
            }
            std::hint::spin_loop();
        }
    }

    /// Diagnostic: pending chunk count. Useful for status reporting
    /// or backpressure decisions in the producer.
    pub fn pending_count(&self) -> usize {
        self.inner.pending.load(Ordering::Acquire)
    }
}

impl Drop for ChainShared {
    fn drop(&mut self) {
        // A chain dropped without a finalize can still hold deposits
        // nothing folded. The stack owns those boxes.
        let mut chain = *self.arrived.get_mut();
        while !chain.is_null() {
            // SAFETY: this is the last owner of the chain, so no task
            // can be reaching the stack.
            let node = *unsafe { Box::from_raw(chain) };
            chain = node.next;
        }
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
        // Only the second arrives, so nothing can fold yet.
        deposit_and_fold(&chain.inner, 1, chunks[1].clone());
        let cursor = with_fold(&chain.inner, |state| state.next)
            .expect("no task holds the fold token in this test");
        assert_eq!(cursor, 0, "nothing folds while index 0 is missing");
        // The first arrives without anyone folding after it.
        with_fold(&chain.inner, |state| {
            if state.waiting.is_empty() {
                state.waiting.push(None);
            }
            state.waiting[0] = Some(chunks[0].clone());
        })
        .expect("no task holds the fold token in this test");
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
