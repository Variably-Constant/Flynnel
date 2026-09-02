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
            }),
        }
    }

    /// Submit a chunk of bytes for hashing. If the global IoPool is
    /// enabled, runs asynchronously on an SMT-sibling thread;
    /// otherwise runs inline on the caller thread.
    pub fn submit_chunk(&self, chunk: Vec<u8>) {
        self.inner.pending.fetch_add(1, Ordering::AcqRel);
        let inner = Arc::clone(&self.inner);
        let task = move || {
            // Update the hasher (briefly hold the mutex).
            if let Ok(mut guard) = inner.hasher.lock()
                && let Some(h) = guard.as_mut()
            {
                h.update(&chunk);
            }
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
        let mut guard = self.inner.hasher.lock().unwrap();
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
    fn chain_finalize_after_finalize_returns_zero_root() {
        let chain = VerifyChain::new();
        chain.submit_chunk(vec![1, 2, 3]);
        let first = chain.clone().finalize();
        let second = chain.finalize();
        assert!(first.iter().any(|&b| b != 0));
        assert_eq!(second, [0u8; 32]);
    }
}
