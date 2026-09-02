//! K_gating: publication-signal dispatch axis.
//!
//! The deque-family design cube exposes K_gating as one of the
//! 6 axes:
//!
//! - **`CounterOnly`** (Chase-Lev family, Fcl): all thieves load
//!   the same `bottom` counter to learn what is publishable.
//!   Publication signal concentrates on ONE cache line. Wins on
//!   smaller-store-buffer cores (in-order ARM, embedded) where
//!   distributed-atomics CAS contention costs more than counter
//!   contention.
//! - **`PerSlot`** (KHL, KHPD): each slot has its own publication
//!   atomic (`seq`). Thieves load `slot[t].seq` on different cache
//!   lines per slot. Distributes contention across many lines.
//!   Wins on store-buffer-rich cores (Zen+, Sapphire Rapids) that
//!   can absorb many concurrent slot-atomic stores.
//!
//! This module provides:
//!
//! - The [`KGating`] enum tag exposed via [`crate::sched::JobPlan`]
//!   for power users that know their host class.
//! - The [`calibrate_k_gating`] function that runs both primitives
//!   in a controlled microbench and returns the winner for this
//!   host.
//! - The [`CALIBRATED_GATING`] static that caches the calibration
//!   result so subsequent calls are zero-cost.
//!
//! The current `crate::sched::arena_local::WorkerCtx` uses KHL
//! exclusively (the Zen+ measurement showed KHL 1.74-1.99x
//! faster than the baseline Chase-Lev deque, while Fcl was only
//! 1.38-1.60x). On hosts where
//! the calibration prefers Fcl, the dispatcher can opt in via the
//! per-call `JobPlan::with_k_gating(KGating::CounterOnly)` tag.

#![allow(clippy::missing_errors_doc)]

use std::sync::OnceLock;
use std::time::Instant;

/// Publication-signal dispatch axis. See module docs.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum KGating {
    /// Single bottom counter (Chase-Lev / Fcl). Publication signal
    /// concentrates on one cache line; wins on smaller-store-buffer
    /// cores.
    CounterOnly,
    /// Per-slot Vyukov sequence (KHL / KHPD). Publication signal
    /// distributes across slot atomics; wins on store-buffer-rich
    /// cores.
    PerSlot,
    /// Use the host calibration's winner (see
    /// [`calibrate_k_gating`]). Default for `JobPlan::new`.
    Auto,
}

impl KGating {
    /// Resolve `Auto` to the calibrated winner; leaves explicit
    /// `CounterOnly` / `PerSlot` unchanged.
    #[inline]
    pub fn resolved(self) -> KGating {
        match self {
            KGating::Auto => *CALIBRATED_GATING.get_or_init(calibrate_k_gating),
            other => other,
        }
    }
}

/// Cached calibration result. Lazily initialized on first
/// `KGating::Auto.resolved()` call.
static CALIBRATED_GATING: OnceLock<KGating> = OnceLock::new();

/// Calibration result with the underlying timings.
#[derive(Debug, Clone, Copy)]
pub struct CalibrationResult {
    /// Per-iter time of the PerSlot (Vyukov) ring on this host.
    pub per_slot_ns: u64,
    /// Per-iter time of the CounterOnly (Chase-Lev) deque on this
    /// host.
    pub counter_only_ns: u64,
    /// Selected winner (smaller time wins).
    pub winner: KGating,
}

/// Run a controlled producer-fast microbench measuring both
/// counter-only and per-slot primitives on this host. Returns the
/// winner. Cached after first call.
///
/// The microbench pushes 64 single-AtomicU64-increment workloads
/// through each primitive serially (no thief contention) and
/// times the drain-to-completion. The serial pattern eliminates
/// noise from thread spawn / scheduler effects; the cross-primitive
/// comparison remains fair because both primitives execute the
/// same workload shape under the same single-thread access
/// pattern.
///
/// Per-host expected outcomes:
/// - Zen+ / Zen 2/3/4 (store-buffer-rich): PerSlot wins
/// - Sapphire Rapids+ (similar): PerSlot wins
/// - ARM Cortex-A (smaller store buffers): CounterOnly may win
/// - In-order embedded cores: CounterOnly likely wins
pub fn calibrate_k_gating() -> KGating {
    run_calibration().winner
}

/// Run the calibration AND return the timing data. Useful for
/// users that want to log or inspect the per-primitive timings.
pub fn calibrate_k_gating_verbose() -> CalibrationResult {
    run_calibration()
}

fn run_calibration() -> CalibrationResult {
    use crate::sched::chase_lev_local::{Steal, new_chase_lev};

    const N: usize = 64;
    const ROUNDS: usize = 4096;

    // Time CounterOnly via ChaseLevLocal<u64>.
    let counter_only_ns = {
        let (w, _s) = new_chase_lev::<u64>(N.next_power_of_two().max(2));
        let t0 = Instant::now();
        for _ in 0..ROUNDS {
            for i in 0..N as u64 {
                w.push(i).expect("push");
            }
            for _ in 0..N {
                match w.pop() {
                    Steal::Success(_) | Steal::Empty | Steal::Retry => {}
                }
            }
        }
        t0.elapsed().as_nanos() as u64 / ROUNDS as u64
    };

    // Time PerSlot via an inline Vyukov ring of u64 slots.
    let per_slot_ns = run_vyukov_ring_calibration(N, ROUNDS);

    let winner = if per_slot_ns < counter_only_ns {
        KGating::PerSlot
    } else {
        KGating::CounterOnly
    };

    CalibrationResult {
        per_slot_ns,
        counter_only_ns,
        winner,
    }
}

/// Inline Vyukov MPMC ring with single-cell u64 slots (no batching
/// to keep the comparison apples-to-apples vs the Chase-Lev path
/// which also uses single-cell slots).
fn run_vyukov_ring_calibration(n: usize, rounds: usize) -> u64 {
    use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};

    let cap = n.next_power_of_two().max(2);
    let mask = (cap - 1) as i64;
    let bottom = AtomicI64::new(0);
    let head = AtomicI64::new(0);
    let slots: Vec<(AtomicU64, std::cell::UnsafeCell<u64>)> = (0..cap)
        .map(|i| (AtomicU64::new(i as u64), std::cell::UnsafeCell::new(0)))
        .collect();

    let t0 = Instant::now();
    for _ in 0..rounds {
        for i in 0..n as u64 {
            let b = bottom.load(Ordering::Relaxed);
            let slot = &slots[(b & mask) as usize];
            while slot.0.load(Ordering::Acquire) != b as u64 {
                std::hint::spin_loop();
            }
            // SAFETY: per-slot seq protocol gates body access;
            // we own the slot for this round.
            unsafe { *slot.1.get() = i; }
            slot.0.store((b as u64) + 1, Ordering::Release);
            bottom.store(b + 1, Ordering::Relaxed);
        }
        for _ in 0..n {
            let t = head.load(Ordering::Acquire);
            let b = bottom.load(Ordering::Acquire);
            if t >= b { break; }
            if head
                .compare_exchange(t, t + 1, Ordering::Acquire, Ordering::Relaxed)
                .is_err()
            {
                continue;
            }
            let slot = &slots[(t & mask) as usize];
            while slot.0.load(Ordering::Acquire) != (t as u64) + 1 {
                std::hint::spin_loop();
            }
            // SAFETY: per-slot seq invariant; sole consumer.
            let v = unsafe { *slot.1.get() };
            std::hint::black_box(v);
            slot.0.store(
                (t as u64) + (cap as u64),
                Ordering::Release,
            );
        }
    }
    let elapsed = t0.elapsed().as_nanos() as u64;
    elapsed / rounds as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gating_variants_are_distinct() {
        assert_ne!(KGating::CounterOnly, KGating::PerSlot);
        assert_ne!(KGating::CounterOnly, KGating::Auto);
        assert_ne!(KGating::PerSlot, KGating::Auto);
    }

    #[test]
    fn resolved_keeps_explicit_choices() {
        assert_eq!(KGating::CounterOnly.resolved(), KGating::CounterOnly);
        assert_eq!(KGating::PerSlot.resolved(), KGating::PerSlot);
        // Auto resolves to a concrete variant, never Auto.
        let auto_resolved = KGating::Auto.resolved();
        assert!(matches!(auto_resolved, KGating::CounterOnly | KGating::PerSlot),
            "Auto must resolve to a concrete variant, got {auto_resolved:?}");
    }

    #[test]
    fn calibration_returns_finite_timings_and_picks_winner() {
        let r = run_calibration();
        assert!(r.per_slot_ns > 0, "per_slot timing must be positive: {r:?}");
        assert!(r.counter_only_ns > 0, "counter_only timing must be positive: {r:?}");
        let expected = if r.per_slot_ns < r.counter_only_ns {
            KGating::PerSlot
        } else {
            KGating::CounterOnly
        };
        assert_eq!(r.winner, expected,
            "winner must match the smaller timing: {r:?}");
    }

    #[test]
    fn calibration_is_cached_for_auto() {
        let r1 = KGating::Auto.resolved();
        let r2 = KGating::Auto.resolved();
        assert_eq!(r1, r2, "auto.resolved() must return the cached value");
    }
}
