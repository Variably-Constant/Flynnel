//! Coherence-distance tier for per-worker deques.
//!
//! The cross-process KHPD architecture maps ONE publication line
//! per topology distance: SMT-sibling, intra-CCX, cross-CCX,
//! cross-NUMA. The in-process scheduler mirrors that shape:
//! each worker carries N deques, one per [`DequeTier`]. The push site
//! picks a tier based on the most-likely-thief distance; the peer-
//! steal site walks tiers in distance order, never reaching CLOSER
//! tiers than the topological distance to the victim.
//!
//! ## Asymmetric steal discipline
//!
//! A thief at distance `D` from the owner is allowed to steal from
//! the owner's tiers `[D..N_TIERS]` - that is, the tier MATCHING the
//! thief's distance, plus any wider tiers. The thief is NOT allowed
//! to steal from CLOSER tiers. This keeps SmtLocal deques exclusive
//! to SMT-sibling pairs (no cross-CCX thief invalidates the SMT
//! sibling's L1d copy) and reserves Public deques for genuinely
//! cross-class work.
//!
//! ## Default tier picks
//!
//! - **Producer-side push default**: [`DequeTier::Public`]. The
//!   Public tier is reachable by every thief regardless of
//!   coherence distance, so a hot producer pushing at the default
//!   does not pin its work to one physical-core pair. See the
//!   doc-comment on [`DequeTier::default`] for the measured
//!   evidence (an SmtLocal default regresses realistic_bench
//!   Heavy/100k 35x because peer-steal is blocked from the
//!   SMT-sibling deque).
//! - **Narrower tiers**: callers that KNOW locality wins (recursive
//!   splits handing off to the SMT sibling; SLAW-style bisection
//!   leaves) call
//!   [`crate::sched::arena_local::WorkerCtx::push_tier`] with a
//!   narrower tier to keep the cache line on the closest core.
//!
//! Tier hinting through the dispatcher-level routing layer lives in
//! the unified [`crate::sched::plan::JobPlan`] surface.
//!
//! ## Number of tiers
//!
//! Four variants enumerated, but [`peer_distance`] only distinguishes
//! three on hosts where NUMA detection beyond cluster_size_log2 is
//! unavailable: SMT-sibling, intra-CCX, Public (cross-CCX collapses
//! into Public). The CrossCcx variant remains in the enum for
//! API stability + topology-extension upgrades.

#![allow(clippy::missing_docs_in_private_items)]

/// Coherence-distance tier label. Indexes into the per-worker deque
/// vector + the steal-allowed matrix.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DequeTier {
    /// Same physical core; SMT siblings share L1d. Cheapest
    /// cross-thread cache transfer (store-to-load forwarding /
    /// shared L1d). Default for push.
    SmtLocal = 0,
    /// Same CCX (or Intel module). Shared L2/L3 cluster; no L1d
    /// sharing across non-sibling cores. ~30 ns cross-thread bounce.
    IntraCcx = 1,
    /// Same CCD/socket, different CCX. Shared L3 across CCXs;
    /// ~50-100 ns bounce.
    CrossCcx = 2,
    /// Cross-NUMA / global. Any worker may steal. ~200 ns bounce.
    Public = 3,
}

/// Number of distinct tiers + array length for per-worker deque
/// matrices.
pub const N_TIERS: usize = 4;

impl DequeTier {
    /// Const conversion to array index.
    pub const fn idx(self) -> usize {
        self as usize
    }

    /// Round-trip from a u8 index. Returns `None` for out-of-range
    /// indices.
    pub const fn from_idx(idx: usize) -> Option<Self> {
        match idx {
            0 => Some(Self::SmtLocal),
            1 => Some(Self::IntraCcx),
            2 => Some(Self::CrossCcx),
            3 => Some(Self::Public),
            _ => None,
        }
    }

    /// Diagnostic label.
    pub const fn label(self) -> &'static str {
        match self {
            Self::SmtLocal => "smt_local",
            Self::IntraCcx => "intra_ccx",
            Self::CrossCcx => "cross_ccx",
            Self::Public => "public",
        }
    }

    /// All four tiers in distance order (closest first). Used by
    /// `find_work` to walk tiers from cheapest to widest.
    pub const fn all() -> [Self; N_TIERS] {
        [Self::SmtLocal, Self::IntraCcx, Self::CrossCcx, Self::Public]
    }
}

impl Default for DequeTier {
    /// Default push target: [`DequeTier::Public`].
    ///
    /// **The default is Public, not SmtLocal, despite SmtLocal being
    /// the tightest cache.** Reason: any peer at any distance must
    /// be able to steal from the default-pushed deque. SmtLocal-tier
    /// pushes are visible ONLY to SMT-sibling thieves (the steal
    /// discipline enforced by [`thief_may_steal`]), so a single hot
    /// producer pushing to SmtLocal pins all its work to one physical
    /// core pair - catastrophic regression on workloads that need
    /// cross-CCX parallelism. Producers that KNOW the locality
    /// benefit applies (e.g., recursive splits that hand off to the
    /// SMT sibling) call [`crate::sched::arena_local::WorkerCtx::push_tier`]
    /// explicitly with a narrower tier.
    fn default() -> Self {
        Self::Public
    }
}

/// Compute the coherence-distance tier between two worker indices,
/// given the host's CCX size (in logical cores). Assumes the common
/// x86 enumeration where SMT siblings are adjacent: (0,1), (2,3), ...
///
/// Returns:
/// - [`DequeTier::SmtLocal`] when the two indices share the same
///   physical core (their indices/2 are equal).
/// - [`DequeTier::IntraCcx`] when they share the same CCX (their
///   indices/ccx_size are equal) but are NOT SMT siblings.
/// - [`DequeTier::Public`] otherwise (cross-CCX / cross-NUMA;
///   collapsed to Public on hosts without finer topology probing).
///
/// `self_idx == peer_idx` returns [`DequeTier::SmtLocal`] (a worker
/// is its own SMT sibling for the purposes of this function); callers
/// should always exclude self before consulting this helper.
#[inline]
pub fn peer_distance(self_idx: usize, peer_idx: usize, ccx_size: usize) -> DequeTier {
    if ccx_size == 0 {
        return DequeTier::Public;
    }
    if self_idx >> 1 == peer_idx >> 1 {
        return DequeTier::SmtLocal;
    }
    if self_idx / ccx_size == peer_idx / ccx_size {
        return DequeTier::IntraCcx;
    }
    DequeTier::Public
}

/// Whether a thief at distance `thief_distance` is allowed to steal
/// from a deque at `owner_tier`. The rule:
///
/// > Thief at distance `D` may steal from tiers `[D..N_TIERS]`.
///
/// I.e. the deque tier must be >= the thief's distance. SmtLocal
/// deques can only be stolen by SMT-sibling thieves (distance == 0);
/// Public deques are open to everyone (any distance allowed).
pub fn thief_may_steal(thief_distance: DequeTier, owner_tier: DequeTier) -> bool {
    (owner_tier as u8) >= (thief_distance as u8)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn idx_round_trip_for_all_four_variants() {
        for t in DequeTier::all() {
            assert_eq!(DequeTier::from_idx(t.idx()), Some(t));
        }
    }

    #[test]
    fn from_idx_rejects_out_of_range() {
        assert!(DequeTier::from_idx(N_TIERS).is_none());
        assert!(DequeTier::from_idx(99).is_none());
    }

    #[test]
    fn all_returns_distance_order() {
        let order = DequeTier::all();
        for w in order.windows(2) {
            assert!((w[0] as u8) < (w[1] as u8), "tier order must be ascending");
        }
    }

    #[test]
    fn default_is_public_for_steal_compatibility() {
        // The default tier MUST be Public so any peer at any
        // distance is allowed to steal the default-pushed work.
        // Narrower defaults pin work to physical-core pairs and
        // wreck cross-CCX parallelism (verified empirically:
        // realistic_bench Heavy/100k regressed 35x when default
        // was SmtLocal).
        assert_eq!(DequeTier::default(), DequeTier::Public);
    }

    #[test]
    fn peer_distance_smt_siblings_adjacent_indices() {
        // (0,1), (2,3), (4,5), ... are SMT siblings on the standard
        // x86 enumeration. CCX size irrelevant for sibling detection.
        assert_eq!(peer_distance(0, 1, 8), DequeTier::SmtLocal);
        assert_eq!(peer_distance(2, 3, 8), DequeTier::SmtLocal);
        assert_eq!(peer_distance(7, 6, 8), DequeTier::SmtLocal);
    }

    #[test]
    fn peer_distance_intra_ccx_non_siblings() {
        // Worker 0 and worker 2 are in the same CCX (both < 8) but
        // not SMT siblings.
        assert_eq!(peer_distance(0, 2, 8), DequeTier::IntraCcx);
        assert_eq!(peer_distance(0, 6, 8), DequeTier::IntraCcx);
    }

    #[test]
    fn peer_distance_cross_ccx_returns_public() {
        // Worker 0 (CCX 0) and worker 8 (CCX 1) are in different CCXs.
        assert_eq!(peer_distance(0, 8, 8), DequeTier::Public);
        assert_eq!(peer_distance(3, 11, 8), DequeTier::Public);
    }

    #[test]
    fn peer_distance_zero_ccx_size_returns_public() {
        // Defensive: ccx_size=0 means no CCX detection; treat all
        // peers as Public-tier reachable.
        assert_eq!(peer_distance(0, 1, 0), DequeTier::Public);
        assert_eq!(peer_distance(5, 99, 0), DequeTier::Public);
    }

    #[test]
    fn thief_may_steal_distance_zero_can_take_all_tiers() {
        // SMT-sibling thief (distance 0): allowed everywhere.
        for tier in DequeTier::all() {
            assert!(thief_may_steal(DequeTier::SmtLocal, tier),
                "SMT-sibling thief must reach {tier:?}");
        }
    }

    #[test]
    fn thief_may_steal_distance_three_only_public() {
        // Cross-NUMA thief (distance 3 = Public): only allowed at
        // the Public tier; SmtLocal/IntraCcx/CrossCcx are blocked.
        assert!(!thief_may_steal(DequeTier::Public, DequeTier::SmtLocal));
        assert!(!thief_may_steal(DequeTier::Public, DequeTier::IntraCcx));
        assert!(!thief_may_steal(DequeTier::Public, DequeTier::CrossCcx));
        assert!(thief_may_steal(DequeTier::Public, DequeTier::Public));
    }

    #[test]
    fn thief_may_steal_distance_one_blocks_smt_local_only() {
        // Intra-CCX thief (distance 1): blocked from SmtLocal,
        // allowed at IntraCcx + wider.
        assert!(!thief_may_steal(DequeTier::IntraCcx, DequeTier::SmtLocal));
        assert!(thief_may_steal(DequeTier::IntraCcx, DequeTier::IntraCcx));
        assert!(thief_may_steal(DequeTier::IntraCcx, DequeTier::CrossCcx));
        assert!(thief_may_steal(DequeTier::IntraCcx, DequeTier::Public));
    }
}
