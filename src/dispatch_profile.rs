//! [`DispatchProfile`]: scheduler-native classification of work for
//! per-dispatch tuning.
//!
//! The scheduler needs two facts about a job to dispatch it well:
//!
//! 1. **Does SMT help or hurt?** Latency-bound work (long dependency
//!    chains stalling on sqrt / div / FMA latency) leaves dispatch
//!    bubbles SMT siblings fill. Port-saturated work (chained IMUL /
//!    saturating FMA throughput) contests the same execution unit on
//!    an SMT sibling and gains nothing.
//! 2. **Roughly how expensive is each element?** Determines the
//!    optimal leaf count for the bisect (per-leaf dispatch overhead
//!    should be a small fraction of per-leaf work) and the
//!    inline-collapse threshold (don't enter the pool at all if the
//!    whole job is under the dispatch floor).
//!
//! `DispatchProfile` names the work in terms of these two
//! scheduler-native facts. It deliberately avoids domain-specific
//! kernel names (sqrt, mul, etc.) so callers across numerical math,
//! signal processing, graphics, ML, and string processing can pick
//! the same profile from their own perspective.
//!
//! Power-user knobs ([`crate::JobPlan::with_smt`],
//! [`crate::JobPlan::with_cost_ns_per_elem`],
//! [`crate::JobPlan::with_oversubscription_log2`],
//! [`crate::JobPlan::with_workers`]) override profile defaults
//! at any layer.

use crate::op_class::OpClass;

/// Scheduler-native classification of a dispatch. Picks SMT
/// activation and per-element cost estimate together so the
/// downstream tuning (leaf-count cap, inline-collapse threshold,
/// backend selection) has the inputs it needs.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum DispatchProfile {
    /// Long-dependency-chain work where SMT-2 siblings on the
    /// same physical core fill the dispatch bubbles between
    /// latency stalls. Default cost estimate is moderate-to-high
    /// per element because latency-bound kernels typically chain
    /// many ops per data element.
    ///
    /// Picks: SMT siblings active, 4x oversubscription (steal
    /// headroom for any per-element cost variance), default cost
    /// 600 ns/elem.
    ///
    /// Workload examples: chained sqrt, chained div, recursive
    /// Newton iteration, FMA chains with register dependencies,
    /// branchy adaptive integrators.
    LatencyBound,

    /// Issue-port-saturated work where SMT siblings contest the
    /// same execution unit and produce no extra throughput.
    /// Default cost estimate is low per element because
    /// throughput-bound kernels are usually one issue per cycle.
    ///
    /// Picks: SMT siblings parked, 2x oversubscription (default
    /// steal headroom), default cost 12 ns/elem.
    ///
    /// Workload examples: chained IMUL on single-issue port,
    /// saturating FMA throughput, packed-loop SIMD with no inter-
    /// element data dependency.
    PortBound,

    /// Memory work where each leaf does many independent loads
    /// with unpredictable addresses (gather / scatter / pointer
    /// chase). The second SMT sibling overlaps its own
    /// cache-miss loads with the first sibling's stalls, which
    /// gives net throughput.
    ///
    /// Picks: SMT siblings active, 2x oversubscription, default
    /// cost 50 ns/elem.
    ///
    /// Workload examples: sparse matvec, pointer-chasing tree
    /// traversals, indirect gather from a permutation array,
    /// graph adjacency walks, hash-table probes.
    MemoryBound,

    /// Streaming sequential memory work where the bottleneck is
    /// SHARED memory bandwidth at the per-core level. SMT
    /// siblings on the same physical core would COMPETE for the
    /// same L2/L3 bandwidth instead of helping; both threads on
    /// the same core run at roughly half the bandwidth of one.
    ///
    /// Picks: SMT siblings parked, 1x oversubscription (avoid
    /// over-fragmenting a steady-state stream), default cost
    /// 50 ns/elem.
    ///
    /// Workload examples: large byte scans (grep), histogram on
    /// large arrays, sequential prefix-sum block sums, per-pixel
    /// image kernels (rgb->gray), CSV parsing on a contiguous
    /// buffer.
    Streaming,

    /// Caller has no profile information. Falls back to
    /// conservative defaults: SMT parked, 2x oversubscription,
    /// no cost estimate. Per-call cost-derived oversubscription
    /// tuning is disabled for `Unspecified` dispatches.
    Unspecified,
}

impl DispatchProfile {
    /// Default per-element cost estimate for this profile in
    /// nanoseconds. Representative values calibrated against
    /// typical workloads in each class on x86-64 silicon; the
    /// caller overrides via [`crate::JobPlan::with_cost_ns_per_elem`]
    /// when better data is available.
    pub fn default_ns_per_elem(self) -> Option<u32> {
        match self {
            Self::LatencyBound => Some(600),
            Self::PortBound => Some(12),
            Self::MemoryBound => Some(50),
            Self::Streaming => Some(50),
            Self::Unspecified => None,
        }
    }

    /// Default leaf-count oversubscription factor for this profile,
    /// as `log2(leaves_per_worker)`. The bisect targets `workers *
    /// 2^oversubscription_log2` leaves. Higher values create more
    /// leaves (more steal headroom for variance); lower values
    /// reduce per-leaf dispatch overhead.
    ///
    /// - `LatencyBound`: 2 (4x oversubscription) - per-element cost
    ///   often varies (sqrt iterations converging at different
    ///   rates, branchy adaptive integrators), so steal headroom
    ///   matters.
    /// - `PortBound`: 1 (2x oversubscription) - per-element cost
    ///   is uniform, so dispatch overhead matters more than steal
    ///   headroom.
    /// - `MemoryBound`: 1 (2x oversubscription) - cache-miss
    ///   patterns are usually uniform across a slice.
    /// - `Unspecified`: 1 (2x oversubscription) - conservative.
    pub fn default_oversubscription_log2(self) -> u8 {
        match self {
            Self::LatencyBound => 2,
            Self::PortBound => 1,
            Self::MemoryBound => 1,
            // Streaming: 1 (2x oversubscription). Empirically
            // validated on rgb_to_gray (4M slots, ~1 ns / slot):
            // oversub_log2=0 measured 1.93x slower than rayon
            // because any straggler worker stalls the whole call
            // (8 leaves on 8 cores = zero recovery), while
            // oversub_log2=1 gives 16 leaves across 8 physical
            // cores so the SLAW splitter's replenish path has
            // headroom. histogram + monte_carlo_pi + transpose
            // show the same regression at oversub=0, confirming 1
            // as the right setting for bandwidth-bound streaming
            // work.
            Self::Streaming => 1,
            Self::Unspecified => 1,
        }
    }

    /// Whether SMT siblings should activate for this profile.
    /// Latency-bound and memory-bound work benefits; port-bound
    /// and unspecified default to no SMT (primaries only).
    pub fn is_latency_bound(self) -> bool {
        matches!(self, Self::LatencyBound | Self::MemoryBound)
    }

    /// Whether SIMC/MIMC mailbox routing fits this profile. The
    /// scheduler auto-routes the right-half of join through the
    /// SMT-sibling mailbox when this is true AND the sibling has
    /// nothing else queued (gated at the call site).
    ///
    /// **Current measurement (realistic_bench, Zen+ R7 2700,
    /// 2026-06-06):** NO profile classification wins by routing
    /// `for_each_chunk`'s recursive bisection through the SMT-
    /// sibling mailbox. The bisect tree fans out across all
    /// primaries via random-victim peer steal; pinning right-
    /// halves to the SMT pair concentrates load + starves
    /// cross-CCX peers (Compute/100k regressed 7x, Heavy/100k
    /// regressed 6x). The mailbox path is exercised through the
    /// `JobPlan::with_mailbox_routing(true)` builder by power
    /// users who know their workload fits the SIMC fan-out
    /// shape (single producer, locality-warm consumer, parallelism-
    /// limited workload).
    pub fn use_mailbox_routing(self) -> bool {
        // Conservative default per the empirical finding above.
        // Future profile variants (e.g. a SimcLocal class) can
        // override; none of the current four does.
        false
    }

    /// Per-profile deque-tier hint. The scheduler's
    /// `arena::join` pushes the right-half to this tier so the
    /// steal discipline pins the work to peers at the right
    /// coherence distance.
    ///
    /// **Current measurement:** same finding as
    /// [`Self::use_mailbox_routing`]. Pinning right-halves to a
    /// narrower tier than Public produces the same SMT-pair load
    /// concentration regression. Power users opt in via
    /// `JobPlan::with_deque_tier_hint(tier)` when they know the
    /// workload's locality structure matches the chosen tier.
    pub fn deque_tier_hint(self) -> Option<crate::sched::deque_tier::DequeTier> {
        None
    }
}

/// `DispatchProfile` is itself the canonical [`OpClass`]
/// implementation: the trait is the abstraction, this enum is the
/// default concrete instance. Domain-specific enums (a numerical-
/// kernel enum in a math crate, a shader-class enum in a graphics
/// crate, etc.) implement `OpClass` by mapping their variants to a
/// `DispatchProfile` and returning that from `dispatch_profile()`.
impl OpClass for DispatchProfile {
    fn is_latency_bound(&self) -> bool {
        DispatchProfile::is_latency_bound(*self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn latency_bound_profiles_activate_smt() {
        assert!(DispatchProfile::LatencyBound.is_latency_bound());
        assert!(DispatchProfile::MemoryBound.is_latency_bound());
        assert!(!DispatchProfile::PortBound.is_latency_bound());
        assert!(!DispatchProfile::Streaming.is_latency_bound());
        assert!(!DispatchProfile::Unspecified.is_latency_bound());
    }

    #[test]
    fn default_cost_estimates_are_set_for_classified_profiles() {
        assert_eq!(DispatchProfile::LatencyBound.default_ns_per_elem(), Some(600));
        assert_eq!(DispatchProfile::PortBound.default_ns_per_elem(), Some(12));
        assert_eq!(DispatchProfile::MemoryBound.default_ns_per_elem(), Some(50));
        assert_eq!(DispatchProfile::Streaming.default_ns_per_elem(), Some(50));
        assert_eq!(DispatchProfile::Unspecified.default_ns_per_elem(), None);
    }

    #[test]
    fn default_oversubscription_log2_matches_design() {
        // LatencyBound: 4x oversubscription = log2 = 2
        assert_eq!(DispatchProfile::LatencyBound.default_oversubscription_log2(), 2);
        // The other profiles: 2x oversubscription = log2 = 1
        assert_eq!(DispatchProfile::PortBound.default_oversubscription_log2(), 1);
        assert_eq!(DispatchProfile::MemoryBound.default_oversubscription_log2(), 1);
        // Streaming: 1 (2x oversubscription). Empirically validated
        // on rgb_to_gray + histogram + monte_carlo_pi: setting 0
        // measured 1.93x rgb_to_gray + 2.11x histogram vs rayon
        // because straggler workers stalled the whole call.
        assert_eq!(DispatchProfile::Streaming.default_oversubscription_log2(), 1);
        assert_eq!(DispatchProfile::Unspecified.default_oversubscription_log2(), 1);
    }

    #[test]
    fn op_class_trait_agrees_with_inherent() {
        for profile in [
            DispatchProfile::LatencyBound,
            DispatchProfile::PortBound,
            DispatchProfile::MemoryBound,
            DispatchProfile::Streaming,
            DispatchProfile::Unspecified,
        ] {
            assert_eq!(
                <DispatchProfile as OpClass>::is_latency_bound(&profile),
                profile.is_latency_bound(),
                "trait must agree with inherent for {profile:?}",
            );
        }
    }
}
