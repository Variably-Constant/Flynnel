//! Declarative workload-shape API for `JobPlan`.
//!
//! The user describes their workload declaratively and the
//! dispatcher picks the primitive + optimization mode. The
//! benefit: user code stops naming `KGating::PerSlot` /
//! `push_burst` / `with_mailbox_routing` directly; the framework
//! maps the shape to those knobs.
//!
//! Hint inference at build-time has zero per-op cost; the mapping
//! shape -> {k_gating, mailbox, oversubscription} runs ONCE at
//! plan construction and the per-call dispatch path stays direct
//! atomic ops.
//!
//! ## The shape taxonomy
//!
//! Five workload shapes that map cleanly to the Flynn-axis
//! taxonomy + the K-hierarchy:
//!
//! | Shape          | Flynn axis | K_gating                    | Burst | Mailbox                       |
//! |----------------|------------|-----------------------------|-------|-------------------------------|
//! | `Streaming`    | SISD       | Auto (host calibration)     | no    | no                            |
//! | `ProducerFast` | SIMC       | PerSlot                     | YES   | no                            |
//! | `WorkSteal`    | MIMD       | Auto (host calibration)     | no    | no                            |
//! | `Cooperative`  | SIMC/MIMC  | PerSlot                     | YES   | YES when `n_cores >= 8`       |
//! | `VariantRace`  | MISD       | PerSlot                     | no    | no                            |
//!
//! ## Mapping vs the existing K-axis hints
//!
//! [`WorkloadShape`] is the HIGH-LEVEL API. Internally it maps to
//! the existing low-level hints already on `JobPlan`:
//! - `k_gating`: which publication signal protocol
//! - `use_mailbox_routing`: SIMC owner-directed hand-off
//! - `oversubscription_log2`: leaf-count multiplier for splitting
//!
//! Power users keep direct access to the low-level hints; the
//! shape API is the convenience surface that captures the common
//! cases without forcing the user to learn the K-axis vocabulary.

#![allow(clippy::missing_errors_doc)]

use crate::sched::k_gating::KGating;

/// Declarative high-level workload-shape hint. Maps to the
/// low-level K-axis hints on [`crate::sched::JobPlan`] via
/// [`JobPlan::with_workload_shape`](crate::sched::JobPlan::with_workload_shape).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum WorkloadShape {
    /// Single-producer streaming: SISD work that goes through the
    /// scheduler only for orchestration. Maps to default K_gating
    /// (Auto -> host-calibrated), no burst, no mailbox.
    Streaming,
    /// Producer-fast burst: SIMC pattern where the producer emits
    /// N jobs rapidly (cooperative_join_n_flat fan-out, parallel
    /// loop chunking). Maps to PerSlot K_gating + burst push (3
    /// jobs per cache-line slot, K_inner=3 amortization).
    ProducerFast {
        /// Approximate burst size (number of jobs the producer
        /// emits between waits). Used to size the oversubscription
        /// hint; larger bursts get more steal headroom.
        burst: u32,
    },
    /// Many-consumer work-stealing: MIMD pattern where independent
    /// workers steal from each other's deques. Maps to host-
    /// calibrated K_gating (PerSlot on store-buffer-rich silicon,
    /// CounterOnly on smaller-buffer cores).
    WorkSteal {
        /// Number of cooperating consumers.
        n_consumers: u32,
        /// Approximate per-consumer batch size.
        batch_size: u32,
    },
    /// Cooperative cross-core work: SIMC/MIMC pattern with
    /// owner-directed mailbox routing for intra-CCX work. Maps to
    /// PerSlot + mailbox routing enabled + burst push.
    Cooperative {
        /// Number of cores participating in the cooperative
        /// dispatch. Driving the K_unified axis.
        n_cores: u32,
    },
    /// Variant racing: MISD pattern where the same work runs in
    /// multiple variants (correct / faithful / fast) and the first
    /// to finish wins. Maps to PerSlot K_gating (each variant gets
    /// its own slot header carrying the variant tag), no burst
    /// (each push is a distinct race entry).
    VariantRace {
        /// Number of variant racers (typically 2-3).
        n_variants: u32,
    },
}

/// Plan-level adjustments derived from a [`WorkloadShape`]. Used
/// by [`crate::JobPlan::with_workload_shape`] to set multiple
/// low-level hints atomically.
#[derive(Copy, Clone, Debug)]
pub struct WorkloadShapeHints {
    /// K_gating axis hint (per-slot vs counter-only publication).
    pub k_gating: KGating,
    /// Whether to enable mailbox routing for the right-half push
    /// in `join`.
    pub use_mailbox_routing: bool,
    /// Oversubscription factor `log2(leaves_per_worker)`. None
    /// means "use the plan's existing oversubscription_log2".
    pub oversubscription_log2: Option<u8>,
    /// Whether the dispatch site should use the burst push path
    /// (push_burst + flush_all) to unlock K_inner=3 amortization.
    /// Read by [`crate::sched::cooperative::cooperative_join_n_flat`]
    /// and similar fan-out call sites.
    pub use_burst: bool,
}

impl WorkloadShape {
    /// Resolve the shape to the low-level hints. Pure mapping;
    /// zero per-op cost.
    pub const fn hints(self) -> WorkloadShapeHints {
        match self {
            WorkloadShape::Streaming => WorkloadShapeHints {
                k_gating: KGating::Auto,
                use_mailbox_routing: false,
                oversubscription_log2: None,
                use_burst: false,
            },
            WorkloadShape::ProducerFast { burst } => WorkloadShapeHints {
                k_gating: KGating::PerSlot,
                use_mailbox_routing: false,
                // Larger bursts get more oversubscription so each
                // worker has steal-headroom for variance.
                oversubscription_log2: Some(if burst >= 64 { 2 } else { 1 }),
                use_burst: true,
            },
            WorkloadShape::WorkSteal {
                n_consumers,
                batch_size: _,
            } => WorkloadShapeHints {
                k_gating: KGating::Auto,
                use_mailbox_routing: false,
                // Many consumers benefit from oversubscription;
                // few don't.
                oversubscription_log2: Some(if n_consumers >= 8 { 2 } else { 1 }),
                use_burst: false,
            },
            WorkloadShape::Cooperative { n_cores } => WorkloadShapeHints {
                k_gating: KGating::PerSlot,
                // Mailbox routing pays off when n_cores >= n_workers
                // (the cooperative gate inside fan_out_in_worker
                // demotes mailbox -> deque when N < n_workers).
                // Enabling at the plan level lets the call site
                // honor the gate without forcing it.
                use_mailbox_routing: n_cores >= 8,
                oversubscription_log2: None,
                use_burst: true,
            },
            WorkloadShape::VariantRace { n_variants: _ } => WorkloadShapeHints {
                k_gating: KGating::PerSlot,
                use_mailbox_routing: false,
                // Each variant racer is its own entry; oversubscription
                // matches racer count.
                oversubscription_log2: Some(0),
                use_burst: false,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn streaming_maps_to_minimal_hints() {
        let h = WorkloadShape::Streaming.hints();
        assert_eq!(h.k_gating, KGating::Auto);
        assert!(!h.use_mailbox_routing);
        assert!(!h.use_burst);
    }

    #[test]
    fn producer_fast_enables_burst_and_per_slot() {
        let h = WorkloadShape::ProducerFast { burst: 64 }.hints();
        assert_eq!(h.k_gating, KGating::PerSlot);
        assert!(h.use_burst);
        assert_eq!(h.oversubscription_log2, Some(2));
    }

    #[test]
    fn producer_fast_small_burst_lower_oversubscription() {
        let h = WorkloadShape::ProducerFast { burst: 8 }.hints();
        assert_eq!(h.oversubscription_log2, Some(1));
    }

    #[test]
    fn work_steal_uses_auto_gating() {
        let h = WorkloadShape::WorkSteal { n_consumers: 4, batch_size: 16 }.hints();
        assert_eq!(h.k_gating, KGating::Auto);
        assert!(!h.use_burst);
    }

    #[test]
    fn cooperative_enables_mailbox_at_threshold() {
        let small = WorkloadShape::Cooperative { n_cores: 4 }.hints();
        assert!(!small.use_mailbox_routing);
        let large = WorkloadShape::Cooperative { n_cores: 16 }.hints();
        assert!(large.use_mailbox_routing);
        assert!(large.use_burst);
    }

    #[test]
    fn variant_race_uses_per_slot_no_burst() {
        let h = WorkloadShape::VariantRace { n_variants: 3 }.hints();
        assert_eq!(h.k_gating, KGating::PerSlot);
        assert!(!h.use_burst);
    }
}
