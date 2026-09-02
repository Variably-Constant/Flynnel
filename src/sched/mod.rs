//! Flynnel K-aware NUMA-aware unified scheduler.
//!
//! A layered dispatch system that maps a job onto the right
//! parallelism envelope based on its `K_outer` (data size), batch
//! size, NUMA topology, and target hardware class. The design borrows
//! the best ideas from rayon (two-word JobRef vtable, 4-state
//! CoreLatch with `Latch::set(*const Self)` self-invalidation
//! discipline, JEC-protected two-phase sleep, Chase-Lev deque via
//! a Chase-Lev work-stealing deque) and reshapes them around
//! Flynnel's K-hierarchy.
//!
//! References: rayon-core 1.13 (Job vtable, CoreLatch, JEC sleep),
//! ARCAS arXiv:2503.11460 (chiplet-aware scheduling; see
//! [`crate::numa_topology::NumaTopology::cluster_size_log2`]),
//! Libfork arXiv:2402.18480 (continuation stealing), Olivier-Prins
//! ROSS '11 (leader-driven cross-NUMA stealing), BLIS
//! Multithreading.md (parallelism at cache-hierarchy boundaries).
//!
//! Layers, top down: [`plan`] (JobPlan + [`pick_tier`]) -> [`arena`]
//! / [`arena_local`] / [`arena_numa`] (work-stealing pools) ->
//! deque primitives ([`chase_lev_local`], [`khl_local`],
//! [`fcl_local`], [`adaptive_worker`], [`injector`],
//! [`flynnel_ring`]). Data-parallel surfaces sit beside them:
//! [`par_iter`], [`cooperative`], [`race`], [`pipeline`],
//! [`k_join`], [`hybrid`]. Adaptive routing and calibration:
//! [`adaptive_profile`], [`adaptive_backend`], [`call_site`],
//! [`split_observer`], [`bg_calibration`]. Each module's own doc
//! carries its contract.

pub mod plan;
pub mod call_site;
pub mod cat;
pub mod latch;
pub mod job;
pub(crate) mod deque;
pub mod chase_lev_local;
pub mod fcl_local;
pub mod fcl_worker;
pub mod khl_local;
pub mod khl_worker;
pub mod adaptive_worker;
pub mod adaptive_profile;
pub mod adaptive_backend;
pub mod adaptive_cooperative;
pub mod adaptive_variant_routing;
pub mod flynnel_ring;
pub mod flynnel_ring_spsc;
pub mod flynnel_ring_mpsc;
pub mod flynnel_ring_composed;
pub mod injector;
pub mod notify_ring;
pub mod k_gating;
pub mod workload_shape;
pub mod dispatch;
pub mod deque_tier;
pub mod arena;
pub mod k_join;
pub mod numa_alloc;
pub mod sleep;
pub mod arena_local;
pub mod arena_numa;
pub mod par_iter;
pub mod race;
pub mod mode_region;
pub mod io_pool;
pub mod bg_calibration;
#[cfg(feature = "verify-chain")]
pub mod verify_chain;
pub mod split_observer;
pub mod prefetch;
pub mod private_deque;
pub mod idempotent;
pub mod cooperative;
pub mod numa_latency;
pub mod pipeline;
pub mod bg_zero;
pub mod hybrid;
pub mod trace;
pub(crate) mod jec_sleep;
#[cfg(feature = "shared-memory-worker-reference")]
pub mod marshal;
#[cfg(feature = "shared-memory-worker-reference")]
pub mod dual_deque;

pub use plan::{BisectVariant, HwClass, JobPlan, SchedTier, kband_for, pick_tier};
pub use call_site::{CallSiteState, PolicyArm, Placement, SiteRef, caller_site, site_for_location};
pub use adaptive_profile::{
    ClassThresholds, ThresholdCalibration, calibrate_class_thresholds,
    calibrate_class_thresholds_into, class_thresholds,
    spawn_class_threshold_calibration,
};
pub use latch::{CoreLatch, Latch};
pub use jec_sleep::{
    reset_spin_stats, set_spin_adaptive, set_spin_window, spin_window, total_idle_yields,
};
pub use job::NUMA_HINT_ANY;
pub use arena::{dispatch_trace_snapshot, dispatch_trace_wait_snapshot, join, join_context, join_default};
pub use io_pool::{IoPool, IoTask, global_io_pool, submit_io_or_inline};
pub use bg_calibration::spawn_calibration;
#[cfg(feature = "verify-chain")]
pub use verify_chain::{VerifyChain, VerifyHasher, FxFallbackHasher, default_hasher};
pub use split_observer::{set_split_multiplier, spawn_observer, split_multiplier};
pub use pipeline::par_map_serial_reduce;
pub use par_iter::{par_map_in_place, par_zip_apply};
pub use prefetch::{
    prefetch_into_l2, prefetch_into_l2_inline, prefetch_into_l3, prefetch_into_l3_inline,
};
pub use k_join::k_join_with_plan;
pub use cooperative::{cooperative_join_n, cooperative_join_n_flat, cooperative_join_n_tree};
pub use hybrid::{SplitReport, hybrid_auto, hybrid_auto_split, hybrid_pipeline, join_hybrid};
pub use numa_latency::{TopologyLatencyTable, topology_latency_table};
#[cfg(feature = "shared-memory-worker-reference")]
pub use marshal::{Marshal, register_marshal_handler};
