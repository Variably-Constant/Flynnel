//! # Flynnel
//!
//! A K-aware, NUMA-aware work-stealing scheduler with extended-Flynn-
//! taxonomy dispatch. Pun-named for Michael J. Flynn (1934-), whose
//! 1966 taxonomy of computer architectures (SISD / SIMD / MISD /
//! MIMD) underlies the per-call execution-class plan this crate
//! exposes.
//!
//! ## Extended Flynn taxonomy
//!
//! | Acronym | Expansion                            | Flynnel axis                          |
//! |---------|--------------------------------------|---------------------------------------|
//! | SISD    | Single Instruction, Single Data      | inline execution                      |
//! | SIMD    | Single Instruction, Multiple Data    | vector lanes (in-kernel)              |
//! | MISD    | Multiple Instruction, Single Data    | `race_variants` (first tolerable wins)|
//! | MIMD    | Multiple Instruction, Multiple Data  | `join`; `explore_select` (all finish, best-by-comparator) |
//! | SIMC    | Single Instruction, Multiple Cores   | `cooperative_join_n` (cross-core SIMD)|
//! | MIMC    | Multiple Instruction, Multiple Cores | heterogeneous roles within MIMC       |
//! | SIMT    | Single Instruction, Multiple Threads | `DispatchBackend::dispatch_kernel` on a registered accelerator; auto-routed via `dispatch_accel` |
//! | MIMT    | Multiple Instruction, Multiple Threads | `join_hybrid` / `hybrid_pipeline` (CPU + accelerator) |
//!
//! ## Quick start
//!
//! ```no_run
//! use flynnel::{JobPlan, DispatchProfile, join};
//!
//! let plan = JobPlan::set_profile(8, 1024, DispatchProfile::PortBound);
//! let (a, b) = join(
//!     &plan,
//!     || (0..1024).sum::<u32>(),
//!     || (1024..2048).sum::<u32>(),
//! );
//! assert_eq!(a + b, (0..2048).sum::<u32>());
//! ```
//!
//! The workhorse primitives ([`join`], [`for_each_chunk`],
//! [`for_each_indexed`], [`for_each_chunk_ref`], [`cooperative_join_n`],
//! [`join_hybrid`], [`hybrid_pipeline`], [`race_variants`],
//! [`CancelToken`], [`k_join`], [`k_join_with_plan`], [`JobPlan`])
//! are re-exported at the crate root following the rayon / tokio
//! convention: short top-level path for the things you call
//! constantly. Specialized variants stay namespaced under [`sched`]
//! where they live.
//!
//! References: rayon-core 1.13 (JobRef vtable, CoreLatch, JEC
//! sleep), ARCAS arXiv:2503.11460 (chiplet-aware scheduling; see
//! [`crate::numa_topology::NumaTopology::cluster_size_log2`]),
//! Libfork arXiv:2402.18480 (continuation stealing), Olivier-Prins
//! ROSS '11 (leader-driven cross-NUMA stealing), arXiv:2401.04494
//! (last-successful-victim-first steal probing), arXiv:2009.00202
//! (prefetching stolen job state into L2 on steal).

#![deny(missing_docs)]
// Module-level docs in this crate intentionally reference private
// implementation details (algorithm constants like MIN_LEAF_ITEMS,
// HEARTBEAT_CYCLES; impl-detail state like the SET / UNSET latch
// constants) when explaining the WHY of a design. Allow rustdoc to
// link to them without warning.
#![allow(rustdoc::private_intra_doc_links)]

pub mod backend;
pub mod cpu_info;
pub mod dispatch_profile;
pub mod flat;
pub mod foundation;
#[cfg(feature = "gpu-peer")]
pub mod gpu_peer;
pub mod numa_topology;
pub mod op_class;
pub mod sched;

pub use backend::{
    Backend, BackendCapabilities, BackendError, BackendRef, DispatchBackend, KernelArg,
    KernelHandle, backend_by_id, backends, cpu_backend, register_backend,
};
pub use backend::accel_op::{
    AccelOpId, AccelReport, accel_op_name, accel_target, bind_accel_kernel,
    bind_accel_kernel_handle, dispatch_accel, register_accel_op,
};
pub use dispatch_profile::DispatchProfile;
pub use foundation::{HwClass, SchedTier, Variant};
pub use numa_topology::{NumaSource, NumaTopology, numa_topology};
pub use op_class::OpClass;

// Workhorse scheduling primitives, re-exported at the crate root so
// the common idiom is `flynnel::join` / `flynnel::for_each_chunk`
// rather than the deeper `flynnel::sched::arena::join` /
// `flynnel::sched::par_iter::for_each_chunk` paths. The deeper paths
// remain valid as aliases for callers that prefer the namespacing.
pub use sched::adaptive_profile::{LeafShape, WorkloadClass};
pub use sched::{BisectVariant, JobPlan};
pub use sched::arena::{join, join_context, join_default};
pub use sched::cooperative::cooperative_join_n;
pub use sched::call_site::{CallSiteState, Placement, SiteRef, caller_site, site_for_location};
pub use sched::cat::{CatCapability, CatError, L3Reservation};
pub use sched::{
    reset_spin_stats, set_spin_adaptive, set_spin_window, spin_window, total_idle_yields,
};
pub use sched::hybrid::{
    SplitReport, hybrid_auto, hybrid_auto_split, hybrid_auto_split_ranges, hybrid_pipeline,
    join_hybrid,
};
pub use sched::k_join::{k_join, k_join_with_plan};
pub use sched::par_iter::{for_each_chunk, for_each_chunk_ref, for_each_indexed};
pub use sched::race::{
    Agreement, Anytime, CancelToken, Settled, StatOpts, StatOutcome, explore_select, race_agree,
    race_any, race_deadline, race_quorum, race_refute, race_statistical, race_tournament,
    race_variants,
};
#[cfg(feature = "shared-memory-worker-reference")]
pub use sched::marshal::{Marshal, register_marshal_handler};
