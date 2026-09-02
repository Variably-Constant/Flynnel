//! Cross-process dispatch backend. Compiled only when the
//! `shared-memory-worker-reference` Cargo feature is enabled.
//!
//! Shared substrate: [`latch_mmf::MmfLatchArena`] (64 B latch
//! cells; the peer publishes result bytes inline and Release-stores
//! `SET`, no response ring), [`pass_registry`] (process-local
//! `closure_id -> handler` table; the wire carries `(closure_id,
//! args)`, never closure code), [`wire`] (`KernelArg`
//! encode / decode).
//!
//! Deque variants, selected per call by
//! [`variant_dispatch::CrossProcessDispatcher`]:
//! [`chase_lev_mmf::MmfChaseLevDeque`] (owner Release-store, thief
//! CAS), [`khl_mmf::MmfKhlDeque`] (cache-line publish via
//! `movdir64b` where available), [`khpd::KhpdDeque`] (batched
//! publication lines), [`urd::UrdDeque`] (per-thief mailbox),
//! [`lcrq_lifo::LohDeque`] (batched producer, LCRQ-FIFO steal).
//! One backend type wraps each ([`SharedMemoryChaseLevBackend`],
//! [`SharedMemoryKhpdBackend`], [`SharedMemoryUrdBackend`],
//! [`SharedMemoryLohBackend`]); [`dispatch_calibration`] populates
//! the routing table from per-host measurements.
//!
//! Trait contract: `register_kernel` wraps
//! [`pass_registry::hash_name`] of the kernel name; every
//! participating process must register the same id before dispatch.
//! `dispatch_one(Box<dyn FnOnce>)` panics and
//! `dispatch_parallel_for` no-ops: closures cannot ship across
//! processes; fan-out comes from attaching more peers to the deque.
//!
//! Versus the CPU backend this buys sandboxing (peer crash cannot
//! corrupt the originator), language interop (any mmap-capable
//! peer), and process-isolated runtimes. Measured per-call latency
//! by coherence tier lives on
//! [`crate::backend::shared_mem::chase_lev_backend`];
//! `examples/chase_lev_mmf_steal.rs` is the cross-process steal
//! end-to-end.

#![allow(clippy::missing_errors_doc)]

pub mod chase_lev_backend;
pub mod chase_lev_mmf;
pub mod dispatch_calibration;
pub mod khl_mmf;
pub mod khpd;
pub mod khpd_backend;
pub mod latch_mmf;
pub mod urd;
pub mod urd_backend;
pub mod lcrq_lifo;
pub mod loh_backend;
pub mod pass_registry;
pub mod variant_dispatch;
pub mod wire;

pub use chase_lev_backend::{DispatchHandle, SharedMemoryChaseLevBackend};
pub use chase_lev_mmf::{
    ARGS_INLINE_BYTES, MmfChaseLevDeque, PushError as ChaseLevPushError, RemoteJobSlot,
    Steal as ChaseLevSteal,
};
pub use latch_mmf::{ERR as LATCH_ERR, MmfLatchArena, RESULT_BYTES, SET as LATCH_SET, UNSET as LATCH_UNSET, LatchError};
pub use khpd::{
    KHPD_ARGS_INLINE_BYTES, KhpdDeque, LINE_ITEMS, LineItem, PushError as KhpdPushError,
    Steal as KhpdSteal, StealResult as KhpdStealResult,
};
pub use khl_mmf::{
    KHL_MMF_LINE_ITEMS, MmfKhlDeque, PublishError as KhlMmfPublishError,
    Steal as KhlMmfSteal, StealResult as KhlMmfStealResult,
    movdir64b, movdir64b_available,
};
pub use khpd_backend::SharedMemoryKhpdBackend;
pub use urd::{
    Drain as UrdDrain, DrainResult as UrdDrainResult, MAILBOX_ITEMS, PublishError as UrdPublishError,
    UrdDeque, WaitStrategy as UrdWaitStrategy,
};
pub use urd_backend::SharedMemoryUrdBackend;
pub use lcrq_lifo::{
    LOH_ARGS_INLINE_BYTES, LohDeque, LohLifoEntry, PushError as LohPushError,
    Steal as LohSteal, StealResult as LohStealResult,
};
pub use loh_backend::SharedMemoryLohBackend;
pub use pass_registry::{Pass, PassError, PassHandler, PassResult, hash_name, register, unregister};
pub use variant_dispatch::{
    CrossProcessDispatcher, CrossProcessDispatcherBuilder, DequeVariant, DispatchedHandle,
    DispatcherRoutingTable, WorkloadShape,
};
pub use dispatch_calibration::{
    CellCalibration, VariantMeasurement, calibrate_cell, calibrate_routing_table,
    estimate_calibration_budget, measure_variant_ns_per_call,
};
