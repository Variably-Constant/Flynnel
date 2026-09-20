//! PowerShell bindings for Flynnel, bound to the Rust directly.
//!
//! The module has two layers. Cmdlets obtain or read something: the
//! host's topology, a job plan, the worker pool, a calibration, a
//! trace. Objects carry the operations their Rust type offers as
//! methods, one method per operation, so a plan is built by chaining
//! the same builders the crate has and a ring is filled by calling the
//! ring.
//!
//! An object owns its Rust value and releases it when disposed or when
//! the garbage collector finalizes it. A failure from a method is an
//! exception; from a cmdlet it is an error record, non-terminating
//! unless the cmdlet cannot go on.
//!
//! # Anything over many items crosses in one call
//!
//! Measured at this boundary in the sibling SubEtha module's
//! `bench/CallShapes.ps1`: a method call costs 1907 ns in PowerShell
//! 7.6 and 651 ns in Windows PowerShell, the same call batched a
//! thousand at a time costs 4.5 and 3.0 ns, and one pipeline record
//! costs 1712 and 7955 ns. The boundary is a hundred times cheaper
//! amortized than per item, and on Windows PowerShell the pipeline
//! alone caps a per-record cmdlet near 125 thousand items a second
//! whatever runs behind it.
//!
//! So every cmdlet and method here that can handle many items takes
//! them all in one call. A per-record form exists only where a script
//! genuinely wants to interleave, and its help states what a record
//! costs.
//!
//! # What this module does not do
//!
//! It never runs a PowerShell script block on a Flynnel worker. A
//! script block runs only on the thread that owns the pipeline: the
//! binding framework's pipeline token is `!Send` and the managed side
//! refuses a stream call reached from another thread. Work that runs on
//! the pool is therefore work this module declares - the kernels over
//! arrays, files and text, the accelerator ops, and the bodies the
//! racing and hybrid cmdlets take by name.

pwrs::export_module! {
    name: "Flynnel",
    cmdlets: [
        host::GetFlynnelCpuInfo,
        host::GetFlynnelTopology,
        host::GetFlynnelNumaDistance,
        host::GetFlynnelNodeCpu,
        host::GetFlynnelLatencyTable,
        host::GetFlynnelHwClass,
        host::GetFlynnelCacheAllocation,
        host::NewFlynnelCacheReservation,
        plan::NewFlynnelPlan,
        plan::ResolveFlynnelPlan,
        plan::GetFlynnelKBand,
        plan::GetFlynnelDispatchProfile,
        pool::StartFlynnelPool,
        pool::GetFlynnelPool,
        pool::GetFlynnelWorker,
        pool::GetFlynnelSpinWindow,
        pool::SetFlynnelSpinWindow,
        pool::SetFlynnelSpinAdaptive,
        pool::ResetFlynnelSpinStats,
        pool::GetFlynnelSplitMultiplier,
        pool::SetFlynnelSplitMultiplier,
        pool::ResetFlynnelSplitStats,
        pool::StartFlynnelSplitObserver,
        pool::NewFlynnelIoPool,
        pool::GetFlynnelIoPool,
        kernels::InvokeFlynnelMap,
        kernels::InvokeFlynnelZip,
        kernels::MeasureFlynnelReduce,
        kernels::GetFlynnelPrefixSum,
        kernels::GetFlynnelHistogram,
        kernels::GetFlynnelDotProduct,
        kernels::SortFlynnelArray,
        kernels::MeasureFlynnelFileHash,
        kernels::TestFlynnelFileHash,
        kernels::SearchFlynnelFile,
        kernels::MeasureFlynnelFileLine,
        kernels::MeasureFlynnelFileByte,
        kernels::SearchFlynnelText,
        kernels::MeasureFlynnelTextCount,
        kernels::SplitFlynnelText,
        kernels::UpdateFlynnelText,
    ],
    classes: [
        host::CpuInfo,
        host::Topology,
        host::NumaDistance,
        host::NodeCpus,
        host::LatencyTable,
        host::HwClassInfo,
        host::CacheAllocation,
        host::CacheReservation,
        plan::Plan,
        plan::ResolvedPlan,
        plan::ProfileRow,
        pool::Pool,
        pool::WorkerStat,
        pool::SpinState,
        pool::SplitState,
        pool::IoPool,
        kernels::Reduction,
        kernels::HistogramBin,
        kernels::FileHash,
        kernels::HashCheck,
        kernels::FileMatch,
        kernels::FileMeasure,
        kernels::TextMatch,
        kernels::TextMeasure,
    ],
    enums: [
        host::Vendor,
        host::NumaSource,
        host::ClusterSource,
        types::DispatchProfile,
        types::SchedTier,
        types::Variant,
        types::HwClass,
        types::BisectVariant,
        types::DequeTier,
        types::LeafShape,
        types::WorkloadClass,
        types::CooperativeRouting,
        types::VariantRouting,
        kernels::MapOp,
        kernels::ZipOp,
        kernels::ReduceOp,
        kernels::TextTransform,
    ],
}

mod host;
mod kernels;
mod plan;
mod pool;
mod types;
