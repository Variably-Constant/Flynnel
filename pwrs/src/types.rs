//! The enums more than one family names.
//!
//! Each is the crate's own enum under the same variant names, so a
//! reader of Flynnel's documentation finds what they expect and the
//! binder completes the members by name. A family-specific enum stays
//! with its family; these are the ones a plan, a kernel, a race and the
//! observation surface all have to speak.

use pwrs::prelude::*;

/// How the scheduler classifies a dispatch, which picks SMT
/// activation and the per-element cost estimate together.
#[psenum(name = "Flynnel.DispatchProfile")]
#[derive(Clone, Copy, Default)]
pub enum DispatchProfile {
    /// Long dependency chains, where an SMT sibling fills the bubbles
    /// between stalls. Siblings active, four times oversubscribed.
    LatencyBound,
    /// Issue-port saturated, where a sibling only contests the same
    /// execution unit. Siblings parked, twice oversubscribed.
    PortBound,
    /// Irregular memory work, where a sibling overlaps its own misses
    /// with the first one's stalls. Siblings active.
    MemoryBound,
    /// Sequential streaming, where two threads on one core halve each
    /// other's bandwidth. Siblings parked, not oversubscribed.
    Streaming,
    /// The caller said nothing. Conservative defaults and no cost
    /// estimate, which also turns off the cost-derived tuning.
    #[default]
    Unspecified,
}

impl From<flynnel::DispatchProfile> for DispatchProfile {
    fn from(p: flynnel::DispatchProfile) -> Self {
        use flynnel::DispatchProfile as P;
        match p {
            P::LatencyBound => Self::LatencyBound,
            P::PortBound => Self::PortBound,
            P::MemoryBound => Self::MemoryBound,
            P::Streaming => Self::Streaming,
            P::Unspecified => Self::Unspecified,
        }
    }
}

impl From<DispatchProfile> for flynnel::DispatchProfile {
    fn from(p: DispatchProfile) -> Self {
        use flynnel::DispatchProfile as P;
        match p {
            DispatchProfile::LatencyBound => P::LatencyBound,
            DispatchProfile::PortBound => P::PortBound,
            DispatchProfile::MemoryBound => P::MemoryBound,
            DispatchProfile::Streaming => P::Streaming,
            DispatchProfile::Unspecified => P::Unspecified,
        }
    }
}

/// Which scheduler tier runs a job.
#[psenum(name = "Flynnel.SchedTier")]
#[derive(Clone, Copy, Default)]
pub enum SchedTier {
    /// No scheduler at all: serial in the caller, which is right when
    /// any dispatch would cost more than the work.
    #[default]
    Inline,
    /// One work-stealing arena inside a single NUMA node.
    Local,
    /// One arena per node, with leader-driven steals between them.
    Hierarchical,
    /// A federation of pools with per-node replication.
    Federated,
}

impl From<flynnel::SchedTier> for SchedTier {
    fn from(t: flynnel::SchedTier) -> Self {
        use flynnel::SchedTier as T;
        match t {
            T::Inline => Self::Inline,
            T::Local => Self::Local,
            T::Hierarchical => Self::Hierarchical,
            T::Federated => Self::Federated,
        }
    }
}

/// The accuracy a primitive is asked for.
#[psenum(name = "Flynnel.Variant")]
#[derive(Clone, Copy, Default)]
pub enum Variant {
    /// Bit-exact and correctly rounded, which is what the
    /// verification chain compares.
    #[default]
    Correct,
    /// Within one unit in the last place, not necessarily correctly
    /// rounded.
    Faithful,
    /// Best effort, with a bounded but unstated error.
    Fast,
}

impl From<flynnel::Variant> for Variant {
    fn from(v: flynnel::Variant) -> Self {
        use flynnel::Variant as V;
        match v {
            V::Correct => Self::Correct,
            V::Faithful => Self::Faithful,
            V::Fast => Self::Fast,
        }
    }
}

impl From<Variant> for flynnel::Variant {
    fn from(v: Variant) -> Self {
        use flynnel::Variant as V;
        match v {
            Variant::Correct => V::Correct,
            Variant::Faithful => V::Faithful,
            Variant::Fast => V::Fast,
        }
    }
}

/// The kernel target a plan selects. A class is a code path, not a
/// device: what this host can reach is what Get-FlynnelBackend reports.
#[psenum(name = "Flynnel.HwClass")]
#[derive(Clone, Copy, Default)]
pub enum HwClass {
    /// Plain scalar, available everywhere.
    #[default]
    Scalar,
    /// SSE2, which every x86_64 has.
    Sse2,
    /// AVX2 with FMA.
    Avx2,
    /// AVX-512 Foundation.
    Avx512f,
    /// AVX-512 BF16.
    Avx512Bf16,
    /// AVX-512 VNNI.
    Avx512Vnni,
    /// AVX-512 VBMI2, the byte and word compress-expand rung.
    Avx512Vbmi2,
    /// VEX-encoded AVX-VNNI, for hybrid clients whose efficiency
    /// cores have no AVX-512.
    AvxVnniVex,
    /// GFNI at 128 or 256 bits without AVX-512.
    Gfni,
    /// AVX-512 FP16, full half-precision arithmetic.
    Avx512Fp16,
    /// AVX10.2, the converged vector ISA.
    Avx10_2,
    /// ARMv8 NEON.
    Neon,
    /// ARMv9 SVE2.
    Sve2,
    /// ARMv9-A Scalable Matrix Extension.
    Sme,
    /// Intel AMX BF16 tiles.
    AmxBf16,
    /// Intel AMX INT8 tiles.
    AmxInt8,
    /// Intel AMX FP16 tiles.
    AmxFp16,
    /// NVIDIA Hopper tensor cores.
    TensorCoreHopper,
    /// NVIDIA Blackwell tensor cores.
    TensorCoreBlackwell,
}

impl HwClass {
    /// Every class, in the order the crate declares them.
    pub(crate) const ALL: [HwClass; 19] = [
        HwClass::Scalar,
        HwClass::Sse2,
        HwClass::Avx2,
        HwClass::Avx512f,
        HwClass::Avx512Bf16,
        HwClass::Avx512Vnni,
        HwClass::Avx512Vbmi2,
        HwClass::AvxVnniVex,
        HwClass::Gfni,
        HwClass::Avx512Fp16,
        HwClass::Avx10_2,
        HwClass::Neon,
        HwClass::Sve2,
        HwClass::Sme,
        HwClass::AmxBf16,
        HwClass::AmxInt8,
        HwClass::AmxFp16,
        HwClass::TensorCoreHopper,
        HwClass::TensorCoreBlackwell,
    ];
}

impl From<flynnel::HwClass> for HwClass {
    fn from(c: flynnel::HwClass) -> Self {
        use flynnel::HwClass as H;
        match c {
            H::Scalar => Self::Scalar,
            H::Sse2 => Self::Sse2,
            H::Avx2 => Self::Avx2,
            H::Avx512f => Self::Avx512f,
            H::Avx512Bf16 => Self::Avx512Bf16,
            H::Avx512Vnni => Self::Avx512Vnni,
            H::Avx512Vbmi2 => Self::Avx512Vbmi2,
            H::AvxVnniVex => Self::AvxVnniVex,
            H::Gfni => Self::Gfni,
            H::Avx512Fp16 => Self::Avx512Fp16,
            H::Avx10_2 => Self::Avx10_2,
            H::Neon => Self::Neon,
            H::Sve2 => Self::Sve2,
            H::Sme => Self::Sme,
            H::AmxBf16 => Self::AmxBf16,
            H::AmxInt8 => Self::AmxInt8,
            H::AmxFp16 => Self::AmxFp16,
            H::TensorCoreHopper => Self::TensorCoreHopper,
            H::TensorCoreBlackwell => Self::TensorCoreBlackwell,
        }
    }
}

impl From<HwClass> for flynnel::HwClass {
    fn from(c: HwClass) -> Self {
        use flynnel::HwClass as H;
        match c {
            HwClass::Scalar => H::Scalar,
            HwClass::Sse2 => H::Sse2,
            HwClass::Avx2 => H::Avx2,
            HwClass::Avx512f => H::Avx512f,
            HwClass::Avx512Bf16 => H::Avx512Bf16,
            HwClass::Avx512Vnni => H::Avx512Vnni,
            HwClass::Avx512Vbmi2 => H::Avx512Vbmi2,
            HwClass::AvxVnniVex => H::AvxVnniVex,
            HwClass::Gfni => H::Gfni,
            HwClass::Avx512Fp16 => H::Avx512Fp16,
            HwClass::Avx10_2 => H::Avx10_2,
            HwClass::Neon => H::Neon,
            HwClass::Sve2 => H::Sve2,
            HwClass::Sme => H::Sme,
            HwClass::AmxBf16 => H::AmxBf16,
            HwClass::AmxInt8 => H::AmxInt8,
            HwClass::AmxFp16 => H::AmxFp16,
            HwClass::TensorCoreHopper => H::TensorCoreHopper,
            HwClass::TensorCoreBlackwell => H::TensorCoreBlackwell,
        }
    }
}

/// Which bisect shape a dispatch uses when the default lazy-steal one
/// is overridden.
#[psenum(name = "Flynnel.BisectVariant")]
#[derive(Clone, Copy, Default)]
pub enum BisectVariant {
    /// Clamp the upfront leaf count to one per worker, so the tree is
    /// shallower than the shipped baseline.
    #[default]
    ProducerMaxLenWorkers,
    /// Start at one leaf per worker and replenish on each observed
    /// steal, the rayon formula.
    RayonStyleReplenish,
}

impl From<flynnel::BisectVariant> for BisectVariant {
    fn from(v: flynnel::BisectVariant) -> Self {
        use flynnel::BisectVariant as B;
        match v {
            B::ProducerMaxLenWorkers => Self::ProducerMaxLenWorkers,
            B::RayonStyleReplenish => Self::RayonStyleReplenish,
        }
    }
}

impl From<BisectVariant> for flynnel::BisectVariant {
    fn from(v: BisectVariant) -> Self {
        use flynnel::BisectVariant as B;
        match v {
            BisectVariant::ProducerMaxLenWorkers => B::ProducerMaxLenWorkers,
            BisectVariant::RayonStyleReplenish => B::RayonStyleReplenish,
        }
    }
}

/// How far a pushed right-half may be stolen from, in cache-coherence
/// distance.
#[psenum(name = "Flynnel.DequeTier")]
#[derive(Clone, Copy, Default)]
pub enum DequeTier {
    /// The same physical core, where SMT siblings share L1d. Only the
    /// sibling may steal.
    #[default]
    SmtLocal,
    /// The same cluster, sharing L2 or L3. About thirty nanoseconds to
    /// bounce a line.
    IntraCcx,
    /// The same socket across clusters. Fifty to a hundred.
    CrossCcx,
    /// Anywhere, including across nodes. About two hundred.
    Public,
}

impl From<flynnel::sched::deque_tier::DequeTier> for DequeTier {
    fn from(t: flynnel::sched::deque_tier::DequeTier) -> Self {
        use flynnel::sched::deque_tier::DequeTier as D;
        match t {
            D::SmtLocal => Self::SmtLocal,
            D::IntraCcx => Self::IntraCcx,
            D::CrossCcx => Self::CrossCcx,
            D::Public => Self::Public,
        }
    }
}

impl From<DequeTier> for flynnel::sched::deque_tier::DequeTier {
    fn from(t: DequeTier) -> Self {
        use flynnel::sched::deque_tier::DequeTier as D;
        match t {
            DequeTier::SmtLocal => D::SmtLocal,
            DequeTier::IntraCcx => D::IntraCcx,
            DequeTier::CrossCcx => D::CrossCcx,
            DequeTier::Public => D::Public,
        }
    }
}

/// What a caller knows about the shape of one leaf's work, which lets
/// the classifier route correctly on the first call instead of after
/// the observer has refined it.
#[psenum(name = "Flynnel.LeafShape")]
#[derive(Clone, Copy, Default)]
pub enum LeafShape {
    /// Compute bound on the issue port: integer multiply, FMA,
    /// compare. Siblings would contest the same port.
    PortCompute,
    /// Compute bound on a long floating-point dependency chain.
    /// Siblings fill the bubbles.
    LatencyCompute,
    /// Sequential streaming, bound by per-core bandwidth.
    Streaming,
    /// Irregular access: gather, scatter, pointer chase.
    Gather,
    /// Mixed or unknown, which falls back to the heuristics.
    #[default]
    Unknown,
}

impl From<flynnel::LeafShape> for LeafShape {
    fn from(s: flynnel::LeafShape) -> Self {
        use flynnel::LeafShape as L;
        match s {
            L::PortCompute => Self::PortCompute,
            L::LatencyCompute => Self::LatencyCompute,
            L::Streaming => Self::Streaming,
            L::Gather => Self::Gather,
            L::Unknown => Self::Unknown,
        }
    }
}

impl From<LeafShape> for flynnel::LeafShape {
    fn from(s: LeafShape) -> Self {
        use flynnel::LeafShape as L;
        match s {
            LeafShape::PortCompute => L::PortCompute,
            LeafShape::LatencyCompute => L::LatencyCompute,
            LeafShape::Streaming => L::Streaming,
            LeafShape::Gather => L::Gather,
            LeafShape::Unknown => L::Unknown,
        }
    }
}

/// The class a site's observer learns from what it measured, or that a
/// caller states outright.
#[psenum(name = "Flynnel.WorkloadClass")]
#[derive(Clone, Copy, Default)]
pub enum WorkloadClass {
    /// Tiny per-item cost, under about fifty nanoseconds, where the
    /// dispatch decision dominates the work.
    #[default]
    FineGrain,
    /// Fifty to five hundred nanoseconds an item, port saturated.
    PortBound,
    /// Over five hundred, on long floating-point chains.
    LatencyBound,
    /// Irregular memory.
    MemoryBound,
    /// Sequential streaming.
    Streaming,
}

impl From<flynnel::WorkloadClass> for WorkloadClass {
    fn from(c: flynnel::WorkloadClass) -> Self {
        use flynnel::WorkloadClass as W;
        match c {
            W::FineGrain => Self::FineGrain,
            W::PortBound => Self::PortBound,
            W::LatencyBound => Self::LatencyBound,
            W::MemoryBound => Self::MemoryBound,
            W::Streaming => Self::Streaming,
        }
    }
}

impl From<WorkloadClass> for flynnel::WorkloadClass {
    fn from(c: WorkloadClass) -> Self {
        use flynnel::WorkloadClass as W;
        match c {
            WorkloadClass::FineGrain => W::FineGrain,
            WorkloadClass::PortBound => W::PortBound,
            WorkloadClass::LatencyBound => W::LatencyBound,
            WorkloadClass::MemoryBound => W::MemoryBound,
            WorkloadClass::Streaming => W::Streaming,
        }
    }
}

/// Which shape an N-way cooperative join takes.
#[psenum(name = "Flynnel.CooperativeRouting")]
#[derive(Clone, Copy, Default)]
pub enum CooperativeRouting {
    /// Defer: a plan defers to the process tag, and the tag defers to
    /// the population heuristic.
    #[default]
    Auto,
    /// The tree bisect, whose amortized setup wins for short
    /// closures.
    ForceTree,
    /// Mailbox distribution, one closure per peer, for an N that
    /// matches the pool.
    ForceMailbox,
    /// Deque fan-out, for closures whose costs differ enough that
    /// mailbox concentration would pin a slow one.
    ForceDeque,
}

impl From<flynnel::sched::adaptive_cooperative::CooperativeRouting> for CooperativeRouting {
    fn from(r: flynnel::sched::adaptive_cooperative::CooperativeRouting) -> Self {
        use flynnel::sched::adaptive_cooperative::CooperativeRouting as C;
        match r {
            C::Auto => Self::Auto,
            C::ForceTree => Self::ForceTree,
            C::ForceMailbox => Self::ForceMailbox,
            C::ForceDeque => Self::ForceDeque,
        }
    }
}

impl From<CooperativeRouting> for flynnel::sched::adaptive_cooperative::CooperativeRouting {
    fn from(r: CooperativeRouting) -> Self {
        use flynnel::sched::adaptive_cooperative::CooperativeRouting as C;
        match r {
            CooperativeRouting::Auto => C::Auto,
            CooperativeRouting::ForceTree => C::ForceTree,
            CooperativeRouting::ForceMailbox => C::ForceMailbox,
            CooperativeRouting::ForceDeque => C::ForceDeque,
        }
    }
}

/// Which bisect-variant routing table the process consults.
#[psenum(name = "Flynnel.VariantRouting")]
#[derive(Clone, Copy, Default)]
pub enum VariantRouting {
    /// Take the CPUID-resolved default for this vendor.
    #[default]
    Auto,
    /// No variant routing: every dispatch uses the default lazy-steal
    /// bisect.
    Default,
    /// Batch-size-adaptive selection for port-bound work, which is
    /// the CPUID default on AMD.
    ComputeBatchAdaptive,
}

impl From<flynnel::sched::adaptive_variant_routing::VariantRouting> for VariantRouting {
    fn from(r: flynnel::sched::adaptive_variant_routing::VariantRouting) -> Self {
        use flynnel::sched::adaptive_variant_routing::VariantRouting as V;
        match r {
            V::Auto => Self::Auto,
            V::Default => Self::Default,
            V::ComputeBatchAdaptive => Self::ComputeBatchAdaptive,
        }
    }
}

impl From<VariantRouting> for flynnel::sched::adaptive_variant_routing::VariantRouting {
    fn from(r: VariantRouting) -> Self {
        use flynnel::sched::adaptive_variant_routing::VariantRouting as V;
        match r {
            VariantRouting::Auto => V::Auto,
            VariantRouting::Default => V::Default,
            VariantRouting::ComputeBatchAdaptive => V::ComputeBatchAdaptive,
        }
    }
}
