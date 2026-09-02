//! Foundation enums: [`Variant`], [`SchedTier`], [`HwClass`].
//!
//! These three small enums form the vocabulary every Flynnel call site
//! speaks. They are deliberately tiny, dependency-free, and stable, so
//! downstream crates that build domain-specific dispatch tensors on top
//! of Flynnel can re-export them or wrap them without pulling in the
//! whole scheduler.

use core::fmt;

// ---------------------------------------------------------------------------
// Variant: precision contract a primitive offers
// ---------------------------------------------------------------------------

/// Precision contract a primitive offers.
///
/// Three-variant precision discipline: every primitive that can support
/// multiple precision variants ships all three; primitives that
/// physically cannot (e.g. equality tests) ship a single variant and
/// document the omission.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Variant {
    /// Bit-exact correctly-rounded result. The verification chain
    /// (BLAKE3 over per-stripe outputs xor Merkle-root agreement
    /// across replicas) uses this variant.
    #[default]
    Correct,
    /// Faithfully-rounded result: within 1 ulp of the exact answer
    /// but not necessarily correctly rounded. Used when the caller
    /// can tolerate +/-1 ulp in exchange for ~2x throughput.
    Faithful,
    /// Best-effort result with bounded but unspecified error. Used by
    /// inner-loop iterates of refinement schemes that recover
    /// precision externally.
    Fast,
}

impl Variant {
    /// Variants ordered by accuracy, highest first.
    pub const ALL: [Variant; 3] = [Variant::Correct, Variant::Faithful, Variant::Fast];

    /// Whether this variant must produce bit-exact results identical
    /// across CPU, GPU, and every distributed replica.
    pub fn requires_bit_exact(self) -> bool {
        matches!(self, Variant::Correct)
    }
}

impl fmt::Display for Variant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Variant::Correct => f.write_str("correct"),
            Variant::Faithful => f.write_str("faithful"),
            Variant::Fast => f.write_str("fast"),
        }
    }
}

// ---------------------------------------------------------------------------
// SchedTier: which scheduler tier runs a given job
// ---------------------------------------------------------------------------

/// Which scheduler tier runs a given job. Selected per call by the
/// dispatch policy from `(k_outer, batch_size, numa_topology, hw_class)`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum SchedTier {
    /// No scheduler: serial in caller. Used for `K_outer <= 4`
    /// where any scheduling overhead exceeds the actual work.
    #[default]
    Inline,
    /// Single-arena work-stealing inside one NUMA node. Child-
    /// stealing, randomized victim selection, Chase-Lev deque.
    Local,
    /// Multi-arena (one per NUMA node) with leader-driven cross-arena
    /// steals per Olivier-Prins ROSS '11 / ARCAS '25.
    Hierarchical,
    /// Multi-pool federation: per-NUMA arenas + tiered storage +
    /// per-NUMA constant replication. FLINT-style pull-pool.
    Federated,
}

impl SchedTier {
    /// Tiers ordered by ascending parallelism cost (Inline cheapest,
    /// Federated heaviest).
    pub const ALL: [SchedTier; 4] = [
        SchedTier::Inline,
        SchedTier::Local,
        SchedTier::Hierarchical,
        SchedTier::Federated,
    ];

    /// Recommended `thread::yield_now()` spin rounds a worker should
    /// run before parking on its condvar. Picked per tier from
    /// empirical observations on Zen+ and Skylake-X.
    pub fn spin_rounds(self) -> u32 {
        match self {
            SchedTier::Inline => 0,
            SchedTier::Local => 8,
            SchedTier::Hierarchical => 32,
            SchedTier::Federated => 0,
        }
    }
}

impl fmt::Display for SchedTier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SchedTier::Inline => f.write_str("inline"),
            SchedTier::Local => f.write_str("local"),
            SchedTier::Hierarchical => f.write_str("hierarchical"),
            SchedTier::Federated => f.write_str("federated"),
        }
    }
}

// ---------------------------------------------------------------------------
// HwClass: hardware class a primitive may target
// ---------------------------------------------------------------------------

/// Hardware class a primitive may target. Selects the concrete kernel
/// variant the scheduler dispatches to. Orthogonal to [`SchedTier`].
///
/// Maps the K-axis hardware regime: vector SIMD at `K_R = 0..6` and
/// matrix-extension regime at `K_R = 10..16`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum HwClass {
    /// Plain scalar code path (always available, conservative
    /// performance).
    #[default]
    Scalar,
    /// SSE2 (every x86_64 has it).
    Sse2,
    /// AVX2 + FMA (Haswell+, Zen 2+).
    Avx2,
    /// AVX-512 Foundation (Skylake-X+, Zen 4+).
    Avx512f,
    /// AVX-512 BF16 (Sapphire Rapids+, Zen 4 client subset).
    Avx512Bf16,
    /// AVX-512 VNNI (Cascade Lake+, Zen 4 desktop).
    Avx512Vnni,
    /// AVX-512 VBMI2: byte/word compress-expand and funnel shifts
    /// (Ice Lake+, Zen 4+). The string/byte-processing rung -
    /// filtering, tokenized scans, index-search kernels. First
    /// consumer: the Lattice index search's AVX-512 matcher form.
    /// Detection: `avx512vbmi2`.
    Avx512Vbmi2,
    /// AVX-VNNI, VEX-encoded: int8 dot on hybrid Intel clients whose
    /// E-cores have no AVX-512 (Alder/Raptor Lake). Detection:
    /// `avxvnni`.
    AvxVnniVex,
    /// GFNI at 128/256-bit without AVX-512: per-byte bit permutation
    /// on the same hybrid parts. Detection: `gfni`.
    Gfni,
    /// AVX-512 FP16: full f16 arithmetic (Sapphire Rapids+); the
    /// half-precision compute rung Bf16 does not cover. Detection:
    /// `avx512fp16`.
    Avx512Fp16,
    /// AVX10.2, the versioned converged vector ISA: 512-bit
    /// guaranteed on every core including E-cores, bf16 arithmetic,
    /// FP8 converts, VMINMAX. Detect via CPUID leaf 0x24's version
    /// field, NOT the discrete flag zoo the AVX-512 era needed.
    Avx10_2,
    /// ARMv8 NEON.
    Neon,
    /// ARMv9 SVE2: scalable vectors between NEON and SME in the ARM
    /// column. Detection: `sve2`.
    Sve2,
    /// ARMv9-A Scalable Matrix Extension (Apple M4+, future ARM
    /// server). Streaming-SVE mode with ZA matrix register.
    Sme,
    /// Intel AMX BF16 tile path (Sapphire Rapids+).
    AmxBf16,
    /// Intel AMX INT8 tile path (Sapphire Rapids+).
    AmxInt8,
    /// Intel AMX FP16 (Granite Rapids+).
    AmxFp16,
    /// NVIDIA Hopper tensor cores (sm_90 + WGMMA + TMA).
    TensorCoreHopper,
    /// NVIDIA Blackwell tensor cores (sm_100 + tcgen05 + tensor
    /// memory + dual-SM 256x256 MMA).
    TensorCoreBlackwell,
}

impl HwClass {
    /// True if this class is in the matrix-extension regime
    /// (`K_R = 10..16`). Matrix-extension regime requires mode-region
    /// batching (see [`crate::sched::mode_region::run_in_region`])
    /// because kernel-entry costs amortize per region, not per op.
    pub fn is_matrix_extension(self) -> bool {
        matches!(
            self,
            HwClass::Sme
                | HwClass::AmxBf16
                | HwClass::AmxInt8
                | HwClass::AmxFp16
                | HwClass::TensorCoreHopper
                | HwClass::TensorCoreBlackwell
        )
    }
}

impl fmt::Display for HwClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            HwClass::Scalar => "scalar",
            HwClass::Sse2 => "sse2",
            HwClass::Avx2 => "avx2",
            HwClass::Avx512f => "avx512f",
            HwClass::Avx512Bf16 => "avx512bf16",
            HwClass::Avx512Vnni => "avx512vnni",
            HwClass::Avx512Vbmi2 => "avx512vbmi2",
            HwClass::AvxVnniVex => "avx-vnni",
            HwClass::Gfni => "gfni",
            HwClass::Avx512Fp16 => "avx512fp16",
            HwClass::Avx10_2 => "avx10.2",
            HwClass::Neon => "neon",
            HwClass::Sve2 => "sve2",
            HwClass::Sme => "sme",
            HwClass::AmxBf16 => "amx-bf16",
            HwClass::AmxInt8 => "amx-int8",
            HwClass::AmxFp16 => "amx-fp16",
            HwClass::TensorCoreHopper => "tc-hopper",
            HwClass::TensorCoreBlackwell => "tc-blackwell",
        };
        f.write_str(s)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn variant_all_has_three_variants() {
        assert_eq!(Variant::ALL.len(), 3);
        assert_eq!(Variant::ALL[0], Variant::Correct);
    }

    #[test]
    fn variant_correct_requires_bit_exact() {
        assert!(Variant::Correct.requires_bit_exact());
        assert!(!Variant::Faithful.requires_bit_exact());
        assert!(!Variant::Fast.requires_bit_exact());
    }

    #[test]
    fn sched_tier_all_has_four_variants() {
        assert_eq!(SchedTier::ALL.len(), 4);
    }

    #[test]
    fn sched_tier_spin_rounds_documented_floor() {
        assert_eq!(SchedTier::Inline.spin_rounds(), 0);
        assert_eq!(SchedTier::Local.spin_rounds(), 8);
        assert_eq!(SchedTier::Hierarchical.spin_rounds(), 32);
        assert_eq!(SchedTier::Federated.spin_rounds(), 0);
    }

    #[test]
    fn sched_tier_default_is_inline() {
        assert_eq!(SchedTier::default(), SchedTier::Inline);
    }

    #[test]
    fn hw_class_matrix_extension_classifier() {
        assert!(!HwClass::Scalar.is_matrix_extension());
        assert!(!HwClass::Avx2.is_matrix_extension());
        assert!(!HwClass::Avx512f.is_matrix_extension());
        assert!(!HwClass::Neon.is_matrix_extension());
        // The 2025/2026 vector rungs are vector regime, not matrix.
        assert!(!HwClass::Avx512Vbmi2.is_matrix_extension());
        assert!(!HwClass::AvxVnniVex.is_matrix_extension());
        assert!(!HwClass::Gfni.is_matrix_extension());
        assert!(!HwClass::Avx512Fp16.is_matrix_extension());
        assert!(!HwClass::Avx10_2.is_matrix_extension());
        assert!(!HwClass::Sve2.is_matrix_extension());
        assert!(HwClass::Sme.is_matrix_extension());
        assert!(HwClass::AmxBf16.is_matrix_extension());
        assert!(HwClass::AmxInt8.is_matrix_extension());
        assert!(HwClass::AmxFp16.is_matrix_extension());
        assert!(HwClass::TensorCoreHopper.is_matrix_extension());
        assert!(HwClass::TensorCoreBlackwell.is_matrix_extension());
    }

    #[test]
    fn hw_class_default_is_scalar() {
        assert_eq!(HwClass::default(), HwClass::Scalar);
    }

    #[test]
    fn sched_tier_display_round_trip() {
        assert_eq!(format!("{}", SchedTier::Inline), "inline");
        assert_eq!(format!("{}", SchedTier::Local), "local");
        assert_eq!(format!("{}", SchedTier::Hierarchical), "hierarchical");
        assert_eq!(format!("{}", SchedTier::Federated), "federated");
    }

    #[test]
    fn hw_class_display_for_every_variant() {
        assert!(!format!("{}", HwClass::Scalar).is_empty());
        assert!(!format!("{}", HwClass::Avx512f).is_empty());
        assert!(!format!("{}", HwClass::AmxBf16).is_empty());
        assert!(!format!("{}", HwClass::TensorCoreBlackwell).is_empty());
        assert_eq!(format!("{}", HwClass::Avx512Vbmi2), "avx512vbmi2");
        assert_eq!(format!("{}", HwClass::AvxVnniVex), "avx-vnni");
        assert_eq!(format!("{}", HwClass::Gfni), "gfni");
        assert_eq!(format!("{}", HwClass::Avx512Fp16), "avx512fp16");
        assert_eq!(format!("{}", HwClass::Avx10_2), "avx10.2");
        assert_eq!(format!("{}", HwClass::Sve2), "sve2");
    }
}
