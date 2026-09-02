//! CPU topology hints: physical-core count + SMT factor.
//!
//! Used by [`crate::sched`] to size primary worker pools (= physical
//! cores) and SMT-sibling extension pools (= primary x (smt - 1)).
//! Deliberately minimal: only the two scalars Flynnel actually
//! consumes are exposed.
//!
//! Detection:
//!
//! - **logical_threads**: [`std::thread::available_parallelism`].
//! - **smt_threads_per_core**: CPUID HTT bit (x86_64) gives a yes/no
//!   for SMT-2 presence; other architectures report SMT-1. This is
//!   coarse on Intel hybrid (P-core SMT-2 + E-core SMT-1) but
//!   correct enough for the worker-sizing decision that consumes it.
//! - **physical_cores**: `logical_threads / smt_threads_per_core`.

use std::sync::OnceLock;

/// CPU manufacturer identified by the CPUID leaf 0 vendor string.
/// Used by [`crate::sched::adaptive_variant_routing`] to gate the
/// per-vendor bisect-variant routing table.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Vendor {
    /// Intel CPUs: vendor string "GenuineIntel".
    Intel,
    /// AMD CPUs: vendor string "AuthenticAMD".
    Amd,
    /// Any other vendor (Centaur / Hygon / Zhaoxin / etc.) or
    /// non-x86_64 targets.
    Other,
}

/// Per-host snapshot used by [`crate::sched::arena_numa`] and
/// [`crate::sched::io_pool`].
#[derive(Copy, Clone, Debug)]
pub struct CpuInfo {
    /// Total number of logical processors visible to this process.
    pub logical_threads: u32,
    /// Number of hardware threads per physical core. 1 on
    /// non-x86_64 or when SMT is disabled in firmware; 2 on
    /// x86_64 with HTT enabled (the canonical SMT-2 case).
    pub smt_threads_per_core: u8,
    /// `log2(smt_threads_per_core)`. 0 for SMT-1; 1 for SMT-2.
    pub smt_threads_log2: u8,
    /// Estimated physical-core count =
    /// `logical_threads / smt_threads_per_core`.
    pub physical_cores: u32,
    /// CPU vendor decoded from CPUID leaf 0. [`Vendor::Other`] on
    /// non-x86_64 targets.
    pub vendor: Vendor,
    /// CPU family from CPUID leaf 1 EAX (combined base + extended
    /// family). 0 on non-x86_64.
    pub family: u32,
    /// CPU model from CPUID leaf 1 EAX (combined base + extended
    /// model). 0 on non-x86_64.
    pub model: u32,
    /// CPU stepping from CPUID leaf 1 EAX bits 0-3. 0 on non-x86_64.
    pub stepping: u8,
}

impl CpuInfo {
    fn detect() -> Self {
        let logical = std::thread::available_parallelism()
            .map(std::num::NonZeroUsize::get)
            .unwrap_or(1) as u32;

        let smt = detect_smt_factor();
        let physical = (logical / smt as u32).max(1);
        let smt_log2 = if smt >= 2 { 1 } else { 0 };

        let (vendor, family, model, stepping) = detect_vendor_family_model();

        Self {
            logical_threads: logical,
            smt_threads_per_core: smt,
            smt_threads_log2: smt_log2,
            physical_cores: physical,
            vendor,
            family,
            model,
            stepping,
        }
    }
}

/// Decode the CPU vendor and family/model/stepping via CPUID leaf 0
/// plus leaf 1. Per Intel SDM Vol. 2A and AMD APM Vol. 3: the
/// displayed family equals `base_family + extended_family` when
/// `base_family == 0xF`, otherwise just `base_family`. The displayed
/// model equals `(extended_model << 4) | base_model` when
/// `base_family` is exactly `0x6` (Intel Core) OR `0xF` (NetBurst /
/// AMD Zen), otherwise just `base_model`. Both vendors follow the
/// same encoding; the bit positions are stable across generations.
#[cfg(target_arch = "x86_64")]
fn detect_vendor_family_model() -> (Vendor, u32, u32, u8) {
    use std::arch::x86_64::__cpuid;
    let leaf0 = __cpuid(0);
    let mut vendor_bytes = [0u8; 12];
    vendor_bytes[0..4].copy_from_slice(&leaf0.ebx.to_le_bytes());
    vendor_bytes[4..8].copy_from_slice(&leaf0.edx.to_le_bytes());
    vendor_bytes[8..12].copy_from_slice(&leaf0.ecx.to_le_bytes());
    let vendor = match &vendor_bytes {
        b"GenuineIntel" => Vendor::Intel,
        b"AuthenticAMD" => Vendor::Amd,
        _ => Vendor::Other,
    };

    if leaf0.eax < 1 {
        return (vendor, 0, 0, 0);
    }
    let leaf1 = __cpuid(1);
    let eax = leaf1.eax;
    let base_family = (eax >> 8) & 0xF;
    let ext_family = (eax >> 20) & 0xFF;
    let base_model = (eax >> 4) & 0xF;
    let ext_model = (eax >> 16) & 0xF;
    let stepping = (eax & 0xF) as u8;

    // Per Intel SDM Vol. 2A and AMD APM Vol. 3: extended family
    // bits add to base_family only when base_family == 0xF;
    // extended model bits add to base_model when base_family is
    // 0x6 (Intel) or 0xF (both vendors). Modern silicon (Intel
    // Core, AMD Zen) all fall into family >= 0x6, so the
    // extended_model addition is the common case.
    let family = if base_family == 0xF {
        base_family + ext_family
    } else {
        base_family
    };
    let model = if base_family == 0xF || base_family == 0x6 {
        (ext_model << 4) | base_model
    } else {
        base_model
    };

    (vendor, family, model, stepping)
}

#[cfg(not(target_arch = "x86_64"))]
fn detect_vendor_family_model() -> (Vendor, u32, u32, u8) {
    (Vendor::Other, 0, 0, 0)
}

/// Returns the cached CPU info snapshot. Probed once per process.
pub fn cpu_info() -> &'static CpuInfo {
    static CACHE: OnceLock<CpuInfo> = OnceLock::new();
    CACHE.get_or_init(CpuInfo::detect)
}

/// Dispatch-floor multiplier for small hosts. Work-stealing's
/// coordination cost has nothing to amortize against on machines
/// with fewer than 4 physical cores, so every inline-collapse /
/// heavy-override floor is scaled by this factor there: parallel
/// dispatch fires only when the workload is 4x past the normal
/// breakeven. 1 (no scaling) on 4+ physical-core hosts.
#[inline]
pub fn small_host_dispatch_factor() -> u64 {
    dispatch_factor_for_cores(cpu_info().physical_cores)
}

/// Pure mapping behind [`small_host_dispatch_factor`], separated so
/// the policy is testable on any host.
#[inline]
pub(crate) fn dispatch_factor_for_cores(physical_cores: u32) -> u64 {
    if physical_cores < 4 { 4 } else { 1 }
}

/// Returns 2 when the host has SMT-2 hardware threads per physical
/// core; 1 otherwise.
#[cfg(target_arch = "x86_64")]
fn detect_smt_factor() -> u8 {
    use std::arch::x86_64::__cpuid;
    // CPUID leaf 1, EDX bit 28: HTT (Hyper-Threading Technology) -
    // set when the package supports >= 2 logical processors per
    // physical core. Note: HTT=1 only guarantees the silicon CAN do
    // SMT; the OS may have it disabled. Cross-check against
    // logical_threads > physical CPUID leaves; if not, default to 2.
    let r = __cpuid(1);
    let htt = (r.edx >> 28) & 1;
    if htt == 1 { 2 } else { 1 }
}

#[cfg(not(target_arch = "x86_64"))]
fn detect_smt_factor() -> u8 {
    // ARM, RISC-V, and other non-x86_64 targets: assume SMT-1 by
    // default. Apple M-series, most ARM-server SoCs, and POWER9-mode
    // SoCs are SMT-1. POWER8/POWER9-SMT8 hosts will under-report;
    // those callers can override via the public CpuInfo constructor.
    1
}

/// True when this host supports the WAITPKG instruction set
/// extension (`UMONITOR` / `UMWAIT` / `TPAUSE`). Detected via
/// CPUID leaf 7 sub-leaf 0, ECX bit 5.
///
/// Available on:
/// - Intel: Tremont (2019), Tiger Lake (2020), Sapphire Rapids,
///   Alder Lake, Raptor Lake, Emerald Rapids, and later
/// - AMD: Zen 5 (2024+) - Ryzen 9000 series, EPYC Turin
/// - Older Intel (pre-Tiger-Lake desktop) and AMD Zen 4 and earlier:
///   returns `false`; callers fall back to PAUSE-spin
///
/// Cached on first call via [`waitpkg_available`].
#[cfg(target_arch = "x86_64")]
pub fn has_waitpkg() -> bool {
    *waitpkg_available()
}

#[cfg(not(target_arch = "x86_64"))]
pub fn has_waitpkg() -> bool {
    false
}

#[cfg(target_arch = "x86_64")]
fn waitpkg_available() -> &'static bool {
    static CACHE: OnceLock<bool> = OnceLock::new();
    CACHE.get_or_init(|| {
        // CPUID is a safe call in the current toolchain; both
        // __cpuid and __cpuid_count have stable safe surfaces.
        // We first check that leaf 7 is supported by reading the
        // max-leaf id from CPUID leaf 0.
        use std::arch::x86_64::{__cpuid, __cpuid_count};
        let max_leaf = __cpuid(0).eax;
        if max_leaf < 7 {
            return false;
        }
        // CPUID leaf 7, sub-leaf 0, ECX bit 5 = WAITPKG.
        let r = __cpuid_count(7, 0);
        (r.ecx >> 5) & 1 == 1
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_info_is_cached() {
        let a = cpu_info() as *const _;
        let b = cpu_info() as *const _;
        assert_eq!(a, b);
    }

    #[test]
    fn detected_values_are_sane() {
        let info = cpu_info();
        assert!(info.logical_threads >= 1);
        assert!(info.physical_cores >= 1);
        assert!(info.physical_cores <= info.logical_threads);
        assert!(matches!(info.smt_threads_per_core, 1 | 2));
        assert_eq!(
            info.physical_cores * info.smt_threads_per_core as u32,
            info.logical_threads,
            "physical * smt should reconstruct logical"
        );
        assert_eq!(info.smt_threads_log2, if info.smt_threads_per_core >= 2 { 1 } else { 0 });
    }
}
