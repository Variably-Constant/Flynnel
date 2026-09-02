//! Adaptive bisect-variant routing migration.
//!
//! Fifth process-global AtomicU8 adaptive surface alongside
//! [`crate::sched::adaptive_worker`] (K_gating),
//! [`crate::sched::adaptive_profile`] (DispatchProfile),
//! [`crate::sched::adaptive_backend`] (Backend selection), and
//! [`crate::sched::adaptive_cooperative`] (cooperative routing).
//! Extends the pattern to bisect-variant selection in
//! [`crate::sched::par_iter::for_each_chunk`].
//!
//! [`VariantRouting::ComputeBatchAdaptive`] picks between
//! `ProducerMaxLenWorkers` (large N) and `RayonStyleReplenish`
//! (small N) for PortBound work; other profiles use the default
//! lazy-steal bisect. Resolved once per dispatch entry: one
//! AtomicU8 Acquire-load + branch (~1 ns); zero cost on the deque
//! hot path; migration is one Release-store.
//!
//! CPUID-resolved default: AMD -> `ComputeBatchAdaptive`, Intel /
//! other -> `Default`. Measured (Xeon Cascade Lake 12T, EPYC 9B14
//! 44T, Zen3 5700G 16T): wins on AMD Compute (Zen3 +37.6% at 10k,
//! +19.6% at 100k, Genoa tied), never regresses AMD Heavy, loses
//! 5-9% on Intel Compute and Heavy/100k. Any host can override via
//! [`migrate_variant_routing`].
//!
//! Precedence: per-plan
//! [`crate::sched::JobPlan::bisect_variant`] wins; else the
//! process-global tag ([`active_variant_routing`]) at plan
//! construction; else the CPUID default ([`cpuid_default_routing`]).

#![allow(clippy::missing_errors_doc)]

use core::sync::atomic::{AtomicU8, Ordering};

use crate::cpu_info::{Vendor, cpu_info};
use crate::dispatch_profile::DispatchProfile;
use crate::sched::plan::BisectVariant;

/// Routing decision for [`crate::sched::par_iter::for_each_chunk`].
///
/// `Auto` defers to [`cpuid_default_routing`]; the other variants
/// pin the routing for the rest of the process.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Default)]
pub enum VariantRouting {
    /// Defer to the CPUID-based default from [`cpuid_default_routing`].
    /// Initial state of the process-global tag.
    #[default]
    Auto,
    /// No experiment-variant routing; every `for_each_chunk` call
    /// uses the default lazy-steal bisect path. Set as the CPUID-
    /// default on Intel and other non-AMD hosts based on the
    /// measured 5-9% regression of `ComputeBatchAdaptive` on those
    /// vendors. Vendor-neutral as a primitive: explicit
    /// `migrate_variant_routing(Default)` makes any host use this
    /// routing.
    Default,
    /// Batch-size-adaptive bisect-variant selector for Compute
    /// (`DispatchProfile::PortBound`) workloads:
    /// [`pick_variant_for_profile`] returns
    /// `ProducerMaxLenWorkers` for `batch_size >=
    /// COMPUTE_BATCH_LARGE_N` (50_000) and `RayonStyleReplenish`
    /// otherwise. All other profiles return `None` (default
    /// lazy-steal bisect).
    ///
    /// CPUID-default on AMD hosts (measured wins: Zen3 Compute/10k
    /// +37.6%, Zen3 Compute/100k +19.6%). Vendor-neutral as a
    /// primitive: explicit `migrate_variant_routing(ComputeBatchAdaptive)`
    /// activates it on Intel or any other vendor for callers who
    /// measure a win on their specific workload.
    ComputeBatchAdaptive,
}

const TAG_AUTO: u8 = 0;
const TAG_DEFAULT: u8 = 1;
const TAG_COMPUTE_BATCH_ADAPTIVE: u8 = 2;

/// Batch-size threshold for the [`VariantRouting::ComputeBatchAdaptive`]
/// routing fork. At
/// `batch_size >= COMPUTE_BATCH_LARGE_N` the routing picks
/// `ProducerMaxLenWorkers` (Zen3 Compute/100k +19.6%); below it
/// picks `RayonStyleReplenish` (Zen3 Compute/10k +37.6%). Genoa
/// 44T is tied on both sides so this fork is safe to fire on any
/// AMD host with vendor == Amd.
pub const COMPUTE_BATCH_LARGE_N: u32 = 50_000;

static ACTIVE_VARIANT_TAG: AtomicU8 = AtomicU8::new(TAG_AUTO);

/// Linkage confirmation marker. When the binary links this
/// module, `nm <bin> | grep __flynnel_marker` returns this
/// symbol, confirming the adaptive variant routing path is
/// present in the build.
#[unsafe(no_mangle)]
pub static __flynnel_marker_adaptive_variant_routing: u8 = 0;

/// Resolve the per-vendor default routing from CPUID. Called by
/// [`active_variant_routing`] when the process-global tag is `Auto`.
#[inline]
pub fn cpuid_default_routing() -> VariantRouting {
    match cpu_info().vendor {
        Vendor::Amd => VariantRouting::ComputeBatchAdaptive,
        Vendor::Intel | Vendor::Other => VariantRouting::Default,
    }
}

/// Read the active [`VariantRouting`] via one AtomicU8 Acquire-load.
/// When the tag is `Auto` (initial process state), delegates to the
/// CPUID-resolved default.
#[inline]
pub fn active_variant_routing() -> VariantRouting {
    match ACTIVE_VARIANT_TAG.load(Ordering::Acquire) {
        TAG_DEFAULT => VariantRouting::Default,
        TAG_COMPUTE_BATCH_ADAPTIVE => VariantRouting::ComputeBatchAdaptive,
        _ => cpuid_default_routing(),
    }
}

/// Migrate the global active variant routing via one AtomicU8
/// Release-store. Subsequent [`crate::sched::JobPlan::new`] /
/// `set_profile` constructions resolve the per-plan
/// `bisect_variant` field through the new routing.
#[inline]
pub fn migrate_variant_routing(routing: VariantRouting) {
    let tag = match routing {
        VariantRouting::Auto => TAG_AUTO,
        VariantRouting::Default => TAG_DEFAULT,
        VariantRouting::ComputeBatchAdaptive => TAG_COMPUTE_BATCH_ADAPTIVE,
    };
    ACTIVE_VARIANT_TAG.store(tag, Ordering::Release);
}

/// Resolve a `(profile, batch_size)` pair to an experiment variant
/// per the active routing. Returns `None` (use the default lazy-
/// steal bisect) when:
///
/// - The active routing is [`VariantRouting::Default`].
/// - The active routing is [`VariantRouting::ComputeBatchAdaptive`] but
///   the profile is not `PortBound` (Compute / Light workloads
///   map to PortBound per
///   [`crate::sched::adaptive_profile::WorkloadClass::to_dispatch_profile`]).
///
/// Otherwise, picks `ProducerMaxLenWorkers` for
/// `batch_size >= COMPUTE_BATCH_LARGE_N` and `RayonStyleReplenish`
/// for smaller batches.
#[inline]
pub fn pick_variant_for_profile(
    profile: DispatchProfile,
    batch_size: u32,
) -> Option<BisectVariant> {
    match active_variant_routing() {
        VariantRouting::Default => None,
        VariantRouting::Auto => unreachable!("Auto resolves via active_variant_routing"),
        VariantRouting::ComputeBatchAdaptive => {
            if profile != DispatchProfile::PortBound {
                return None;
            }
            if batch_size >= COMPUTE_BATCH_LARGE_N {
                Some(BisectVariant::ProducerMaxLenWorkers)
            } else {
                Some(BisectVariant::RayonStyleReplenish)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Mutex, MutexGuard, OnceLock};

    fn global_test_lock() -> MutexGuard<'static, ()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
            .lock()
            .expect("global_test_lock poisoned by prior test panic")
    }

    struct TestGuard {
        _lock: MutexGuard<'static, ()>,
    }
    impl TestGuard {
        fn new() -> Self {
            let lock = global_test_lock();
            migrate_variant_routing(VariantRouting::Auto);
            Self { _lock: lock }
        }
    }
    impl Drop for TestGuard {
        fn drop(&mut self) {
            migrate_variant_routing(VariantRouting::Auto);
        }
    }

    #[test]
    fn default_active_routing_resolves_from_cpuid() {
        let _guard = TestGuard::new();
        let active = active_variant_routing();
        // Whichever the host CPUID resolves to must match the
        // cpuid_default_routing helper directly.
        assert_eq!(active, cpuid_default_routing());
    }

    #[test]
    fn migration_changes_active_routing() {
        let _guard = TestGuard::new();
        migrate_variant_routing(VariantRouting::Default);
        assert_eq!(active_variant_routing(), VariantRouting::Default);
        migrate_variant_routing(VariantRouting::ComputeBatchAdaptive);
        assert_eq!(active_variant_routing(), VariantRouting::ComputeBatchAdaptive);
        migrate_variant_routing(VariantRouting::Auto);
        assert_eq!(active_variant_routing(), cpuid_default_routing());
    }

    #[test]
    fn pick_variant_force_default_returns_none() {
        let _guard = TestGuard::new();
        migrate_variant_routing(VariantRouting::Default);
        assert_eq!(pick_variant_for_profile(DispatchProfile::PortBound, 10_000), None);
        assert_eq!(pick_variant_for_profile(DispatchProfile::PortBound, 100_000), None);
        assert_eq!(pick_variant_for_profile(DispatchProfile::LatencyBound, 100_000), None);
    }

    #[test]
    fn pick_variant_amd_compute_picks_by_batch_size() {
        let _guard = TestGuard::new();
        migrate_variant_routing(VariantRouting::ComputeBatchAdaptive);
        // Small N -> RayonStyleReplenish
        assert_eq!(
            pick_variant_for_profile(DispatchProfile::PortBound, 10_000),
            Some(BisectVariant::RayonStyleReplenish)
        );
        // At threshold -> ProducerMaxLenWorkers
        assert_eq!(
            pick_variant_for_profile(DispatchProfile::PortBound, COMPUTE_BATCH_LARGE_N),
            Some(BisectVariant::ProducerMaxLenWorkers)
        );
        // Large N -> ProducerMaxLenWorkers
        assert_eq!(
            pick_variant_for_profile(DispatchProfile::PortBound, 100_000),
            Some(BisectVariant::ProducerMaxLenWorkers)
        );
        // Non-PortBound profile -> None
        assert_eq!(
            pick_variant_for_profile(DispatchProfile::LatencyBound, 100_000),
            None
        );
        assert_eq!(
            pick_variant_for_profile(DispatchProfile::MemoryBound, 100_000),
            None
        );
    }

    #[test]
    fn cpuid_default_matches_host_vendor() {
        // No guard needed: this test only reads, doesn't mutate the
        // global tag. cpu_info() is cached per-process.
        let info = cpu_info();
        let expected = match info.vendor {
            Vendor::Amd => VariantRouting::ComputeBatchAdaptive,
            _ => VariantRouting::Default,
        };
        assert_eq!(cpuid_default_routing(), expected);
    }
}
