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

static ACTIVE_VARIANT_TAG: AtomicU8 = AtomicU8::new(ACTIVE_VARIANT_TAG_DEFAULT);

/// The tag [`ACTIVE_VARIANT_TAG`] holds before anything migrates it.
const ACTIVE_VARIANT_TAG_DEFAULT: u8 = TAG_AUTO;

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
    match routing_of_tag(ACTIVE_VARIANT_TAG.load(Ordering::Acquire)) {
        VariantRouting::Auto => cpuid_default_routing(),
        named => named,
    }
}

/// The routing a stored tag names, `Auto` for any value no routing is
/// stored as.
#[inline]
const fn routing_of_tag(tag: u8) -> VariantRouting {
    match tag {
        TAG_DEFAULT => VariantRouting::Default,
        TAG_COMPUTE_BATCH_ADAPTIVE => VariantRouting::ComputeBatchAdaptive,
        _ => VariantRouting::Auto,
    }
}

/// The tag a routing is stored as.
#[inline]
const fn tag_of_routing(routing: VariantRouting) -> u8 {
    match routing {
        VariantRouting::Auto => TAG_AUTO,
        VariantRouting::Default => TAG_DEFAULT,
        VariantRouting::ComputeBatchAdaptive => TAG_COMPUTE_BATCH_ADAPTIVE,
    }
}

/// Migrate the global active variant routing via one AtomicU8
/// Release-store. Subsequent [`crate::sched::JobPlan::new`] /
/// `set_profile` constructions resolve the per-plan
/// `bisect_variant` field through the new routing.
#[inline]
pub fn migrate_variant_routing(routing: VariantRouting) {
    ACTIVE_VARIANT_TAG.store(tag_of_routing(routing), Ordering::Release);
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
    pick_variant_under(active_variant_routing(), profile, batch_size)
}

/// The same choice under a routing the caller names, so a reader of
/// this decision does not have to move the process-wide tag to see it.
/// `Auto` resolves to the CPUID default, as [`active_variant_routing`]
/// does.
#[inline]
fn pick_variant_under(
    routing: VariantRouting,
    profile: DispatchProfile,
    batch_size: u32,
) -> Option<BisectVariant> {
    let resolved = match routing {
        VariantRouting::Auto => cpuid_default_routing(),
        named => named,
    };
    match resolved {
        VariantRouting::Default => None,
        VariantRouting::Auto => unreachable!("Auto resolved above"),
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

    /// Every routing survives the round trip the global tag performs,
    /// and any other tag reads as Auto.
    ///
    /// Over the tag rather than the process-wide cell, so the tests in
    /// this module neither wait for each other nor answer each other's
    /// reads.
    #[test]
    fn every_routing_survives_its_tag() {
        for routing in [
            VariantRouting::Auto,
            VariantRouting::Default,
            VariantRouting::ComputeBatchAdaptive,
        ] {
            assert_eq!(routing_of_tag(tag_of_routing(routing)), routing);
        }
        assert_eq!(routing_of_tag(TAG_AUTO), VariantRouting::Auto);
        assert_eq!(routing_of_tag(200), VariantRouting::Auto);
    }

    #[test]
    fn the_cell_starts_on_the_cpuid_default() {
        assert_eq!(
            routing_of_tag(ACTIVE_VARIANT_TAG_DEFAULT),
            VariantRouting::Auto,
            "the cell's initial tag defers to CPUID"
        );
        assert_eq!(
            pick_variant_under(VariantRouting::Auto, DispatchProfile::PortBound, 10_000),
            pick_variant_under(cpuid_default_routing(), DispatchProfile::PortBound, 10_000),
            "Auto picks what this host's CPUID default picks"
        );
    }

    #[test]
    fn pick_variant_force_default_returns_none() {
        for (profile, batch) in [
            (DispatchProfile::PortBound, 10_000),
            (DispatchProfile::PortBound, 100_000),
            (DispatchProfile::LatencyBound, 100_000),
        ] {
            assert_eq!(
                pick_variant_under(VariantRouting::Default, profile, batch),
                None
            );
        }
    }

    #[test]
    fn pick_variant_amd_compute_picks_by_batch_size() {
        let routing = VariantRouting::ComputeBatchAdaptive;
        // Small N -> RayonStyleReplenish
        assert_eq!(
            pick_variant_under(routing, DispatchProfile::PortBound, 10_000),
            Some(BisectVariant::RayonStyleReplenish)
        );
        // At threshold -> ProducerMaxLenWorkers
        assert_eq!(
            pick_variant_under(routing, DispatchProfile::PortBound, COMPUTE_BATCH_LARGE_N),
            Some(BisectVariant::ProducerMaxLenWorkers)
        );
        // Large N -> ProducerMaxLenWorkers
        assert_eq!(
            pick_variant_under(routing, DispatchProfile::PortBound, 100_000),
            Some(BisectVariant::ProducerMaxLenWorkers)
        );
        // Non-PortBound profile -> None
        assert_eq!(
            pick_variant_under(routing, DispatchProfile::LatencyBound, 100_000),
            None
        );
        assert_eq!(
            pick_variant_under(routing, DispatchProfile::MemoryBound, 100_000),
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
