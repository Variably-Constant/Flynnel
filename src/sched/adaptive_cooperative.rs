//! Adaptive cooperative-routing migration.
//!
//! The same AtomicU-tag pattern proven in
//! [`crate::sched::adaptive_worker`] (K_gating),
//! [`crate::sched::adaptive_profile`] (DispatchProfile), and
//! [`crate::sched::adaptive_backend`] (Backend selection) extended
//! to cooperative-routing selection
//! (Tree / FlatDeque / FlatMailbox).
//!
//! [`crate::sched::cooperative::cooperative_join_n`] consults the
//! routing (Tree / FlatDeque / FlatMailbox) once per call entry:
//! zero cost on the deque hot path, one AtomicU8 Acquire-load per
//! call, one Release-store per migration.
//!
//! Precedence: per-plan `JobPlan::cooperative_routing` (when not
//! `Auto`) wins; else the process-global
//! [`active_cooperative_routing`] tag; else the population
//! heuristic (`N < n_workers` -> tree, `N >= n_workers` ->
//! mailbox). [`migrate_cooperative_routing`] composes with the
//! other adaptive axes (Backend, DispatchProfile, KGating), each
//! an independent AtomicU8.

#![allow(clippy::missing_errors_doc)]

use core::sync::atomic::{AtomicU8, Ordering};

/// Routing decision for [`crate::sched::cooperative::cooperative_join_n`].
///
/// `Auto` is the default at every layer (per-plan field default
/// alongside the process-global initial value); the call falls
/// through to the population heuristic when both layers are `Auto`.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Default)]
pub enum CooperativeRouting {
    /// Defer to the next layer in the precedence chain. Per-plan
    /// `Auto` defers to the process-global tag; global `Auto`
    /// defers to the population heuristic.
    #[default]
    Auto,
    /// Force the tree-bisect shape via
    /// [`crate::sched::cooperative::cooperative_join_n_tree`].
    /// Pick this when per-closure work is short (sub-100us) and
    /// the tree's amortized setup wins over the flat fan-out's
    /// per-StackJob cost.
    ForceTree,
    /// Force the mailbox-distribute shape via
    /// [`crate::sched::cooperative::cooperative_join_n_flat_mailbox`].
    /// Pick this when N matches the host worker pool size and
    /// each closure should land on a specific peer's mailbox.
    ForceMailbox,
    /// Force the deque fan-out shape via
    /// [`crate::sched::cooperative::cooperative_join_n_flat`].
    /// Pick this when broad random peer-steal load balance is
    /// preferred over owner-directed mailbox routing (e.g.
    /// heterogeneous-cost closures where mailbox concentration
    /// can pin a slow closure on one worker while others idle).
    ForceDeque,
}

/// Encoded active-routing tags stored in [`ACTIVE_COOPERATIVE_TAG`].
const TAG_AUTO: u8 = 0;
const TAG_FORCE_TREE: u8 = 1;
const TAG_FORCE_MAILBOX: u8 = 2;
const TAG_FORCE_DEQUE: u8 = 3;

/// Global active-routing tag. Read by
/// [`crate::sched::cooperative::cooperative_join_n`] when the
/// per-plan `cooperative_routing` field is `Auto`; flipped by
/// [`migrate_cooperative_routing`]. Initial value: `Auto` (defer
/// to the population heuristic).
static ACTIVE_COOPERATIVE_TAG: AtomicU8 = AtomicU8::new(ACTIVE_COOPERATIVE_TAG_DEFAULT);

/// The tag [`ACTIVE_COOPERATIVE_TAG`] holds before anything migrates it.
const ACTIVE_COOPERATIVE_TAG_DEFAULT: u8 = TAG_AUTO;

/// Linkage confirmation marker. When the binary links this
/// module, `nm <bin> | grep __flynnel_marker` returns this
/// symbol, confirming the adaptive cooperative routing dispatch
/// path is present in the build.
#[unsafe(no_mangle)]
pub static __flynnel_marker_adaptive_cooperative: u8 = 0;

/// Read the active [`CooperativeRouting`] via one AtomicU8
/// Acquire-load. Consumed by `cooperative_join_n` when the
/// per-plan field is `Auto`.
#[inline]
pub fn active_cooperative_routing() -> CooperativeRouting {
    routing_of_tag(ACTIVE_COOPERATIVE_TAG.load(Ordering::Acquire))
}

/// The routing a stored tag names, `Auto` for any value no routing is
/// stored as.
#[inline]
const fn routing_of_tag(tag: u8) -> CooperativeRouting {
    match tag {
        TAG_FORCE_TREE => CooperativeRouting::ForceTree,
        TAG_FORCE_MAILBOX => CooperativeRouting::ForceMailbox,
        TAG_FORCE_DEQUE => CooperativeRouting::ForceDeque,
        _ => CooperativeRouting::Auto,
    }
}

/// The tag a routing is stored as.
#[inline]
const fn tag_of_routing(routing: CooperativeRouting) -> u8 {
    match routing {
        CooperativeRouting::Auto => TAG_AUTO,
        CooperativeRouting::ForceTree => TAG_FORCE_TREE,
        CooperativeRouting::ForceMailbox => TAG_FORCE_MAILBOX,
        CooperativeRouting::ForceDeque => TAG_FORCE_DEQUE,
    }
}

/// Migrate the global active cooperative routing via one AtomicU8
/// Release-store. Subsequent
/// [`crate::sched::cooperative::cooperative_join_n`] calls that
/// see a per-plan `cooperative_routing == Auto` consult the new
/// value via one Acquire-load.
#[inline]
pub fn migrate_cooperative_routing(routing: CooperativeRouting) {
    ACTIVE_COOPERATIVE_TAG.store(tag_of_routing(routing), Ordering::Release);
}

#[cfg(test)]
mod tests {
    use super::*;
    /// Every routing survives the round trip the global tag performs,
    /// the cell starts on Auto, and any other tag reads as Auto.
    ///
    /// Over the tag rather than the process-wide cell, so this test
    /// neither waits for the one below nor answers its reads. That
    /// leaves exactly one test in the module writing the cell.
    #[test]
    fn every_routing_survives_its_tag_and_the_cell_starts_auto() {
        for routing in [
            CooperativeRouting::Auto,
            CooperativeRouting::ForceTree,
            CooperativeRouting::ForceMailbox,
            CooperativeRouting::ForceDeque,
        ] {
            assert_eq!(routing_of_tag(tag_of_routing(routing)), routing);
        }
        assert_eq!(
            routing_of_tag(ACTIVE_COOPERATIVE_TAG_DEFAULT),
            CooperativeRouting::Auto
        );
        assert_eq!(routing_of_tag(200), CooperativeRouting::Auto);
    }

    /// The one test that writes the process-wide cell, because what it
    /// asserts is that a store on one thread reaches another. It
    /// restores Auto before it returns.
    #[test]
    fn migration_propagates_across_threads() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::thread;
        use std::time::{Duration, Instant};

        migrate_cooperative_routing(CooperativeRouting::ForceTree);

        let observed_tree = Arc::new(AtomicBool::new(false));
        let observed_mailbox = Arc::new(AtomicBool::new(false));

        let ot = Arc::clone(&observed_tree);
        let om = Arc::clone(&observed_mailbox);
        // The producer spins until it has seen both values or the
        // deadline passes. The deadline bounds a broken build rather
        // than pacing the test: the main thread waits for the first
        // observation and flips on it, so neither side depends on the
        // other reaching a point within some interval.
        let deadline = Instant::now() + Duration::from_secs(10);
        let producer = thread::spawn(move || {
            while Instant::now() < deadline {
                match active_cooperative_routing() {
                    CooperativeRouting::ForceTree => {
                        ot.store(true, Ordering::Relaxed);
                    }
                    CooperativeRouting::ForceMailbox => {
                        om.store(true, Ordering::Relaxed);
                    }
                    _ => {}
                }
                if ot.load(Ordering::Relaxed) && om.load(Ordering::Relaxed) {
                    return;
                }
                std::hint::spin_loop();
            }
        });

        // Flip once the producer has actually observed the first value,
        // not after an interval it is assumed to observe one in. A
        // thread that has not been scheduled yet has observed nothing,
        // and a flip before its first load leaves it nothing to see -
        // which is a failure of the test's pacing rather than of the
        // migration it is checking.
        let handshake = Instant::now() + Duration::from_secs(10);
        while !observed_tree.load(Ordering::Relaxed) && Instant::now() < handshake {
            std::hint::spin_loop();
        }
        assert!(
            observed_tree.load(Ordering::Relaxed),
            "producer never saw ForceTree in 10 s, so the migration did not reach another thread"
        );
        migrate_cooperative_routing(CooperativeRouting::ForceMailbox);

        producer.join().expect("producer thread should not panic");

        assert!(
            observed_mailbox.load(Ordering::Relaxed),
            "producer never saw ForceMailbox after migration"
        );
        migrate_cooperative_routing(CooperativeRouting::Auto);
    }

    #[test]
    fn default_routing_value_is_auto() {
        assert_eq!(CooperativeRouting::default(), CooperativeRouting::Auto);
    }
}
