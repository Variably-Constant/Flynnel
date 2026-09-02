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
static ACTIVE_COOPERATIVE_TAG: AtomicU8 = AtomicU8::new(TAG_AUTO);

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
    match ACTIVE_COOPERATIVE_TAG.load(Ordering::Acquire) {
        TAG_FORCE_TREE => CooperativeRouting::ForceTree,
        TAG_FORCE_MAILBOX => CooperativeRouting::ForceMailbox,
        TAG_FORCE_DEQUE => CooperativeRouting::ForceDeque,
        _ => CooperativeRouting::Auto,
    }
}

/// Migrate the global active cooperative routing via one AtomicU8
/// Release-store. Subsequent
/// [`crate::sched::cooperative::cooperative_join_n`] calls that
/// see a per-plan `cooperative_routing == Auto` consult the new
/// value via one Acquire-load.
#[inline]
pub fn migrate_cooperative_routing(routing: CooperativeRouting) {
    let tag = match routing {
        CooperativeRouting::Auto => TAG_AUTO,
        CooperativeRouting::ForceTree => TAG_FORCE_TREE,
        CooperativeRouting::ForceMailbox => TAG_FORCE_MAILBOX,
        CooperativeRouting::ForceDeque => TAG_FORCE_DEQUE,
    };
    ACTIVE_COOPERATIVE_TAG.store(tag, Ordering::Release);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Mutex, MutexGuard, OnceLock};

    /// Process-wide mutex serializing the tests in this module so
    /// parallel test runs do not race on the shared
    /// ACTIVE_COOPERATIVE_TAG global: tests that mutate shared
    /// process state serialize on one lock.
    fn global_test_lock() -> MutexGuard<'static, ()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
            .lock()
            .expect("global_test_lock poisoned by prior test panic")
    }

    /// RAII guard: acquires the serializing lock and restores the
    /// default `Auto` routing on drop so cross-test state does not
    /// leak between runs.
    struct TestGuard {
        _lock: MutexGuard<'static, ()>,
    }
    impl TestGuard {
        fn new() -> Self {
            let lock = global_test_lock();
            migrate_cooperative_routing(CooperativeRouting::Auto);
            Self { _lock: lock }
        }
    }
    impl Drop for TestGuard {
        fn drop(&mut self) {
            migrate_cooperative_routing(CooperativeRouting::Auto);
        }
    }

    #[test]
    fn default_active_routing_is_auto() {
        let _guard = TestGuard::new();
        assert_eq!(active_cooperative_routing(), CooperativeRouting::Auto);
    }

    #[test]
    fn migration_changes_active_routing() {
        let _guard = TestGuard::new();

        migrate_cooperative_routing(CooperativeRouting::ForceTree);
        assert_eq!(active_cooperative_routing(), CooperativeRouting::ForceTree);

        migrate_cooperative_routing(CooperativeRouting::ForceMailbox);
        assert_eq!(
            active_cooperative_routing(),
            CooperativeRouting::ForceMailbox
        );

        migrate_cooperative_routing(CooperativeRouting::ForceDeque);
        assert_eq!(
            active_cooperative_routing(),
            CooperativeRouting::ForceDeque
        );

        migrate_cooperative_routing(CooperativeRouting::Auto);
        assert_eq!(active_cooperative_routing(), CooperativeRouting::Auto);
    }

    #[test]
    fn migration_propagates_across_threads() {
        let _guard = TestGuard::new();
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::Arc;
        use std::thread;
        use std::time::{Duration, Instant};

        migrate_cooperative_routing(CooperativeRouting::ForceTree);

        let observed_tree = Arc::new(AtomicBool::new(false));
        let observed_mailbox = Arc::new(AtomicBool::new(false));

        let ot = Arc::clone(&observed_tree);
        let om = Arc::clone(&observed_mailbox);
        // Deadline-based loop: spin until BOTH values observed or
        // 500 ms elapses. Iteration-count loops race with the main
        // thread's 5 ms sleep on fast hosts (1M atomic loads finish
        // in <5 ms on Zen+, so producer can exit before the second
        // migration ever lands).
        let deadline = Instant::now() + Duration::from_millis(500);
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

        // Give the producer enough iterations to observe Tree, then flip.
        std::thread::sleep(Duration::from_millis(5));
        migrate_cooperative_routing(CooperativeRouting::ForceMailbox);

        producer.join().expect("producer thread should not panic");

        assert!(observed_tree.load(Ordering::Relaxed), "producer never saw ForceTree");
        assert!(
            observed_mailbox.load(Ordering::Relaxed),
            "producer never saw ForceMailbox after migration"
        );
    }

    #[test]
    fn default_routing_value_is_auto() {
        assert_eq!(CooperativeRouting::default(), CooperativeRouting::Auto);
    }
}
