//! `Parker`: per-worker park / unpark primitive with a yield-N-then-
//! park spin floor.
//!
//! Built on `std::thread::{park, current().unpark()}`. The std
//! primitive provides the permit-based race resolution: if `unpark`
//! is called before `park`, the permit is stored and the next
//! `park` returns immediately. That eliminates the lost-wakeup
//! window the rayon JEC protocol exists to solve, in exchange for
//! the (cheap) cost of always calling `unpark` even when no one is
//! parked.
//!
//!
//! ## Spin floor policy
//!
//! - Local tier: 8 rounds of `thread::yield_now()` before parking
//!   (per [`crate::sched::SchedTier::spin_rounds`]). Sub-microsecond
//!   work avoids the syscall.
//! - Hierarchical tier: 32 rounds. Multi-microsecond work amortizes
//!   the park / unpark pair.
//! - Federated tier: 0 rounds (direct park). Federated jobs are
//!   millisecond-scale; throughput beats latency.
//!
//! `Parker` accepts the spin-round count at construction so it
//! works across tiers without conditional plumbing.

use core::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::thread::{self, Thread};

/// Selects how the [`Parker`] waits after the spin floor is exhausted.
///
/// Picked at construction time via [`WaitStrategy::pick`]:
/// - WAITPKG-capable silicon (Intel Tremont/Tiger Lake+, AMD Zen 5+)
///   -> [`WaitStrategy::Waitpkg`]: UMONITOR + UMWAIT halt the logical
///   CPU sub-100ns until the watched cache line transitions or the
///   TSC deadline fires. No kernel syscall.
/// - AMD silicon without WAITPKG (Excavator onward, so every Zen
///   before Zen 5) -> [`WaitStrategy::Monitorx`]: MONITORX + MWAITX,
///   the same wake-on-store shape from user mode.
/// - All other silicon -> [`WaitStrategy::StdPark`]: the original
///   `std::thread::park()` path (kernel condvar; ~1us syscall on Linux
///   futex / Windows WaitForSingleObject).
///
/// The strategy is observable via [`Parker::wait_strategy`] for
/// diagnostics + per-host bench gating.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WaitStrategy {
    /// `std::thread::park()`. Permits-based; cross-platform; always
    /// works. Kernel transition on the wait path (~1us).
    StdPark,
    /// `UMONITOR` + `UMWAIT`. Halts the logical CPU; wake-on-store
    /// to the monitored cache line. Sub-100ns wake; no syscall.
    /// Available if and only if [`crate::cpu_info::has_waitpkg`]
    /// is true.
    Waitpkg,
    /// `MONITORX` + `MWAITX`. AMD's user-mode monitor-wait, the same
    /// wake-on-store behavior as the WAITPKG pair. Available if and
    /// only if [`crate::cpu_info::has_monitorx`] is true.
    ///
    /// MWAITX takes a relative count of cycles in EBX where UMWAIT
    /// takes an absolute TSC deadline in EDX:EAX. The two are not the
    /// same quantity, and how far one EBX unit reaches is a property
    /// of the part rather than a constant: on a Ryzen 9 7900X one unit
    /// measures half an RDTSC cycle. Nothing in CPUID reports that
    /// ratio, and a busy host also returns from MWAITX early by an
    /// amount that varies between runs.
    /// [`Parker::wait_via_monitorx`] therefore re-arms toward its own
    /// deadline rather than trusting one instruction to reach it,
    /// which is what makes the arm indifferent to both.
    Monitorx,
}

/// Whether a monitor wait on this host suspends the core, which the
/// CPUID bit does not answer and only the length of a real wait
/// reveals. A Ryzen 7 2700 under load returns from MWAITX at once.
///
/// Never set back once false: a wrong false costs the kernel park
/// that was there anyway, a wrong true costs a spin over the
/// instruction pair on every park.
static MONITOR_HOLDS: AtomicBool = AtomicBool::new(true);

/// Separate waits that have found the monitor not holding.
///
/// The verdict is permanent and process-wide, so it is not taken on
/// one wait's evidence. A host whose monitor never holds supplies
/// these in microseconds; a host whose monitor holds has to be
/// unlucky this many times in separate waits.
static MONITOR_DOUBTS: AtomicU32 = AtomicU32::new(0);

/// Waits that must independently find the monitor not holding before
/// the process stops trying.
///
/// One was too few. On a 7900X, whose monitor holds, a single
/// unlucky wait condemned the process and every later group in that
/// run measured the kernel park under a MONITORX label; two runs of
/// one binary read 0.99 us and 5.08 us for the same cell depending
/// only on whether that happened before or after the cell ran.
const DOUBTS_BEFORE_GIVING_UP: u32 = 8;

/// Whether re-arming a monitor is still believed to be worth it.
fn monitor_holds() -> bool {
    MONITOR_HOLDS.load(Ordering::Relaxed)
}

/// How long the arms took in the most recent wait that doubted the
/// monitor, in RDTSC cycles.
///
/// The verdict says a wait got nowhere; this says how fast. Arms
/// totalling about four times the instruction pair mean the monitor
/// never armed at all, while hundreds of thousands of cycles mean it
/// armed and something ended it, and those want different answers.
/// Nothing in the verdict distinguishes them.
static LAST_DOUBT_CYCLES: AtomicU64 = AtomicU64::new(0);

/// Record that one wait did not suspend the core, and stop trying
/// once enough separate waits have said so.
fn note_monitor_does_not_hold(span_cycles: u64) {
    LAST_DOUBT_CYCLES.store(span_cycles, Ordering::Relaxed);
    if MONITOR_DOUBTS.fetch_add(1, Ordering::Relaxed) + 1 >= DOUBTS_BEFORE_GIVING_UP {
        MONITOR_HOLDS.store(false, Ordering::Relaxed);
    }
}

/// Waits that have found the monitor not holding, and how long the
/// arms took in the most recent of them.
///
/// For benches and diagnostics. A run that reports the monitor gave
/// up says nothing about why, and the two causes it cannot separate
/// need different fixes.
pub fn monitor_doubts() -> (u32, u64) {
    (
        MONITOR_DOUBTS.load(Ordering::Relaxed),
        LAST_DOUBT_CYCLES.load(Ordering::Relaxed),
    )
}

/// Whether any wait has found this host's monitor not to hold.
///
/// Exposed for the bench and for diagnostics: a MONITORX run whose
/// numbers look like the StdPark ones beside it has usually fallen
/// back, and without this that is indistinguishable from the wait
/// being no faster.
pub fn monitor_wait_held() -> bool {
    monitor_holds()
}

impl WaitStrategy {
    /// Pick the best wait strategy for this host: WAITPKG where the
    /// silicon has it, else MONITORX, else
    /// [`WaitStrategy::StdPark`].
    ///
    /// WAITPKG is preferred where both are present because UMWAIT
    /// takes the deadline this parker wants directly, so that arm
    /// reaches its bound in one instruction.
    pub fn pick() -> Self {
        if crate::cpu_info::has_waitpkg() {
            Self::Waitpkg
        } else if crate::cpu_info::has_monitorx() {
            Self::Monitorx
        } else {
            Self::StdPark
        }
    }
}

/// Per-worker park primitive. One `Parker` per worker thread; the
/// thread that owns it parks via [`Self::park_until`], and any
/// other thread wakes it via [`Self::unpark`].
///
/// On WAITPKG-capable hosts the wait path bypasses the kernel
/// condvar entirely - the producer's `unpark` increments
/// `wake_counter` (one cache-line store), and the parked thread's
/// `UMWAIT` returns as soon as the cache-line transition is observed
/// by the hardware monitor. Sub-100ns wake instead of ~1us syscall.
///
/// On non-WAITPKG hosts the wake_counter increment is still issued
/// (it costs one atomic add) but the wait path falls through to
/// `std::thread::park()` as before.
/// One cache line per parker, because a monitor wait watches the line
/// `wake_counter` sits in and wakes on any store to it. Without this
/// two parkers share a line and each one's unpark fires the other's
/// monitor, and an `Arc`'s refcounts sit immediately before the data
/// so every clone and drop fires it too. Both read as the monitor
/// failing to hold.
#[derive(Debug)]
#[repr(align(64))]
pub struct Parker {
    /// Cached `Thread` handle for cross-thread unpark.
    thread: Thread,
    /// Shutdown signal: set by the arena's drop / explicit
    /// shutdown path. When `true`, [`Self::park_until`] returns
    /// `false` to break the worker loop.
    shutdown: AtomicBool,
    /// How many `thread::yield_now()` rounds to spin before
    /// actually calling `thread::park()`. Picked per tier per
    /// [`crate::sched::SchedTier::spin_rounds`].
    spin_rounds: u32,
    /// Monotonic wake counter. Producers increment on `unpark`;
    /// the WAITPKG path snapshots before park + UMONITOR-watches
    /// the counter's cache line.
    wake_counter: AtomicU64,
    /// Wait strategy chosen at construction time.
    wait_strategy: WaitStrategy,
}

impl Parker {
    /// Construct a Parker owned by the calling thread. Captures
    /// the current `Thread` handle for later cross-thread unpark.
    /// Wait strategy is auto-picked via [`WaitStrategy::pick`].
    pub fn new(spin_rounds: u32) -> Self {
        Self::with_strategy(spin_rounds, WaitStrategy::pick())
    }

    /// Construct a Parker with an explicit wait strategy. Used by
    /// benches + tests that need to A/B against the auto-picked
    /// strategy. Callers must not pass [`WaitStrategy::Waitpkg`] on
    /// a host where [`crate::cpu_info::has_waitpkg`] returns false
    /// (the inline `UMONITOR`/`UMWAIT` opcodes would raise `#UD`).
    pub fn with_strategy(spin_rounds: u32, wait_strategy: WaitStrategy) -> Self {
        Self {
            thread: thread::current(),
            shutdown: AtomicBool::new(false),
            spin_rounds,
            wake_counter: AtomicU64::new(0),
            wait_strategy,
        }
    }

    /// Observable wait strategy. Used by benches + diagnostics to
    /// confirm which path the Parker is on.
    pub fn wait_strategy(&self) -> WaitStrategy {
        self.wait_strategy
    }

    /// Block the calling thread until `is_ready` returns `true`,
    /// shutdown is signalled, or the thread is unparked.
    ///
    /// Returns `true` when `is_ready()` was observed or the thread
    /// was unparked; returns `false` on shutdown.
    ///
    /// Polling sequence:
    /// 1. Loop `spin_rounds` times calling `thread::yield_now()`
    ///    between polls. Cheapest path: a worker about to receive
    ///    work via unpark stays out of the parker.
    /// 2. After the spin floor, `thread::park()` ONCE. If we wake
    ///    via unpark (regardless of predicate state) we return
    ///    `true` and let the caller re-attempt the work search.
    ///    This is important when the caller has out-of-band signals
    ///    (e.g., wake-on-push from a peer) that don't update the
    ///    predicate's observed state - the peer's deque might have
    ///    work but the predicate doesn't see it. Returning on
    ///    unpark hands control back to the caller, which then walks
    ///    the peer stealers in its main loop.
    pub fn park_until<F: FnMut() -> bool>(&self, mut is_ready: F) -> bool {
        // Snapshot wake_counter ahead of the spin floor so the waitpkg
        // path can detect any unpark that fires after this snapshot
        // (whether during the spin floor or during the UMWAIT itself).
        let initial_wake = self.wake_counter.load(Ordering::Acquire);

        for _ in 0..self.spin_rounds {
            if self.shutdown.load(Ordering::Acquire) {
                return false;
            }
            if is_ready() {
                return true;
            }
            thread::yield_now();
        }
        if self.shutdown.load(Ordering::Acquire) {
            return false;
        }
        if is_ready() {
            return true;
        }

        // Dispatch on wait strategy. Either path returns to the
        // caller on wake (real or spurious); the caller's loop
        // re-attempts the work search and re-enters park_until
        // when still empty.
        match self.wait_strategy {
            WaitStrategy::StdPark => {
                thread::park();
            }
            // A monitor wait returns on its deadline as well as on a
            // wake, so the deadline re-reads state and the wait is
            // re-entered. Only a wake, a shutdown or a ready
            // predicate returns to the caller.
            WaitStrategy::Waitpkg | WaitStrategy::Monitorx => loop {
                match self.wait_strategy {
                    WaitStrategy::Monitorx => self.wait_via_monitorx(initial_wake),
                    _ => self.wait_via_waitpkg(initial_wake),
                }
                if self.wake_counter.load(Ordering::Acquire) != initial_wake {
                    break;
                }
                if self.shutdown.load(Ordering::Acquire) {
                    break;
                }
                if is_ready() {
                    break;
                }
            },
        }

        // Final shutdown check before returning so a shutdown
        // unpark surfaces cleanly.
        if self.shutdown.load(Ordering::Acquire) {
            return false;
        }
        true
    }

    /// Wake the parked thread if any. Unconditional: if no thread
    /// is parked, the permit is stored for the next park. This
    /// trades a no-op syscall on the empty case for not having to
    /// track an explicit "is this worker parked" flag.
    ///
    /// Increments [`Self::wake_counter`] before anything else, so the
    /// WAITPKG observer's monitor fires; then calls
    /// `thread::unpark()` so
    /// the [`WaitStrategy::StdPark`] path also wakes. Both are
    /// needed because the Parker is constructed knowing its
    /// strategy but the caller does not need to: this method works
    /// for both strategies uniformly.
    pub fn unpark(&self) {
        // Release-store on wake_counter happens-before the parked
        // observer's Acquire-load post-UMWAIT, so the observer sees
        // any state the producer published prior to unpark.
        self.wake_counter.fetch_add(1, Ordering::Release);
        self.thread.unpark();
    }

    /// Signal shutdown. The parked thread observes this on its
    /// next park return and exits its loop.
    ///
    /// Routes through [`Self::unpark`] so the wake_counter increments
    /// and the std::thread permit fires - the waitpkg observer
    /// returns from UMWAIT on the cache-line transition and then
    /// observes `shutdown == true` on its post-park check.
    pub fn shutdown(&self) {
        self.shutdown.store(true, Ordering::Release);
        self.unpark();
    }

    /// WAITPKG wait path: snapshot wake_counter (caller's), arm
    /// UMONITOR on its cache line, double-check the counter to
    /// catch a wake that landed between caller's snapshot and our
    /// UMONITOR setup, then UMWAIT until the line transitions OR
    /// a TSC deadline fires (~10ms).
    ///
    /// Returning here does not mean wake_counter actually changed -
    /// UMWAIT can return on signals, interrupts, or its hint
    /// expiry. The caller (`park_until`) re-checks `is_ready` and
    /// `shutdown` after wake_via_waitpkg returns and decides
    /// whether to re-park.
    #[cfg(target_arch = "x86_64")]
    fn wait_via_waitpkg(&self, initial_wake: u64) {
        // 10 ms deadline cap so a missed wake (e.g. shutdown raced
        // with a UMONITOR that armed after the shutdown unpark)
        // does not block forever. The deadline TSC is computed
        // assuming a ~2.5 GHz TSC frequency; off-by-2x error is
        // immaterial because the caller re-enters park_until on
        // spurious return anyway.
        const WAIT_DEADLINE_NS: u64 = 10_000_000;
        const TSC_HZ_ESTIMATE: u64 = 2_500_000_000;
        let cycles =
            WAIT_DEADLINE_NS.saturating_mul(TSC_HZ_ESTIMATE / 1_000_000_000);
        // SAFETY: `_rdtsc` is a no-side-effect read of the TSC
        // counter; available on every x86_64 CPU produced this
        // century.
        let now = unsafe { core::arch::x86_64::_rdtsc() };
        let deadline = now.wrapping_add(cycles);
        let lo = deadline as u32;
        let hi = (deadline >> 32) as u32;

        let addr = (&raw const self.wake_counter).cast::<u8>();

        // UMONITOR rax: arm hardware monitor on the cache line
        // containing the wake_counter. Any store to that line wakes
        // UMWAIT, including one to another field of this Parker;
        // `repr(align(64))` on the struct is what keeps a different
        // parker's stores out of it.
        //
        // SAFETY: caller (Parker::new -> WaitStrategy::pick) only
        // installs the Waitpkg strategy when has_waitpkg() returned
        // true, so UMONITOR is not a `#UD`. `addr` is a stable
        // pointer to a live AtomicU64 field of `self`.
        unsafe {
            core::arch::asm!(
                "umonitor rax",
                in("rax") addr,
                options(nostack, preserves_flags),
            );
        }

        // Race window: an unpark that fired between the caller's
        // initial_wake snapshot and the UMONITOR arming would not
        // wake UMWAIT (the monitor was not yet armed). Re-check
        // wake_counter; if it advanced, skip UMWAIT entirely.
        if self.wake_counter.load(Ordering::Acquire) != initial_wake {
            return;
        }

        // UMWAIT ecx, edx:eax with ecx = wake hint:
        //   1 = C0.1 (light wait, fastest wake)
        //   0 = C0.2 (deeper wait, lower power, slower wake)
        // We pick C0.1 for the scheduler's latency-sensitive
        // workload. EDX:EAX carries the absolute TSC deadline.
        //
        // SAFETY: same WAITPKG-available reasoning as the UMONITOR
        // above. UMWAIT modifies CF on return (timeout-vs-wake);
        // we drop preserves_flags accordingly.
        unsafe {
            core::arch::asm!(
                "umwait {hint:e}",
                hint = in(reg) 1u32,
                in("eax") lo,
                in("edx") hi,
                options(nostack),
            );
        }
    }

    /// Non-x86_64 stub. The WAITPKG strategy cannot be installed on
    /// non-x86_64 targets (the CPUID probe in `crate::cpu_info`
    /// returns false), so this branch is unreachable in practice.
    /// Fall through to `thread::park()` as a defensive default.
    #[cfg(not(target_arch = "x86_64"))]
    fn wait_via_waitpkg(&self, _initial_wake: u64) {
        thread::park();
    }

    /// MONITORX wait path: the same protocol
    /// [`Self::wait_via_waitpkg`] runs, over AMD's user-mode pair, and
    /// bounded to the same 10 ms.
    ///
    /// Re-arms toward the deadline instead of asking one instruction
    /// to reach it. MWAITX counts EBX units, and how far a unit
    /// reaches is not in CPUID: half an RDTSC cycle on a Ryzen 9
    /// 7900X, a whole one on a Ryzen 7 2700. A busy host also returns
    /// early by a varying amount. Both cost iterations here and
    /// neither is read.
    ///
    /// Returning does not mean `wake_counter` changed: the budget can
    /// run out and the monitor can fire on an unrelated store to the
    /// watched line. `park_until` re-checks `is_ready` and
    /// `shutdown`.
    #[cfg(target_arch = "x86_64")]
    fn wait_via_monitorx(&self, initial_wake: u64) {
        // Matches the WAITPKG arm's cap and its reasoning: a missed
        // wake must not block forever, and the estimate may be off by
        // a factor without mattering, because a short budget costs a
        // re-park and a long one is cut short by the shutdown and
        // wake checks below.
        const WAIT_DEADLINE_NS: u64 = 10_000_000;
        const TSC_HZ_ESTIMATE: u64 = 2_500_000_000;
        // A monitor that keeps firing on traffic to a neighbouring
        // address would otherwise spin here for the whole budget. The
        // count bounds that case on its own, without assuming any
        // iteration actually waits.
        const MAX_ARMS: u32 = 256;
        // This many arms inside `ARMS_TOO_FAST_CYCLES` means the
        // monitor is not holding, and the wait parks instead. Four
        // honoured budgets take tens of milliseconds; four unheld
        // ones take about four times the instruction pair, measured
        // at 2369 cycles on a 7900X and 1606 on a 2700.
        const ARMS_BEFORE_JUDGING: u32 = 4;
        // Below this, four arms are impossibly fast for a monitor
        // that armed at all. Unheld they cost about four times the
        // pair, 9,500 cycles on a 7900X and 6,400 on a 2700. Armed
        // and cut short by something else they measured 420,791 on a
        // 7900X under load, about 22 microseconds each. This sits
        // roughly eight times clear of both.
        //
        // A million was tried, and is why this figure is measured
        // rather than reasoned: it caught the interrupted case, so a
        // monitor that was working got condemned for being woken by
        // the load the arm exists to survive.
        const ARMS_TOO_FAST_CYCLES: u64 = 50_000;

        // Read before anything else this function does. Whether the
        // monitor holds is a property of the part, so a finding by any
        // parker in this process applies to all of them and is never
        // re-tested; and on a part that has given up, everything below
        // is cost paid on the way to a kernel park. A clock read alone
        // measures 206 ns on this project's Windows host and 973 on
        // its Linux guest.
        if !monitor_holds() {
            thread::park();
            return;
        }

        let budget = WAIT_DEADLINE_NS.saturating_mul(TSC_HZ_ESTIMATE / 1_000_000_000);
        // SAFETY: `_rdtsc` is a no-side-effect read of the TSC
        // counter; available on every x86_64 CPU produced this
        // century.
        let start = unsafe { core::arch::x86_64::_rdtsc() };

        let addr = (&raw const self.wake_counter).cast::<u8>();

        let mut arms = 0u32;
        for _ in 0..MAX_ARMS {
            arms += 1;
            // MONITORX rax: arm the monitor on the line holding
            // wake_counter. ECX carries extensions and EDX hints, both
            // zero, which is the only defined combination.
            //
            // SAFETY: Parker::new -> WaitStrategy::pick only installs
            // this strategy when has_monitorx() returned true, so the
            // opcode is not a `#UD`. `addr` points at a live AtomicU64
            // field of `self`. Encoded as bytes because the mnemonic
            // needs a target feature this crate does not set, and the
            // encoding is fixed.
            // The zeros arrive as operands rather than through `xor`,
            // because `xor` writes flags and this block promises not
            // to. MONITORX itself leaves them alone, so the promise
            // holds only while no instruction here breaks it.
            unsafe {
                core::arch::asm!(
                    ".byte 0x0f, 0x01, 0xfa",
                    in("rax") addr,
                    in("ecx") 0u32,
                    in("edx") 0u32,
                    options(nostack, preserves_flags),
                );
            }

            // An unpark between the caller's snapshot and the arming
            // above would not wake MWAITX, because the monitor was not
            // yet armed. Checked after every arm, not just the first,
            // since each iteration re-opens the window.
            if self.wake_counter.load(Ordering::Acquire) != initial_wake {
                return;
            }

            let spent = unsafe { core::arch::x86_64::_rdtsc() }.wrapping_sub(start);
            let Some(left) = budget.checked_sub(spent) else {
                return;
            };
            // EBX is 32 bits. A remaining budget past that asks for
            // the longest wait the register can express and the next
            // iteration asks for the rest.
            let ask = u32::try_from(left).unwrap_or(u32::MAX);

            // MWAITX: EAX = 0 requests C0, the light state matching
            // the WAITPKG arm's C0.1 hint; ECX bit 1 enables the EBX
            // timer; EBX carries the count.
            //
            // rbx is reserved by LLVM and cannot be an operand, so it
            // is saved and restored inside the block. That is why this
            // block does not claim `nostack`: it pushes, and a red
            // zone below rsp would be live under that promise.
            //
            // SAFETY: same MONITORX-available reasoning as above.
            // MWAITX reports timer-versus-wake exit in CF, so flags
            // are not preserved.
            unsafe {
                core::arch::asm!(
                    "push rbx",
                    "mov ebx, {ask:e}",
                    ".byte 0x0f, 0x01, 0xfb",
                    "pop rbx",
                    ask = in(reg) ask,
                    inout("eax") 0u32 => _,
                    inout("ecx") 2u32 => _,
                );
            }

            if self.wake_counter.load(Ordering::Acquire) != initial_wake {
                return;
            }
            if self.shutdown.load(Ordering::Acquire) {
                return;
            }

            // Arms and clock together, not the length of any single
            // return: an interrupt lengthens some returns, so a rule
            // over consecutive short ones never fires on a host whose
            // monitor is not holding.
            // The arm count gates the clock read, not the other way
            // round: the first arms cannot trip this and a clock read
            // costs 206 ns on the Windows host here and 973 on the
            // Linux guest.
            if arms >= ARMS_BEFORE_JUDGING {
                let span = unsafe { core::arch::x86_64::_rdtsc() }.wrapping_sub(start);
                if span < ARMS_TOO_FAST_CYCLES {
                    note_monitor_does_not_hold(span);
                    thread::park();
                    return;
                }
            }
        }
    }

    /// Non-x86_64 stub, for the same reason as the WAITPKG one: the
    /// CPUID probe cannot report MONITORX off x86_64, so this strategy
    /// is never installed there.
    #[cfg(not(target_arch = "x86_64"))]
    fn wait_via_monitorx(&self, _initial_wake: u64) {
        thread::park();
    }

    /// Test whether shutdown has been signalled. Workers can poll
    /// this between job executions to exit promptly.
    pub fn is_shutdown(&self) -> bool {
        self.shutdown.load(Ordering::Acquire)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::AtomicU32;
    use std::thread;
    use std::time::{Duration, Instant};

    #[test]
    fn park_until_returns_immediately_when_ready() {
        let p = Parker::new(8);
        let t0 = Instant::now();
        let ok = p.park_until(|| true);
        let elapsed = t0.elapsed();
        assert!(ok);
        assert!(elapsed < Duration::from_millis(10),
            "park_until with ready=true must be fast; took {elapsed:?}");
    }

    #[test]
    fn park_until_returns_false_on_shutdown() {
        let p = Arc::new(Parker::new(8));
        let p_signal = Arc::clone(&p);
        let signal = thread::spawn(move || {
            thread::sleep(Duration::from_millis(50));
            p_signal.shutdown();
        });
        let ok = p.park_until(|| false);
        signal.join().unwrap();
        assert!(!ok, "park_until must return false after shutdown");
    }

    #[test]
    fn park_until_wakes_on_unpark_from_other_thread() {
        // Owner thread parks. Helper thread unparks after 50 ms.
        // The owner observes `is_ready` becoming true and returns.
        let ready = Arc::new(AtomicU32::new(0));
        let ready_clone = Arc::clone(&ready);

        let (tx, rx) = std::sync::mpsc::channel::<Arc<Parker>>();

        let owner = thread::spawn(move || {
            let p = Arc::new(Parker::new(8));
            tx.send(Arc::clone(&p)).unwrap();
            let t0 = Instant::now();
            let ok = p.park_until(|| ready.load(Ordering::Acquire) == 1);
            (ok, t0.elapsed())
        });

        let p_owner = rx.recv().expect("owner must send its parker");
        thread::sleep(Duration::from_millis(50));
        ready_clone.store(1, Ordering::Release);
        p_owner.unpark();

        let (ok, elapsed) = owner.join().unwrap();
        assert!(ok, "park_until must return true after ready becomes true");
        // Should wake within ~100 ms.
        assert!(elapsed < Duration::from_millis(500),
            "park_until took too long: {elapsed:?}");
    }

    #[test]
    fn park_until_spin_floor_eight_rounds_no_park() {
        // With spin_rounds=8 and is_ready becoming true on round 3,
        // park_until should return without ever calling park().
        // We can't observe park directly, but we can verify the
        // sequence completes quickly.
        let p = Parker::new(8);
        let mut polls = 0u32;
        let ok = p.park_until(|| {
            polls += 1;
            polls >= 3
        });
        assert!(ok);
        assert_eq!(polls, 3);
    }

    #[test]
    fn park_until_zero_spin_floor_goes_straight_to_park() {
        // With spin_rounds=0, park_until skips the yield loop. We
        // verify by setting ready=true synchronously - the call
        // returns at the loop's first iteration.
        let p = Parker::new(0);
        let ok = p.park_until(|| true);
        assert!(ok);
    }

    #[test]
    fn unpark_before_park_is_observable_via_permit() {
        // std::thread::park's permit semantics: unpark before park
        // stores a permit; next park returns immediately. We test
        // this through park_until: helper unparks before owner
        // calls park_until. The owner's first park sees the
        // permit and returns; the subsequent re-check observes
        // ready=true.
        let p = Arc::new(Parker::new(0)); // 0 spin so we go to park fast
        let ready = Arc::new(AtomicU32::new(0));
        let p_clone = Arc::clone(&p);
        let ready_clone = Arc::clone(&ready);

        // Pre-store an unpark permit on the owner thread before
        // it calls park_until. We do this by having the owner be
        // the main thread, and a helper that unparks then sets
        // ready.
        // (Easier: just stage the unpark via a delayed thread
        //  before the owner's park_until call.)

        let signal = thread::spawn(move || {
            // Caller's thread::current() is captured inside p_clone
            // when the main thread instantiates Parker. The unpark
            // targets the main thread (the parker's owner).
            ready_clone.store(1, Ordering::Release);
            p_clone.unpark();
        });
        signal.join().unwrap();

        let ok = p.park_until(|| ready.load(Ordering::Acquire) == 1);
        assert!(ok);
    }

    #[test]
    fn is_shutdown_reflects_shutdown_call() {
        let p = Parker::new(8);
        assert!(!p.is_shutdown());
        p.shutdown();
        assert!(p.is_shutdown());
    }

    #[test]
    fn wait_strategy_pick_matches_cpuid() {
        // The order is what is asserted, not any one host's answer:
        // WAITPKG wins where present, MONITORX takes the hosts that
        // have only it, and the kernel park is what is left.
        let want = if crate::cpu_info::has_waitpkg() {
            WaitStrategy::Waitpkg
        } else if crate::cpu_info::has_monitorx() {
            WaitStrategy::Monitorx
        } else {
            WaitStrategy::StdPark
        };
        assert_eq!(WaitStrategy::pick(), want);
    }

    #[test]
    fn no_two_parkers_can_share_a_cache_line() {
        // A monitor wait watches the line wake_counter sits in, so a
        // neighbour sharing it turns that neighbour's every unpark
        // into a wake here. Pinned by alignment rather than by size,
        // because adding a field must not be able to undo it.
        assert_eq!(std::mem::align_of::<Parker>(), 64);
        assert_eq!(std::mem::size_of::<Parker>() % 64, 0);
    }

    #[test]
    fn the_parker_never_picks_a_wait_its_host_cannot_execute() {
        // The strategy names an instruction, so picking one the CPUID
        // probe did not confirm is a `#UD` on the idle path rather
        // than a wrong answer. Read from the strategy toward the
        // probe, which is the direction that catches a `pick` whose
        // arms have been reordered out from under the detections.
        match WaitStrategy::pick() {
            WaitStrategy::Waitpkg => assert!(crate::cpu_info::has_waitpkg()),
            WaitStrategy::Monitorx => assert!(crate::cpu_info::has_monitorx()),
            WaitStrategy::StdPark => {
                assert!(!crate::cpu_info::has_waitpkg());
                assert!(!crate::cpu_info::has_monitorx());
            }
        }
    }

    #[test]
    fn the_hosts_own_strategy_wakes_on_unpark() {
        // The other wake tests name a strategy, so on a host whose
        // pick() differs from all of them the arm that actually runs
        // in production goes unexercised. This one parks on whatever
        // this host chose, which is the only test here that executes
        // the MONITORX path on a MONITORX host.
        // Constructed on the thread that parks, and handed back over
        // a channel. A Parker captures `thread::current()` at
        // construction and unparks that handle, so one built here and
        // parked over there sends the permit to this thread and the
        // parked one never wakes. The StdPark arm is where that
        // shows: a monitor wait returns on the cache-line store and
        // never reads the handle at all.
        let ready = Arc::new(AtomicU32::new(0));
        let woke = Arc::new(AtomicU32::new(0));
        let (tx, rx) = std::sync::mpsc::channel::<Arc<Parker>>();

        let ready_thread = Arc::clone(&ready);
        let woke_thread = Arc::clone(&woke);
        let owner = thread::spawn(move || {
            let p = Arc::new(Parker::new(0));
            tx.send(Arc::clone(&p)).expect("send parker");
            let ok = p.park_until(|| ready_thread.load(Ordering::Acquire) == 1);
            woke_thread.store(1, Ordering::Release);
            ok
        });

        let p = rx.recv().expect("recv parker");
        thread::sleep(Duration::from_millis(20));
        ready.store(1, Ordering::Release);
        p.unpark();

        assert!(
            owner.join().expect("owner thread joins"),
            "park_until must report a wake rather than a shutdown"
        );
        assert_eq!(woke.load(Ordering::Acquire), 1);
    }

    #[test]
    fn with_strategy_stdpark_round_trips_like_default() {
        // Explicitly construct a StdPark Parker; same semantics as
        // the original implementation.
        let p = Parker::with_strategy(8, WaitStrategy::StdPark);
        assert_eq!(p.wait_strategy(), WaitStrategy::StdPark);
        let ok = p.park_until(|| true);
        assert!(ok, "StdPark park_until must return true on ready=true");
    }

    #[test]
    fn unpark_increments_wake_counter() {
        // Verify the WAITPKG observer mechanism is wired: every
        // unpark MUST bump wake_counter so the WAITPKG path's
        // double-check after UMONITOR catches the wake.
        let p = Parker::new(8);
        let before = p.wake_counter.load(Ordering::Acquire);
        p.unpark();
        let after = p.wake_counter.load(Ordering::Acquire);
        assert_eq!(after, before + 1, "unpark must increment wake_counter");
    }

    #[test]
    fn shutdown_increments_wake_counter() {
        // shutdown routes through unpark so the WAITPKG observer
        // also wakes on shutdown (not just the std::thread permit).
        let p = Parker::new(8);
        let before = p.wake_counter.load(Ordering::Acquire);
        p.shutdown();
        let after = p.wake_counter.load(Ordering::Acquire);
        assert_eq!(after, before + 1, "shutdown must increment wake_counter via unpark");
        assert!(p.is_shutdown());
    }

    #[test]
    fn waitpkg_strategy_wakes_on_unpark_when_available() {
        // Skip on hosts without WAITPKG (the UMONITOR/UMWAIT opcodes
        // would #UD). Per-architecture cpuid check; on Zen+ R7 2700
        // this returns false and the test is a no-op.
        if !crate::cpu_info::has_waitpkg() {
            eprintln!(
                "skip waitpkg_strategy_wakes_on_unpark_when_available: \
                 host has no WAITPKG (cpuid leaf 7 ECX bit 5 = 0)"
            );
            return;
        }
        // WAITPKG-capable host: park with Waitpkg strategy + unpark
        // from helper thread. Owner thread must observe the wake
        // within the 10ms deadline.
        let ready = Arc::new(AtomicU32::new(0));
        let ready_clone = Arc::clone(&ready);
        let (tx, rx) = std::sync::mpsc::channel::<Arc<Parker>>();
        let owner = thread::spawn(move || {
            let p = Arc::new(Parker::with_strategy(8, WaitStrategy::Waitpkg));
            tx.send(Arc::clone(&p)).unwrap();
            let t0 = Instant::now();
            let ok = p.park_until(|| ready.load(Ordering::Acquire) == 1);
            (ok, t0.elapsed())
        });
        let p_owner = rx.recv().expect("owner must send its parker");
        thread::sleep(Duration::from_millis(20));
        ready_clone.store(1, Ordering::Release);
        p_owner.unpark();
        let (ok, elapsed) = owner.join().unwrap();
        assert!(ok, "Waitpkg park_until must return true on unpark");
        // Cap should be well under 100ms; the 10ms UMWAIT deadline
        // bounds the worst case to ~10ms even if UMWAIT misses the
        // wake.
        assert!(elapsed < Duration::from_millis(100),
            "Waitpkg park_until took {elapsed:?}, expected < 100ms");
    }

    #[test]
    fn shutdown_unparks_so_blocked_thread_exits() {
        // Parker MUST be constructed inside the thread that will
        // park on it, because `Parker::new` captures
        // `thread::current()` for the unpark target. A Parker
        // built in main and parked-on by a spawned thread would
        // unpark main, not the spawned thread, and deadlock.
        let (tx, rx) = std::sync::mpsc::channel::<Arc<Parker>>();
        let owner = thread::spawn(move || {
            let p = Arc::new(Parker::new(8));
            tx.send(Arc::clone(&p)).unwrap();
            p.park_until(|| false)
        });
        let p_owner = rx.recv().expect("owner must send its parker");
        thread::sleep(Duration::from_millis(50));
        p_owner.shutdown();
        let ok = owner.join().unwrap();
        assert!(!ok, "shutdown must surface as park_until -> false");
    }
}
