//! The worker pool a plan dispatches into, and the dials that change
//! how it waits.
//!
//! Every dial here is process-wide. A script that changes one changes
//! it for everything in the session that dispatches through this
//! module's library, so each Set cmdlet reads the value back and
//! writes what is now in force rather than leaving a caller to assume
//! the write landed.

use pwrs::prelude::*;

/// The pool as it stands.
#[psclass(name = "Flynnel.Pool")]
#[derive(Clone, Default)]
pub struct Pool {
    /// Per-node arenas. One on a host with a single node.
    pub node_count: u64,
    /// Whether there is only one, which is the shape most desktops
    /// have and the one with no cross-node steal at all.
    pub is_single_node: bool,
    /// Every worker thread, primaries and SMT extensions together.
    pub total_workers: u64,
    /// Primary workers, one per physical core.
    pub primary_workers: u64,
    /// SMT extension workers, which park until a dispatch asks for
    /// them.
    pub smt_extension_workers: u64,
    /// Workers in the node the calling thread is on.
    pub local_workers: u64,
    /// How much of the pool's pushing went through the burst path
    /// rather than the auto-flush one, between zero and one. It is
    /// 0.5 when nothing has pushed yet, which is not a measurement of
    /// a half-burst workload.
    pub burst_ratio: f32,
    /// Whether anything has pushed at all, which is what tells a
    /// starting 0.5 from a measured one.
    pub has_pushed: bool,
}

pub(crate) fn pool_snapshot() -> Pool {
    let arena = flynnel::sched::arena::global_local_arena();
    let pushed: u64 = arena
        .iter_worker_stats()
        .map(|s| {
            s.single_pushes.load(std::sync::atomic::Ordering::Relaxed)
                + s.burst_pushes.load(std::sync::atomic::Ordering::Relaxed)
        })
        .sum();
    Pool {
        node_count: arena.node_count() as u64,
        is_single_node: arena.is_single_numa(),
        total_workers: arena.total_workers() as u64,
        primary_workers: arena.primary_workers() as u64,
        smt_extension_workers: arena.smt_extension_workers() as u64,
        local_workers: arena.local_worker_count() as u64,
        burst_ratio: arena.global_burst_ratio(),
        has_pushed: pushed > 0,
    }
}

/// Starts the worker pool if it is not already running and writes what
/// it is: the per-node arenas, the primary workers and the SMT
/// extensions.
///
/// Safe to call twice. The pool is process-wide and starts on the
/// first dispatch anyway; this is how a script starts it deliberately,
/// before timing something, rather than paying for the start inside
/// the first measurement.
///
/// # Examples
///
/// `Start-FlynnelPool`
///
/// `(Start-FlynnelPool).PrimaryWorkers`
#[cmdlet(
    verb = "Start",
    noun = "FlynnelPool",
    alias = "Start-FlyPool",
    output = ["Flynnel.Pool"]
)]
#[derive(Default)]
pub struct StartFlynnelPool {}

impl Cmdlet for StartFlynnelPool {
    fn process(&mut self, ps: &Pipeline<'_>) -> PsResult<()> {
        ps.write(pool_snapshot())
    }
}

/// Reads the worker pool as it stands.
///
/// Starts the pool if nothing has dispatched yet, because there is no
/// honest answer about a pool that does not exist; Start-FlynnelPool
/// is the same call under a name that says so.
///
/// # Examples
///
/// `Get-FlynnelPool`
///
/// `Get-FlynnelPool | Format-List`
#[cmdlet(
    verb = "Get",
    noun = "FlynnelPool",
    alias = "Get-FlyPool",
    output = ["Flynnel.Pool"]
)]
#[derive(Default)]
pub struct GetFlynnelPool {}

impl Cmdlet for GetFlynnelPool {
    fn process(&mut self, ps: &Pipeline<'_>) -> PsResult<()> {
        ps.write(pool_snapshot())
    }
}

/// One worker's counters.
#[psclass(name = "Flynnel.WorkerStat")]
#[derive(Clone, Default)]
pub struct WorkerStat {
    /// Index of the row in the pool's statistics table.
    pub index: u64,
    /// Whether this row is a worker. The table runs past the pool's
    /// workers into the external slots a foreign thread pushes
    /// through, and those are not workers however idle they read.
    pub is_worker: bool,
    /// Jobs taken from its own deque.
    pub local_pops: u64,
    /// Jobs it stole from a peer.
    pub peer_steal_hits: u64,
    /// Probe rounds that found nothing to steal. High against the
    /// hits means the pool is idle, not contended.
    pub peer_steal_misses: u64,
    /// Times a peer stole from it, which is the pressure signal the
    /// lazy bisect reads.
    pub times_stolen_from: u64,
    /// Pushes through the auto-flush path, the join right-half shape.
    pub single_pushes: u64,
    /// Pushes through the burst path, the fan-out shape.
    pub burst_pushes: u64,
    /// Pushes a full deque refused, where the caller ran the job
    /// inline instead of waiting for a thief.
    pub push_refusals: u64,
    /// Burst against single for this worker, between zero and one,
    /// and 0.5 when it has pushed nothing.
    pub burst_ratio: f32,
}

/// Reads every worker's counters: what it popped, what it stole, what
/// was stolen from it, and how it pushed.
///
/// Every worker comes back in one call rather than one call per
/// worker, which on a 24-thread host is the difference between one
/// crossing and twenty-four.
///
/// # Examples
///
/// `Get-FlynnelWorker`
///
/// `Get-FlynnelWorker | Sort-Object TimesStolenFrom -Descending | Select-Object -First 5`
#[cmdlet(
    verb = "Get",
    noun = "FlynnelWorker",
    alias = "Get-FlyWorker",
    output = ["Flynnel.WorkerStat"]
)]
#[derive(Default)]
pub struct GetFlynnelWorker {
    /// Also write the external slots a foreign thread pushes through.
    /// They sit past the workers in the same table and are not
    /// workers; IsWorker tells them apart.
    #[param]
    pub include_external_slot: bool,
}

/// Every worker's counters, and optionally the external slots behind
/// them.
///
/// The stats table is longer than the pool: the entries past the
/// worker count belong to the external slots a foreign thread pushes
/// through. They are real rows and they are not workers, and emitting
/// them unmarked gives a caller a set of permanently idle workers that
/// do not exist. They are named rather than dropped, and left out
/// unless asked for.
///
/// Built here rather than inside the cmdlet so the Flynnel drive's
/// `pool\workers` level answers the same rows, and builds the whole
/// level in one pass rather than one call into Rust per child.
pub(crate) fn worker_rows(include_external_slot: bool) -> Vec<WorkerStat> {
    use std::sync::atomic::Ordering::Relaxed;
    let arena = flynnel::sched::arena::global_local_arena();
    let workers = arena.total_workers();
    let mut out = Vec::new();
    for (index, s) in arena.iter_worker_stats().enumerate() {
        let is_worker = index < workers;
        if !is_worker && !include_external_slot {
            continue;
        }
        out.push(WorkerStat {
            index: index as u64,
            is_worker,
            local_pops: s.local_pops.load(Relaxed),
            peer_steal_hits: s.peer_steal_hits.load(Relaxed),
            peer_steal_misses: s.peer_steal_misses.load(Relaxed),
            times_stolen_from: s.times_stolen_from.load(Relaxed),
            single_pushes: s.single_pushes.load(Relaxed),
            burst_pushes: s.burst_pushes.load(Relaxed),
            push_refusals: s.push_refusals.load(Relaxed),
            burst_ratio: s.burst_ratio(),
        });
    }
    out
}

impl Cmdlet for GetFlynnelWorker {
    fn process(&mut self, ps: &Pipeline<'_>) -> PsResult<()> {
        for row in worker_rows(self.include_external_slot) {
            ps.write(row)?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------
// Spinning
// ---------------------------------------------------------------------

/// How a worker waits before it parks.
#[psclass(name = "Flynnel.SpinState")]
#[derive(Clone, Default)]
pub struct SpinState {
    /// Yield rounds a worker runs before parking on its condvar.
    pub window_rounds: u32,
    /// Yields the pool has made since the counter was last reset.
    pub total_idle_yields: u64,
    /// Whether the adaptive controller is deciding the window. False
    /// means WindowRounds is pinned where someone set it.
    pub adaptive: bool,
    /// Times the controller has reached a decision, counted from
    /// process start and never reset. Zero while Adaptive is true says
    /// the workload never parked often enough to gather the evidence,
    /// which is a different state from a controller that decided and
    /// left the window where it found it.
    pub adapt_decisions: u64,
}

pub(crate) fn spin_snapshot() -> SpinState {
    SpinState {
        window_rounds: flynnel::spin_window(),
        total_idle_yields: flynnel::total_idle_yields(),
        adaptive: flynnel::sched::spin_adaptive(),
        adapt_decisions: flynnel::spin_adapt_decisions(),
    }
}

/// Reads how long a worker spins before it parks, and how many idle
/// yields the pool has made since the counter was last reset.
///
/// # Examples
///
/// `Get-FlynnelSpinWindow`
///
/// `(Get-FlynnelSpinWindow).TotalIdleYields`
#[cmdlet(
    verb = "Get",
    noun = "FlynnelSpinWindow",
    alias = "Get-FlySpinWindow",
    output = ["Flynnel.SpinState"]
)]
#[derive(Default)]
pub struct GetFlynnelSpinWindow {}

impl Cmdlet for GetFlynnelSpinWindow {
    fn process(&mut self, ps: &Pipeline<'_>) -> PsResult<()> {
        ps.write(spin_snapshot())
    }
}

/// Sets the spin window, in yield rounds, and writes back what is now
/// in force.
///
/// Pinning the window takes the adaptive controller out of the
/// decision. That matters when the controller is what you were
/// measuring: pinned, both arms of such a comparison are the same arm.
/// Set-FlynnelSpinAdaptive turns the controller back on.
///
/// This is process-wide.
///
/// # Examples
///
/// `Set-FlynnelSpinWindow -Rounds 8`
///
/// `Set-FlynnelSpinWindow -Rounds 500`
#[cmdlet(
    verb = "Set",
    noun = "FlynnelSpinWindow",
    alias = "Set-FlySpinWindow",
    output = ["Flynnel.SpinState"]
)]
#[derive(Default)]
pub struct SetFlynnelSpinWindow {
    /// Yield rounds before a worker parks.
    #[param(mandatory, position = 0)]
    pub rounds: u32,
}

impl Cmdlet for SetFlynnelSpinWindow {
    fn process(&mut self, ps: &Pipeline<'_>) -> PsResult<()> {
        flynnel::set_spin_window(self.rounds);
        let now = spin_snapshot();
        if now.window_rounds != self.rounds {
            pwrs::warning!(
                ps,
                "asked for {} rounds and the window reads {}; the controller clamps it to \
                 the range it can use",
                self.rounds,
                now.window_rounds
            )?;
        }
        ps.write(now)
    }
}

/// Turns the adaptive spin controller on or off and writes the window
/// now in force.
///
/// With it on the window shrinks under a bursty-idle workload and
/// grows back; with it off the window stays where it was last set.
///
/// This is process-wide.
///
/// # Examples
///
/// `Set-FlynnelSpinAdaptive -On`
///
/// `Set-FlynnelSpinAdaptive -Off`
#[cmdlet(
    verb = "Set",
    noun = "FlynnelSpinAdaptive",
    alias = "Set-FlySpinAdaptive",
    default_parameter_set = "On",
    output = ["Flynnel.SpinState"]
)]
#[derive(Default)]
pub struct SetFlynnelSpinAdaptive {
    /// Let the controller move the window.
    #[param(set = "On")]
    pub on: bool,
    /// Leave the window where it was last set.
    #[param(set = "Off")]
    pub off: bool,
}

impl Cmdlet for SetFlynnelSpinAdaptive {
    fn process(&mut self, ps: &Pipeline<'_>) -> PsResult<()> {
        flynnel::set_spin_adaptive(!self.off);
        ps.write(spin_snapshot())
    }
}

/// Zeroes the idle-yield counter, so a following measurement counts
/// only its own yields.
///
/// # Examples
///
/// `Reset-FlynnelSpinStats`
///
/// `Reset-FlynnelSpinStats; Invoke-FlynnelMap -InputObject $a -Operation Square; Get-FlynnelSpinWindow`
#[cmdlet(
    verb = "Reset",
    noun = "FlynnelSpinStats",
    alias = "Reset-FlySpinStats",
    output = ["Flynnel.SpinState"]
)]
#[derive(Default)]
pub struct ResetFlynnelSpinStats {}

impl Cmdlet for ResetFlynnelSpinStats {
    fn process(&mut self, ps: &Pipeline<'_>) -> PsResult<()> {
        flynnel::reset_spin_stats();
        ps.write(spin_snapshot())
    }
}

// ---------------------------------------------------------------------
// The split multiplier and what the observer sees
// ---------------------------------------------------------------------

/// The leaf-splitting multiplier and the window it was derived from.
#[psclass(name = "Flynnel.SplitState")]
#[derive(Clone, Default)]
pub struct SplitState {
    /// The multiplier in force, between one and eight.
    pub multiplier: u32,
    /// Leaves the observer has timed in the current window.
    pub leaf_count: u64,
    /// Items those leaves covered.
    pub leaf_items: u64,
    /// Mean nanoseconds a leaf took, null until enough leaves have
    /// reported to mean anything.
    pub mean_leaf_ns: Option<u64>,
    /// Nanoseconds an item took, null on the same condition.
    pub per_item_ns: Option<u64>,
    /// Spread of the per-item cost, in parts per thousand of the mean
    /// squared. Null until the window has enough leaves.
    pub per_item_cv2_per_mille: Option<u64>,
    /// Spread of the whole-leaf cost, on the same scale and the same
    /// condition.
    pub leaf_cv2_per_mille: Option<u64>,
}

pub(crate) fn split_snapshot() -> SplitState {
    let stats = flynnel::sched::split_observer::snapshot_leaf_stats();
    SplitState {
        multiplier: flynnel::sched::split_observer::split_multiplier(),
        leaf_count: stats.count,
        leaf_items: stats.items,
        mean_leaf_ns: flynnel::sched::split_observer::observed_mean_leaf_ns(),
        per_item_ns: flynnel::sched::split_observer::observed_per_item_ns(stats),
        per_item_cv2_per_mille: flynnel::sched::split_observer::per_item_cv_squared_per_mille(
            stats,
        ),
        leaf_cv2_per_mille: flynnel::sched::split_observer::leaf_cv_squared_per_mille(stats),
    }
}

/// Reads the leaf-splitting multiplier and the window the observer
/// derived it from: how many leaves it timed, what they covered, and
/// the spread of their cost.
///
/// A figure the window is too sparse to support comes back as nothing
/// rather than as zero, because a zero spread and an unmeasured one
/// are different facts and only one of them is about the workload.
///
/// # Examples
///
/// `Get-FlynnelSplitMultiplier`
///
/// `(Get-FlynnelSplitMultiplier).PerItemCv2PerMille`
#[cmdlet(
    verb = "Get",
    noun = "FlynnelSplitMultiplier",
    alias = "Get-FlySplitMultiplier",
    output = ["Flynnel.SplitState"]
)]
#[derive(Default)]
pub struct GetFlynnelSplitMultiplier {}

impl Cmdlet for GetFlynnelSplitMultiplier {
    fn process(&mut self, ps: &Pipeline<'_>) -> PsResult<()> {
        ps.write(split_snapshot())
    }
}

/// Sets the leaf-splitting multiplier and writes back what is now in
/// force.
///
/// The value is clamped to between one and eight. With the observer
/// running, it overwrites this on its next window, so a pinned
/// multiplier only holds while the observer is not started.
///
/// This is process-wide.
///
/// # Examples
///
/// `Set-FlynnelSplitMultiplier -Value 4`
#[cmdlet(
    verb = "Set",
    noun = "FlynnelSplitMultiplier",
    alias = "Set-FlySplitMultiplier",
    output = ["Flynnel.SplitState"]
)]
#[derive(Default)]
pub struct SetFlynnelSplitMultiplier {
    /// The multiplier, clamped to one through eight.
    #[param(mandatory, position = 0)]
    pub value: u32,
}

impl Cmdlet for SetFlynnelSplitMultiplier {
    fn process(&mut self, ps: &Pipeline<'_>) -> PsResult<()> {
        flynnel::sched::split_observer::set_split_multiplier(self.value);
        let now = split_snapshot();
        if now.multiplier != self.value {
            pwrs::warning!(
                ps,
                "asked for {} and the multiplier reads {}; it is clamped to one through eight",
                self.value,
                now.multiplier
            )?;
        }
        ps.write(now)
    }
}

/// Zeroes the observer's leaf window, so the next figures describe
/// only what follows.
///
/// # Examples
///
/// `Reset-FlynnelSplitStats`
#[cmdlet(
    verb = "Reset",
    noun = "FlynnelSplitStats",
    alias = "Reset-FlySplitStats",
    output = ["Flynnel.SplitState"]
)]
#[derive(Default)]
pub struct ResetFlynnelSplitStats {}

impl Cmdlet for ResetFlynnelSplitStats {
    fn process(&mut self, ps: &Pipeline<'_>) -> PsResult<()> {
        flynnel::sched::split_observer::reset_leaf_stats();
        ps.write(split_snapshot())
    }
}

/// Starts the background observer that retunes the split multiplier
/// from the pool's measured steal rate.
///
/// Safe to call twice: the observer starts once per process, and a
/// second call writes the state without starting another.
///
/// The observer runs on the IO pool and resubmits itself each window,
/// so without an IO pool it cannot start. The crate's own start is a
/// silent no-op in that case; this warns instead, because a multiplier
/// nothing is retuning reads exactly like one that is.
///
/// # Examples
///
/// `Start-FlynnelSplitObserver`
#[cmdlet(
    verb = "Start",
    noun = "FlynnelSplitObserver",
    alias = "Start-FlySplitObserver",
    output = ["Flynnel.SplitState"]
)]
#[derive(Default)]
pub struct StartFlynnelSplitObserver {}

impl Cmdlet for StartFlynnelSplitObserver {
    fn process(&mut self, ps: &Pipeline<'_>) -> PsResult<()> {
        let has_io_pool = flynnel::sched::global_io_pool().is_some();
        flynnel::sched::split_observer::spawn_observer();
        if !has_io_pool {
            pwrs::warning!(
                ps,
                "there is no IO pool in this process, so the observer did not start and the \
                 split multiplier will stay where it is; set FLYNNEL_SCHED_SMT_AS_IO=1 before \
                 the pool starts, or build one with New-FlynnelIoPool"
            )?;
        }
        ps.write(split_snapshot())
    }
}

// ---------------------------------------------------------------------
// The IO pool
// ---------------------------------------------------------------------

/// A pool of threads for blocking work, kept off the scheduler's own
/// workers so a blocking task cannot occupy one.
#[psclass(name = "Flynnel.IoPool", mode = proxy)]
pub struct IoPool {
    /// Threads in this pool.
    pub worker_count: u64,
    #[psfield(skip)]
    inner: std::sync::Arc<flynnel::sched::io_pool::IoPool>,
}

/// The operations of a `Flynnel.IoPool`.
#[psmethods]
impl IoPool {
    /// A pool of WorkerCount threads for blocking work.
    ///
    /// A script reaches this as `[Flynnel.IoPool]::new(4)`.
    /// New-FlynnelIoPool builds its pool here as well, so the two routes
    /// make the same object, and nothing submits to either:
    /// New-FlynnelIoPool's help says why.
    pub fn new(worker_count: u32) -> PsResult<IoPool> {
        if worker_count == 0 {
            return Err(PsError::new(
                ErrorCategory::InvalidArgument,
                "FlynnelArgument",
                "WorkerCount must be at least one",
            )
            .terminating());
        }
        let inner = flynnel::sched::io_pool::IoPool::new(worker_count as usize);
        Ok(IoPool {
            worker_count: inner.worker_count() as u64,
            inner,
        })
    }

    /// Threads in this pool.
    pub fn workers(&self) -> PsResult<u64> {
        Ok(self.inner.worker_count() as u64)
    }
}

/// Makes a pool of threads for blocking work.
///
/// # Nothing submits to the pool this returns
///
/// One cmdlet does route to a pool: `Measure-FlynnelFileHash
/// -UseIoPool` reads its files off one instead of off the arena. It
/// uses the process-wide pool from `global_io_pool()`, which is not
/// the object this cmdlet returns, so a pool made here starts threads
/// that sit idle however the kernels are called.
///
/// The switch cannot be pointed at that object either. The Rust pool
/// it holds is a skipped field, which is what stops a value copy
/// carrying the handle, and a class with one cannot be taken as a
/// typed cmdlet parameter.
///
/// The process-wide pool is built from the environment before the
/// first dispatch: `global_io_pool()` is a `OnceLock`, so a session
/// that has already dispatched cannot gain one afterwards.
///
/// What this cmdlet is good for, then, is starting a pool and reading
/// the width it actually got. It is said plainly because the
/// alternative is a script that creates a pool, sees the worker count
/// it asked for, passes `-UseIoPool`, and concludes the two are
/// connected.
///
/// A script block could not be submitted in any case: it runs only on
/// the thread that owns the pipeline.
///
/// # Examples
///
/// `$io = New-FlynnelIoPool -WorkerCount 4`
///
/// `(New-FlynnelIoPool -WorkerCount 4).WorkerCount`
#[cmdlet(
    verb = "New",
    noun = "FlynnelIoPool",
    alias = "New-FlyIoPool",
    output = ["Flynnel.IoPool"]
)]
#[derive(Default)]
pub struct NewFlynnelIoPool {
    /// How many threads.
    #[param(mandatory, position = 0)]
    pub worker_count: u32,
}

impl Cmdlet for NewFlynnelIoPool {
    fn process(&mut self, ps: &Pipeline<'_>) -> PsResult<()> {
        ps.write(IoPool::new(self.worker_count)?)
    }
}

/// Reads the process's own IO pool.
///
/// A process where it was never started has none, and this writes a
/// warning and nothing rather than an empty pool, because an empty
/// pool and an absent one are different states.
///
/// # Examples
///
/// `Get-FlynnelIoPool`
#[cmdlet(
    verb = "Get",
    noun = "FlynnelIoPool",
    alias = "Get-FlyIoPool",
    output = ["Flynnel.IoPool"]
)]
#[derive(Default)]
pub struct GetFlynnelIoPool {}

impl Cmdlet for GetFlynnelIoPool {
    fn process(&mut self, ps: &Pipeline<'_>) -> PsResult<()> {
        let Some(pool) = flynnel::sched::io_pool::global_io_pool() else {
            pwrs::warning!(
                ps,
                "this process has no global IO pool: nothing has started one, which is not \
                 the same as one with no workers"
            )?;
            return Ok(());
        };
        ps.write(IoPool {
            worker_count: pool.worker_count() as u64,
            inner: std::sync::Arc::clone(pool),
        })
    }
}
