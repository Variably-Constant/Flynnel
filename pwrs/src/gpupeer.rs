//! The GPU as a peer of the scheduler, and what bounds a piece of its
//! work.
//!
//! # What is bound here so far
//!
//! The watchdog reading, which is the one thing in this family a caller
//! needs before a peer exists and can get on a host that has no card at
//! all. Everything else in the family - the peer itself, its lanes and
//! tickets, waves, resident and wide ops, the VRAM pool, L2
//! persistence, the mirror buffers and the linear algebra - waits on a
//! live peer and lands in later slices.
//!
//! # Why the watchdog comes first
//!
//! On Windows, timeout detection and recovery resets a device whose
//! work runs too long, and the peer's poller is a bounded-quantum
//! kernel precisely so it stays under that bound. A caller sizing any
//! piece of device work is sizing it against this number, and on a host
//! whose TDR level is zero there is no number to size against. That is
//! a fact about the machine, not about a peer, so it is readable
//! without one.

use std::sync::Mutex;

use pwrs::prelude::*;

use flynnel::gpu_peer::wave::Frontier as CrateFrontier;
use flynnel::gpu_peer::wave::plan::{Imbalance, PlanInputs, plan as plan_wave};
use flynnel::gpu_peer::watchdog::{self, DriverModel as CrateDriverModel};
use flynnel::gpu_peer::{GpuPeer, GpuPeerConfig, GpuPeerError};

use crate::host::arg_err;

/// How a device's driver presents it, which decides whether the
/// watchdog covers it.
#[psenum(name = "Flynnel.DriverModel")]
#[derive(Clone, Copy, Default)]
pub enum DriverModelKind {
    /// The model could not be read. Treated as covered, because only a
    /// Tcc reading rules the watchdog out and guessing the other way
    /// ends in a device reset.
    #[default]
    Unknown,
    /// Windows Display Driver Model, which the watchdog covers.
    Wddm,
    /// Tesla Compute Cluster, which the watchdog does not cover.
    Tcc,
    /// Microsoft Compute Driver Model, which the watchdog covers.
    Mcdm,
}

impl From<CrateDriverModel> for DriverModelKind {
    fn from(m: CrateDriverModel) -> Self {
        match m {
            CrateDriverModel::Wddm => DriverModelKind::Wddm,
            CrateDriverModel::Tcc => DriverModelKind::Tcc,
            CrateDriverModel::Mcdm => DriverModelKind::Mcdm,
        }
    }
}

/// The watchdog that applies to one device, and what was read to decide
/// it.
#[psclass(name = "Flynnel.PeerWatchdog")]
#[derive(Clone, Default)]
pub struct PeerWatchdog {
    /// The CUDA device ordinal this row describes.
    pub ordinal: u32,
    /// Whether a watchdog applies at all. False means no bound: on a
    /// Tcc device, at TDR level zero, or on a platform with no
    /// equivalent.
    pub applies: bool,
    /// How long one piece of device work may run before the device is
    /// reset. Null when nothing applies, which is not the same as zero.
    pub delay_ns: Option<u64>,
    /// The same bound in seconds, null on the same terms.
    pub delay_seconds: Option<f64>,
    /// The driver model as the device reported it.
    pub driver_model: DriverModelKind,
    /// Whether that model was read or guessed. False means the read
    /// failed and the row's model column is Unknown; the device is
    /// still treated as covered.
    pub driver_model_known: bool,
    /// Why the read failed, when it did.
    pub driver_model_problem: Option<String>,
    /// Everything that was read and what each read returned, including
    /// the reads that failed. This is the column to quote when a bound
    /// looks wrong, because it names which of the two reads decided it.
    pub basis: String,
}

/// Reads which watchdog bounds a piece of work on one GPU, and what was
/// read to decide it.
///
/// On Windows this is the timeout detection and recovery setting: the
/// driver model says whether the device is covered, and the registry
/// says for how long. Both are read from the host rather than assumed.
/// No other platform has an equivalent, and there the answer is that
/// nothing applies.
///
/// Applies false is a real answer with three causes, and Basis says
/// which: a Tcc device is outside the watchdog, TDR level zero turns
/// detection off, and a platform without TDR has no watchdog to read.
///
/// Where a read fails the device is treated as covered and the
/// documented two second delay is taken. That direction is deliberate:
/// a watchdog that is present and treated as absent ends in a device
/// reset, while one treated as present only shortens slices. Basis
/// names every failed read so a bound taken this way is never mistaken
/// for one that was measured.
///
/// This needs no peer and no CUDA context. It does load NVML once per
/// device to read the driver model, which costs tens of milliseconds;
/// the reading is cached for the life of the process, because neither
/// the hardware nor the driver configuration can change under a running
/// process.
///
/// # Examples
///
/// `Get-FlynnelPeerWatchdog`
///
/// `Get-FlynnelPeerWatchdog -Ordinal 1 | Select-Object Applies, DelaySeconds, Basis`
#[cmdlet(
    verb = "Get",
    noun = "FlynnelPeerWatchdog",
    alias = "Get-FlyPeerWatchdog",
    output = ["Flynnel.PeerWatchdog"]
)]
#[derive(Default)]
pub struct GetFlynnelPeerWatchdog {
    /// Which CUDA device to read. The first one when unset.
    #[param(position = 0)]
    pub ordinal: Option<u32>,
}

impl Cmdlet for GetFlynnelPeerWatchdog {
    fn process(&mut self, ps: &Pipeline<'_>) -> PsResult<()> {
        let ordinal = self.ordinal.unwrap_or(0);
        let (model, state) = watchdog::detect_with_model(ordinal as usize);
        let (kind, known, problem) = match model {
            Ok(m) => (DriverModelKind::from(m), true, None),
            Err(err) => (DriverModelKind::Unknown, false, Some(err)),
        };
        ps.write(PeerWatchdog {
            ordinal,
            applies: state.applies(),
            delay_ns: state.delay_ns,
            delay_seconds: state.delay_ns.map(|ns| ns as f64 / 1_000_000_000.0),
            driver_model: kind,
            driver_model_known: known,
            driver_model_problem: problem,
            basis: state.basis,
        })
    }
}

// ---------------------------------------------------------------------
// The wave planner
// ---------------------------------------------------------------------

/// How a wave keeps its frontier.
#[psenum(name = "Flynnel.Frontier")]
#[derive(Clone, Copy, Default)]
pub enum FrontierKind {
    /// One frontier across the team, fixed by a barrier every
    /// generation. Load stays even and each generation waits for its
    /// slowest block.
    #[default]
    Global,
    /// One frontier per block, met and dealt out evenly every so many
    /// generations.
    Partition,
}

/// The frontier a wave should keep, and what the model prices it at.
#[psclass(name = "Flynnel.WavePlan")]
#[derive(Clone, Default)]
pub struct WavePlan {
    /// How the frontier is kept.
    pub frontier: FrontierKind,
    /// Generations between rebalances. Null on a global frontier,
    /// which has none, and also on a partition the model says should
    /// never rebalance, which is a different answer; RebalancesAtAll
    /// separates them.
    pub rebalance_every: Option<u32>,
    /// Whether the chosen plan rebalances at all.
    pub rebalances_at_all: bool,
    /// What the model expects the chosen frontier to cost per
    /// generation.
    pub cost_per_generation_ns: f64,
    /// What it expects a global frontier to cost per generation, which
    /// is the barrier. Carried beside the chosen cost so the margin is
    /// readable without a second call.
    pub global_cost_ns: f64,
    /// How much cheaper the chosen frontier is than a global one. Zero
    /// when the plan is global.
    pub saving_ns: f64,
    /// Blocks in the team the plan was chosen for.
    pub width: u32,
    /// Whether an imbalance was supplied. False means the plan is
    /// global because nothing has been observed yet, not because a
    /// global frontier won a comparison.
    pub imbalance_supplied: bool,
}

/// Chooses how a wave should keep its frontier, from what the device
/// costs and what an earlier run observed.
///
/// Two costs trade against each other. A global frontier pays a barrier
/// every generation and idles no block. A partition pays a rebalance
/// once every so many generations and idles blocks in between as their
/// frontiers diverge; one that never rebalances pays nothing and idles
/// until the divergence reaches the team width.
///
/// With no imbalance supplied the answer is a global frontier, and
/// ImbalanceSupplied false says that is because nothing has been
/// observed rather than because a comparison chose it. A wave run that
/// way records the imbalance the next plan needs.
///
/// This is the cost model alone. It launches nothing, needs no peer and
/// no device, and answers the same on a host with no card: the costs
/// are arguments. Measure them on the machine that will run the wave
/// and pass them here.
///
/// # Examples
///
/// `Get-FlynnelWavePlan -Width 32 -BarrierNs 4000 -GenerationNs 90000`
///
/// `Get-FlynnelWavePlan -Width 32 -BarrierNs 4000 -GenerationNs 90000
///     -ImbalancePerMille 1800 -ImbalanceOverGenerations 8`
#[cmdlet(
    verb = "Get",
    noun = "FlynnelWavePlan",
    alias = "Get-FlyWavePlan",
    output = ["Flynnel.WavePlan"]
)]
#[derive(Default)]
pub struct GetFlynnelWavePlan {
    /// Blocks in the team.
    #[param(mandatory, position = 0)]
    pub width: u32,
    /// What one cross-block barrier costs at this width.
    #[param(mandatory, position = 1)]
    pub barrier_ns: f64,
    /// What one generation of the program takes.
    #[param(mandatory, position = 2)]
    pub generation_ns: f64,
    /// A rebalance's fixed cost, apart from the ids it moves.
    #[param]
    pub rebalance_fixed_ns: Option<f64>,
    /// What moving one pending id through staging costs at a
    /// rebalance.
    #[param]
    pub copy_ns_per_id: Option<f64>,
    /// Ids typically pending when a rebalance happens.
    #[param]
    pub pending_ids: Option<f64>,
    /// The largest block's frontier over the mean on an earlier run, in
    /// thousandths, where 1000 is balanced. Without it the plan is
    /// global.
    #[param]
    pub imbalance_per_mille: Option<u32>,
    /// Generations of independent growth that imbalance accumulated
    /// over. One for a global frontier's per-generation share.
    #[param]
    pub imbalance_over_generations: Option<u32>,
}

impl Cmdlet for GetFlynnelWavePlan {
    fn process(&mut self, ps: &Pipeline<'_>) -> PsResult<()> {
        if self.width == 0 {
            return Err(arg_err("Width must be above zero").terminating());
        }
        for (name, v) in [
            ("BarrierNs", self.barrier_ns),
            ("GenerationNs", self.generation_ns),
        ] {
            if !v.is_finite() || v < 0.0 {
                return Err(
                    arg_err(format!("{name} must be a finite cost at or above zero"))
                        .terminating(),
                );
            }
        }
        // An imbalance is a pair and half of one is not a reading. A
        // per-mille with no generation count would silently become one
        // generation, which prices a divergence as growing far faster
        // than it was observed to.
        let imbalance = match (self.imbalance_per_mille, self.imbalance_over_generations) {
            (Some(per_mille), Some(over_generations)) => {
                if over_generations == 0 {
                    return Err(arg_err(
                        "ImbalanceOverGenerations must be at least one; an imbalance over no \
                         generations is not a reading",
                    )
                    .terminating());
                }
                Some(Imbalance {
                    per_mille,
                    over_generations,
                })
            }
            (None, None) => None,
            _ => {
                return Err(arg_err(
                    "an imbalance needs both ImbalancePerMille and ImbalanceOverGenerations; \
                     a ratio without the generations it grew over does not price a divergence",
                )
                .terminating());
            }
        };

        let inputs = PlanInputs {
            width: self.width,
            barrier_ns: self.barrier_ns,
            rebalance_fixed_ns: self.rebalance_fixed_ns.unwrap_or(0.0),
            copy_ns_per_id: self.copy_ns_per_id.unwrap_or(0.0),
            pending_ids: self.pending_ids.unwrap_or(0.0),
            generation_ns: self.generation_ns,
            imbalance,
        };
        let chosen = plan_wave(&inputs);
        let (kind, every) = match chosen.frontier {
            CrateFrontier::Global => (FrontierKind::Global, None),
            CrateFrontier::Partition { rebalance_every } => (
                FrontierKind::Partition,
                rebalance_every.map(|n| n.get()),
            ),
        };
        ps.write(WavePlan {
            frontier: kind,
            rebalance_every: every,
            rebalances_at_all: every.is_some(),
            cost_per_generation_ns: chosen.cost_per_generation_ns,
            global_cost_ns: chosen.global_cost_ns,
            saving_ns: (chosen.global_cost_ns - chosen.cost_per_generation_ns).max(0.0),
            width: self.width,
            imbalance_supplied: imbalance.is_some(),
        })
    }
}

// ---------------------------------------------------------------------
// The peer's lifecycle
// ---------------------------------------------------------------------

/// The one peer this process holds.
///
/// One rather than a table, because a peer owns a device context, a
/// mapped region and a resident kernel, and two of them on one device
/// contend for all three. `Get-FlynnelGpuPeer` means the live one, and
/// there is only ever a live one.
static PEER: Mutex<Option<GpuPeer>> = Mutex::new(None);

/// What the running peer was asked for, kept beside it so a clamped
/// team width can be reported as a clamp rather than as the only
/// number there is.
static REQUESTED_BLOCKS: Mutex<u32> = Mutex::new(0);

/// The peer slot.
fn peer_slot() -> std::sync::MutexGuard<'static, Option<GpuPeer>> {
    match PEER.lock() {
        Ok(guard) => guard,
        // Taken over rather than propagated, and that is a decision
        // about this lock rather than an error going unread. The panic
        // that poisoned it happened beside the peer, not inside it, so
        // the peer is intact; refusing every later call would strand a
        // device context with nothing able to release it.
        Err(poisoned) => poisoned.into_inner(),
    }
}

/// The requested team width, on the same terms as [`peer_slot`].
fn requested_blocks() -> std::sync::MutexGuard<'static, u32> {
    match REQUESTED_BLOCKS.lock() {
        Ok(guard) => guard,
        // A poisoned lock still holds the number some earlier call
        // wrote, and that number is what this reports.
        Err(poisoned) => poisoned.into_inner(),
    }
}

/// A peer error as an error record, with the category that matches what
/// went wrong rather than one category for the family.
fn peer_err(err: GpuPeerError) -> PsError {
    let category = match &err {
        GpuPeerError::NoDevice(_) | GpuPeerError::Unavailable(_) => ErrorCategory::DeviceError,
        GpuPeerError::Driver(_) => ErrorCategory::NotSpecified,
        GpuPeerError::Io(_) => ErrorCategory::WriteError,
        GpuPeerError::PayloadTooLarge { .. } | GpuPeerError::ReapOutOfOrder { .. } => {
            ErrorCategory::InvalidArgument
        }
        GpuPeerError::Timeout => ErrorCategory::OperationTimeout,
    };
    PsError::new(category, "FlynnelGpuPeer", err.to_string()).terminating()
}

/// What a peer is built from.
#[psclass(name = "Flynnel.GpuPeerConfig")]
#[derive(Clone)]
pub struct PeerConfig {
    /// Backing file for the shared region. Empty means a per-process
    /// file in the temp directory, removed when the peer goes. A fixed
    /// path is what makes the region attachable by another process.
    pub region_path: String,
    /// Lanes, each served by its own consumer.
    pub lanes: u32,
    /// Slot size in bytes, including the sixteen byte descriptor.
    pub slot_bytes: u32,
    /// Ring depth per lane.
    pub slots_per_lane: u32,
    /// How long one resident quantum runs before it exits. This is the
    /// number the watchdog bounds, so Get-FlynnelPeerWatchdog is what
    /// says whether a value is safe on this host.
    pub quantum_ns: u64,
    /// How long rank zero waits for the rest of its block team before
    /// retiring the slot. Read only when BlocksPerLane is above one.
    pub barrier_deadline_ns: u64,
    /// Idle time after which a resident quantum parks.
    pub idle_exit_ns: u64,
    /// Which CUDA device.
    pub device_ordinal: u32,
    /// Bytes of device memory per resident block.
    pub vram_block_bytes: u32,
    /// Resident blocks. Zero disables the pool.
    pub vram_blocks: u32,
    /// Blocks serving each lane. One keeps a lane on a single
    /// multiprocessor; above one a lane is worked by a team and a
    /// single doorbell spreads across the device. The peer clamps this
    /// to the device's multiprocessor count at init and reports the
    /// size it actually ran.
    pub blocks_per_lane: u32,
    /// Blocks serving each lane individually, one entry per lane, for
    /// a peer whose lanes run teams of different widths. Empty runs
    /// every lane at BlocksPerLane.
    pub lane_teams: Vec<u32>,
}

impl Default for PeerConfig {
    fn default() -> Self {
        // Taken from the crate's own default rather than restated, so
        // the two cannot drift.
        let d = GpuPeerConfig::default();
        Self {
            region_path: String::new(),
            lanes: d.lanes,
            slot_bytes: d.slot_bytes,
            slots_per_lane: d.slots_per_lane,
            quantum_ns: d.quantum_ns,
            barrier_deadline_ns: d.barrier_deadline_ns,
            idle_exit_ns: d.idle_exit_ns,
            device_ordinal: d.device_ordinal as u32,
            vram_block_bytes: d.vram_block_bytes,
            vram_blocks: d.vram_blocks,
            blocks_per_lane: d.blocks_per_lane,
            lane_teams: d.lane_teams.clone(),
        }
    }
}

impl PeerConfig {
    fn to_crate(&self) -> GpuPeerConfig {
        GpuPeerConfig {
            region_path: if self.region_path.is_empty() {
                None
            } else {
                Some(std::path::PathBuf::from(&self.region_path))
            },
            lanes: self.lanes,
            slot_bytes: self.slot_bytes,
            slots_per_lane: self.slots_per_lane,
            quantum_ns: self.quantum_ns,
            barrier_deadline_ns: self.barrier_deadline_ns,
            idle_exit_ns: self.idle_exit_ns,
            device_ordinal: self.device_ordinal as usize,
            vram_block_bytes: self.vram_block_bytes,
            vram_blocks: self.vram_blocks,
            blocks_per_lane: self.blocks_per_lane,
            lane_teams: self.lane_teams.clone(),
            // Deliberately not exposed. A user op is CUDA C source
            // compiled by NVRTC at init, and handing a driver a source
            // that came from a script is the boundary this module
            // declines for bind_accel_kernel too. There is also
            // exactly one flynnel_user_op hook per module, so a second
            // source string would not be a second op.
            user_ops_cuda: None,
            user_ops_nvrtc_options: Vec::new(),
        }
    }
}

/// Builds the settings a peer is created from, starting at the crate's
/// own defaults.
///
/// Every parameter is optional and an omitted one keeps the default, so
/// a config differing in one field is one parameter rather than twelve.
///
/// There is no parameter for user-op source. A user op is CUDA C
/// compiled by NVRTC when the peer starts, and this module does not
/// hand a driver a source that came from a script; the same reasoning
/// keeps Register cmdlets off the accelerator-op family.
///
/// # Examples
///
/// `New-FlynnelGpuPeerConfig`
///
/// `New-FlynnelGpuPeerConfig -Lanes 8 -VramBlocks 64 -VramBlockBytes 1048576`
#[cmdlet(
    verb = "New",
    noun = "FlynnelGpuPeerConfig",
    alias = "New-FlyGpuPeerConfig",
    output = ["Flynnel.GpuPeerConfig"]
)]
#[derive(Default)]
pub struct NewFlynnelGpuPeerConfig {
    /// A fixed path for the region, so another process can attach it.
    #[param]
    pub region_path: Option<String>,
    /// Lanes in the region, each served by its own consumer.
    #[param]
    pub lanes: Option<u32>,
    /// Slot size in bytes.
    #[param]
    pub slot_bytes: Option<u32>,
    /// Ring depth per lane.
    #[param]
    pub slots_per_lane: Option<u32>,
    /// How long one resident quantum runs before it exits. The
    /// watchdog is what bounds this, so Get-FlynnelPeerWatchdog says
    /// whether a value is safe here.
    #[param]
    pub quantum_ns: Option<u64>,
    /// How long rank zero waits for the rest of its block team before
    /// retiring the slot. Read only when BlocksPerLane is above one.
    #[param]
    pub barrier_deadline_ns: Option<u64>,
    /// Idle time before a quantum parks.
    #[param]
    pub idle_exit_ns: Option<u64>,
    /// CUDA device ordinal.
    #[param]
    pub device_ordinal: Option<u32>,
    /// Bytes per resident block.
    #[param]
    pub vram_block_bytes: Option<u32>,
    /// Resident blocks; zero disables the pool.
    #[param]
    pub vram_blocks: Option<u32>,
    /// Blocks serving each lane. The peer clamps this to the device's
    /// multiprocessor count and reports the width it ran.
    #[param]
    pub blocks_per_lane: Option<u32>,
    /// Blocks per lane individually, one entry per lane.
    #[param]
    pub lane_teams: Option<Vec<u32>>,
}

impl Cmdlet for NewFlynnelGpuPeerConfig {
    fn process(&mut self, ps: &Pipeline<'_>) -> PsResult<()> {
        let mut c = PeerConfig::default();
        if let Some(v) = self.region_path.take() {
            c.region_path = v;
        }
        if let Some(v) = self.lanes {
            c.lanes = v;
        }
        if let Some(v) = self.slot_bytes {
            c.slot_bytes = v;
        }
        if let Some(v) = self.slots_per_lane {
            c.slots_per_lane = v;
        }
        if let Some(v) = self.quantum_ns {
            c.quantum_ns = v;
        }
        if let Some(v) = self.barrier_deadline_ns {
            c.barrier_deadline_ns = v;
        }
        if let Some(v) = self.idle_exit_ns {
            c.idle_exit_ns = v;
        }
        if let Some(v) = self.device_ordinal {
            c.device_ordinal = v;
        }
        if let Some(v) = self.vram_block_bytes {
            c.vram_block_bytes = v;
        }
        if let Some(v) = self.vram_blocks {
            c.vram_blocks = v;
        }
        if let Some(v) = self.blocks_per_lane {
            c.blocks_per_lane = v;
        }
        if let Some(v) = self.lane_teams.take() {
            c.lane_teams = v;
        }
        if c.lanes == 0 {
            return Err(arg_err("Lanes must be above zero").terminating());
        }
        // Refused here rather than at init, because init's refusal
        // arrives after a device context has been made and torn down.
        if !c.lane_teams.is_empty() && c.lane_teams.len() != c.lanes as usize {
            return Err(arg_err(format!(
                "LaneTeams has {} entries and Lanes is {}; a per-lane team width needs one \
                 entry for every lane",
                c.lane_teams.len(),
                c.lanes
            ))
            .terminating());
        }
        ps.write(c)
    }
}

/// The live peer, and what it measured about this host when it started.
#[psclass(name = "Flynnel.GpuPeer")]
#[derive(Clone, Default)]
pub struct PeerRow {
    /// Whether a peer is running in this process. False means every
    /// other column is empty rather than describing one.
    pub running: bool,
    /// Lanes the region was built with.
    pub lanes: u32,
    /// Slot size in bytes.
    pub slot_bytes: u32,
    /// Ring depth per lane.
    pub slots_per_lane: u32,
    /// Blocks actually serving a lane, after the peer clamped the
    /// requested width to the device's multiprocessor count. Below
    /// BlocksPerLaneRequested means the clamp fired.
    pub team_size: u32,
    /// Blocks per lane the config asked for.
    pub blocks_per_lane_requested: u32,
    /// Whether the peer narrowed the team. A team wider than the
    /// device loses ranks at its barrier, so the clamp is a correction
    /// rather than a preference.
    pub team_narrowed: bool,
    /// Whether starting the peer replaced a CUDA context the caller
    /// had built. The peer works on the device primary context, so a
    /// consumer holding its own finds its launches on the primary one
    /// after this.
    pub displaced_foreign_context: bool,
    /// Free blocks in the resident pool.
    pub pool_free_blocks: u32,
    /// Blocks in the resident pool. Zero means the pool is disabled,
    /// which is a setting rather than an exhausted pool.
    pub pool_total_blocks: u32,
    /// Doorbell round trip, minimum observed at init.
    pub rtt_min_ns: u64,
    /// Doorbell round trip, median at init.
    pub rtt_median_ns: u64,
    /// Doorbell round trip, 99th percentile at init.
    pub rtt_p99_ns: u64,
    /// One-way visibility bound.
    pub one_way_ns: u64,
    /// Cross-device clock alignment error.
    pub clock_err_ns: u64,
    /// The timed-lock margin the self-test actually validated.
    pub delta_ns: u64,
    /// Kernel launch and synchronize baseline, median at init.
    pub launch_ns: u64,
    /// Whether the doorbell handshake completed and was measured.
    pub doorbell_ok: bool,
    /// Whether the timed-lock self-test passed with no violations.
    pub timed_lock_ok: bool,
    /// Whether cross-device compare-and-swap conserved claims, which
    /// only a coherent link gives.
    pub sys_atomics_ok: bool,
    /// Contended rounds the CPU side saw in the granting self-test, of
    /// 150. The evidence behind TimedLockOk: a pass with no contention
    /// tested nothing.
    pub lock_cpu_contended: u32,
    /// Contended rounds the GPU side saw, of 150.
    pub lock_gpu_contended: u32,
}

/// The row for a live peer, or the empty row when there is none.
fn peer_row(slot: &Option<GpuPeer>, requested: u32) -> PeerRow {
    let Some(peer) = slot else {
        return PeerRow::default();
    };
    let g = peer.geometry();
    let c = peer.calibration();
    let (free, total) = peer.pool_stats();
    let team = peer.team_size();
    PeerRow {
        running: true,
        lanes: g.lanes,
        slot_bytes: g.slot_bytes,
        slots_per_lane: g.slots_per_lane,
        team_size: team,
        blocks_per_lane_requested: requested,
        team_narrowed: requested > 0 && team < requested,
        displaced_foreign_context: peer.displaced_foreign_context(),
        pool_free_blocks: free as u32,
        pool_total_blocks: total,
        rtt_min_ns: c.rtt_min_ns,
        rtt_median_ns: c.rtt_median_ns,
        rtt_p99_ns: c.rtt_p99_ns,
        one_way_ns: c.one_way_ns,
        clock_err_ns: c.clock_err_ns,
        delta_ns: c.delta_ns,
        launch_ns: c.launch_ns,
        doorbell_ok: c.doorbell_ok,
        timed_lock_ok: c.timed_lock_ok,
        sys_atomics_ok: c.sys_atomics_ok,
        lock_cpu_contended: c.lock_cpu_contended,
        lock_gpu_contended: c.lock_gpu_contended,
    }
}

/// Reads the peer this process is running, or says there is none.
///
/// Running false is a row, never an absent one: a script asking whether
/// a peer exists has to get an answer it can branch on.
///
/// Every timing column was measured when the peer started and is not
/// re-measured by this call. They describe the host as it was at init:
/// the doorbell round trip, the cross-device clock error, the
/// timed-lock margin the self-test validated, and the launch baseline.
///
/// # Examples
///
/// `Get-FlynnelGpuPeer`
///
/// `if ((Get-FlynnelGpuPeer).Running) { Submit-FlynnelGpuOp ... }`
#[cmdlet(
    verb = "Get",
    noun = "FlynnelGpuPeer",
    alias = "Get-FlyGpuPeer",
    output = ["Flynnel.GpuPeer"]
)]
#[derive(Default)]
pub struct GetFlynnelGpuPeer {}

impl Cmdlet for GetFlynnelGpuPeer {
    fn process(&mut self, ps: &Pipeline<'_>) -> PsResult<()> {
        let requested = *requested_blocks();
        let slot = peer_slot();
        ps.write(peer_row(&slot, requested))
    }
}

/// Starts the GPU peer: maps the shared region, registers it with the
/// driver, launches the resident poller and calibrates this host.
///
/// One peer per process. A peer owns a device context, a mapped region
/// and a resident kernel, and a second one on the same device would
/// contend for all three, so a call made while one is running is
/// refused and says so. Remove-FlynnelGpuPeer tears the running one
/// down first.
///
/// A host with no loadable CUDA driver is refused before anything is
/// created. That order matters: the check is made against the driver's
/// own loadability rather than by attempting the call and catching
/// what comes back, because the failing shape this family has already
/// had once was a CUDA entry point reached before the driver was known
/// to be there.
///
/// Starting a peer makes the device primary context current on the
/// calling thread. A consumer that built its own context finds its
/// later launches on the primary one instead, and the row says so
/// through DisplacedForeignContext rather than leaving it to be
/// discovered at a launch far from the cause.
///
/// The team width is clamped to the device's multiprocessor count,
/// because a team wider than the device loses ranks at its barrier.
/// TeamSize is what ran and TeamNarrowed says whether the clamp fired.
///
/// # Examples
///
/// `New-FlynnelGpuPeer`
///
/// `New-FlynnelGpuPeer -Config (New-FlynnelGpuPeerConfig -Lanes 8)`
#[cmdlet(
    verb = "New",
    noun = "FlynnelGpuPeer",
    alias = "New-FlyGpuPeer",
    output = ["Flynnel.GpuPeer"]
)]
#[derive(Default)]
pub struct NewFlynnelGpuPeer {
    /// The settings to start with. The defaults when unset.
    #[param(position = 0)]
    pub config: Option<PeerConfig>,
}

impl Cmdlet for NewFlynnelGpuPeer {
    fn process(&mut self, ps: &Pipeline<'_>) -> PsResult<()> {
        let mut slot = peer_slot();
        if slot.is_some() {
            return Err(PsError::new(
                ErrorCategory::ResourceExists,
                "FlynnelGpuPeer",
                "a GPU peer is already running in this process; Remove-FlynnelGpuPeer tears \
                 it down first. One peer owns the device context, the mapped region and the \
                 resident kernel, and a second would contend for all three",
            )
            .terminating());
        }
        if !flynnel::backend::detect::cuda_available() {
            return Err(PsError::new(
                ErrorCategory::DeviceError,
                "FlynnelGpuPeer",
                "no loadable CUDA driver on this host, so there is no device for a peer to \
                 join. Get-FlynnelBackend reports what this host has",
            )
            .terminating());
        }
        let config = match self.config.take() {
            Some(c) => c,
            None => PeerConfig::default(),
        };
        let requested = config.blocks_per_lane;
        let peer = GpuPeer::init(config.to_crate()).map_err(peer_err)?;
        *requested_blocks() = requested;
        *slot = Some(peer);
        ps.write(peer_row(&slot, requested))
    }
}

/// Tears the running peer down: stops the poller, unregisters the
/// region and releases the device memory, now rather than whenever the
/// process ends.
///
/// This is the deterministic teardown, and it is a cmdlet rather than a
/// Dispose on a handle because the peer is one per process. A handle
/// released by the garbage collector would free a device context at a
/// moment nothing chose.
///
/// Removing when nothing is running is not an error. It writes false
/// and warns, so a cleanup block does not have to ask first.
///
/// # Examples
///
/// `Remove-FlynnelGpuPeer`
#[cmdlet(
    verb = "Remove",
    noun = "FlynnelGpuPeer",
    alias = "Remove-FlyGpuPeer",
    output = ["System.Boolean"]
)]
#[derive(Default)]
pub struct RemoveFlynnelGpuPeer {}

impl Cmdlet for RemoveFlynnelGpuPeer {
    fn process(&mut self, ps: &Pipeline<'_>) -> PsResult<()> {
        let mut slot = peer_slot();
        let had = slot.is_some();
        // Dropped inside the lock, so a second Remove cannot find the
        // slot empty while this teardown is still running.
        *slot = None;
        *requested_blocks() = 0;
        if !had {
            pwrs::warning!(ps, "no GPU peer was running, so nothing was torn down")?;
        }
        ps.write(had)
    }
}
