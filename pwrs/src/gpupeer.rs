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

use pwrs::prelude::*;

use flynnel::gpu_peer::wave::Frontier as CrateFrontier;
use flynnel::gpu_peer::wave::plan::{Imbalance, PlanInputs, plan as plan_wave};
use flynnel::gpu_peer::watchdog::{self, DriverModel as CrateDriverModel};

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
