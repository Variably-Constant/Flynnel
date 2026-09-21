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

use flynnel::gpu_peer::watchdog::{self, DriverModel as CrateDriverModel};

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
