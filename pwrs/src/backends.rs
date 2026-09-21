//! What devices this host has, what each can do, and which
//! accelerator operations are registered against them.
//!
//! # An absent device is a row, never a missing row
//!
//! Every backend the crate names gets a row, on every host. A row
//! saying `Registered` false and `Available` false is the answer for a
//! machine without that device, and it is a different answer from no
//! row at all - which is what a caller would get if the family only
//! listed what it found, and which reads identically to a capability
//! the module forgot to bind. This module ships with every one of the
//! crate's features on, so the code for each backend is compiled in
//! everywhere and its absence is always a runtime fact.
//!
//! # Three questions that are not the same question
//!
//! `Registered` is whether something has put an implementation in the
//! process's registry. `Available` is whether the host's probe finds
//! the runtime - a library that loads, a device node that opens.
//! `Detected` is whether the crate's own sweep put it in the list it
//! would pick from. They come apart in both directions: a CUDA
//! runtime can load on a machine with no card, and a backend a
//! consumer registered by hand is registered without any probe having
//! passed. Reporting one of the three as though it were the others is
//! how a script ends up dispatching at a device that is not there.
//!
//! # What a script cannot do here
//!
//! It cannot register a backend. `register_backend` takes an
//! `Arc<dyn DispatchBackend>` and a script has no Rust type to build
//! one from, so there is no `Register-FlynnelBackend` and its absence
//! is deliberate rather than pending. The same holds for
//! `register_accel_op`, which takes the CPU implementation as a
//! closure. What a script can do is read what the process already
//! holds, which is what this family is.

use pwrs::prelude::*;

use flynnel::backend::detect;
use flynnel::backend::registry::{
    backend_by_id, backends as registered_backends, ensure_default_registered,
};
use flynnel::backend::{Backend as CrateBackend, BackendCapabilities};

/// Which kind of device a backend drives.
///
/// The crate's own `Backend` carries a device id in most of its
/// variants, so it is a data-carrying enum: the kind comes across as
/// this, and the id rides beside it on the row. A fieldless enum is
/// what a shell can compare and complete against.
#[psenum(name = "Flynnel.BackendKind")]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BackendKind {
    /// The host's own cores. Always registered and always available.
    #[default]
    Cpu,
    /// An NVIDIA device through CUDA.
    Cuda,
    /// An AMD device through ROCm.
    Rocm,
    /// An Apple device through Metal.
    Metal,
    /// A tensor processing unit.
    Tpu,
    /// An Apple Neural Engine.
    Ane,
    /// A WebAssembly runtime, which is a backend in the dispatch sense
    /// without being a device.
    Wasm,
    /// A peer process reached through a memory-mapped deque.
    SharedMemoryWorker,
    /// A backend a consumer registered under an id the taxonomy does
    /// not name.
    Custom,
}

impl BackendKind {
    /// Every kind that can be enumerated without knowing an id.
    ///
    /// Custom is absent on purpose: its variant carries a caller's own
    /// u32 and there is no set of them to walk, so a Custom backend
    /// appears in a listing only when it has been registered.
    pub(crate) const ENUMERABLE: [BackendKind; 8] = [
        BackendKind::Cpu,
        BackendKind::Cuda,
        BackendKind::Rocm,
        BackendKind::Metal,
        BackendKind::Tpu,
        BackendKind::Ane,
        BackendKind::Wasm,
        BackendKind::SharedMemoryWorker,
    ];

    /// The crate's backend for this kind at a device id.
    pub(crate) fn to_crate(self, device_id: u32) -> CrateBackend {
        match self {
            BackendKind::Cpu => CrateBackend::Cpu,
            BackendKind::Cuda => CrateBackend::Cuda { device_id },
            BackendKind::Rocm => CrateBackend::Rocm { device_id },
            BackendKind::Metal => CrateBackend::Metal { device_id },
            BackendKind::Tpu => CrateBackend::Tpu { device_id },
            BackendKind::Ane => CrateBackend::Ane,
            BackendKind::Wasm => CrateBackend::Wasm { device_id },
            BackendKind::SharedMemoryWorker => CrateBackend::SharedMemoryWorker {
                backend_id: device_id,
            },
            BackendKind::Custom => CrateBackend::Custom(device_id),
        }
    }

    /// Whether this host's probe finds the runtime behind the kind.
    ///
    /// The probes are the crate's own and each answers for the kind
    /// rather than for a particular device, which is why this takes no
    /// id. The CPU is not probed: it is the one backend that cannot be
    /// absent.
    fn available(self) -> bool {
        match self {
            BackendKind::Cpu => true,
            BackendKind::Cuda => detect::cuda_available(),
            BackendKind::Rocm => detect::rocm_available(),
            BackendKind::Metal => detect::metal_available(),
            BackendKind::Tpu => detect::tpu_available(),
            BackendKind::Ane => detect::ane_available(),
            BackendKind::Wasm => detect::wasm_available(),
            BackendKind::SharedMemoryWorker => detect::shared_memory_worker_available(),
            // A consumer's own backend has no probe the crate could
            // run, so the only honest answer is whether it is there.
            BackendKind::Custom => false,
        }
    }

    /// What the crate's probe actually looks at, so a false reading
    /// can be argued with rather than only believed.
    fn probe(self) -> &'static str {
        match self {
            BackendKind::Cpu => "not probed: the CPU backend cannot be absent",
            BackendKind::Cuda => "loads the CUDA driver library and asks it for a device count",
            BackendKind::Rocm => "loads the ROCm runtime library",
            BackendKind::Metal => "compiled for a target where Metal exists",
            BackendKind::Tpu => "looks for a TPU device node or the runtime library",
            BackendKind::Ane => "compiled for an Apple target with a Neural Engine",
            BackendKind::Wasm => "the wasm runtime is compiled into this build",
            BackendKind::SharedMemoryWorker => "the shared-memory worker support is compiled in",
            BackendKind::Custom => "no probe: a consumer's backend is known only by registration",
        }
    }
}

impl From<CrateBackend> for BackendKind {
    fn from(b: CrateBackend) -> Self {
        match b {
            CrateBackend::Cpu => BackendKind::Cpu,
            CrateBackend::Cuda { .. } => BackendKind::Cuda,
            CrateBackend::Rocm { .. } => BackendKind::Rocm,
            CrateBackend::Metal { .. } => BackendKind::Metal,
            CrateBackend::Tpu { .. } => BackendKind::Tpu,
            CrateBackend::Ane => BackendKind::Ane,
            CrateBackend::Wasm { .. } => BackendKind::Wasm,
            CrateBackend::SharedMemoryWorker { .. } => BackendKind::SharedMemoryWorker,
            CrateBackend::Custom(_) => BackendKind::Custom,
        }
    }
}

/// The device id carried by a backend, or zero for the variants that
/// carry none.
fn device_id_of(b: CrateBackend) -> u32 {
    match b {
        CrateBackend::Cpu | CrateBackend::Ane => 0,
        CrateBackend::Cuda { device_id }
        | CrateBackend::Rocm { device_id }
        | CrateBackend::Metal { device_id }
        | CrateBackend::Tpu { device_id }
        | CrateBackend::Wasm { device_id } => device_id,
        CrateBackend::SharedMemoryWorker { backend_id } => backend_id,
        CrateBackend::Custom(id) => id,
    }
}

// Flynnel.Placement is not bound here. It reports where a dispatch
// ran, and nothing in this slice runs one: these cmdlets read the
// registry and the routing decision without launching anything, so
// there is no placement for them to report. It arrives with
// the cmdlets that produce one, rather than being exported now as a
// type nothing in the module ever answers with.

// ---------------------------------------------------------------------
// Backends
// ---------------------------------------------------------------------

/// One backend, and the three separate questions about it.
#[psclass(name = "Flynnel.Backend")]
#[derive(Clone, Default)]
pub struct BackendRow {
    /// Which kind of device.
    pub kind: BackendKind,
    /// The device or backend id it carries. Zero for the kinds that
    /// carry none, which is not a device zero.
    pub device_id: u32,
    /// The crate's own short name for it.
    pub name: String,
    /// Whether it dispatches one instruction across many threads, the
    /// SIMT shape. False for the CPU, the Neural Engine, wasm and a
    /// shared-memory peer.
    pub is_simt: bool,
    /// Whether an implementation is in this process's registry. A
    /// consumer can register one without any probe having passed.
    pub registered: bool,
    /// Whether the host's probe finds the runtime. A CUDA runtime can
    /// load on a machine with no card, so this is not the same as a
    /// usable device.
    pub available: bool,
    /// Whether the crate's own sweep put it in the list a plan would
    /// pick from.
    pub detected: bool,
    /// What the probe looks at, so a reading can be argued with.
    pub probe: String,
    /// Lanes one instruction drives. One on a backend that is not
    /// SIMT. Zero when nothing is registered, because a capability is
    /// a property of an implementation and there is none to ask.
    pub simt_width: u32,
    /// Threads the backend will keep in flight. Zero when nothing is
    /// registered.
    pub max_threads_in_flight: u32,
    /// What one launch costs before any work happens. Zero when
    /// nothing is registered.
    pub launch_latency_ns: u32,
    /// Host-to-device bandwidth. Zero on the CPU backend, which moves
    /// nothing, and also zero when nothing is registered;
    /// CapabilitiesKnown is what tells those two apart.
    pub h2d_bandwidth_bytes_per_sec: u64,
    /// Whether the four capability columns came from an
    /// implementation. False means nothing is registered and they are
    /// zeros rather than measurements.
    pub capabilities_known: bool,
    /// Streaming multiprocessors on this CUDA device, which is what a
    /// launch geometry is sized against. Zero unless SmCountKnown.
    pub sm_count: u32,
    /// Whether SmCount was read. Only a CUDA device answers it, and
    /// only when the driver loaded and reported one, so a zero here
    /// with this false is an unread number rather than a device with
    /// no multiprocessors.
    pub sm_count_known: bool,
}

fn capabilities_into(row: &mut BackendRow, caps: &BackendCapabilities) {
    row.simt_width = caps.simt_width;
    row.max_threads_in_flight = caps.max_threads_in_flight;
    row.launch_latency_ns = caps.launch_latency_ns;
    row.h2d_bandwidth_bytes_per_sec = caps.h2d_bw_bytes_per_sec;
    row.capabilities_known = true;
}

pub(crate) fn row_for(backend: CrateBackend, detected: &[CrateBackend]) -> BackendRow {
    let kind = BackendKind::from(backend);
    let mut row = BackendRow {
        kind,
        device_id: device_id_of(backend),
        name: backend.name().to_string(),
        is_simt: backend.is_simt(),
        registered: false,
        available: kind.available(),
        detected: detected.contains(&backend),
        probe: kind.probe().to_string(),
        ..BackendRow::default()
    };
    if let Some(implementation) = backend_by_id(&backend) {
        row.registered = true;
        capabilities_into(&mut row, &implementation.capabilities());
    }
    // Only a CUDA device has one, and only when the driver loaded and
    // answered. Asked of the device this row names rather than of
    // device zero, so a second card reports its own.
    if kind == BackendKind::Cuda
        && let Some(count) = detect::cuda_sm_count(row.device_id as usize)
    {
        row.sm_count = count;
        row.sm_count_known = true;
    }
    row
}

/// Reads every backend this build carries, whether each is registered,
/// whether this host's probe finds it, and what a registered one can
/// do.
///
/// A backend the host does not have is a row saying so, never a
/// missing row: this module ships with every backend feature on, so
/// the code is present everywhere and absence is always a runtime
/// fact. A missing row would read like a capability nobody bound.
///
/// Registered, Available and Detected are three different questions
/// and they come apart in both directions. Read the one you mean.
///
/// There is no matching Register cmdlet. Registering takes a Rust
/// implementation of the backend trait and a script has no way to make
/// one; that is a property of the boundary rather than something
/// waiting to be built.
///
/// # Examples
///
/// `Get-FlynnelBackend`
///
/// `Get-FlynnelBackend | Where-Object Available | Format-Table Kind, Name, SimtWidth`
#[cmdlet(
    verb = "Get",
    noun = "FlynnelBackend",
    alias = "Get-FlyBackend",
    output = ["Flynnel.Backend"]
)]
#[derive(Default)]
pub struct GetFlynnelBackend {
    /// Only this kind.
    #[param(position = 0)]
    pub kind: Option<BackendKind>,
    /// The device id to ask about, for the kinds that carry one.
    /// Defaults to zero, which is the first device rather than none.
    #[param(position = 1)]
    pub device_id: Option<u32>,
}

impl Cmdlet for GetFlynnelBackend {
    fn process(&mut self, ps: &Pipeline<'_>) -> PsResult<()> {
        // Touching the registry is what registers the CPU backend: it
        // arrives on first access, and a listing that ran before
        // anything had touched it would report the one backend that
        // cannot be absent as unregistered. The answer is checked
        // rather than discarded, because a false here would mean every
        // row below reads against a registry that did not initialize.
        if !ensure_default_registered() {
            pwrs::warning!(
                ps,
                "the CPU backend is not registered, which should not be reachable: every \
                 row below reads a registry that failed to initialise, so Registered is \
                 unreliable on all of them"
            )?;
        }
        let detected = detect::detect_all();
        let device_id = self.device_id.unwrap_or(0);

        if let Some(kind) = self.kind {
            return ps.write(row_for(kind.to_crate(device_id), &detected));
        }

        let mut written: Vec<CrateBackend> = Vec::new();
        for kind in BackendKind::ENUMERABLE {
            let backend = kind.to_crate(device_id);
            written.push(backend);
            ps.write(row_for(backend, &detected))?;
        }
        // Anything registered that the walk above does not cover: a
        // Custom backend, or a second device of a kind already listed.
        // Enumerating kinds alone would hide both, and a registered
        // backend missing from a listing of backends is the one error
        // this family exists to avoid.
        for implementation in registered_backends() {
            let backend = implementation.id();
            if !written.contains(&backend) {
                ps.write(row_for(backend, &detected))?;
            }
        }
        Ok(())
    }
}

/// What a probe found, and what it looked at.
#[psclass(name = "Flynnel.BackendProbe")]
#[derive(Clone, Default)]
pub struct BackendProbe {
    /// Which kind was probed.
    pub kind: BackendKind,
    /// The crate's short name for it.
    pub name: String,
    /// What the probe answered, now rather than at startup.
    pub available: bool,
    /// Whether an implementation is registered, which is a separate
    /// question and does not follow from the probe.
    pub registered: bool,
    /// What the probe looks at.
    pub probe: String,
}

/// Runs the host's detection probe now and writes what it found.
///
/// Get-FlynnelBackend reports the same Available column; this is the
/// form for asking again, after plugging something in or loading a
/// runtime, without reading everything else.
///
/// The probe answers for the kind rather than for one device, so it
/// takes no device id. A runtime that loads is not a device that
/// works: Available true and a failing dispatch are consistent, and
/// the Probe column says what was actually looked at.
///
/// # Examples
///
/// `Test-FlynnelBackend`
///
/// `if ((Test-FlynnelBackend -Kind Cuda).Available) { ... }`
#[cmdlet(
    verb = "Test",
    noun = "FlynnelBackend",
    alias = "Test-FlyBackend",
    output = ["Flynnel.BackendProbe"]
)]
#[derive(Default)]
pub struct TestFlynnelBackend {
    /// Only this kind. Every enumerable kind when unset.
    #[param(position = 0)]
    pub kind: Option<BackendKind>,
}

impl Cmdlet for TestFlynnelBackend {
    fn process(&mut self, ps: &Pipeline<'_>) -> PsResult<()> {
        let one = |kind: BackendKind| -> BackendProbe {
            let backend = kind.to_crate(0);
            BackendProbe {
                kind,
                name: backend.name().to_string(),
                available: kind.available(),
                registered: backend_by_id(&backend).is_some(),
                probe: kind.probe().to_string(),
            }
        };
        match self.kind {
            Some(kind) => ps.write(one(kind)),
            None => {
                for kind in BackendKind::ENUMERABLE {
                    ps.write(one(kind))?;
                }
                Ok(())
            }
        }
    }
}

// ---------------------------------------------------------------------
// Accelerator operations
// ---------------------------------------------------------------------

/// One registered accelerator operation.
#[psclass(name = "Flynnel.AccelOp")]
#[derive(Clone, Default)]
pub struct AccelOp {
    /// The name it registered under.
    pub name: String,
    /// Bytes each item moves across the bus, which the routing gate
    /// reads to decide whether a transfer is worth making. Zero means
    /// the op did not declare a per-item cost.
    pub bytes_per_item: u32,
    /// Whether any backend has a kernel bound for it. False means
    /// every dispatch of this op runs its CPU implementation, whatever
    /// devices the host has.
    pub has_binding: bool,
    /// The kinds a kernel is bound on, in binding order.
    pub bound_kinds: Vec<BackendKind>,
    /// Their device ids, in the same order.
    pub bound_device_ids: Vec<u32>,
}

/// Reads every accelerator operation registered in this process.
///
/// An operation is a CPU implementation plus zero or more device
/// kernels bound against it; a dispatch picks between them. The list
/// is empty in a session where nothing has registered one, and that is
/// the ordinary state: the crate registers its linear-algebra
/// operations only when a caller asks it to, so an empty list means
/// nothing has asked rather than that the build lacks them.
///
/// There is no matching Register cmdlet. Registering an operation
/// takes its CPU implementation as a Rust closure, which a script
/// cannot supply.
///
/// # Examples
///
/// `Get-FlynnelAccelOp`
///
/// `Get-FlynnelAccelOp | Where-Object { -not $_.HasBinding }`
#[cmdlet(
    verb = "Get",
    noun = "FlynnelAccelOp",
    alias = "Get-FlyAccelOp",
    output = ["Flynnel.AccelOp"]
)]
#[derive(Default)]
pub struct GetFlynnelAccelOp {
    /// Only operations whose name contains this text.
    #[param(position = 0)]
    pub name: Option<String>,
}

impl Cmdlet for GetFlynnelAccelOp {
    fn process(&mut self, ps: &Pipeline<'_>) -> PsResult<()> {
        for op in flynnel::registered_accel_ops() {
            if let Some(filter) = &self.name
                && !op.name.contains(filter.as_str())
            {
                continue;
            }
            ps.write(AccelOp {
                name: op.name,
                bytes_per_item: op.bytes_per_item,
                has_binding: !op.kernels.is_empty(),
                bound_kinds: op.kernels.iter().map(|b| BackendKind::from(*b)).collect(),
                bound_device_ids: op.kernels.iter().map(|b| device_id_of(*b)).collect(),
            })?;
        }
        Ok(())
    }
}

/// Where one operation would run under a plan.
#[psclass(name = "Flynnel.AccelTarget")]
#[derive(Clone, Default)]
pub struct AccelTarget {
    /// The operation asked about.
    pub name: String,
    /// Whether a device would be used at all. False means the CPU
    /// implementation would run, and the remaining columns are empty
    /// rather than describing a device.
    pub routed_to_backend: bool,
    /// The kind it would route to.
    pub kind: BackendKind,
    /// Its device id.
    pub device_id: u32,
    /// The kernel handle that would be launched.
    pub kernel_handle: u64,
}

/// Reads which backend an operation would route to under a plan,
/// without running it.
///
/// This is the routing decision alone. It launches nothing, moves
/// nothing across the bus, and answers what the plan's hint, the
/// bindings and the registry resolve to right now.
///
/// A false RoutedToBackend is a real answer and the common one: it
/// means the CPU implementation would run, whether because nothing is
/// bound, because the plan's hint names a backend with no kernel for
/// this operation, or because no plan hint picks a device.
///
/// # Examples
///
/// `Get-FlynnelAccelTarget -Name flynnel.linalg.gemm_batched_f64`
///
/// `Get-FlynnelAccelTarget -Name gemm -Plan $plan`
#[cmdlet(
    verb = "Get",
    noun = "FlynnelAccelTarget",
    alias = "Get-FlyAccelTarget",
    output = ["Flynnel.AccelTarget"]
)]
#[derive(Default)]
pub struct GetFlynnelAccelTarget {
    /// The operation's registered name.
    #[param(mandatory, position = 0)]
    pub name: String,
    /// The plan whose backend hint decides the routing. A bare plan
    /// when unset, which routes by the registry alone.
    #[param(position = 1)]
    pub plan: Option<crate::plan::Plan>,
}

impl Cmdlet for GetFlynnelAccelTarget {
    fn process(&mut self, ps: &Pipeline<'_>) -> PsResult<()> {
        let registered = flynnel::registered_accel_ops();
        let Some(found) = registered.iter().find(|r| r.name == self.name) else {
            // Naming what is there rather than only what is missing: an
            // empty registry and a misspelt name are different
            // mistakes and the message has to tell them apart.
            let known = if registered.is_empty() {
                "no operations are registered in this process".to_string()
            } else {
                format!(
                    "registered: {}",
                    registered
                        .iter()
                        .map(|r| r.name.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            };
            return Err(PsError::new(
                ErrorCategory::ObjectNotFound,
                "FlynnelUnknownAccelOp",
                format!("no accelerator operation named '{}'; {known}", self.name),
            )
            .terminating());
        };

        let plan = match &self.plan {
            Some(p) => p.to_job_plan()?,
            None => flynnel::JobPlan::bare(0, 1),
        };
        let row = match flynnel::accel_target(&plan, found.op) {
            Some((backend, handle)) => AccelTarget {
                name: found.name.clone(),
                routed_to_backend: true,
                kind: BackendKind::from(backend),
                device_id: device_id_of(backend),
                kernel_handle: handle.0,
            },
            None => AccelTarget {
                name: found.name.clone(),
                routed_to_backend: false,
                ..AccelTarget::default()
            },
        };
        ps.write(row)
    }
}
