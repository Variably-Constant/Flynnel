//! The host's own facts: the processors, the NUMA layout, the measured
//! inter-core latency, and the L3 ways this process can reserve.
//!
//! Everything here is what the scheduler reads before it sizes
//! anything, so it is the family to look at first when a plan resolves
//! to a width that surprises you.

use pwrs::prelude::*;

/// The error for a host facility this machine does not have.
pub(crate) fn unsupported_err(what: &str, detail: impl std::fmt::Display) -> PsError {
    PsError::new(
        ErrorCategory::DeviceError,
        "FlynnelUnsupported",
        format!("{what} is not available on this host: {detail}"),
    )
}

/// The error for an argument the host family cannot take.
pub(crate) fn arg_err(message: impl Into<String>) -> PsError {
    PsError::new(
        ErrorCategory::InvalidArgument,
        "FlynnelArgument",
        message.into(),
    )
}

// ---------------------------------------------------------------------
// Processors
// ---------------------------------------------------------------------

/// The CPU manufacturer, from the CPUID leaf 0 vendor string.
#[psenum(name = "Flynnel.Vendor")]
#[derive(Clone, Copy, Default)]
pub enum Vendor {
    /// Vendor string `GenuineIntel`.
    Intel,
    /// Vendor string `AuthenticAMD`.
    Amd,
    /// Any other vendor, or a target that is not x86_64.
    #[default]
    Other,
}

impl From<flynnel::cpu_info::Vendor> for Vendor {
    fn from(v: flynnel::cpu_info::Vendor) -> Self {
        match v {
            flynnel::cpu_info::Vendor::Intel => Vendor::Intel,
            flynnel::cpu_info::Vendor::Amd => Vendor::Amd,
            flynnel::cpu_info::Vendor::Other => Vendor::Other,
        }
    }
}

/// What the scheduler knows about this host's processors.
#[psclass(name = "Flynnel.CpuInfo")]
#[derive(Clone, Default)]
pub struct CpuInfo {
    /// Logical processors this process can see.
    pub logical_threads: u32,
    /// Hardware threads per physical core: one where SMT is off or the
    /// target is not x86_64, two on x86_64 with it on.
    pub smt_threads_per_core: u8,
    /// Physical cores, the logical count divided by the SMT factor.
    /// The count the pool sizes its primary workers to.
    pub physical_cores: u32,
    /// The manufacturer, which gates the per-vendor bisect routing.
    pub vendor: Vendor,
    /// CPU family from CPUID leaf 1, zero off x86_64.
    pub family: u32,
    /// CPU model from CPUID leaf 1, zero off x86_64.
    pub model: u32,
    /// CPU stepping from CPUID leaf 1, zero off x86_64.
    pub stepping: u8,
    /// Whether UMONITOR and UMWAIT are present, which is what lets a
    /// worker wait without spinning.
    pub has_waitpkg: bool,
    /// The factor every dispatch floor is scaled by here. Four on a
    /// host with fewer than four physical cores, where stealing has
    /// nothing to amortize against; one otherwise.
    pub small_host_dispatch_factor: u64,
}

/// Reads what the scheduler knows about this host's processors: the
/// logical and physical counts, the SMT factor, the vendor and model,
/// and the dispatch floor those imply.
///
/// # Examples
///
/// `Get-FlynnelCpuInfo`
///
/// `(Get-FlynnelCpuInfo).PhysicalCores`
#[cmdlet(
    verb = "Get",
    noun = "FlynnelCpuInfo",
    alias = "Get-FlyCpuInfo",
    output = ["Flynnel.CpuInfo"]
)]
#[derive(Default)]
pub struct GetFlynnelCpuInfo {}

/// This host's processor facts, as one row.
///
/// Built here rather than inside the cmdlet so the Flynnel drive's
/// `host\cpu` leaf can answer the same object rather than a second
/// rendering of it. Two renderings of one reading drift, and a script
/// comparing them would be comparing this module against itself.
pub(crate) fn cpu_info_row() -> CpuInfo {
    let info = flynnel::cpu_info::cpu_info();
    CpuInfo {
        logical_threads: info.logical_threads,
        smt_threads_per_core: info.smt_threads_per_core,
        physical_cores: info.physical_cores,
        vendor: info.vendor.into(),
        family: info.family,
        model: info.model,
        stepping: info.stepping,
        has_waitpkg: flynnel::cpu_info::has_waitpkg(),
        small_host_dispatch_factor: flynnel::cpu_info::small_host_dispatch_factor(),
    }
}

impl Cmdlet for GetFlynnelCpuInfo {
    fn process(&mut self, ps: &Pipeline<'_>) -> PsResult<()> {
        ps.write(cpu_info_row())
    }
}

// ---------------------------------------------------------------------
// NUMA
// ---------------------------------------------------------------------

/// Which detection path produced the topology.
#[psenum(name = "Flynnel.NumaSource")]
#[derive(Clone, Copy, Default)]
pub enum NumaSource {
    /// `/sys/devices/system/node/*` on Linux.
    LinuxSysfs,
    /// `GetLogicalProcessorInformationEx` on Windows.
    WindowsGlpiEx,
    /// Neither probe ran: one node holding every CPU.
    #[default]
    Fallback,
}

impl From<flynnel::numa_topology::NumaSource> for NumaSource {
    fn from(s: flynnel::numa_topology::NumaSource) -> Self {
        match s {
            flynnel::numa_topology::NumaSource::LinuxSysfs => NumaSource::LinuxSysfs,
            flynnel::numa_topology::NumaSource::WindowsGlpiEx => NumaSource::WindowsGlpiEx,
            flynnel::numa_topology::NumaSource::Fallback => NumaSource::Fallback,
        }
    }
}

/// Which probe produced the cluster size, the smallest cache-coherent
/// group that shares one last-level slice.
#[psenum(name = "Flynnel.ClusterSource")]
#[derive(Clone, Copy, Default)]
pub enum ClusterSource {
    /// No cluster probe ran, or none returned a useful value.
    #[default]
    None,
    /// AMD Zen, through the CPUID L3-sharing leaf.
    AmdCpuidCcx,
    /// Intel, through the CPUID module domain.
    IntelCpuidModule,
    /// AArch64 Linux, through the sysfs cluster id.
    ArmSysfsCluster,
    /// AArch64 macOS, through the perflevel sysctl.
    AppleSysctlPerflevel,
}

impl From<flynnel::numa_topology::ClusterSource> for ClusterSource {
    fn from(s: flynnel::numa_topology::ClusterSource) -> Self {
        use flynnel::numa_topology::ClusterSource as C;
        match s {
            C::None => ClusterSource::None,
            C::AmdCpuidCcx => ClusterSource::AmdCpuidCcx,
            C::IntelCpuidModule => ClusterSource::IntelCpuidModule,
            C::ArmSysfsCluster => ClusterSource::ArmSysfsCluster,
            C::AppleSysctlPerflevel => ClusterSource::AppleSysctlPerflevel,
        }
    }
}

/// The NUMA layout this process sees.
#[psclass(name = "Flynnel.Topology")]
#[derive(Clone, Default)]
pub struct Topology {
    /// Distinct NUMA nodes visible to this process.
    pub node_count: u32,
    /// Whether there is more than one, which is what makes the
    /// per-node arena worth composing.
    pub is_multi_node: bool,
    /// The NUMA node of each logical CPU, indexed by its OS id.
    pub node_of_cpu: Vec<u32>,
    /// The distance matrix flattened row by row, node count squared
    /// entries. Ten means the same node and higher means farther; on
    /// Linux this is the SLIT, and on Windows it is ten on the
    /// diagonal and twenty elsewhere because Win32 reports no
    /// distances.
    pub distances: Vec<u8>,
    /// Log2 of the logical processors in one cache-sharing cluster.
    /// Zero where the probe found no cluster structure, which a
    /// single-die mesh part genuinely has none of.
    pub cluster_size_log2: u8,
    /// Which probe produced the cluster size.
    pub cluster_source: ClusterSource,
    /// Which probe produced the rest.
    pub source: NumaSource,
}

pub(crate) fn topology_snapshot() -> Topology {
    let topo = flynnel::numa_topology::numa_topology();
    Topology {
        node_count: topo.num_nodes,
        is_multi_node: topo.is_multi_node(),
        node_of_cpu: topo.node_of_cpu.clone(),
        distances: topo.distances.iter().flatten().copied().collect(),
        cluster_size_log2: topo.cluster_size_log2,
        cluster_source: topo.cluster_source.into(),
        source: topo.source.into(),
    }
}

/// Reads this process's NUMA layout: how many nodes there are, which
/// node each CPU belongs to, the distance between every pair, and the
/// cache-sharing cluster size with the probe that found it.
///
/// The whole matrix comes back in one object rather than one row per
/// pair, so a script pays one call for the layout however many nodes
/// the host has.
///
/// # Examples
///
/// `Get-FlynnelTopology`
///
/// `(Get-FlynnelTopology).NodeCount`
#[cmdlet(
    verb = "Get",
    noun = "FlynnelTopology",
    alias = "Get-FlyTopology",
    output = ["Flynnel.Topology"]
)]
#[derive(Default)]
pub struct GetFlynnelTopology {}

impl Cmdlet for GetFlynnelTopology {
    fn process(&mut self, ps: &Pipeline<'_>) -> PsResult<()> {
        ps.write(topology_snapshot())
    }
}

/// One pair of NUMA nodes and the distance between them.
#[psclass(name = "Flynnel.NumaDistance")]
#[derive(Clone, Default)]
pub struct NumaDistance {
    /// The node measured from.
    pub from: u32,
    /// The node measured to.
    pub to: u32,
    /// Ten for the same node, higher the farther apart they are.
    pub distance: u8,
}

/// Reads the distance between two NUMA nodes, or every pair when
/// neither is named.
///
/// # Examples
///
/// `Get-FlynnelNumaDistance -From 0 -To 1`
///
/// `Get-FlynnelNumaDistance | Format-Table`
#[cmdlet(
    verb = "Get",
    noun = "FlynnelNumaDistance",
    alias = "Get-FlyNumaDistance",
    output = ["Flynnel.NumaDistance"]
)]
#[derive(Default)]
pub struct GetFlynnelNumaDistance {
    /// The node to measure from. With it absent, every pair is written.
    #[param(position = 0)]
    pub from: Option<u32>,
    /// The node to measure to. With it absent, every node reachable
    /// from From is written.
    #[param(position = 1)]
    pub to: Option<u32>,
}

impl Cmdlet for GetFlynnelNumaDistance {
    fn process(&mut self, ps: &Pipeline<'_>) -> PsResult<()> {
        let topo = flynnel::numa_topology::numa_topology();
        let n = topo.num_nodes;
        let check = |node: u32, which: &str| -> PsResult<()> {
            if node >= n {
                return Err(arg_err(format!(
                    "{which} names node {node} and this host has {n}, numbered 0 to {}",
                    n.saturating_sub(1)
                )));
            }
            Ok(())
        };
        if let Some(from) = self.from {
            check(from, "From")?;
        }
        if let Some(to) = self.to {
            check(to, "To")?;
        }
        let froms: Vec<u32> = match self.from {
            Some(f) => vec![f],
            None => (0..n).collect(),
        };
        let tos: Vec<u32> = match self.to {
            Some(t) => vec![t],
            None => (0..n).collect(),
        };
        for from in froms {
            for &to in &tos {
                ps.write(NumaDistance {
                    from,
                    to,
                    distance: topo.distance(from, to),
                })?;
            }
        }
        Ok(())
    }
}

/// A NUMA node and the logical CPUs in it.
#[psclass(name = "Flynnel.NodeCpus")]
#[derive(Clone, Default)]
pub struct NodeCpus {
    /// The node.
    pub node: u32,
    /// Its logical CPU ids.
    pub cpus: Vec<u32>,
    /// How many, so a table reads without expanding the list.
    pub count: u32,
}

/// Reads the logical CPUs in a NUMA node, or in every node when none is
/// named.
///
/// # Examples
///
/// `Get-FlynnelNodeCpu -Node 0`
///
/// `Get-FlynnelNodeCpu | Select-Object Node, Count`
#[cmdlet(
    verb = "Get",
    noun = "FlynnelNodeCpu",
    alias = "Get-FlyNodeCpu",
    output = ["Flynnel.NodeCpus"]
)]
#[derive(Default)]
pub struct GetFlynnelNodeCpu {
    /// The node to read. With it absent, every node is written.
    #[param(position = 0)]
    pub node: Option<u32>,
}

impl Cmdlet for GetFlynnelNodeCpu {
    fn process(&mut self, ps: &Pipeline<'_>) -> PsResult<()> {
        let topo = flynnel::numa_topology::numa_topology();
        let n = topo.num_nodes;
        if let Some(node) = self.node
            && node >= n
        {
            return Err(arg_err(format!(
                "Node names {node} and this host has {n}, numbered 0 to {}",
                n.saturating_sub(1)
            )));
        }
        let nodes: Vec<u32> = match self.node {
            Some(node) => vec![node],
            None => (0..n).collect(),
        };
        for node in nodes {
            let cpus = topo.cpus_in_node(node);
            ps.write(NodeCpus {
                node,
                count: cpus.len() as u32,
                cpus,
            })?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------
// Measured inter-core latency
// ---------------------------------------------------------------------

/// The measured cost of one cache line changing hands between cores.
#[psclass(name = "Flynnel.LatencyTable")]
#[derive(Clone, Default)]
pub struct LatencyTable {
    /// Cores the table covers.
    pub core_count: u32,
    /// The matrix flattened row by row, core count squared entries in
    /// nanoseconds. A cell is zero on the diagonal.
    pub latency_ns: Vec<u32>,
    /// The mean of everything off the diagonal.
    pub mean_offdiag_ns: f64,
    /// The cheapest pair, which approximates the intra-cluster cost.
    pub min_offdiag_ns: u32,
    /// The dearest pair, which approximates the cross-socket cost.
    pub max_offdiag_ns: u32,
    /// Ping-pong iterations the calibration used.
    pub iters: u32,
    /// What the calibration sweep itself cost.
    pub calibration_wall_ns: u64,
    /// The matrix as text, rows and columns in core order.
    pub matrix: String,
}

/// Reads the measured round-trip cost of a cache line between every
/// pair of cores.
///
/// The table is calibrated once per process by a ping-pong sweep and
/// cached, so the first call in a session pays for the sweep and later
/// calls do not. A host where the sweep could not run - no affinity
/// control, or a budget that ran out - has no table, and this writes a
/// warning and nothing rather than a table of zeros.
///
/// # Examples
///
/// `Get-FlynnelLatencyTable`
///
/// `(Get-FlynnelLatencyTable).Matrix`
#[cmdlet(
    verb = "Get",
    noun = "FlynnelLatencyTable",
    alias = "Get-FlyLatencyTable",
    output = ["Flynnel.LatencyTable"]
)]
#[derive(Default)]
pub struct GetFlynnelLatencyTable {}

/// The inter-core latency table, or None on a host where the sweep
/// could not run.
///
/// Built here rather than inside the cmdlet so the Flynnel drive's
/// `host\latency` leaf answers the same object. None is the honest
/// answer and each caller says so in its own idiom: the cmdlet warns
/// and writes nothing, the drive reports a leaf that exists and is
/// empty.
pub(crate) fn latency_table_row() -> Option<LatencyTable> {
    let table = flynnel::sched::numa_latency::topology_latency_table()?;
    let n = table.n();
    let mut latency_ns = Vec::with_capacity(n * n);
    for src in 0..n {
        for dst in 0..n {
            latency_ns.push(table.latency_ns(src, dst));
        }
    }
    Some(LatencyTable {
        core_count: n as u32,
        latency_ns,
        mean_offdiag_ns: table.mean_offdiag_ns(),
        min_offdiag_ns: table.min_offdiag_ns(),
        max_offdiag_ns: table.max_offdiag_ns(),
        iters: table.iters,
        calibration_wall_ns: table.calibration_wall_ns,
        matrix: table.format_as_matrix(),
    })
}

impl Cmdlet for GetFlynnelLatencyTable {
    fn process(&mut self, ps: &Pipeline<'_>) -> PsResult<()> {
        let Some(row) = latency_table_row() else {
            pwrs::warning!(
                ps,
                "this host has no inter-core latency table: the ping-pong sweep could not \
                 pin threads or did not finish inside its budget, so nothing was measured"
            )?;
            return Ok(());
        };
        ps.write(row)
    }
}

// ---------------------------------------------------------------------
// Hardware classes
// ---------------------------------------------------------------------

/// A kernel target the scheduler can dispatch to.
#[psclass(name = "Flynnel.HwClassInfo")]
#[derive(Clone, Default)]
pub struct HwClassInfo {
    /// The class, which a plan takes through WithHwClass.
    pub class: crate::types::HwClass,
    /// Its short name, as the crate prints it.
    pub name: String,
    /// Whether it is in the matrix-extension regime, which is entered
    /// as a mode region rather than one operation at a time.
    pub is_matrix_extension: bool,
}

/// Reads every hardware class a plan can target, and whether each is a
/// tile class that has to be entered as a mode region.
///
/// This is the list of targets, not a probe of this machine: what the
/// host can actually reach is what Get-FlynnelBackend reports, because
/// a class is a kernel selector and a backend is a device.
///
/// # Examples
///
/// `Get-FlynnelHwClass`
///
/// `Get-FlynnelHwClass | Where-Object IsMatrixExtension`
#[cmdlet(
    verb = "Get",
    noun = "FlynnelHwClass",
    alias = "Get-FlyHwClass",
    output = ["Flynnel.HwClassInfo"]
)]
#[derive(Default)]
pub struct GetFlynnelHwClass {}

impl Cmdlet for GetFlynnelHwClass {
    fn process(&mut self, ps: &Pipeline<'_>) -> PsResult<()> {
        for class in crate::types::HwClass::ALL {
            let native: flynnel::HwClass = class.into();
            ps.write(HwClassInfo {
                class,
                name: native.to_string(),
                is_matrix_extension: native.is_matrix_extension(),
            })?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------
// L3 cache allocation
// ---------------------------------------------------------------------

/// What this host exposes for reserving L3 ways.
#[psclass(name = "Flynnel.CacheAllocation")]
#[derive(Clone, Default)]
pub struct CacheAllocation {
    /// Whether ways can be reserved here at all. False on every
    /// non-Linux host and wherever resctrl is not mounted.
    pub supported: bool,
    /// Allocatable partitions, the classes of service.
    pub closid_count: u32,
    /// Bits in the capacity bitmask, which is the number of L3 ways.
    pub way_count: u32,
    /// The fewest contiguous ways a valid reservation may take.
    pub min_ways: u32,
    /// L3 domains, one per node or chiplet slice.
    pub domain_count: u32,
}

/// Reads whether this host can reserve L3 ways, and how many there are
/// to reserve.
///
/// A host without it is not an error: the row comes back with Supported
/// false and zero counts, because an absent row and an absent facility
/// read alike to a script.
///
/// # Examples
///
/// `Get-FlynnelCacheAllocation`
///
/// `if ((Get-FlynnelCacheAllocation).Supported) { ... }`
#[cmdlet(
    verb = "Get",
    noun = "FlynnelCacheAllocation",
    alias = "Get-FlyCacheAllocation",
    output = ["Flynnel.CacheAllocation"]
)]
#[derive(Default)]
pub struct GetFlynnelCacheAllocation {}

/// What this host can carve out of its last-level cache, as one row.
///
/// Built here rather than inside the cmdlet so the Flynnel drive's
/// `host\cache` leaf answers the same object.
pub(crate) fn cache_allocation_row() -> CacheAllocation {
    let cap = flynnel::sched::cat::CatCapability::detect();
    CacheAllocation {
        supported: cap.supported,
        closid_count: cap.num_closids,
        way_count: cap.cbm_bits,
        min_ways: cap.min_cbm_bits,
        domain_count: cap.num_domains,
    }
}

impl Cmdlet for GetFlynnelCacheAllocation {
    fn process(&mut self, ps: &Pipeline<'_>) -> PsResult<()> {
        ps.write(cache_allocation_row())
    }
}

/// A live reservation of L3 ways, with this process bound to it.
///
/// The ways are given back when the object is released, disposed or
/// collected, which is what makes it safe to hand to a script: a
/// session that ends without releasing does not leave a class of
/// service standing.
#[psclass(name = "Flynnel.CacheReservation", mode = proxy)]
pub struct CacheReservation {
    /// The resctrl group the ways were taken in.
    pub name: String,
    /// The first way of the range.
    pub first_way: u32,
    /// How many ways.
    pub way_count: u32,
    #[psfield(skip)]
    inner: Option<flynnel::sched::cat::L3Reservation>,
}

/// The operations of a `Flynnel.CacheReservation`.
#[psmethods]
impl CacheReservation {
    /// The resctrl schemata line the reservation wrote, which is what
    /// the kernel will act on.
    pub fn schemata(&self) -> PsResult<String> {
        let Some(inner) = self.inner.as_ref() else {
            return Err(unsupported_err("this reservation", "it has been released"));
        };
        inner
            .schemata()
            .map_err(|e| unsupported_err("reading the schemata", e))
    }

    /// Gives the ways back and re-homes this process to the default
    /// group. Safe to call twice.
    pub fn release(&mut self) -> PsResult<()> {
        self.inner = None;
        Ok(())
    }
}

/// Reserves a contiguous range of L3 ways for this process and returns
/// the reservation, which gives them back when it is released, disposed
/// or collected.
///
/// Refuses on a host with no resctrl, and refuses a range that does not
/// fit the host's way count, naming both numbers.
///
/// # Examples
///
/// `$ways = New-FlynnelCacheReservation -Name hot -FirstWay 0 -NumWays 4`
///
/// `try { ... } finally { $ways.Release() }`
#[cmdlet(
    verb = "New",
    noun = "FlynnelCacheReservation",
    alias = "New-FlyCacheReservation",
    output = ["Flynnel.CacheReservation"]
)]
#[derive(Default)]
pub struct NewFlynnelCacheReservation {
    /// The resctrl group to take the ways in.
    #[param(mandatory, position = 0)]
    pub name: String,
    /// The first way of the range.
    #[param(mandatory, position = 1)]
    pub first_way: u32,
    /// How many contiguous ways to take.
    #[param(mandatory, position = 2)]
    pub num_ways: u32,
}

impl Cmdlet for NewFlynnelCacheReservation {
    fn process(&mut self, ps: &Pipeline<'_>) -> PsResult<()> {
        let inner = flynnel::sched::cat::L3Reservation::reserve_ways(
            &self.name,
            self.first_way,
            self.num_ways,
        )
        .map_err(|e| unsupported_err("reserving L3 ways", e))?;
        ps.write(CacheReservation {
            name: self.name.clone(),
            first_way: self.first_way,
            way_count: self.num_ways,
            inner: Some(inner),
        })
    }
}
