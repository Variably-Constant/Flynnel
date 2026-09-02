//! NUMA topology probe: per-CPU node membership + node-to-node
//! distance matrix.
//!
//! ## Per-platform NUMA detection
//!
//! - **Linux**: reads `/sys/devices/system/node/nodeN/cpulist` for
//!   per-CPU node membership and `/sys/devices/system/node/nodeN/distance`
//!   for the SLIT (System Locality Information Table).
//! - **Windows**: calls `GetLogicalProcessorInformationEx(RelationNumaNode)`
//!   for node membership. Win32 does NOT expose SLIT, so distances
//!   are filled uniformly: 10 for intra-node, 20 for inter-node.
//! - **macOS / other**: single-node fallback (Apple M-series is one
//!   NUMA domain; other platforms get the conservative default).
//!
//! ## Chiplet / cluster-size detection
//!
//! Independent of the macro NUMA partitioning above, this module also
//! probes the size of the **local cache-sharing cluster** - i.e. how
//! many logical processors share the L3 (or whatever the local cache
//! hierarchy's last-level slice is). That number drives chiplet-aware
//! arena partitioning per ARCAS ([arXiv:2503.11460](https://arxiv.org/abs/2503.11460)).
//!
//! | Vendor / arch                | Probe used                                                                 | Returns                                  |
//! |------------------------------|----------------------------------------------------------------------------|------------------------------------------|
//! | x86_64 + `AuthenticAMD`      | CPUID leaf `0x8000_001D` sub-leaf 3 (L3 cache, deterministic-cache leaf)   | CCX size (cores sharing one L3 slice)    |
//! | x86_64 + `GenuineIntel`      | CPUID leaf `1Fh` v2 extended topology, Module domain                       | Module/tile size (Sapphire Rapids+ tile) |
//! | aarch64 + Linux              | `/sys/devices/system/cpu/cpu0/topology/cluster_id` consensus across CPUs   | DSU cluster size (ARMv8 DynamIQ)         |
//! | aarch64 + macOS              | `sysctl hw.perflevel0.physicalcpu`                                         | Apple Silicon P-cluster size             |
//! | other                        | (none - returns 0)                                                         | 0                                        |
//!
//! In every case the returned value is `log2(cluster_size_in_logical_processors)`,
//! rounded down. For AMD CCX = 8 (Zen 1-4) this is 3; for Apple M1 Pro
//! (8 P-cores) this is 3; for Sapphire Rapids 1-tile SKU with ~15
//! cores it floors to 3 (log2(15) = 3); for Graviton (one big cluster)
//! this is `log2(total_cores)`.
//!
//! ## Public API
//!
//! - [`numa_topology()`] returns a cached `&'static NumaTopology`.
//! - [`NumaTopology::detect()`] forces a fresh probe (test use).
//! - [`NumaTopology::fallback()`] returns the single-node default.

use std::sync::OnceLock;

/// Per SLIT convention, 10 = same NUMA node.
const DEFAULT_LOCAL_DISTANCE: u8 = 10;
/// Conservative cross-node estimate used when the OS does not expose
/// real distances (Windows). The SLIT canonical "different node"
/// value is 20-30 on real silicon; we pick 20.
const DEFAULT_REMOTE_DISTANCE: u8 = 20;

/// NUMA topology snapshot. Holds variable-sized data (one entry per
/// logical processor, one row per NUMA node) so it is not `Copy`.
#[derive(Clone, Debug)]
pub struct NumaTopology {
    /// Number of distinct NUMA nodes visible to this process.
    pub num_nodes: u32,
    /// NUMA node id of each logical processor, indexed by OS
    /// logical-CPU id. Length equals the platform's logical-thread
    /// count.
    pub node_of_cpu: Vec<u32>,
    /// Symmetric `num_nodes x num_nodes` distance matrix. Convention:
    /// 10 = same node, higher = farther. On Linux this is the SLIT.
    /// On Windows we fill 10 on the diagonal and 20 off-diagonal
    /// because Win32 does not expose distances.
    pub distances: Vec<Vec<u8>>,
    /// `log2(logical processors per local cache-sharing cluster)` -
    /// the size of the smallest cache-coherent group that shares one
    /// L3 slice (or equivalent local cluster cache). Zero on
    /// platforms where the probe can't run or finds no cluster
    /// structure (single-die mesh CPUs like Skylake-X return 0 here,
    /// because there's no chiplet boundary to partition at).
    ///
    /// See the module docs for the per-vendor probe table.
    pub cluster_size_log2: u8,
    /// Which probe produced [`Self::cluster_size_log2`]. Used by
    /// the arena to emit a useful diagnostic when chiplet
    /// partitioning is engaged.
    pub cluster_source: ClusterSource,
    /// Source of the NUMA probe: which detection path ran.
    pub source: NumaSource,
}

/// Tag for which detection path produced this snapshot.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum NumaSource {
    /// `/sys/devices/system/node/*` on Linux.
    LinuxSysfs,
    /// `GetLogicalProcessorInformationEx` on Windows.
    WindowsGlpiEx,
    /// Fallback: single node, all logical CPUs in node 0.
    Fallback,
}

/// Which probe produced [`NumaTopology::cluster_size_log2`].
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ClusterSource {
    /// No cluster probe ran or none returned a useful value.
    None,
    /// AMD Zen via CPUID `0x8000_001D` L3-sharing leaf.
    AmdCpuidCcx,
    /// Intel via CPUID `1Fh` Module domain (Sapphire Rapids+).
    IntelCpuidModule,
    /// AArch64 Linux via `/sys/devices/system/cpu/cpu0/topology/cluster_id`
    /// consensus count (ARM DynamIQ Shared Unit clusters).
    ArmSysfsCluster,
    /// AArch64 macOS via `sysctl hw.perflevel0.physicalcpu`
    /// (Apple Silicon performance-core cluster).
    AppleSysctlPerflevel,
}

/// Number of logical processors visible to this process. Uses
/// [`std::thread::available_parallelism`]; falls back to 1 if the
/// platform cannot report it.
fn logical_threads() -> usize {
    std::thread::available_parallelism()
        .map(std::num::NonZeroUsize::get)
        .unwrap_or(1)
}

impl NumaTopology {
    /// Single-NUMA-node fallback when probing fails or the platform
    /// does not expose NUMA. Sizes [`Self::node_of_cpu`] from the
    /// available logical-thread count.
    pub fn fallback() -> Self {
        let n = logical_threads().max(1);
        Self {
            num_nodes: 1,
            node_of_cpu: vec![0; n],
            distances: vec![vec![DEFAULT_LOCAL_DISTANCE]],
            cluster_size_log2: 0,
            cluster_source: ClusterSource::None,
            source: NumaSource::Fallback,
        }
    }

    /// Force a fresh NUMA topology probe. The cached
    /// [`numa_topology()`] value is the same shape on the next call.
    pub fn detect() -> Self {
        let mut topo = primary_detect().unwrap_or_else(Self::fallback);
        let (size_log2, src) = detect_cluster_size_log2();
        topo.cluster_size_log2 = size_log2;
        topo.cluster_source = src;
        // Validate that distances is a square matrix sized to num_nodes.
        // Probes can produce malformed data on exotic hosts; clamp to a
        // sane shape rather than panicking downstream.
        if topo.distances.len() as u32 != topo.num_nodes
            || topo
                .distances
                .iter()
                .any(|row| row.len() as u32 != topo.num_nodes)
        {
            topo.distances = build_uniform_distance_matrix(topo.num_nodes);
        }
        topo
    }

    /// Distance between two NUMA nodes. Returns 10 (same) if either
    /// index is out of range (defensive fallback).
    #[inline]
    pub fn distance(&self, a: u32, b: u32) -> u8 {
        let (i, j) = (a as usize, b as usize);
        if i < self.distances.len() && j < self.distances[i].len() {
            self.distances[i][j]
        } else {
            DEFAULT_LOCAL_DISTANCE
        }
    }

    /// True if this host has more than one NUMA node.
    #[inline]
    pub fn is_multi_node(&self) -> bool {
        self.num_nodes >= 2
    }

    /// Returns the indices (logical CPU ids) of all logical processors
    /// in the given NUMA node.
    pub fn cpus_in_node(&self, node: u32) -> Vec<u32> {
        self.node_of_cpu
            .iter()
            .enumerate()
            .filter_map(|(i, &n)| if n == node { Some(i as u32) } else { None })
            .collect()
    }
}

/// Returns the cached NUMA topology snapshot. Probed once per process.
pub fn numa_topology() -> &'static NumaTopology {
    static CACHE: OnceLock<NumaTopology> = OnceLock::new();
    CACHE.get_or_init(NumaTopology::detect)
}

/// Build a uniform distance matrix (10 diagonal, 20 off-diagonal). Used
/// as a fallback when the primary probe produced malformed data.
fn build_uniform_distance_matrix(num_nodes: u32) -> Vec<Vec<u8>> {
    let n = num_nodes.max(1) as usize;
    let mut m = vec![vec![DEFAULT_REMOTE_DISTANCE; n]; n];
    for (i, row) in m.iter_mut().enumerate() {
        row[i] = DEFAULT_LOCAL_DISTANCE;
    }
    m
}

fn primary_detect() -> Option<NumaTopology> {
    #[cfg(target_os = "linux")]
    {
        detect_linux()
    }
    #[cfg(target_os = "windows")]
    {
        detect_windows()
    }
    #[cfg(not(any(target_os = "linux", target_os = "windows")))]
    {
        None
    }
}

// ---------------------------------------------------------------------------
// Linux: /sys/devices/system/node
// ---------------------------------------------------------------------------

#[cfg(target_os = "linux")]
fn detect_linux() -> Option<NumaTopology> {
    use std::fs;
    use std::path::Path;

    let root = Path::new("/sys/devices/system/node");
    let entries = fs::read_dir(root).ok()?;

    let mut nodes: Vec<u32> = Vec::new();
    for entry in entries.flatten() {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if let Some(rest) = name.strip_prefix("node") {
            if let Ok(id) = rest.parse::<u32>() {
                nodes.push(id);
            }
        }
    }
    if nodes.is_empty() {
        return None;
    }
    nodes.sort_unstable();

    let lt = logical_threads().max(1);
    let mut node_of_cpu = vec![0u32; lt];

    for &node in &nodes {
        let cpulist_path = root.join(format!("node{node}")).join("cpulist");
        let Ok(s) = fs::read_to_string(&cpulist_path) else { continue };
        for cpu in parse_cpulist(s.trim()) {
            if (cpu as usize) < lt {
                node_of_cpu[cpu as usize] = node;
            }
        }
    }

    let num_nodes = nodes.iter().max().copied().unwrap_or(0) + 1;
    let mut distances = build_uniform_distance_matrix(num_nodes);
    for &node in &nodes {
        let dist_path = root.join(format!("node{node}")).join("distance");
        let Ok(s) = fs::read_to_string(&dist_path) else { continue };
        let row: Vec<u8> = s
            .split_ascii_whitespace()
            .filter_map(|tok| tok.parse::<u8>().ok())
            .collect();
        if row.len() as u32 >= num_nodes {
            distances[node as usize] = row[..num_nodes as usize].to_vec();
        }
    }

    Some(NumaTopology {
        num_nodes,
        node_of_cpu,
        distances,
        cluster_size_log2: 0,
        cluster_source: ClusterSource::None,
        source: NumaSource::LinuxSysfs,
    })
}

/// Parse a Linux cpulist string ("0-3,5,7-9") into a flat Vec of CPU ids.
#[cfg(target_os = "linux")]
fn parse_cpulist(s: &str) -> Vec<u32> {
    let mut out = Vec::new();
    for token in s.split(',') {
        let token = token.trim();
        if token.is_empty() {
            continue;
        }
        if let Some((lo, hi)) = token.split_once('-') {
            if let (Ok(lo), Ok(hi)) = (lo.parse::<u32>(), hi.parse::<u32>()) {
                for cpu in lo..=hi {
                    out.push(cpu);
                }
            }
        } else if let Ok(cpu) = token.parse::<u32>() {
            out.push(cpu);
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Windows: GetLogicalProcessorInformationEx
// ---------------------------------------------------------------------------

#[cfg(target_os = "windows")]
#[allow(non_camel_case_types, non_snake_case, dead_code, clippy::upper_case_acronyms)]
mod win {
    /// Win32 `KAFFINITY` (ULONG_PTR) is pointer-sized.
    pub type KAFFINITY = usize;

    /// `LOGICAL_PROCESSOR_RELATIONSHIP::RelationNumaNode`.
    pub const RELATION_NUMA_NODE: u32 = 1;
    pub const ERROR_INSUFFICIENT_BUFFER: u32 = 122;

    #[repr(C)]
    #[derive(Copy, Clone)]
    pub struct GROUP_AFFINITY {
        pub mask: KAFFINITY,
        pub group: u16,
        pub reserved: [u16; 3],
    }

    /// Header common to every variant of
    /// `SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX`. We read the size
    /// and stride through the buffer; the body shape depends on
    /// `relationship`.
    #[repr(C)]
    #[derive(Copy, Clone)]
    pub struct SlpiExHeader {
        pub relationship: u32,
        pub size: u32,
    }

    /// `NUMA_NODE_RELATIONSHIP` body that follows [`SlpiExHeader`]
    /// when `relationship == RELATION_NUMA_NODE`. On pre-Win10 builds
    /// the GroupCount field is reserved; we read only the first
    /// GroupMask which covers Windows 10 + every consumer system
    /// Flynnel targets.
    #[repr(C)]
    #[derive(Copy, Clone)]
    pub struct NumaNodeRelationship {
        pub node_number: u32,
        pub reserved: [u8; 20],
        pub group_mask: GROUP_AFFINITY,
    }

    #[link(name = "kernel32")]
    unsafe extern "system" {
        pub fn GetLogicalProcessorInformationEx(
            relationship_type: u32,
            buffer: *mut u8,
            returned_length: *mut u32,
        ) -> i32;
        pub fn GetLastError() -> u32;
    }
}

#[cfg(target_os = "windows")]
fn detect_windows() -> Option<NumaTopology> {
    // First call: pass null buffer to learn required size.
    let mut required: u32 = 0;
    // SAFETY: `GetLogicalProcessorInformationEx` with a null
    // `Buffer` and a writable `&mut required` is the documented
    // size-query call (returns 0, sets `GetLastError() =
    // ERROR_INSUFFICIENT_BUFFER`, fills `required` with the
    // needed byte count). `&mut required` is a live local;
    // null lpBuffer is documented valid for the query call.
    let ok = unsafe {
        win::GetLogicalProcessorInformationEx(
            win::RELATION_NUMA_NODE,
            core::ptr::null_mut(),
            &mut required,
        )
    };
    if ok != 0 || required == 0 {
        return None;
    }
    // SAFETY: `GetLastError` is a no-argument thread-local
    // accessor with no preconditions.
    if unsafe { win::GetLastError() } != win::ERROR_INSUFFICIENT_BUFFER {
        return None;
    }

    let mut buf: Vec<u8> = vec![0u8; required as usize];
    let mut len = required;
    // SAFETY: `buf` has `required` bytes (matching the size
    // the previous probe returned); `&mut len` is a live local
    // initialized to that same size; `RELATION_NUMA_NODE` is a
    // valid relationship constant.
    let ok2 = unsafe {
        win::GetLogicalProcessorInformationEx(
            win::RELATION_NUMA_NODE,
            buf.as_mut_ptr(),
            &mut len,
        )
    };
    if ok2 == 0 {
        return None;
    }

    let lt = logical_threads().max(1);
    let mut node_of_cpu = vec![0u32; lt];
    let mut max_node: u32 = 0;
    let mut node_seen: Vec<bool> = Vec::new();

    let mut offset: usize = 0;
    while offset + core::mem::size_of::<win::SlpiExHeader>() <= len as usize {
        // SAFETY: the `while` predicate guarantees the
        // header bytes are inside the buffer; `read_unaligned`
        // tolerates any alignment of the source pointer.
        let header: win::SlpiExHeader = unsafe {
            core::ptr::read_unaligned(buf.as_ptr().add(offset) as *const _)
        };
        let entry_size = header.size as usize;
        if entry_size == 0 || offset + entry_size > len as usize {
            break;
        }
        if header.relationship == win::RELATION_NUMA_NODE {
            let body_offset = offset + core::mem::size_of::<win::SlpiExHeader>();
            if body_offset + core::mem::size_of::<win::NumaNodeRelationship>() <= len as usize {
                // SAFETY: the enclosing `if` confirms the body
                // bytes are in-bounds of `buf`; `read_unaligned`
                // tolerates the unaligned source pointer.
                let body: win::NumaNodeRelationship = unsafe {
                    core::ptr::read_unaligned(
                        buf.as_ptr().add(body_offset) as *const _,
                    )
                };
                let node = body.node_number;
                if node > max_node {
                    max_node = node;
                }
                if (node as usize) >= node_seen.len() {
                    node_seen.resize(node as usize + 1, false);
                }
                node_seen[node as usize] = true;

                // GROUP_AFFINITY: each bit i of mask = "logical CPU
                // (group * 64 + i) belongs to this node." We support
                // group 0 only here; multi-group hosts (>64 CPUs)
                // require iterating all GroupCount masks, which the
                // pre-Win10 NumaNodeRelationship shape does not expose.
                let mask = body.group_mask.mask;
                let group = body.group_mask.group;
                let base = (group as usize) << 6;
                for bit in 0..(core::mem::size_of::<win::KAFFINITY>() << 3) {
                    if (mask >> bit) & 1 == 1 {
                        let cpu = base + bit;
                        if cpu < lt {
                            node_of_cpu[cpu] = node;
                        }
                    }
                }
            }
        }
        offset += entry_size;
    }

    let num_nodes = node_seen.iter().filter(|&&seen| seen).count() as u32;
    let num_nodes = num_nodes.max(1);
    let distances = build_uniform_distance_matrix(num_nodes);

    Some(NumaTopology {
        num_nodes,
        node_of_cpu,
        distances,
        cluster_size_log2: 0,
        cluster_source: ClusterSource::None,
        source: NumaSource::WindowsGlpiEx,
    })
}

// ---------------------------------------------------------------------------
// Local-cluster size detection (cores sharing one L3 slice / cluster cache)
//
// Architecture-portable dispatch: the right probe depends on (vendor, arch,
// OS). Every probe returns log2(cluster_size_in_logical_processors); zero
// means "no probe matched or no cluster structure detected" (e.g. Intel
// Skylake-X mesh-on-die, single-die ARMv7).
// ---------------------------------------------------------------------------

fn detect_cluster_size_log2() -> (u8, ClusterSource) {
    #[cfg(target_arch = "x86_64")]
    {
        // AMD path: existing CPUID 0x8000_001D L3-sharing leaf.
        if is_amd_cpu() {
            // SAFETY: `amd_ccx_threads_log2` calls CPUID,
            // which is unconditionally available on all
            // x86_64 CPUs (it is part of the base ISA from
            // Pentium onward).
            let n = unsafe { amd_ccx_threads_log2() };
            if n > 0 {
                return (n, ClusterSource::AmdCpuidCcx);
            }
        }
        // Intel path: CPUID 1Fh Module domain (Sapphire Rapids+).
        if is_intel_cpu() {
            // SAFETY: same CPUID-is-base-ISA justification as
            // the AMD branch directly above.
            let n = unsafe { intel_module_threads_log2() };
            if n > 0 {
                return (n, ClusterSource::IntelCpuidModule);
            }
        }
        (0, ClusterSource::None)
    }
    #[cfg(target_arch = "aarch64")]
    {
        // ARM Linux path: sysfs cluster_id consensus count.
        #[cfg(target_os = "linux")]
        {
            let n = arm_linux_cluster_size_log2();
            if n > 0 {
                return (n, ClusterSource::ArmSysfsCluster);
            }
        }
        // Apple Silicon macOS path: sysctl perflevel0.physicalcpu.
        #[cfg(target_os = "macos")]
        {
            let n = apple_perflevel0_size_log2();
            if n > 0 {
                return (n, ClusterSource::AppleSysctlPerflevel);
            }
        }
        (0, ClusterSource::None)
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        (0, ClusterSource::None)
    }
}

// ---------------------------------------------------------------------------
// x86_64 vendor detection (probe CPUID leaf 0 EBX/EDX/ECX)
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
fn is_amd_cpu() -> bool {
    use std::arch::x86_64::__cpuid;
    // __cpuid is safe to call on x86_64 - the CPUID instruction is
    // always available on this arch (it was added in 1993). The std
    // intrinsic doesn't require an `unsafe` block on x86_64 targets.
    let r = __cpuid(0);
    // EBX = "Auth", EDX = "enti", ECX = "cAMD".
    r.ebx == 0x6874_7541 && r.edx == 0x6974_6e65 && r.ecx == 0x444d_4163
}

#[cfg(target_arch = "x86_64")]
fn is_intel_cpu() -> bool {
    use std::arch::x86_64::__cpuid;
    let r = __cpuid(0);
    // EBX = "Genu", EDX = "ineI", ECX = "ntel".
    r.ebx == 0x756e_6547 && r.edx == 0x4965_6e69 && r.ecx == 0x6c65_746e
}

// ---------------------------------------------------------------------------
// AMD CCX size via CPUID 0x8000_001D sub-leaf 3 (Zen L3 cache leaf)
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
unsafe fn amd_ccx_threads_log2() -> u8 {
    use std::arch::x86_64::__cpuid_count;
    // Confirm AMD extended cache leaf is present.
    let max_ext = __cpuid_count(0x8000_0000, 0).eax;
    if max_ext < 0x8000_001D {
        return 0;
    }
    // Walk sub-leaves until we find the L3 cache (level == 3) or run
    // out. AMD encodes the same layout as Intel leaf 4:
    //   EAX[4:0]   = cache type (0 = end)
    //   EAX[7:5]   = level
    //   EAX[25:14] = NumSharingCache minus 1 (= logical processors
    //                that share this cache)
    for sub in 0u32..16 {
        let r = __cpuid_count(0x8000_001D, sub);
        let cache_type = r.eax & 0x1F;
        if cache_type == 0 {
            break;
        }
        let level = (r.eax >> 5) & 0x07;
        if level == 3 {
            let num_sharing = ((r.eax >> 14) & 0x0FFF) + 1;
            // Round down to pow2 for our log2 representation; CCX
            // size on real Zen silicon is always a power of two (4 or
            // 8 in Zen 1-4 today). If it isn't, log2_u32 still
            // returns the floor which keeps callers conservative.
            return log2_u32(num_sharing);
        }
    }
    0
}

// ---------------------------------------------------------------------------
// Intel module-domain size via CPUID 1Fh (v2 extended topology)
//
// Leaf 1Fh enumerates topology domains starting at sub-leaf 0. Each
// sub-leaf reports:
//   EAX[4:0]   = shift count for next-domain x2APIC partition
//   EBX[15:0]  = logical processors at this domain level (cumulative
//                from leaf-0 up through this domain)
//   ECX[15:8]  = domain type (0=Invalid, 1=SMT, 2=Core, 3=Module,
//                4=Tile, 5=Die, 6=DieGrp - per Intel's
//                SDM-Processor-Topology-Enumeration reference repo,
//                cpuid_topology.h CPU_DOMAIN enum)
// Iteration terminates when EBX == 0.
//
// We want ModuleDomain=3 because on Sapphire Rapids that's the
// chiplet-tile size (cores sharing one L3 slice). EBX[15:0] is
// cumulative through the domain, so it directly gives logical
// processors per module.
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
unsafe fn intel_module_threads_log2() -> u8 {
    use std::arch::x86_64::__cpuid_count;
    // Confirm 1Fh is supported (max basic CPUID >= 0x1F).
    let max_basic = __cpuid_count(0, 0).eax;
    if max_basic < 0x1F {
        return 0;
    }
    const MODULE_DOMAIN: u32 = 3;
    for sub in 0u32..32 {
        let r = __cpuid_count(0x1F, sub);
        if r.ebx & 0xFFFF == 0 {
            // Domain enumeration terminates.
            break;
        }
        let domain_type = (r.ecx >> 8) & 0xFF;
        if domain_type == MODULE_DOMAIN {
            let lp_count = r.ebx & 0xFFFF;
            return log2_u32(lp_count);
        }
    }
    0
}

// ---------------------------------------------------------------------------
// AArch64 Linux: /sys/devices/system/cpu/cpuX/topology/cluster_id
//
// Kernel >= 5.16 exposes cluster_id per CPU. Cores sharing the same
// cluster_id are in the same DSU (DynamIQ Shared Unit) cluster,
// sharing local L2/L3 cache. We count how many CPUs share cpu0's
// cluster_id; that gives the cluster size from cpu0's vantage.
//
// On heterogeneous topologies (Apple Silicon under Asahi Linux,
// Cortex-X+A series big.LITTLE) cpu0 typically lands on a P-cluster
// or the first cluster, giving a conservative-but-correct local
// cluster size for partitioning.
// ---------------------------------------------------------------------------

#[cfg(all(target_arch = "aarch64", target_os = "linux"))]
fn arm_linux_cluster_size_log2() -> u8 {
    use std::fs;
    let cpu0_cluster_id =
        fs::read_to_string("/sys/devices/system/cpu/cpu0/topology/cluster_id")
            .ok()
            .and_then(|s| s.trim().parse::<i32>().ok());
    let Some(cpu0_id) = cpu0_cluster_id else { return 0 };
    if cpu0_id < 0 {
        // Kernel returns -1 when cluster topology is unknown.
        return 0;
    }
    let mut count: u32 = 0;
    let lt = logical_threads().max(1) as u32;
    for cpu in 0..lt {
        let path = format!("/sys/devices/system/cpu/cpu{cpu}/topology/cluster_id");
        if let Ok(s) = fs::read_to_string(&path) {
            if let Ok(id) = s.trim().parse::<i32>() {
                if id == cpu0_id {
                    count += 1;
                }
            }
        }
    }
    log2_u32(count.max(1))
}

// ---------------------------------------------------------------------------
// AArch64 macOS: sysctl hw.perflevel0.physicalcpu
//
// On Apple Silicon (M1/M2/M3/M4) macOS exposes performance levels via
// sysctl. perflevel0 is the highest performance level (P-cores);
// perflevel1 is the next level (E-cores). For local-cluster
// partitioning we want the P-cluster size since that's where the
// hottest L2 group lives.
// ---------------------------------------------------------------------------

#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
fn apple_perflevel0_size_log2() -> u8 {
    // Raw libc FFI for sysctlbyname so we don't pull the `libc` crate
    // as a build-graph dep just for one Darwin-only call site.
    // Signature per macOS's <sys/sysctl.h>:
    //   int sysctlbyname(const char *name, void *oldp, size_t *oldlenp,
    //                    const void *newp, size_t newlen);
    unsafe extern "C" {
        fn sysctlbyname(
            name: *const core::ffi::c_char,
            oldp: *mut core::ffi::c_void,
            oldlenp: *mut usize,
            newp: *const core::ffi::c_void,
            newlen: usize,
        ) -> core::ffi::c_int;
    }
    let mut value: i64 = 0;
    let mut size: usize = core::mem::size_of::<i64>();
    // SAFETY: sysctlbyname reads an OID into a typed buffer we own.
    // Both pointers are valid and aligned; size is initialized to the
    // buffer's exact byte size; the C string is null-terminated.
    let rc = unsafe {
        sysctlbyname(
            c"hw.perflevel0.physicalcpu".as_ptr(),
            &mut value as *mut i64 as *mut core::ffi::c_void,
            &mut size,
            core::ptr::null(),
            0,
        )
    };
    if rc != 0 || value <= 0 {
        return 0;
    }
    log2_u32(value as u32)
}

fn log2_u32(x: u32) -> u8 {
    if x <= 1 {
        0
    } else {
        (31 - x.leading_zeros()) as u8
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fallback_is_single_node() {
        let f = NumaTopology::fallback();
        assert_eq!(f.num_nodes, 1);
        assert_eq!(f.distances, vec![vec![DEFAULT_LOCAL_DISTANCE]]);
        assert_eq!(f.source, NumaSource::Fallback);
        assert_eq!(f.cluster_size_log2, 0);
        assert_eq!(f.cluster_source, ClusterSource::None);
        for &n in &f.node_of_cpu {
            assert_eq!(n, 0);
        }
    }

    #[test]
    fn fallback_node_of_cpu_matches_logical_threads() {
        let f = NumaTopology::fallback();
        assert_eq!(f.node_of_cpu.len(), logical_threads().max(1));
    }

    #[test]
    fn numa_topology_is_cached() {
        let a = numa_topology() as *const _;
        let b = numa_topology() as *const _;
        assert_eq!(a, b);
    }

    #[test]
    fn detect_produces_sane_snapshot() {
        let t = NumaTopology::detect();
        assert!(t.num_nodes >= 1, "num_nodes must be >= 1");
        assert_eq!(
            t.distances.len() as u32,
            t.num_nodes,
            "distance matrix must have num_nodes rows"
        );
        for row in &t.distances {
            assert_eq!(
                row.len() as u32,
                t.num_nodes,
                "distance matrix must be square"
            );
        }
        for i in 0..t.num_nodes as usize {
            assert_eq!(
                t.distances[i][i],
                DEFAULT_LOCAL_DISTANCE,
                "diagonal entry [{i}][{i}] must be 10"
            );
        }
        let lt = logical_threads().max(1);
        assert_eq!(t.node_of_cpu.len(), lt);
        for &n in &t.node_of_cpu {
            assert!(
                n < t.num_nodes,
                "node id {n} out of range (num_nodes = {})",
                t.num_nodes
            );
        }
    }

    #[test]
    fn distance_accessor_handles_oob() {
        let t = NumaTopology::fallback();
        assert_eq!(t.distance(99, 0), DEFAULT_LOCAL_DISTANCE);
        assert_eq!(t.distance(0, 99), DEFAULT_LOCAL_DISTANCE);
    }

    #[test]
    fn cpus_in_node_round_trips_against_node_of_cpu() {
        let t = numa_topology();
        for node in 0..t.num_nodes {
            for cpu in t.cpus_in_node(node) {
                assert_eq!(
                    t.node_of_cpu[cpu as usize], node,
                    "cpu {cpu} reported in node {node} but node_of_cpu says {}",
                    t.node_of_cpu[cpu as usize]
                );
            }
        }
    }

    #[test]
    fn is_multi_node_matches_num_nodes() {
        assert!(!NumaTopology::fallback().is_multi_node());
        let t = numa_topology();
        assert_eq!(t.is_multi_node(), t.num_nodes >= 2);
    }

    #[test]
    fn parse_cpulist_handles_ranges_and_singletons() {
        #[cfg(target_os = "linux")]
        {
            assert_eq!(super::parse_cpulist("0-3"), vec![0, 1, 2, 3]);
            assert_eq!(super::parse_cpulist("0,2,4"), vec![0, 2, 4]);
            assert_eq!(super::parse_cpulist("0-1,3,5-6"), vec![0, 1, 3, 5, 6]);
            assert_eq!(super::parse_cpulist(""), Vec::<u32>::new());
        }
    }

    #[test]
    fn log2_u32_matches_intrinsic() {
        for k in 0u32..16 {
            assert_eq!(log2_u32(1u32 << k) as u32, k, "log2({})", 1u32 << k);
        }
        assert_eq!(log2_u32(0), 0);
        assert_eq!(log2_u32(1), 0);
    }

    #[test]
    fn cluster_detection_returns_sane_value() {
        let (n, src) = detect_cluster_size_log2();
        assert!(n <= 7, "cluster_size_log2 {n} unreasonably large");
        // If we got a non-zero size we must also have named the source.
        if n > 0 {
            assert_ne!(src, ClusterSource::None);
        } else {
            assert_eq!(src, ClusterSource::None);
        }
    }

    #[test]
    fn cluster_size_is_recorded_on_detect_snapshot() {
        let t = NumaTopology::detect();
        // Mirrors `cluster_detection_returns_sane_value` at the
        // snapshot layer to catch regressions where `detect()`
        // forgets to populate the field from the probe.
        assert!(t.cluster_size_log2 <= 7);
        if t.cluster_size_log2 > 0 {
            assert_ne!(t.cluster_source, ClusterSource::None);
        } else {
            assert_eq!(t.cluster_source, ClusterSource::None);
        }
    }

    #[test]
    fn build_uniform_distance_matrix_diagonal_is_local() {
        for n in 1u32..6 {
            let m = build_uniform_distance_matrix(n);
            assert_eq!(m.len() as u32, n);
            for (i, row) in m.iter().enumerate() {
                assert_eq!(row.len() as u32, n);
                assert_eq!(
                    row[i], DEFAULT_LOCAL_DISTANCE,
                    "[{i}][{i}] not local for n={n}"
                );
                for (j, &cell) in row.iter().enumerate() {
                    if i != j {
                        assert_eq!(cell, DEFAULT_REMOTE_DISTANCE);
                    }
                }
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn vendor_probes_are_mutually_exclusive() {
        // A CPU can't simultaneously be both AMD and Intel.
        assert!(!(is_amd_cpu() && is_intel_cpu()));
    }
}
