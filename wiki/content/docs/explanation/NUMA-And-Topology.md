---
title: NUMA and Topology
weight: 5
---

Flynnel probes three orthogonal pieces of host hardware at startup. All three cache their results in process-level `OnceLock`s and run at most once per process per probe.

## `CpuInfo`

Defined in [`src/cpu_info.rs`](https://github.com/markusmcnugen/flynnel/blob/main/src/cpu_info.rs); accessed via `flynnel::cpu_info::cpu_info() -> &'static CpuInfo`.

```rust
pub struct CpuInfo {
    pub logical_threads: u32,
    pub smt_threads_per_core: u8,
    pub smt_threads_log2: u8,
    pub physical_cores: u32,
}
```

| Field | Source |
|-------|--------|
| `logical_threads` | `std::thread::available_parallelism()` |
| `smt_threads_per_core` | x86_64: CPUID leaf 1, EDX bit 28 (HTT) gives 2 if set, 1 otherwise. Non-x86_64: always 1. |
| `smt_threads_log2` | 0 for SMT-1; 1 for SMT-2. |
| `physical_cores` | `logical_threads / smt_threads_per_core`. |

Notes:

- HTT=1 only guarantees the silicon *can* do SMT; OS may have it disabled. The detection cross-checks against `logical_threads` to avoid over-reporting.
- Intel hybrid CPUs (P-cores SMT-2 + E-cores SMT-1) get a coarse SMT-2 classification. The two scalars `cpu_info()` exposes are correct enough for the worker-sizing decisions Flynnel consumes them for; precise per-class detail lives in consumer crates.
- POWER8 / POWER9-SMT8 hosts under-report. Construct a `CpuInfo` value directly to override.

## `NumaTopology`

Defined in [`src/numa_topology.rs`](https://github.com/markusmcnugen/flynnel/blob/main/src/numa_topology.rs); accessed via `flynnel::numa_topology() -> &'static NumaTopology`.

```rust
pub struct NumaTopology {
    pub num_nodes: u32,
    pub node_of_cpu: Vec<u32>,
    pub distances: Vec<Vec<u8>>,
    pub cluster_size_log2: u8,
    pub cluster_source: ClusterSource,
    pub source: NumaSource,
}

pub enum NumaSource {
    LinuxSysfs,
    WindowsGlpiEx,
    Fallback,
}

pub enum ClusterSource {
    None,
    AmdCpuidCcx,
    IntelCpuidModule,
    ArmSysfsCluster,
    AppleSysctlPerflevel,
}
```

| Field | Meaning |
|-------|---------|
| `num_nodes` | Number of distinct NUMA nodes visible to this process. |
| `node_of_cpu[i]` | NUMA node id of logical CPU `i`. Length equals `logical_threads`. |
| `distances[i][j]` | Symmetric NUMA distance matrix. SLIT convention: 10 = same node, higher = farther. |
| `cluster_size_log2` | `log2(logical processors per local cache-sharing cluster)`. The size of the smallest cache-coherent group sharing one L3 slice (or equivalent). Zero on platforms where no chiplet boundary was detected (Skylake-X mesh-on-die, single-die ARMv7). |
| `cluster_source` | Which probe populated `cluster_size_log2`. See the cluster-detection table below. |
| `source` | Which NUMA-detection path produced this snapshot. |

### Per-platform NUMA detection

| Platform | Path |
|----------|------|
| Linux | Reads `/sys/devices/system/node/nodeN/cpulist` for per-CPU node membership and `/sys/devices/system/node/nodeN/distance` for the SLIT. |
| Windows | Calls `GetLogicalProcessorInformationEx(RelationNumaNode)` for node membership. Win32 does NOT expose SLIT, so distances are filled uniformly: 10 for intra-node, 20 for inter-node. |
| macOS / other | Single-node fallback. Apple M-series is one NUMA domain. |

### Chiplet / cluster-size detection

Independent of the macro NUMA partitioning above, the probe also detects the size of the **local cache-sharing cluster** - i.e. how many logical processors share the L3 slice (or the equivalent local cache). That number drives chiplet-aware arena partitioning per ARCAS ([arXiv:2503.11460](https://arxiv.org/abs/2503.11460)).

| Vendor / arch                | Probe used                                                                 | Returns                                  |
|------------------------------|----------------------------------------------------------------------------|------------------------------------------|
| x86_64 + `AuthenticAMD`      | CPUID leaf `0x8000_001D` sub-leaf 3 (L3 cache, deterministic-cache leaf)   | CCX size (cores sharing one L3 slice)    |
| x86_64 + `GenuineIntel`      | CPUID leaf `1Fh` v2 extended topology, Module domain                       | Module/tile size (Sapphire Rapids+ tile) |
| aarch64 + Linux              | `/sys/devices/system/cpu/cpu0/topology/cluster_id` consensus across CPUs   | DSU cluster size (ARMv8 DynamIQ)         |
| aarch64 + macOS              | `sysctl hw.perflevel0.physicalcpu`                                         | Apple Silicon P-cluster size             |
| other                        | (none - returns 0)                                                         | 0                                        |

In every case the field stores `log2(cluster_size_in_logical_processors)` rounded down. For AMD CCX = 8 (Zen 1-4) this is 3; for Apple M1 Pro (8 P-cores) this is 3; for Sapphire Rapids 1-tile SKU with ~15 cores it floors to 3; for Graviton with one big cluster this is `log2(total_cores)`.

The Intel Module-domain code is the analogue of AMD's CCX code: same shape, different CPUID leaf, exact match for what `cluster_size_log2` is meant to represent. The Module domain (type 3 in CPUID 1Fh per Intel's [SDM-Processor-Topology-Enumeration reference repo](https://github.com/intel/SDM-Processor-Topology-Enumeration) `cpuid_topology.h` `CPU_DOMAIN` enum) is the chiplet boundary on Sapphire Rapids and later.

### Methods

- `topo.distance(a: u32, b: u32) -> u8` - distance between two NUMA nodes. Returns 10 (local default) if either index is out of range.
- `topo.is_multi_node() -> bool` - `num_nodes >= 2`.
- `topo.cpus_in_node(node: u32) -> Vec<u32>` - logical CPU ids in the given node.
- `NumaTopology::fallback()` - single-NUMA fallback (used on macOS, on detection failure, and in tests).
- `NumaTopology::detect()` - force a fresh probe. The cached `numa_topology()` value retains the same shape afterward.

## `TopologyLatencyTable`

Defined in [`src/sched/numa_latency.rs`](https://github.com/markusmcnugen/flynnel/blob/main/src/sched/numa_latency.rs); accessed via `flynnel::sched::topology_latency_table() -> Option<&'static TopologyLatencyTable>`.

Per-core sync-latency calibration table built by ping-pong measurements between pinned cores. The `K_unified` / `K_core` dispatch policy uses these numbers to decide whether cooperation amortizes its sync cost on the host.

```rust
pub struct TopologyLatencyTable {
    pub n: usize,
    pub iters: u32,
    pub entries_ns: Vec<u64>,
}
```

`entries_ns[src * n + dst]` is the mean per-iteration round-trip time in nanoseconds between cores `src` and `dst` (one-way sync cost is half this).

### Methods

- `table.lookup(src, dst) -> u64` - latency in nanoseconds; zero if out of range.
- `table.format_as_matrix() -> String` - human-readable matrix for debug.
- `table.min_offdiag()` / `table.max_offdiag()` / `table.mean_offdiag()` - summary statistics excluding the zero-diagonal.

### Calibration

- `calibrate_table(iters: u32) -> Option<TopologyLatencyTable>` - explicit one-shot calibration.
- `topology_latency_table() -> Option<&'static TopologyLatencyTable>` - cached lazy-init; returns `None` when `core_affinity::get_core_ids()` reports no pinnable cores (uncommon).
- `force_calibrate() -> Option<TopologyLatencyTable>` - runs a fresh calibration regardless of the cached value.
- `CALIBRATION_BUDGET: Duration = Duration::from_millis(500)` - total time budget for the calibration sweep; the iter count auto-adjusts to fit.

### Constants

- `DEFAULT_PING_PONG_ITERS = 200`
- `MIN_PING_PONG_ITERS = 32`
- `MAX_PING_PONG_ITERS = 4096`

## How they compose

When a [`JobPlan`](JobPlan-Reference.md) is dispatched:

1. [`pick_tier(plan, numa_topology())`](Sched-Module-Reference.md#pick_tier) picks the scheduler tier. Reads `topo.is_multi_node()` to decide whether `Hierarchical` collapses to `Local`.
2. The arena reads `cpu_info()` to size per-NUMA-node worker counts (primaries = physical cores in that node; SMT extension = primaries * (smt - 1)).
3. NUMA-aware allocations (`NumaAlloc`, `bg_zero::prepare`) read `numa_topology()` to pick the local node.
4. Cooperative-vector calls ([`cooperative_join_n`](Sched-Module-Reference.md#cooperative_join_n)) consult `topology_latency_table()` when present to decide whether the per-cooperative-call sync cost is worth the parallelism.

All three probes are decoupled: a binary that only uses CPU dispatch never touches `TopologyLatencyTable`; the cached `OnceLock` only runs its calibration when the cooperative path actually fires.
