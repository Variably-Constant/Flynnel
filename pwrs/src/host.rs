//! The host's own facts: what the CPU is, and what the scheduler reads
//! from it before it sizes anything.

use pwrs::prelude::*;

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
    /// Whether the UMONITOR and UMWAIT instructions are present, which
    /// is what lets a worker wait without spinning.
    pub has_waitpkg: bool,
    /// The factor every dispatch floor is scaled by on this host. Four
    /// on a host with fewer than four physical cores, where stealing
    /// has nothing to amortize against; one otherwise.
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

impl Cmdlet for GetFlynnelCpuInfo {
    fn process(&mut self, ps: &Pipeline<'_>) -> PsResult<()> {
        let info = flynnel::cpu_info::cpu_info();
        ps.write(CpuInfo {
            logical_threads: info.logical_threads,
            smt_threads_per_core: info.smt_threads_per_core,
            physical_cores: info.physical_cores,
            vendor: info.vendor.into(),
            family: info.family,
            model: info.model,
            stepping: info.stepping,
            has_waitpkg: flynnel::cpu_info::has_waitpkg(),
            small_host_dispatch_factor: flynnel::cpu_info::small_host_dispatch_factor(),
        })
    }
}
