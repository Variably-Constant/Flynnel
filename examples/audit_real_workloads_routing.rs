//! Pre-bench audit: for each of the 13 real-workload bench cells,
//! print the actual JobPlan fields that the scheduler resolves to,
//! and compare against the canonical DispatchProfile the workload
//! would want a knowledgeable user to pin.
//!
//! Run with:
//! ```text
//! cargo run --release --example audit_real_workloads_routing
//! ```
//!
//! Output format per workload:
//!
//! ```
//! image_blur (N=1048576):
//!   JobPlan::new(default)      use_smt=false  ns/elem=Some(12)  oversub=Some(1)  variant=Some(...)
//!   set_profile(PortBound)     use_smt=false  ns/elem=Some(12)  oversub=Some(1)  variant=Some(...)
//!   canonical: PortBound       MATCH
//! ```

use flynnel::sched::adaptive_profile::active_workload_class;
use flynnel::{DispatchProfile, JobPlan};

#[derive(Copy, Clone)]
struct WorkloadInfo {
    name: &'static str,
    n: u32,
    canonical: DispatchProfile,
    bottleneck: &'static str,
}

fn workloads() -> Vec<WorkloadInfo> {
    vec![
        WorkloadInfo { name: "image_blur",      n: 1024 * 1024,         canonical: DispatchProfile::PortBound,    bottleneck: "port (integer ADD pipeline saturated)" },
        WorkloadInfo { name: "merkle",          n: 16_384,              canonical: DispatchProfile::LatencyBound, bottleneck: "latency (long hash chain per leaf, ~500 cycles)" },
        WorkloadInfo { name: "grep",            n: 16,                  canonical: DispatchProfile::MemoryBound,  bottleneck: "memory (streaming byte scan, dcache-bound)" },
        WorkloadInfo { name: "sort",            n: 1_000_000,           canonical: DispatchProfile::PortBound,    bottleneck: "port (branchy compare-and-swap)" },
        WorkloadInfo { name: "pagerank",        n: 10_000,              canonical: DispatchProfile::MemoryBound,  bottleneck: "memory (indirect-gather from rank[])" },
        WorkloadInfo { name: "rle",             n: 256,                 canonical: DispatchProfile::PortBound,    bottleneck: "port (byte compare + push)" },
        WorkloadInfo { name: "conway",          n: 512 * 512,           canonical: DispatchProfile::PortBound,    bottleneck: "port (stencil add + match)" },
        WorkloadInfo { name: "spmv",            n: 100_000,             canonical: DispatchProfile::MemoryBound,  bottleneck: "memory (indirect-gather from x[])" },
        WorkloadInfo { name: "histogram",       n: 16,                  canonical: DispatchProfile::MemoryBound,  bottleneck: "memory (streaming + scatter into bins)" },
        WorkloadInfo { name: "nbody",           n: 1024,                canonical: DispatchProfile::LatencyBound, bottleneck: "latency (sqrt chain per pair, FP pipeline stalls)" },
        WorkloadInfo { name: "monte_carlo_pi",  n: 10,                  canonical: DispatchProfile::PortBound,    bottleneck: "port (RNG IMUL + compare)" },
        WorkloadInfo { name: "kmeans",          n: 100,                 canonical: DispatchProfile::PortBound,    bottleneck: "port (FMA-bound distance computation)" },
        WorkloadInfo { name: "csv_scan",        n: 64,                  canonical: DispatchProfile::PortBound,    bottleneck: "port (byte compare + parse)" },
    ]
}

fn fmt_profile(p: DispatchProfile) -> &'static str {
    match p {
        DispatchProfile::LatencyBound => "LatencyBound",
        DispatchProfile::PortBound => "PortBound",
        DispatchProfile::MemoryBound => "MemoryBound",
        DispatchProfile::Streaming => "Streaming",
        DispatchProfile::Unspecified => "Unspecified",
    }
}

fn print_plan_row(label: &str, plan: &JobPlan) {
    println!(
        "  {:<28}  use_smt={:<5}  ns/elem={:?}  oversub={:?}  bisect_variant={:?}",
        label,
        plan.use_smt,
        plan.estimated_per_item_ns,
        plan.oversubscription_log2,
        plan.bisect_variant,
    );
}

fn main() {
    println!("Process-global active WorkloadClass = {:?}", active_workload_class());
    println!();
    println!("Per-workload audit of real_workloads bench routing:");
    println!("(K_outer fixed at 6 to match the bench file)");
    println!();

    let mut mismatches = Vec::new();
    let mut matches = Vec::new();

    for w in workloads() {
        println!("==== {} (N={}, bottleneck={}) ====", w.name, w.n, w.bottleneck);
        let default_plan = JobPlan::new(6, w.n);
        let canonical_plan = JobPlan::set_profile(6, w.n, w.canonical);
        print_plan_row("JobPlan::new (default)", &default_plan);
        print_plan_row(
            &format!("set_profile({})", fmt_profile(w.canonical)),
            &canonical_plan,
        );

        let mismatch = default_plan.use_smt != canonical_plan.use_smt
            || default_plan.estimated_per_item_ns != canonical_plan.estimated_per_item_ns
            || default_plan.oversubscription_log2 != canonical_plan.oversubscription_log2;
        if mismatch {
            println!(
                "  >>> MISMATCH: default routing differs from canonical {} <<<",
                fmt_profile(w.canonical)
            );
            mismatches.push((w.name, w.canonical));
        } else {
            println!("  ok: default routing matches canonical {}", fmt_profile(w.canonical));
            matches.push(w.name);
        }
        println!();
    }

    println!("======================================================================");
    println!("Audit summary");
    println!("======================================================================");
    println!("Total workloads:   {}", 13);
    println!("Matched canonical: {}", matches.len());
    println!("Mismatched:        {}", mismatches.len());
    if !mismatches.is_empty() {
        println!();
        println!("Mismatched workloads (default routing != canonical profile):");
        for (name, canonical) in &mismatches {
            println!("  - {}: should use {}", name, fmt_profile(*canonical));
        }
        println!();
        println!("Fix: use `JobPlan::set_profile(K, batch, DispatchProfile::*)` in");
        println!("the bench for these workloads, OR call `migrate_workload_class`");
        println!("with the appropriate class before constructing each plan.");
    }
}
