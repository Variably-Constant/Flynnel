//! Print what the static classifier picks for each real_workloads bench.
//! Run with `cargo run --release --example _classifier_audit`.
//!
//! Compares the classifier's chosen WorkloadClass against the bench's
//! actual workload shape (input size vs slot count, per-slot cost).
//! Where the chosen class is wrong, that's a classifier-input bug
//! the bench is exposing: the JobPlan's batch_size + per_item_ns
//! describe the input population, not the actual dispatch shape.

use flynnel::sched::adaptive_profile::infer_class_static;

#[allow(non_snake_case)]
struct Cell {
    name: &'static str,
    k_outer: u8,
    batch_size: u32,
    ns: Option<u32>,
    // Actual dispatch shape that the parallel-for receives
    n_slots: u32,
    slot_kind: &'static str,
}

fn main() {
    let cells = [
        Cell { name: "blur",        k_outer: 6, batch_size: 1024*1024,  ns: None,     n_slots: 1024*1024, slot_kind: "byte" },
        Cell { name: "merkle",      k_outer: 6, batch_size: 16384,      ns: Some(600),n_slots: 16384,     slot_kind: "leaf-hash" },
        Cell { name: "grep",        k_outer: 6, batch_size: 16*1024*1024, ns: Some(1), n_slots: 16,       slot_kind: "1MB chunk" },
        Cell { name: "sort",        k_outer: 6, batch_size: 1_000_000,  ns: None,     n_slots: 0,         slot_kind: "join-recurse" },
        Cell { name: "pagerank",    k_outer: 6, batch_size: 10_000,     ns: Some(600),n_slots: 10_000,    slot_kind: "row" },
        Cell { name: "rle",         k_outer: 6, batch_size: 256,        ns: None,     n_slots: 256,       slot_kind: "64KB block" },
        Cell { name: "conway",      k_outer: 6, batch_size: 512*512,    ns: None,     n_slots: 512*512,   slot_kind: "byte" },
        Cell { name: "spmv",        k_outer: 6, batch_size: 100_000,    ns: Some(600),n_slots: 100_000,   slot_kind: "row" },
        Cell { name: "histogram_v15", k_outer: 6, batch_size: 16*1024*1024, ns: Some(1), n_slots: 16*1024*1024, slot_kind: "u32 item (reduce_chunks)" },
        Cell { name: "histogram_v14", k_outer: 6, batch_size: 16*1024*1024, ns: Some(1), n_slots: 16, slot_kind: "1MB chunk slot (PREV)" },
        Cell { name: "nbody",       k_outer: 6, batch_size: 1024,       ns: Some(1000),n_slots: 1024,     slot_kind: "particle" },
        Cell { name: "monte_carlo_pi", k_outer: 6, batch_size: 10_000_000, ns: Some(10), n_slots: 10,    slot_kind: "1M-sample slot" },
        Cell { name: "kmeans_v15",  k_outer: 6, batch_size: 100_000,    ns: Some(100),n_slots: 100_000*8, slot_kind: "f64 (reduce_chunks)" },
        Cell { name: "csv_scan",    k_outer: 6, batch_size: 16*1024*1024, ns: Some(10), n_slots: 64,     slot_kind: "256KB chunk slot" },
        Cell { name: "prefix_sum",  k_outer: 6, batch_size: 4*1024*1024, ns: Some(1),  n_slots: 4*1024*1024, slot_kind: "u64 slot" },
        Cell { name: "rgb_to_gray", k_outer: 6, batch_size: 4*1024*1024, ns: Some(5),  n_slots: 4*1024*1024, slot_kind: "byte" },
        Cell { name: "word_count_v15", k_outer: 6, batch_size: 16*1024*1024, ns: Some(1), n_slots: 16*1024*1024, slot_kind: "byte (reduce_chunks)" },
        Cell { name: "transpose",   k_outer: 6, batch_size: 2048*2048,  ns: Some(5),  n_slots: 2048*2048, slot_kind: "f64 slot" },
    ];

    println!("{:<22} {:>10} {:>10} {:>10} {:>14} {:>30} {}",
        "bench", "k_outer", "batch", "ns", "class", "n_slots", "slot_kind");
    println!("{}", "-".repeat(130));
    for c in cells.iter() {
        let class = infer_class_static(c.k_outer, c.batch_size, c.ns);
        let ns_str = c.ns.map(|n| n.to_string()).unwrap_or_else(|| "-".into());
        let mismatch = if c.n_slots > 0 && c.n_slots != c.batch_size {
            " <- MISMATCH: batch_size != n_slots"
        } else { "" };
        println!("{:<22} {:>10} {:>10} {:>10} {:>14?} {:>30} {}{}",
            c.name, c.k_outer, c.batch_size, ns_str, class, c.n_slots, c.slot_kind, mismatch);
    }
}
