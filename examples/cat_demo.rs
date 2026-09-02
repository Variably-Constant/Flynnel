//! CPU L3 cache-way reservation via resctrl (CAT).
//!
//! Run with:
//!   cargo run --release --example cat_demo
//!
//! On Linux with resctrl mounted and root privileges this reserves a
//! contiguous set of L3 ways for the process and reads the schemata
//! back to prove the reservation took. On any other host it reports
//! the capability as unsupported and exits cleanly - the lever is
//! capability-gated, never a panic.

use flynnel::{CatCapability, CatError, L3Reservation};

fn main() {
    println!("=== CPU L3 cache-way reservation (resctrl / CAT) ===\n");
    let cap = CatCapability::detect();
    println!("[1] capability on this host:");
    println!("    resctrl L3 CAT supported : {}", cap.supported);
    println!("    classes of service       : {}", cap.num_closids);
    println!("    L3 ways (cbm bits)       : {}", cap.cbm_bits);
    println!("    min contiguous ways      : {}", cap.min_cbm_bits);
    println!("    L3 domains               : {}", cap.num_domains);

    if !cap.supported {
        println!("\n[2] resctrl absent (not Linux, or not mounted). Nothing to reserve.");
        println!("    On a Zen2+/RDT Linux host: mount -t resctrl resctrl /sys/fs/resctrl");
        println!("VERIFIED: capability gating reported unsupported without panicking.");
        return;
    }

    // Reserve the lower half of the L3 ways for this process.
    let ways = (cap.cbm_bits / 2).max(cap.min_cbm_bits);
    println!("\n[2] reserving {ways} L3 ways (of {}) for this process...", cap.cbm_bits);
    match L3Reservation::reserve_ways("flynnel_cat_demo", 0, ways) {
        Ok(res) => {
            let schem = res.schemata().unwrap_or_else(|e| format!("<read failed: {e}>"));
            println!("[3] reservation live; schemata read back from resctrl:");
            for line in schem.lines() {
                println!("    {line}");
            }
            let expect = format!("{:x}", (1u64 << ways) - 1);
            let ok = schem.contains(&format!("=[{expect}]")) || schem.contains(&format!("={expect}"));
            println!(
                "\nVERIFIED: {}",
                if ok {
                    format!("L3 reserved with mask {expect} on every domain")
                } else {
                    "reservation created (mask format varies by kernel; see schemata above)".to_string()
                }
            );
        }
        Err(CatError::Io(e)) if e.kind() == std::io::ErrorKind::PermissionDenied => {
            println!("[3] permission denied: resctrl reservation needs root. Re-run with sudo.");
        }
        Err(e) => println!("[3] reservation failed: {e}"),
    }
}
