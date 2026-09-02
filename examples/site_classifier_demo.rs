//! Per-call-site classifier demo: two interleaved workloads with
//! opposite shapes each converge to their OWN learned class, with
//! zero cross-contamination between them.
//!
//! Run with:
//!   cargo run --release --example site_classifier_demo
//!
//! The demo attaches caller-owned `CallSiteState` statics via
//! `JobPlan::with_site` so it can read back what each site learned.
//! Production code normally skips this: every generic dispatch
//! entry automatically attaches the site resolved from the
//! caller's source location (`track_caller` chain).

use flynnel::{CallSiteState, JobPlan, SiteRef};

static LIGHT_SITE: CallSiteState = CallSiteState::new();
static HEAVY_SITE: CallSiteState = CallSiteState::new();

fn main() {
    println!("=== Per-call-site classifier demo ===\n");
    println!("Interleaving two opposite workload shapes for 8 rounds:");
    println!("  light: uniform byte increments over 1M u8 (streaming shape)");
    println!("  heavy: irregular sqrt chains, 64 items, 10x cost spread\n");

    let mut light_data = vec![0u8; 1 << 20];
    let mut heavy_data: Vec<f64> = (1..=64u32).map(f64::from).collect();

    for round in 1..=8u32 {
        let light_plan = JobPlan::new(3, light_data.len() as u32)
            .with_site(SiteRef::new(&LIGHT_SITE));
        flynnel::for_each_chunk(&light_plan, &mut light_data, |chunk| {
            for b in chunk.iter_mut() {
                *b = b.wrapping_add(1);
            }
        });

        let heavy_plan = JobPlan::new(6, heavy_data.len() as u32)
            .with_site(SiteRef::new(&HEAVY_SITE));
        flynnel::for_each_chunk(&heavy_plan, &mut heavy_data, |chunk| {
            for x in chunk.iter_mut() {
                // Irregular cost: item value drives the chain depth,
                // giving the high per-leaf variance heartbeat-shaped
                // workloads exhibit.
                let iters = 2_000 + (*x as u64 % 17) * 2_000;
                let mut v = *x;
                for _ in 0..iters {
                    v = v.sqrt() + 1.0;
                }
                *x = v;
            }
        });

        println!(
            "round {round}: light = {:?} (cv2 {:?}, {} leaves) | heavy = {:?} (cv2 {:?}, {} leaves)",
            LIGHT_SITE.learned_class(),
            LIGHT_SITE.cv2_per_mille(),
            LIGHT_SITE.leaf_count(),
            HEAVY_SITE.learned_class(),
            HEAVY_SITE.cv2_per_mille(),
            HEAVY_SITE.leaf_count(),
        );
    }

    let light = LIGHT_SITE.learned_class();
    let heavy = HEAVY_SITE.learned_class();
    println!("\nfinal: light site = {light:?}, heavy site = {heavy:?}");
    assert!(
        light.is_some() && heavy.is_some(),
        "both sites must have classified after 8 rounds"
    );
    assert_ne!(
        light, heavy,
        "opposite workload shapes must learn different classes"
    );
    println!("VERIFIED: sites classified independently with no cross-contamination.");
}
