//! Fischer timed lock: cross-device mutual exclusion with NO atomic
//! read-modify-write - only plain stores, bounded delays, and a
//! verified visibility bound.
//!
//! Protocol (identical on both sides; the GPU contender lives in
//! `kernels/gpu_peer.cu` as `flynnel_peer_fischer`):
//!
//! ```text
//! loop {
//!     wait until x == 0
//!     x = me;  fence
//!     delay(Delta)            // Delta > one-way visibility bound
//!     if x == me { break }    // silence for Delta == exclusivity
//! }
//! critical section
//! x = 0;  fence
//! ```
//!
//! Correct exactly when every store becomes visible to the other side
//! within Delta. Delta is NOT a compile-time constant: it is derived
//! from the host's measured one-way latency + clock error at init and
//! then VALIDATED by a real cross-device contention self-test before
//! the `timed_lock_ok` capability is granted (see the calibration
//! module). The occupancy word doubles as a violation detector: under
//! a correct mutex it is never observed at any value other than 1
//! inside the critical section.

use std::time::{Duration, Instant};

use super::layout::{HDR_FISCHER_CS_OFF, HDR_FISCHER_VIOL_OFF, HDR_FISCHER_X_OFF};
use super::region::PeerRegion;

/// CPU contender identity in the lock word (zero means free).
pub const OWNER_CPU: u32 = 1;
/// GPU contender identity in the lock word (zero means free).
pub const OWNER_GPU: u32 = 2;

/// Busy-wait for `d` (sub-millisecond precision matters here; a
/// sleeping wait would blow straight past Delta).
#[inline]
pub(crate) fn spin_for(d: Duration) {
    let t0 = Instant::now();
    while t0.elapsed() < d {
        core::hint::spin_loop();
    }
}

/// CPU-side acquire of the region's Fischer word. Returns `false` on
/// timeout (lock never observed free / claim never survived Delta).
pub fn acquire(region: &PeerRegion, delta: Duration, timeout: Duration) -> bool {
    acquire_with_contention(region, delta, timeout).is_some()
}

/// [`acquire`] that also reports whether the round CONTENDED (saw the
/// lock held, or lost a claim recheck). `None` = timeout. The
/// calibration self-test uses the contention evidence: a pass in
/// which neither side ever contended proves nothing about mutual
/// exclusion.
pub(crate) fn acquire_with_contention(
    region: &PeerRegion,
    delta: Duration,
    timeout: Duration,
) -> Option<bool> {
    let t0 = Instant::now();
    let mut contended = false;
    loop {
        if t0.elapsed() > timeout {
            return None;
        }
        while region.load_u32(HDR_FISCHER_X_OFF) != 0 {
            contended = true;
            if t0.elapsed() > timeout {
                return None;
            }
            core::hint::spin_loop();
        }
        region.store_u32(HDR_FISCHER_X_OFF, OWNER_CPU);
        region.release_fence();
        spin_for(delta);
        if region.load_u32(HDR_FISCHER_X_OFF) == OWNER_CPU {
            return Some(contended);
        }
        contended = true;
    }
}

/// CPU-side release.
pub fn release(region: &PeerRegion) {
    region.release_fence();
    region.store_u32(HDR_FISCHER_X_OFF, 0);
    region.release_fence();
}

/// Enter the critical section with the occupancy violation detector
/// (calibration self-test path): increments the shared occupancy word,
/// checks it reads exactly 1, holds for `cs_hold`, decrements.
pub(crate) fn critical_section_checked(region: &PeerRegion, cs_hold: Duration) {
    let cs = region.load_i32(HDR_FISCHER_CS_OFF);
    region.store_i32(HDR_FISCHER_CS_OFF, cs + 1);
    region.release_fence();
    if region.load_i32(HDR_FISCHER_CS_OFF) != 1 {
        let v = region.load_u32(HDR_FISCHER_VIOL_OFF);
        region.store_u32(HDR_FISCHER_VIOL_OFF, v + 1);
    }
    spin_for(cs_hold);
    let cs = region.load_i32(HDR_FISCHER_CS_OFF);
    region.store_i32(HDR_FISCHER_CS_OFF, cs - 1);
    region.release_fence();
}
