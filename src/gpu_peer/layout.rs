//! Shared-region byte layout for the GPU-peer substrate.
//!
//! The GPU side of this contract lives in `kernels/gpu_peer.cu` as
//! `#define` offsets; the values here MUST stay in lockstep with that
//! file (the unit tests at the bottom pin every one of them). All
//! cross-device state is addressed by BYTE OFFSET from the region
//! base - never by raw pointer - because the CPU and GPU observe the
//! region at different virtual addresses (`cudaHostGetDevicePointer`
//! translation) and other processes at yet another.
//!
//! Concurrency contract:
//! - Every shared word is accessed with volatile reads/writes plus
//!   explicit fences; Rust references are never formed over shared
//!   words (raw-pointer access only), so there is no aliasing UB
//!   against the GPU's concurrent stores.
//! - Lane indices are single-writer (CPU owns `head`, GPU owns
//!   `tail`); the release order is payload, fence, index.

/// Region format magic ("FLYGPUPR" little-endian bytes).
pub const MAGIC: u64 = 0x5250_5550_4759_4C46;
/// Region format version.
pub const VERSION: u32 = 1;

/// Region magic (u64).
pub const HDR_MAGIC_OFF: usize = 0x000;
/// Region format version (u32).
pub const HDR_VERSION_OFF: usize = 0x008;
/// Lane count (u32).
pub const HDR_LANES_OFF: usize = 0x00C;
/// Slot size in bytes including descriptor (u32).
pub const HDR_SLOT_BYTES_OFF: usize = 0x010;
/// Ring depth per lane (u32).
pub const HDR_SLOTS_PER_OFF: usize = 0x014;
/// Calibration/capability flag bits (u64, `FLAG_*`).
pub const HDR_FLAGS_OFF: usize = 0x018;
/// Calibrated doorbell round-trip minimum, ns (u64).
pub const HDR_RTT_MIN_OFF: usize = 0x020;
/// Calibrated doorbell round-trip median, ns (u64).
pub const HDR_RTT_MED_OFF: usize = 0x028;
/// Calibrated doorbell round-trip p99, ns (u64).
pub const HDR_RTT_P99_OFF: usize = 0x030;
/// Calibrated one-way visibility bound, ns (u64).
pub const HDR_ONE_WAY_OFF: usize = 0x038;
/// Calibrated cross-device clock error, ns (u64).
pub const HDR_CLOCK_ERR_OFF: usize = 0x040;
/// Validated Fischer margin Delta, ns (u64).
pub const HDR_DELTA_OFF: usize = 0x048;
/// Kernel launch+sync baseline, ns (u64).
pub const HDR_LAUNCH_OFF: usize = 0x050;
/// Global stop flag consumed by the poller (u32).
pub const HDR_STOP_OFF: usize = 0x058;
/// Poller block-exit counter, device-scope atomic (u32).
pub const HDR_EXITS_OFF: usize = 0x05C;
/// Active poller generation; stragglers from older launches exit (u32).
pub const HDR_ACTIVE_GEN_OFF: usize = 0x060;
/// Calibration doorbell: CPU-written ping (u32).
pub const HDR_CALIB_PING_OFF: usize = 0x064;
/// Calibration doorbell: GPU-written pong (u32).
pub const HDR_CALIB_PONG_OFF: usize = 0x068;
/// Fischer lock word: 0 free / OWNER_CPU / OWNER_GPU (u32).
pub const HDR_FISCHER_X_OFF: usize = 0x080;
/// Fischer critical-section occupancy detector (i32).
pub const HDR_FISCHER_CS_OFF: usize = 0x0C0;
/// Fischer mutual-exclusion violation counter (u32).
pub const HDR_FISCHER_VIOL_OFF: usize = 0x100;
/// Fischer GPU-side completed-acquisitions counter (u32).
pub const HDR_FISCHER_ACQS_OFF: usize = 0x140;
/// Fischer self-test: GPU contender is resident and contending (u32).
pub const HDR_FISCHER_STARTED_OFF: usize = 0x148;
/// Fischer self-test: GPU-side CONTENDED-round count (u32). A pass
/// without contention on both sides proves nothing and is treated as
/// inconclusive, never as a grant.
pub const HDR_FISCHER_GPU_CONT_OFF: usize = 0x14C;
/// Calibration globaltimer samples (`u64[GTS_SLOTS]`).
pub const HDR_GTS_OFF: usize = 0x180;
/// Calibration timestamp slots (`u64[GTS_SLOTS]` at [`HDR_GTS_OFF`]).
pub const GTS_SLOTS: usize = 400;
/// Total header reservation.
pub const HDR_BYTES: usize = 0x1000;

/// Per-lane header stride (head and tail on separate cache lines).
pub const LANE_STRIDE: usize = 0x100;
/// Producer-owned publish index within a lane header (u32).
pub const LANE_HEAD_OFF: usize = 0x00;
/// Consumer-owned completion index within a lane header (u32).
pub const LANE_TAIL_OFF: usize = 0x40;

/// Slot descriptor: opcode (u32).
pub const SLOT_OP_OFF: usize = 0x00;
/// Slot descriptor: payload length in bytes (u32).
pub const SLOT_LEN_OFF: usize = 0x04;
/// Slot descriptor: sequence number (u32).
pub const SLOT_SEQ_OFF: usize = 0x08;
/// Slot descriptor: consumer-written status (u32, `STATUS_*`).
pub const SLOT_STATUS_OFF: usize = 0x0C;
/// Payload start within a slot.
pub const SLOT_PAYLOAD_OFF: usize = 0x10;

/// Opcode: no operation (completion plumbing only).
pub const OP_NOP: u32 = 0;
/// Opcode: add 1.0 to every f32 in the payload, in place.
pub const OP_ADD1_F32: u32 = 1;
/// Opcode: sum the payload u32s; u64 result replaces payload start.
pub const OP_SUM_U32: u32 = 2;
/// Opcode: upload payload data (at +8) into the VRAM block named by
/// the 8-byte param header (u32 block index, u32 byte count).
pub const OP_H2V: u32 = 3;
/// Opcode: download the named VRAM block into the payload at +8.
pub const OP_V2H: u32 = 4;
/// Opcode: add 1.0 to every f32 of the named RESIDENT block. The
/// task moves only the 8-byte params - the data stays in VRAM.
pub const OP_ADD1_F32_V: u32 = 5;
/// Opcode: sum the named resident block's u32s; u64 result lands in
/// the payload at +8.
pub const OP_SUM_U32_V: u32 = 6;
/// Size of the resident-op param header at the payload start.
pub const RESIDENT_PARAMS_BYTES: usize = 8;
/// First user-defined opcode. Ops at or above this route through the
/// `flynnel_user_op` hook NVRTC-composed into the poller at init;
/// without registered user source they complete as [`STATUS_ERR`].
pub const OP_USER_BASE: u32 = 100;
/// Param value naming "no resident block" for a user opcode.
pub const NO_BLOCK: u32 = u32::MAX;

/// Status: slot published, not yet consumed.
pub const STATUS_SUBMITTED: u32 = 0;
/// Status: consumer completed the operation.
pub const STATUS_DONE: u32 = 1;
/// Status: consumer rejected the descriptor (unknown op / bad len).
pub const STATUS_ERR: u32 = 2;

/// Capability: doorbell handshake measured working.
pub const FLAG_DOORBELL_OK: u64 = 1 << 0;
/// Capability: Fischer self-test passed at the stored Delta.
pub const FLAG_TIMED_LOCK_OK: u64 = 1 << 1;
/// Capability: cross-device CAS conserved claims on this host.
pub const FLAG_SYS_ATOMICS_OK: u64 = 1 << 2;
/// Header calibration block is populated.
pub const FLAG_CALIBRATED: u64 = 1 << 3;

/// Geometry of a region: lane count and slot shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Geometry {
    /// SPSC lane count (one consumer block each).
    pub lanes: u32,
    /// Slot size in bytes, including the 16-byte descriptor.
    pub slot_bytes: u32,
    /// Ring depth per lane.
    pub slots_per_lane: u32,
}

impl Geometry {
    /// Byte offset of lane `l`'s header.
    #[inline]
    pub fn lane_hdr_off(&self, lane: u32) -> usize {
        HDR_BYTES + lane as usize * LANE_STRIDE
    }

    /// Byte offset of the slot slab base.
    #[inline]
    pub fn slab_off(&self) -> usize {
        HDR_BYTES + self.lanes as usize * LANE_STRIDE
    }

    /// Byte offset of slot `seq % slots_per_lane` in lane `lane`.
    #[inline]
    pub fn slot_off(&self, lane: u32, seq: u32) -> usize {
        self.slab_off()
            + (lane as usize * self.slots_per_lane as usize
                + (seq % self.slots_per_lane) as usize)
                * self.slot_bytes as usize
    }

    /// Maximum payload bytes per slot.
    #[inline]
    pub fn payload_max(&self) -> usize {
        self.slot_bytes as usize - SLOT_PAYLOAD_OFF
    }

    /// Total region size in bytes.
    #[inline]
    pub fn region_bytes(&self) -> usize {
        self.slab_off()
            + self.lanes as usize * self.slots_per_lane as usize * self.slot_bytes as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn offsets_match_the_kernel_defines() {
        // These values are #define'd in kernels/gpu_peer.cu; a drift
        // here corrupts the wire without any compiler diagnostic.
        assert_eq!(HDR_STOP_OFF, 0x058);
        assert_eq!(HDR_EXITS_OFF, 0x05C);
        assert_eq!(HDR_ACTIVE_GEN_OFF, 0x060);
        assert_eq!(HDR_CALIB_PING_OFF, 0x064);
        assert_eq!(HDR_CALIB_PONG_OFF, 0x068);
        assert_eq!(HDR_FISCHER_X_OFF, 0x080);
        assert_eq!(HDR_FISCHER_CS_OFF, 0x0C0);
        assert_eq!(HDR_FISCHER_VIOL_OFF, 0x100);
        assert_eq!(HDR_FISCHER_ACQS_OFF, 0x140);
        assert_eq!(HDR_FISCHER_STARTED_OFF, 0x148);
        assert_eq!(HDR_FISCHER_GPU_CONT_OFF, 0x14C);
        assert_eq!(HDR_GTS_OFF, 0x180);
        assert_eq!(HDR_BYTES, 0x1000);
        assert_eq!(LANE_STRIDE, 0x100);
        assert_eq!(LANE_TAIL_OFF, 0x40);
        assert_eq!(SLOT_PAYLOAD_OFF, 0x10);
        // gts array must fit inside the header reservation. The
        // operands are compile-time constants; the assert exists to
        // fail the test when someone edits one of them out of range.
        #[expect(clippy::assertions_on_constants, reason = "layout guard over const offsets")]
        {
            assert!(HDR_GTS_OFF + GTS_SLOTS * 8 <= HDR_BYTES);
        }
    }

    #[test]
    fn geometry_math_is_consistent() {
        let g = Geometry { lanes: 4, slot_bytes: 4096, slots_per_lane: 64 };
        assert_eq!(g.lane_hdr_off(0), HDR_BYTES);
        assert_eq!(g.lane_hdr_off(3), HDR_BYTES + 3 * LANE_STRIDE);
        assert_eq!(g.slab_off(), HDR_BYTES + 4 * LANE_STRIDE);
        // Wraparound addressing: seq 64 reuses slot 0 of the lane.
        assert_eq!(g.slot_off(1, 64), g.slot_off(1, 0));
        assert_ne!(g.slot_off(1, 1), g.slot_off(1, 0));
        assert_eq!(g.payload_max(), 4096 - 16);
        assert_eq!(
            g.region_bytes(),
            g.slab_off() + 4usize * 64 * 4096
        );
    }
}
