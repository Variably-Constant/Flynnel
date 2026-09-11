//! Shared-region byte layout for the GPU-peer substrate.
//!
//! The GPU side of this contract lives in `kernels/gpu_peer.cu` as
//! `#define` offsets; the values here have to stay in lockstep with that
//! file (the unit tests at the bottom pin every one of them). All
//! cross-device state is addressed by byte offset from the region
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
/// Reserved header word (u32). The poller counts block exits per lane
/// at [`HDR_LANE_EXITS_OFF`].
pub const HDR_EXITS_OFF: usize = 0x05C;
/// Reserved header word (u32). The poller's generation tag is per lane
/// at [`HDR_LANE_GEN_OFF`].
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
/// Count of barrier expiries: block teams that did not fully arrive
/// before rank 0's deadline (u32, device-scope atomic).
///
/// Read with [`super::GpuPeer::barrier_stalls`]. Zero on any run where
/// no team missed its deadline, which is every run on an unloaded host
/// observed so far.
pub const HDR_STALL_COUNT_OFF: usize = 0x150;
/// Largest ring depth (`head - tail`) observed at a barrier expiry
/// (u32, device-scope atomic maximum).
///
/// This separates the two mechanisms that can produce a stall. A team
/// split by a quantum boundary landing between two ranks' clock reads
/// happens on whatever the lane was holding, so a trickle shows one or
/// two. A lane relaunched late and draining a backlog claims from a
/// full ring, so it shows a depth near `slots_per_lane`. The count says
/// stalls happened; this says which kind.
pub const HDR_STALL_MAX_DEPTH_OFF: usize = 0x154;
/// Longest time rank 0 spent at the team barrier on a slot the whole
/// team did reach, in nanoseconds (u32, device-scope atomic maximum).
///
/// This is the margin the deadline is spending. The deadline is a whole
/// quantum, chosen as the figure already to hand rather than measured;
/// this says how much of it a healthy team actually needs. A deadline
/// can safely be shortened to some multiple of this and no further,
/// and shortening it is what decides how long a caller waits before
/// being told a team was lost.
pub const HDR_BARRIER_WAIT_MAX_OFF: usize = 0x158;
/// Calibration globaltimer samples (`u64[GTS_SLOTS]`).
pub const HDR_GTS_OFF: usize = 0x180;
/// Calibration timestamp slots (`u64[GTS_SLOTS]` at [`HDR_GTS_OFF`]).
pub const GTS_SLOTS: usize = 400;
/// Per-lane poller block-exit counters, device-scope atomic
/// (`u32[MAX_POLLER_LANES]`). A lane's quantum is drained when its
/// counter has advanced by the lane's block team size, which is what
/// lets one lane relaunch while another is still working.
pub const HDR_LANE_EXITS_OFF: usize = 0xE00;
/// Per-lane active poller generation (`u32[MAX_POLLER_LANES]`). A
/// straggler exits at its next poll when an older launch of its own
/// lane is superseded; one lane never supersedes another.
pub const HDR_LANE_GEN_OFF: usize = 0xF00;
/// Lanes addressable by the two per-lane arrays above, which occupy
/// the header from [`HDR_LANE_EXITS_OFF`] to [`HDR_BYTES`].
pub const MAX_POLLER_LANES: usize = 64;
/// Total header reservation.
pub const HDR_BYTES: usize = 0x1000;

/// Byte offset of `lane`'s exit counter.
#[inline]
pub const fn lane_exits_off(lane: u32) -> usize {
    HDR_LANE_EXITS_OFF + (lane as usize) * 4
}

/// Byte offset of `lane`'s active generation word.
#[inline]
pub const fn lane_gen_off(lane: u32) -> usize {
    HDR_LANE_GEN_OFF + (lane as usize) * 4
}

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
/// Opcode: add 1.0 to every f32 of the named resident block. The
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
/// Return value from `flynnel_user_op` that keeps the slot in the ring
/// instead of retiring it (`FLYNNEL_USER_YIELD` in the kernel).
///
/// The poller runs the same slot again on its next pass, after its stop,
/// generation and quantum checks, so an op can pace itself across passes
/// and quanta while its state stays in VRAM. Only rank 0's thread 0
/// decides. A yielded slot reports [`STATUS_SUBMITTED`] until an op run
/// retires it, and runs only while its lane has a resident quantum;
/// [`super::GpuPeer::wait_status`] relaunches the lane as it waits.
pub const USER_OP_YIELD: u32 = 0x5949_4C44;
/// Opcode of Flynnel's own wave calibration op (`FLW_OP_CALIBRATE` in the
/// kernel), dispatched ahead of the user hook when user ops are composed.
/// It lies in the user range so every rank of a team runs it; a user op
/// must not use this value.
pub const OP_WAVE_CALIBRATE: u32 = 0xFFFF_FF00;

/// Status: slot published, not yet consumed.
pub const STATUS_SUBMITTED: u32 = 0;
/// Status: consumer completed the operation.
pub const STATUS_DONE: u32 = 1;
/// Status: consumer rejected the descriptor (unknown op / bad len), or
/// the user op reported failure.
pub const STATUS_ERR: u32 = 2;
/// Status: a block team did not fully arrive before rank 0's deadline,
/// so the slot was retired without every rank's contribution.
///
/// Distinct from [`STATUS_ERR`] because the two want different
/// responses: a lost rank may be fixed by a retry or a smaller
/// `blocks_per_lane`, while an op reporting failure will not be. Only
/// reachable when `blocks_per_lane > 1`.
///
/// A caller testing `!= STATUS_DONE` treats this as a failure without
/// change; one that wants the distinction reads the raw word from
/// [`super::GpuPeer::wait_status`].
pub const STATUS_TEAM_INCOMPLETE: u32 = 3;

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

    /// Status words are `#define`d in kernels/gpu_peer.cu and read back
    /// here from the slot descriptor, so a value that drifts on one side
    /// is a caller misreading an outcome with nothing to say so. The
    /// offsets below are pinned for the same reason; these were not.
    #[test]
    fn status_words_match_the_kernel_defines() {
        assert_eq!(STATUS_SUBMITTED, 0);
        assert_eq!(STATUS_DONE, 1);
        assert_eq!(STATUS_ERR, 2);
        assert_eq!(STATUS_TEAM_INCOMPLETE, 3);
        // FLYNNEL_USER_YIELD in the kernel. It is a return value, not a
        // status, and must differ from 0 (success) and from the small
        // codes consumers already return for failure.
        assert_eq!(USER_OP_YIELD, 0x5949_4C44);
        // FLW_OP_CALIBRATE in the kernel: inside the user range, so every
        // rank runs it, and distinct from the ~0 the kernel uses for a
        // refused op.
        assert_eq!(OP_WAVE_CALIBRATE, 0xFFFF_FF00);
        #[expect(clippy::assertions_on_constants, reason = "guard over const opcodes")]
        {
            assert!(OP_WAVE_CALIBRATE >= OP_USER_BASE && OP_WAVE_CALIBRATE != u32::MAX);
        }

        // Each outcome must be its own word: a caller distinguishing a
        // lost rank from a failed op can only do so while these differ.
        let all = [STATUS_SUBMITTED, STATUS_DONE, STATUS_ERR, STATUS_TEAM_INCOMPLETE];
        for (i, a) in all.iter().enumerate() {
            for b in &all[i + 1..] {
                assert_ne!(a, b, "two status words collide");
            }
        }
    }

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
        assert_eq!(HDR_LANE_EXITS_OFF, 0xE00);
        assert_eq!(HDR_LANE_GEN_OFF, 0xF00);
        assert_eq!(HDR_BYTES, 0x1000);
        assert_eq!(LANE_STRIDE, 0x100);
        assert_eq!(LANE_TAIL_OFF, 0x40);
        assert_eq!(SLOT_PAYLOAD_OFF, 0x10);
        // gts array must fit inside the header reservation. The
        // operands are compile-time constants; the assert exists to
        // fail the test when someone edits one of them out of range.
        #[expect(clippy::assertions_on_constants, reason = "layout guard over const offsets")]
        {
            assert!(HDR_GTS_OFF + GTS_SLOTS * 8 <= HDR_LANE_EXITS_OFF);
            assert!(HDR_LANE_EXITS_OFF + MAX_POLLER_LANES * 4 <= HDR_LANE_GEN_OFF);
            assert!(HDR_LANE_GEN_OFF + MAX_POLLER_LANES * 4 <= HDR_BYTES);
        }
    }

    #[test]
    fn per_lane_poller_words_are_distinct_and_in_range() {
        assert_eq!(lane_exits_off(0), HDR_LANE_EXITS_OFF);
        assert_eq!(lane_gen_off(0), HDR_LANE_GEN_OFF);
        assert_eq!(lane_exits_off(3), HDR_LANE_EXITS_OFF + 12);
        assert_eq!(lane_gen_off(3), HDR_LANE_GEN_OFF + 12);
        let last = (MAX_POLLER_LANES - 1) as u32;
        assert!(lane_exits_off(last) + 4 <= HDR_LANE_GEN_OFF);
        assert!(lane_gen_off(last) + 4 <= HDR_BYTES);
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
