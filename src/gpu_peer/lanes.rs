//! Lamport SPSC lanes over the shared region.
//!
//! One lane = one single-producer/single-consumer ring: the CPU
//! producer owns `head`, the consuming GPU block owns `tail`. Indices
//! are free-running u32 sequence numbers (wrapping compare via
//! `wrapping_sub`); slot addressing is `seq % slots_per_lane`. There
//! are NO atomics on the hot path in either direction - correctness
//! rests on single-writer indices plus release ordering (payload,
//! fence, index), which PCIe posted-write ordering and x86 store
//! ordering both provide. This is the protocol shape the probe suite
//! validated end-to-end with data verification.
//!
//! Backpressure/reuse: a slot is reusable only after the SUBMITTER has
//! reaped its result ([`LaneSet::reap`]), not merely after the GPU
//! consumed it - results are written in place, so `head - reaped <
//! slots_per_lane` is the submit gate. Reaping is in-order per lane
//! (it is an SPSC ring; out-of-order reaps would tear the gate).

use super::GpuPeerError;
use super::layout::{
    Geometry, LANE_HEAD_OFF, LANE_TAIL_OFF, SLOT_LEN_OFF, SLOT_OP_OFF, SLOT_PAYLOAD_OFF,
    SLOT_SEQ_OFF, SLOT_STATUS_OFF, STATUS_SUBMITTED,
};

/// Word-granular access to the shared region. Implemented by the real
/// CUDA-registered region and by a plain in-memory buffer for
/// CPU-only protocol tests.
pub trait RegionWords {
    /// The region's lane/slot geometry.
    fn geometry(&self) -> Geometry;
    /// Volatile u32 load at a byte offset.
    fn load_u32(&self, off: usize) -> u32;
    /// Volatile u32 store at a byte offset.
    fn store_u32(&self, off: usize, v: u32);
    /// Bulk payload write at a byte offset.
    fn write_bytes(&self, off: usize, src: &[u8]);
    /// Bulk payload read at a byte offset.
    fn read_bytes(&self, off: usize, dst: &mut [u8]);
    /// Full fence ordering payload writes before index publishes.
    fn release_fence(&self);
}

impl RegionWords for super::region::PeerRegion {
    #[inline]
    fn geometry(&self) -> Geometry {
        Self::geometry(self)
    }
    #[inline]
    fn load_u32(&self, off: usize) -> u32 {
        Self::load_u32(self, off)
    }
    #[inline]
    fn store_u32(&self, off: usize, v: u32) {
        Self::store_u32(self, off, v)
    }
    #[inline]
    fn write_bytes(&self, off: usize, src: &[u8]) {
        Self::write_bytes(self, off, src)
    }
    #[inline]
    fn read_bytes(&self, off: usize, dst: &mut [u8]) {
        Self::read_bytes(self, off, dst)
    }
    #[inline]
    fn release_fence(&self) {
        Self::release_fence(self)
    }
}

/// Completion ticket: which lane, which sequence number.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Ticket {
    /// Lane the submission went to.
    pub lane: u32,
    /// Sequence number within the lane.
    pub seq: u32,
}

/// Producer-side state for all lanes of a region. Single-producer by
/// construction: methods take `&mut self`.
pub struct LaneSet {
    geometry: Geometry,
    next_lane: u32,
    /// Next sequence to submit per lane (mirror of the shared head;
    /// the shared word is the published copy).
    head: Vec<u32>,
    /// First unreaped sequence per lane (CPU-private).
    reaped: Vec<u32>,
}

impl LaneSet {
    /// Fresh producer state for a region of the given geometry.
    pub fn new(geometry: Geometry) -> Self {
        Self {
            geometry,
            next_lane: 0,
            head: vec![0; geometry.lanes as usize],
            reaped: vec![0; geometry.lanes as usize],
        }
    }

    /// In-flight (submitted, not yet reaped) count for a lane.
    #[inline]
    pub fn in_flight(&self, lane: u32) -> u32 {
        self.head[lane as usize].wrapping_sub(self.reaped[lane as usize])
    }

    /// Total in-flight across lanes.
    pub fn in_flight_total(&self) -> u32 {
        (0..self.geometry.lanes).map(|l| self.in_flight(l)).sum()
    }

    /// Try to submit `payload` under `op` on the next round-robin lane
    /// with a free slot. Returns `None` when every lane is full
    /// (backpressure) or `Err` when the payload exceeds slot capacity.
    pub fn try_submit<R: RegionWords>(
        &mut self,
        r: &R,
        op: u32,
        payload: &[u8],
    ) -> Result<Option<Ticket>, GpuPeerError> {
        if payload.len() > self.geometry.payload_max() {
            return Err(GpuPeerError::PayloadTooLarge {
                len: payload.len(),
                max: self.geometry.payload_max(),
            });
        }
        let lanes = self.geometry.lanes;
        for probe in 0..lanes {
            let lane = (self.next_lane + probe) % lanes;
            if self.in_flight(lane) < self.geometry.slots_per_lane {
                self.next_lane = (lane + 1) % lanes;
                return Ok(Some(self.submit_on(r, lane, op, payload)));
            }
        }
        Ok(None)
    }

    /// Try to submit on a SPECIFIC lane (per-handle lane affinity:
    /// same-lane FIFO order is the dependency order for tasks
    /// touching one resident block). `None` = that lane is full.
    pub(crate) fn try_submit_on<R: RegionWords>(
        &mut self,
        r: &R,
        lane: u32,
        op: u32,
        payload: &[u8],
    ) -> Result<Option<Ticket>, GpuPeerError> {
        if payload.len() > self.geometry.payload_max() {
            return Err(GpuPeerError::PayloadTooLarge {
                len: payload.len(),
                max: self.geometry.payload_max(),
            });
        }
        if self.in_flight(lane) >= self.geometry.slots_per_lane {
            return Ok(None);
        }
        Ok(Some(self.submit_on(r, lane, op, payload)))
    }

    fn submit_on<R: RegionWords>(&mut self, r: &R, lane: u32, op: u32, payload: &[u8]) -> Ticket {
        let seq = self.head[lane as usize];
        let slot = self.geometry.slot_off(lane, seq);
        r.store_u32(slot + SLOT_OP_OFF, op);
        r.store_u32(slot + SLOT_LEN_OFF, payload.len() as u32);
        r.store_u32(slot + SLOT_SEQ_OFF, seq);
        r.store_u32(slot + SLOT_STATUS_OFF, STATUS_SUBMITTED);
        if !payload.is_empty() {
            r.write_bytes(slot + SLOT_PAYLOAD_OFF, payload);
        }
        // Release: everything above drains before the head store
        // publishes the slot to the consumer.
        r.release_fence();
        let new_head = seq.wrapping_add(1);
        self.head[lane as usize] = new_head;
        r.store_u32(self.geometry.lane_hdr_off(lane) + LANE_HEAD_OFF, new_head);
        Ticket { lane, seq }
    }

    /// Whether the consumer has finished `ticket`'s slot (tail passed
    /// the sequence). Reading `tail` is the acquire for the result
    /// bytes the consumer wrote before its release fence.
    #[inline]
    pub fn is_done<R: RegionWords>(&self, r: &R, t: Ticket) -> bool {
        let tail = r.load_u32(self.geometry.lane_hdr_off(t.lane) + LANE_TAIL_OFF);
        (tail.wrapping_sub(t.seq) as i32) > 0
    }

    /// Slot status word for a completed ticket.
    #[inline]
    pub fn status<R: RegionWords>(&self, r: &R, t: Ticket) -> u32 {
        r.load_u32(self.geometry.slot_off(t.lane, t.seq) + SLOT_STATUS_OFF)
    }

    /// Copy a completed ticket's result payload into `dst`.
    pub fn read_result<R: RegionWords>(&self, r: &R, t: Ticket, dst: &mut [u8]) {
        let slot = self.geometry.slot_off(t.lane, t.seq);
        r.read_bytes(slot + SLOT_PAYLOAD_OFF, dst);
    }

    /// Release `ticket`'s slot for reuse. In-order per lane: `ticket`
    /// must be the lane's oldest unreaped submission.
    pub fn reap(&mut self, t: Ticket) -> Result<(), GpuPeerError> {
        if self.reaped[t.lane as usize] != t.seq {
            return Err(GpuPeerError::ReapOutOfOrder {
                lane: t.lane,
                expected: self.reaped[t.lane as usize],
                got: t.seq,
            });
        }
        self.reaped[t.lane as usize] = t.seq.wrapping_add(1);
        Ok(())
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::gpu_peer::layout::{OP_ADD1_F32, STATUS_DONE};
    use std::cell::UnsafeCell;

    /// Plain-memory region for CPU-only protocol tests: same byte
    /// layout, no CUDA, interior mutability so a fake consumer can
    /// flip indices through a shared reference like the GPU does.
    pub(crate) struct MemRegion {
        buf: UnsafeCell<Vec<u8>>,
        g: Geometry,
    }
    // SAFETY: test-only; the SPSC discipline under test is exactly the
    // producer/consumer split that makes the shared access sound.
    unsafe impl Sync for MemRegion {}

    impl MemRegion {
        pub(crate) fn new(g: Geometry) -> Self {
            Self { buf: UnsafeCell::new(vec![0u8; g.region_bytes()]), g }
        }
        fn ptr(&self, off: usize) -> *mut u8 {
            // SAFETY: bounds enforced by callers via geometry math.
            unsafe { (*self.buf.get()).as_mut_ptr().add(off) }
        }
    }

    impl RegionWords for MemRegion {
        fn geometry(&self) -> Geometry {
            self.g
        }
        fn load_u32(&self, off: usize) -> u32 {
            // SAFETY: volatile word access, no references formed.
            unsafe { (self.ptr(off) as *const u32).read_volatile() }
        }
        fn store_u32(&self, off: usize, v: u32) {
            // SAFETY: as above.
            unsafe { (self.ptr(off) as *mut u32).write_volatile(v) }
        }
        fn write_bytes(&self, off: usize, src: &[u8]) {
            // SAFETY: as above.
            unsafe { core::ptr::copy_nonoverlapping(src.as_ptr(), self.ptr(off), src.len()) }
        }
        fn read_bytes(&self, off: usize, dst: &mut [u8]) {
            // SAFETY: as above.
            unsafe { core::ptr::copy_nonoverlapping(self.ptr(off), dst.as_mut_ptr(), dst.len()) }
        }
        fn release_fence(&self) {
            core::sync::atomic::fence(core::sync::atomic::Ordering::SeqCst);
        }
    }

    /// Consume every published slot once, like the poller kernel:
    /// read slot, mark DONE, bump tail.
    pub(crate) fn fake_consume_all(r: &MemRegion) {
        let g = r.geometry();
        for lane in 0..g.lanes {
            let hdr = g.lane_hdr_off(lane);
            let head = r.load_u32(hdr + LANE_HEAD_OFF);
            let mut tail = r.load_u32(hdr + LANE_TAIL_OFF);
            while tail != head {
                let slot = g.slot_off(lane, tail);
                r.store_u32(slot + SLOT_STATUS_OFF, STATUS_DONE);
                r.release_fence();
                tail = tail.wrapping_add(1);
                r.store_u32(hdr + LANE_TAIL_OFF, tail);
            }
        }
    }

    fn g4() -> Geometry {
        Geometry { lanes: 2, slot_bytes: 256, slots_per_lane: 4 }
    }

    #[test]
    fn submit_publishes_descriptor_and_head() {
        let r = MemRegion::new(g4());
        let mut ls = LaneSet::new(g4());
        let t = ls
            .try_submit(&r, OP_ADD1_F32, &[1, 2, 3, 4])
            .expect("payload fits")
            .expect("lane free");
        assert_eq!(t, Ticket { lane: 0, seq: 0 });
        let slot = g4().slot_off(0, 0);
        assert_eq!(r.load_u32(slot + SLOT_OP_OFF), OP_ADD1_F32);
        assert_eq!(r.load_u32(slot + SLOT_LEN_OFF), 4);
        assert_eq!(r.load_u32(g4().lane_hdr_off(0) + LANE_HEAD_OFF), 1);
        assert!(!ls.is_done(&r, t), "not done until the consumer moves tail");
    }

    #[test]
    fn backpressure_gates_on_reap_not_on_consume() {
        let r = MemRegion::new(g4());
        let mut ls = LaneSet::new(g4());
        // Fill both lanes completely: 2 lanes x 4 slots.
        let mut tickets = Vec::new();
        for _ in 0..8 {
            tickets.push(
                ls.try_submit(&r, OP_ADD1_F32, &[0u8; 8]).expect("fits").expect("free"),
            );
        }
        assert!(ls.try_submit(&r, OP_ADD1_F32, &[0u8; 8]).expect("fits").is_none());
        // GPU consuming does NOT free slots for reuse...
        fake_consume_all(&r);
        assert!(ls.try_submit(&r, OP_ADD1_F32, &[0u8; 8]).expect("fits").is_none());
        // ...reaping does.
        assert!(ls.is_done(&r, tickets[0]));
        assert_eq!(ls.status(&r, tickets[0]), STATUS_DONE);
        ls.reap(tickets[0]).expect("in order");
        assert!(ls.try_submit(&r, OP_ADD1_F32, &[0u8; 8]).expect("fits").is_some());
    }

    #[test]
    fn reap_enforces_lane_order() {
        let r = MemRegion::new(g4());
        let mut ls = LaneSet::new(g4());
        let t0 = ls.try_submit(&r, OP_ADD1_F32, &[]).expect("fits").expect("free");
        let t1 = ls.try_submit(&r, OP_ADD1_F32, &[]).expect("fits").expect("free");
        // t0 lane 0, t1 lane 1 (round-robin); same-lane order enforced:
        let t2 = ls.try_submit(&r, OP_ADD1_F32, &[]).expect("fits").expect("free");
        assert_eq!(t2.lane, t0.lane);
        fake_consume_all(&r);
        assert!(ls.reap(t2).is_err(), "seq 1 before seq 0 must be rejected");
        ls.reap(t0).expect("oldest first");
        ls.reap(t2).expect("now in order");
        ls.reap(t1).expect("other lane independent");
    }

    #[test]
    fn oversized_payload_is_rejected() {
        let r = MemRegion::new(g4());
        let mut ls = LaneSet::new(g4());
        let big = vec![0u8; g4().payload_max() + 1];
        assert!(matches!(
            ls.try_submit(&r, OP_ADD1_F32, &big),
            Err(GpuPeerError::PayloadTooLarge { .. })
        ));
    }

    #[test]
    fn sequence_wraparound_keeps_flowing() {
        let r = MemRegion::new(g4());
        let mut ls = LaneSet::new(g4());
        // Drive one lane far past u32 slot counts by cycling
        // submit -> consume -> reap many times.
        let mut last = None;
        for _ in 0..1000 {
            let t = ls.try_submit(&r, OP_ADD1_F32, &[0u8; 4]).expect("fits").expect("free");
            fake_consume_all(&r);
            assert!(ls.is_done(&r, t));
            ls.reap(t).expect("in order");
            last = Some(t);
        }
        let last = last.expect("looped");
        assert_eq!(ls.in_flight_total(), 0);
        // 1000 submissions over 2 lanes -> seq advanced to 500 per lane.
        assert_eq!(last.seq, 499);
    }
}
