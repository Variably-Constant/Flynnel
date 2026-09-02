//! One model for data residence AND execution side.
//!
//! [`hybrid_auto_resident`] routes each call through the SAME
//! per-call-site, per-size-bucket placement EWMAs the crate-level
//! learned hybrid uses - but over a [`MirrorBuf`], whose bytes live
//! on the host, on the device (resident block), or both. Any
//! transfer a placement flip requires (fetch before CPU work on a
//! device-valid mirror, re-upload before device work on a
//! host-valid mirror) executes INSIDE the timed section, so the
//! model prices data movement and execution together. Sticky
//! residence is an emergent property of measurement: flip-flopping
//! pays the transfer every time and loses the EWMA comparison, so
//! the model parks the data where the work wins. No separate
//! residency planner exists, and none is needed.

use std::time::Instant;

use crate::sched::call_site::{Placement, caller_site};
use crate::sched::plan::JobPlan;

use super::{GpuPeer, GpuPeerError, ResidentHandle, STATUS_DONE, layout};

/// Which copy of a [`MirrorBuf`] currently holds the truth.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MirrorState {
    /// Host bytes are authoritative; device copy stale or absent.
    Host,
    /// Device block is authoritative; host bytes stale.
    Device,
    /// Both copies agree.
    Both,
}

/// A buffer whose residence the placement model controls: host bytes
/// plus an optional device-resident twin.
pub struct MirrorBuf {
    host: Vec<u8>,
    handle: Option<ResidentHandle>,
    state: MirrorState,
}

impl MirrorBuf {
    /// Start host-resident.
    pub fn new(data: Vec<u8>) -> Self {
        Self { host: data, handle: None, state: MirrorState::Host }
    }

    /// Where the truth currently lives.
    pub fn state(&self) -> MirrorState {
        self.state
    }

    /// Host view; synchronizes from the device first when the device
    /// copy is authoritative.
    pub fn host_bytes<'a>(&'a mut self, peer: &mut GpuPeer) -> Result<&'a [u8], GpuPeerError> {
        if self.state == MirrorState::Device {
            let h = self.handle.as_ref().expect("device-valid implies handle");
            let mut out = vec![0u8; h.len()];
            peer.fetch(h, &mut out)?;
            self.host = out;
            self.state = MirrorState::Both;
        }
        Ok(&self.host)
    }

    /// Drop the device twin (host copy made authoritative first).
    pub fn evict(&mut self, peer: &mut GpuPeer) -> Result<(), GpuPeerError> {
        if self.state == MirrorState::Device {
            self.host_bytes(peer)?;
        }
        if let Some(h) = self.handle.take() {
            peer.unpin(h)?;
        }
        self.state = MirrorState::Host;
        Ok(())
    }

    fn ensure_device(&mut self, peer: &mut GpuPeer) -> Result<(), GpuPeerError> {
        match (self.state, self.handle.is_some()) {
            (MirrorState::Host, false) => {
                self.handle = Some(peer.pin(&self.host)?);
                self.state = MirrorState::Both;
            }
            (MirrorState::Host, true) => {
                let h = *self.handle.as_ref().expect("checked");
                peer.write_resident(&h, &self.host)?;
                self.state = MirrorState::Both;
            }
            _ => {}
        }
        Ok(())
    }

    fn ensure_host(&mut self, peer: &mut GpuPeer) -> Result<(), GpuPeerError> {
        if self.state == MirrorState::Device {
            self.host_bytes(peer)?;
        }
        Ok(())
    }
}

/// Run one step of a repeated workload over `mirror`, choosing CPU
/// or device execution - and therefore residence - from the call
/// site's learned placement model. `cpu_impl` mutates the host bytes
/// in place; `op` is the equivalent resident opcode (built-in `_V`
/// or user op). Returns the placement taken.
///
/// Race placements run BOTH sides on equal inputs and keep the CPU
/// result (matching the crate-level learned hybrid), recording both
/// measured costs - transfers included - into the site's EWMAs.
#[track_caller]
pub fn hybrid_auto_resident<C>(
    plan: &JobPlan,
    peer: &mut GpuPeer,
    mirror: &mut MirrorBuf,
    op: u32,
    mut cpu_impl: C,
) -> Result<Placement, GpuPeerError>
where
    C: FnMut(&mut [u8]),
{
    let plan_owned = plan.with_site_if_none(caller_site());
    let site = plan_owned.site.expect("attached above").get();
    let batch = plan_owned.batch_size;

    let run_cpu = |peer: &mut GpuPeer, mirror: &mut MirrorBuf, cpu_impl: &mut C| -> Result<u64, GpuPeerError> {
        let t0 = Instant::now();
        mirror.ensure_host(peer)?;          // transfer priced in
        cpu_impl(&mut mirror.host);
        mirror.state = MirrorState::Host;   // device copy now stale
        Ok(t0.elapsed().as_nanos() as u64)
    };
    let run_dev = |peer: &mut GpuPeer, mirror: &mut MirrorBuf| -> Result<u64, GpuPeerError> {
        let t0 = Instant::now();
        mirror.ensure_device(peer)?;        // transfer priced in
        let h = *mirror.handle.as_ref().expect("ensured");
        let t = peer.submit_resident(op, &h)?;
        let s = peer.wait(t, std::time::Duration::from_secs(10))?;
        peer.reap(t)?;
        if s != STATUS_DONE {
            return Err(GpuPeerError::Unavailable("resident op rejected"));
        }
        mirror.state = MirrorState::Device; // host copy now stale
        Ok(t0.elapsed().as_nanos() as u64)
    };

    match site.choose_placement(batch) {
        Placement::Cpu => {
            let ns = run_cpu(peer, mirror, &mut cpu_impl)?;
            site.record_placement(batch, Some(ns), None);
            Ok(Placement::Cpu)
        }
        Placement::Backend => {
            let ns = run_dev(peer, mirror)?;
            site.record_placement(batch, None, Some(ns));
            Ok(Placement::Backend)
        }
        Placement::Race => {
            // Both sides transform the SAME input exactly once: the
            // input is normalized to the host (untimed), snapshotted,
            // the device leg runs from it (upload priced into its
            // timing), the CPU leg runs on the snapshot, and the CPU
            // result stays authoritative - the device application is
            // discarded with the stale device copy.
            mirror.ensure_host(peer)?;
            let mut snapshot = mirror.host.clone();
            let dev_ns = run_dev(peer, mirror)?;
            let t0 = Instant::now();
            cpu_impl(&mut snapshot);
            let cpu_ns = t0.elapsed().as_nanos() as u64;
            mirror.host = snapshot;
            mirror.state = MirrorState::Host;
            site.record_placement(batch, Some(cpu_ns), Some(dev_ns));
            Ok(Placement::Race)
        }
    }
}

/// Convenience: the resident opcode paired with a matching CPU
/// closure for the built-in ADD1 op (demo/test symmetry helper).
pub fn add1_f32_cpu(bytes: &mut [u8]) {
    for c in bytes.chunks_exact_mut(4) {
        let v = f32::from_le_bytes([c[0], c[1], c[2], c[3]]) + 1.0;
        c.copy_from_slice(&v.to_le_bytes());
    }
}

/// The matching resident opcode for [`add1_f32_cpu`].
pub const ADD1_F32_RESIDENT: u32 = layout::OP_ADD1_F32_V;
