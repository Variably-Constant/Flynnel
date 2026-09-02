//! Multi-peer group: N GPU peers under ONE handle namespace, or used
//! individually - both from the same type.
//!
//! Each peer owns its own region, lanes, calibration, and resident
//! pool (per-device constants stay per-device: a group spanning a
//! PCIe card and a coherent-link card carries BOTH calibrations).
//! The group adds placement across peers: unified `pin` picks the
//! peer with the most free blocks, `migrate` moves a resident block
//! between peers through the host bridge (fetch from the source
//! region, pin into the destination), and every [`GroupHandle`]
//! names its peer so routing is explicit and cheap.

use std::time::Duration;

use super::lanes::Ticket;
use super::{GpuPeer, GpuPeerConfig, GpuPeerError, ResidentHandle};

/// A resident handle qualified by which peer holds it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GroupHandle {
    /// Index into the group's peer list.
    pub peer: usize,
    /// The peer-local handle.
    pub handle: ResidentHandle,
}

/// N peers, one scheduler surface.
pub struct PeerGroup {
    peers: Vec<GpuPeer>,
}

impl PeerGroup {
    /// Initialize one peer per config (e.g. one per device ordinal).
    /// Fails if ANY peer fails - a group with silently missing
    /// members would skew placement.
    pub fn init(configs: Vec<GpuPeerConfig>) -> Result<Self, GpuPeerError> {
        let mut peers = Vec::with_capacity(configs.len());
        for c in configs {
            peers.push(GpuPeer::init(c)?);
        }
        if peers.is_empty() {
            return Err(GpuPeerError::Unavailable("empty peer group"));
        }
        Ok(Self { peers })
    }

    /// Peer count.
    pub fn len(&self) -> usize {
        self.peers.len()
    }
    /// True when the group has no peers (never constructed so).
    pub fn is_empty(&self) -> bool {
        self.peers.is_empty()
    }
    /// Direct access to a member peer (individual-use mode).
    pub fn peer_mut(&mut self, idx: usize) -> &mut GpuPeer {
        &mut self.peers[idx]
    }
    /// Member calibrations (per-device constants differ by design).
    pub fn calibrations(&self) -> Vec<super::PeerCalibration> {
        self.peers.iter().map(|p| p.calibration()).collect()
    }

    /// Unified pin: place on the peer with the most free pool blocks.
    pub fn pin(&mut self, data: &[u8]) -> Result<GroupHandle, GpuPeerError> {
        let peer = (0..self.peers.len())
            .max_by_key(|&i| self.peers[i].pool_stats().0)
            .unwrap_or(0);
        let handle = self.peers[peer].pin(data)?;
        Ok(GroupHandle { peer, handle })
    }

    /// Submit a resident task on the owning peer.
    pub fn submit_resident(
        &mut self,
        op: u32,
        h: &GroupHandle,
    ) -> Result<(usize, Ticket), GpuPeerError> {
        Ok((h.peer, self.peers[h.peer].submit_resident(op, &h.handle)?))
    }

    /// Wait + reap a ticket on its peer; returns the status word.
    pub fn wait_reap(
        &mut self,
        peer: usize,
        t: Ticket,
        timeout: Duration,
    ) -> Result<u32, GpuPeerError> {
        let s = self.peers[peer].wait(t, timeout)?;
        self.peers[peer].reap(t)?;
        Ok(s)
    }

    /// Download a group handle's block.
    pub fn fetch(&mut self, h: &GroupHandle, out: &mut [u8]) -> Result<(), GpuPeerError> {
        self.peers[h.peer].fetch(&h.handle, out)
    }

    /// Release a group handle's block.
    pub fn unpin(&mut self, h: GroupHandle) -> Result<(), GpuPeerError> {
        self.peers[h.peer].unpin(h.handle)
    }

    /// Move a resident block to another peer through the host
    /// bridge: fetch from the source region, pin into the
    /// destination, release the source block. Returns the new
    /// handle.
    pub fn migrate(
        &mut self,
        h: GroupHandle,
        to_peer: usize,
    ) -> Result<GroupHandle, GpuPeerError> {
        if to_peer >= self.peers.len() {
            return Err(GpuPeerError::Unavailable("migrate target out of range"));
        }
        if to_peer == h.peer {
            return Ok(h);
        }
        let mut buf = vec![0u8; h.handle.len()];
        self.peers[h.peer].fetch(&h.handle, &mut buf)?;
        let new_handle = self.peers[to_peer].pin(&buf)?;
        self.peers[h.peer].unpin(h.handle)?;
        Ok(GroupHandle { peer: to_peer, handle: new_handle })
    }
}
