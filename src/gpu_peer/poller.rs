//! Bounded-quantum persistent poller lifecycle.
//!
//! The consumer kernel runs for at most a quantum (watchdog-safe on
//! display-attached GPUs), exits when idle, and is relaunched on
//! demand. "Running vs parked" is tracked WITHOUT stream queries:
//! every exiting block bumps a device-scope-atomic exit counter in
//! the region header (exact among GPU threads), so
//! `exits / lanes == launches` means every launched quantum has fully
//! drained. A generation word invalidates stragglers: a relaunch
//! bumps the active generation and any block from an older launch
//! exits at its next poll, so two generations never both consume.
//!
//! Wake-from-idle therefore costs one kernel launch (the calibrated
//! `launch_ns`, ~tens of microseconds); a continuously fed queue
//! never pays it because the resident quantum keeps consuming.

use std::sync::Arc;

use cudarc::driver::sys as cu;
use cudarc::driver::{CudaFunction, CudaStream, LaunchConfig, PushKernelArg};

use super::GpuPeerError;
use super::layout::{HDR_ACTIVE_GEN_OFF, HDR_EXITS_OFF, HDR_STOP_OFF};
use super::region::PeerRegion;

pub struct Poller {
    stream: Arc<CudaStream>,
    func: CudaFunction,
    /// Quanta launched so far; also the current generation tag.
    launches: u32,
    quantum_ns: u64,
    idle_exit_ns: u64,
    /// Resident-pool geometry passed to every quantum (base 0 = none).
    vram_base: u64,
    vram_block_bytes: u32,
    vram_blocks: u32,
    /// Blocks serving each lane. Above 1, a lane is worked by a team
    /// of consecutive blocks: rank 0 owns the ring and the descriptor,
    /// every rank runs the user op over its share of the work. This is
    /// what lets a doorbell op use the whole device rather than the
    /// single SM one block occupies.
    blocks_per_lane: u32,
    /// While paused, the poller holds no resident quantum and
    /// `ensure_running` will not launch one - the device is left free
    /// for a heavy wide op. Cleared by `resume`.
    paused: bool,
}

impl Poller {
    #[expect(
        clippy::too_many_arguments,
        reason = "internal constructor mirroring the kernel's parameter list; a config struct would restate GpuPeerConfig"
    )]
    pub fn new(
        stream: Arc<CudaStream>,
        func: CudaFunction,
        quantum_ns: u64,
        idle_exit_ns: u64,
        vram_base: u64,
        vram_block_bytes: u32,
        vram_blocks: u32,
        blocks_per_lane: u32,
    ) -> Self {
        Self {
            stream,
            func,
            launches: 0,
            quantum_ns,
            idle_exit_ns,
            vram_base,
            vram_block_bytes,
            vram_blocks,
            blocks_per_lane: blocks_per_lane.max(1),
            paused: false,
        }
    }

    /// Number of fully drained quanta according to the exit counter.
    fn completed_launches(&self, region: &PeerRegion) -> u32 {
        // Every block of every lane's team increments the exit
        // counter, so a drained quantum is lanes * blocks_per_lane
        // exits, not lanes.
        let blocks = region.geometry().lanes.max(1) * self.blocks_per_lane.max(1);
        region.load_u32(HDR_EXITS_OFF) / blocks
    }

    /// Launch a new quantum when no quantum is resident. Callers
    /// invoke this after every submit and periodically while waiting,
    /// which closes the exit-vs-new-work race: a quantum that idled
    /// out just before a submit is simply relaunched by that submit.
    pub fn ensure_running(&mut self, region: &PeerRegion) -> Result<(), GpuPeerError> {
        if self.paused {
            return Ok(()); // held off so a wide op owns the device
        }
        if self.completed_launches(region) < self.launches {
            return Ok(()); // a quantum is still resident (or draining)
        }
        let generation = self.launches.wrapping_add(1);
        self.launches = generation;
        region.store_u32(HDR_ACTIVE_GEN_OFF, generation);
        region.release_fence();

        let g = region.geometry();
        let dev_base = region.dev_base();
        let lanes = g.lanes;
        let slot_bytes = g.slot_bytes;
        let slots_per_lane = g.slots_per_lane;
        let quantum = self.quantum_ns;
        let idle = self.idle_exit_ns;
        let mut b = self.stream.launch_builder(&self.func);
        b.arg(&dev_base);
        b.arg(&lanes);
        b.arg(&slot_bytes);
        b.arg(&slots_per_lane);
        b.arg(&quantum);
        b.arg(&idle);
        b.arg(&generation);
        b.arg(&self.vram_base);
        b.arg(&self.vram_block_bytes);
        b.arg(&self.vram_blocks);
        b.arg(&self.blocks_per_lane);
        // SAFETY: argument types match flynnel_peer_poller(u8*, u32,
        // u32, u32, u64, u64, u32, u8*, u32, u32, u32); the grid gives
        // each lane its team of consecutive blocks, which is what the
        // kernel's lane = blockIdx.x / blocks_per_lane assumes;
        // dev_base is the live registered mapping and vram_base the
        // live (or absent = 0) pool.
        unsafe {
            b.launch(LaunchConfig {
                grid_dim: (lanes * self.blocks_per_lane, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map_err(|e| GpuPeerError::Driver(format!("poller launch: {e:?}")))?;
        // Push the buffered launch to the GPU now; the caller is
        // about to spin on results.
        // SAFETY: valid stream; NOT_READY is the benign busy answer.
        let _rc: cu::CUresult = unsafe { cu::cuStreamQuery(self.stream.cu_stream()) };
        Ok(())
    }

    /// Quiesce the poller: force any resident quantum to exit and hold
    /// off relaunches until [`Self::resume`]. The kernel busy-polls
    /// while resident, so a live poller steals SM occupancy and L2
    /// bandwidth from a concurrent wide op; pausing it hands the whole
    /// device to that op. Syncs the stream so the exit is observed
    /// (the SMs are actually free) before returning.
    pub fn pause(&mut self, region: &PeerRegion) -> Result<(), GpuPeerError> {
        if self.paused {
            return Ok(());
        }
        region.store_u32(HDR_STOP_OFF, 1);
        region.release_fence();
        self.stream
            .synchronize()
            .map_err(|e| GpuPeerError::Driver(format!("poller pause sync: {e:?}")))?;
        // Clear the stop flag now that the quantum has drained, so a
        // later resume + relaunch starts clean.
        region.store_u32(HDR_STOP_OFF, 0);
        region.release_fence();
        self.paused = true;
        Ok(())
    }

    /// Undo [`Self::pause`]. The next submit relaunches a fresh
    /// quantum via `ensure_running` (a new generation supersedes any
    /// straggler, and the exit counter already matched `launches`
    /// after the paused drain).
    pub fn resume(&mut self) {
        self.paused = false;
    }

    /// Whether the poller is currently held off.
    #[inline]
    pub fn is_paused(&self) -> bool {
        self.paused
    }

    /// Stop consumption and drain the stream. A sync failure during
    /// teardown is unactionable; the stop flag alone already ends the
    /// resident quantum.
    pub fn shutdown(&mut self, region: &PeerRegion) {
        region.store_u32(HDR_STOP_OFF, 1);
        region.release_fence();
        self.stream.synchronize().ok();
    }
}
