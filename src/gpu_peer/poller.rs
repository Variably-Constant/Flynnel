//! Bounded-quantum persistent poller lifecycle.
//!
//! The consumer kernel runs for at most a quantum (watchdog-safe on
//! display-attached GPUs), exits when idle, and is relaunched on
//! demand. Running versus parked is tracked per lane and without
//! stream queries: every exiting block bumps its own lane's
//! device-scope-atomic exit counter in the region header (exact among
//! GPU threads), so a lane whose counter has advanced by its team size
//! has fully drained and may be relaunched. Each lane is launched as
//! its own grid, so a lane inside a long op does not hold a parked
//! lane's relaunch behind it.
//!
//! A lane is never served by two teams at once, and that is the
//! property the per-lane counter exists to hold. The ring tail
//! advances only when an op retires (`gpu_peer.cu`, at the end of the
//! slot body), so a second team on the same lane would read the same
//! unretired slot and run it a second time. The per-lane generation
//! word catches a straggler that outlives its launch, but it cannot
//! prevent that double read on its own: a block already inside an op
//! is past the generation check until the op returns.
//!
//! Wake-from-idle therefore costs one kernel launch (the calibrated
//! `launch_ns`, ~tens of microseconds); a continuously fed queue
//! never pays it because the resident quantum keeps consuming.

use std::sync::Arc;

use cudarc::driver::sys as cu;
use cudarc::driver::{CudaFunction, CudaStream, LaunchConfig, PushKernelArg};

use super::GpuPeerError;
use super::layout::{HDR_STOP_OFF, lane_exits_off, lane_gen_off};
use super::region::PeerRegion;

pub struct Poller {
    /// One stream per lane, indexed by lane. Work on a single stream
    /// runs in sequence, so lanes share none.
    streams: Vec<Arc<CudaStream>>,
    func: CudaFunction,
    /// Quanta launched so far per lane; each entry is also that lane's
    /// current generation tag. Grown to the region's lane count on
    /// first use.
    launches: Vec<u32>,
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
    /// How long rank 0 waits for its team before retiring the slot as
    /// incomplete. Separate from the quantum: a healthy team assembles
    /// in microseconds, so tying this to the quantum made a caller wait
    /// a quarter second to learn a team was lost.
    barrier_deadline_ns: u64,
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
        streams: Vec<Arc<CudaStream>>,
        func: CudaFunction,
        quantum_ns: u64,
        idle_exit_ns: u64,
        vram_base: u64,
        vram_block_bytes: u32,
        vram_blocks: u32,
        blocks_per_lane: u32,
        barrier_deadline_ns: u64,
    ) -> Self {
        Self {
            streams,
            func,
            launches: Vec::new(),
            quantum_ns,
            idle_exit_ns,
            vram_base,
            vram_block_bytes,
            vram_blocks,
            blocks_per_lane: blocks_per_lane.max(1),
            barrier_deadline_ns: barrier_deadline_ns.max(1),
            paused: false,
        }
    }

    /// Quanta of `lane` that have fully drained, according to that
    /// lane's exit counter.
    fn completed_launches(&self, region: &PeerRegion, lane: u32) -> u32 {
        // Every rank of the lane's team increments the lane's own
        // counter, so a drained quantum is blocks_per_lane exits.
        region.load_u32(lane_exits_off(lane)) / self.blocks_per_lane.max(1)
    }

    /// Launch a new quantum for `lane` when none is resident on it.
    /// Callers invoke this after every submit and periodically while
    /// waiting, which closes the exit-vs-new-work race: a quantum that
    /// idled out just before a submit is simply relaunched by that
    /// submit.
    ///
    /// Lanes are launched and counted independently, so a lane inside a
    /// long op does not hold up a parked lane's relaunch. A grid serves
    /// one lane's team, and a lane never has two teams at once because
    /// its own previous team must have drained before this relaunches
    /// it.
    pub fn ensure_running(
        &mut self,
        region: &PeerRegion,
        lane: u32,
    ) -> Result<(), GpuPeerError> {
        if self.paused {
            return Ok(()); // held off so a wide op owns the device
        }
        let lane_count = region.geometry().lanes.max(1);
        if lane >= lane_count {
            return Err(GpuPeerError::Unavailable("lane beyond the region geometry"));
        }
        if self.launches.len() < lane_count as usize {
            self.launches.resize(lane_count as usize, 0);
        }
        let launched = self.launches[lane as usize];
        if self.completed_launches(region, lane) < launched {
            return Ok(()); // a quantum is still resident (or draining)
        }
        let generation = launched.wrapping_add(1);
        region.store_u32(lane_gen_off(lane), generation);
        region.release_fence();

        let g = region.geometry();
        let dev_base = region.dev_base();
        let lanes = g.lanes;
        let slot_bytes = g.slot_bytes;
        let slots_per_lane = g.slots_per_lane;
        let quantum = self.quantum_ns;
        let idle = self.idle_exit_ns;
        let stream = match self.streams.get(lane as usize) {
            Some(s) => Arc::clone(s),
            None => return Err(GpuPeerError::Unavailable("no stream for lane")),
        };
        let mut b = stream.launch_builder(&self.func);
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
        let barrier_deadline = self.barrier_deadline_ns;
        b.arg(&barrier_deadline);
        b.arg(&self.blocks_per_lane);
        b.arg(&lane);
        // SAFETY: argument types match flynnel_peer_poller(u8*, u32,
        // u32, u32, u64, u64, u32, u8*, u32, u32, u64, u32, u32); the grid
        // is one lane's team of consecutive blocks, which is what the
        // kernel's lane = lane_base + blockIdx.x / blocks_per_lane
        // assumes; dev_base is the live registered mapping and
        // vram_base the live (or absent = 0) pool.
        unsafe {
            b.launch(LaunchConfig {
                grid_dim: (self.blocks_per_lane, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map_err(|e| GpuPeerError::Driver(format!("poller launch: {e:?}")))?;
        // Counted only once the driver accepted the launch: a failed
        // one leaves no team to drain, and a lane whose count ran ahead
        // of its exits would never be relaunched.
        self.launches[lane as usize] = generation;
        // Push the buffered launch to the GPU now; the caller is
        // about to spin on results.
        // SAFETY: valid stream; NOT_READY is the benign busy answer.
        let _rc: cu::CUresult = unsafe { cu::cuStreamQuery(stream.cu_stream()) };
        Ok(())
    }

    /// Ensure every lane has a resident quantum. For callers with no
    /// single lane in view, such as a submit waiting on whichever ring
    /// drains first.
    pub fn ensure_running_all(&mut self, region: &PeerRegion) -> Result<(), GpuPeerError> {
        for lane in 0..region.geometry().lanes.max(1) {
            self.ensure_running(region, lane)?;
        }
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
        for (lane, s) in self.streams.iter().enumerate() {
            s.synchronize()
                .map_err(|e| GpuPeerError::Driver(format!("poller pause sync lane {lane}: {e:?}")))?;
        }
        // Clear the stop flag now that the quantum has drained, so a
        // later resume + relaunch starts clean.
        region.store_u32(HDR_STOP_OFF, 0);
        region.release_fence();
        self.paused = true;
        Ok(())
    }

    /// Undo [`Self::pause`]. The next submit relaunches that lane via
    /// `ensure_running`; the paused drain left every lane's exit
    /// counter matching its launch count, so each lane is eligible.
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
        for (lane, s) in self.streams.iter().enumerate() {
            if let Err(e) = s.synchronize() {
                // Teardown has nothing left to try, and the stop flag
                // has already ended the resident quantum, so this is
                // reported rather than propagated.
                eprintln!("flynnel gpu_peer: lane {lane} shutdown sync failed: {e:?}");
            }
        }
    }
}
