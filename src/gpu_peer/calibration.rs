//! Host calibration: every timing constant the substrate relies on is
//! MEASURED on the running machine at init, validated by live
//! self-tests, and stored in the region header so attaching processes
//! inherit the same numbers. Nothing here is baked from any reference
//! box: a host with a coherent CPU-GPU link measures a smaller
//! visibility bound (and may even pass the system-atomics probe), and
//! every derived constant tightens automatically.
//!
//! Measured quantities:
//! - doorbell round-trip (min / median / p99) against a resident
//!   kernel: the substrate's signalling latency;
//! - CPU/GPU clock offset error via paired QPC/globaltimer sampling
//!   (integer-domain differencing; best-RTT-quartile spread);
//! - launch+sync baseline: the cost a wake-from-idle pays and the
//!   number the doorbell path is beating;
//! - visibility bound `L = p99/2 + clock_err`, Fischer margin
//!   `Delta = clamp(10 x L, 5us ..= 100us)` - then VALIDATED by a
//!   real CPU-vs-GPU contention run (escalating x2 on violation);
//! - system-atomics conservation probe deciding `sys_atomics_ok`.

use std::sync::Arc;
use std::time::{Duration, Instant};

use cudarc::driver::sys as cu;
use cudarc::driver::{CudaFunction, CudaStream, LaunchConfig, PushKernelArg};

use super::GpuPeerError;
use super::layout::{
    self, FLAG_CALIBRATED, FLAG_DOORBELL_OK, FLAG_SYS_ATOMICS_OK, FLAG_TIMED_LOCK_OK, GTS_SLOTS,
    HDR_CALIB_PING_OFF, HDR_CALIB_PONG_OFF, HDR_FISCHER_ACQS_OFF, HDR_FISCHER_CS_OFF,
    HDR_FISCHER_VIOL_OFF, HDR_FISCHER_X_OFF, HDR_GTS_OFF,
};
use super::region::PeerRegion;
use super::timed_lock;

/// Host-measured constants + capability flags for one (host, GPU,
/// driver) combination.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PeerCalibration {
    /// Doorbell round-trip, minimum observed.
    pub rtt_min_ns: u64,
    /// Doorbell round-trip, median.
    pub rtt_median_ns: u64,
    /// Doorbell round-trip, 99th percentile.
    pub rtt_p99_ns: u64,
    /// One-way visibility bound (p99 / 2 + clock error).
    pub one_way_ns: u64,
    /// Cross-device clock alignment error (best-quartile spread / 2).
    pub clock_err_ns: u64,
    /// Fischer margin actually validated by the self-test.
    pub delta_ns: u64,
    /// Kernel launch + synchronize baseline (median).
    pub launch_ns: u64,
    /// Doorbell handshake completed and measured.
    pub doorbell_ok: bool,
    /// Fischer self-test passed at `delta_ns` (0 violations).
    pub timed_lock_ok: bool,
    /// Cross-device CAS conserved claims (coherent-link hosts).
    pub sys_atomics_ok: bool,
    /// Evidence behind `timed_lock_ok`: CPU-side contended rounds in
    /// the granting self-test run (of 150).
    pub lock_cpu_contended: u32,
    /// Evidence behind `timed_lock_ok`: GPU-side contended rounds in
    /// the granting self-test run (of 150).
    pub lock_gpu_contended: u32,
}

impl PeerCalibration {
    /// Persist into the region header (other processes attach and
    /// read these instead of re-calibrating).
    pub fn store(&self, r: &PeerRegion) {
        r.store_u64(layout::HDR_RTT_MIN_OFF, self.rtt_min_ns);
        r.store_u64(layout::HDR_RTT_MED_OFF, self.rtt_median_ns);
        r.store_u64(layout::HDR_RTT_P99_OFF, self.rtt_p99_ns);
        r.store_u64(layout::HDR_ONE_WAY_OFF, self.one_way_ns);
        r.store_u64(layout::HDR_CLOCK_ERR_OFF, self.clock_err_ns);
        r.store_u64(layout::HDR_DELTA_OFF, self.delta_ns);
        r.store_u64(layout::HDR_LAUNCH_OFF, self.launch_ns);
        let mut flags = FLAG_CALIBRATED;
        if self.doorbell_ok {
            flags |= FLAG_DOORBELL_OK;
        }
        if self.timed_lock_ok {
            flags |= FLAG_TIMED_LOCK_OK;
        }
        if self.sys_atomics_ok {
            flags |= FLAG_SYS_ATOMICS_OK;
        }
        r.store_u64(layout::HDR_FLAGS_OFF, flags);
    }
}

/// Derive the Fischer margin from a measured visibility bound.
/// Clamped: below 5us the OS scheduler jitter on the CPU side eats
/// the margin; above 100us the lock is slower than useful and the
/// capability is refused instead.
pub fn derive_delta_ns(one_way_ns: u64) -> u64 {
    (one_way_ns.saturating_mul(10)).clamp(5_000, 100_000)
}

/// The kernel handles calibration needs.
pub struct CalibKernels {
    /// Doorbell ping-pong + globaltimer sampler.
    pub calib_pong: CudaFunction,
    /// GPU-side Fischer contender for the timed-lock self-test.
    pub fischer: CudaFunction,
    /// `atomicCAS_system` claim racer for the atomics probe.
    pub cas_probe: CudaFunction,
}

fn one_one() -> LaunchConfig {
    LaunchConfig { grid_dim: (1, 1, 1), block_dim: (1, 1, 1), shared_mem_bytes: 0 }
}

/// Force submission of buffered work without blocking (the WDDM
/// queue holds launches back until a flush; a resident spinning
/// kernel must actually reach the GPU before the CPU pings it).
fn flush_stream(stream: &Arc<CudaStream>) {
    // SAFETY: valid stream handle; NOT_READY is the expected benign
    // result for a stream with a resident kernel.
    let _rc: cu::CUresult = unsafe { cu::cuStreamQuery(stream.cu_stream()) };
}

fn now_ns(t0: Instant) -> u64 {
    t0.elapsed().as_nanos() as u64
}

/// Run the full calibration. `region` must be freshly created (lane
/// traffic has not started).
pub fn calibrate(
    region: &PeerRegion,
    stream: &Arc<CudaStream>,
    kernels: &CalibKernels,
) -> Result<PeerCalibration, GpuPeerError> {
    let dev_base = region.dev_base();

    // ---- launch+sync baseline (rounds = 0 exits immediately) ----
    let mut launch_samples = Vec::with_capacity(32);
    for _ in 0..32 {
        let zero_rounds: u32 = 0;
        let t0 = Instant::now();
        let mut b = stream.launch_builder(&kernels.calib_pong);
        b.arg(&dev_base);
        b.arg(&zero_rounds);
        // SAFETY: argument types match the kernel signature
        // (unsigned char* base, u32 rounds); dev_base is the live
        // registered mapping.
        unsafe { b.launch(one_one()) }
            .map_err(|e| GpuPeerError::Driver(format!("calib launch: {e:?}")))?;
        stream
            .synchronize()
            .map_err(|e| GpuPeerError::Driver(format!("calib sync: {e:?}")))?;
        launch_samples.push(now_ns(t0));
    }
    launch_samples.sort_unstable();
    let launch_ns = launch_samples[launch_samples.len() / 2];

    // ---- doorbell RTT + clock pairing ----
    let rounds: u32 = (GTS_SLOTS - 8) as u32;
    region.store_u32(HDR_CALIB_PING_OFF, 0);
    region.store_u32(HDR_CALIB_PONG_OFF, 0);
    let mut b = stream.launch_builder(&kernels.calib_pong);
    b.arg(&dev_base);
    b.arg(&rounds);
    // SAFETY: as above.
    unsafe { b.launch(one_one()) }
        .map_err(|e| GpuPeerError::Driver(format!("pong launch: {e:?}")))?;
    flush_stream(stream);

    let wall = Instant::now();
    let mut rtt = Vec::with_capacity(rounds as usize);
    let mut pairs: Vec<(u64, u64, u64)> = Vec::with_capacity(rounds as usize); // (rtt, gts, mid)
    let mut doorbell_ok = true;
    for i in 1..=rounds {
        let t0 = now_ns(wall);
        region.store_u32(HDR_CALIB_PING_OFF, i);
        region.release_fence();
        loop {
            if region.load_u32(HDR_CALIB_PONG_OFF) == i {
                break;
            }
            if now_ns(wall).saturating_sub(t0) > 2_000_000_000 {
                doorbell_ok = false;
                break;
            }
            core::hint::spin_loop();
        }
        if !doorbell_ok {
            break;
        }
        let t1 = now_ns(wall);
        let gts = region.load_u64(HDR_GTS_OFF + (i as usize) * 8);
        rtt.push(t1 - t0);
        pairs.push((t1 - t0, gts, t0 + (t1 - t0) / 2));
    }
    stream
        .synchronize()
        .map_err(|e| GpuPeerError::Driver(format!("pong drain: {e:?}")))?;

    if !doorbell_ok || rtt.len() < 16 {
        // Without a doorbell there is no substrate; report what we
        // know and refuse the capabilities.
        let cal = PeerCalibration {
            rtt_min_ns: 0,
            rtt_median_ns: 0,
            rtt_p99_ns: 0,
            one_way_ns: 0,
            clock_err_ns: 0,
            delta_ns: 0,
            launch_ns,
            doorbell_ok: false,
            timed_lock_ok: false,
            sys_atomics_ok: false,
            lock_cpu_contended: 0,
            lock_gpu_contended: 0,
        };
        cal.store(region);
        return Ok(cal);
    }

    rtt.sort_unstable();
    let rtt_min_ns = rtt[0];
    let rtt_median_ns = rtt[rtt.len() / 2];
    let rtt_p99_ns = rtt[(rtt.len() * 99) / 100];

    // Clock error: difference the (gts - mid) offsets against the
    // first sample in the INTEGER domain (both clocks carry
    // epoch-sized magnitudes a double cannot hold to ns precision),
    // then take the spread over the best-RTT quartile.
    pairs.sort_unstable_by_key(|p| p.0);
    let q = (pairs.len() / 4).clamp(4, pairs.len());
    let (_, g0, m0) = pairs[0];
    let mut mn = i64::MAX;
    let mut mx = i64::MIN;
    for &(_, g, m) in pairs.iter().take(q) {
        let off = (g.wrapping_sub(g0) as i64) - (m.wrapping_sub(m0) as i64);
        mn = mn.min(off);
        mx = mx.max(off);
    }
    let clock_err_ns = ((mx - mn) / 2).max(1) as u64;

    let one_way_ns = rtt_p99_ns / 2 + clock_err_ns;
    let mut delta_ns = derive_delta_ns(one_way_ns);

    // ---- Fischer self-test at the derived Delta ----
    // Same grant discipline as the atomics flag: a violation
    // escalates Delta x2; a run without contention evidence is
    // INCONCLUSIVE and retried, never counted as a pass; the grant
    // requires two consecutive contended zero-violation runs.
    let mut timed_lock_ok = false;
    let mut passes = 0u32;
    let mut lock_cpu_contended = 0u32;
    let mut lock_gpu_contended = 0u32;
    for _attempt in 0..6 {
        match fischer_self_test(region, stream, kernels, delta_ns)? {
            SelfTest::Pass { cpu_contended, gpu_contended } => {
                passes += 1;
                lock_cpu_contended = cpu_contended;
                lock_gpu_contended = gpu_contended;
                if passes >= 2 {
                    timed_lock_ok = true;
                    break;
                }
            }
            SelfTest::Violated => {
                passes = 0;
                delta_ns = (delta_ns * 2).min(100_000);
            }
            SelfTest::Inconclusive => {
                passes = 0;
            }
        }
    }

    // ---- system-atomics conservation probe ----
    // The flag UNLOCKS protocols, so a false positive is worse than a
    // false negative: grant only on two consecutive fully-contended
    // conserving runs.
    let sys_atomics_ok = cas_conservation_probe(region, stream, kernels)?
        && cas_conservation_probe(region, stream, kernels)?;

    let cal = PeerCalibration {
        rtt_min_ns,
        rtt_median_ns,
        rtt_p99_ns,
        one_way_ns,
        clock_err_ns,
        delta_ns,
        launch_ns,
        doorbell_ok: true,
        timed_lock_ok,
        sys_atomics_ok,
        lock_cpu_contended,
        lock_gpu_contended,
    };
    cal.store(region);
    Ok(cal)
}

/// Outcome of one Fischer self-test run.
enum SelfTest {
    /// Zero violations WITH contention evidence on both sides.
    Pass { cpu_contended: u32, gpu_contended: u32 },
    /// Mutual exclusion observed broken at this Delta.
    Violated,
    /// Completed without enough contention (or did not complete):
    /// proves nothing either way.
    Inconclusive,
}

/// One CPU-vs-GPU Fischer contention run.
fn fischer_self_test(
    region: &PeerRegion,
    stream: &Arc<CudaStream>,
    kernels: &CalibKernels,
    delta_ns: u64,
) -> Result<SelfTest, GpuPeerError> {
    const ACQS: u32 = 150;
    const CS_NS: u64 = 2_000;
    // A pass without real interleaving proves nothing: both sides
    // must have contended at least this many rounds.
    const CONTENTION_FLOOR: u32 = ACQS / 10;
    region.store_u32(HDR_FISCHER_X_OFF, 0);
    region.store_i32(HDR_FISCHER_CS_OFF, 0);
    region.store_u32(HDR_FISCHER_VIOL_OFF, 0);
    region.store_u32(HDR_FISCHER_ACQS_OFF, 0);
    region.store_u32(layout::HDR_FISCHER_STARTED_OFF, 0);
    region.store_u32(layout::HDR_FISCHER_GPU_CONT_OFF, 0);
    region.release_fence();

    let dev_base = region.dev_base();
    let acqs = ACQS;
    let cs = CS_NS;
    let mut b = stream.launch_builder(&kernels.fischer);
    b.arg(&dev_base);
    b.arg(&acqs);
    b.arg(&delta_ns);
    b.arg(&cs);
    // SAFETY: argument types match flynnel_peer_fischer(u8*, u32, u64, u64).
    unsafe { b.launch(one_one()) }
        .map_err(|e| GpuPeerError::Driver(format!("fischer launch: {e:?}")))?;
    flush_stream(stream);

    // Overlap is forced, not hoped for: contend only once the GPU
    // contender is resident (its first act is raising `started`).
    let t0 = Instant::now();
    while region.load_u32(layout::HDR_FISCHER_STARTED_OFF) == 0 {
        if t0.elapsed() > Duration::from_secs(3) {
            stream.synchronize().ok();
            return Ok(SelfTest::Inconclusive);
        }
        core::hint::spin_loop();
    }

    let delta = Duration::from_nanos(delta_ns);
    let cs_hold = Duration::from_nanos(CS_NS);
    let deadline = Duration::from_secs(6);
    let mut cpu_done = 0u32;
    let mut cpu_contended = 0u32;
    for _ in 0..ACQS {
        match timed_lock::acquire_with_contention(region, delta, deadline) {
            Some(contended) => {
                if contended {
                    cpu_contended += 1;
                }
                timed_lock::critical_section_checked(region, cs_hold);
                timed_lock::release(region);
                cpu_done += 1;
            }
            None => break,
        }
    }
    // Wait for the GPU side to finish its acquisitions.
    let t0 = Instant::now();
    while region.load_u32(HDR_FISCHER_ACQS_OFF) < ACQS {
        if t0.elapsed() > Duration::from_secs(6) {
            break;
        }
        std::thread::yield_now();
    }
    stream
        .synchronize()
        .map_err(|e| GpuPeerError::Driver(format!("fischer drain: {e:?}")))?;
    let gpu_done = region.load_u32(HDR_FISCHER_ACQS_OFF);
    let gpu_contended = region.load_u32(layout::HDR_FISCHER_GPU_CONT_OFF);
    let violations = region.load_u32(HDR_FISCHER_VIOL_OFF);
    if violations > 0 {
        return Ok(SelfTest::Violated);
    }
    if cpu_done == ACQS
        && gpu_done == ACQS
        && cpu_contended >= CONTENTION_FLOOR
        && gpu_contended >= CONTENTION_FLOOR
    {
        Ok(SelfTest::Pass { cpu_contended, gpu_contended })
    } else {
        Ok(SelfTest::Inconclusive)
    }
}

/// GPU `atomicCAS_system` vs CPU `compare_exchange` claim race over
/// scratch slots (the slot slab, which carries no lane traffic during
/// calibration). Claims must be conserved exactly for the flag.
fn cas_conservation_probe(
    region: &PeerRegion,
    stream: &Arc<CudaStream>,
    kernels: &CalibKernels,
) -> Result<bool, GpuPeerError> {
    let g = region.geometry();
    let slab = g.slab_off();
    let slab_bytes = g.region_bytes() - slab;
    // n slots + gpu_won + started cells, all inside the scratch slab.
    let n = ((slab_bytes / 4).saturating_sub(32)).min(60_000) as u32;
    if n < 4_096 {
        // Region too small to probe meaningfully; refuse the flag.
        return Ok(false);
    }
    for i in 0..n as usize {
        region.store_u32(slab + i * 4, 0);
    }
    let gpu_won_off = slab + n as usize * 4;
    let started_off = gpu_won_off + 4;
    region.store_u32(gpu_won_off, 0);
    region.store_u32(started_off, 0);
    region.release_fence();

    let dev_slots = region.dev_base() + slab as u64;
    let dev_won = region.dev_base() + gpu_won_off as u64;
    let dev_started = region.dev_base() + started_off as u64;
    let mut b = stream.launch_builder(&kernels.cas_probe);
    b.arg(&dev_slots);
    b.arg(&n);
    b.arg(&dev_won);
    b.arg(&dev_started);
    // SAFETY: argument types match
    // flynnel_peer_cas_probe(u32*, u32, u32*, u32*).
    // One narrow block on purpose: with thousands of GPU threads the
    // sweep finishes in microseconds and the CPU never contends; 64
    // threads make both sides sweep for a comparable duration so the
    // claim fronts interleave densely for the whole run.
    unsafe {
        b.launch(LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (64, 1, 1),
            shared_mem_bytes: 0,
        })
    }
    .map_err(|e| GpuPeerError::Driver(format!("cas probe launch: {e:?}")))?;
    flush_stream(stream);

    // Genuine overlap is a validity requirement, not a hope: wait
    // until the kernel reports it is resident and claiming (top-down)
    // before the CPU starts claiming bottom-up; the walks collide
    // wherever the two sides meet.
    let t0 = Instant::now();
    while region.load_u32(started_off) == 0 {
        if t0.elapsed() > Duration::from_secs(3) {
            stream.synchronize().ok();
            return Ok(false); // never started: inconclusive, refuse
        }
        core::hint::spin_loop();
    }

    // CPU contends through real atomic CAS on the same words.
    let base = region.base_addr() as usize;
    let mut cpu_won = 0u64;
    for i in 0..n as usize {
        // SAFETY: 4-aligned live mapping word inside the region; the
        // probe's entire purpose is contending this word with the
        // GPU's CAS, so an atomic view is the correct access.
        let a = unsafe { core::sync::atomic::AtomicU32::from_ptr((base + slab + i * 4) as *mut u32) };
        if a
            .compare_exchange(
                0,
                1,
                core::sync::atomic::Ordering::SeqCst,
                core::sync::atomic::Ordering::SeqCst,
            )
            .is_ok()
        {
            cpu_won += 1;
        }
    }
    stream
        .synchronize()
        .map_err(|e| GpuPeerError::Driver(format!("cas probe drain: {e:?}")))?;
    let gpu_won = region.load_u32(gpu_won_off) as u64;
    let mut unclaimed = 0u64;
    for i in 0..n as usize {
        if region.load_u32(slab + i * 4) == 0 {
            unclaimed += 1;
        }
    }
    // Conservation AND real contention: every slot claimed exactly
    // once, with BOTH sides winning a meaningful share (>= 5% each).
    // A lopsided sweep means the race barely happened and proves
    // nothing - refuse the flag.
    let conserved = unclaimed == 0 && cpu_won + gpu_won == n as u64;
    let floor = n as u64 / 20;
    let contended = cpu_won >= floor && gpu_won >= floor;
    Ok(conserved && contended)
}

#[cfg(test)]
mod tests {
    use super::derive_delta_ns;

    #[test]
    fn delta_derivation_clamps() {
        assert_eq!(derive_delta_ns(0), 5_000);
        assert_eq!(derive_delta_ns(100), 5_000);
        assert_eq!(derive_delta_ns(1_600), 16_000);
        assert_eq!(derive_delta_ns(2_500), 25_000);
        assert_eq!(derive_delta_ns(50_000), 100_000);
    }
}
