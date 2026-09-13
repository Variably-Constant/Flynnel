//! One device at a time, across test binaries as well as within one.
//!
//! Each GPU test file declares its own `static GPU: Mutex<()>`, which
//! serializes the tests inside that binary and nothing else. Cargo runs
//! the test binaries concurrently, so gpu_peer_team,
//! gpu_peer_failure_paths, real_gpu_consumer_shapes and the parity
//! suites all drive the device at once.
//!
//! That is visible in the numbers. The 64-block barrier wait in
//! gpu_peer_team printed 51264, 53440 and 412416 ns across three runs
//! on one host, an eight times swing decided by which other binary was
//! resident. A test asserting a bound on that measures the neighbour.
//!
//! The lock here is a file, because the binaries are separate processes
//! and a `Mutex` cannot reach across them. It is advisory: a process
//! that does not ask still gets the device.

use std::fs::OpenOptions;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

/// Whether a foreign process was resident on the device while a timed
/// region ran, so a reading taken across it is about the neighbour.
///
/// The lock above excludes Flynnel's own test binaries and nothing
/// else. A process from another project holds the device regardless,
/// and a wall-clock assertion then measures it. The driver exposes
/// memory but not utilization, so memory is the signal here: a
/// neighbour's allocation is visible where its load is not.
///
/// The threshold is measured rather than chosen. Sample before the
/// device work is set up and again once its buffers exist, and the
/// difference is this run's own footprint on this host. Nothing more is
/// allocated after that, so a rise during the timed region larger than
/// the run's own footprint is somebody else.
///
/// Sampling runs across the region, not at its ends: a neighbour that
/// arrives and leaves between two endpoint samples is invisible to
/// both.
///
/// This module is compiled into every test binary that declares it, and
/// each uses a different part, so what one binary does not call is dead
/// code there under `-D warnings`.
#[allow(dead_code)]
pub struct DeviceMemoryWatch {
    before_setup: Option<u64>,
    after_setup: Option<u64>,
    peak: Option<u64>,
}

#[allow(dead_code)]
impl DeviceMemoryWatch {
    /// Sample before any device buffers exist.
    pub fn before_setup() -> Self {
        Self {
            before_setup: flynnel::gpu_peer::device_memory_in_use().map(|(used, _)| used),
            after_setup: None,
            peak: None,
        }
    }

    /// Sample once the run's own buffers exist and before the timed
    /// region starts. Everything above this is another process.
    pub fn setup_done(&mut self) {
        self.after_setup = flynnel::gpu_peer::device_memory_in_use().map(|(used, _)| used);
        self.peak = self.after_setup;
    }

    /// Take a reading inside the timed region.
    pub fn sample(&mut self) {
        if let Some((used, _)) = flynnel::gpu_peer::device_memory_in_use() {
            self.peak = Some(self.peak.map_or(used, |p| p.max(used)));
        }
    }

    /// This run's own device footprint, in bytes.
    fn own_footprint(&self) -> Option<u64> {
        match (self.before_setup, self.after_setup) {
            (Some(before), Some(after)) => Some(after.saturating_sub(before)),
            _ => None,
        }
    }

    /// Bytes that appeared after this run finished allocating.
    fn arrived(&self) -> Option<u64> {
        match (self.after_setup, self.peak) {
            (Some(after), Some(peak)) => Some(peak.saturating_sub(after)),
            _ => None,
        }
    }

    /// Whether a wall-clock reading taken across this region is about
    /// this crate rather than about a neighbour.
    ///
    /// Unknown when the driver would not answer, and an unknown reading
    /// is treated as measurable: refusing to assert whenever the query
    /// fails would turn a driver quirk into a suite that never checks
    /// anything.
    pub fn measurable(&self) -> bool {
        match (self.arrived(), self.own_footprint()) {
            (Some(arrived), Some(own)) => arrived <= own,
            _ => true,
        }
    }

    /// What the device did across the region, for a message that has to
    /// explain why a reading was not used.
    pub fn describe(&self) -> String {
        match (self.before_setup, self.after_setup, self.peak) {
            (Some(before), Some(after), Some(peak)) => format!(
                "device memory {before} bytes before setup, {after} after this run's own \
                 buffers ({} of its own), peaking at {peak} during the region, so {} arrived \
                 from elsewhere",
                after.saturating_sub(before),
                peak.saturating_sub(after)
            ),
            _ => "device memory unavailable from the driver".to_string(),
        }
    }
}

/// Held for as long as one test needs the device. Dropping it releases
/// the device to the next waiter, in any test binary.
pub struct DeviceLock {
    path: PathBuf,
}

impl Drop for DeviceLock {
    fn drop(&mut self) {
        if let Err(e) = std::fs::remove_file(&self.path) {
            // The file should be exactly the one this guard created, so
            // its absence means another process judged it abandoned and
            // took the device while this test still held it. That is the
            // one way this lock fails to do its job, and it is worth
            // saying rather than dropping.
            eprintln!(
                "flynnel gpu test lock: releasing {} failed ({e}); another \
                 process took it as abandoned, so two held the device",
                self.path.display()
            );
        }
    }
}

/// How long a lock file may sit before a waiter treats it as abandoned
/// and takes the device anyway.
///
/// A process killed between creating the file and dropping its guard
/// leaves the file behind, and with no steal every later run on that
/// host waits forever. The window is longer than any GPU test here:
/// the slowest holds a rank past a 400 ms barrier deadline on top of an
/// NVRTC compile, and the whole gpu_peer_team binary runs in about
/// three seconds.
const STALE_AFTER: Duration = Duration::from_secs(120);

/// The longest a caller waits before concluding the holder is stuck
/// rather than slow. Above every GPU binary's runtime put together.
const GIVE_UP_AFTER: Duration = Duration::from_secs(600);

/// Whether the lock file has sat long enough to be treated as left
/// behind by a process that died holding it.
fn abandoned(path: &Path) -> bool {
    let created = match std::fs::metadata(path).and_then(|m| m.modified()) {
        Ok(t) => t,
        // Released between the failed create and this check, so there is
        // nothing to steal and the next create attempt will win it.
        Err(e) if e.kind() == ErrorKind::NotFound => return false,
        // On Windows a file another holder is still deleting refuses its
        // metadata with access denied; it is being released, not abandoned.
        Err(e) if e.kind() == ErrorKind::PermissionDenied => return false,
        Err(e) => panic!("gpu test lock: cannot read {}: {e}", path.display()),
    };
    match created.elapsed() {
        Ok(age) => age > STALE_AFTER,
        // A timestamp in the future means the clock moved under us. The
        // file is not old, and guessing an age from a bad clock would
        // steal a lock somebody holds.
        Err(skew) => {
            eprintln!(
                "flynnel gpu test lock: {} is stamped {} in the future; \
                 waiting rather than treating it as abandoned",
                path.display(),
                skew.duration().as_secs()
            );
            false
        }
    }
}

/// Wait for exclusive use of the device, then take it.
///
/// The returned guard holds the device until it drops, including while
/// a panic unwinds, so a failing test does not strand the next one.
pub fn device() -> DeviceLock {
    let path = std::env::temp_dir().join("flynnel-gpu-test.lock");
    let waiting_since = Instant::now();
    loop {
        match OpenOptions::new().write(true).create_new(true).open(&path) {
            Ok(_) => return DeviceLock { path },
            Err(e) if e.kind() == ErrorKind::AlreadyExists => {}
            // On Windows, creating a file that another holder is still
            // deleting fails with access denied until the delete completes,
            // which is the same contention as a file that still exists.
            Err(e) if e.kind() == ErrorKind::PermissionDenied => {}
            // A path that cannot be created at all is not contention.
            // Spinning on it would wait the full ten minutes and then
            // report a stuck holder that never existed.
            Err(e) => panic!("gpu test lock: cannot create {}: {e}", path.display()),
        }
        // A failed removal means another waiter cleared it first, which
        // reaches the same state; retry either way.
        if abandoned(&path) && std::fs::remove_file(&path).is_ok() {
            continue;
        }
        assert!(
            waiting_since.elapsed() < GIVE_UP_AFTER,
            "waited {} seconds for {}, longer than every GPU test in this \
             suite put together. Something holds the device and is not \
             releasing it.",
            GIVE_UP_AFTER.as_secs(),
            path.display()
        );
        std::thread::sleep(Duration::from_millis(25));
    }
}
