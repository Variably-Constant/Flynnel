//! Reference TPU backend driven by a Python-JAX child process.
//! Compiled only when the `tpu-jax-reference` Cargo feature is
//! enabled.
//!
//! ## How it works
//!
//! At construction the backend locates a Python interpreter (tries
//! `python3` first, then `python`), spawns it with the embedded
//! `tpu_jax_bridge.py` script, and exchanges a `ping` handshake.
//! The handshake verifies that:
//!
//! 1. The interpreter starts.
//! 2. `import jax` succeeds (JAX is installed).
//! 3. `jax.devices()` reports at least one device.
//!
//! Any of these failing returns
//! [`crate::backend::BackendError::DeviceUnavailable`], so a binary
//! built with `--features tpu-jax-reference` runs unchanged on hosts
//! without Python / without JAX / without TPU - the routing helper
//! falls back to the CPU backend.
//!
//! The bridge script is `include_str!`-baked into the Rust binary at
//! compile time. At construction it is written to a temp file under
//! the platform temp directory; the path is passed to the child as
//! its script argument. The temp file is cleaned up when the
//! backend drops.
//!
//! ## Wire protocol
//!
//! See `src/backend/tpu_jax_bridge.py` for the protocol contract.
//! Every Rust call (`register_kernel`, `dispatch_kernel`, the
//! `ping` handshake, `shutdown`) serializes a one-line JSON
//! request, writes it to the child's stdin, reads one line from the
//! child's stdout, and parses the JSON response.
//!
//! Concurrency: the bridge is request-response over one pipe, and a
//! response carries no request id, so it can only be matched to its
//! request by order. One thread owns the pipe and performs every
//! exchange; callers hand it a serialized request and a reply slot
//! and wait for that slot. Two callers therefore cannot interleave
//! lines, which is what the protocol needs, and no caller excludes
//! another from anything but the pipe itself. Throughput-wise the
//! bridge is single-flight either way, which matches JAX's actual TPU
//! launch semantics (per-device).

#![allow(clippy::missing_errors_doc)]

use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread::JoinHandle;

use serde::{Deserialize, Serialize};

use crate::backend::{
    Backend, BackendCapabilities, BackendError, DispatchBackend, KernelArg, KernelHandle,
};
use crate::sched::notify_ring::{NotifyHub, NotifySendResult, NotifySender};

/// Embedded Python bridge script. Compile-time `include_str!` so the
/// crate ships as a single artifact (no external file dependency at
/// runtime).
const BRIDGE_PY: &str = include_str!("tpu_jax_bridge.py");

/// Reference TPU backend that drives a Python-JAX child process.
pub struct TpuJaxBackend {
    device_id: u32,
    caps: BackendCapabilities,
    /// What the handshake reported. Fixed once the child has
    /// answered, so it is read without asking the bridge.
    devices: Vec<String>,
    /// Requests to the thread that owns the pipe.
    hub: NotifyHub<BridgeRequest>,
    /// Cached sender, so a transaction does not clone the hub.
    tx: NotifySender<BridgeRequest>,
    /// The owning thread, taken by `Drop`, which holds this
    /// exclusively and so needs nothing to guard it.
    owner: Option<JoinHandle<()>>,
}

/// State the owning thread holds. It is reached from that thread
/// alone, which is what removes the exclusion the pipe used to need.
struct BridgeState {
    child: Option<Child>,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    script_path: PathBuf,
    devices: Vec<String>,
}

/// One exchange for the owning thread: a serialized request and the
/// slot its response goes to.
struct BridgeRequest {
    body: String,
    reply: std::sync::mpsc::Sender<Result<String, BackendError>>,
}

/// Responses the owning thread produced for a caller that had already
/// stopped waiting. Nonzero means a caller timed out or was dropped
/// mid-transaction, not that the bridge lost anything: the exchange
/// completed and the line was read.
static REPLIES_UNCLAIMED: AtomicU64 = AtomicU64::new(0);

/// Responses that completed with nobody left to take them, since
/// process start.
pub fn replies_unclaimed() -> u64 {
    REPLIES_UNCLAIMED.load(Ordering::Relaxed)
}

impl std::fmt::Debug for TpuJaxBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TpuJaxBackend")
            .field("device_id", &self.device_id)
            .finish()
    }
}

impl TpuJaxBackend {
    /// Spawn the JAX bridge on the primary TPU device.
    pub fn new() -> Result<Self, BackendError> {
        Self::with_device(0)
    }

    /// Spawn the JAX bridge targeting `device_id` (passed through as
    /// the identity tag on the resulting [`Backend::Tpu`] id only;
    /// JAX itself manages device selection per its own
    /// `jax.devices()` order).
    pub fn with_device(device_id: u32) -> Result<Self, BackendError> {
        let script_path = write_bridge_script()?;
        // The handshake runs on the constructing thread, before the
        // state moves and before any thread starts: a backend that
        // cannot answer `ping` is not constructed, so the owning
        // thread never starts for one. It is also what chooses the
        // interpreter, because starting one proves nothing.
        let (state, pong) = open_bridge(script_path)?;
        let devices = pong.devices;

        // A ring deep enough that callers hand over their requests
        // rather than queue on the handover itself. The bridge is
        // single-flight, so depth past the callers that can be waiting
        // buys nothing and this is already more than the pool has
        // workers on the hosts here.
        const BRIDGE_RING_CAPACITY: usize = 256;
        let hub = NotifyHub::<BridgeRequest>::new(BRIDGE_RING_CAPACITY, 1);
        let tx = hub.sender();
        let hub_for_owner = hub.clone();
        let owner = std::thread::Builder::new()
            .name(format!("flynnel-tpu-jax-{device_id}"))
            .spawn(move || owner_loop(state, &hub_for_owner))
            .map_err(|_| BackendError::DeviceUnavailable(Backend::Tpu { device_id }))?;

        Ok(Self {
            device_id,
            caps: probe_capabilities(),
            devices,
            hub,
            tx,
            owner: Some(owner),
        })
    }

    /// Devices the JAX runtime reported during the handshake (e.g.
    /// `["TpuDevice(id=0, ...)"]`). Useful for telemetry.
    pub fn devices(&self) -> Vec<String> {
        self.devices.clone()
    }
}

/// The thread that owns the pipe: one exchange at a time, in the order
/// the requests arrived, then the polite shutdown once the hub closes.
fn owner_loop(mut state: BridgeState, hub: &NotifyHub<BridgeRequest>) {
    let rx = hub.register_consumer();
    while let Some(request) = rx.recv() {
        let outcome = exchange(&mut state, &request.body);
        if let Err(unclaimed) = request.reply.send(outcome) {
            // The caller stopped waiting before its answer arrived.
            // The exchange itself completed, so the pipe is still in
            // step; what is lost is one response nobody wants.
            REPLIES_UNCLAIMED.fetch_add(1, Ordering::Relaxed);
            drop(unclaimed);
        }
    }
    // The hub is closed, so no further request can arrive and this
    // thread is the only one that can still reach the child.
    let farewell = serde_json::json!({"op": "shutdown"});
    match writeln!(state.stdin, "{farewell}") {
        Ok(()) => {}
        // The child is already gone, which is the common way to reach
        // this line and needs no answer beyond not waiting for one.
        Err(gone) => drop(gone),
    }
    if let Some(mut child) = state.child.take() {
        match child.wait() {
            // The bridge was told to shut down a few lines above, so
            // a status other than success is the child having ended
            // of something else, which the next run would otherwise
            // meet as an unexplained missing bridge.
            Ok(status) => {
                if !status.success() {
                    eprintln!("tpu_jax: the bridge process ended with {status}");
                }
            }
            Err(unwaitable) => {
                eprintln!("tpu_jax: the bridge process could not be waited for: {unwaitable}");
            }
        }
    }
    match std::fs::remove_file(&state.script_path) {
        Ok(()) => {}
        // The temp directory may have been swept already.
        Err(absent) => drop(absent),
    }
}

impl Drop for TpuJaxBackend {
    fn drop(&mut self) {
        // Closing the hub is what ends the owning thread's loop and
        // starts its shutdown; joining is what makes the child's exit
        // and the script's removal have happened before this returns.
        self.hub.shutdown();
        if let Some(owner) = self.owner.take() {
            match owner.join() {
                Ok(()) => {}
                // The owning thread panicked. Drop cannot recover, and
                // the child is reaped by the OS.
                Err(panicked) => drop(panicked),
            }
        }
    }
}

impl DispatchBackend for TpuJaxBackend {
    fn id(&self) -> Backend {
        Backend::Tpu {
            device_id: self.device_id,
        }
    }

    fn capabilities(&self) -> BackendCapabilities {
        self.caps
    }

    fn dispatch_parallel_for(&self, count: u32, work: &(dyn Fn(u32) + Send + Sync)) {
        // The closure body is a Rust closure, not a JAX function;
        // there is no codegen path from Rust source to TPU XLA. To
        // keep parity with the other backends' parallel-for shape
        // we run the closure host-side, fanned out across worker
        // threads. For real TPU compute, consumers use the
        // `dispatch_kernel` handle path with a Python source body.
        if count == 0 {
            return;
        }
        std::thread::scope(|scope| {
            let threads = (count as usize).min(
                std::thread::available_parallelism()
                    .map(std::num::NonZeroUsize::get)
                    .unwrap_or(1),
            );
            let chunks = count.div_ceil(threads as u32);
            for t in 0..threads as u32 {
                let lo = t.saturating_mul(chunks);
                let hi = (lo + chunks).min(count);
                if lo >= hi {
                    continue;
                }
                scope.spawn(move || {
                    for i in lo..hi {
                        work(i);
                    }
                });
            }
        });
    }

    fn dispatch_one(&self, work: Box<dyn FnOnce() + Send>) {
        std::thread::spawn(work);
    }

    fn register_kernel(&self, name: &str, source: &[u8]) -> Result<KernelHandle, BackendError> {
        let source_str = std::str::from_utf8(source).map_err(|e| {
            BackendError::KernelCompile(format!("source must be UTF-8 Python: {e}"))
        })?;
        let req = RegisterRequest {
            op: "register",
            name,
            source: source_str,
        };
        let resp: RegisterResponse = transact(&self.tx, &req)?;
        if !resp.ok {
            return Err(BackendError::KernelCompile(
                resp.error.unwrap_or_else(|| "register failed".into()),
            ));
        }
        let handle =
            resp.handle.ok_or_else(|| BackendError::KernelCompile("missing handle".into()))?;
        Ok(KernelHandle(handle))
    }

    fn dispatch_kernel(
        &self,
        handle: KernelHandle,
        count: u32,
        args: &[KernelArg<'_>],
    ) -> Result<(), BackendError> {
        let json_args = args
            .iter()
            .map(arg_to_json)
            .collect::<Result<Vec<_>, _>>()?;
        let req = DispatchRequest {
            op: "dispatch",
            handle: handle.0,
            count,
            args: json_args,
        };
        let resp: PlainResponse = transact(&self.tx, &req)?;
        if !resp.ok {
            return Err(BackendError::Launch(
                resp.error.unwrap_or_else(|| "dispatch failed".into()),
            ));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Wire types
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct PingRequest<'a> {
    op: &'a str,
}

#[derive(Deserialize)]
struct PingResponse {
    ok: bool,
    #[serde(default)]
    devices: Vec<String>,
    #[serde(default)]
    error: Option<String>,
    #[serde(default)]
    #[allow(dead_code)]
    jax_version: Option<String>,
}

#[derive(Serialize)]
struct RegisterRequest<'a> {
    op: &'a str,
    name: &'a str,
    source: &'a str,
}

#[derive(Deserialize)]
struct RegisterResponse {
    ok: bool,
    #[serde(default)]
    handle: Option<u64>,
    #[serde(default)]
    error: Option<String>,
}

#[derive(Serialize)]
struct DispatchRequest<'a> {
    op: &'a str,
    handle: u64,
    count: u32,
    args: Vec<serde_json::Value>,
}

#[derive(Deserialize)]
struct PlainResponse {
    ok: bool,
    #[serde(default)]
    error: Option<String>,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn write_bridge_script() -> Result<PathBuf, BackendError> {
    let mut path = std::env::temp_dir();
    let pid = std::process::id();
    path.push(format!("flynnel_tpu_jax_bridge_{pid}.py"));
    std::fs::write(&path, BRIDGE_PY).map_err(|e| {
        BackendError::DeviceUnavailable(Backend::Tpu { device_id: 0 })
            .map_io_context(format!("write bridge script: {e}"))
    })?;
    Ok(path)
}

/// Start the bridge on the first interpreter that answers `ping`.
///
/// Starting is not the test. Windows ships an execution alias named
/// `python3` that starts, says where Python can be installed and
/// exits, so a spawn that succeeds says nothing about whether an
/// interpreter is there, and a host carrying that alias beside a real
/// `python` has the working one second in this list. The handshake is
/// the test, and an interpreter that fails it is reaped and the next
/// one tried.
fn open_bridge(script_path: PathBuf) -> Result<(BridgeState, PingResponse), BackendError> {
    let mut last = String::from("no python3 or python interpreter on PATH");
    for interpreter in ["python3", "python"] {
        let started = Command::new(interpreter)
            .arg(&script_path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn();
        let mut child = match started {
            Ok(child) => child,
            Err(absent) => {
                last = format!("{interpreter}: {absent}");
                continue;
            }
        };
        let (Some(stdin), Some(stdout)) = (child.stdin.take(), child.stdout.take()) else {
            last = format!("{interpreter}: started without both pipes");
            reap(interpreter, child);
            continue;
        };
        let mut state = BridgeState {
            child: Some(child),
            stdin,
            stdout: BufReader::new(stdout),
            script_path: script_path.clone(),
            devices: Vec::new(),
        };
        match ping_handshake(&mut state) {
            Ok(pong) => {
                state.devices = pong.devices.clone();
                return Ok((state, pong));
            }
            Err(refused) => {
                last = format!("{interpreter}: {refused}");
                if let Some(child) = state.child.take() {
                    reap(interpreter, child);
                }
            }
        }
    }
    match std::fs::remove_file(&script_path) {
        Ok(()) => {}
        // No interpreter got as far as owning the script, so nothing
        // else will remove it, and a temp directory that has already
        // been swept is the ordinary way to reach this.
        Err(absent) => drop(absent),
    }
    Err(BackendError::DeviceUnavailable(Backend::Tpu { device_id: 0 }).map_io_context(last))
}

/// End a child that will not be used and wait for it, so a rejected
/// interpreter leaves no process behind.
fn reap(interpreter: &str, mut child: Child) {
    match child.kill() {
        Ok(()) => {}
        Err(gone) => {
            eprintln!("tpu_jax: the {interpreter} child was already gone: {gone}");
        }
    }
    match child.wait() {
        Ok(_ended) => {}
        Err(unwaitable) => {
            eprintln!("tpu_jax: the {interpreter} child could not be waited for: {unwaitable}");
        }
    }
}

fn ping_handshake(state: &mut BridgeState) -> Result<PingResponse, BackendError> {
    let req = PingRequest { op: "ping" };
    let req_json = serde_json::to_string(&req).map_err(|e| {
        BackendError::DeviceUnavailable(Backend::Tpu { device_id: 0 })
            .map_io_context(format!("ping serialize: {e}"))
    })?;
    writeln!(state.stdin, "{req_json}").map_err(|e| {
        BackendError::DeviceUnavailable(Backend::Tpu { device_id: 0 })
            .map_io_context(format!("ping stdin write: {e}"))
    })?;
    state.stdin.flush().map_err(|e| {
        BackendError::DeviceUnavailable(Backend::Tpu { device_id: 0 })
            .map_io_context(format!("ping stdin flush: {e}"))
    })?;
    let mut line = String::new();
    state.stdout.read_line(&mut line).map_err(|e| {
        BackendError::DeviceUnavailable(Backend::Tpu { device_id: 0 })
            .map_io_context(format!("ping stdout read: {e}"))
    })?;
    if line.trim().is_empty() {
        return Err(BackendError::DeviceUnavailable(Backend::Tpu { device_id: 0 })
            .map_io_context("ping: bridge sent empty response".into()));
    }
    let pong: PingResponse = serde_json::from_str(line.trim()).map_err(|e| {
        BackendError::DeviceUnavailable(Backend::Tpu { device_id: 0 })
            .map_io_context(format!("ping parse `{}`: {e}", line.trim()))
    })?;
    if !pong.ok {
        return Err(BackendError::DeviceUnavailable(Backend::Tpu { device_id: 0 })
            .map_io_context(
                pong.error.unwrap_or_else(|| "ping reported not-ok".into()),
            ));
    }
    if pong.devices.is_empty() {
        return Err(BackendError::DeviceUnavailable(Backend::Tpu { device_id: 0 })
            .map_io_context("ping: jax reported zero devices".into()));
    }
    Ok(pong)
}

/// One request out and one response line back, performed by whichever
/// thread owns the pipe.
fn exchange(state: &mut BridgeState, body: &str) -> Result<String, BackendError> {
    writeln!(state.stdin, "{body}")
        .map_err(|e| BackendError::Launch(format!("stdin write: {e}")))?;
    state
        .stdin
        .flush()
        .map_err(|e| BackendError::Launch(format!("stdin flush: {e}")))?;
    let mut line = String::new();
    state
        .stdout
        .read_line(&mut line)
        .map_err(|e| BackendError::Launch(format!("stdout read: {e}")))?;
    Ok(line)
}

/// Hand one request to the thread that owns the pipe and wait for its
/// answer.
///
/// The wait is on this request's own reply slot rather than on the
/// pipe, so a second caller queues behind this one at the hub instead
/// of excluding it from anything.
fn transact<Req, Resp>(tx: &NotifySender<BridgeRequest>, req: &Req) -> Result<Resp, BackendError>
where
    Req: Serialize,
    Resp: for<'de> Deserialize<'de>,
{
    let body = serde_json::to_string(req)
        .map_err(|e| BackendError::Launch(format!("request serialize: {e}")))?;
    let (reply, answer) = std::sync::mpsc::channel();
    match tx.send(BridgeRequest { body, reply }) {
        NotifySendResult::Ok => {}
        NotifySendResult::Closed(refused) => {
            drop(refused);
            return Err(BackendError::Launch(
                "the bridge is shut down, so this request was not sent".into(),
            ));
        }
    }
    let line = answer
        .recv()
        .map_err(|_| BackendError::Launch("the bridge thread ended before answering".into()))??;
    serde_json::from_str(line.trim())
        .map_err(|e| BackendError::Launch(format!("response parse `{}`: {e}", line.trim())))
}

fn arg_to_json(arg: &KernelArg<'_>) -> Result<serde_json::Value, BackendError> {
    let v = match *arg {
        KernelArg::I32(v) => serde_json::json!({"i32": v}),
        KernelArg::I64(v) => serde_json::json!({"i64": v}),
        KernelArg::U32(v) => serde_json::json!({"u32": v}),
        KernelArg::U64(v) => serde_json::json!({"u64": v}),
        KernelArg::F32(v) => serde_json::json!({"f32": v}),
        KernelArg::F64(v) => serde_json::json!({"f64": v}),
        KernelArg::DevicePtr(p) => serde_json::json!({"device_ptr": p as u64}),
        KernelArg::HostSlice(_) => return Err(BackendError::NotSupported),
    };
    Ok(v)
}

fn probe_capabilities() -> BackendCapabilities {
    BackendCapabilities {
        // TPU "MXU lane" sized at the 128-wide systolic array.
        simt_width: 128,
        // TPU v4 + v5 are good for hundreds of thousands of in-
        // flight tiles; nominal upper bound.
        max_threads_in_flight: 200_000,
        // Python-JAX dispatch round-trip dominated by JSON encode +
        // subprocess pipe latency: ~100us is realistic.
        launch_latency_ns: 100_000,
        // PCIe class on standalone TPU edge; cloud TPU is hosted
        // through high-bandwidth interconnect that's not bottle-
        // necked by host transfer.
        h2d_bw_bytes_per_sec: 25_000_000_000,
    }
}

/// Internal helper trait so the BackendError construction sites
/// above can attach an I/O context message without exposing the
/// detail in the public error variant. Keeps DeviceUnavailable as a
/// single-variant tag while still surfacing the cause via Display.
trait WithIoContext: Sized {
    fn map_io_context(self, msg: String) -> Self;
}

impl WithIoContext for BackendError {
    fn map_io_context(self, msg: String) -> Self {
        match self {
            BackendError::DeviceUnavailable(b) => {
                eprintln!("[flynnel::tpu_jax] {}: {msg}", b.name());
                BackendError::DeviceUnavailable(b)
            }
            other => other,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn construction_matches_python_jax_availability() {
        let res = TpuJaxBackend::new();
        // No assertion about Ok vs Err: hosts vary. We only assert
        // that construction returns a typed Result, not a panic /
        // process abort.
        match res {
            Ok(b) => {
                assert_eq!(b.id(), Backend::Tpu { device_id: 0 });
                assert_eq!(b.capabilities().simt_width, 128);
                let devices = b.devices();
                assert!(
                    !devices.is_empty(),
                    "successful ping must report at least one device"
                );
            }
            Err(BackendError::DeviceUnavailable(Backend::Tpu { device_id })) => {
                assert_eq!(device_id, 0);
            }
            Err(other) => panic!("unexpected construction error: {other}"),
        }
    }

    #[test]
    fn capabilities_have_expected_shape() {
        let caps = probe_capabilities();
        assert_eq!(caps.simt_width, 128);
        assert_eq!(caps.h2d_bw_bytes_per_sec, 25_000_000_000);
        assert!(caps.launch_latency_ns >= 10_000);
    }
}
