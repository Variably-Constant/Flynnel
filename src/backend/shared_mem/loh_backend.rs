//! `DispatchBackend` impl over an MMF-backed LOH deque + MMF latch
//! arena. Companion to [`super::chase_lev_backend::SharedMemoryChaseLevBackend`]
//! that targets bursty-dispatch workloads where many items are pushed
//! per coherence interval.
//!
//! ## When to pick LOH over Chase-Lev
//!
//! - **Bursty dispatch** (parallel-for fan-out, fork-join leaves):
//!   the per-burst migration amortizes the per-item ring-tail update
//!   over the whole batch. Chase-Lev pays a Release-store on `bottom`
//!   per item; LOH pays one Release-store on `tail` per batch.
//! - **Single-item request-reply**: LOH degenerates to one push +
//!   one auto-flush per item, no amortization, so it does NOT beat
//!   Chase-Lev. Use Chase-Lev for that shape.

#![allow(clippy::missing_errors_doc)]

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use super::chase_lev_backend::DispatchHandle;
use super::lcrq_lifo::{self, LOH_ARGS_INLINE_BYTES, LohDeque, LohLifoEntry, Steal};
use super::latch_mmf::{ERR, MmfLatchArena, SET, UNSET};
use super::pass_registry::{self, Pass};
use super::wire;

use crate::backend::{
    Backend, BackendCapabilities, BackendError, DispatchBackend, KernelArg, KernelHandle,
};

/// `DispatchBackend` impl over an MMF LOH deque + MMF latch arena.
pub struct SharedMemoryLohBackend {
    deque: LohDeque,
    latches: MmfLatchArena,
    backend_id: u32,
    deque_path: PathBuf,
    latches_path: PathBuf,
    caps: BackendCapabilities,
    dispatched: AtomicU64,
}

impl SharedMemoryLohBackend {
    /// Create the deque + latch arena files. Truncates if they exist.
    /// `flush_threshold` is the LIFO length at which auto-flush
    /// fires; the originator's hot path absorbs `flush_threshold - 1`
    /// pushes without touching the ring.
    pub fn create(
        backend_id: u32,
        deque_path: impl Into<PathBuf>,
        latches_path: impl Into<PathBuf>,
        deque_capacity: usize,
        latch_capacity: usize,
        flush_threshold: usize,
    ) -> Result<Self, BackendError> {
        let deque_path = deque_path.into();
        let latches_path = latches_path.into();
        let deque = LohDeque::create(&deque_path, deque_capacity, flush_threshold)
            .map_err(|e| BackendError::Memory(format!("create LOH deque: {e}")))?;
        let latches = MmfLatchArena::create(&latches_path, latch_capacity)
            .map_err(|e| BackendError::Memory(format!("create latch arena: {e}")))?;
        let caps = Self::caps_for(deque.capacity());
        Ok(Self {
            deque,
            latches,
            backend_id,
            deque_path,
            latches_path,
            caps,
            dispatched: AtomicU64::new(0),
        })
    }

    /// Attach to an existing LOH deque + latch arena.
    pub fn open(
        backend_id: u32,
        deque_path: impl Into<PathBuf>,
        latches_path: impl Into<PathBuf>,
        flush_threshold: usize,
    ) -> Result<Self, BackendError> {
        let deque_path = deque_path.into();
        let latches_path = latches_path.into();
        let deque = LohDeque::open(&deque_path, flush_threshold)
            .map_err(|e| BackendError::Memory(format!("open LOH deque: {e}")))?;
        let latches = MmfLatchArena::open(&latches_path)
            .map_err(|e| BackendError::Memory(format!("open latch arena: {e}")))?;
        let caps = Self::caps_for(deque.capacity());
        Ok(Self {
            deque,
            latches,
            backend_id,
            deque_path,
            latches_path,
            caps,
            dispatched: AtomicU64::new(0),
        })
    }

    fn caps_for(capacity: usize) -> BackendCapabilities {
        BackendCapabilities {
            simt_width: 1,
            max_threads_in_flight: capacity.max(1) as u32,
            launch_latency_ns: 250,
            h2d_bw_bytes_per_sec: 0,
        }
    }

    /// Path of the LOH deque file.
    pub fn deque_path(&self) -> &std::path::Path {
        &self.deque_path
    }

    /// Path of the latch arena file.
    pub fn latches_path(&self) -> &std::path::Path {
        &self.latches_path
    }

    /// Total dispatches issued through this backend since creation.
    pub fn dispatched(&self) -> u64 {
        self.dispatched.load(Ordering::Relaxed)
    }

    /// Owner-side: stage one Marshal-shaped item in the local LIFO.
    /// May auto-flush per the configured threshold. Returns a
    /// [`DispatchHandle`] the caller waits on with
    /// [`Self::wait_handle`].
    pub fn dispatch_marshal(
        &self,
        closure_id: u32,
        args: &[u8],
    ) -> Result<DispatchHandle, BackendError> {
        if args.len() > LOH_ARGS_INLINE_BYTES {
            return Err(BackendError::Launch(format!(
                "LOH marshal args length {} exceeds slot capacity {}",
                args.len(),
                LOH_ARGS_INLINE_BYTES
            )));
        }
        let latch_offset = self.latches.alloc();
        let entry = LohLifoEntry::new(closure_id, latch_offset, args).map_err(|e| {
            BackendError::Launch(format!("build LOH entry for closure_id={closure_id}: {e:?}"))
        })?;
        self.deque
            .push(entry)
            .map_err(|e| BackendError::Launch(format!("LOH push failed: {e:?}")))?;
        self.dispatched.fetch_add(1, Ordering::Relaxed);
        Ok(DispatchHandle { latch_offset })
    }

    /// Owner-side: stage N Marshal-shaped items then explicitly flush
    /// at the end so the thief sees every item by the time this call
    /// returns. The per-push auto-flush at `flush_threshold` fires
    /// inside the staging loop whenever the LIFO reaches the
    /// threshold, so for N > flush_threshold the batch produces
    /// ceil(N / flush_threshold) auto-flushes plus one explicit final
    /// flush; for N < flush_threshold only the final flush runs.
    /// Either way, LOH's amortization win comes from grouping the
    /// per-flush `tail.fetch_add(batch)` ring update instead of the
    /// per-item bookkeeping Chase-Lev pays. Returns one
    /// [`DispatchHandle`] per item, in caller-supplied order.
    pub fn dispatch_marshal_batch(
        &self,
        items: &[(u32, &[u8])],
    ) -> Result<Vec<DispatchHandle>, BackendError> {
        let mut handles = Vec::with_capacity(items.len());
        for (closure_id, args) in items {
            if args.len() > LOH_ARGS_INLINE_BYTES {
                return Err(BackendError::Launch(format!(
                    "LOH marshal args length {} exceeds slot capacity {}",
                    args.len(),
                    LOH_ARGS_INLINE_BYTES
                )));
            }
            let latch_offset = self.latches.alloc();
            let entry = LohLifoEntry::new(*closure_id, latch_offset, args).map_err(|e| {
                BackendError::Launch(format!("build LOH entry: {e:?}"))
            })?;
            self.deque
                .push(entry)
                .map_err(|e| BackendError::Launch(format!("LOH push failed: {e:?}")))?;
            handles.push(DispatchHandle { latch_offset });
        }
        // Force flush after the burst even if the auto-flush threshold
        // wasn't reached, so the thief sees every item by the time
        // this call returns.
        self.deque
            .flush()
            .map_err(|e| BackendError::Launch(format!("LOH flush failed: {e:?}")))?;
        self.dispatched
            .fetch_add(items.len() as u64, Ordering::Relaxed);
        Ok(handles)
    }

    /// Owner-side: explicit flush of any pending LIFO items.
    pub fn flush(&self) -> Result<usize, BackendError> {
        self.deque
            .flush()
            .map_err(|e| BackendError::Launch(format!("LOH flush failed: {e:?}")))
    }

    /// Peer-side: steal one slot, execute its registered handler,
    /// publish the reply into the latch cell.
    pub fn drain_one(&self) -> Result<Option<()>, BackendError> {
        let slot = loop {
            match self.deque.steal() {
                Steal::Success(s) => break s,
                Steal::Empty => return Ok(None),
                Steal::Retry => continue,
            }
        };
        let pass = Pass {
            closure_id: slot.closure_id,
            args: slot.args().to_vec(),
        };
        match pass_registry::execute(&pass) {
            Ok(reply) => self
                .latches
                .publish(slot.latch_offset, &reply)
                .map_err(|e| BackendError::Launch(format!("latch publish failed: {e:?}")))?,
            Err(err) => {
                let msg = err.to_string();
                let bytes = msg.as_bytes();
                let n = bytes.len().min(lcrq_lifo::LOH_SLOT_SIZE);
                self.latches
                    .publish_err(slot.latch_offset, &bytes[..n])
                    .map_err(|e| {
                        BackendError::Launch(format!("latch publish_err failed: {e:?}"))
                    })?;
            }
        };
        Ok(Some(()))
    }

    /// Originator-side non-blocking poll.
    pub fn poll_handle(
        &self,
        handle: DispatchHandle,
    ) -> Result<Option<Result<Vec<u8>, String>>, BackendError> {
        let is_set = self
            .latches
            .is_set(handle.latch_offset)
            .map_err(|e| BackendError::Launch(format!("latch poll: {e:?}")))?;
        if !is_set {
            return Ok(None);
        }
        let mut buf = Vec::new();
        let state = self
            .latches
            .read_result(handle.latch_offset, &mut buf)
            .map_err(|e| BackendError::Launch(format!("latch read: {e:?}")))?;
        self.latches
            .reset(handle.latch_offset)
            .map_err(|e| BackendError::Launch(format!("latch reset: {e:?}")))?;
        match state {
            SET => Ok(Some(Ok(buf))),
            ERR => Ok(Some(Err(String::from_utf8_lossy(&buf).into_owned()))),
            UNSET => Ok(None),
            other => Err(BackendError::Launch(format!(
                "latch read returned unexpected state: {other}"
            ))),
        }
    }

    /// Originator-side blocking wait.
    pub fn wait_handle(
        &self,
        handle: DispatchHandle,
        iter_budget: u32,
    ) -> Result<Result<Vec<u8>, String>, BackendError> {
        for _ in 0..iter_budget {
            if let Some(r) = self.poll_handle(handle)? {
                return Ok(r);
            }
            std::hint::spin_loop();
        }
        loop {
            if let Some(r) = self.poll_handle(handle)? {
                return Ok(r);
            }
            std::thread::yield_now();
        }
    }
}

impl DispatchBackend for SharedMemoryLohBackend {
    fn id(&self) -> Backend {
        Backend::SharedMemoryWorker {
            backend_id: self.backend_id,
        }
    }

    fn capabilities(&self) -> BackendCapabilities {
        self.caps
    }

    fn dispatch_parallel_for(&self, _count: u32, _work: &(dyn Fn(u32) + Send + Sync)) {
        // Closures don't cross processes. Cross-process fan-out is
        // achieved by attaching multiple peer processes to the same
        // LOH deque and relying on the per-thief CAS-on-head to
        // distribute slots.
    }

    fn dispatch_one(&self, _work: Box<dyn FnOnce() + Send>) {
        panic!(
            "SharedMemoryLohBackend does not support dispatch_one; \
             use register_kernel + dispatch_kernel (Marshal path)"
        );
    }

    fn register_kernel(&self, name: &str, _source: &[u8]) -> Result<KernelHandle, BackendError> {
        let id = pass_registry::hash_name(name);
        Ok(KernelHandle(id as u64))
    }

    fn dispatch_kernel(
        &self,
        handle: KernelHandle,
        _count: u32,
        args: &[KernelArg<'_>],
    ) -> Result<(), BackendError> {
        let closure_id = handle.0 as u32;
        let args_blob = wire::encode_args(args)?;
        self.dispatch_marshal(closure_id, &args_blob).map(|_| ())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::shared_mem::pass_registry::{hash_name, register, unregister};

    fn temp_paths(label: &str) -> (PathBuf, PathBuf) {
        let mut d = std::env::temp_dir();
        let mut l = std::env::temp_dir();
        let pid = std::process::id();
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        d.push(format!("flynnel_loh_be_d_{pid}_{nonce}_{label}.bin"));
        l.push(format!("flynnel_loh_be_l_{pid}_{nonce}_{label}.bin"));
        (d, l)
    }

    #[test]
    fn id_carries_backend_id() {
        let (d, l) = temp_paths("id");
        let be = SharedMemoryLohBackend::create(7, &d, &l, 4, 4, 1).expect("create");
        match be.id() {
            Backend::SharedMemoryWorker { backend_id } => assert_eq!(backend_id, 7),
            other => panic!("wrong id: {other:?}"),
        }
        std::fs::remove_file(&d).ok();
        std::fs::remove_file(&l).ok();
    }

    #[test]
    fn dispatch_marshal_round_trips_single_item() {
        let (d, l) = temp_paths("single");
        // flush_threshold = 1 makes per-item dispatch behave like a
        // single-item-in-flight backend (each push immediately migrates).
        let be = SharedMemoryLohBackend::create(0, &d, &l, 4, 4, 1).expect("create");

        let id = hash_name("flynnel.test.loh.adder");
        register(id, |args| {
            let a = u32::from_le_bytes(args[0..4].try_into().unwrap());
            let b = u32::from_le_bytes(args[4..8].try_into().unwrap());
            Ok((a + b).to_le_bytes().to_vec())
        });

        let mut args = [0u8; 8];
        args[..4].copy_from_slice(&13u32.to_le_bytes());
        args[4..].copy_from_slice(&29u32.to_le_bytes());
        let h = be.dispatch_marshal(id, &args).expect("dispatch");
        be.drain_one().expect("drain").expect("had work");
        let r = be.wait_handle(h, 1024).expect("wait").expect("ok branch");
        let v = u32::from_le_bytes(r[..4].try_into().unwrap());
        assert_eq!(v, 42);

        unregister(id);
        std::fs::remove_file(&d).ok();
        std::fs::remove_file(&l).ok();
    }

    #[test]
    fn dispatch_marshal_batch_amortizes_flush() {
        let (d, l) = temp_paths("batch");
        // flush_threshold = 16; the batch flush at end of
        // dispatch_marshal_batch forces a single migration over all
        // 8 items, never waiting for the auto-flush threshold.
        let be = SharedMemoryLohBackend::create(0, &d, &l, 16, 32, 16).expect("create");

        let id = hash_name("flynnel.test.loh.batch_adder");
        register(id, |args| {
            let a = u32::from_le_bytes(args[0..4].try_into().unwrap());
            let b = u32::from_le_bytes(args[4..8].try_into().unwrap());
            Ok((a + b).to_le_bytes().to_vec())
        });

        let mut payloads = Vec::with_capacity(8);
        for i in 0..8u32 {
            let mut p = [0u8; 8];
            p[..4].copy_from_slice(&i.to_le_bytes());
            p[4..].copy_from_slice(&(i * 2).to_le_bytes());
            payloads.push(p);
        }
        let items: Vec<(u32, &[u8])> = payloads.iter().map(|p| (id, p.as_slice())).collect();
        let handles = be.dispatch_marshal_batch(&items).expect("dispatch_batch");
        assert_eq!(handles.len(), 8);

        // Drain all 8.
        for _ in 0..8 {
            loop {
                match be.drain_one().expect("drain") {
                    Some(()) => break,
                    None => std::hint::spin_loop(),
                }
            }
        }
        // Verify every handle's result.
        for (i, h) in handles.iter().enumerate() {
            let r = be.wait_handle(*h, 1024).expect("wait").expect("ok");
            let v = u32::from_le_bytes(r[..4].try_into().unwrap());
            assert_eq!(v, i as u32 + i as u32 * 2);
        }

        unregister(id);
        std::fs::remove_file(&d).ok();
        std::fs::remove_file(&l).ok();
    }

    #[test]
    fn unknown_closure_id_publishes_err() {
        let (d, l) = temp_paths("unknown");
        let be = SharedMemoryLohBackend::create(0, &d, &l, 4, 4, 1).expect("create");
        let h = be.dispatch_marshal(0xDEAD_BEEF, &[1, 2]).expect("dispatch");
        be.drain_one().expect("drain").expect("had work");
        let r = be.wait_handle(h, 1024).expect("wait");
        match r {
            Err(msg) => assert!(msg.contains("no handler"), "got: {msg}"),
            Ok(_) => panic!("expected Err"),
        }
        std::fs::remove_file(&d).ok();
        std::fs::remove_file(&l).ok();
    }

    #[test]
    fn dispatch_one_panics() {
        let (d, l) = temp_paths("panic");
        let be = SharedMemoryLohBackend::create(0, &d, &l, 4, 4, 1).expect("create");
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            be.dispatch_one(Box::new(|| ()));
        }));
        assert!(r.is_err());
        std::fs::remove_file(&d).ok();
        std::fs::remove_file(&l).ok();
    }

    #[test]
    fn payload_too_large_rejected() {
        let (d, l) = temp_paths("oversize");
        let be = SharedMemoryLohBackend::create(0, &d, &l, 4, 4, 1).expect("create");
        let big = vec![0u8; LOH_ARGS_INLINE_BYTES + 1];
        let err = be.dispatch_marshal(0, &big).expect_err("oversize");
        match err {
            BackendError::Launch(msg) => assert!(msg.contains("exceeds slot capacity")),
            other => panic!("expected Launch, got {other:?}"),
        }
        std::fs::remove_file(&d).ok();
        std::fs::remove_file(&l).ok();
    }
}
