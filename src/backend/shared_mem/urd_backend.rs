//! `DispatchBackend` impl over [`super::urd::UrdDeque`] + MMF
//! latch arena. Push-based cross-process dispatch with one
//! per-thief mailbox cache line; the owner picks the target by
//! round-robin (or by an explicit thief index).

#![allow(clippy::missing_errors_doc)]

use std::path::PathBuf;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use super::chase_lev_backend::DispatchHandle;
use super::khpd::{KHPD_ARGS_INLINE_BYTES, LineItem};
use super::latch_mmf::{ERR, MmfLatchArena, SET, UNSET};
use super::pass_registry::{self, Pass};
use super::urd::{Drain, MAILBOX_ITEMS, UrdDeque};

use crate::backend::{Backend, BackendCapabilities, BackendError, DispatchBackend, KernelHandle};

/// `DispatchBackend` impl over a URD deque + MMF latch arena.
pub struct SharedMemoryUrdBackend {
    deque: UrdDeque,
    latches: MmfLatchArena,
    backend_id: u32,
    deque_path: PathBuf,
    latches_path: PathBuf,
    caps: BackendCapabilities,
    dispatched: AtomicU64,
    /// Owner-side staging buffer. Items accumulate here until
    /// `flush()` (or auto-flush at MAILBOX_ITEMS) publishes one
    /// mailbox-worth via [`UrdDeque::publish_round_robin`].
    pending: Mutex<Vec<LineItem>>,
}

impl SharedMemoryUrdBackend {
    /// Create the URD deque + latch arena.
    pub fn create(
        backend_id: u32,
        deque_path: impl Into<PathBuf>,
        latches_path: impl Into<PathBuf>,
        n_mailboxes: usize,
        latch_capacity: usize,
    ) -> Result<Self, BackendError> {
        let deque_path = deque_path.into();
        let latches_path = latches_path.into();
        let deque = UrdDeque::create(&deque_path, n_mailboxes)
            .map_err(|e| BackendError::Memory(format!("create URD deque: {e}")))?;
        let latches = MmfLatchArena::create(&latches_path, latch_capacity)
            .map_err(|e| BackendError::Memory(format!("create latch arena: {e}")))?;
        let caps = Self::caps_for(deque.n_mailboxes());
        Ok(Self {
            deque,
            latches,
            backend_id,
            deque_path,
            latches_path,
            caps,
            dispatched: AtomicU64::new(0),
            pending: Mutex::new(Vec::with_capacity(MAILBOX_ITEMS)),
        })
    }

    /// Attach to an existing URD deque + latch arena.
    pub fn open(
        backend_id: u32,
        deque_path: impl Into<PathBuf>,
        latches_path: impl Into<PathBuf>,
    ) -> Result<Self, BackendError> {
        let deque_path = deque_path.into();
        let latches_path = latches_path.into();
        let deque = UrdDeque::open(&deque_path)
            .map_err(|e| BackendError::Memory(format!("open URD deque: {e}")))?;
        let latches = MmfLatchArena::open(&latches_path)
            .map_err(|e| BackendError::Memory(format!("open latch arena: {e}")))?;
        let caps = Self::caps_for(deque.n_mailboxes());
        Ok(Self {
            deque,
            latches,
            backend_id,
            deque_path,
            latches_path,
            caps,
            dispatched: AtomicU64::new(0),
            pending: Mutex::new(Vec::with_capacity(MAILBOX_ITEMS)),
        })
    }

    fn caps_for(n_mailboxes: usize) -> BackendCapabilities {
        BackendCapabilities {
            simt_width: 1,
            max_threads_in_flight: (n_mailboxes * MAILBOX_ITEMS).max(1) as u32,
            launch_latency_ns: 150,
            h2d_bw_bytes_per_sec: 0,
        }
    }

    /// Path of the URD deque file.
    pub fn deque_path(&self) -> &std::path::Path {
        &self.deque_path
    }

    /// Path of the latch arena file.
    pub fn latches_path(&self) -> &std::path::Path {
        &self.latches_path
    }

    /// Total items dispatched.
    pub fn dispatched(&self) -> u64 {
        self.dispatched.load(Ordering::Relaxed)
    }

    /// Owner-side: stage one item + auto-flush at MAILBOX_ITEMS.
    /// Returns a [`DispatchHandle`] the caller waits on.
    pub fn dispatch_marshal(
        &self,
        closure_id: u32,
        args: &[u8],
    ) -> Result<DispatchHandle, BackendError> {
        if args.len() > KHPD_ARGS_INLINE_BYTES {
            return Err(BackendError::Launch(format!(
                "URD args length {} exceeds slot capacity {}",
                args.len(),
                KHPD_ARGS_INLINE_BYTES
            )));
        }
        let latch_offset = self.latches.alloc();
        let item = LineItem::new(closure_id, latch_offset, args)
            .map_err(|e| BackendError::Launch(format!("URD build item: {e:?}")))?;
        let mut p = self.pending.lock().expect("URD pending poisoned");
        p.push(item);
        if p.len() >= MAILBOX_ITEMS {
            let batch = p.drain(..).collect::<Vec<_>>();
            drop(p);
            self.deque
                .publish_round_robin(&batch)
                .map_err(|e| BackendError::Launch(format!("URD publish: {e:?}")))?;
        }
        self.dispatched.fetch_add(1, Ordering::Relaxed);
        Ok(DispatchHandle { latch_offset })
    }

    /// Owner-side: stage N items, then publish in round-robin
    /// batches of MAILBOX_ITEMS items each. This is the
    /// producer-fast entry point.
    pub fn dispatch_marshal_batch(
        &self,
        items: &[(u32, &[u8])],
    ) -> Result<Vec<DispatchHandle>, BackendError> {
        let mut handles = Vec::with_capacity(items.len());
        // Build all the LineItems up-front so we don't lock /
        // unlock per item.
        let mut line_items = Vec::with_capacity(items.len());
        for (closure_id, args) in items {
            if args.len() > KHPD_ARGS_INLINE_BYTES {
                return Err(BackendError::Launch(format!(
                    "URD args length {} exceeds slot capacity {}",
                    args.len(),
                    KHPD_ARGS_INLINE_BYTES
                )));
            }
            let latch_offset = self.latches.alloc();
            let item = LineItem::new(*closure_id, latch_offset, args)
                .map_err(|e| BackendError::Launch(format!("URD build item: {e:?}")))?;
            line_items.push(item);
            handles.push(DispatchHandle { latch_offset });
        }
        // Publish in MAILBOX_ITEMS-sized chunks via round-robin.
        for chunk in line_items.chunks(MAILBOX_ITEMS) {
            self.deque
                .publish_round_robin(chunk)
                .map_err(|e| BackendError::Launch(format!("URD publish: {e:?}")))?;
        }
        self.dispatched
            .fetch_add(items.len() as u64, Ordering::Relaxed);
        Ok(handles)
    }

    /// Owner-side: explicit flush of any pending staged items.
    pub fn flush(&self) -> Result<usize, BackendError> {
        let mut p = self.pending.lock().expect("URD pending poisoned");
        if p.is_empty() {
            return Ok(0);
        }
        let batch = p.drain(..).collect::<Vec<_>>();
        drop(p);
        for chunk in batch.chunks(MAILBOX_ITEMS) {
            self.deque
                .publish_round_robin(chunk)
                .map_err(|e| BackendError::Launch(format!("URD publish: {e:?}")))?;
        }
        Ok(batch.len())
    }

    /// Peer-side: drain one mailbox (`mailbox_idx`) and execute
    /// every item via the local pass_registry.
    pub fn drain_mailbox(&self, mailbox_idx: usize) -> Result<Option<usize>, BackendError> {
        let result = match self.deque.drain_mailbox(mailbox_idx) {
            Drain::Success(r) => r,
            Drain::Empty => return Ok(None),
        };
        for i in 0..result.n_items {
            let item = result.items[i];
            let pass = Pass {
                closure_id: item.closure_id,
                args: item.args_inline.to_vec(),
            };
            match pass_registry::execute(&pass) {
                Ok(reply) => self
                    .latches
                    .publish(item.latch_offset, &reply)
                    .map_err(|e| {
                        BackendError::Launch(format!("URD latch publish: {e:?}"))
                    })?,
                Err(err) => {
                    let msg = err.to_string();
                    let bytes = msg.as_bytes();
                    let n = bytes.len().min(64);
                    self.latches
                        .publish_err(item.latch_offset, &bytes[..n])
                        .map_err(|e| {
                            BackendError::Launch(format!("URD latch publish_err: {e:?}"))
                        })?;
                }
            }
        }
        Ok(Some(result.n_items))
    }

    /// Originator-side non-blocking poll on a single dispatch handle.
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

impl DispatchBackend for SharedMemoryUrdBackend {
    fn id(&self) -> Backend {
        Backend::SharedMemoryWorker {
            backend_id: self.backend_id,
        }
    }

    fn capabilities(&self) -> BackendCapabilities {
        self.caps
    }

    fn dispatch_parallel_for(&self, _count: u32, _work: &(dyn Fn(u32) + Send + Sync)) {}

    fn dispatch_one(&self, _work: Box<dyn FnOnce() + Send>) {
        panic!(
            "SharedMemoryUrdBackend does not support dispatch_one; \
             use dispatch_marshal / dispatch_marshal_batch"
        );
    }

    fn register_kernel(&self, name: &str, _source: &[u8]) -> Result<KernelHandle, BackendError> {
        let id = pass_registry::hash_name(name);
        Ok(KernelHandle(id as u64))
    }

    fn dispatch_kernel(
        &self,
        _handle: KernelHandle,
        _count: u32,
        _args: &[crate::backend::KernelArg<'_>],
    ) -> Result<(), BackendError> {
        // Same rationale as KHPD: 8-byte inline args don't fit the
        // generic KernelArg wire envelope. Callers route through
        // dispatch_marshal / dispatch_marshal_batch with raw 8-byte
        // payloads.
        Err(BackendError::NotSupported)
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
        d.push(format!("flynnel_urd_be_d_{pid}_{nonce}_{label}.bin"));
        l.push(format!("flynnel_urd_be_l_{pid}_{nonce}_{label}.bin"));
        (d, l)
    }

    #[test]
    fn dispatch_marshal_batch_round_trips_across_mailboxes() {
        let (d, l) = temp_paths("batch");
        // n_mailboxes=4 so 6 items (2 batches of MAILBOX_ITEMS=3
        // each) land in DIFFERENT mailboxes via round-robin. This
        // eliminates the `publish_to` spin-wait that would deadlock
        // under parallel-test load when only a single mailbox is
        // available and no concurrent drain thread is running. The
        // test drains all 4 mailboxes sequentially after dispatch.
        let be = SharedMemoryUrdBackend::create(0, &d, &l, 4, 64).expect("create");
        let id = hash_name("flynnel.test.urd.adder");
        register(id, |args| {
            let a = u32::from_le_bytes(args[0..4].try_into().unwrap());
            let b = u32::from_le_bytes(args[4..8].try_into().unwrap());
            Ok((a + b).to_le_bytes().to_vec())
        });

        let mut payloads = Vec::with_capacity(6);
        for i in 0..6u32 {
            let mut p = [0u8; 8];
            p[..4].copy_from_slice(&i.to_le_bytes());
            p[4..].copy_from_slice(&(i * 2).to_le_bytes());
            payloads.push(p);
        }
        let items: Vec<(u32, &[u8])> =
            payloads.iter().map(|p| (id, p.as_slice())).collect();
        // With 4 mailboxes and 2 batches, round-robin lands batch 0
        // in mailbox 0 and batch 1 in mailbox 1. Neither publish
        // call spin-waits because both target mailboxes are EMPTY
        // (fresh init).
        let handles = be.dispatch_marshal_batch(&items).expect("dispatch");
        assert_eq!(handles.len(), 6);

        // Drain every mailbox (sequential; no concurrency required).
        let mut total_drained = 0usize;
        for mb in 0..4 {
            if let Some(n) = be.drain_mailbox(mb).expect("drain") {
                total_drained += n;
            }
        }
        assert_eq!(total_drained, 6, "all 6 items must drain across mailboxes 0..3");

        // Every handle's latch was published by drain_mailbox; wait
        // resolves immediately. The handle->mailbox mapping is
        // round-robin-positional, so handles[0..3] land in
        // mailbox 0's published batch and handles[3..6] in mailbox 1.
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
    fn dispatch_one_panics() {
        let (d, l) = temp_paths("panic");
        let be = SharedMemoryUrdBackend::create(0, &d, &l, 1, 8).expect("create");
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            be.dispatch_one(Box::new(|| ()));
        }));
        assert!(r.is_err());
        std::fs::remove_file(&d).ok();
        std::fs::remove_file(&l).ok();
    }
}
