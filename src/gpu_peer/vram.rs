//! The device-resident block pool: VRAM the scheduler OWNS by index.
//!
//! The CPU never dereferences this memory - it addresses blocks by
//! index exactly like the host-side region addresses slots, and the
//! GPU touches the bytes through the resident opcodes. That index
//! discipline is the whole residency story: many small tasks can
//! re-read the same device-resident block while each task moves only
//! an 8-byte param header across the bus.
//!
//! Placement stays a scheduler decision, so the free list and block
//! states live CPU-side only; nothing here is shared memory.

use std::sync::Arc;

use cudarc::driver::sys as cu;
use cudarc::driver::{CudaContext, CudaStream};

use super::GpuPeerError;

/// A pool of fixed-size VRAM blocks addressed by index.
pub struct VramPool {
    base: u64,
    block_bytes: u32,
    blocks: u32,
    free: Vec<u32>,
    ctx: Arc<CudaContext>,
}

impl VramPool {
    /// Allocate `blocks x block_bytes` of device memory (zeroed).
    pub fn new(
        ctx: &Arc<CudaContext>,
        stream: &Arc<CudaStream>,
        block_bytes: u32,
        blocks: u32,
    ) -> Result<Self, GpuPeerError> {
        let total = block_bytes as usize * blocks as usize;
        let slice = stream
            .alloc_zeros::<u8>(total)
            .map_err(|e| GpuPeerError::Driver(format!("vram pool alloc: {e:?}")))?;
        // Leak for a stable raw device address; freed in Drop while
        // the context is still alive (field order in GpuPeer).
        let base = slice.leak();
        Ok(Self {
            base,
            block_bytes,
            blocks,
            free: (0..blocks).rev().collect(),
            ctx: Arc::clone(ctx),
        })
    }

    /// Raw device base address (kernel argument).
    #[inline]
    pub fn base(&self) -> u64 {
        self.base
    }
    /// Block size in bytes.
    #[inline]
    pub fn block_bytes(&self) -> u32 {
        self.block_bytes
    }
    /// Total block count.
    #[inline]
    pub fn blocks(&self) -> u32 {
        self.blocks
    }

    /// Raw device address of block `idx` - the base a full-grid
    /// kernel targets when a resident op needs the whole device
    /// rather than one lane's block.
    #[inline]
    pub fn block_ptr(&self, idx: u32) -> u64 {
        debug_assert!(idx < self.blocks);
        self.base + idx as u64 * self.block_bytes as u64
    }
    /// Blocks currently available.
    #[inline]
    pub fn free_blocks(&self) -> usize {
        self.free.len()
    }

    /// Claim a block index, or `None` when the pool is exhausted
    /// (the caller decides eviction policy; the pool never evicts
    /// silently).
    pub fn alloc(&mut self) -> Option<u32> {
        self.free.pop()
    }

    /// Claim `need` consecutive block indices and return the first,
    /// or `None` when no free run that long exists. Searches the
    /// free set in index order, so release order never decides
    /// whether a span is available; the lowest-indexed fitting run
    /// wins.
    pub fn alloc_span(&mut self, need: u32) -> Option<u32> {
        let first = span_start(&self.free, need)?;
        self.free.retain(|&b| b < first || b >= first + need);
        Some(first)
    }

    /// Return a block to the pool.
    pub fn release(&mut self, block: u32) {
        debug_assert!(block < self.blocks);
        debug_assert!(!self.free.contains(&block), "double release of block {block}");
        self.free.push(block);
    }
}

/// First index of the lowest run of `need` consecutive values in
/// `free` (any order), or `None`.
fn span_start(free: &[u32], need: u32) -> Option<u32> {
    if need == 0 || need as usize > free.len() {
        return None;
    }
    let mut sorted = free.to_vec();
    sorted.sort_unstable();
    let need_us = need as usize;
    let mut run_start = 0usize;
    for i in 1..=sorted.len() {
        let contiguous = i < sorted.len() && sorted[i] == sorted[i - 1] + 1;
        if !contiguous {
            if i - run_start >= need_us {
                return Some(sorted[run_start]);
            }
            run_start = i;
        }
    }
    None
}

impl Drop for VramPool {
    fn drop(&mut self) {
        if self.ctx.bind_to_thread().is_ok() {
            // SAFETY: `base` came from a leaked device allocation
            // owned solely by this pool; freeing once at drop with
            // the context bound is the paired release. Teardown
            // failure is unactionable.
            let _rc: cu::CUresult = unsafe { cu::cuMemFree_v2(self.base) };
        }
    }
}

#[cfg(test)]
mod tests {
    // Pool bookkeeping is pure CPU logic; exercised without a device
    // through the index math only (constructor needs a GPU, so the
    // free-list discipline is tested via the same Vec operations the
    // pool performs).
    #[test]
    fn free_list_indexing_is_lifo_and_complete() {
        let blocks = 8u32;
        let mut free: Vec<u32> = (0..blocks).rev().collect();
        assert_eq!(free.pop(), Some(0));
        assert_eq!(free.pop(), Some(1));
        free.push(0);
        assert_eq!(free.pop(), Some(0));
        // Drain fully: every index appears exactly once.
        let mut seen: Vec<u32> = std::mem::take(&mut free);
        seen.sort_unstable();
        assert_eq!(seen, (2..blocks).collect::<Vec<_>>());
    }

    #[test]
    fn span_start_ignores_free_list_order() {
        // Released out of order: a LIFO pop sequence could never
        // produce 4..8 consecutively, but the run exists.
        let free = vec![9u32, 4, 6, 5, 7, 2, 0];
        assert_eq!(super::span_start(&free, 4), Some(4));
        assert_eq!(super::span_start(&free, 1), Some(0));
        assert_eq!(super::span_start(&free, 2), Some(4));
        assert_eq!(super::span_start(&free, 5), None);
        assert_eq!(super::span_start(&free, 0), None);
        assert_eq!(super::span_start(&[], 1), None);
    }
}
