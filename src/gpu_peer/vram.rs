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

    /// Return a block to the pool.
    pub fn release(&mut self, block: u32) {
        debug_assert!(block < self.blocks);
        debug_assert!(!self.free.contains(&block), "double release of block {block}");
        self.free.push(block);
    }
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
}
