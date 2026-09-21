//! URD: UMWAIT Rendezvous Deque, MMF-backed.
//!
//! Per-thief mailbox cache lines instead of a shared deque. Each
//! mailbox is one 64-byte line carrying state (8 B) + `MAILBOX_ITEMS
//! = 3` line items (48 B) + 8 B trailing padding. The owner picks a
//! mailbox by round-robin (or by an explicit thief target) and
//! writes the items in; the addressed thief observes the
//! cache-line transition and reads its items. There is **no
//! shared head/tail counter on the steal path** - each thief has
//! its own state byte and never CASes a contended atomic.
//!
//! ## Wait strategy: runtime dispatch on WAITPKG
//!
//! Thieves idle on their mailbox's state byte. The wait primitive
//! is chosen at runtime by [`crate::cpu_info::has_waitpkg`]:
//!
//! - **WAITPKG available** (Intel Tremont/Tiger Lake+, AMD Zen 5+):
//!   thief uses `UMONITOR` + `UMWAIT` to halt until the cache line
//!   transitions OR a TSC deadline fires. Power-efficient; the
//!   thief doesn't burn pipeline slots polling.
//! - **WAITPKG not available**: thief uses [`std::hint::spin_loop`]
//!   (PAUSE on x86) in a tight Acquire-load loop on the state byte.
//!
//! `MONITORX` + `MWAITX` is implemented as a third strategy and is
//! not picked automatically: measured against the spin it is slower
//! in three cells of four. See [`WaitStrategy::pick`] for the table
//! and for what the measurement does not cover.
//!
//! Both branches end the wait when `state` carries the ready bit
//! for the expected epoch.
//!
//! ## Why this shape vs Chase-Lev / LOH / KHPD
//!
//! Chase-Lev / LOH / KHPD are *pull-based*: thieves CAS the deque
//! to discover work. URD is *push-based*: the owner picks the
//! target thief by writing its mailbox. Two architectural
//! consequences:
//!
//! 1. **No CAS contention at the steal site.** N thieves on a
//!    shared deque CAS the same head; under contention each push
//!    racing N thieves can take O(N) failed CASes. URD's per-thief
//!    mailbox has zero contention because the owner is the only
//!    writer and the thief is the only reader.
//! 2. **Owner-controlled distribution.** Round-robin / locality-
//!    aware / variance-driven targeting is the owner's choice, not
//!    a thief's victim-pick. The orchestrator gets explicit say in
//!    which thief does what.
//!
//! Trade-offs: the thief is bound to one mailbox (no work-stealing
//! between mailboxes), the owner pays one mailbox-spin per publish
//! if the previous batch hasn't been consumed yet, and the
//! per-mailbox slot count is fixed.

#![allow(clippy::missing_errors_doc)]

use std::fs::{File, OpenOptions};
use std::io;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

use memmap2::{MmapMut, MmapOptions};

use super::khpd::LineItem;
use crate::cpu_info::{has_monitorx, has_waitpkg};

/// Magic byte sequence marking a valid URD file. ASCII "FURD".
pub const URD_MAGIC: u64 = 0x4655_5244_0000_0001;

/// Cache line size; one mailbox per cache line.
pub const URD_MAILBOX_SIZE: usize = 64;

/// Items per mailbox: state (8 B) + 3 * 16 = 56 B; 8 B trailing pad.
pub const MAILBOX_ITEMS: usize = 3;

/// Wait strategy chosen at runtime per CPUID.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WaitStrategy {
    /// `std::hint::spin_loop` (PAUSE on x86). Universally available
    /// fallback.
    PauseSpin,
    /// `UMONITOR` + `UMWAIT`. Available on Intel Tremont/Tiger Lake+
    /// and AMD Zen 5+; detected via CPUID leaf 7 ECX bit 5.
    Waitpkg,
    /// `MONITORX` + `MWAITX`. Available on AMD from Excavator
    /// onward, so on every Zen; detected via CPUID Fn8000_0001
    /// ECX bit 29.
    Monitorx,
}

impl WaitStrategy {
    /// Returns the wait strategy for this host: WAITPKG where the
    /// silicon has it, otherwise the PAUSE-spin.
    ///
    /// # MONITORX is implemented and is deliberately not picked here
    ///
    /// The thief's spin is already the fastest wake there is, because
    /// it never stopped looking, and the measurement says a monitor
    /// wait does not beat it. On a Ryzen 9 7900X, best median of
    /// three runs of `benches/urd_thief_wait.rs`, publish to drained:
    ///
    /// ```text
    ///                PAUSE-spin   MONITORX
    ///   idle   50us     1.54 us    2.00 us
    ///   idle  500us     1.69 us    1.60 us
    ///   load   50us     1.66 us    1.92 us
    ///   load  500us     1.63 us    1.85 us
    /// ```
    ///
    /// Slower in three cells of four, so it does not pass a rule that
    /// nothing may get slower anywhere, and it is not the default.
    ///
    /// What a monitor wait would buy the thief is the core it stops
    /// occupying, and that bench cannot see it: it times the thief's
    /// own wake, not the throughput of whatever else could have used
    /// the core meanwhile. Establishing the benefit needs a harness
    /// that measures the co-runner, and until one exists the trade is
    /// unmeasured in the direction that would justify it.
    ///
    /// [`UrdDeque::set_wait_strategy`] still selects it, which is how
    /// the numbers above were taken.
    pub fn pick() -> Self {
        if has_waitpkg() {
            Self::Waitpkg
        } else {
            Self::PauseSpin
        }
    }
}

/// State word packed-bit layout: top 32 bits = epoch, bits 16..32 =
/// n_items, bits 0..16 = claim (0 = EMPTY, 1 = READY).
const STATE_EMPTY: u64 = 0;
const CLAIM_READY: u64 = 1;

/// Header sits at file offset 0.
#[repr(C, align(64))]
pub struct UrdHeader {
    /// Magic constant.
    pub magic: u64,
    /// Number of mailboxes (one per thief).
    pub n_mailboxes: u64,
    /// Pid of the owner process; informational. Cleared on
    /// `close_owner()`.
    pub owner_pid: AtomicU64,
    /// Shutdown epoch counter.
    pub epoch: AtomicU64,
    /// Padding to push `rr_cursor` to its own cache line.
    pub _pad_meta: [u8; 24],
    /// Round-robin cursor the owner uses to pick the next target
    /// mailbox when no explicit target is supplied.
    pub rr_cursor: AtomicU64,
    /// Padding to round to two cache lines.
    pub _pad_rr: [u8; 56],
}

/// One per-thief mailbox cache line.
#[repr(C, align(64))]
pub struct Mailbox {
    /// State word: `(epoch:32) << 32 | (n_items:16) << 16 | claim:16`.
    pub state: AtomicU64,
    /// Inline items.
    pub items: [LineItem; MAILBOX_ITEMS],
    /// Trailing padding.
    pub _pad: [u8; 8],
}

/// Total file size for a URD with `n_mailboxes` mailboxes.
pub const fn urd_file_size(n_mailboxes: usize) -> usize {
    std::mem::size_of::<UrdHeader>() + n_mailboxes * URD_MAILBOX_SIZE
}

/// Outcome of [`UrdDeque::publish_to`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PublishError {
    /// Caller-supplied target mailbox index is out of range.
    BadTarget(usize),
    /// More items than [`MAILBOX_ITEMS`] passed in one publish call.
    TooManyItems,
    /// Args payload exceeded [`super::khpd::KHPD_ARGS_INLINE_BYTES`].
    PayloadTooLarge,
}

/// Outcome of [`UrdDeque::drain_mailbox`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Drain {
    /// Got items.
    Success(DrainResult),
    /// Mailbox empty (no published items past this thief's last
    /// consume).
    Empty,
}

/// Items pulled from a mailbox.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DrainResult {
    /// How many of the `items` slots are filled.
    pub n_items: usize,
    /// The items.
    pub items: [LineItem; MAILBOX_ITEMS],
}

/// MMF-backed URD. Single owner, N pre-configured thieves.
pub struct UrdDeque {
    _file: File,
    mmap: MmapMut,
    n_mailboxes: usize,
    wait_strategy: WaitStrategy,
}

// SAFETY: All fields are Send. The mmap handle is Send+Sync per
// memmap2. Every mailbox access goes through the per-mailbox
// state-atomic protocol.
unsafe impl Send for UrdDeque {}
// SAFETY: same justification as the Send impl directly above.
unsafe impl Sync for UrdDeque {}

impl UrdDeque {
    /// Create a fresh URD file with `n_mailboxes` mailboxes (one
    /// per intended thief). Minimum 1. The round-robin cursor
    /// reduces modulo `n_mailboxes` (no pow2 requirement so a
    /// single-thief bench can use n=1).
    pub fn create<P: AsRef<Path>>(path: P, n_mailboxes: usize) -> io::Result<Self> {
        let n_mailboxes = n_mailboxes.max(1);
        let size = urd_file_size(n_mailboxes);

        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path.as_ref())?;
        file.set_len(size as u64)?;

        // SAFETY: `map_mut` is unsafe because the kernel cannot
        // prevent another process from truncating the file. This
        // call site upholds the soundness contract by writing only
        // through the per-mailbox state-atomic protocol.
        let mut mmap = unsafe { MmapOptions::new().len(size).map_mut(&file)? };

        let header_ptr = mmap.as_mut_ptr() as *mut UrdHeader;
        // SAFETY: mmap is page-aligned (>= 64-byte alignment); the
        // map covers `urd_file_size(n_mailboxes)` bytes by
        // construction.
        unsafe {
            (*header_ptr).magic = URD_MAGIC;
            (*header_ptr).n_mailboxes = n_mailboxes as u64;
            (*header_ptr).owner_pid = AtomicU64::new(std::process::id() as u64);
            (*header_ptr).epoch = AtomicU64::new(0);
            std::ptr::write_bytes((*header_ptr)._pad_meta.as_mut_ptr(), 0, 24);
            (*header_ptr).rr_cursor = AtomicU64::new(0);
            std::ptr::write_bytes((*header_ptr)._pad_rr.as_mut_ptr(), 0, 56);
        }

        // Zero all mailboxes (state == STATE_EMPTY).
        let mailboxes_start = std::mem::size_of::<UrdHeader>();
        // SAFETY: write_bytes covers the unwritten tail of the map.
        unsafe {
            std::ptr::write_bytes(
                mmap.as_mut_ptr().add(mailboxes_start),
                0,
                n_mailboxes * URD_MAILBOX_SIZE,
            );
        }

        mmap.flush()?;
        let wait_strategy = WaitStrategy::pick();
        Ok(Self {
            _file: file,
            mmap,
            n_mailboxes,
            wait_strategy,
        })
    }

    /// Open an existing URD file.
    pub fn open<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(path.as_ref())?;
        let size = file.metadata()?.len() as usize;
        if size < std::mem::size_of::<UrdHeader>() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "urd file too small",
            ));
        }
        // SAFETY: same protocol-only-access justification as `create`.
        let mmap = unsafe { MmapOptions::new().len(size).map_mut(&file)? };
        let header_ptr = mmap.as_ptr() as *const UrdHeader;
        // SAFETY: map size verified to cover header.
        let (magic, n_mailboxes) =
            unsafe { ((*header_ptr).magic, (*header_ptr).n_mailboxes as usize) };
        if magic != URD_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("urd magic mismatch {magic:#x}"),
            ));
        }
        if n_mailboxes == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "urd n_mailboxes must be >= 1",
            ));
        }
        if size < urd_file_size(n_mailboxes) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "urd file size below header expected",
            ));
        }
        let wait_strategy = WaitStrategy::pick();
        Ok(Self {
            _file: file,
            mmap,
            n_mailboxes,
            wait_strategy,
        })
    }

    /// Number of configured mailboxes.
    pub fn n_mailboxes(&self) -> usize {
        self.n_mailboxes
    }

    /// Wait strategy this URD instance picked (per CPUID).
    pub fn wait_strategy(&self) -> WaitStrategy {
        self.wait_strategy
    }

    /// Override the wait strategy for benches and tests that need to
    /// A/B against the auto-picked one.
    ///
    /// The thief's choice is the one worth measuring rather than
    /// assuming: a `PAUSE`-spin is already the fastest wake there is,
    /// because it never stopped looking, so replacing it with a
    /// monitor wait trades some of that latency for a core it stops
    /// occupying. Which way that comes out is a property of the host
    /// and cannot be read off the instruction set.
    ///
    /// # Panics
    ///
    /// Panics on a strategy this host cannot execute, since the
    /// alternative is a `#UD` inside the steal path with nothing to
    /// say which caller asked for it.
    pub fn set_wait_strategy(&mut self, strategy: WaitStrategy) {
        match strategy {
            WaitStrategy::Waitpkg => assert!(
                has_waitpkg(),
                "this host has no WAITPKG, so UMONITOR would be an illegal instruction"
            ),
            WaitStrategy::Monitorx => assert!(
                has_monitorx(),
                "this host has no MONITORX, so MONITORX would be an illegal instruction"
            ),
            WaitStrategy::PauseSpin => {}
        }
        self.wait_strategy = strategy;
    }

    /// Owner pid at create time, or 0 after `close_owner()`.
    pub fn owner_pid(&self) -> u64 {
        self.header().owner_pid.load(Ordering::Acquire)
    }

    /// Owner shutdown: zero pid + advance epoch.
    pub fn close_owner(&self) {
        self.header().owner_pid.store(0, Ordering::Release);
        self.header().epoch.fetch_add(1, Ordering::Release);
    }

    fn header(&self) -> &UrdHeader {
        // SAFETY: header is at the start of the map.
        unsafe { &*(self.mmap.as_ptr() as *const UrdHeader) }
    }

    fn mailbox_ptr(&self, idx: usize) -> *mut Mailbox {
        let off = std::mem::size_of::<UrdHeader>() + idx * URD_MAILBOX_SIZE;
        // SAFETY: idx < n_mailboxes by caller contract; `off` is
        // in-bounds + 64-byte aligned.
        unsafe { self.mmap.as_ptr().add(off) as *mut Mailbox }
    }

    fn mailbox(&self, idx: usize) -> &Mailbox {
        // SAFETY: same as mailbox_ptr.
        unsafe { &*self.mailbox_ptr(idx) }
    }

    /// Owner-side: publish `items` to mailbox `target`. Spins until
    /// the mailbox is EMPTY (the previous batch has been consumed),
    /// writes items, Release-stores state to READY. Returns the
    /// number of items published.
    pub fn publish_to(
        &self,
        target: usize,
        items: &[LineItem],
    ) -> Result<usize, PublishError> {
        if target >= self.n_mailboxes {
            return Err(PublishError::BadTarget(target));
        }
        if items.len() > MAILBOX_ITEMS {
            return Err(PublishError::TooManyItems);
        }
        if items.is_empty() {
            return Ok(0);
        }
        let mb = self.mailbox(target);
        // Spin-wait for the mailbox to be EMPTY (previous batch
        // consumed). The owner is on the WRITE side so a brief
        // PAUSE-spin is the right primitive here regardless of the
        // thief's WAITPKG availability.
        loop {
            let s = mb.state.load(Ordering::Acquire);
            if s == STATE_EMPTY {
                break;
            }
            std::hint::spin_loop();
        }
        // Write items.
        // SAFETY: mailbox is in-bounds + aligned; we have exclusive
        // access until the Release-store below transitions state to
        // READY.
        unsafe {
            let mb_ptr = self.mailbox_ptr(target);
            for (i, item) in items.iter().enumerate() {
                (*mb_ptr).items[i] = *item;
            }
        }
        let epoch = self.header().epoch.load(Ordering::Relaxed);
        let new_state = (epoch << 32) | ((items.len() as u64) << 16) | CLAIM_READY;
        mb.state.store(new_state, Ordering::Release);
        Ok(items.len())
    }

    /// Owner-side: publish `items` to the next round-robin mailbox.
    /// Returns `(target, n_published)`.
    pub fn publish_round_robin(
        &self,
        items: &[LineItem],
    ) -> Result<(usize, usize), PublishError> {
        let cursor = self.header().rr_cursor.fetch_add(1, Ordering::Relaxed) as usize;
        // Use modulo (not bit-mask) so non-pow2 mailbox counts work;
        // common case is `n_mailboxes` being a small constant the
        // compiler reduces to a strength-reduced multiply.
        let target = cursor % self.n_mailboxes;
        let n = self.publish_to(target, items)?;
        Ok((target, n))
    }

    /// Thief-side: drain own mailbox if it has READY items.
    /// `mailbox_idx` is the thief's pre-assigned mailbox. Returns
    /// [`Drain::Empty`] when the state byte is EMPTY (no work
    /// published yet).
    pub fn drain_mailbox(&self, mailbox_idx: usize) -> Drain {
        if mailbox_idx >= self.n_mailboxes {
            return Drain::Empty;
        }
        let mb = self.mailbox(mailbox_idx);
        let s = mb.state.load(Ordering::Acquire);
        if s & 0xFFFF != CLAIM_READY {
            return Drain::Empty;
        }
        let n_items = ((s >> 16) & 0xFFFF) as usize;
        let n_items = n_items.min(MAILBOX_ITEMS);
        // SAFETY: state's READY bit is set; the publisher's Release-
        // store synchronizes-with our Acquire-load above so item
        // bytes are visible.
        let result = unsafe {
            DrainResult {
                n_items,
                items: (*self.mailbox_ptr(mailbox_idx)).items,
            }
        };
        // Release the mailbox: state -> EMPTY. The owner's next
        // publish_to(target) spin sees EMPTY and writes.
        mb.state.store(STATE_EMPTY, Ordering::Release);
        Drain::Success(result)
    }

    /// Thief-side: block (per the host's [`WaitStrategy`]) until
    /// the mailbox transitions to READY, then drain it.
    ///
    /// On WAITPKG-capable hardware the thief uses `UMONITOR` +
    /// `UMWAIT` to halt; otherwise it uses `PAUSE`-spin. The
    /// deadline is expressed as the absolute TSC value at which
    /// `UMWAIT` should return even if the line hasn't transitioned;
    /// `u64::MAX` means "no deadline" (wait indefinitely - protocol
    /// risk if the owner never publishes).
    pub fn wait_and_drain(&self, mailbox_idx: usize, deadline_tsc: u64) -> Drain {
        if mailbox_idx >= self.n_mailboxes {
            return Drain::Empty;
        }
        let mb = self.mailbox(mailbox_idx);
        let state_addr = (&raw const mb.state).cast::<u8>();
        loop {
            let s = mb.state.load(Ordering::Acquire);
            if s & 0xFFFF == CLAIM_READY {
                break;
            }
            match self.wait_strategy {
                WaitStrategy::PauseSpin => std::hint::spin_loop(),
                WaitStrategy::Waitpkg => {
                    // SAFETY: WAITPKG was confirmed available by
                    // CPUID at construction time. `state_addr` is a
                    // valid pointer into the mmap; UMONITOR arms the
                    // hardware monitor on its cache line. UMWAIT
                    // suspends until the monitor fires, an interrupt
                    // arrives, or the TSC deadline is reached. The
                    // double-check on the next loop iteration
                    // re-validates the state byte.
                    unsafe { wait_with_waitpkg(state_addr, deadline_tsc) };
                }
                WaitStrategy::Monitorx => {
                    // SAFETY: MONITORX was confirmed available by
                    // CPUID at construction time. Same pointer and
                    // same double-check as the WAITPKG arm; MWAITX
                    // suspends on the same terms, and returning early
                    // costs one more turn of this loop.
                    unsafe { wait_with_monitorx(state_addr, deadline_tsc) };
                }
            }
        }
        self.drain_mailbox(mailbox_idx)
    }

    /// Force any dirty pages to disk.
    pub fn flush_to_disk(&self) -> io::Result<()> {
        self.mmap.flush()
    }
}

/// `UMONITOR` + `UMWAIT` wait primitive. Halts the calling logical
/// CPU until the monitored cache line transitions OR the TSC
/// reaches `deadline_tsc` (whichever comes first). Pass
/// `deadline_tsc = u64::MAX` for "no deadline".
///
/// # Safety
///
/// The caller must have confirmed WAITPKG is available via
/// [`crate::cpu_info::has_waitpkg`] - executing `UMONITOR` /
/// `UMWAIT` on hardware without WAITPKG raises an illegal-
/// instruction trap (`#UD`).
///
/// `state_addr` must be a valid pointer into accessible memory;
/// `UMONITOR` reads no payload, only the address.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn wait_with_waitpkg(state_addr: *const u8, deadline_tsc: u64) {
    use std::arch::asm;
    let lo = deadline_tsc as u32;
    let hi = (deadline_tsc >> 32) as u32;
    // Two separate asm blocks: UMONITOR needs the address in RAX,
    // UMWAIT needs EAX (low half of RAX) for the deadline low
    // dword. We cannot bind RAX and EAX to different values in
    // one asm! call, so we split. The monitor stays armed across
    // the second asm block; UMWAIT in C0.1 (hint = 1) is the light
    // wait state with low wake latency.
    //
    // SAFETY: caller-asserted WAITPKG availability + valid pointer.
    unsafe {
        asm!(
            "umonitor rax",
            in("rax") state_addr,
            options(nostack, preserves_flags),
        );
        asm!(
            "umwait {hint:e}",
            hint = in(reg) 1u32,
            in("eax") lo,
            in("edx") hi,
            options(nostack),
        );
    }
}

#[cfg(not(target_arch = "x86_64"))]
#[inline(always)]
unsafe fn wait_with_waitpkg(_state_addr: *const u8, _deadline_tsc: u64) {
    // Non-x86_64: WAITPKG cannot be available; this function is
    // never called on those targets (the WaitStrategy::Waitpkg
    // branch is gated on has_waitpkg() which returns false).
    std::hint::spin_loop();
}

/// `MONITORX` + `MWAITX` wait primitive, AMD's user-mode pair.
/// Halts the calling logical core until the monitored line
/// transitions, an interrupt arrives, or the timer runs out.
///
/// Takes the same absolute `deadline_tsc` the WAITPKG primitive
/// takes and converts it, because MWAITX counts a relative number
/// of EBX units rather than reaching a TSC value. The conversion
/// passes the remaining cycle count through unscaled, which is
/// deliberate: an EBX unit is never longer than a TSC cycle (on a
/// Ryzen 9 7900X it is half of one), so the wait can only end at or
/// before the deadline, never after it. Undershooting costs another
/// turn of the caller's loop; overshooting would hold a thief past
/// a deadline it was given. That asymmetry is why this does not try
/// to measure the ratio and correct for it.
///
/// # Safety
///
/// Caller must have confirmed MONITORX via
/// [`crate::cpu_info::has_monitorx`] - executing `MONITORX` /
/// `MWAITX` without it raises an illegal-instruction trap (`#UD`).
///
/// `state_addr` must be a valid pointer into accessible memory;
/// `MONITORX` reads no payload, only the address.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn wait_with_monitorx(state_addr: *const u8, deadline_tsc: u64) {
    use std::arch::asm;

    // SAFETY: `_rdtsc` is a no-side-effect read of the TSC counter.
    let now = unsafe { std::arch::x86_64::_rdtsc() };
    // A deadline already past asks for the shortest wait the timer
    // can express rather than the longest, which is what a zero
    // would otherwise be read as.
    let remaining = deadline_tsc.saturating_sub(now).max(1);
    let ask = u32::try_from(remaining).unwrap_or(u32::MAX);

    // Split for the same reason the WAITPKG pair is split: MONITORX
    // needs the address in RAX and MWAITX needs EAX for its hints,
    // and one asm! call cannot bind both. The monitor stays armed
    // across the two blocks.
    //
    // Encoded as bytes because the mnemonics need a target feature
    // this crate does not set; the encodings are fixed.
    //
    // SAFETY: caller-asserted MONITORX availability + valid pointer.
    unsafe {
        // The zeros arrive as operands rather than through `xor`,
        // because `xor` writes flags and this block promises not to.
        // MONITORX itself leaves them alone, so the promise holds
        // only while no instruction here breaks it.
        asm!(
            ".byte 0x0f, 0x01, 0xfa",
            in("rax") state_addr,
            in("ecx") 0u32,
            in("edx") 0u32,
            options(nostack, preserves_flags),
        );
        // EAX = 0 requests C0, matching the WAITPKG arm's C0.1 hint;
        // ECX bit 1 enables the EBX timer; EBX carries the count.
        //
        // rbx is reserved by LLVM and cannot be an operand, so it is
        // saved and restored inside the block. That is why this one
        // does not claim `nostack`: it pushes, and a red zone below
        // rsp would be live under that promise. MWAITX reports
        // timer-versus-wake exit in CF, so flags are not preserved
        // either.
        asm!(
            "push rbx",
            "mov ebx, {ask:e}",
            ".byte 0x0f, 0x01, 0xfb",
            "pop rbx",
            ask = in(reg) ask,
            inout("eax") 0u32 => _,
            inout("ecx") 2u32 => _,
        );
    }
}

#[cfg(not(target_arch = "x86_64"))]
#[inline(always)]
unsafe fn wait_with_monitorx(_state_addr: *const u8, _deadline_tsc: u64) {
    // Non-x86_64: MONITORX cannot be available, for the same reason
    // the WAITPKG stub above is never called.
    std::hint::spin_loop();
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::Ordering as O;
    use std::thread;

    fn temp_path(name: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        let pid = std::process::id();
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        p.push(format!("flynnel_urd_{pid}_{nonce}_{name}.bin"));
        p
    }

    /// Delete a test's backing file, reporting a failure rather than
    /// discarding it. A leaked mapping shows up here first, and a
    /// silent drop would leave the next run to find the file instead.
    fn remove_temp(path: &std::path::Path) {
        if let Err(e) = std::fs::remove_file(path) {
            eprintln!("urd test: could not remove {}: {e}", path.display());
        }
    }

    fn item(id: u32) -> LineItem {
        LineItem::new(id, u32::MAX, &id.to_le_bytes()).expect("item")
    }

    #[test]
    fn create_open_round_trips_header() {
        let path = temp_path("create_open");
        let _u = UrdDeque::create(&path, 4).expect("create");
        let o = UrdDeque::open(&path).expect("open");
        assert_eq!(o.n_mailboxes(), 4);
        assert_eq!(o.owner_pid(), std::process::id() as u64);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn wait_strategy_matches_host() {
        let path = temp_path("strategy");
        let u = UrdDeque::create(&path, 2).expect("create");
        let s = u.wait_strategy();
        // WAITPKG where the host has it, the spin otherwise. MONITORX
        // is never picked, and that is asserted below rather than
        // left implicit, because it is a measurement result and not
        // an oversight.
        if has_waitpkg() {
            assert_eq!(s, WaitStrategy::Waitpkg);
        } else {
            assert_eq!(s, WaitStrategy::PauseSpin);
        }
        remove_temp(&path);
    }

    #[test]
    fn monitorx_is_available_to_ask_for_and_is_never_chosen_on_its_own() {
        // It loses to the spin on latency, so defaulting to it would
        // be a regression; it stays reachable because the bench that
        // found that out selects it, and because the trade it makes
        // has not been measured in the direction that would justify
        // it. A later measurement changes `pick`, and this test is
        // what will notice.
        assert_ne!(WaitStrategy::pick(), WaitStrategy::Monitorx);

        if has_monitorx() {
            let path = temp_path("strategy_monitorx");
            let mut u = UrdDeque::create(&path, 2).expect("create");
            u.set_wait_strategy(WaitStrategy::Monitorx);
            assert_eq!(u.wait_strategy(), WaitStrategy::Monitorx);
            remove_temp(&path);
        }
    }

    #[test]
    fn the_thief_never_picks_a_wait_its_host_cannot_execute() {
        // The strategy names an instruction, so picking one the CPUID
        // probe did not confirm is a `#UD` on the steal path rather
        // than a wrong answer. Read from the strategy toward the
        // probe, which is the direction that catches a `pick` whose
        // arms have been reordered out from under the detections.
        //
        // The spin carries no precondition, and asserting one here
        // would be asserting something else entirely: this host has
        // MONITORX and is picked onto the spin anyway, because the
        // spin measured faster. That is a choice and not a failure to
        // detect, so the executability test must not encode it.
        // `monitorx_is_available_to_ask_for_and_is_never_chosen_on_its_own`
        // is where the choice is pinned.
        let path = temp_path("strategy_executable");
        let u = UrdDeque::create(&path, 2).expect("create");
        match u.wait_strategy() {
            WaitStrategy::Waitpkg => assert!(has_waitpkg()),
            WaitStrategy::Monitorx => assert!(has_monitorx()),
            WaitStrategy::PauseSpin => {}
        }
        remove_temp(&path);
    }

    #[test]
    fn publish_then_drain_round_trips() {
        let path = temp_path("publish_drain");
        let u = UrdDeque::create(&path, 2).expect("create");
        let items = [item(1), item(2), item(3)];
        let n = u.publish_to(0, &items).expect("publish");
        assert_eq!(n, 3);
        match u.drain_mailbox(0) {
            Drain::Success(r) => {
                assert_eq!(r.n_items, 3);
                assert_eq!(r.items[0].closure_id, 1);
                assert_eq!(r.items[1].closure_id, 2);
                assert_eq!(r.items[2].closure_id, 3);
            }
            Drain::Empty => panic!("expected ready mailbox"),
        }
        // After drain, mailbox is EMPTY.
        assert!(matches!(u.drain_mailbox(0), Drain::Empty));
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn publish_round_robin_cycles_targets() {
        let path = temp_path("rr");
        let u = UrdDeque::create(&path, 4).expect("create");
        let items = [item(1)];
        let (t0, _) = u.publish_round_robin(&items).expect("rr 0");
        // Consume so the next publish doesn't spin on a busy mb;
        // bare statements discard the Drain return value (it's a
        // Copy enum, so `drop` would be a no-op clippy warning).
        u.drain_mailbox(t0);
        let (t1, _) = u.publish_round_robin(&items).expect("rr 1");
        u.drain_mailbox(t1);
        let (t2, _) = u.publish_round_robin(&items).expect("rr 2");
        u.drain_mailbox(t2);
        let (t3, _) = u.publish_round_robin(&items).expect("rr 3");
        // The four picks must cover all mailboxes (mod n_mailboxes).
        let mut targets = [t0, t1, t2, t3];
        targets.sort();
        assert_eq!(targets, [0, 1, 2, 3]);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn too_many_items_rejected() {
        let path = temp_path("too_many");
        let u = UrdDeque::create(&path, 2).expect("create");
        let items: Vec<LineItem> = (0..(MAILBOX_ITEMS + 1) as u32).map(item).collect();
        let err = u.publish_to(0, &items).expect_err("too many");
        assert_eq!(err, PublishError::TooManyItems);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn bad_target_rejected() {
        let path = temp_path("bad_target");
        let u = UrdDeque::create(&path, 2).expect("create");
        let err = u.publish_to(99, &[item(1)]).expect_err("bad target");
        assert_eq!(err, PublishError::BadTarget(99));
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn empty_publish_is_noop() {
        let path = temp_path("empty");
        let u = UrdDeque::create(&path, 2).expect("create");
        assert_eq!(u.publish_to(0, &[]).expect("publish"), 0);
        assert!(matches!(u.drain_mailbox(0), Drain::Empty));
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn close_owner_zeros_pid_and_advances_epoch() {
        let path = temp_path("close");
        let u = UrdDeque::create(&path, 2).expect("create");
        let before = u.header().epoch.load(O::Acquire);
        u.close_owner();
        assert_eq!(u.owner_pid(), 0);
        assert_eq!(u.header().epoch.load(O::Acquire), before + 1);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn four_thieves_no_double_take() {
        // Stress: owner publishes round-robin to 4 mailboxes; 4
        // thieves each drain their own mailbox. Sum invariant.
        //
        // Shrunk to n=400 (from a larger N) so the test stays under
        // the cargo-test 60s starvation timeout when many other
        // stress tests are running in parallel. The thieves use
        // `yield_now` instead of `spin_loop` so they release CPU to
        // the publisher and to other concurrent test threads under
        // CPU saturation. A 30-second wall-clock deadline on the
        // thief loops surfaces a clear panic if the URD primitive
        // ever stalls under contention instead of silently hanging.
        use std::sync::atomic::AtomicUsize;
        use std::time::{Duration, Instant};
        let path = temp_path("stress");
        let u = Arc::new(UrdDeque::create(&path, 4).expect("create"));
        let n = 400usize;
        let consumed = Arc::new(AtomicUsize::new(0));
        let sum = Arc::new(AtomicUsize::new(0));

        let mut thieves = Vec::new();
        for tid in 0..4 {
            let u = Arc::clone(&u);
            let consumed = Arc::clone(&consumed);
            let sum = Arc::clone(&sum);
            thieves.push(thread::spawn(move || {
                let deadline = Instant::now() + Duration::from_secs(30);
                while consumed.load(O::Relaxed) < n {
                    assert!(
                        Instant::now() < deadline,
                        "URD stress: thief {tid} starved past 30s deadline",
                    );
                    match u.drain_mailbox(tid) {
                        Drain::Success(r) => {
                            for i in 0..r.n_items {
                                consumed.fetch_add(1, O::Relaxed);
                                sum.fetch_add(
                                    r.items[i].closure_id as usize,
                                    O::Relaxed,
                                );
                            }
                        }
                        Drain::Empty => std::thread::yield_now(),
                    }
                }
            }));
        }

        // Publisher: round-robin batches of MAILBOX_ITEMS items.
        let mut pushed = 0usize;
        while pushed < n {
            let want = MAILBOX_ITEMS.min(n - pushed);
            let mut batch = [item(0); MAILBOX_ITEMS];
            for slot in batch.iter_mut().take(want) {
                *slot = item(pushed as u32);
                pushed += 1;
            }
            u.publish_round_robin(&batch[..want]).expect("rr publish");
        }

        for t in thieves {
            t.join().expect("thief");
        }
        let expected: usize = (0..n).sum();
        assert_eq!(
            sum.load(O::Relaxed),
            expected,
            "URD stress sum invariant (every item consumed exactly once)"
        );
        std::fs::remove_file(&path).ok();
    }
}
