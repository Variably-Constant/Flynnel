//! What the shared-memory staging buffers pay for their mutex.
//!
//! `khpd`, `urd_backend` and `lcrq_lifo` each hold a `Mutex<Vec<_>>`
//! that an owner pushes into and a publish drains. Every other lock
//! removed in this campaign was on a read path many threads reach at
//! once, and the call-site registry measured at 1.27 us on zen3 and
//! 2.51 us on pc2 under that shape. These three are different: khpd
//! documents its buffer as owner-side only ("Only the owner process
//! may stage") and its `unsafe impl Sync` names the mutex as part of
//! the soundness argument. So the question is not whether a lock-free
//! staging buffer can be written. It is whether this lock costs
//! anything, because the standing criterion forbids a change that
//! makes something slower and a rewrite here would have to restate a
//! safety argument to buy it.
//!
//! # Shape
//!
//! Four cells, all in one process and interleaved so a clock or a
//! frequency change moves all of them together:
//!
//!   control  building the item and dropping it, with no buffer at
//!            all. The floor.
//!   mutex    the item pushed into a `Mutex<Vec<_>>`, drained every
//!            `BATCH` pushes, which is what these three do.
//!   stack    the item pushed onto a compare-exchange stack and the
//!            whole stack swapped out every `BATCH` pushes, which is
//!            what a lock-free staging buffer would do.
//!   thread   the item pushed into a thread-local `Vec`, drained the
//!            same way. Not a candidate, because the buffer is shared
//!            by contract; it is here as the floor a staging buffer
//!            could reach if nothing were shared at all, so the other
//!            two can be read against something rather than only
//!            against each other.
//!   khpd     the shipped path, `KhpdDeque::stage` into a real deque
//!            with the publish it fires every `LINE_ITEMS` stages,
//!            against its own control. The four above are replicas
//!            written here and can only say which shape is fastest;
//!            this one says whether that shows through the path that
//!            changed.
//!
//! The item is a `u64` pair rather than a real `LineItem`, because
//! what is being timed is the buffer and not the item: a heavier item
//! would add the same constant to all four cells and shrink every
//! difference toward the floor. The khpd cell builds a real `LineItem`
//! and its control builds the same one.
//!
//! # Quiet and loaded
//!
//! Every cell runs alone and then with `threads - 1` siblings pushing
//! into the same buffer. The quiet row is the one the design contract
//! predicts, since the buffer is documented owner-side. The loaded row
//! is what happens if that contract is ever broken by a caller
//! reaching a backend through an `Arc<dyn DispatchBackend>`, which the
//! `&self` signature permits. For the khpd cell the timed thread stays
//! the deque's one publisher and every sibling steals from its ring,
//! which is the contention the deque is built for, and the cell
//! reports how many of its publishes found the ring full.
//!
//! Each cell is read as its median over `REPEATS`, and the control is
//! subtracted. The control is interleaved with the other three rather
//! than sampled at the ends, so it is not sitting on the extremes of
//! the warm-up curve while everything measured against it averages
//! over the whole of it. A figure smaller than the control's own span
//! across its repeats was not resolved by the run.

use std::cell::RefCell;
use std::hint::black_box;
use std::sync::atomic::{AtomicBool, AtomicPtr, Ordering};
use std::sync::{Arc, Barrier, Mutex};
use std::time::Instant;

use flynnel::backend::shared_mem::khpd::{KhpdDeque, LINE_ITEMS, LineItem, PushError, Steal};

/// Pushes before a drain, matching the `LINE_ITEMS` flush these
/// buffers do.
const BATCH: usize = 8;

/// Pushes per timed cell in the quiet row.
const CALLS: u64 = 20_000_000;

/// Pushes per timed cell in the loaded row.
///
/// Ten times smaller than the quiet count, because the contended mutex
/// cell runs at roughly 800 ns per push where the quiet one runs at 5.
/// At the quiet count the loaded row alone took minutes per repeat and
/// held a leased box for an hour over a difference that is three
/// orders of magnitude wide and resolved in seconds.
const LOADED_CALLS: u64 = 2_000_000;

/// Pushes through the real `KhpdDeque` in the shipped-path cell.
///
/// Small because this cell publishes. `LINE_ITEMS` is 3, so a publish
/// fires every third stage, and a publisher with nothing draining the
/// ring spins once the ring wraps. Keeping the count well under
/// `REAL_CAPACITY * LINE_ITEMS` means the ring never wraps and no
/// consumer is needed.
const REAL_CALLS: u64 = 100_000;

/// Publication lines in the deque the shipped-path cell stages into.
///
/// Derived rather than written down. Every repeat stages `REAL_CALLS`
/// items and publishes one line per `LINE_ITEMS` of them, and nothing
/// drains the ring, so every repeat of the run has to fit rather than
/// just one. A fixed 65536 covered one repeat and the seventh filled
/// the ring, which surfaced as `PushError::Full`.
const REAL_CAPACITY: usize = REPEATS * (REAL_CALLS as usize).div_ceil(LINE_ITEMS) + REPEATS;

/// Publication lines in the deque the loaded shipped-path cell stages
/// into. Consumers drain it, so it only has to absorb bursts; a
/// publisher that finds it full waits, and every such wait in the timed
/// cell is counted and printed.
const REAL_LOADED_CAPACITY: usize = 4096;

/// Timed cells per shape. The median is taken, so an odd count has a
/// middle.
const REPEATS: usize = 7;

/// A staged item, sized to what these buffers actually carry rather
/// than to a real LineItem.
#[derive(Clone, Copy)]
struct Item {
    id: u64,
    offset: u64,
}

struct Node {
    item: Item,
    next: *mut Node,
}

/// A compare-exchange staging stack: push never waits, and a drain
/// takes the whole stack in one swap.
struct Staging {
    head: AtomicPtr<Node>,
}

impl Staging {
    const fn new() -> Self {
        Self {
            head: AtomicPtr::new(core::ptr::null_mut()),
        }
    }

    fn push(&self, item: Item) {
        let node = Box::into_raw(Box::new(Node {
            item,
            next: core::ptr::null_mut(),
        }));
        loop {
            let head = self.head.load(Ordering::Acquire);
            // SAFETY: not published until the compare-exchange below
            // succeeds, so nothing else reaches it.
            unsafe { (*node).next = head };
            if self
                .head
                .compare_exchange(head, node, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                return;
            }
        }
    }

    /// Take everything staged and answer how many there were.
    fn drain(&self) -> usize {
        let mut chain = self.head.swap(core::ptr::null_mut(), Ordering::AcqRel);
        let mut taken = 0usize;
        while !chain.is_null() {
            // SAFETY: the swap took the whole chain, so this call owns
            // every node on it and takes each exactly once.
            let node = *unsafe { Box::from_raw(chain) };
            chain = node.next;
            black_box(node.item.id);
            taken += 1;
        }
        taken
    }
}

impl Drop for Staging {
    fn drop(&mut self) {
        self.drain();
    }
}

fn drain_vec(buffer: &mut Vec<Item>) -> usize {
    let taken = buffer.len();
    for item in buffer.drain(..) {
        black_box(item.id);
    }
    taken
}

/// One timed run of a cell, in nanoseconds per push.
fn cell(calls: u64, mut body: impl FnMut(u64)) -> f64 {
    let start = Instant::now();
    for i in 0..calls {
        body(i);
    }
    start.elapsed().as_nanos() as f64 / calls as f64
}

fn median(mut xs: Vec<f64>) -> f64 {
    xs.sort_by(|a, b| a.partial_cmp(b).expect("a timing is never NaN"));
    xs[xs.len() / 2]
}

fn spread_pct(xs: &[f64]) -> f64 {
    let lo = xs.iter().copied().fold(f64::INFINITY, f64::min);
    let hi = xs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if lo <= 0.0 {
        f64::INFINITY
    } else {
        (hi - lo) / lo * 100.0
    }
}

fn item_at(i: u64) -> Item {
    Item {
        id: i,
        offset: i << 6,
    }
}

fn lock_vec(buffer: &Mutex<Vec<Item>>) -> std::sync::MutexGuard<'_, Vec<Item>> {
    // A panicking pusher would poison this, and the buffer it guards
    // is a staging vector a panic leaves consistent.
    match buffer.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

fn push_mutex(buffer: &Mutex<Vec<Item>>, i: u64) {
    let mut held = lock_vec(buffer);
    held.push(item_at(i));
    if held.len() >= BATCH {
        black_box(drain_vec(&mut held));
    }
}

fn push_stack(buffer: &Staging, i: u64) {
    buffer.push(item_at(i));
    if i as usize % BATCH == BATCH - 1 {
        black_box(buffer.drain());
    }
}

fn measure(calls: u64, shared_mutex: &Mutex<Vec<Item>>, shared_stack: &Staging) -> [Vec<f64>; 4] {
    thread_local! {
        static LOCAL: RefCell<Vec<Item>> = const { RefCell::new(Vec::new()) };
    }

    let mut control = Vec::with_capacity(REPEATS);
    let mut mutex_cell = Vec::with_capacity(REPEATS);
    let mut stack_cell = Vec::with_capacity(REPEATS);
    let mut thread_cell = Vec::with_capacity(REPEATS);

    for _ in 0..REPEATS {
        control.push(cell(calls, |i| {
            black_box(item_at(i).offset);
        }));
        mutex_cell.push(cell(calls, |i| push_mutex(shared_mutex, i)));
        stack_cell.push(cell(calls, |i| push_stack(shared_stack, i)));
        thread_cell.push(cell(calls, |i| {
            LOCAL.with(|buffer| {
                let mut held = buffer.borrow_mut();
                held.push(item_at(i));
                if held.len() >= BATCH {
                    black_box(drain_vec(&mut held));
                }
            });
        }));
    }
    [control, mutex_cell, stack_cell, thread_cell]
}

/// Publish the calling thread's staged items, waiting out a full ring
/// one yield at a time. Answers how many times the ring was full.
fn publish_or_wait(deque: &KhpdDeque) -> u64 {
    let mut full = 0u64;
    loop {
        match deque.publish() {
            Ok(_) => return full,
            Err(PushError::Full) => {
                full += 1;
                std::thread::yield_now();
            }
            Err(other) => panic!("publish failed: {other:?}"),
        }
    }
}

fn line_item(i: u64) -> LineItem {
    LineItem::new(i as u32, (i & 0xFFFF) as u32, &i.to_le_bytes())
        .expect("eight bytes fits the inline payload")
}

/// The shipped path: stage into a real `KhpdDeque` and let it
/// auto-publish, which is the sequence `dispatch_marshal` runs.
///
/// This is here because the four cells above are replicas of the three
/// shapes, written in this file. They answer which shape is fastest.
/// They cannot answer whether the shape shows through the real path's
/// other costs, and `LINE_ITEMS` is 3, so the real path publishes every
/// third stage and the publish may well dominate the push the shapes
/// differ on. A harness pointed away from the code that changed prints
/// the same clean rows whether or not the change did anything.
///
/// The third figure is how many publishes in the timed cells found the
/// ring full and waited. Quiet, the ring is sized so it is zero; loaded,
/// a nonzero count says part of the cell's time went to the consumers
/// rather than to the buffer.
fn measure_shipped(deque: &KhpdDeque) -> (Vec<f64>, Vec<f64>, u64) {
    let mut control = Vec::with_capacity(REPEATS);
    let mut khpd_cell = Vec::with_capacity(REPEATS);
    let mut full_waits = 0u64;
    for _ in 0..REPEATS {
        control.push(cell(REAL_CALLS, |i| {
            black_box(line_item(i));
        }));
        khpd_cell.push(cell(REAL_CALLS, |i| {
            let staged = deque.stage(line_item(i)).expect("staging never fails");
            if staged >= LINE_ITEMS {
                full_waits += publish_or_wait(deque);
            }
        }));
    }
    (control, khpd_cell, full_waits)
}

fn report_shipped(label: &str, control: Vec<f64>, khpd_cell: Vec<f64>, full_waits: u64) {
    let floor = median(control.clone());
    let control_spread = spread_pct(&control);
    println!(
        "{label} floor={floor:.4} ns control_spread={control_spread:.2}% resolution_floor={:.4} ns",
        floor * control_spread / 100.0
    );
    let khpd_median = median(khpd_cell.clone());
    println!(
        "{label} khpd_stage={:.4} ns over_floor={:.4} ns spread={:.2}% full_waits={full_waits} \
         (includes the publish that fires every {LINE_ITEMS} stages)",
        khpd_median,
        khpd_median - floor,
        spread_pct(&khpd_cell)
    );
}

fn report(label: &str, cells: [Vec<f64>; 4]) {
    let [control, mutex_cell, stack_cell, thread_cell] = cells;
    let floor = median(control.clone());
    let control_spread = spread_pct(&control);
    println!(
        "{label} floor={floor:.4} ns control_spread={control_spread:.2}% resolution_floor={:.4} ns",
        floor * control_spread / 100.0
    );
    for (name, xs) in [
        ("mutex", &mutex_cell),
        ("stack", &stack_cell),
        ("thread", &thread_cell),
    ] {
        let m = median(xs.clone());
        println!(
            "{label} {name}={:.4} ns over_floor={:.4} ns spread={:.2}%",
            m,
            m - floor,
            spread_pct(xs)
        );
    }
    let mutex_over = median(mutex_cell) - floor;
    let stack_over = median(stack_cell) - floor;
    if stack_over > 0.0 {
        println!("{label} mutex_over_stack={:.3}x", mutex_over / stack_over);
    } else {
        println!("{label} mutex_over_stack=unreadable, the stack cell did not clear the floor");
    }
}

fn thread_count() -> usize {
    match std::env::args().nth(1) {
        Some(arg) => match arg.parse::<usize>() {
            Ok(n) if n >= 1 => n,
            Ok(n) => panic!("thread count must be at least 1, got {n}"),
            Err(err) => panic!("thread count {arg:?} is not a number: {err}"),
        },
        None => std::thread::available_parallelism()
            .expect("the host reports its parallelism")
            .get(),
    }
}

fn main() {
    let threads = thread_count();
    let shared_mutex: Arc<Mutex<Vec<Item>>> = Arc::new(Mutex::new(Vec::with_capacity(BATCH)));
    let shared_stack: Arc<Staging> = Arc::new(Staging::new());

    println!(
        "staging_buffer_cost threads={threads} batch={BATCH} calls={CALLS} \
         loaded_calls={LOADED_CALLS} real_calls={REAL_CALLS} repeats={REPEATS}"
    );
    report("quiet", measure(CALLS, &shared_mutex, &shared_stack));

    // The shipped path, against the replicas above. Its own control,
    // because it runs a different number of calls and builds a real
    // LineItem per call rather than an Item.
    let deque_path = std::env::temp_dir().join(format!(
        "flynnel_staging_cost_{}.khpd",
        std::process::id()
    ));
    let deque = KhpdDeque::create(&deque_path, REAL_CAPACITY)
        .expect("a temp-dir deque of the sized capacity");
    let (real_control, khpd_cell, full_waits) = measure_shipped(&deque);
    report_shipped("shipped", real_control, khpd_cell, full_waits);
    drop(deque);
    if let Err(e) = std::fs::remove_file(&deque_path) {
        println!("shipped note: the deque file at {deque_path:?} outlived the run: {e}");
    }

    let stop = Arc::new(AtomicBool::new(false));
    let barrier = Arc::new(Barrier::new(threads));
    let mut helpers = Vec::with_capacity(threads.saturating_sub(1));
    for _ in 0..threads.saturating_sub(1) {
        let stop = Arc::clone(&stop);
        let barrier = Arc::clone(&barrier);
        let shared_mutex = Arc::clone(&shared_mutex);
        let shared_stack = Arc::clone(&shared_stack);
        helpers.push(std::thread::spawn(move || {
            barrier.wait();
            let mut i = 0u64;
            while !stop.load(Ordering::Relaxed) {
                push_mutex(&shared_mutex, i);
                push_stack(&shared_stack, i);
                i += 1;
            }
        }));
    }
    if threads > 1 {
        barrier.wait();
    }
    report("loaded", measure(LOADED_CALLS, &shared_mutex, &shared_stack));
    stop.store(true, Ordering::Relaxed);
    for h in helpers {
        h.join().expect("a helper thread panicked");
    }

    // The shipped path under load: the timed thread is the deque's one
    // publisher, as the owner is in production, and every sibling
    // steals lines from it, so the ring stays a ring and the contention
    // is the one the deque is built for. The times a publish finds the
    // ring full are counted and printed, because a cell dominated by
    // waiting for ring space measures the consumers rather than the
    // buffer.
    //
    // Siblings do not publish: the deque has one owner and its
    // contention is thieves against that owner, which is what these
    // consumers are.
    let loaded_path = std::env::temp_dir().join(format!(
        "flynnel_staging_cost_loaded_{}.khpd",
        std::process::id()
    ));
    let deque = Arc::new(
        KhpdDeque::create(&loaded_path, REAL_LOADED_CAPACITY)
            .expect("a temp-dir deque of the loaded capacity"),
    );
    let consumers = threads.saturating_sub(1);
    let stop = Arc::new(AtomicBool::new(false));
    let barrier = Arc::new(Barrier::new(consumers + 1));
    let mut helpers = Vec::with_capacity(consumers);
    for _ in 0..consumers {
        let stop = Arc::clone(&stop);
        let barrier = Arc::clone(&barrier);
        let deque = Arc::clone(&deque);
        helpers.push(std::thread::spawn(move || {
            barrier.wait();
            while !stop.load(Ordering::Relaxed) {
                match deque.steal_line() {
                    Steal::Success(line) => {
                        for k in 0..line.n_items {
                            black_box(line.items[k].closure_id);
                        }
                    }
                    Steal::Empty | Steal::Retry => std::thread::yield_now(),
                }
            }
        }));
    }
    barrier.wait();
    println!("shipped_loaded consumers={consumers} publishers=1");
    let (real_control, khpd_cell, full_waits) = measure_shipped(&deque);
    report_shipped("shipped_loaded", real_control, khpd_cell, full_waits);
    stop.store(true, Ordering::Relaxed);
    for h in helpers {
        h.join().expect("a helper thread panicked");
    }
    drop(deque);
    if let Err(e) = std::fs::remove_file(&loaded_path) {
        println!("shipped_loaded note: the deque file at {loaded_path:?} outlived the run: {e}");
    }
}
