//! A/B microbench: a lock held across a pipe round trip, against a
//! thread that owns the pipe.
//!
//! The jax bridge speaks one request and one response over one pipe
//! with nothing in the protocol to match a reply to its request, so
//! the exchanges have to happen one at a time. It used to get that by
//! holding a `Mutex<BridgeState>` across the whole round trip, so a
//! caller excluded every other caller for the length of it. It now
//! gets it from a thread that owns the pipe: a caller hands over a
//! serialized request and a reply slot and waits on its own slot, so
//! a second caller queues at the hub instead of being shut out.
//!
//! Both transports are built here and timed in one process against
//! one child, which is what makes the comparison a comparison. Only
//! the shipped one exists in the crate; the lock arm lives in this
//! file so that measuring it does not mean putting a lock back into
//! the code.
//!
//! ## Bench-audit
//!
//! - **Same payload across A/B**: the same child, the same line in,
//!   the same line back, the same caller count, the same round-trip
//!   count. The two arms differ in how a caller gets its turn and in
//!   nothing else.
//! - **The child is an echo, not jax.** What changed is the
//!   transport, and a real bridge call spends milliseconds inside
//!   python where the transport spends microseconds, so timing the
//!   real one would report python's variance and call it a result.
//!   The echo keeps the pipe, the line discipline and the
//!   one-at-a-time constraint, and removes only the part that is the
//!   same in both arms.
//! - **The replica is the two operations the shipped path performs**,
//!   a `writeln!` with a flush and a `read_line`, over the same hub
//!   and the same per-caller reply channel. It is a model and can
//!   drift from the code it models; what keeps it honest is that the
//!   owner arm's queueing is the crate's own `NotifyHub`.
//! - **One caller and many.** With one caller there is no contention
//!   and the lock is a few cycles, so that row is where never-slower
//!   has to hold. With many, the lock is what one caller does to the
//!   others, which is the thing being removed.
//! - **Idle and loaded.** A caller shut out of the lock loses its
//!   core only when something else wants it.
//! - **The callers are started inside the span**, which adds the same
//!   thread starts to both arms. At the widest row that is a few
//!   hundred microseconds against a span of several milliseconds, and
//!   being equal in both arms it can dilute the difference between
//!   them but cannot reverse it. A crew outliving the iterations
//!   would carry less, at the cost of a second copy of one.
//!
//! Needs an interpreter on PATH for the echo child, tried as `python3`
//! then `python`, and the one it takes is the first that echoes a probe
//! line back rather than the first that starts. Without one it reports
//! that it measured nothing rather than reporting an empty pass.

use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use criterion::{Criterion, criterion_group, criterion_main};

use flynnel::sched::notify_ring::{NotifyHub, NotifySendResult, NotifySender};

/// Reads lines and writes each one straight back. The flush matters:
/// without it python buffers and the reader waits forever.
const ECHO: &str =
    "import sys\nfor line in sys.stdin:\n    sys.stdout.write(line)\n    sys.stdout.flush()\n";

/// Round trips whose answer did not come back, counted rather than
/// asserted so a failure is reported beside the timing it was taken
/// with instead of aborting the run that found it.
static FAILED: AtomicU64 = AtomicU64::new(0);

/// The first failure of each kind, said once.
///
/// A child that has gone makes every later call fail the same way, and
/// a line each would bury the run's own output under thousands of
/// copies of one fact, in a log a chain then greps. The count in
/// `FAILED` is what says how many there were.
///
/// One gate per transport, not one for the file: the two arms fail for
/// their own reasons, and a single gate would let whichever failed
/// first silence the other, which is the one thing a comparison
/// between them must not do.
static SAID_OWNED: std::sync::Once = std::sync::Once::new();
static SAID_LOCKED: std::sync::Once = std::sync::Once::new();

fn say_once(gate: &'static std::sync::Once, what: impl FnOnce() -> String) {
    gate.call_once(|| eprintln!("{}", what()));
}

/// The ends of the pipe, together, because the two arms both need
/// exactly one thing to own or to lock.
struct Pipe {
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

impl Pipe {
    /// One line out and one line back. This is the whole of what the
    /// shipped `exchange` does.
    ///
    /// A read that returns nothing is the child having gone, which
    /// reaches a caller as a successful read of an empty line and
    /// would let a whole run time an empty loop. It is an error here.
    fn round_trip(&mut self, body: &str) -> std::io::Result<String> {
        writeln!(self.stdin, "{body}")?;
        self.stdin.flush()?;
        let mut line = String::new();
        if self.stdout.read_line(&mut line)? == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "the echo child closed its output",
            ));
        }
        Ok(line)
    }
}

/// The echo child and its pipe, killed and reaped when dropped.
struct Echo {
    child: Child,
}

impl Echo {
    /// Start the echo on the first interpreter that echoes.
    ///
    /// Starting one is not the test. Windows ships an execution alias
    /// named `python3` that starts, says where Python can be
    /// installed and exits, so a spawn that succeeds proves nothing
    /// and the pipe behind it returns end of file on every read. A
    /// bench that took it would time an empty read loop and report
    /// the two transports as identically fast, which is the one wrong
    /// answer that looks like a result.
    fn start() -> Option<(Self, Pipe)> {
        for interpreter in ["python3", "python"] {
            let mut child = match Command::new(interpreter)
                .arg("-c")
                .arg(ECHO)
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .stderr(Stdio::null())
                .spawn()
            {
                Ok(child) => child,
                Err(absent) => {
                    eprintln!("jax_bridge_transport: {interpreter} did not start: {absent}");
                    continue;
                }
            };
            let (Some(stdin), Some(stdout)) = (child.stdin.take(), child.stdout.take()) else {
                eprintln!("jax_bridge_transport: {interpreter} started without both pipes");
                drop(Self { child });
                continue;
            };
            let mut pipe = Pipe {
                stdin,
                stdout: BufReader::new(stdout),
            };
            match pipe.round_trip(PROBE) {
                Ok(back) if back.trim() == PROBE => return Some((Self { child }, pipe)),
                Ok(other) => {
                    eprintln!(
                        "jax_bridge_transport: {interpreter} answered {:?} rather than echoing",
                        other.trim()
                    );
                }
                Err(broken) => {
                    eprintln!("jax_bridge_transport: {interpreter} did not echo: {broken}");
                }
            }
            drop(pipe);
            drop(Self { child });
        }
        None
    }
}

/// The line `start` sends to decide whether a child is an echo.
const PROBE: &str = "{\"op\":\"probe\"}";

impl Drop for Echo {
    fn drop(&mut self) {
        // Both arms drop their end of the pipe before this runs, and a
        // closed stdin is what ends the echo loop, so the child has
        // usually gone by itself. Asking first means a kill is only
        // issued to a child that is genuinely still there, and a kill
        // that then fails is worth saying rather than expected.
        match self.child.try_wait() {
            Ok(Some(_ended)) => return,
            Ok(None) => match self.child.kill() {
                Ok(()) => {}
                Err(refused) => {
                    eprintln!(
                        "jax_bridge_transport: the echo child would not be killed: {refused}"
                    );
                }
            },
            Err(unpollable) => {
                eprintln!("jax_bridge_transport: the echo child could not be polled: {unpollable}");
            }
        }
        match self.child.wait() {
            Ok(_ended) => {}
            Err(unwaitable) => {
                eprintln!(
                    "jax_bridge_transport: the echo child could not be waited for: {unwaitable}"
                );
            }
        }
    }
}

/// One request and the slot its answer goes into.
struct Request {
    body: String,
    reply: std::sync::mpsc::Sender<std::io::Result<String>>,
}

/// The shipped shape: a thread owns the pipe, callers queue at the hub
/// and wait on a slot of their own.
struct Owned {
    tx: NotifySender<Request>,
    hub: NotifyHub<Request>,
    worker: Option<std::thread::JoinHandle<()>>,
}

impl Owned {
    fn new(mut pipe: Pipe) -> Self {
        let hub = NotifyHub::<Request>::new(256, 1);
        let tx = hub.sender();
        let hub_for_owner = hub.clone();
        let worker = std::thread::spawn(move || {
            // Registered on the thread that will receive, and not on
            // the one that spawns it. A NotifyReceiver captures the
            // parker of whichever thread
            // registered it, and a send wakes that thread, so
            // registering on the spawning thread and moving the
            // receiver across wakes the wrong one: the worker parks,
            // nobody wakes it, and the first call never returns. That
            // is what the shipped bridge does too, by registering
            // inside its owner loop.
            let rx = hub_for_owner.register_consumer();
            while let Some(request) = rx.recv() {
                let outcome = pipe.round_trip(&request.body);
                if request.reply.send(outcome).is_err() {
                    // The caller stopped waiting. The exchange itself
                    // completed, so the pipe is still in step.
                    FAILED.fetch_add(1, Ordering::Relaxed);
                }
            }
        });
        Self {
            tx,
            hub,
            worker: Some(worker),
        }
    }

    fn call(&self, body: &str) {
        let (reply, answer) = std::sync::mpsc::channel();
        match self.tx.send(Request {
            body: body.to_string(),
            reply,
        }) {
            NotifySendResult::Ok => {}
            NotifySendResult::Closed(refused) => {
                drop(refused);
                FAILED.fetch_add(1, Ordering::Relaxed);
                return;
            }
        }
        match answer.recv() {
            Ok(Ok(_line)) => {}
            Ok(Err(broken)) => {
                say_once(&SAID_OWNED, || {
                    format!("jax_bridge_transport: owned round trip failed: {broken}")
                });
                FAILED.fetch_add(1, Ordering::Relaxed);
            }
            Err(_) => {
                FAILED.fetch_add(1, Ordering::Relaxed);
            }
        }
    }
}

impl Drop for Owned {
    fn drop(&mut self) {
        self.hub.shutdown();
        if let Some(worker) = self.worker.take()
            && worker.join().is_err()
        {
            eprintln!("jax_bridge_transport: the owning thread panicked");
            FAILED.fetch_add(1, Ordering::Relaxed);
        }
    }
}

/// The shape that was replaced: a caller holds the lock for the length
/// of the round trip, so it excludes every other caller for that long.
struct Locked {
    pipe: Mutex<Pipe>,
}

impl Locked {
    fn new(pipe: Pipe) -> Self {
        Self {
            pipe: Mutex::new(pipe),
        }
    }

    fn call(&self, body: &str) {
        match self.pipe.lock() {
            Ok(mut held) => match held.round_trip(body) {
                Ok(_line) => {}
                Err(broken) => {
                    say_once(&SAID_LOCKED, || {
                        format!("jax_bridge_transport: locked round trip failed: {broken}")
                    });
                    FAILED.fetch_add(1, Ordering::Relaxed);
                }
            },
            Err(_poisoned) => {
                FAILED.fetch_add(1, Ordering::Relaxed);
            }
        }
    }
}

/// Busy threads occupying half the host, joined when this is dropped.
struct Load {
    stop: Arc<AtomicU32>,
    threads: Vec<std::thread::JoinHandle<()>>,
}

impl Load {
    fn spawn() -> Self {
        let n = std::thread::available_parallelism()
            .map(|p| p.get() / 2)
            .unwrap_or(1)
            .max(1);
        let stop = Arc::new(AtomicU32::new(0));
        let threads = (0..n)
            .map(|_| {
                let stop = Arc::clone(&stop);
                std::thread::spawn(move || {
                    let mut x = 0u64;
                    while stop.load(Ordering::Relaxed) == 0 {
                        x = std::hint::black_box(
                            x.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1),
                        );
                    }
                })
            })
            .collect();
        Self { stop, threads }
    }
}

impl Drop for Load {
    fn drop(&mut self) {
        self.stop.store(1, Ordering::Relaxed);
        for t in self.threads.drain(..) {
            if t.join().is_err() {
                eprintln!(
                    "jax_bridge_transport: a load thread panicked, so this run's loaded rows carried less load than they report"
                );
            }
        }
    }
}

/// `callers` threads each doing `each` round trips through `call`.
fn fan_out<F>(callers: usize, each: u32, call: F)
where
    F: Fn(&str) + Sync,
{
    std::thread::scope(|scope| {
        for c in 0..callers {
            let call = &call;
            scope.spawn(move || {
                for i in 0..each {
                    call(&format!("{{\"op\":\"echo\",\"c\":{c},\"i\":{i}}}"));
                }
            });
        }
    });
}

/// Returns whether any row was timed, so the closing line can tell a
/// clean run from one that measured nothing. Every round trip coming
/// back is true of a run with no round trips in it, and on its own it
/// reads like a pass.
fn bench_shapes(c: &mut Criterion, loaded: bool) -> bool {
    let mut timed = false;
    let cores = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(4);
    let suffix = if loaded { "loaded" } else { "idle" };
    let _load = if loaded { Some(Load::spawn()) } else { None };

    let mut group = c.benchmark_group(format!("jax_transport/{suffix}"));
    group.measurement_time(Duration::from_secs(10));

    // Deduplicated, because criterion refuses two benchmarks with one
    // id and a host of two cores would otherwise ask for c2 twice.
    let mut widths = vec![1usize, 2, cores.max(2)];
    widths.dedup();
    for callers in widths {
        let Some((owned_child, owned_pipe)) = Echo::start() else {
            eprintln!(
                "jax_bridge_transport: no interpreter on PATH echoed a line, nothing measured"
            );
            group.finish();
            return timed;
        };
        let owned = Owned::new(owned_pipe);
        group.bench_function(format!("owner_thread/c{callers}"), |b| {
            b.iter(|| fan_out(callers, 8, |body| owned.call(body)));
        });
        timed = true;
        drop(owned);
        drop(owned_child);

        let Some((locked_child, locked_pipe)) = Echo::start() else {
            eprintln!(
                "jax_bridge_transport: no interpreter on PATH echoed a line, nothing measured"
            );
            group.finish();
            return timed;
        };
        let locked = Locked::new(locked_pipe);
        group.bench_function(format!("held_lock/c{callers}"), |b| {
            b.iter(|| fan_out(callers, 8, |body| locked.call(body)));
        });
        drop(locked);
        drop(locked_child);
    }

    group.finish();
    timed
}

fn bench_all(c: &mut Criterion) {
    eprintln!(
        "jax_bridge_transport: cores={}",
        std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(0)
    );
    let idle = bench_shapes(c, false);
    let loaded = bench_shapes(c, true);
    let failed = FAILED.load(Ordering::Relaxed);
    if !idle && !loaded {
        eprintln!(
            "jax_bridge_transport: nothing measured in either shape, so there is no result here to read"
        );
    } else if failed > 0 {
        eprintln!(
            "jax_bridge_transport: {failed} round trips did not come back, so these timings are of a run that was not doing all of its work"
        );
    } else {
        eprintln!("jax_bridge_transport: every round trip came back");
    }
}

criterion_group!(benches, bench_all);
criterion_main!(benches);
