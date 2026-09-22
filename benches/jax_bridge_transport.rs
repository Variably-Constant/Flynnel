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
//! Needs `python3` on PATH for the echo child. Without one it reports
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

/// The ends of the pipe, together, because the two arms both need
/// exactly one thing to own or to lock.
struct Pipe {
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

impl Pipe {
    /// One line out and one line back. This is the whole of what the
    /// shipped `exchange` does.
    fn round_trip(&mut self, body: &str) -> std::io::Result<String> {
        writeln!(self.stdin, "{body}")?;
        self.stdin.flush()?;
        let mut line = String::new();
        self.stdout.read_line(&mut line)?;
        Ok(line)
    }
}

/// The echo child and its pipe, killed and reaped when dropped.
struct Echo {
    child: Child,
}

impl Echo {
    fn spawn() -> Option<(Self, Pipe)> {
        let mut child = Command::new("python3")
            .arg("-c")
            .arg(ECHO)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .ok()?;
        let stdin = child.stdin.take()?;
        let stdout = child.stdout.take()?;
        Some((
            Self { child },
            Pipe {
                stdin,
                stdout: BufReader::new(stdout),
            },
        ))
    }
}

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
        let rx = hub.register_consumer();
        let worker = std::thread::spawn(move || {
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
                eprintln!("jax_bridge_transport: owned round trip failed: {broken}");
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
        if let Some(worker) = self.worker.take() {
            if worker.join().is_err() {
                eprintln!("jax_bridge_transport: the owning thread panicked");
                FAILED.fetch_add(1, Ordering::Relaxed);
            }
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
                    eprintln!("jax_bridge_transport: locked round trip failed: {broken}");
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

fn bench_shapes(c: &mut Criterion, loaded: bool) {
    let cores = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(4);
    let suffix = if loaded { "loaded" } else { "idle" };
    let _load = if loaded { Some(Load::spawn()) } else { None };

    let mut group = c.benchmark_group(format!("jax_transport/{suffix}"));
    group.measurement_time(Duration::from_secs(10));

    for callers in [1usize, 2, cores.max(2)] {
        let Some((owned_child, owned_pipe)) = Echo::spawn() else {
            eprintln!("jax_bridge_transport: no python3 on PATH, nothing measured");
            return;
        };
        let owned = Owned::new(owned_pipe);
        group.bench_function(format!("owner_thread/c{callers}"), |b| {
            b.iter(|| fan_out(callers, 8, |body| owned.call(body)));
        });
        drop(owned);
        drop(owned_child);

        let Some((locked_child, locked_pipe)) = Echo::spawn() else {
            eprintln!("jax_bridge_transport: no python3 on PATH, nothing measured");
            return;
        };
        let locked = Locked::new(locked_pipe);
        group.bench_function(format!("held_lock/c{callers}"), |b| {
            b.iter(|| fan_out(callers, 8, |body| locked.call(body)));
        });
        drop(locked);
        drop(locked_child);
    }

    group.finish();
}

fn bench_all(c: &mut Criterion) {
    eprintln!(
        "jax_bridge_transport: cores={}",
        std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(0)
    );
    bench_shapes(c, false);
    bench_shapes(c, true);
    let failed = FAILED.load(Ordering::Relaxed);
    if failed > 0 {
        eprintln!(
            "jax_bridge_transport: {failed} round trips did not come back, so these timings are of a run that was not doing all of its work"
        );
    } else {
        eprintln!("jax_bridge_transport: every round trip came back");
    }
}

criterion_group!(benches, bench_all);
criterion_main!(benches);
