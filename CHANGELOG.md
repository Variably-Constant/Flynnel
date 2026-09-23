# Changelog

All notable changes to `flynnel`, by version. Numbers are
measurements from `benches/` and `tests/` on the two bench hosts, an
RTX 3070 with a Ryzen 7 2700 (16 threads) and an RTX 5070 with a
Ryzen 9 7900X (24 threads); the wiki carries the full tables.

## Unreleased

### Changed

- `external_dispatch`'s documentation says what the caller does: it
  hands the join to a worker, spins for the plan's budget, then parks
  until the latch sets, and runs none of the join's work itself.

- The parker records what it did, under `FLYNNEL_TRACE=1`. `ParkEnter`
  (kind 14) on every park, payload the strategy it chose - 0 kernel
  park, 1 WAITPKG, 2 MONITORX - plus 16 when that park is one half of a
  controller probe and is therefore being timed. `WorkerWake` (8) on a
  sampled park, payload that wake's cost in nanoseconds. `WaitSwitch`
  (15) when the controller moves, payload the strategy now in use. The
  strategy is recorded per park rather than read from the controller
  afterwards, because the controller reports where it ended up and a run
  that switched partway through looks from its report exactly like one
  that started there.

  None of the three has yet been observed to fire. The one trace taken
  produced 110,458 rows over 201 dispatches with no park event among
  them: `examples/trace_dispatch` dispatched back to back and no worker
  went idle long enough to park. Its fourth argument is now a gap in
  milliseconds, applied between dispatches and once more before the
  timed call, which is the condition under which these events have
  anything to record. A run at gap zero still records none.

- The park path reads the trace switch before it builds the payload.
  Rust evaluates arguments before the call, so passing the payload to
  `trace::emit` computed it on every park and discarded it inside when
  tracing was off.

  Measured by `examples/trace_predicate_cost` on a Zen 3 Linux guest,
  five cells interleaved in one process, 500,000,000 calls a cell, seven
  repeats, the control interleaved with the rest and its span across its
  own repeats printed as the run's resolution floor. Nanoseconds over
  that control, on the two runs of six that passed their own readability
  test:

  quiet, control 0.5958 ns at a 2.11 per cent spread, floor 0.0126 ns -
  payload alone 0.5603, whole emit 1.2316.

  loaded at 8 spinners on 16 cores, control 1.8348 ns at a 4.00 per cent
  spread, floor 0.0734 ns - payload alone 0.4705, emit with the payload
  passed as an argument 1.2533, emit with the switch read first 0.1299.
  The difference between those last two is 1.1234 ns a park, against a
  floor of 0.0734.

  The other four runs declared themselves unreadable on control spreads
  of 45.46, 76.70, 79.96 and 62.51 per cent and are not quoted. An
  earlier batch was unreadable for a different reason, a control sampled
  only at the run's two endpoints while every other cell was
  interleaved, which put the one cell everything is subtracted from on
  the extremes of the warm-up curve.

- `FLYNNEL_LEVER_ALLOWED_WIDTH` defaults on. A plan's worker count is
  capped by the CPUs the process may use at that moment, re-read every
  250 ms, so a process whose affinity mask or cgroup quota narrows after
  the pool is spawned stops chunking for threads that cannot reach a
  core; `tests/affinity_follows_process_mask.rs` holds that on Linux,
  FreeBSD and Windows. Its price is the re-read on a host whose mask
  never changes, measured twice paired by trial on a 24-thread
  bare-metal box at 4096 reps of uniform work over 40 trials with no
  decision moving between the arms: 1.0006 at a 0.13 per cent bound
  over 40 clean pairs, retained 1.0005 at 0.17, control 1.0000 at 0.10,
  on the code that ships; and 1.0000 at 0.15 over 23 clean pairs,
  retained 1.0014 at 0.21, control 1.0000 at 0.30, on a tree whose
  Windows probe was still `available_parallelism`. Off restores the
  shipped sizing, the arena's spawned width whatever the host allows.

- `FLYNNEL_LEVER_ONCORE_SPREAD` is documented as a correctness lever and
  carries its measured price. It swaps the classifier's input from wall
  time to the thread clock, which excludes descheduled time, so what it
  changes is proportional to how much descheduling a host does: the
  learned class differs between its two arms in 29 trials of 40 on a
  Linux guest, 6 of 40 on a FreeBSD guest and 1 of 40 on bare metal. A
  host that does not deschedule has nothing for it to correct.

  Its cost, every figure an upper bound rather than a resolved
  difference: 0.9978 at a 0.22 per cent bound on a 24-thread bare-metal
  box over 32 clean pairs of 40, control 0.9988 at 0.38, and 1.3 to 1.9
  on a Linux guest. That cell was read twice - an earlier rotation gave
  0.9964 at 1.90 over 7 pairs while an unrelated process held a core
  continuously, putting the box's idle floor at 1.81 cores against the
  harness's 1.4-core gate, so most trials were dropped for a condition
  none of them caused. The difference between the two bounds is the
  floor, not the lever. On that
  guest at an 8 second window over 40 trials it reads 0.9814 at a 2.52
  per cent bound in the trials where the class moves and 0.9872 at 4.34
  where it holds - about the same either way. An earlier reading at a 2
  second window put those at 1.0000 and 0.9759 and suggested the routing
  change paid for the bracket; the longer window cut the bound from 6.44
  to 2.52 and reversed the ordering, so that was noise.

  No speed-up is claimed, and the reason is structural. Four cells were
  screened on bare metal for one where the class could move at all -
  reps 4096 uniform, 4096 irregular, 8192 and 16384 - and none did.
  `classify_observed` returns `PortBound` below 500 ns and never reads
  cv^2; raising reps lifts the mean past that gate while averaging the
  variance out of cv^2, so the two conditions a class change needs pull
  apart. Where the lever acts, the host is too noisy to resolve what it
  did; where the host resolves it, the lever has nothing to do.

- The split multiplier is decided from the window passed to it rather
  than from the globals the sampler reads. `sample_and_compute` summed
  process-global arena counters and then asserted against them, so the
  test covering its sparse-window short-circuit raced every other test
  in the binary: a concurrent dispatch could push those counters back
  over the floor between the test's reset and its assert. It failed once
  in a gate and passed five times on re-run - alone, in the full
  parallel suite and single-threaded - which is the shape that gets a
  gate ignored. The decision now takes the window as arguments and the
  wrapper reads the globals, so the three tests replacing it are
  deterministic: nine consecutive suite runs on the guest that caught it
  read 749 passed and none failed. The early return still skips
  `reset_leaf_stats`, which is what lets a window too sparse to read
  accumulate its leaves into the next one.

- The adaptive spin controller is measured on what it does rather than
  on throughput. Where it shrinks the window, idle yields fall to
  between 0.22 and 0.51 of the arm without it; where it does not, they
  read 1.000 to 1.019. Four rotations, two hosts, two operating systems,
  with the unshrunk trials of each rotation as its own control.

  No throughput effect survives a change of host or load. Pinning the
  window directly with `FLYNNEL_SPIN_WINDOW_ROUNDS` - 8 against 500,
  the controller out of the comparison and every trial usable - gives
  1.0250 and 1.0271 on a Linux guest at half and full load, and 0.9980
  and 1.0655 on a FreeBSD guest, at bounds of 3.9 to 9.0 per cent. The
  ratio is flat across load, so the lever cannot show faster-under-load
  whatever its size: there is nothing for load to change.

  How often the controller shrinks at all is a host property, not a
  setting: 1, 17, 18 and 21 arms in 40 on one guest and 39 in 40 on the
  other, at identical settings.

- `examples/zen3_lever_ab.sh` runs on FreeBSD as well as Linux. It takes
  busy cores from `kern.cp_time` where `/proc/stat` is absent, the core
  count from `hw.ncpu` where `nproc` is, and it strips the padding
  FreeBSD's `wc` puts around a count - which had made every arm line
  unparseable, so a 40-trial rotation reported forty trials and no pairs
  rather than failing. It also takes the SMT prior as an argument,
  without which `effective_use_smt` returns on its first line and the
  window lever's two arms are the same arm.

- `examples/zen3_lever_ab.sh` runs a lever A/B paired by trial on Linux,
  and `examples/paired_arms_report.py` takes an optional field to split
  its pairs on. The three-arm rotation compares arms that ran at
  different times, so drift between them lands in the ratio: its null
  read 11 per cent on a quiet 16-core guest where the paired shape reads
  0.61 per cent on ten pairs on a busier host. Every speed lever here is
  smaller than the first figure and larger than the second.

  The split exists because a mechanism can fire in one run and not the
  next at identical settings. The adaptive spin controller leaves its
  window at the tuned default in about three runs in four, so one median
  over every trial averages the trials where it acted with the trials
  where it did not and reports neither. Split, the trials where it held
  are a control for the ones where it moved, from the same rotation on
  the same box.

- All three rotation readers assert engagement themselves rather than
  leaving it to whoever reads the log.
  `examples/paired_arms_report.py` lists which decisions differ between
  the two arms before printing any ratio, and says so plainly when none
  does; `examples/lever_report.py` and `examples/smt_report.py` compare
  each arm's resolved switch state against its label and refuse
  mismatched cells by name. A log written before its harness reported
  lever states is described as unrecorded rather than treated as a pass.

  An arm label naming a value neither line reports is described as
  unverified rather than counted as a mismatch, because refusing the
  table on it discards a sound run: an arm that pins a width rather than
  flipping a switch names something the levers line does not carry. That
  width is read from the engagement line under the same name and
  compared literally, so a pin that did not take is caught the way a
  lever arm is. `examples/paired_armstate_defect_sample.log` and
  `examples/paired_pinstate_defect_sample.log` are what the two refusals
  are checked against.

  `levers::describe()` carries `spin_adaptive` and `spin_window`, which
  it did not. It is documented as every switch and its state, and the
  two that decide how the pool parks were absent, so a row printed from
  it could not say whether adaptation ran; the harnesses fetched
  `spin_adaptive` separately, which is why the engagement line had it
  and the levers line did not. That puts a width on a line that
  otherwise carries booleans, so the label check now decides from the
  values it found rather than from which line they came from: a field
  holding only `true` or `false` reads the label as a switch, anything
  else compares literally. Without that, an arm labelled
  `spin_window=8` would have been compared against `false` and a sound
  arm reported as a mismatch.

  Two levers now ship on, so an arm that expressed "off" by leaving its
  variable unset ran the lever on: all three arms of a rotation became
  one arm, with a tight null, because two identical arms agree. The
  rotations at `examples/lever_rounds.sh` and `examples/width_rounds.sh`
  now name the value on both arms, and `examples/smt_recovery.rs` and
  `examples/class_migration_under_load.rs` print the resolved states the
  way `throughput_under_load` already did.

  A switch's own flag is not counted as a decision it made. Including
  `spin_adaptive`, which differs between the arms by construction, made
  the moved-decision test pass for every rotation of the spin lever
  whatever the controller went on to do: a 432-arm run has it deciding
  36 to 50 times per process and leaving the window at the tuned default
  on both arms, which that test called engaged. A lever can be read,
  then decide, and still change nothing, and only the third says a
  throughput row is a reading of it.

- `examples/pc2_lever_ab.ps1` takes the engagement markers from its
  caller as `-ReadPattern` and `-ActedPattern`, and reports them as two
  figures rather than one. It counted `oncore_items`, which is one
  lever's marker, so every other lever's rotation passed the check
  without the switch having reached anything: a 40-trial run of the SMT
  window lever reported 80 engaged rows of 80 on a counter its
  mechanism never touches. A caller naming no pattern gets the word
  `unjudged` instead of a zero, and a run whose read pattern never
  matches exits non-zero rather than reporting a comparison between two
  arms that were the same arm.

- A stored calibration is served once an independent draw has agreed
  with it, where before it was served if the spread across its own nine
  samples sat under `PROVISIONAL_SPREAD_PER_MILLE`. `CpuCalibration`
  carries a `confirmations` count, raised when another draw's dispatch,
  collapse and wake figures all land within that bound;
  `is_trustworthy` reads the count. A record with none is stored and not
  served, so a fresh stamp serves from its third start.

  On a 12-core host with at most two other processes running, 40 draws
  gave within-draw spreads of 273 to 7429, median 1136, and not one met
  the bound of 250. The same draws, paired, disagreed on their dispatch
  medians by 0 to 206, median 74. So nothing served on that host and
  every process paid a 13.9 to 23.1 ms draw, while the figure the draws
  agreed on went unread.

  All three figures must agree because all three are served. Over 42
  consecutive pairs there, dispatch agreed 40 times, collapse 38 and
  wake 37; a pair agreeing on dispatch alone published a collapse of
  29183 against a 10733 median of the same 43 draws.

  `LAYOUT_VERSION` 7 to 8, which invalidates stored tables by stamp.

  Against the never-slower criterion, both halves on that host: serving
  removes the draw at start, and routing shows no difference at 1.05 per
  cent resolution over 14 paired trials whose control resolves to 0.24.

- `PublishOutcome::Published` and `KeptIncumbent` carry the agreement
  count, and `KeptIncumbent`'s two figures are dispatch costs in
  nanoseconds. The line reporting them named them parts per mille of
  occupancy, which cannot exceed 1000.

### Added

- `FLYNNEL_LEVER_JOIN_PARK` and `FLYNNEL_LEVER_SLOT_PARK_NOW`, both off.
  A thread that waits by calling `yield_now` while its process holds
  more runnable threads than cores gives its core to a ready thread for
  the rest of that thread's time slice, and sees a latch already set
  milliseconds late. On a 24-thread Windows host, with 18 spinning
  threads in the process, an outside caller's dispatch of 1e6 items
  finished every leaf within about 1 to 3 ms and completed as much as
  30 ms later, at a loaded p99 of 33.9 ms against a quiet 1.44.
  `JOIN_PARK` parks a join waiter in the kernel once its spin budget is
  spent and it finds nothing to steal, and the thief that sets the right
  half's latch wakes it. `SLOT_PARK_NOW` takes the yield rounds out of
  the outside caller's slot wait, which has already spun for its budget.
  The right half's latch is a `JoinLatch`, which names the forking
  worker's own parker by address, so a set reads one pointer and clones
  no `Arc`. `total_join_parks()` counts the parks, so a run can show it
  reached them.

  Both stay off, on four arms of that reproducer (neither, each, both;
  three rounds, positions rotating). On the 24-thread host `JOIN_PARK`
  took the loaded median from 3.02 to 0.95 ms and the p99 from 31.3 to
  1.8 ms, with calls of 5 ms or more falling from 211 to none, at no
  quiet cost: 0.468 ms against 0.471. On a 16-vCPU Linux guest it raised
  the quiet median from about 1.3 to 3.1 ms in each round, with the
  serial control level across the arms, because a parked thread halts
  its vCPU and the wake through the hypervisor costs far more than the
  yield it replaced. `SLOT_PARK_NOW` moved nothing on either host.

- `FLYNNEL_LEVER_JOIN_PARK_OVERSUBSCRIBED`, off: the join park, taken
  only while a yield somewhere in the process has lately given its core
  away. A yield with nothing else ready returns in about a microsecond,
  and one that loses the core returns a time slice later, so the pool's
  idle rounds, which yield with no latch pending, time their yields
  while the switch is on, and one of 2 ms or more marks the process
  oversubscribed for the next 10 ms. A join waiter whose spin budget is
  spent parks while that mark is fresh and yields otherwise.
  `total_long_yields()` counts the readings and `yield_histogram()`
  counts every timed yield in log2 buckets of microseconds.

  The line is read from those buckets. On the 16-vCPU guest, processes
  with no spinners saw 13 and 37 of about 7.9 million yields reach 2 ms,
  and processes with 12 spinners saw 10,106 and 27,393 of about 7.3
  million. A first line of 100 us did not separate them: the quiet
  processes read about 1,970 each, because a guest's vCPU is descheduled
  mid-yield often enough to look like a lost core, and parked 15,400 to
  21,300 times at a quiet cost of about 6 per cent. Its quiet and loaded
  cost at 2 ms, on bare metal and on a guest, are being measured, and it
  stays off until they are.

- Two trace events under `FLYNNEL_TRACE=1`. `LatchSet` (kind 20) comes
  just before a thief sets the latch of the stolen job it ran.
  `JoinLastYield` (21) comes before the `JoinWaitEnd` of a join wait
  that yielded, with payload the last yield's length in microseconds.
  A last yield that spans the awaited half's `LatchSet` is a waiter
  that was away while its latch was set, which is how the case above
  was told apart from a thief that set its latch late.

- `total_self_rescues()`, `total_park_events()` and
  `total_rescue_events()`: how pool workers leave the idle path, whether
  parking, rescued inside the spin window, or rescued by the recheck
  after publishing sleep.

- `examples/oversubscribed_caller`: an outside caller's dispatch timed
  with and without spinning threads in the same process, beside the
  same work run serially, reporting the tail. With `FLYNNEL_TRACE=1` it
  dumps every thread's trace so a slow call can be split.

- **`MONITORX`/`MWAITX`, so AMD parts before Zen 5 stop falling back
  to the kernel.** `cpu_info::has_monitorx` reads CPUID `Fn8000_0001`
  ECX bit 29, and the worker parker gained a third arm.

  **Which wait a worker uses is measured, not decided.** A monitor
  wait beats a kernel park on some parts and loses on others, and on
  a Ryzen 7 2700 it does both: five to nine times faster on an idle
  host and twenty times slower on a loaded one. Nothing readable at
  construction separates those, because CPUID reports the instruction
  and not what it costs, and the cost moves with the load.

  So `WaitController` times the parks the pool performs anyway. A
  probe is two consecutive timed parks, one per arm; the unparker
  stamps its TSC only when the waiter armed it, out of the cache line
  it is already writing `wake_counter` into, so a wake between probes
  pays one relaxed load and no clock read, and a host with no second
  strategy never touches the controller at all.

  **The two halves of a probe are back to back**, so every baseline
  reading has a challenger reading beside it in time. That pairing is
  the control, and it is not decoration: a quiet draw and a loaded
  draw of one quantity differ on these hosts by 350 to 4000 times,
  far more than the two waits differ from each other, so readings
  gathered over different stretches would compare the machine's mood.

  **The pair is also what decides.** Each one credits a point to
  whichever arm was cheaper in it, saturating at 64, and the process
  moves at a net 24; a pair whose readings sit within a fifth of each
  other separates nothing and scores neither way. Comparing the two
  means instead answers to their largest draws, and a wake latency
  has a long tail. Measured on a 7900X, the controller read its
  challenger at 17939 ns over 100 samples while a pinned arm measured
  that same wait at 888 ns in the same group under the same load,
  because a handful of draws near 340 us carried the average; it
  declined a wait nearly seven times cheaper on that. Within a pair
  the slower reading is the slower wait under the conditions that
  applied to both, and how large the loss was never enters. The means
  are kept and reported, because a tail that costs 340 us is worth
  seeing, and they no longer move the process.

  **How often a probe runs is itself adaptive**, because a probe
  spends one park on the arm not in use, and on a host that settled
  against the monitor wait that park costs twenty times what the
  chosen one costs. At a fixed one park in sixty-four that is about a
  sixth of every park for the life of the process, on exactly the
  host that already decided against it. The interval doubles to 4096
  parks once the score saturates, meaning one arm has won every
  recent pair, and collapses to 64 as soon as the score falls back
  inside the switching threshold, which is the order coming apart
  and happens before it flips. Probing never stops, so no verdict
  outlives its evidence and the choice moves back when a host gets
  busy.

  A process starts on WAITPKG or the kernel park, exactly as before,
  and leaves that only on measurement. `Parker::with_strategy` pins an
  arm and is never sampled, so the benches that hold one still cannot
  move the default they inform.

  **The URD thief waits in two stages.** Its spin is the fastest wake
  there is, because it never stopped looking, and it is also the most
  expensive way to be idle, because it holds a logical CPU throughout.
  Those are not alternatives to choose between, which is what the
  earlier comparison treated them as: the thief now spins for
  `SPIN_BEFORE_WAIT` polls and only then reaches whatever monitor wait
  the host has, so a publish already in flight is caught at spin
  latency and never reaches the second stage at all.

  Measured with no floor, so the monitor wait was doing the spin's job
  as well as its own, it was slower in three cells of four (1.54
  against 2.00 us idle at a 50 us inter-arrival, 1.66 against 1.92 us
  under load; best median of three runs on a 7900X).

  What the second stage buys is narrower than a core, and the
  difference decides whether it is worth having. A `PAUSE`-spin takes
  both the physical core's execution resources and a logical CPU the
  operating system could have given elsewhere. `MWAITX` halts the
  logical processor, handing the core's issue width to its SMT
  sibling, but it does not deschedule the thread: the kernel still
  counts that logical CPU as occupied. A pool with more runnable work
  than CPUs wants the second of those, and the instruction does not
  supply it.

  Neither is visible in the rows above, which time the thief rather
  than its neighbours. The co-runner group in
  `benches/urd_thief_wait.rs` times the neighbours and oversubscribes
  deliberately, one per logical CPU, which is the arrangement where
  the difference has somewhere to show. It has not run on any host,
  so the choice rests on the instruction being present rather than on
  it being cheaper here. Nothing outside that bench calls
  `UrdDeque::wait_and_drain`.

  The gap this closes is the whole AMD line from 2015 to Zen 5.
  `has_waitpkg` was the only probe, and it is false on Zen 1 through
  4, so every such host took the fallback although AMD's user-mode
  monitor-wait has been present throughout. For the parker that
  fallback is a syscall, not a spin.

  **Each arm re-checks and waits again rather than trusting one
  instruction, because `MWAITX`'s unit is not a constant.** It counts
  a relative number of `EBX` units where `UMWAIT` takes an absolute
  TSC deadline, and one unit measures 0.4999 RDTSC cycles on a Ryzen 9
  7900X against 0.9987 on a Ryzen 7 2700 - a factor of two between two
  AMD parts, with nothing in CPUID reporting which applies. Both
  figures are least-squares fits over a sweep from 10,000 to 2,000,000
  units, worst residual 1.8% and 0.4%; the pair's fixed cost is 2369
  and 1606 cycles. Neither number is read anywhere in the crate. The
  code passes the remaining count through unscaled, which can only
  undershoot a deadline and never overshoot it, and the difference
  costs an iteration.

  **And a monitor wait that decodes is not a monitor wait that
  holds.** On a Ryzen 9 7900X the arm beats the kernel park in every
  cell measured - 4.7 to 1.2 us idle and 5.1 to 0.7 us under load at a
  50 us inter-arrival, over three runs - but the identical code on a
  Ryzen 7 2700 is a severe regression under load: 444 to 755 us
  against the 15 to 25 us of the park it replaced. There the monitor
  does not survive to be waited on, `MWAITX` returns straight back,
  and re-arming becomes a spin over a 1600-cycle instruction pair.
  CPUID reports bit 29 on both parts and distinguishes them not at
  all; only the length of a real wait does.

  So the wait checks that it held. Four arms inside 50,000 RDTSC
  cycles is a monitor that never armed, and the wait then ends in
  `thread::park`. This answers a narrower question than the
  controller above and sits underneath it: whether the instruction
  does anything at all on this host, which decides whether there is a
  second strategy worth timing. Which of two working waits is cheaper
  is the controller's job, and no threshold can answer it.

  **There are three cases and the threshold has to separate all
  three**, which is why that figure is measured rather than argued.
  Four arms on a monitor that never armed cost about four times the
  instruction pair: 9,476 cycles on a 7900X, 6,424 on a 2700. Four
  arms on a monitor that armed and was cut short by something else
  measured 420,791 cycles on a 7900X under load, about 22
  microseconds each. A monitor that holds covers the whole budget in
  one arm. The middle case is a monitor doing its job while the host
  is busy, and an earlier threshold of a million cycles swallowed it:
  a working monitor was condemned for being woken by exactly the load
  the arm exists to survive.

  **The verdict is not taken on one wait.** It is permanent and
  process-wide, so eight separate waits have to agree before the
  process stops trying. A host whose monitor never holds supplies
  those in microseconds; one whose monitor holds needs eight
  independent coincidences. Requiring only one was tried and is why
  this is written down: a single unlucky wait on a 7900X condemned
  the process, and every group after it in that run measured the
  kernel park under a MONITORX label, so two runs of one binary read
  0.99 us and 5.08 us for the same cell depending only on when it
  fired.

  Once taken, the verdict is never cleared: a wrong `false` costs the
  kernel park that was already there, a wrong `true` costs the
  regression on every park, and the two directions are not worth the
  same.

  **Each `Parker` owns a cache line.** A monitor wait watches the line
  its `wake_counter` sits in and wakes on any store to it. The struct
  was 32 bytes with no alignment, so two parkers shared a line and
  each one's unpark fired the other's monitor, and an `Arc`'s
  refcounts sit immediately before its data so every clone and drop
  fired it too. In a pool the monitor therefore never held. Found by
  a consumer's workload rather than by this crate's bench, which
  constructs one parker and so has no neighbour to be disturbed by.

  `benches/parker_wait_strategy.rs` carries all three arms in one
  process, each at a 50 us and a 500 us inter-arrival and each both
  idle and against busy threads occupying half the host, plus a group
  that allocates a parker per logical CPU the way an arena does and
  times one of them while a thread unparks the rest.

  It names the group the fallback fired in, not merely that it fired.
  A fallen-back row is numerically identical to the control row
  beside it, and because the verdict is process-wide one group can
  turn every later group into the kernel park, so a run-level notice
  leaves every row after it unreadable and every row before it
  indistinguishable from those.

  Both benches print the tree they were built from. Two hosts here
  carry a directory called `Flynnel-verify` and they are different
  checkouts, so a run could otherwise be against source several
  commits behind the one being reasoned about with nothing in the
  output saying so.

- **A PowerShell module, `pwrs/`, binding the scheduler surface
  directly to the Rust.** 109 cmdlets, 79 classes and 36 enumerations
  over PWRS, plus a read-only `Flynnel:` drive. It is a binary module:
  no marshalling layer, no second implementation, and the objects a
  cmdlet writes are the crate's own readings.

  The families: the host's topology, CPU facts, inter-core latency
  table and cache allocation; job plans that resolve a worker count,
  leaf shape and execution tier; the worker pool with its spin, split
  and IO dials; kernels over arrays, files and text that run on the
  scheduler's own threads; the trace ring, leaf statistics, occupancy
  and call sites; the calibration store; the nine in-process ring
  shapes; backends and accelerator ops; the hybrid CPU-and-device
  shapes; the verify chain and the CGRA mode region; the GPU peer's
  watchdog, wave planner, linear-algebra chooser and peer lifecycle;
  and the cross-process router and pass registry.

  **Anything over many items crosses in one call.** Measured at this
  boundary: a method call costs 1907 ns on PowerShell 7.6 and 651 ns
  on Windows PowerShell 5.1, the same call batched a thousand at a
  time costs 4.5 and 3.0 ns, and one pipeline record costs 1712 and
  7955 ns. So every cmdlet that can take many items takes them all at
  once, and a per-record form exists only where a script wants to
  interleave, with its cost in its own help.

  **No cmdlet runs a script block on a worker.** A script block runs
  only on the thread owning the pipeline: the binding framework's
  pipeline token is `!Send` and the managed side refuses a stream call
  from another thread. Work that runs on the pool is therefore work
  the module declares.

  **Objects construct from script as well as from their cmdlets.**
  `[Flynnel.JobPlan]::new(8, 100000)` builds what `New-FlynnelPlan 8
  100000` builds, and `[Flynnel.GpuPeerConfig]::new()` builds the
  crate's default peer settings, as `New-FlynnelGpuPeerConfig` does.
  Each cmdlet calls its constructor, so the two routes cannot drift,
  and the suites check them equal at every arity on both editions. The
  five singly built proxy classes have constructors on the same terms,
  and the ring shapes that come as several objects build from one
  factory type. A constructor has no pipeline, so where a cmdlet warns,
  it refuses or builds as asked: a plan with a PerItemNs and no
  TaskOverheadNs, which New-FlynnelPlan warns leaves the leaf-width
  model nothing to solve, is built as given.

  **Sizes take a suffix on both editions.** PowerShell 7 binds the
  string `'2MB'` to a numeric parameter by itself, and Windows
  PowerShell 5.1 refuses it, which is the form a size takes when it
  comes from a variable, a CSV or a settings file. So 18 byte-size and
  large-count parameters carry a transform that reads the forms 7
  reads, exactly as 7 reads them, and hands everything else back
  unchanged. On the 24-thread host a bind through it costs 0.8 to
  0.9 us more on PowerShell 7.6 and 6 to 10 us more on 5.1, whose
  binder charges 5 to 8 us for any transform at all.

  **A completeness gate, `pwrs/src/bin/census.rs`, reads the crate and
  fails on any public item neither bound nor recorded in
  `census.toml` with one of six fixed reasons.** A binding missing a
  function is invisible otherwise: the module imports, every suite
  passes, and the gap shows up when someone needs the thing. A test
  written against the module cannot find it, because the module is
  what is incomplete. 368 items remain uncovered, all of them in the
  GPU-peer and cross-process families and in the racing arms that need
  a body able to decline or disagree.

  Built on PoWerRuSt 0.2.0. 631 tests pass on Windows under
  PowerShell 7.6.6 and the same 631 under Windows PowerShell 5.1, and
  628 on a Linux guest under 7.6.5, which skips three more; each suite
  prints the host and edition before it
  asserts, because one pipeline record costs three times more on 5.1
  and a figure without its edition cannot be compared with one from
  the other.

- `benches/verify_chain_fold.rs`, which measures what ordering the
  verify chain's fold costs, per chunk and per byte, with a quiet arm
  and a loaded arm in one process.

  Measured on the 24-thread bare-metal box under the measurement
  lease, three arms - a baseline, the same code against that baseline,
  and the unordered fold patched back. The ordering costs about 11 per
  cent on a sequential stream of 64-byte chunks and about 43 per cent
  on a four-producer fan-out of them; at 4 KiB and above it disappears
  into the hasher.

  Only two of the five cells are readable and the control is what says
  which. Identical code against its own baseline moved +3.05, +0.92,
  +5.16, +10.46 and +7.45 per cent, four of them at p = 0.00, so
  criterion's own significance test cannot see between-run drift at
  all: it is computed from the spread within one run. The three cells
  reporting the unordered fold as slower are mechanically impossible -
  it does strictly less work - and all three sit in the drift
  direction. A comparison taken this way resolves nothing below about
  ten per cent on either host.

- `tests/affinity_follows_process_mask.rs` holds what
  `FLYNNEL_LEVER_ALLOWED_WIDTH` does. It starts the arena at full width,
  narrows the process affinity mask to two CPUs, and asserts that
  `allowed_parallelism` follows, that `resolved_workers` caps by it, and
  that both come back when the mask widens - so it measures a pool that
  outlived the change rather than one sized after it. The behavior was
  measured from outside on a guest before this, which left a ratchet
  here silent.

  It occupies its own test binary because the mask is process-wide: a
  test running beside it would be narrowed by it and timed through a
  mask it did not set. A host that cannot narrow, or allows fewer than
  four CPUs, fails rather than returning early, because `eprintln` in a
  test binary is captured and a skip reports the same green as a run. It
  polls for the width instead of sleeping the re-read cadence, so it
  carries no second copy of `RECHECK_INTERVAL_MS`.

  `libc` joins the dev-dependencies for the Linux and FreeBSD arms,
  whose affinity calls differ in name, arguments and set type. The
  Windows arm declares the kernel32 symbols it calls, so no binding
  crate is added for it.

  Running it found that `FLYNNEL_LEVER_ALLOWED_WIDTH` capped by nothing
  on Windows. The Linux and FreeBSD arms passed; the Windows arm
  narrowed the process mask to two CPUs on a 24-thread host and
  `allowed_parallelism` still read 24.
  `std::thread::available_parallelism` documents the reason - it "may
  overcount the amount of parallelism available on systems limited by
  process-wide affinity masks, or job object limitations" - so
  `sched::host_width` was reporting the machine there rather than the
  share of it the process may use.

  `sched::host_width` now reads `GetProcessAffinityMask` on Windows and
  `available_parallelism` elsewhere, so the same question is asked by
  the route each platform answers it on and the lever caps on all three.
  A failed call falls back to `available_parallelism` rather than
  publishing a width nothing measured, and the mask is per processor
  group, so above 64 CPUs it describes the group. Both the module and
  the architecture page had said the width honoured the mask without
  naming a platform, and now say which call each platform uses.

  The test reads the mask back through `sched_getaffinity`,
  `cpuset_getaffinity` or `GetProcessAffinityMask` and asserts on that
  before the width, so a mask that never took and a mask the platform
  ignored fail with different messages. Without that the two arrive as
  the same assertion, which is how the Windows result read at first.

### Fixed

- **A monitor wait reported its own deadline as a wake.**
  `Parker::park_until` dispatched to the wait once and returned `true`
  whatever came back, but `UMWAIT` returns on its TSC deadline as well
  as on a store to the watched line. An idle worker was therefore
  handed back to its caller every deadline with nothing to do, and the
  caller re-entered through the whole spin floor; with the 10 ms cap
  that is a worker waking a hundred times a second to find the same
  empty deque. It also broke the shutdown contract outright, returning
  `true` before any shutdown arrived. Both monitor arms now re-read
  state on the deadline and wait again, which is what the deadline was
  for: catching a store that went missing, not announcing one that
  never happened.

  Present since the WAITPKG arm was written and never executed, because
  no host this project measures on has WAITPKG. The MONITORX arm below
  is what ran it for the first time.

- **The MONITORX inline-asm blocks promised flags they clobbered.**
  Both cleared `ECX` and `EDX` with `xor` inside a block marked
  `options(preserves_flags)`. `xor` writes flags, so the promise was
  false and the caller's next conditional read whatever the block left
  behind; on a Zen 4 host this surfaced as `park_until` returning
  `true` straight through a shutdown. The zeros are operands now, so
  nothing in either block writes flags ahead of the wait instruction.

- **`VerifyChain` rooted over the order chunks finished, not the order
  they were submitted.** A hash chain is ordered: `update(a)` then
  `update(b)` is not `update(b)` then `update(a)`. Submitting to the IO
  pool means tasks complete in whatever order the pool runs them, and
  each folded itself in as its own task finished, so two runs over
  identical chunks could root differently. A chain whose entire purpose
  is deciding whether a CPU trace and a device trace are bit-exact
  would have reported a mismatch between identical ones.

  The index is now taken on the submitting thread and a ready prefix is
  folded as it completes, with `finalize` folding whatever is left.
  Proved to fire by restoring the old behaviour: both new tests fail
  on it - `arrivals [3, 1, 0, 2] rooted differently from submission
  order`, and `nothing folds while index 0 is missing` - and pass on
  the fix. Its cost is in Added, above.

- Two intra-doc links in `JobPlan::k_inner_log2` pointed at items
  behind the `gpu-peer` feature, so they resolved with that feature on
  and dangled with it off. Neither `cargo check` configuration saw it,
  because neither runs rustdoc, and rustdoc's own default for the lint
  is a warning a gate reading exit codes cannot act on. The crate now
  denies `rustdoc::broken_intra_doc_links`, which found both, and the
  doc leg of the gate can fail.

- `registered_accel_ops()` is new, because the accelerator-op registry
  could not be read at all: `registry()` was private, `AccelOp` was
  private, and `AccelOpId`'s field was private, so a caller holding no
  id from `register_accel_op` had no way to reach an op. That is the
  position anything inspecting the process from outside is in. It
  returns the id, name, per-item byte estimate and bound backends
  under one read lock, with two tests: that a registered op appears
  with its bytes and bindings, and that every listed id resolves back
  to the name it was listed under, since the id is an index into a
  private vector and a mispaired listing would send every later
  `accel_target` at the wrong op.

- `watchdog::detect_with_model()` is new. `detect` read the device's
  driver model and threw it away, so a caller reporting both the model
  and the watchdog state would have loaded NVML twice for a fact that
  cannot change under a running process. Both entry points share one
  cache.

- `.gitignore` ignored `/target` only at the workspace root, so
  `pwrs/target` was untracked and unignored.

## 0.6.0 - 2026-09-13

### Removed

- `JobPlan::k_gating`, `JobPlan::with_k_gating` and
  `WorkloadShapeHints::k_gating`. The field was written by three
  constructors, the builder and the shape mapping, and read by nothing.
  It could not be wired either: every worker carries a KHL and an Fcl
  backing at once with an atomic tag naming the live one, and a job
  pushed to the other is reachable only while that worker's orphan-drain
  flag is set - a flag any pop or steal clears on finding that backing
  empty, so a push racing a probe strands the job. Honoring the hint per
  dispatch means probing both backings on every steal, which is the cost
  the flag exists to reclaim.

  A hint with no reader cannot report that it is doing nothing, so it
  reads as working forever. Deprecating it would have left that on the
  surface for another release.

  `KGating` itself is untouched and still load-bearing:
  `calibrate_k_gating`, the per-worker `AtomicU32` tags,
  `LocalArena::migrate_all_workers_k_gating` and
  `AdaptiveDispatcher::migrate_k_gating` are the granularity K-gating
  actually has. Callers pinning a backing should use the migrate call.

  `WorkloadShape::hints()` now returns the hints that steer.

### Added

- `CallSiteState::window_cv2_range_per_mille()` reports the smallest and
  largest per-window cv^2 across every classifier tick.
  `window_cv2_per_mille` holds the latest tick of what is often
  thousands, and on a 16-core Linux guest five identical runs of the same
  harness cell read 1, 0, 527, 209 and 217 per mille, spanning both the
  uniform edge at 50 and the high-variance edge at 500. So one reading
  cannot say which regimes a run passed through, and a maximum below the
  uniform edge is what says a spread-driven mechanism was never consulted
  in its own regime. `examples/throughput_under_load` and
  `examples/smt_recovery` print it as `cv2_window_min` and
  `cv2_window_max`, `examples/class_migration_under_load` as a
  `window_cv2_range` column, and `examples/lever_report.py` and
  `examples/smt_report.py` carry them through.
  `throughput_under_load` also takes the mode-2 block size as a ninth
  argument, because the block that clears the edge differs by host.

  The dispatch path is unchanged. `JobPlan::effective_use_smt` reads the
  latest tick deliberately: it is consulted per dispatch and wants the
  classification current at that moment, which ages as later windows
  replace it. Only the reporting was wrong.

- `spin_adapt_decisions()` reports how many times the adaptive spin
  controller passed its 256-event gate and reached a decision. The
  window alone cannot answer whether the controller ran: a
  rescue-dominated workload grows the window and is clamped to the
  tuned default it started from, so `spin_window()` reads 500 whether
  the controller decided on every park or never gathered the evidence.
  Probed on a 24-thread host with `FLYNNEL_ADAPTIVE_SPIN=1`, a
  `Streaming` shape at 8192 reps and a `FineGrain` shape at 512 both
  reported a window of 500, and which of the two had happened was not
  recoverable from any published figure.
  `examples/throughput_under_load` prints it as `spin_adapts` beside
  `spin_window`.

- `JobPlan::optimal_chunk_count_for(workers, n)` sizes the Tiny-Tasks
  model over an item count the caller names, where
  `optimal_chunk_count` sizes it over the plan's own `batch_size`. The
  model was consulted at one site in the crate,
  `collect_indexed_tiny_tasks`, which allocates and returns a `Vec`, so
  a kernel accumulating into a caller's buffer could not reach it and
  had to hand-roll a leaf floor from a constant of its own. Reported by
  a consumer with the constant attached.
- `gpu_peer::device_memory_in_use()` reports bytes in use and the
  device total for the calling thread's current context, covering the
  whole device rather than this process. A caller sizing an allocation
  against the total alone collides with a neighbour the total cannot
  see. Memory is what the driver exposes; utilization is not on this
  surface, so a process holding a context while launching nothing
  registers as its allocation and not as load.
- `JobPlan::resolved_workers()` answers how many workers will actually
  run this plan's work: the arena's primaries, plus its SMT siblings
  when the plan wakes them, capped by `worker_cap`. `effective_workers`
  takes the pool width as an argument and there was no supported route
  to it from outside the crate. The chunked walks divide by this count.

  Under the default worker sizing it equals the arena's whole width,
  because the per-node count defaults to the node's logical threads and
  no SMT extension is spawned when the primaries already cover them. It
  diverges only under `FLYNNEL_SCHED_PHYSICAL_ONLY=on`,
  `FLYNNEL_SCHED_SMT=off`, or an explicit `FLYNNEL_SCHED_WORKERS` below
  the logical count, where the siblings exist and park unless a
  dispatch with `use_smt` wakes them. There `NumaArena::total_workers`
  counts threads that exist rather than threads that run, and sizing a
  plan against it overstates parallelism twofold, asking for chunks
  about 1.4 times too narrow. The arena's own doc gave that 24-against-12
  example without saying it describes the non-default sizing.

### Changed

- `for_each_chunk_indexed_min_leaf` and `for_each_chunk_triple_min_leaf`
  derive their leaf width from the Tiny-Tasks model when the plan
  carries both `estimated_per_item_ns` and `task_overhead_ns`, treating
  the caller's `min_leaf` as a floor under the derived width rather
  than as the whole answer. A per-item cost the entry probe measured
  counts as carried, so a plan supplying only `task_overhead_ns` reaches
  the model on this host's own measurement. A plan carrying neither
  hint, or only the per-item cost, splits exactly where it did: no call
  site in `src/` sets `task_overhead_ns`.
- Every helper taking `min_leaf` documents what the parameter bounds.
  It is a floor on leaf size, not a granularity leaves are multiples
  of, so `items.len() / min_leaf` caps the leaf count at any worker
  count; a consumer read it as a granularity, tied it to a cache
  blocking width, and turned a four-cell blocking sweep into a sweep of
  parallel-leaf count. `for_each_chunk_min_leaf` bounds from the
  opposite side, capping what `adaptive_min_leaf` may pick, and
  `for_each_chunk_ref` takes a chunk width rather than a bisect bound.
- A declared matrix-extension `hw_class` no longer promotes a batch of
  one. One item is one leaf, so there is nothing to split and the
  dispatch can only add its own cost. A consumer's sweep on an
  AMX-emulating strip measures a one-tile grid at 2,388 ns serial
  against 2,827 ns dispatched. The promotion is still too eager above
  one: the same sweep has a 2x2 grid at 0.94x and a 4x4 at 2.71x, so
  the crossover lies between 4 and 16 tiles and is not yet measured.
- A declared matrix-extension `hw_class` promotes the scheduler tier.
  `with_hw_class` set a field that nothing outside tests read, under a
  `SchedTier` doc naming `hw_class` among the inputs the tier is
  selected from. The smallest tile a caller can mean by `AmxBf16`,
  `AmxInt8`, `Sme` or a tensor-core class is a multiply-accumulate over
  a 16x16 block, orders above the dispatch the inline fold weighs
  against, so the tier selection now treats such a class as heavy per
  item alongside an explicit per-item cost and `use_smt`. The vector
  classes say nothing the batch size does not and still steer nothing.
  `SchedTier`'s doc now lists what the selection reads.
- Five noisy-host mechanisms ship behind runtime switches, every one
  DEFAULTING OFF, read once per process from the environment:
  `FLYNNEL_LEVER_ONCORE_SPREAD`, `FLYNNEL_LEVER_BATCH_WEIGHT`,
  `FLYNNEL_LEVER_SMT_WINDOW`, `FLYNNEL_LEVER_ALLOWED_WIDTH` and
  `FLYNNEL_LEVER_CALIBRATION_REFUSAL`. `sched::levers::describe` reports
  their state for a harness to print beside a result.

  Off is what shipped before them, so a consumer who sets nothing gets
  the behavior it already had. Each has to earn its default against the
  same code without it: never slower on a quiet host, faster on a
  contended one. A switch is what makes that measurable - one binary
  runs both arms, so the comparison carries no difference in commit,
  harness, build or machine.

  Environment rather than cargo features, because a feature is chosen at
  build time and an A/B across one needs two binaries that differ in
  more than the feature.

  The entries below describe what each switch does when it is on.

- `FLYNNEL_LEVER_ONCORE_SPREAD`: a call site's classifier takes its
  per-item SPREAD from the thread's own clock, and keeps its mean on
  wall time.

  Wall time rises both because the work is irregular and because the
  thread lost its core, and preemption lands on some leaves and not
  others, so it reaches a wall-time spread as variance indistinguishable
  from the work's own. A thread's clock does not advance off a core, so
  a preempted leaf reports what it cost rather than what it waited. The
  sampled path brackets each leaf with both clocks; a window that
  carried no on-core reading is classified on wall time, as every window
  was before.

  WHICH DISPATCH ENTRIES IT REACHES. The bracket is taken by the
  sampled leaf recorder, and only the plain steal-driven bisect calls
  that, so the switch changes what `for_each_chunk` and
  `for_each_chunk_min_leaf` record and nothing else. The indexed and
  triple bisects time every leaf through a recorder with no bracket, so
  a site dispatching through `for_each_chunk_indexed`,
  `for_each_chunk_indexed_min_leaf` or `for_each_chunk_triple_min_leaf`
  takes no on-core reading whatever this switch is set to. A caller can
  tell which case it is in from `CallSiteState::oncore_items`, which
  stays at zero when no leaf was bracketed.

  Measured paired inside one process, where the two figures cover the
  same leaves and differ only in which clock timed them: across a load
  event the wall spread rose more than the on-core spread in every one
  of six trials on the gather shape, with the two agreeing on the quiet
  window either side of it.

  The on-core ticks are never converted to nanoseconds. On Windows that
  clock counts executed cycles against a fixed-rate elapsed counter, so
  a conversion would carry the achieved-to-base clock ratio into every
  figure; a cv^2 is a ratio and a common factor cancels out of it. They
  carry their own item count, because the sampled path takes a reading
  for some leaves and not others and each figure divides by the items it
  covers.

  Two clock pairs per sampled leaf, taken only on the sampled path.
  Each worker pays for its own sampled leaves, so what a dispatch adds
  is that cost times the sampled leaves ONE worker ran, not times the
  whole dispatch's.

  The cost is not a constant, and not a constant per target either.
  Measured with `examples/clock_cost.rs`, in nanoseconds, where the
  bracket is two pairs and the last column is that amortized at the
  sample stride:

  | host | thread clock | elapsed | bracket | per leaf |
  |---|---|---|---|---|
  | Ryzen 9 7900X, Windows, quiet | 206.8 | 6.1 | 425.7 | 53.2 |
  | Zen3 Linux guest | 670.6 | 28.6 | 1398.4 | 174.8 |
  | Zen3 Linux guest, busier | 973.4 | 38.1 | 2022.8 | 252.9 |

  On Linux and FreeBSD the thread-clock half is
  `clock_gettime(CLOCK_THREAD_CPUTIME_ID)`, which the vDSO fast path
  refuses, so it enters the kernel; on Windows it is
  `QueryThreadCycleTime` against `rdtsc`. Two readings on one guest
  differ by 1.45x, so the machine and what runs on it vary more than the
  target does. A caller sizing anything against this figure should read
  it on the host rather than assume one.

  What it is against: at a per-item cost of 1 ns a 256-item leaf is
  256 ns, and the bracket is most of it. At 2 us an item the same leaf
  is 512 us and the bracket is a fraction of a percent. The switch is
  also only consultable in the second regime, because
  `classify_observed` returns on the mean alone below `port_heavy_ns`
  and never reads the spread this improves.
- `FLYNNEL_LEVER_BATCH_WEIGHT`: a leaf batch is recorded at the share of
  its interval the pool spent on a core, and every sum a call site keeps
  carries that weight as a factor along with the count they are divided
  by, so the means do not move and only a contended batch's influence
  does. The pool ticks are recorded either way, because that clock pair
  is read regardless and a site's reported occupancy should not depend
  on which arm is running. `leaf_count` stays a
  count of leaves, because the sample guards and the classifier quantum
  read it, and a batch whose ends did not both carry an on-core reading
  weighs a whole batch: no reading is not evidence of contention.

  This weighting reaches the class through the SPLIT rather than through
  the figure the classifier reads - the chain runs weight, cv^2, SMT
  decision, leaf floor, split - so it separated from the baseline on an
  adaptive routing and on neither pinned one. It is kept because a
  contended sample counting for less is right on its own terms, and it
  is not what the spread change above rests on.

  It changes a figure only where batches DIFFER in the share they held.
  Each statistic divides a weighted total by a weighted count, so a
  share every batch in the window shares cancels exactly: a host that is
  contended steadily moves nothing here however contended it is. What
  the weighting is for is a window carrying contended batches and quiet
  ones together, which is what `examples/throughput_under_load.rs` makes
  with its `duty_ms` argument.
- `FLYNNEL_LEVER_SMT_WINDOW`: `JobPlan::effective_use_smt` decides from
  the window the site's classifier last read rather than from its
  lifetime cv^2, falling back
  to the lifetime figure only until a window has been classified. The
  lifetime figure is an equal-weight average over every leaf a site has
  ever run, so a contended stretch could be diluted only by running
  enough later leaves to outweigh it, and an SMT decision resting on it
  did not come back when the host went quiet. The window is also the
  quantity `cv2_low_per_mille` bounds when a class is decided.
- `FLYNNEL_LEVER_ALLOWED_WIDTH`: `JobPlan::resolved_workers` caps by the
  CPUs the process may currently use, through the new
  `sched::host_width::allowed_parallelism`. The
  pool is spawned once and its threads outlive a change to the process
  affinity mask or the cgroup CPU quota, so a host that has narrowed
  since startup left the arena counting workers that cannot reach a
  core. `std::thread::available_parallelism` honours both and is re-read
  at most every 250 ms, because the answer costs a syscall and on Linux
  a cgroup read. A failed query says so once, naming the error and the
  width it keeps, and holds the last successful reading: one is a width
  a genuinely pinned process has, so resolving an error to it would
  silence the pool wherever the query is unsupported.

  The cap binds only where the allowed width has actually narrowed. A
  busy neighbor does not move the affinity mask or the cgroup quota, so
  on a merely contended host this switch caps nothing and changes
  nothing. `examples/width_narrowing.rs` reports the allowed width and
  the resolved worker count beside each window so a run can say which
  case it was in.
- `CpuCalibration` records the occupancy its draw ran at, and
  `CALIBRATION` layout version rises from 4 to 5, so the first start on
  any host after this measures again. The spread a record already
  carried says whether its samples agreed with each other; it cannot say
  whether they agreed on the wrong number, which is what a draw taken
  while a neighbour held half the machine produces - every sample slow,
  and slow by about the same amount. The figure gates nothing: the share
  of free cores is a continuous property of a host rather than a state
  it is in, so no cutoff separates a contended draw from a clean one,
  and a reader holding two records can prefer the better-drawn one.
  `OCCUPANCY_UNRECORDED` is distinct from zero, which is the share a
  thread that never reached a core genuinely reports.
- `CpuCalibration::new` takes the occupancy as a sixth argument, which
  is a breaking change for a caller constructing one directly.
- `OccupancySample::per_mille` reports the same fraction as `percent` at
  a thousandth rather than a hundredth. Ten per mille is the whole
  distance between a pool that held its cores and one that lost a
  hundredth of them, which is what a reader comparing two records is
  looking at.
- `FLYNNEL_LEVER_CALIBRATION_REFUSAL`: a calibration draw no longer
  displaces one taken on a quieter host.
  `publish` overwrote whatever the table held, so a process measuring
  while a neighbor held half the machine replaced a better-drawn record
  and published its own for every later process to read.
  `WriterGuard::publish_if_better` compares the two records instead.

  It is an ordering between two records, not a threshold on either. No
  occupancy is called good or bad: the share of free cores is a
  continuous property of a host rather than a state it is in, so a
  cutoff would be a policy about how much of a machine a calibration
  insists on and would have to be argued as one. Two records can still
  be compared without settling that.

  The comparison runs only where it can decide. An incumbent whose own
  samples disagreed is replaced whatever it was drawn at, because its
  spread already says it does not describe the host. Where either record
  lacks an occupancy the two cannot be ordered and the offer is
  published, so a platform with no thread clock cannot freeze the table.
  A tie publishes, because the fresher draw describes the host now. A
  refusal returns `PublishOutcome::KeptIncumbent` naming both figures
  rather than passing silently.

  THE COMPARISON IS UNREACHABLE AND THE SWITCH CHANGES NOTHING, on any
  host. A consumer setting it should expect no behavior change until
  that is fixed.

  `publish_if_better` has one caller: the branch `stored_or_measured`
  takes when the stored record is absent or FAILED
  `CpuCalibration::is_trustworthy`. `prefers_incumbent` returns `None`
  on its first line unless the incumbent PASSED the same check. A record
  that passes is returned before that branch is reached and is never
  offered for comparison, so the two conditions cannot both hold.

  The unit tests build a `CalibrationStore` directly and call the guard
  with an incumbent of their choosing, which is why they pass while the
  path a process takes is never exercised.

  Making a stored record something a fresh draw can displace is what
  would deliver an incumbent to the comparison. Until then
  `CpuCalibration::occupancy` is provenance a reader can inspect and
  nothing acts on.
- `sched::par_iter::sample_iqr_per_mille` reports the interquartile
  range of a sorted sample set over its median, beside the existing
  `sample_spread_per_mille`, which reports the full range over the same
  median. Nothing reads either, and the pair is reported together
  because having both is what showed neither can serve.

  A range is defined by the two samples a median exists to survive, so
  one scheduling hiccup in nine sets it. The interquartile range fixes
  that and does not fix the thing that matters: measured across three
  load levels on two hosts, both INVERT. Under saturation every sample
  in a draw is slowed by about the same factor, so the samples agree
  with each other while the medians independent draws produce scatter
  enormously. On a Zen3 guest the interquartile range read 128 per
  mille quiet against 86 saturated, while the medians of independent
  draws went from 83 to 432,299.

  The consequence for a caller: a dispersion figure taken over one
  draw's samples does not say whether that draw's median is
  reproducible, and under load it says the opposite. The figure that
  does is the disagreement between two draws, which on the same data
  separated those conditions by 14,149 times where the samples' own
  spread separated them by 2.4.
- `examples/clock_cost.rs` times what a thread-clock read costs on the
  running host and what the sampled leaf bracket amortizes to at a given
  stride, timing each half as the platform runs it -
  `clock_gettime(CLOCK_THREAD_CPUTIME_ID)` against `Instant::elapsed` on
  Linux and FreeBSD, `QueryThreadCycleTime` against `rdtsc` on Windows.
  It refuses to run where there is no thread clock rather than reporting
  the cost of returning an absence.
- `examples/width_narrowing.rs` times dispatch throughput while the CPUs
  the process may use are taken away from under it, printing its pid
  before it starts so a driver can narrow it partway through. It exits 3
  when every window saw one allowed width, because a run the narrowing
  never reached produces the same rows as a switch with no effect.
- `sched::spin_adaptive` reports whether the park-versus-rescue
  controller is running, after the environment has been read. The window
  `sched::spin_window` returns sits at its tuned default both when the
  controller is off and when it is on and the evidence keeps it there,
  so a caller reporting only the window cannot tell those apart.
- `examples/throughput_under_load.rs` takes the per-item cost and
  whether it varies with the index. A workload whose items all cost the
  same gives the call-site classifier a per-item spread of zero, and
  every adaptive mechanism here reads that classifier, so such a
  workload cannot show any of them doing anything. The irregular shape
  varies the cost over `1 ..= 2 * reps - 1`, keeping the mean at `reps`
  so the two shapes are comparable in total work.

### Fixed

- The GPU tests no longer report a neighbour as a Flynnel regression.
  `gpu_peer_team`'s barrier assertion and `gpu_peer_wave_calibration`
  are wall-clock, and the device lock in the test tree excludes
  Flynnel's own test binaries and nothing else, so a process from
  another project holds the device and the reading becomes about it.
  A 48-block team cannot assemble until that many streaming
  multiprocessors are free, so it is the size that suffers: on one host
  the same binary at the same commit read 267,104 ns quiet and
  2,369,440 ns loaded, a spread of about nine times, while teams of 2,
  4 and 8 barely moved. Both tests now sample device memory across
  their timed region through the new `gpu_peer::device_memory_in_use`
  and report a reading taken against foreign residency as unmeasured
  rather than failed, printing rather than skipping quietly. The
  threshold is the run's own measured footprint, not a chosen number.
  Deadline expiries are still asserted unconditionally: a team that
  missed its deadline missed it whatever else was resident.
- The wiki's `JobPlan` reference carried four claims the code does not
  support, each the published form of a defect this release corrects.
  `task_span_ns` was documented as "currently unused in
  `optimal_chunk_count`; reserved for span-aware extensions" and the
  formula beside it divided by the overhead alone. `k_gating`'s entry
  described a per-call backing selector and the builder table said the
  hint made "this dispatch land on a pinned backing"; both entries are
  gone with the field they described. `k_inner_log2`
  described a kernel naming `mul_slice` and `add_slice` that this crate
  does not contain. `hw_class` named only a mode-region path and not the
  tier promotion. The reference now also documents
  `optimal_chunk_count_for` and `resolved_workers`, and the scheduler
  reference states what `min_leaf` bounds on each helper that takes it,
  which is the sentence a consumer needed and did not have.
- `with_task_span_ns` and `with_k_inner_log2` steer something. Both
  shipped in 0.5.0 setting a field that nothing in the crate read, so a
  caller setting either got no error, no effect, and no way to tell.
  `task_span_ns` is now the other half of a chunk's fixed cost in
  `JobPlan::optimal_chunk_count`, which divides by `overhead + span`
  rather than by overhead alone: a larger span yields fewer, coarser
  chunks, and an unset or zero span leaves every existing split exactly
  where it was. `k_inner_log2` resolves through the new
  `JobPlan::k_inner_lanes` to the cell count
  `gpu_peer::linalg::cpu::gemm_batched_lanes` carries per k-iteration,
  which `gemm_tandem_batched` now uses for the CPU half of its split.
  Every output cell accumulates over `k` in the same order against the
  same operands at every width, so the result is bit-identical to the
  one-cell loop and the device-parity oracle `cpu::gemm_batched` is
  untouched.
- `k_inner_log2`'s documentation described a kernel this crate does not
  contain, naming `mul_slice`, `add_slice`, `FpN<N>` widths and a
  scalar fallback, and recommended widths per instruction set. It now
  states what reads the hint and what that does, and recommends no
  width: which one pays is a property of the host and the matrix shape,
  and `benches/gemm_k_inner.rs` measures it.
- `tests/no_hint_is_silently_dropped.rs` asserted only that these two
  builders stored their value, which a hint wired to no reader passes
  forever - under a file whose name claims that cannot happen. Both now
  have an assertion that the hint changes what it names, and the header
  says a round-trip assertion is not coverage.

## 0.5.0 - 2026-09-12

This release is **0.5.0, not 0.4.1**: it changes public signatures under
`sched::occupancy`, and in a `0.x` version the minor position is where a
breaking change goes.

Two things a consumer has to do rather than read:

- **A manifest pinning `flynnel = "0.4"` will not resolve this release.**
  Cargo reads `"0.4"` as `>=0.4.0, <0.5.0`, so the build keeps resolving
  the older crate, reports nothing, and looks exactly as it would if the
  release had never happened. Change the requirement to `"0.5"`.
- **`thread_on_core_ticks`, `OccupancySample` and
  `OccupancySample::percent` have new signatures, and
  `HAS_THREAD_CLOCK` is gone.** The Changed entry below says what each
  became and why. A caller that only read the occupancy percentage needs
  to decide what it does when the figure is absent, which is the point
  of the change.

### Changed

- A call site's learned class is decided from the cost of one item and
  the spread of that cost, not from whole-leaf times. A leaf's time
  scales with the items in it, and how finely work is split follows from
  the class, so classifying leaf times let a class hold itself in place:
  a uniform-item site that once left `Streaming` could not return, and
  measured 18 percent below `Streaming` on fine-grain work for as long
  as it stayed there. Every leaf recorder now carries its item count,
  and `CallSiteState::per_item_ns` and `per_item_cv2_per_mille` report
  what the class was decided from. A window whose samples carry no item
  count, such as the heartbeat's serial spans, still classifies on leaf
  times.
- The recursion floor comes from the site's measured per-item cost
  rather than from `use_smt` standing in for "the items are heavy". A
  class that turns SMT on no longer drops the floor to one item, which
  is how a 5 ns item could be handed its own dispatch.
- Leaf times reach the classifiers in nanoseconds. `read_tsc` returns
  raw counter ticks on x86_64, and every consumer - the class bands,
  `effective_use_smt`, the split multiplier, the site's window - read
  them as nanoseconds, so every boundary sat out by the tick rate: a
  factor of 4.66 on one bench host. The conversion happens once per
  batch, from a rate measured once per process. On other targets the
  fallback clock now measures elapsed time rather than reading about
  zero.
- A thread's on-core count says whether it was measured, so a platform
  that cannot read one can no longer be mistaken for a thread that got
  no core. `thread_on_core_ticks` returns `ThreadTicks`, either
  `Measured(ticks)` or `Absent(reason)`, where the reason separates a
  platform with no such clock from a clock that failed to read - the
  first is permanent and the second is a fault. `OccupancySample`
  becomes an enum with a `Measured` and an `Unmeasured` case, and
  `percent` returns `Option<u32>`, whose `None` is an interval nobody
  measured. `HAS_THREAD_CLOCK` is gone: it stated the same fact as the
  returned value, and two mechanisms for one fact drift apart.

  This changes the signatures of `thread_on_core_ticks`,
  `OccupancySample` and `OccupancySample::percent`, all public under
  `sched::occupancy`.

  A batch of leaves whose on-core count is absent at either end now
  contributes neither figure to its call site, where before it would
  have added elapsed time against no on-core time and read as a site
  whose workers never held a core. The ratio is therefore taken only
  over batches that were measured, on every platform.

  The arm that reports no clock is compiled by none of the gated hosts,
  which are Windows, Linux and FreeBSD. The behavior a platform without
  a clock would get is tested on every host by constructing the values
  directly, and a separate test asserts that each gated host does read
  its own thread clock - so an arm that goes dead fails a test instead
  of reporting zeros.
- A global frontier's `imbalance_per_mille` is taken over the blocks a
  generation could reach rather than over the team. The frontier is
  dealt in runs of one block's threads from rank 0, so a generation of
  `n` ids reaches only its first `ceil(n / threads per block)` blocks,
  256 of them at the poller's launch; measuring the
  largest block's share against the whole team therefore scored a
  generation that reached ONE block as the worst imbalance possible,
  `1000 * team`. Since the figure keeps the largest reading of the wave,
  and a wave with fewer roots than a block has threads always starts
  with such a generation, the counter reported `1000 * team` for the
  whole run whatever the later generations did. It was measured at
  exactly 8000, 24000 and 48000 across 36 cells at teams 8, 24 and 48,
  with no variation at any depth or in any round - a constant, not a
  measurement. A partition is unaffected: every block there has a region
  of its own, so the blocks that could hold work are the team.

- A call site's per-item cv^2, the figure that decides a class above
  the port-heavy boundary, is formed in 128 bits from the recorder's
  scaled sums with no integer variance in between. It used to move in
  steps of `1000 / (mean^2 >> 16)` per mille - 26 at 1.6 us per item
  and 333 at the 500 ns bottom of the heavy band - against class edges
  at 50 and 500, so a site whose items cost a few microseconds read a
  cv^2 of 0, 25 or 51 depending on how a window's sums rounded and
  crossed the Streaming/MemoryBound edge tick to tick. Measured on a
  1.6 us-per-item shape on a quiet host: the class flapped on both the
  adaptive and the pinned arms before, and holds one class on every
  trial after, at the same wall. `CallSiteState::per_item_cv2_per_mille`
  and `window_cv2_per_mille` report the corrected figure; the integer
  mean is the one rounding left, at most `2 / mean` per mille.

### Added

- A call site measures what acting on its class is worth. The site runs
  a routing A/B on its own counters: one arm is the plan its learned
  class re-derives, the other the plan as the caller built it. It
  explores both, keeps the faster by moving average, and re-tests on a
  cadence, so a class that stops describing the work costs the trial
  rate rather than every dispatch until something notices.

- `CallSiteState::window_mean_ns`, `window_cv2_per_mille` and
  `window_ticks` report the delta window the latest classifier tick
  classified: its mean leaf time in nanoseconds and its cv^2 per mille,
  which are what the site's learned class is decided from, and how many
  windows have been classified. `cv2_per_mille` stays the site's
  lifetime figure. `examples/class_migration_under_load` prints them
  beside the class, with the sizes, counts, mean times and cv^2 of the
  leaves each row's interval ran, and takes `adaptive` or `pinned`
  routing so a class's effect on the split can be told from the load's.
- A user op can keep its slot. Returning `USER_OP_YIELD`
  (`FLYNNEL_USER_YIELD` in the CUDA source) leaves the slot in the ring
  with no status written, and the poller runs the same op again on its
  next pass, after its stop, generation and quantum checks. So an op can
  pace itself across passes and quanta while its state stays in VRAM.
  Only thread 0 of rank 0 decides. A failing op, or a team that does not
  assemble, still retires as before. The pre-generated PTX is rebuilt
  from the new source.
- A user op may address a `pin_bulk` span. Its byte count is now bounded
  by the end of the resident pool rather than by one pool block.
- `gpu_peer::watchdog` reads which GPU watchdog can reset a device. On
  Windows it reads the TDR level and delay from the registry and the
  driver model from NVML, both loaded at run time with no new
  dependency. TCC devices are outside TDR and level 0 disables it, and
  every failed read is named in the result.
- `gpu_peer::wave`: segmented waves across a lane's team. A wave runs a
  set of segments in generations on every block of the team, with its
  state in one resident span. `kernels/gpu_peer_wave.cu` is composed
  with the user source and provides:
  - a push index and an arena over `atomicAdd`;
  - a per-generation cross-block barrier with its own deadline and
    instrumentation. On a global frontier, block 0 reads the push count
    once every other block has arrived and publishes the next range, and
    every block takes its range from that;
  - a global frontier, or per-block frontiers that deal pending
    segments out evenly every N generations;
  - a slice end at which block 0 waits for every block;
  - failure carry that names the lowest failing segment;
  - a self-timed stop from the measured longest generation against a
    budget derived from the detected watchdog.

  A slice that stops early yields to run again on the device, or retires
  for the host to continue. `GpuPeer::create_wave`, `submit_wave`,
  `wave_stats` and `release_wave` drive it from the host.
  `wave::plan::plan` chooses between a global frontier and a partition,
  and the rebalance interval, from the calibrated barrier and rebalance
  costs and an observed imbalance. A best interval of one generation is
  planned as a global frontier.
- `GpuPeer::calibrate_waves` measures a device's wave costs at its team
  size: the barrier, the start skew, a rebalance's fixed cost and its
  copy per pending id, the fixed and per-segment cost of a slice round
  trip, and the longest generation. It runs Flynnel's own tree op
  (`layout::OP_WAVE_CALIBRATE`), which composing any user source brings
  into the poller module. The costs stay on the peer and are stored on
  the device record, and `GpuPeer::wave_costs` returns them. A later
  peer on the same device at the same team size starts with them.
  `WaveCosts::plan_inputs` turns the costs and a wave's stats into the
  planner's inputs. A global frontier records its per-generation
  imbalance. `WaveStats` also reports the ids moved by rebalances, the
  time block 0 spent in them, and the summed slice time.
- A wave can keep a reorder buffer (`WaveSpec::rob`), with a record per
  segment id and a row per root.
  - Segments link their children with `flw_push_child` and report
    themselves expanded, refused or retired.
  - While the other blocks wait at a barrier, block 0 commits each row in
    pre-order over (parent, ordinal). It stops at the first pending or
    refused segment, and the row keeps its lowest refusing id.
  - `GpuPeer::wave_rows` reads the rows.
  - Between slices, `push_wave_segments` and `report_wave_segments` add
    host-run segments to the same span.
- `GpuPeer::fetch_bulk_at` and `write_resident_bulk_at` read and write a
  resident span at an offset.
- `GpuPeerConfig::lane_teams` gives each lane its own team width, so one
  peer serves narrow waves from a one-block lane and wide ones from a
  wide lane. A wave's barrier is paid by every block of its team whether
  or not that block was dealt work: a global frontier reaches only its
  first `ceil(n / threads per block)` blocks, so a wave whose
  generations fit inside one block's threads ran faster on a team of one
  than on any wider team at every depth measured, and the only way to
  get both widths from Flynnel was two peers. Now
  `GpuPeer::create_wave_on_lane` lays a wave out for a named lane's
  team and pins it there; `GpuPeer::lane_team_size` reports each lane's
  width; `GpuPeer::pin_bulk_on_lane` chooses a handle's lane. Each entry
  is clamped to the device's streaming multiprocessor count and reported
  the way `blocks_per_lane` is, and a config naming some lanes but not
  all is refused rather than padded. The kernel is unchanged: every lane
  was already launched as its own grid at a width the kernel is handed.
  `create_wave` refuses a peer whose lanes differ, since the lane the
  pool would pick may not run the width the wave was laid out for;
  `team_size` and `calibrate_waves` describe lane 0.
- `GpuPeerConfig::user_ops_nvrtc_options` passes NVRTC options to the
  composed user-op module, such as `--fmad=false` for a kernel that must
  match the host bit for bit. A composed module is compiled once per
  process for each source and set of options, and
  `GpuPeer::user_ops_compiles` counts the compilations.
- `GpuPeer::team_size`, the blocks each lane actually runs.

### Changed

- `GpuPeer::init` clamps `blocks_per_lane` to the device's streaming
  multiprocessor count whenever the driver reports one, and says so on
  stderr. A team wider than the device loses ranks at its barrier.
  Measured on an RTX 5070 with 48 SMs, 64-block teams expired generation
  barriers in both passes of the barrier probe (2,496 and 704), retired
  slots incomplete and stalled the kernel, while teams of up to 32
  blocks ran clean. `team_size` returns the size in use, and it is the
  `team_size` a user op receives.

- An oversubscription factor the caller set with
  `with_oversubscription_log2` now reaches the seed-depth route, which
  is what `for_each_chunk` and `for_each_chunk_indexed` take when the
  plan carries an authoritative per-item estimate. It was read only on
  the routes that take a split budget, so the override did nothing
  exactly when the caller had supplied the per-item cost the docs ask
  for.

  It shifts the depth rather than scaling the target.
  `adaptive_seed_depth` derives a depth from the estimate, applies the
  worker floor and the hysteresis stabiliser, then adds the log2, capped
  at one leaf per item. Both of those stabilisers already work in depth,
  and the two orders agree only when the target is already a power of
  two: on a target of 51 with a factor of 2, scaling the target gives
  102 and seeds 128, while shifting the depth rounds 51 to 64 and then
  doubles it.

  Only a factor the caller set applies. One that arrived from a profile
  or a learned class does not, and neither does the process-global split
  multiplier, so on this route the only oversubscription is one the
  caller asked for. `oversubscription_log2_explicit` is what separates
  the two.

  A plan that sets no factor seeds what it seeded before.

### Fixed

- A deque-shape cooperative fan-out wider than one worker's ring could
  hang, with the dispatching worker spinning and every other worker
  parked. `cooperative_join_n_flat` at N = 1024 on a 24-worker host is
  where it was caught; any caller reaching the deque shape with more
  than `ADAPTIVE_SLOT_CAPACITY * JOBS_PER_SLOT` closures could reach
  it, which is 768 at the shipped values.

  The fan-out pushed all N-1 closures onto one worker's tier before
  waking anybody. Those pushes go through the burst path, which
  publishes to a ring of 256 slots holding three jobs each, and
  publishing to a full ring spins until a consumer frees a slot. The
  only wake sat below the push loop. So past 768 jobs the producer
  waited for a consumer nobody had woken and could not reach the line
  that would have woken one.

  ```text
     N   jobs pushed   ring holds 768
   512          511    fits
   768          767    fits, just
  1024         1023    over by 255
  ```

  Whether it hung was down to timing: if peers left awake by a previous
  dispatch happened to drain the ring, the push completed. That is why
  it presented as an intermittent hang at one width rather than a
  reproducible failure at a threshold.

  The first slot's worth still bursts and is then published and
  broadcast, so peers drain while the rest is pushed - the deque path's
  per-push wake is measured at 3.7x on N=8 and is kept. Past that the
  push refuses instead of waiting, and the caller runs a refused
  closure inline. That bound is the one `try_push_tier` already applied
  to the single-push path after the 65,536-item hang; the burst path
  had not taken it.

  Below one slot's worth nothing changed, so narrow fan-outs take the
  same path as before.

  The widths that do change are measured, because deferring the first
  broadcast had been measured at 3.7x on N = 8 and the early wake had
  to be shown not to undo it. Both trees at N = 8, 12, 16, 24 and 32,
  four passes alternating which tree ran first, every width in both
  registration orders, on a 24-worker Zen 4 under a measurement lease.

  No width regresses. Nine of the ten cells run faster on the fixed
  tree, by 0.9 to 7.7 percent, and the tenth is 0.9 percent the other
  way, which is inside the spread of one binary measured against itself
  across two passes.

  How much of that is the change is not separable on this host, and the
  reason is worth stating rather than rounding away. Three of the four
  arms in that sweep reach the same push loop at these widths: the
  mailbox entry point gates at 32 times the worker count, which is 768
  here, so below it that arm runs the deque branch too. They agree with
  each other, which is a second reading rather than a control. The one
  arm this change cannot reach is the rayon comparison, and across
  these ten cells it moves between 7 percent faster and 32 percent
  slower with no pattern, so it bounds nothing. The tree-shape arm is a
  control at N = 8, 12 and 16, where it moves under half a percent in
  one registration order and up to 4 percent in the other.

  So the claim this entry makes is the one the measurement supports:
  the fix does not cost the narrow widths anything.

- The crate builds and lints on Linux and FreeBSD. Three defects sat in
  code one platform compiles and the other does not, so a Windows-only
  gate could not see any of them: an undocumented public function in the
  non-Windows arms of `thread_on_core_ticks`, which failed
  `deny(missing_docs)`; a nested `if let` pair in the Linux NUMA probe,
  which clippy rejects at `-D warnings`; and the TDR defaults in
  `examples/gpu_peer_generation_barrier`, read only by `cfg(windows)`
  code and therefore dead elsewhere.

  A malformed entry under `/sys/devices/system/node` is now reported on
  stderr rather than discarded. An entry whose name does not begin with
  `node` is not a node and is skipped as before; one that does and whose
  suffix is not a number is named, since nothing else would say the
  topology was read from fewer nodes than the directory offered.

  The leaf-floor test that covers the per-item recursion floor asserted
  an exact leaf count, which is reachable only above a particular
  `pool_dispatch_cost_ns`. That figure is measured per process and moves
  with load, so the test passed on a loaded host and failed on a quiet
  one. It now asserts what the floor owes on any host: never above the
  caller's cap, never one item per leaf, and unless the cap binds first,
  one item fewer would not cover a dispatch.

- A host with no NVIDIA driver gets the error the API documents instead
  of an aborted process. cudarc resolves its driver symbols lazily and
  panics when it cannot load libcuda, and both `CudaBackend::with_device`
  and `GpuPeer::init` reached a driver call with no probe - the second
  while documenting that it never panics when no device is present,
  which is the contract a CPU-only fallback rests on. Both now prove the
  driver loadable first, through the cached availability probe and then
  `is_culib_present` against cudarc's own candidate library names. In
  `GpuPeer::init` the gate sits above the current-context read, which
  reaches the driver as well. On a GPU-less host the lib suite goes from
  seven aborting tests to none.

- `GpuPeer::create_wave` reads a device's watchdog once per process
  rather than once per wave. `watchdog::detect` loaded NVML, initialized
  it, read the driver model and shut it down on every call, and
  `create_wave` calls it for every wave whose slice budget is `Detected`.
  Measured on an RTX 5070 by a consumer crate, that was a flat 24.8 to
  28.2 ms per wave across spans from 120 to 41 000 ids - indifferent to
  span size because none of it was the span - and 99.7 percent of the
  time spent setting a wave up. The driver model and the TDR settings
  describe hardware and a driver configuration a running process cannot
  change, so the reading is kept per device.

- Occupancy reads a thread's own CPU time on FreeBSD, where it
  previously read nothing. `thread_on_core_ticks` had a Linux arm and a
  fallback returning zero, and FreeBSD took the fallback, so a spinning
  thread there reported no time on core. The per-thread CPU clock exists
  on that kernel; its id is 14 where Linux numbers the same clock 3, so
  one implementation now covers both and the id is chosen per kernel.

  The zero that fallback returns is no longer ambiguous. It meant both
  that a platform has no such clock and that a thread genuinely spent no
  time on a core, and `OccupancySample::percent` guarded only a
  zero-length interval, so a real interval with no clock read 0 where
  the module documents full occupancy. `HAS_THREAD_CLOCK` states which
  case a reader holds, and `OccupancyWindow::sample` reports the
  interval as owned where the fraction is unknowable - the reading that
  leaves a scheduler decision where it sat, against the one that would
  move it by treating every interval as idle.

### Scheduler

- The cooperative fan-out's mailbox gate moves from the worker count to
  32 times it. Below the gate a fan-out takes the deque shape, which
  distributes through the parent's own deque and random peer-steal;
  at or above it, the mailbox shape, which pushes each closure to one
  worker's mailbox.

  A caller reaching `cooperative_join_n` at a width between the pool
  count and 32 times it was previously routed to the mailbox shape and
  is now routed to the deque one. Nothing about either shape changed and
  both remain callable directly.

  Measured on a 24-worker Zen 4, every width run in both registration
  orders, comparing the deque fan-out against the routed entry point a
  caller actually reaches:

  ```text
     N   xpool    deque   routed   faster
    24      1     19.65    22.77   deque   by 13.0%
    64      3     34.01    38.23   deque   by 13.3%
   256     11    108.16   121.39   deque   by 12.1%
   512     21    197.77   202.81   deque   by  2.5%
   640     27    233.15   261.77   deque   by 12.3%
   768     32    270.85   262.27   routed  by  3.2%
   896     37    313.77   295.30   routed  by  5.9%
  1024     43    384.93   322.43   routed  by 19.4%
  ```

  Microseconds, mean of both orders. The crossing lies between 640 and
  768. Below the pool count both flynnel shapes run the same code and
  agree to within 1.5 percent, which is the control the rest rests on.

  The design's argument for owner-directed placement is that it beats
  random peer-steal once a fan-out is deep enough to be worth directing.
  That holds, and it holds much later than the pool width: mailbox mode
  leaves the parent and every untargeted worker idle for the wait,
  because peer-steal reads deques and cannot reach a closure sitting in
  a mailbox. Until a fan-out is wide enough that nearly every worker
  gets a target, that idleness costs more than the directed placement
  saves.

  The gate is a multiple of the pool rather than a constant, because the
  crossing is a property of how many workers must be reached before
  directed placement repays its cost. The multiple is measured on one
  host and one workload shape; a host whose sync costs differ will put
  the crossing elsewhere.

  Applied where the fallback is the deque fan-out. The `Auto` arm of
  `cooperative_join_n` still sends anything at or above the pool count
  onward, because its own fallback is the tree shape and routing the
  newly-excluded band there would be slower than either.

- `sched::occupancy` reports what fraction of a measured interval the
  measuring thread was actually on a core. Nothing consumes it: no
  learner refuses a window on it and no dispatch is routed on it, so the
  scheduler behaves exactly as it did without it.

  It is reported rather than acted on because what figure marks a
  contended measurement is not known. A threshold chosen ahead of the
  distribution describes whoever picked it, and this one would have been
  picked against a scale that turned out to be wrong by the host's clock
  rate. `FLYNNEL_OCCUPANCY` prints the host calibration's figure; the
  seed-depth bench prints one per arm.

  Every adaptive input in the scheduler was derived from wall time: the
  calibrated dispatch cost and its two thresholds, a call site's
  coefficient of variation, the policy-arm averages. Wall time rises and
  spreads for two unrelated reasons - the work is expensive or
  irregular, and the thread is not getting its core - and nothing
  separated them. Preemption lands on some leaves and not others, so it
  reaches the classifier as variance rather than as a uniform slowdown:
  a site whose leaves are uniform reads as irregular, migrates class,
  and the class it lands on selects a different fan-out shape and a
  different SMT setting. `Streaming` parks SMT siblings because both
  threads would contend for the bandwidth it is already saturating;
  `Gather` wakes them to interleave cache-miss loads. So a busy host
  moved the scheduler's choice rather than only its speed, and the
  choice outlived the load that caused it. What a wrong class costs is
  not measured.

  `OccupancyWindow` reads both counters at construction and again at
  `sample`, so they cover the same interval by construction rather than
  by a caller pairing them, and both come from the same clock so their
  ratio is a fraction. On Windows that is `QueryThreadCycleTime` against
  the timestamp counter, because the thread figure counts cycles rather
  than time: divided by
  a nanosecond clock it yields achieved GHz, which on a 4 GHz part reads
  100 percent for anything above a quarter of one core. On Linux both
  sides are nanoseconds through `CLOCK_THREAD_CPUTIME_ID`. The two
  Windows counters advance for different reasons - one per executed
  cycle, one at a fixed rate - so a boosted core reads over 1.0 and is
  clamped. A platform with no thread clock reports full occupancy, so it
  behaves as it did rather than reading every interval as idle.

  What a dispatch reports is its pool's occupancy, not its caller's:
  each worker contributes its own
  on-core and elapsed counts for the leaves it ran, summed per call
  site, and a dispatch takes the difference across itself. Measuring the
  calling thread instead gave the opposite of the intended reading,
  because a dispatch is exactly the interval in which the caller stops
  working and waits for its leaves - the better the spread, the longer
  the wait and the lower the figure. On an idle host the regimes ordered
  backwards from load: the floor-bound one that barely parallelizes read
  100 percent, the one that spreads furthest read 60.

  The counts ride the existing leaf buffer, read once when a site's
  batch opens and once when it flushes, so they cost two clock reads per
  batch of up to four leaves rather than two per leaf. A dispatch whose
  leaves never reached that path reports nothing rather than zero, which
  would read as total contention instead of as no reading.

- `sched::calibration_store` persists measured dispatch costs and device
  capabilities per host in a memory-mapped file, behind the new
  `persisted-calibration` feature, which is on by default and pulls in
  `memmap2`. Without it every process measures its own at first use, and
  that measurement is only as good as the host was quiet - the entry
  below records a 62 percent spread across nine processes on one host,
  two of which installed a threshold about 40 percent below the median
  of their own sweeps.

  A process that finds a table for its own host reads it and measures
  nothing. One that does not takes the writer lease, measures, and
  publishes. The table is keyed by a `HostStamp` - vendor, cpuid
  signature, architecture, operating system, primary and total worker
  counts, layout version - so a different chip, a different core count
  or a different probe set is a different table rather than a stale one.
  `LAYOUT_VERSION` enters the stamp, so raising it makes the next start
  on every host measure again.

  Three things make a read safe. The magic is written last, after every
  other field is in place, so a process attaching to a region another is
  still laying out sees a zero magic and waits rather than reading an
  unwritten record as real. Readers take the record under a SeqLock: the
  writer raises `seq_version` to an odd value before touching the
  payload and to the next even value after, and a reader that observes
  an odd version, or a different version either side of its copy,
  retries, so a record is never half old and half new. A writer that
  dies mid-measurement leaves its pid in the header and stops beating,
  and a later start whose heartbeat has not advanced within
  `LEASE_GRACE_EPOCHS` takes the lease from it - without which one
  killed process would stop every later start on the host from ever
  calibrating.

  A record whose own nine samples spread more than
  `PROVISIONAL_SPREAD_PER_MILLE` of their median, which is 250, is
  marked provisional and a later start on a quieter host overwrites it.
  The calibration already takes nine samples and keeps their median, so
  the spread costs nothing to compute and is the sharpest available
  signal for whether the host was quiet.

  It is scheduler surface rather than a reference backend, which matters
  for a consumer that turns the default set off. `--no-default-features`
  is the way to drop the CUDA, JAX, WASM and GPU-peer backends, and it
  takes this with them unless it is named back:
  `features = ["persisted-calibration"]`. Nothing reports the loss,
  because a process that measures its own calibration behaves exactly
  like one that read a table - it is simply the only process the
  measurement ever serves.

  The directory is `FLYNNEL_CALIBRATION_DIR` when set, otherwise
  `%LOCALAPPDATA%\flynnel\calibration` on Windows and
  `$XDG_CACHE_HOME/flynnel/calibration` or `~/.cache/flynnel/calibration`
  elsewhere. A host with more than `MAX_ACCEL` devices records the first
  eight and leaves the rest unrecorded rather than overflowing.

- `for_each_chunk_min_leaf` takes the recursion floor from the caller,
  as `for_each_chunk_indexed_min_leaf` and `for_each_chunk_triple_min_leaf`
  already did. `for_each_chunk` was the only member of the family with
  the floor fixed at `MIN_LEAF_ITEMS`, so a heavy-per-element op on a
  plain `&mut [T]` had no way to ask for a leaf per item and ran a small
  batch serially. `for_each_chunk` now delegates with the old default,
  so its behavior is unchanged.

- `cooperative_join_n`'s `Auto` arm compares the fan-out against the
  calling thread's own node rather than against every node summed.
  `NumaArena::total_workers` spans all nodes; the two gates inside
  `cooperative_join_n_flat_mailbox` read `ctx.sleep.worker_count()`,
  which counts one node, because each worker's context carries a clone
  of its own arena's sleep coordinator. On a single-node host a sum of
  one term equals that term and the two agreed. On a two-node host with
  24 workers each, `Auto` routed to mailbox at 48 while the gate inside
  that path opened at 24, so a fan-out between the two went to the tree
  shape where the population heuristic intended mailbox.
  `NumaArena::local_worker_count` is the new accessor, resolving through
  the node lookup that already existed. This host is single-node, so the
  band is empty here and the change is a no-op on it.

  Whether mailbox beats deque at the widths where it engages has since
  been measured, and it does not. Against the deque fan-out on the same
  closures, mailbox costs 13 percent more at N=24, 22 at 32, 9.7 at 56
  and 14 at 64 - four widths, one direction. Below the gate the two arms
  run the same code and match to within 1.5 percent, which is the
  control that makes the rest readable.

  So the fan-out a caller gets through `cooperative_join_n` is not the
  fastest one the crate has at those widths, and where rayon has been
  seen to win it is winning against the routed shape rather than against
  the deque fan-out, which beats it at every width measured. Where the
  gate belongs instead waits on the sweep reaching past 64.

- Seed-depth hysteresis is on. A change of seeded leaf count now needs
  two consecutive dispatches asking for it, so one estimate landing the
  far side of a power-of-two boundary cannot halve or double the fan-out
  by itself. `FLYNNEL_SEED_HYSTERESIS=0` turns it off, and
  `set_seed_hysteresis` switches it at run time so both arms of a
  comparison fit in one process; arms measured across separate processes
  carry whatever else differed between them.

  Measured against the defect rather than only against the clock. Over
  one bench group of 32768 items whose estimate alternates across a
  boundary every dispatch, the number of dispatches seeding a different
  leaf count from the one before them:

  | | flips | cost |
  |---|---|---|
  | off | 15013 | - |
  | on | 0 | ~2.3% |

  Zero in both registration orders. The cost appears only in that
  straddling group; with the estimate held still, and with it so far
  under the worker floor that it reaches the decision not at all, the
  arm with hysteresis measured slightly faster than the arm without.
  So it is charged to callers who are crossing a boundary and to nobody
  else.

  An estimate-smoothing arm was measured beside it and is not shipped.
  It was free, and it removed 8 percent of the flips in one order and
  none in the other - not distinguishable from no effect. Reading cost
  alone would have kept it and dropped hysteresis, since one was free
  and the other was not; the flip count is what separates a mechanism
  that works from one that only looks affordable.

- `cooperative_join_n_flat_mailbox` picks its mode and its mailbox
  targets from the worker count. Both read `ctx.stealers.len()`, which
  is the worker count plus `EXTERNAL_SLOT_COUNT`, so the two uses were
  the same binding and moved together: the gate opened at 32 above the
  pool width, and the round-robin at `(ctx.index + 1 + i) % n_workers`
  spread targets across the whole stealer table.

  The first fan-out wide enough to open the gate was therefore also the
  first whose targets reached past the last worker. Indices at or above
  the worker count are external slots, and a slot's mailbox is drained
  only by a thread that has claimed that slot; `find_work` pops a
  thread's own mailbox and peer-steal reads deques, so nothing else
  reaches one. `push_to_mailbox` accepts those indices because the
  mailbox vector is allocated for slots too, and the ring holds 16
  against a per-target depth of 1, so the full-mailbox fallback onto
  the caller's deque never fires. The join's latch counts every
  dispatched closure, so a fan-out at that width does not complete.

  On a 24-worker host the gate opened at 56, where roughly 32 of 55
  dispatched closures were routed to slot mailboxes. A 56-closure
  fan-out sat 39 minutes inside a 2 second warm-up while the same
  fan-out in deque mode completed. No caller reached this: the gate had
  never opened, because `cooperative_join_n` routes on the worker count
  and so never asked for a fan-out wide enough.

  Mailbox mode now engages at N at or above the worker count instead of
  32 above it, so a fan-out in that band takes owner-directed
  distribution where it previously demoted to the shared deque. That
  band has not been measured against deque mode on any host.
  `cooperative_join_n_flat` and its callers are unaffected: the changed
  binding is read only by the gate and by the mailbox arm's target
  computation, and a deque fan-out reaches neither.

- `NumaArena::primary_workers` and `NumaArena::smt_extension_workers`
  sum the counts `LocalArena` already published per node.
  `total_workers` was the only count reachable across nodes and counts
  SMT siblings whether or not they are awake. A consumer sizing to
  physical cores gated a kernel on `total_workers() <= 16`, read 24 on
  a twelve-core host, and never ran that kernel there.

- The collapse and wake crossovers are the median of the calibration's
  nine sweeps rather than the fastest. Nine were already run and eight
  discarded. A crossover is derived from two timings rather than being
  one, so the fastest of several runs is whichever run's cancellation
  noise fell furthest in one direction: biased low, and far wider in
  spread than the median of the same samples. The timing points inside
  a sweep keep their own minimum, where suppressing that noise is what
  it is for.

  Measured across nine processes on a Ryzen 9 7900X, both statistics
  taken from the same samples: the fastest spans 42400 to 68600 ns, a
  62 percent spread; the median spans 68200 to 71000, 4.1 percent. Two
  of the nine installed a threshold about 40 percent below the median
  of their own sweeps. A consumer reported the same 57 percent spread
  from four of its own processes, which is what prompted this.

  The threshold rises, so more work collapses inline, and that is worth
  what it costs. Measured with the two thresholds pinned as the only
  difference between arms, alternating, on a host at 1.12 of 24 cores:
  a cell whose work falls between them runs 86.3 us dispatched against
  9.3 us inline. A cell under both thresholds reads 6.3 and 6.4 across
  the arms and one over both reads 92.7 and 93.3, so the two controls
  make the same decision in both arms and the middle cell is the
  measurement. The eight pre-existing workload cells move between -0.05
  and +0.03 with the sign flipping four up and four down, which is
  noise. So the old statistic's low draws were not merely unstable, they
  dispatched work that cost nine times more dispatched than inline.

  `benches/cold_workloads.rs` gains the three cells this was measured
  with. Every shape it had sat orders above the threshold - the smallest
  is 32 ms of work - so none of them reached the decision at all.

  The spread figures above were taken on a loaded host and are not
  comparable to the 14.4 us in the 0.3.0 entry; only the two spreads,
  computed from one set of samples, are compared.

- `inline_collapse_threshold_ns` and `pool_dispatch_cost_ns` say that
  their figures are per process rather than properties of the machine:
  measured once on first use and cached for that process's life, so a
  figure read in one process does not describe another. The 0.3.0 entry
  below gives a range across six processes for the Ryzen 7 2700 but a
  single 14.4 us figure for the 7900X, which reads as a constant of
  that chip and was one draw. A consumer compared a threshold printed
  in one process against behavior in another and reported a
  contradiction in the collapse control flow; there was none.

### GPU peer

- A peer start reuses the host's stored record for its device instead of
  re-measuring the round trip, when there is one and it was taken on a
  device nothing else was resident on. The round trip, the clock error
  and the launch baseline are properties of a host, device and driver
  combination and are taken as they stand.

  The margin and the atomics flag are not taken. They are re-established
  on every start whichever path it takes, because what they assert is
  how this device behaved under contention, and only a run on it can say
  that. A stored record therefore shortens a start without ever standing
  in for a safety property.

  Whether the stored record is usable is decided the same way the CPU
  half is. The peer already times the round trip at three points, so how
  far apart they fell is the device's equivalent of the CPU sweep's
  sample spread and costs nothing extra: another process resident on the
  device stretches the tail without moving the minimum, which is what
  that spread reads. The capability fields are read from the driver and
  do not vary with what else is running, so they are stored beside the
  timings and reused unconditionally.

  Every way of failing to reach the table ends in a full measurement,
  which is what a start did before there was a table. What it does not
  do is fail quietly: a table that cannot be read and a device that has
  never been measured produce the same calibration, and a diagnostic on
  stderr is the only thing separating them.

  A device publish carries the host's CPU half through untouched rather
  than writing a default beside it. That path measured a device, not a
  host, and a default record reads as a completed measurement of one
  sample with zero spread, which is the shape a reader trusts most.

### Tests

- `benches/seed_depth_stability.rs` measures what seed-depth hysteresis
  and estimate smoothing cost against each other and against neither, in
  three regimes: an estimate that alternates across a power-of-two
  boundary every dispatch, one held still, and one so far under the
  worker floor that the estimate reaches the decision not at all. The
  two cost regimes are what decide whether either mechanism is
  shippable, because a cost there is paid by every caller including
  those that can never cross a boundary.

  Every regime is registered twice, in opposite arm order. Criterion
  runs arms sequentially, so a load arriving during a group lands on
  some arms and not others, and an effect present in one order and
  absent in the other is the box while one whose ratio survives both is
  the code. On the first run this separated the groups rather than the
  arms: the four arms of the straddle group, which ran first, agree
  across both orders to within 3 percent, while the two groups that ran
  after a compile storm arrived disagree between orders by 61 and 68
  percent, which is unreadable rather than noisy. Each arm owns a
  distinct `CallSiteState`, so the depth one settles on never becomes
  another's starting condition, and each prints its site's occupancy and
  suppressed-migration count so a degraded run marks itself.

- `FLYNNEL_SEED_DEPTH` prints, per distinct workload shape, the leaf
  count the cost model asked for beside the power of two it rounded to.
  The two differ by up to a factor of two and only the second is
  dispatched, so a caller reading its own fan-out had no way to see
  which of the two it got. Deduplicated on the pair the result depends
  on, so a sweep prints once per size rather than once per dispatch.

  It also reports the recursion floor beside the seeded count, as a
  `leaf floor:` line carrying the caller's floor and the effective one.
  They are two separate decisions taken from two separate inputs, and
  only the second was observable: with an authoritative per-item
  estimate the floor is `pool_dispatch_cost_ns()` divided by that
  estimate, and that numerator is measured once per process. On the
  bench host the sweep reports a caller floor of 256 against an
  effective floor of 92, so the figure a caller gets is neither the
  default nor a constant of the machine, and it could move under them
  with every printed number staying still.

  Both reports read the environment once rather than per dispatch.
  `env::var_os` takes the process environment lock and allocates on
  each call, and the seed-depth report sat on a per-dispatch path
  paying that to answer a question whose answer cannot change.

- `CallSiteState::seed_depth_flips` counts dispatches at a site that
  seeded a different leaf count from the one before them. Throughput
  cannot express that: a flip between two adjacent depths costs little
  either way, so an arm can be fast and unstable or slow and steady and
  the timings rank them the same. It is what the seed-depth mechanisms
  were finally measured against, and it reversed which of them shipped.

- `examples/trace_mailbox_hang` runs a mailbox fan-out at a chosen width
  with the caller's join push and wait recorded and every closure
  emitting its own index on entry and exit, so a fan-out that does not
  complete says which closures never ran rather than only that it
  stopped. A worker that never dumps its buffer never woke, since a
  parked worker flushes only when woken. It takes a repeat count,
  because a single call is the one thing a standalone reproduction does
  that a criterion sweep does not.

- `benches/seed_depth_smoothness.rs` measures what an input size costs
  for crossing a fan-out boundary, at sizes close enough together that
  only the leaf count differs. The leaf count is rounded up to a power
  of two as the last step, so it is a step function of the input size:
  two calls whose sizes differ by one percent can seed 32 leaves and 64.
  Nothing is noisy and nothing flips - a consumer sweeping sizes reads
  the discontinuity as its own kernel changing behavior.

  It does not hunt for the boundary. The division is integer and the
  floor is the pool width, so the count is 32 until the quotient reaches
  33 - items at or above `33e6 / est`, which is 660_000 at 50 ns an item
  on 24 workers. Six sizes bracket that with the closest pair five
  thousand apart, and the per-item work is tuned to the same 50 ns the
  plan is told. On a host of another width the boundary moves and the
  reported leaf count per size says where it fell.

  What it settles is whether the step clears the within-size spread. A
  step buried under that spread is a fact about the code with no
  consequence, which is a result rather than an absence of one.

  Measured, 24 workers at 50 ns an item, every size run forward and
  reversed: crossing 655_000 to 665_000 the larger input is faster per
  element, by 2.6 percent in one order and 6.8 in the other. Both orders
  agree on the direction, which is the counter-intuitive one - the
  sawtooth costs the smaller input, so a consumer's throughput curve
  gets faster as `n` grows past a boundary. The reason is load balance:
  32 leaves on 24 workers leaves eight workers holding two and a tail
  about twice one leaf, where 64 leaves is 2.67 each and averages the
  imbalance out.

  It does not clear the spread. The same size measured in the two orders
  differed by 3.5 percent at 655_000, so the step is present and is not
  a number a consumer can rely on measuring. It is documented beside
  `adaptive_seed_depth` and where the fan-out helpers are described, as
  the reason a reader's curve has a step in it rather than as a figure
  to depend on. Nothing in the fan-out changed: interpolating between
  boundaries or seeding a ragged split would give up the power-of-two
  bisect for an effect this size.

- `examples/pair_orders.py` pairs each arm's forward reading against its
  reversed one across a whole log. It compares an arm's movement against
  its GROUP's median movement rather than against a threshold: the raw
  cross-order difference is unpaired, carrying whatever the machine did
  between two readings at opposite ends of a group, while criterion's
  interval is a within-arm estimate - so testing one against the other
  rejects arms that merely drifted with everything around them.

  It refuses rather than under-reports. A timing line it cannot
  attribute ends the run, having once dropped sixteen arms while
  reporting twenty as though that were all of them; an arm present in
  only one order is named; an unrecognized time unit is an error rather
  than a guess; and the per-arm figures are read as key-value pairs
  rather than by position, so a field added or removed leaves the rest
  readable instead of reporting the column as absent.

- `benches/simc_cooperative.rs` registers every size twice, in opposite
  arm order. Criterion measures arms sequentially, so a load arriving
  partway through a group moves the arms it covers and not the others,
  and a confidence interval does not catch it: an interval describes
  spread within an arm, so two arms can hold tight disjoint intervals
  and still be separated by the machine.

  That is not hypothetical for this sweep. Its founding figures and a
  re-run at the same commit disagree by 53 percent at N=16 and by a
  factor of five at N=8, and the re-run's rayon numbers are incoherent
  on their own terms - 73 us at N=8 against 21.5 us at N=16, slower
  with less work. The original argument that rayon beat
  `cooperative_join_n_flat` by 16 percent rested on non-overlapping
  intervals, which is the evidence this defeats, so those numbers are
  withdrawn rather than re-argued.

  A ratio surviving both orders is the code; one present in one order
  and absent or reversed in the other is the host, and the group is
  unreadable rather than noisy. The four shapes are now an enum behind
  one dispatch function, so both orders run identical code and differ
  only in registration sequence. The sweep doubles to roughly six and a
  half minutes, which buys telling a contaminated run from a clean one
  without a second quiet window to compare against.

- `FLYNNEL_BENCH_STALL_REPORT=1` makes every closure in
  `benches/simc_cooperative.rs` record that it started and that it
  finished, and arms a watchdog that prints what a dispatch reached
  once it stops advancing. It is off by default because it puts two
  atomic stores and one lock acquisition in each closure, which moves
  the numbers that bench exists to take: a run with it on diagnoses and
  does not measure.

  It answers a stall with sets rather than an event log - the indices
  that never started, the ones that started and never finished, and the
  threads that ran anything. Those are bounded by the fan-out width
  whatever the iteration count, where an event log is bounded by
  iterations times width, which at this bench's counts is tens of
  millions of records across forty-nine threads.

  The three readings are distinct. Indices that never started with
  workers missing from the thread list is work nothing was woken to
  take; indices that never started with every worker present is work
  routed where none of them looks; started-and-unfinished is a closure
  still running or a thread that died inside one.

  It is what located the deque fan-out hang recorded under Fixed. The
  stalled dispatch reported all 1024 closures unstarted, every worker
  present from earlier dispatches and none from this one, and nothing
  started-and-unfinished. The last closure runs inline on the calling
  thread before the wait loop, so its never having started placed the
  hang in the push loop and ruled out the wait loop. The stall rates
  measured before it could say how often the hang happened but never
  where it was.

- The GPU test binaries take one cross-process lock on the device.
  Each file declared its own `static Mutex`, which serializes the tests
  inside that binary and nothing else, and cargo runs the binaries
  concurrently; two of them had no lock at all. The 64-block barrier
  wait in `gpu_peer_team` printed 51264, 53440 and 412416 ns across
  three runs on one host, an eight times swing decided by which other
  binary was resident, and an assertion bounding it was reading its
  neighbor. Serialized, the wait reads 55008 ns. The lock is a file,
  since a `Mutex` does not reach across processes, and it is advisory:
  a process that does not ask still gets the device. A lock left behind
  by a killed process is reclaimed after 120 seconds.

## 0.4.0 - 2026-09-07

### Breaking

- `GpuPeer::read_result` returns `Result<(), GpuPeerError>` and refuses
  a `dst` longer than the slot's payload with `PayloadTooLarge`, naming
  both the length and the capacity. It previously copied `dst.len()`
  bytes with no check, so a buffer longer than the payload read across
  the slot boundary into the next slot's descriptor, and past the last
  slot outside the mapping. The only guard was a `debug_assert` against
  the whole region, which is the wrong bound and is compiled out in
  release. Every submit path already refused an oversized buffer the
  same way; the read path now matches them. Callers testing the result
  need a `?` or an `expect`.

### GPU peer

- `STATUS_TEAM_INCOMPLETE` is set when a block team does not fully
  arrive before rank 0's deadline. It was `STATUS_ERR`, which is also
  what a user op returning non-zero sets, so a caller learned a slot
  was refused and nothing more. A failing op still outranks an
  incomplete team. Additive: a caller testing `!= STATUS_DONE` is
  unaffected.

- `GpuPeerConfig::barrier_deadline_ns` sets how long rank 0 waits for
  its team, defaulting to 5 ms. It was the poller quantum, 250 ms. On
  an RTX 5070 with a Ryzen 9 7900X, over thirty-two submissions per
  size, a team that assembles costs 1.7 us at two blocks, 5.2 us at
  four, 13.7 us at eight and 53.4 us at sixty-four, the widest team any
  consumer here configures; a loaded host roughly doubles the small
  sizes. The margin at 64 is the one that matters, because that setting
  was chosen while the 250 ms quantum bounded the wait, and it is 94
  times the measured cost. Only consulted when `blocks_per_lane > 1`.

- `GpuPeer::barrier_stalls` reports how many teams missed that deadline
  and the largest ring depth at one; `GpuPeer::barrier_wait_max_ns`
  reports the longest wait on a slot the whole team reached. Depth
  separates a boundary landing on a lightly fed lane, which shows one
  or two, from a lane draining a backlog, which shows a depth near the
  ring size. A consumer measuring 12500 queries per arm saw three and
  six expiries leave no mark on either signal it had: the 99th
  percentile there is the 125th slowest query, and recall moved 0.0005
  across arms counting 0, 3 and 6.

- `GpuPeer::displaced_foreign_context` reports whether `init` replaced a
  CUDA context that was current on the calling thread. The peer
  operates on the device primary context and makes it current where it
  binds, and a consumer holding its own context sees no error when that
  happens: a device pointer from the displaced context still reads
  back correctly through `cuMemcpyDtoH`. The displacement is also
  written to stderr.

- `blocks_per_lane > 1` has tests. The barrier, the rank-0 retirement
  and the follower generation wait had none, and a consumer runs teams
  in production.

### Scheduler

- The host profile's dispatch cost is observed rather than
  differenced. It was a dispatched timing minus a serial one at the
  same size, which recovers a quantity smaller than either operand, so
  the noise on the dispatched side exceeded the thing being measured.
  Across twelve draws on an idle Ryzen 7 2700 it read 1, 400 and 10200
  ns among values clustered near 2000, with one draw at 250400; the 1
  was the floor engaging on a difference that had reached zero. It is
  now the median of a thousand timings of one join whose halves are a
  single item each, so the dispatch is what is timed. The same twelve
  draws now span 2100 to 5500 ns and the floor never engages. The
  workload cells are unchanged: across four alternating runs per arm,
  every cell differs by less than its own repeat spread. This value
  sizes leaves through `adaptive_min_leaf`; the collapse threshold that
  decides routing was already stable and is untouched.

- A leaf shape the caller names outranks an estimate the caller did
  not. `JobPlan::new` seeds `estimated_per_item_ns` from the
  process-active profile and marks it non-explicit; the shape
  classifier's fine-grain guard read that estimate without asking
  whether the caller supplied it, so below roughly 4200 items an
  explicit `with_leaf_shape` was classified fine-grain and discarded.
  The plan still reported carrying the shape, so nothing a caller could
  inspect showed the drop. `with_estimated_per_item_ns` is unchanged: a
  figure the caller passes still governs. A consumer dispatching 32 to
  a few hundred items with `LatencyCompute` and no estimate was routed
  as though no shape had been given at all five of its sites, on work
  measured at 38 to 48 percent device wait.

- `with_cost_ns_per_elem` and `with_estimated_per_item_ns` are the same
  call. The first stored the figure and returned; the second stored it
  and re-ran the static classifier, so `use_smt`,
  `oversubscription_log2`, `use_mailbox_routing` and `deque_tier_hint`
  followed from the caller's cost. The README and the JobPlan reference
  both say the two are equivalent and that the first is the canonical
  name, so a caller taking the documented advice set a field and
  changed no routing. At every size tested the two produced different
  plans from identical input.

- A profile the caller named survives a cost hint given after it. That
  classifier pass overwrote `use_smt` and the rest whether or not the
  profile had been set explicitly, so `set_profile(MemoryBound)`
  followed by a cost hint routed as whatever the classifier inferred
  while the plan still reported `MemoryBound`. The builder's own
  documentation says the hint overrides the size-only guess `new` made;
  a named profile is not that guess. Both remain in force now: the
  estimate lands, the profile stands.

- The host calibration no longer subtracts past zero. Its crossover
  interpolation took the span between two timed counts as
  `a - a_prev` over `u64`, guarded only by the gap having narrowed and
  never by the larger count having cost more. A contended host times
  2n faster than n often enough to matter: that panics in debug, and
  in release it wraps, producing a collapse threshold that then governs
  every dispatch for the life of the process. Reachable from
  `calibrate_inline_collapse_threshold`, so a busy machine at process
  start is the whole condition.

- `calibrate_host_dispatch`'s documentation says that installing the
  three figures into the process globals is the effect of calling it,
  not of using what it returns. A call site that binds the result and
  only prints it has still armed the whole program.

- `FLYNNEL_HOST_PROFILE_NS=<dispatch>,<collapse>,<wake>` pins the host
  dispatch profile to three nanosecond counts and skips the
  calibration. It exists for comparing this scheduler against one that
  never measures its own dispatch cost: unpinned, the two arms differ
  in whether they adapt as well as in how they schedule, and a ratio
  between them mixes the two. All three fields are required, zero is
  refused in any of them because an installed collapse threshold of
  zero is what marks a profile as uncalibrated, and anything that does
  not parse is reported on stderr and the host measured instead. Unset,
  which is the default, nothing changes.

## 0.3.0 - 2026-09-06

### Scheduler
- The dispatch constants a call meets are measured on the running
  host, not set by hand. `par_iter::host_dispatch_profile` measures,
  once per process at first use (13 to 20 ms on the Ryzen 7 2700) or
  earlier by `par_iter::calibrate_host_dispatch`, the pool's dispatch
  cost, the collapse crossover (floored at that cost) and the
  polling-versus-wake crossover, from a compute-bound body serial
  against dispatched, sixteen warm-up dispatches then five sweeps
  each. Readers: the inline collapse in every walker and the tier
  pick's heavy-item override (the 50 us floor and 800 us cap of 0.2.3
  and the tier pick's own 50 us constant are gone), leaf sizing from
  an explicit or probed per-item cost (one dispatch cost of work per
  leaf, in place of a 5 us target), the probe's decision to dispatch
  its tail (in place of workers times 5 us times a small-host
  factor), and the switch between polling and the sleep-counter wake
  path (in place of 200 us). `INLINE_COLLAPSE_FLOOR_NS` and
  `INLINE_COLLAPSE_CAP_NS` are removed; `inline_collapse_threshold_ns`
  and `calibrate_inline_collapse_threshold` remain and read the
  profile. Measured on the Ryzen 7 2700 across six processes:
  dispatch cost 4.9 to 12.7 us, collapse 16.0 to 30.2 us, wake 14.0
  to 19.3 us; on the Ryzen 9 7900X the collapse threshold measures
  14.4 us, so a 29 us slice add there dispatches (the 50 us floor of
  0.2.3 had collapsed it to 27 us where the pool finishes in 11). The
  256-item leaf floor for hint-less work and the probe sizes keep
  their bench-sweep values.
- An idle worker probes the held external slots' deques every round
  and draws its random victims from the workers alone. The stealer
  table holds the workers and the thirty-two external slots, and a
  random pick over all of them reached a caller's wrapped join once
  in twelve rounds on a 16-worker pool: traced on the Ryzen 7 2700,
  a 10k-item light call's job started 4.1 us after the push and its
  helpers arrived over the following 22 us; with the probe, 0.6 us
  and 3.4 us.
- A worker waiting on a stolen join half polls the latch in a spin
  before it yields, where it yielded every round before, which cost
  one to three microseconds per level of an eight-deep steal chain
  (10 us of drain after the last leaf on the traced call, 2 us now).
  An external caller waiting on its slot job spins on the same
  budget before parking, over a fixed 256-spin floor that absorbs a
  fast-completion race.
- `JobPlan::spin_before_yield_ns` and its builder
  `with_spin_before_yield_ns` set that budget, and
  `effective_spin_before_yield_ns` resolves it: the caller's value
  when set, else this host's measured collapse threshold, dropping
  to zero when the plan's per-item cost is at least that threshold.
  A waiter spins because the half it waits on is about to finish;
  when one item outlasts the whole spin that premise is false and
  the spin only denies a core to the thief being waited on. The
  decision reads the per-item cost rather than `use_smt`, which is a
  claim about execution ports and says nothing about how long to
  poll: a caller who sets `with_smt` for a memory-stall reason keeps
  the spin its item cost earns. `par_iter::measured_collapse_threshold_ns`
  is public and reports `None` until the host profile is measured.
- Gate-shaped cells on a Ryzen 9 7900X, built there with
  `-C target-cpu=native` and measured on an idle host, medians of
  five interleaved runs of 1000 calls each with the busy-core count
  sampled before every round (median 0.07 of 24). A light body at
  10k items 13.5 to 11.1 us, which is 3.70x to 4.50x over serial. A
  body of about 80 ns per item through
  `for_each_chunk_triple_min_leaf` at 1k 10.4 to 9.0 us and at 10k
  44.5 to 40.5 us. Two cells are flat: the light body at 100k, 56.9
  to 56.8 us, and a heavy body at 100k, 633.6 to 633.9 us.
- The cold-dispatch bench on a Ryzen 7 2700, from a binary
  cross-built on the other host so the measuring machine stayed
  idle, medians of seven interleaved rounds, as a ratio against
  rayon where lower is better. The one movement outside its own
  noise is 16384 items of 10us, 1.14 to 1.04. Every other shape sits
  inside the spread described next, the heavy ones flat at 1.00.
- What "inside the spread" means here, because it decides which of
  the figures above are claims. The same base binary measured in two
  separate sessions disagrees with itself by up to 13 points at 128
  items of 500us and 6 points at 1024 items of 100us, against 2
  points at 16384 items of 10us. A cell's own repeat spread is the
  floor a difference has to clear, and this bench does not resolve a
  few points either way at its mid sizes. Earlier sessions reported
  movement at those cells in both directions, which was the spread
  rather than the code.
- Probing every worker peer per round instead of four was measured
  and is not adopted: the light cells gained within noise and the
  heavy cell lost the same.
- A body that the inline collapse ran on the calling thread, and that
  then took longer than the collapse threshold which admitted it,
  latches its call site. Later calls there dispatch whatever the
  caller's per-item estimate says. The collapse trusts that estimate,
  and one low enough to admit work that overruns costs the difference
  between running inline and running on the pool. Measured on an idle
  Ryzen 9 7900X by sweeping a deliberately scaled estimate against a
  fixed workload: an 80 ns-per-item body at 1000 items took 43.8 us
  with its estimate set to an eighth of the truth, against 9.0 us at
  every factor from a quarter to eight times; after the change the
  same point reads 9.8 us against 9.3 to 9.6 across the rest. An
  estimate eight times too high still costs a light body at 10k items
  about 47 percent through leaf over-splitting, which is monotone in
  the error rather than a cliff.
- The host dispatch profile takes the fastest of nine timings at every
  point and every crossover sweep, where it took a median of five.
  Interference only adds time to a timing measurement, so the fastest
  run reads the cost itself and a median reads whatever else the host
  was doing. Measured across 60 processes on an idle Ryzen 9 7900X:
  the dispatch cost, which divides into leaf sizing, narrows from a
  900 to 3500 ns band to 700 to 1700, standard deviation 551 to 198;
  the wake threshold from 2700 to 6203 down to 1300 to 2100, deviation
  992 to 217. The collapse threshold's band narrows from 14564 to
  24030 down to 6957 to 14165, deviation 1888 to 1193, with its
  coefficient of variation unchanged at 0.10: that value is
  interpolated from a doubling sweep, so its precision is bounded by
  the sweep's granularity rather than by sample noise.
- A call site's averaged costs (policy arms, hybrid placement, and
  the tandem split's per-item cost on each side) weight their first
  eight samples equally before the update turns exponential at
  1/8. The first sample of a site is a cold one, and seeded straight
  into the exponential average it held half the weight into the
  sixth sample: on the gemm tandem parity test the CPU share had
  reached 474 to 572 per mille after six rounds against a device 2.2
  times slower per item, so the run-to-run spread crossed the 500
  the test asserts.
- Three documented per-call overrides now reach the dispatch
  entries, where they were previously ignored. The leaf-count
  oversubscription factor was read nowhere: every entry used the
  process-global split observer's multiplier, so
  `with_oversubscription_log2` changed nothing.
  `effective_leaves_per_worker` resolves it, and a factor the caller
  set skips the observer entirely, while a profile-derived or
  class-derived one still defers to it; the new
  `oversubscription_log2_explicit` field tells the two apart. The
  worker cap reached only `for_each_chunk`, and
  `effective_workers` now applies it at the triple, indexed,
  collect, token-bucket and reduce entries as well. A cap of one,
  documented as forcing serial execution on the calling thread,
  dispatched into the pool at every entry; `runs_on_caller` resolves
  it from the plan alone, so a capped call touches neither the pool
  nor the host profile.
- `examples/trace_dispatch` traces one dispatch of a chosen shape
  with `FLYNNEL_TRACE=1`, and the trace records the slot push, the
  caller's wait end, and the wrapped join's start and end on the
  worker; `trace::worker_flushes_done` counts completed worker dumps
  so a tracer can wait for them before exit.

- `FLYNNEL_PROFILE_SAMPLES=1` prints every sample behind each point of
  the host dispatch calibration, so the spread can be read off one run
  rather than inferred from repeated ones. The estimator is unchanged
  and still takes the fastest of nine, but that is now documented as
  the biased-low choice it is: on a Ryzen 7 2700 a dispatched point
  read 6200 ns against a 9600 ns median of the same nine samples,
  while serial points repeat to within a percent. The bias is kept
  because both corrections measured worse. Taking the median of each
  point separately gave dispatch costs from 300 to 6700 ns over six
  calibrations of an idle host, since it differences two numbers of
  comparable size never observed together; pairing the samples and
  differencing within each iteration held a spread comparable to the
  current one but drew 38700 ns on one calibration, which raised the
  collapse threshold and took the light 10k cell from 25.8 to 49.4 us.
  Under-estimating dispatch spends overhead on work that did not need
  the pool; over-estimating it forfeits the parallelism outright.

### GPU peer

- An opcode the kernel does not recognize is marked failed instead of
  completing. The dispatch chain had no branch for one, so `op` kept
  its submitted value and the slot was written `STATUS_DONE`; the
  comment claiming otherwise had never been true. A caller who mistyped
  an opcode, or submitted a user op before its source was composed in,
  received success and read the payload it had submitted back as a
  result. `GpuPeer::wait` could not protect against it, because the
  slot never carried a failed status to convert into an error. Measured
  against 0.3.0 on an RTX 5070: opcode 42, past the last built-in and
  below `OP_USER_BASE`, returned `Ok(STATUS_DONE)` from both `wait` and
  `wait_status`; both now report the failure.

### Tests

- `tests/gpu_peer_status_contract.rs` exercises the failed-slot
  contract against a slot that genuinely fails on the device rather
  than a constructed status: `wait` reports an error, `wait_status`
  hands back the raw failed word, and an `OP_NOP` control still
  completes, so the test can tell the two apart.
- `for_each_chunk_small_input_runs_serial` supplies an explicit
  per-item cost, which is what the inline collapse gates on, so the
  routing it asserts follows the plan rather than a live probe of the
  body. Without one it failed on a loaded host, and identically on a
  tree predating the change it was blamed on.
  `for_each_chunk_small_input_without_an_estimate` keeps the probed
  path and asserts what holds there whichever way it routes: every
  item processed exactly once.

### GPU peer

- A lane's poller launches, drains and counts independently of the
  others. Each lane carries its own launch count, exit counter
  (`HDR_LANE_EXITS_OFF`) and generation word (`HDR_LANE_GEN_OFF`), and
  runs as its own grid on its own stream; before, one stream carried
  every lane and no lane relaunched until every block of the previous
  launch had exited. A submit on a lane whose block had parked
  therefore waited for whichever block was still working, for as long
  as that block's work lasted. On an RTX 3070 with one lane held by a
  device-side spin and another timed after a 10 ms gap, the timed
  lane's p50 was 139.918 ms against a 150 ms feeder and 389.871 ms
  against a 400 ms feeder, the second past the 250 ms quantum; both now
  read 0.140 ms and 0.122 ms against a control of 0.24 ms, with no
  separation at 1, 2 or 4 blocks per lane. A lane still holds one team
  at a time, which is what keeps a second team from reading a slot the
  first has not retired: the ring tail advances only at retirement, so
  the generation word cannot prevent that read once a block is inside
  an op. `examples/gpu_peer_lane_stall` is the measuring arm.
- `GpuPeer::init` rejects a lane count above `MAX_POLLER_LANES` (64),
  which is what the per-lane header words address.
- Both sides of the team barrier are bounded by the quantum: rank 0
  stops waiting for a rank that never arrives and marks the slot
  `STATUS_ERR`, and a follower stops waiting for a retirement that will
  not be published.
- `GpuPeer::wait` returns `Err(Unavailable)` for a slot that completed
  with a failed status, so a caller testing only for an error cannot
  read an unfilled payload as an answer. `GpuPeer::wait_status` returns
  the raw status word instead.
- `GpuPeer::submit_user_on_lane` places a user op on a caller-chosen
  lane, for diagnostics that need a particular lane warm.

## 0.2.3 - 2026-09-05

### Scheduler
- `for_each_chunk_triple`, `for_each_chunk_triple_min_leaf`,
  `for_each_chunk_indexed` and `for_each_chunk_indexed_min_leaf` (and
  so `for_each_indexed` and `for_each_chunk_ref`) run the body once on
  the calling thread when the caller's explicit per-item estimate
  puts the total under the collapse threshold, as `for_each_chunk`
  already did. A 1000-item slice add estimated at 1 ns per item:
  13.7 us to 0.5 us on the 2700, 5.1 us to 0.3 us on the 7900X (serial
  0.4 and 0.1); 10000 items: 37.7 to 4.4 us and 14.2 to 1.5 us.
- The collapse threshold is measured per host.
  `par_iter::inline_collapse_threshold_ns` answers the floor of 50 us
  until `par_iter::calibrate_inline_collapse_threshold` has run; the
  first query starts that calibration on a thread of its own, and a
  process may run it at start-up. The calibration times a
  compute-bound body over doubling item counts, serial against
  dispatched, and takes the interpolated crossover, three sweeps,
  median, clamped to 50..=800 us. Both bench hosts measure the floor.
  `INLINE_COLLAPSE_FLOOR_NS` and `INLINE_COLLAPSE_CAP_NS` are public.

### Tests
- The core-pair ping-pong test takes the best of five measurements.

## 0.2.2 - 2026-09-05

Everything in 0.2.1 (yanked) plus:

### Benches
- `gpu_linalg` measures every call of a tandem cell from one call
  site: the tandem helpers learn their share per calling source
  location, so the earlier tables had timed cells at an unlearned
  split. Re-measured tables on both hosts; per-side times balance
  (n = 64 eigen on the 3070: 181 ms CPU side against 189 ms device).

### Tests
- The LOH stress test's final flush loops until it lands; a full ring
  had left entries in the LIFO and the thieves spinning (one run in
  five).
- Clippy clean on all targets.

## 0.2.1 - 2026-09-04 (yanked)

Yanked the same day: it shipped while the tandem bench defect above
was open. All of it is in 0.2.2.

### Scheduler
- `for_each_indexed(plan, n, min_leaf, f)`: `f(i)` for every index
  once, over a zero-sized slice, so the probe, per-site statistics
  and lazy-steal bisect apply unchanged. `for_each_chunk_ref(plan,
  items, min_leaf, f)`: the read-only chunk walk at a fixed width.
- `CancelToken::new`, `cancel` and `Default`, for a race the caller
  composes on `join` or the walkers.
- `hybrid_auto_split_ranges(plan, n, cpu, backend)`: the learned
  CPU/backend split by index range; the share is learned per call
  site and per log2 batch bucket
  (`CallSiteState::split_cpu_share_per_mille_for`,
  `record_split_for`), so the backend side may work on resident data.
- The `for_each_chunk` probe confirms a single trusted first-item
  reading with up to three more items (minimum taken): a cold first
  call at a site measured 30 to 79 us for a one-add item and had sent
  a light batch to the pool at a one-item leaf.
- All four walkers and the token are re-exported at the crate root.

### GPU peer
- Batched LU with partial pivoting (`kernels/linalg_lu_f64.cu`):
  `launch_getrf`, `launch_getrs` (with an identity flag for the
  inverse), `getrf_batched`, `getrs_batched`, `getri_batched`,
  `lu_det_batched`; bit-exact with `cpu::getrf_batched` at n <= 64;
  the device leads the pool by 1.3x to 31x from n = 16.
- `gemm_tandem_batched`, `syev_tandem_batched`,
  `gesvd_tandem_batched`: the batch split between the device and the
  CPU pool by the call site's learned share, eigenvalues ascending
  and singular values descending for every item whichever side
  computed it (`sort_eigenpairs_ascending`,
  `sort_singular_descending`).
- `group_by_shape`, `gather_items`, `scatter_items` for ragged inputs.
- The Frobenius inner product `"ij,ij->"` is covered by the einsum
  parity tests.

### Tests
- The profile migration tests hold a lock shared with the tests that
  read the global profile; the inline-join order test builds an
  inline plan explicitly.

## 0.2.0 - 2026-09-02

### Scheduler
- The KHL ring's owner pops newest (Chase-Lev discipline) and thieves
  take oldest; noop dispatch of 10000 items: 41 to 53 us against 430
  to 660 us with the owner popping oldest, rayon 100 to 160 us.
- A push into a full deque is refused and the fork runs inline
  instead of blocking; `WorkerStats::push_refusals` counts them.
  This removed an intermittent `collect_indexed` hang at 65536 items
  with a one-item leaf.
- `Sleep::debug_state`, `LocalArena::debug_snapshot` and
  `NumaArena::debug_snapshot` for hang diagnosis.

### GPU peer
- House-owned f64 linear algebra over resident VRAM blocks, driver-JIT
  PTX, no vendor library: einsum, batched GEMM, Jacobi symmetric
  eigen and one-sided Jacobi SVD, each with a CPU reference and an
  `accel_op` registration; the Jacobi kernel shape (block per matrix
  or thread per matrix) is picked from measurement
  (`JACOBI_THREAD_SHAPE_BATCH_PER_N`).
- Symmetric eigen and SVD by Householder reduction and bisection with
  inverse iteration for n >= 32 (`syev_bisect_batched`,
  `gesvd_bisect_batched`); `syev_auto_batched` and
  `gesvd_auto_batched` route by the measured rule
  (`SYEV_BISECT_MIN_N = 32`, `GESVD_BISECT_MIN_N = 64`).
- Ozaki-scheme f64 GEMM on the int8 tensor cores (`gpu_peer::ozaki`),
  explicit only, held to its stated error bound.
- `pin_bulk` allocates contiguous spans and unpins whole spans.

### Benches and docs
- `gpu_linalg` bench with section selection
  (`FLYNNEL_BENCH_SECTIONS`), a GPU clock ramp before every timing,
  and cells over the pool's capacity skipped; measured tables for
  every op on both hosts in the wiki.
- Every repository URL points at `Variably-Constant/Flynnel`.

## 0.1.0 - 2026-09-02

First published crate: the K-aware, NUMA-aware work-stealing
scheduler with extended-Flynn-taxonomy dispatch (`join`,
`for_each_chunk`, `cooperative_join_n`, `join_hybrid`,
`hybrid_pipeline`, the racing family, `k_join`), per-call `JobPlan`
with dispatch profiles and call-site learning, the backend registry
with CUDA, TPU-JAX, WebAssembly and shared-memory reference backends,
and the GPU-peer substrate (memory-mapped lanes, doorbell dispatch,
resident VRAM blocks, wide kernels).
