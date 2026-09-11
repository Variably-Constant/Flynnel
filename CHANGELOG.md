# Changelog

All notable changes to `flynnel`, by version. Numbers are
measurements from `benches/` and `tests/` on the two bench hosts, an
RTX 3070 with a Ryzen 7 2700 (16 threads) and an RTX 5070 with a
Ryzen 9 7900X (24 threads); the wiki carries the full tables.

## Unreleased

### Added

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
