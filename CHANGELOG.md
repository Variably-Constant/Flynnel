# Changelog

All notable changes to `flynnel`, by version. Numbers are
measurements from `benches/` and `tests/` on the two bench hosts, an
RTX 3070 with a Ryzen 7 2700 (16 threads) and an RTX 5070 with a
Ryzen 9 7900X (24 threads); the wiki carries the full tables.

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
