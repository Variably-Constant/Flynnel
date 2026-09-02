//! End-to-end demo of the AdaptiveDispatcher unified API.
//!
//! Exercises the four Flynn-axis entry points in sequence:
//! - Streaming (SISD) - inline execution
//! - ProducerFast (SIMC) - cooperative fan-out with K_inner=3 burst
//! - WorkSteal (MIMD) - recursive bisection over a slice
//! - Cooperative with mailbox (SIMC/MIMC) - per-worker mailbox routing
//!
//! Then demonstrates the runtime K_gating migration: flips every
//! worker's active backing from PerSlot (KHL) to CounterOnly (Fcl)
//! and back, with subsequent dispatches landing on each backing.
//!
//! Run with:
//! ```sh
//! cargo run --example adaptive_dispatcher_demo --release
//! ```

use flynnel::backend::Backend;
use flynnel::sched::adaptive_profile::WorkloadClass;
use flynnel::sched::dispatch::AdaptiveDispatcher;
use flynnel::sched::k_gating::KGating;
use flynnel::sched::workload_shape::WorkloadShape;

fn main() {
    println!("=== Flynnel AdaptiveDispatcher unified-API demo ===\n");

    // 1. Streaming (SISD): inline.
    println!("[1] WorkloadShape::Streaming (SISD) - inline execution");
    let r = AdaptiveDispatcher::new()
        .with_shape(WorkloadShape::Streaming)
        .execute_streaming(|| {
            let mut acc = 0u64;
            for i in 0..10_000u64 {
                acc = acc.wrapping_add(i);
            }
            acc
        });
    println!("    result: {r} (expected 49995000)\n");
    assert_eq!(r, 49995000);

    // 2. ProducerFast (SIMC): fan-out + K_inner=3 burst.
    println!("[2] WorkloadShape::ProducerFast{{burst:32}} (SIMC) - cooperative fan-out");
    let closures: Vec<Box<dyn FnOnce() -> u64 + Send>> = (0..32u64)
        .map(|i| Box::new(move || i * i) as _)
        .collect();
    let t0 = std::time::Instant::now();
    let results = AdaptiveDispatcher::new()
        .with_shape(WorkloadShape::ProducerFast { burst: 32 })
        .execute_cooperative(closures);
    let elapsed = t0.elapsed();
    let total: u64 = results.iter().sum();
    let expected: u64 = (0..32u64).map(|i| i * i).sum();
    println!("    32 closures executed in {elapsed:?}");
    println!("    sum of squares: {total} (expected {expected})");
    assert_eq!(total, expected);
    println!();

    // 3. WorkSteal (MIMD): recursive bisection.
    println!("[3] WorkloadShape::WorkSteal{{n:8, batch:256}} (MIMD) - parallel for");
    let mut items: Vec<u32> = (0..10_000u32).collect();
    let t0 = std::time::Instant::now();
    AdaptiveDispatcher::new()
        .with_shape(WorkloadShape::WorkSteal {
            n_consumers: 8,
            batch_size: 256,
        })
        .execute_for_each(&mut items, |slice| {
            for x in slice.iter_mut() {
                *x = x.wrapping_mul(3).wrapping_add(1);
            }
        });
    let elapsed = t0.elapsed();
    println!("    10000 in-place updates in {elapsed:?}");
    for (i, x) in items.iter().enumerate().take(5) {
        let expected = (i as u32).wrapping_mul(3).wrapping_add(1);
        assert_eq!(*x, expected, "item {i} expected {expected} got {x}");
    }
    println!("    first 5 items match expected; in-place mutation confirmed\n");

    // 4. Cooperative with mailbox (SIMC/MIMC at large N).
    println!("[4] WorkloadShape::Cooperative{{n_cores:16}} (SIMC/MIMC) - mailbox routing");
    let closures: Vec<Box<dyn FnOnce() -> u64 + Send>> = (0..16u64)
        .map(|i| {
            Box::new(move || {
                // small amount of work per closure
                let mut acc = i;
                for j in 0..100 {
                    acc = acc.wrapping_mul(31).wrapping_add(j);
                }
                acc
            }) as _
        })
        .collect();
    let t0 = std::time::Instant::now();
    let results = AdaptiveDispatcher::new()
        .with_shape(WorkloadShape::Cooperative { n_cores: 16 })
        .execute_cooperative_mailbox(closures);
    let elapsed = t0.elapsed();
    println!("    16 closures via mailbox routing in {elapsed:?}");
    println!("    first 3 results: {} {} {}", results[0], results[1], results[2]);
    assert_eq!(results.len(), 16);
    println!();

    // 5. Runtime K_gating migration: flip every worker's active
    //    backing from PerSlot (KHL) to CounterOnly (Fcl), run a
    //    dispatch on the new backing, flip back, run another.
    println!("[5] Runtime K_gating migration: PerSlot -> CounterOnly -> PerSlot");

    let dispatcher = AdaptiveDispatcher::new();
    let arena = flynnel::sched::arena::global_local_arena();
    println!("    burst ratio before migration: {:.3}",
        arena.global_burst_ratio());

    // Migrate every worker to CounterOnly (Fcl active).
    let t0 = std::time::Instant::now();
    dispatcher.migrate_k_gating(KGating::CounterOnly);
    let migrate_ns = t0.elapsed().as_nanos();
    println!("    migrate PerSlot -> CounterOnly took {} ns", migrate_ns);

    // Run a dispatch on the Fcl backing.
    let closures: Vec<Box<dyn FnOnce() -> u64 + Send>> = (0..16u64)
        .map(|i| Box::new(move || i) as _)
        .collect();
    let t0 = std::time::Instant::now();
    let r1 = AdaptiveDispatcher::new()
        .with_shape(WorkloadShape::ProducerFast { burst: 16 })
        .execute_cooperative(closures);
    let fcl_elapsed = t0.elapsed();
    println!("    16-closure dispatch on Fcl backing: {fcl_elapsed:?}, sum {}",
        r1.iter().sum::<u64>());

    // Migrate back to PerSlot (KHL active).
    let t0 = std::time::Instant::now();
    dispatcher.migrate_k_gating(KGating::PerSlot);
    let migrate_ns_back = t0.elapsed().as_nanos();
    println!("    migrate CounterOnly -> PerSlot took {} ns", migrate_ns_back);

    // Run another dispatch on the KHL backing.
    let closures: Vec<Box<dyn FnOnce() -> u64 + Send>> = (0..16u64)
        .map(|i| Box::new(move || i * 2) as _)
        .collect();
    let t0 = std::time::Instant::now();
    let r2 = AdaptiveDispatcher::new()
        .with_shape(WorkloadShape::ProducerFast { burst: 16 })
        .execute_cooperative(closures);
    let khl_elapsed = t0.elapsed();
    println!("    16-closure dispatch on KHL backing: {khl_elapsed:?}, sum {}",
        r2.iter().sum::<u64>());

    println!("    burst ratio after all dispatches: {:.3}",
        arena.global_burst_ratio());
    println!();

    // 6. Runtime DispatchProfile migration via WorkloadClass.
    //    Same AtomicU-tag pattern as K_gating; just one byte
    //    instead of per-(worker, tier) tags. Migration cost: ~1ns.
    println!("[6] Runtime DispatchProfile migration via WorkloadClass");
    println!("    initial profile: {:?}", dispatcher.active_dispatch_profile());

    // Migrate to Heavy: scheduler default plan now activates SMT
    // siblings + 4x oversubscription + LatencyBound cost estimate.
    let t0 = std::time::Instant::now();
    dispatcher.migrate_workload_class(WorkloadClass::LatencyBound);
    let m1 = t0.elapsed().as_nanos();
    println!("    migrate -> Heavy took {} ns; profile now {:?}",
        m1, dispatcher.active_dispatch_profile());

    // Dispatch a Heavy-class workload: each item does ~1000 sqrt
    // iters to simulate latency-bound work. The Heavy class's
    // SMT activation should let siblings fill the dispatch
    // bubbles between FP latency stalls.
    let mut items: Vec<f64> = (1..=20_000).map(|i| i as f64).collect();
    let t0 = std::time::Instant::now();
    AdaptiveDispatcher::new()
        .with_shape(WorkloadShape::WorkSteal {
            n_consumers: 8,
            batch_size: 256,
        })
        .execute_for_each(&mut items, |slice| {
            for x in slice.iter_mut() {
                let mut v = *x;
                for _ in 0..200 {
                    v = v.sqrt() * 1.0001 + 1.0;
                }
                *x = v;
            }
        });
    let heavy_elapsed = t0.elapsed();
    println!("    Heavy workload (20k items x 200 sqrt) in {heavy_elapsed:?}");

    // Migrate to Memory: SMT active, MemoryBound cost estimate.
    let t0 = std::time::Instant::now();
    dispatcher.migrate_workload_class(WorkloadClass::MemoryBound);
    let m2 = t0.elapsed().as_nanos();
    println!("    migrate -> Memory took {} ns; profile now {:?}",
        m2, dispatcher.active_dispatch_profile());

    // Migrate to Compute: SMT parked, PortBound cost estimate.
    let t0 = std::time::Instant::now();
    dispatcher.migrate_workload_class(WorkloadClass::PortBound);
    let m3 = t0.elapsed().as_nanos();
    println!("    migrate -> Compute took {} ns; profile now {:?}",
        m3, dispatcher.active_dispatch_profile());

    // Verify per-call class override takes precedence over global
    // AND actually dispatches: hold global at Compute (PortBound)
    // and exercise `with_workload_class(Heavy)` across all four
    // WorkloadShape variants. Each dispatch should see the Heavy
    // override applied to its built JobPlan (use_smt=true,
    // estimated_per_item_ns=Some(600), oversubscription=4x) without
    // mutating the global, which we re-read after each call.
    println!("    per-call with_workload_class(Heavy) override + 4 shape dispatches");
    println!("    (global remains PortBound throughout; only the dispatcher's plan changes)");

    // 6a. Streaming (SISD) with explicit Heavy.
    let r6a = AdaptiveDispatcher::new()
        .with_workload_class(WorkloadClass::LatencyBound)
        .with_shape(WorkloadShape::Streaming)
        .execute_streaming(|| {
            let mut acc = 0u64;
            for i in 0..5_000u64 {
                acc = acc.wrapping_add(i * i);
            }
            acc
        });
    let expected_6a: u64 = (0..5_000u64).map(|i| i.wrapping_mul(i)).fold(0u64, u64::wrapping_add);
    assert_eq!(r6a, expected_6a);
    println!("      [6a] Streaming + Heavy override: result {r6a} (matches expected); global still {:?}",
        dispatcher.active_dispatch_profile());

    // 6b. ProducerFast (SIMC) with explicit Heavy.
    let closures: Vec<Box<dyn FnOnce() -> u64 + Send>> = (0..16u64)
        .map(|i| Box::new(move || i.wrapping_mul(7).wrapping_add(3)) as _)
        .collect();
    let r6b = AdaptiveDispatcher::new()
        .with_workload_class(WorkloadClass::LatencyBound)
        .with_shape(WorkloadShape::ProducerFast { burst: 16 })
        .execute_cooperative(closures);
    let expected_6b: u64 = (0..16u64).map(|i| i.wrapping_mul(7).wrapping_add(3)).sum();
    let got_6b: u64 = r6b.iter().sum();
    assert_eq!(got_6b, expected_6b);
    println!("      [6b] ProducerFast{{burst:16}} + Heavy override: 16 closures, sum {got_6b} (matches); global still {:?}",
        dispatcher.active_dispatch_profile());

    // 6c. WorkSteal (MIMD) with explicit Heavy - per-element
    // latency-bound sqrt chain to exercise the SMT-active code
    // path the Heavy class's use_smt=true should engage.
    let mut items_6c: Vec<f64> = (1..=8_000).map(|i| i as f64).collect();
    AdaptiveDispatcher::new()
        .with_workload_class(WorkloadClass::LatencyBound)
        .with_shape(WorkloadShape::WorkSteal {
            n_consumers: 8,
            batch_size: 256,
        })
        .execute_for_each(&mut items_6c, |slice| {
            for x in slice.iter_mut() {
                let mut v = *x;
                for _ in 0..100 {
                    v = v.sqrt() * 1.0001 + 1.0;
                }
                *x = v;
            }
        });
    assert!(items_6c[0].is_finite() && items_6c[0] > 1.0);
    println!("      [6c] WorkSteal{{n:8,batch:256}} + Heavy override: 8000 items mutated, items[0]={:.3}; global still {:?}",
        items_6c[0], dispatcher.active_dispatch_profile());

    // 6d. Cooperative mailbox (SIMC/MIMC) with explicit Heavy.
    let closures: Vec<Box<dyn FnOnce() -> u64 + Send>> = (0..16u64)
        .map(|i| {
            Box::new(move || {
                let mut acc = i;
                for j in 0..100 {
                    acc = acc.wrapping_mul(31).wrapping_add(j);
                }
                acc
            }) as _
        })
        .collect();
    let r6d = AdaptiveDispatcher::new()
        .with_workload_class(WorkloadClass::LatencyBound)
        .with_shape(WorkloadShape::Cooperative { n_cores: 16 })
        .execute_cooperative_mailbox(closures);
    assert_eq!(r6d.len(), 16);
    println!("      [6d] Cooperative{{n_cores:16}} + Heavy override: 16 closures via mailbox; first result={}; global still {:?}",
        r6d[0], dispatcher.active_dispatch_profile());

    // Final check: global was NEVER mutated by any of the four
    // per-call overrides. This is the contract: with_workload_class
    // is per-dispatcher (per-call), not a global migration.
    assert!(
        matches!(dispatcher.active_dispatch_profile(), flynnel::DispatchProfile::PortBound),
        "global active profile must remain PortBound; per-call override leaked to global"
    );
    println!("    global profile unchanged after 4 per-call Heavy overrides: {:?} (contract upheld)",
        dispatcher.active_dispatch_profile());

    println!();

    // 7. Backend-adaptive routing: same AtomicU-tag pattern as
    //    K_gating + DispatchProfile, extended to backend selection.
    //    CPU is auto-registered; the demo registers Cuda
    //    explicitly under `#[cfg(feature = "cuda-reference")]` so
    //    `migrate -> Cuda{device_id:0}` lands on a real CudaBackend
    //    when the feature is enabled AND a CUDA driver is present.
    //    On `cargo run --example adaptive_dispatcher_demo --release`
    //    (no feature) the migration still works as the documented
    //    graceful-fallback path - the active-backend tag flips,
    //    resolve_active_backend() falls back to CPU, and the
    //    `fell_back: bool` surface tells the caller.
    println!("[7] Runtime backend migration via active-backend tag");

    use std::sync::Arc;
    use std::sync::atomic::{AtomicU32, Ordering};

    // Try to register Cuda when the cuda-reference feature is
    // compiled in AND the host has a usable CUDA driver. If either
    // is absent, the demo still runs and exercises the fallback
    // path - the print line below makes the registration outcome
    // explicit so users on a CUDA host know whether they engaged
    // the real backend or the fallback.
    #[cfg(feature = "cuda-reference")]
    let cuda_registered = match flynnel::backend::cuda::CudaBackend::new() {
        Ok(b) => {
            flynnel::backend::register_backend(Arc::new(b));
            println!("    [setup] cuda-reference feature ON + CudaBackend::new() succeeded -> Cuda registered");
            true
        }
        Err(e) => {
            println!("    [setup] cuda-reference feature ON but CudaBackend::new() returned {e:?}; falling through to CPU fallback path");
            false
        }
    };
    #[cfg(not(feature = "cuda-reference"))]
    let cuda_registered = {
        println!("    [setup] cuda-reference feature OFF; Cuda backend not compiled in. Re-run with --features cuda-reference to engage real CUDA.");
        false
    };

    // Same pattern for WASM: wasmtime is pure-Rust so WasmBackend::new()
    // succeeds anywhere the feature compiles in. Re-run with
    // --features wasm-reference to engage the real WASM sandbox.
    #[cfg(feature = "wasm-reference")]
    let wasm_registered = match flynnel::backend::wasm::WasmBackend::new() {
        Ok(b) => {
            flynnel::backend::register_backend(Arc::new(b));
            println!("    [setup] wasm-reference feature ON + WasmBackend::new() succeeded -> Wasm registered");
            true
        }
        Err(e) => {
            println!("    [setup] wasm-reference feature ON but WasmBackend::new() returned {e:?}; falling through to CPU fallback path");
            false
        }
    };
    #[cfg(not(feature = "wasm-reference"))]
    let wasm_registered = {
        println!("    [setup] wasm-reference feature OFF; Wasm backend not compiled in. Re-run with --features wasm-reference to engage the wasmtime sandbox.");
        false
    };

    // Same pattern for TPU JAX. TpuJaxBackend::new() spawns the
    // embedded Python+JAX bridge subprocess; on hosts without
    // python3 + jax installed, the constructor returns
    // BackendError::DeviceUnavailable and the demo falls back to
    // CPU. Re-run with --features tpu-jax-reference (and python3 +
    // jax on PATH) to engage the real TPU.
    #[cfg(feature = "tpu-jax-reference")]
    let tpu_registered = match flynnel::backend::tpu_jax::TpuJaxBackend::new() {
        Ok(b) => {
            flynnel::backend::register_backend(Arc::new(b));
            println!("    [setup] tpu-jax-reference feature ON + TpuJaxBackend::new() succeeded -> Tpu registered");
            true
        }
        Err(e) => {
            println!("    [setup] tpu-jax-reference feature ON but TpuJaxBackend::new() returned {e:?}; falling through to CPU fallback path");
            false
        }
    };
    #[cfg(not(feature = "tpu-jax-reference"))]
    let tpu_registered = {
        println!("    [setup] tpu-jax-reference feature OFF; Tpu backend not compiled in. Re-run with --features tpu-jax-reference to engage the JAX bridge.");
        false
    };

    let initial_backend = dispatcher.active_backend_id();
    let (b_ref, fell_back) = dispatcher.resolve_active_backend();
    println!("    initial active backend: {:?} (resolved: {:?}, fell_back={})",
        initial_backend, b_ref.id(), fell_back);
    println!("    capabilities: {:?}", b_ref.capabilities());

    // Dispatch through the active backend (CPU). Counter is
    // incremented atomically per index by the parallel-for work
    // closure.
    let counter = Arc::new(AtomicU32::new(0));
    let c = Arc::clone(&counter);
    let fell_back_cpu = AdaptiveDispatcher::new().execute_indexed(1000, move |_i| {
        c.fetch_add(1, Ordering::Relaxed);
    });
    println!("    1000-element parallel-for via Cpu backend: counter={}, fell_back={}",
        counter.load(Ordering::Relaxed), fell_back_cpu);

    // Migrate to Cuda. If we registered above, this resolves to
    // the real CudaBackend; otherwise it falls back to CPU.
    let t0 = std::time::Instant::now();
    dispatcher.migrate_backend(Backend::Cuda { device_id: 0 });
    let m_back = t0.elapsed().as_nanos();
    println!("    migrate -> Cuda{{device_id:0}} took {} ns", m_back);
    let (b_ref, fell_back) = dispatcher.resolve_active_backend();
    println!("    active_backend_id() reports {:?}", dispatcher.active_backend_id());
    println!("    resolve_active_backend() returned {:?}, fell_back={}",
        b_ref.id(), fell_back);
    let counter = Arc::new(AtomicU32::new(0));
    let c = Arc::clone(&counter);
    let fell_back_cuda = AdaptiveDispatcher::new().execute_indexed(500, move |_i| {
        c.fetch_add(1, Ordering::Relaxed);
    });
    println!("    500-element parallel-for after Cuda migration: counter={}, fell_back={}",
        counter.load(Ordering::Relaxed), fell_back_cuda);
    if cuda_registered {
        assert!(!fell_back_cuda,
            "Cuda was registered but resolve still fell back to CPU - this is a real bug");
        println!("    (real CudaBackend dispatched the work; no fallback)");
    } else {
        println!("    (fell back to CPU because Cuda is not registered on this host)");
    }

    // Same migration story for WASM. The Backend::Wasm variant
    // resolves to the WasmBackend impl if registered above; if
    // wasm-reference is off, resolves to CPU + fell_back=true.
    let t0 = std::time::Instant::now();
    dispatcher.migrate_backend(Backend::Wasm { device_id: 0 });
    let m_wasm = t0.elapsed().as_nanos();
    println!("    migrate -> Wasm{{device_id:0}} took {} ns", m_wasm);
    let (b_ref, fell_back_wasm_resolve) = dispatcher.resolve_active_backend();
    println!("    active_backend_id() reports {:?}", dispatcher.active_backend_id());
    println!("    resolve_active_backend() returned {:?}, fell_back={}",
        b_ref.id(), fell_back_wasm_resolve);
    let counter = Arc::new(AtomicU32::new(0));
    let c = Arc::clone(&counter);
    let fell_back_wasm = AdaptiveDispatcher::new().execute_indexed(300, move |_i| {
        c.fetch_add(1, Ordering::Relaxed);
    });
    println!("    300-element parallel-for after Wasm migration: counter={}, fell_back={}",
        counter.load(Ordering::Relaxed), fell_back_wasm);
    if wasm_registered {
        assert!(!fell_back_wasm,
            "Wasm was registered but resolve still fell back to CPU - this is a real bug");
        println!("    (real WasmBackend dispatched the work; no fallback)");
    } else {
        println!("    (fell back to CPU because Wasm is not registered on this host)");
    }

    // Same migration story for TPU.
    let t0 = std::time::Instant::now();
    dispatcher.migrate_backend(Backend::Tpu { device_id: 0 });
    let m_tpu = t0.elapsed().as_nanos();
    println!("    migrate -> Tpu{{device_id:0}} took {} ns", m_tpu);
    let (b_ref, fell_back_tpu_resolve) = dispatcher.resolve_active_backend();
    println!("    active_backend_id() reports {:?}", dispatcher.active_backend_id());
    println!("    resolve_active_backend() returned {:?}, fell_back={}",
        b_ref.id(), fell_back_tpu_resolve);
    let counter = Arc::new(AtomicU32::new(0));
    let c = Arc::clone(&counter);
    let fell_back_tpu = AdaptiveDispatcher::new().execute_indexed(200, move |_i| {
        c.fetch_add(1, Ordering::Relaxed);
    });
    println!("    200-element parallel-for after Tpu migration: counter={}, fell_back={}",
        counter.load(Ordering::Relaxed), fell_back_tpu);
    if tpu_registered {
        assert!(!fell_back_tpu,
            "Tpu was registered but resolve still fell back to CPU - this is a real bug");
        println!("    (real TpuJaxBackend dispatched the work; no fallback)");
    } else {
        println!("    (fell back to CPU because Tpu is not registered on this host)");
    }

    // Migrate back to CPU explicitly.
    dispatcher.migrate_backend(Backend::Cpu);
    println!("    migrated back to Cpu; active_backend_id() = {:?}",
        dispatcher.active_backend_id());

    println!("\n=== Demo complete: unified API + K_gating + DispatchProfile + Backend adaptive ===");
}
