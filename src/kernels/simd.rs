//! The instruction-set level each block of a declared kernel runs at.
//!
//! A library built for the x86-64 floor compiles a block's work for SSE2
//! alone. [`run_at`] runs a [`Block`] through a function compiled for one
//! x86-64 psABI level with `#[target_feature]`. A block whose `run_block`
//! is `#[inline(always)]`, with each helper its loop calls marked the
//! same, compiles inside every one of those functions, so it has a copy
//! per level and the level a job holds picks the copy. Code that does not
//! inline into them, which includes a closure's body larger than the
//! compiler's inlining threshold, compiles once at the baseline, and every
//! level runs that one copy. Every copy answers the same bits for every
//! value but a NaN: no level contracts a multiply and an add or reorders
//! floating-point arithmetic. A NaN comes out a NaN at every level, its
//! sign and payload free to differ: the C runtime's rounding keeps a
//! signaling NaN that the rounding instruction quiets, and an encoding
//! free to swap an addition's operands picks which of two NaNs survives.
//!
//! The level is the highest one whose every extension the CPU reports,
//! with the operating system enabling the state it needs, read once per
//! process and capped by `PWRS_CPU_MAX` as PoWerRuSt caps it: `x86-64`,
//! `x86-64-v2`, `x86-64-v3` or `x86-64-v4` names a cap, and `native`,
//! empty, unset or any other value is none.

use std::sync::OnceLock;

/// One job's block work, which [`run_at`] runs at a level.
///
/// An implementation whose loop can use a wider level marks `run_block`
/// `#[inline(always)]`, and each helper in that loop too, so its body
/// compiles inside each level's function. One whose time goes to I/O or to
/// code built outside the crate leaves the mark off, and every level runs
/// its one copy.
pub(crate) trait Block {
    /// Run block `block` of `phase`.
    fn run_block(&self, phase: usize, block: usize);
}

/// A closure run as a block, with nothing forcing its body into the level
/// functions: the compiler keeps one copy of a body larger than its
/// inlining threshold, and every level runs it. For work whose time goes
/// to I/O or to code built outside the crate.
pub(crate) struct OneCopy<F: Fn(usize, usize)>(pub(crate) F);

impl<F: Fn(usize, usize)> Block for OneCopy<F> {
    fn run_block(&self, phase: usize, block: usize) {
        (self.0)(phase, block);
    }
}

/// An x86-64 psABI microarchitecture level.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Level {
    /// x86-64: SSE and SSE2.
    Base,
    /// x86-64-v2: adds SSE3, SSSE3, SSE4.1, SSE4.2, POPCNT and CMPXCHG16B.
    V2,
    /// x86-64-v3: adds AVX, AVX2, BMI1, BMI2, F16C, FMA, LZCNT, MOVBE and
    /// XSAVE.
    V3,
    /// x86-64-v4: adds AVX-512 F, BW, CD, DQ and VL.
    V4,
}

impl Level {
    #[cfg(test)]
    pub(crate) const ALL: [Level; 4] = [Level::Base, Level::V2, Level::V3, Level::V4];

    /// The cap a `PWRS_CPU_MAX` value names, or `None` for no cap.
    fn cap_named(value: &str) -> Option<Level> {
        match value.trim() {
            "x86-64" => Some(Level::Base),
            "x86-64-v2" => Some(Level::V2),
            "x86-64-v3" => Some(Level::V3),
            "x86-64-v4" => Some(Level::V4),
            _no_cap => None,
        }
    }
}

/// The cap `PWRS_CPU_MAX` names, or `None` for no cap.
fn cap() -> Option<Level> {
    match std::env::var("PWRS_CPU_MAX") {
        Ok(value) => Level::cap_named(&value),
        Err(std::env::VarError::NotPresent) => None,
        Err(std::env::VarError::NotUnicode(_value)) => None,
    }
}

/// The highest level this CPU offers.
#[cfg(target_arch = "x86_64")]
fn offered() -> Level {
    let v2 = is_x86_feature_detected!("sse3")
        && is_x86_feature_detected!("ssse3")
        && is_x86_feature_detected!("sse4.1")
        && is_x86_feature_detected!("sse4.2")
        && is_x86_feature_detected!("popcnt")
        && is_x86_feature_detected!("cmpxchg16b");
    let v3 = v2
        && is_x86_feature_detected!("avx")
        && is_x86_feature_detected!("avx2")
        && is_x86_feature_detected!("bmi1")
        && is_x86_feature_detected!("bmi2")
        && is_x86_feature_detected!("f16c")
        && is_x86_feature_detected!("fma")
        && is_x86_feature_detected!("lzcnt")
        && is_x86_feature_detected!("movbe")
        && is_x86_feature_detected!("xsave");
    let v4 = v3
        && is_x86_feature_detected!("avx512f")
        && is_x86_feature_detected!("avx512bw")
        && is_x86_feature_detected!("avx512cd")
        && is_x86_feature_detected!("avx512dq")
        && is_x86_feature_detected!("avx512vl");
    if v4 {
        Level::V4
    } else if v3 {
        Level::V3
    } else if v2 {
        Level::V2
    } else {
        Level::Base
    }
}

/// The highest level this CPU offers.
#[cfg(not(target_arch = "x86_64"))]
fn offered() -> Level {
    Level::Base
}

/// The level this process runs blocks at: the highest the CPU offers,
/// capped by `PWRS_CPU_MAX`, read once.
pub(crate) fn process_level() -> Level {
    static LEVEL: OnceLock<Level> = OnceLock::new();
    *LEVEL.get_or_init(|| match cap() {
        Some(cap) => offered().min(cap),
        None => offered(),
    })
}

/// The level a job made now runs its blocks at.
#[cfg(not(test))]
pub(crate) fn for_new_job() -> Level {
    process_level()
}

#[cfg(test)]
thread_local! {
    static FORCED: std::cell::Cell<Option<Level>> = const { std::cell::Cell::new(None) };
}

/// The level a job made now runs its blocks at: the one [`with_level`]
/// forced on this thread, or the process's.
#[cfg(test)]
pub(crate) fn for_new_job() -> Level {
    FORCED
        .with(std::cell::Cell::get)
        .unwrap_or_else(process_level)
}

/// Every level this CPU offers, lowest first, whatever `PWRS_CPU_MAX`
/// says.
#[cfg(test)]
pub(crate) fn levels_offered() -> Vec<Level> {
    let top = offered();
    Level::ALL.into_iter().filter(|&l| l <= top).collect()
}

/// `f`'s answer, with every job `f` makes on this thread running its
/// blocks at `level`, which this CPU must offer.
#[cfg(test)]
pub(crate) fn with_level<R>(level: Level, f: impl FnOnce() -> R) -> R {
    assert!(
        level <= offered(),
        "{level:?} is above what this CPU offers"
    );
    let before = FORCED.with(|forced| forced.replace(Some(level)));
    let answer = f();
    FORCED.with(|forced| forced.set(before));
    answer
}

/// Asserts that `answer` returns at every level this CPU offers what it
/// returns at the baseline, naming `what` and the level where it does not.
#[cfg(test)]
pub(crate) fn assert_every_level_agrees<T: PartialEq>(what: &str, answer: impl Fn() -> T) {
    let base = with_level(Level::Base, &answer);
    for level in levels_offered() {
        assert!(
            with_level(level, &answer) == base,
            "{what} at {level:?} answered other bits than at the baseline"
        );
    }
}

/// Block `block` of `phase` of `work`, run through the function compiled
/// for `level`, which must be one [`process_level`] handed out, or in a
/// test one `with_level` forced.
#[cfg(target_arch = "x86_64")]
#[inline]
pub(crate) fn run_at<B: Block>(level: Level, work: &B, phase: usize, block: usize) {
    match level {
        // SAFETY: a level above Base comes from `offered`, directly or
        // capped below it, so the CPU runs every extension the function it
        // names was compiled with.
        Level::V4 => unsafe { v4(work, phase, block) },
        // SAFETY: as above.
        Level::V3 => unsafe { v3(work, phase, block) },
        // SAFETY: as above.
        Level::V2 => unsafe { v2(work, phase, block) },
        Level::Base => work.run_block(phase, block),
    }
}

/// Block `block` of `phase` of `work`, run as compiled; no target but
/// x86-64 has a ladder.
#[cfg(not(target_arch = "x86_64"))]
#[inline]
pub(crate) fn run_at<B: Block>(_level: Level, work: &B, phase: usize, block: usize) {
    work.run_block(phase, block);
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b")]
fn v2<B: Block>(work: &B, phase: usize, block: usize) {
    work.run_block(phase, block);
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(
    enable = "sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,bmi1,bmi2,f16c,fma,lzcnt,movbe,xsave"
)]
fn v3<B: Block>(work: &B, phase: usize, block: usize) {
    work.run_block(phase, block);
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(
    enable = "sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,bmi1,bmi2,f16c,fma,lzcnt,movbe,xsave,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
)]
fn v4<B: Block>(work: &B, phase: usize, block: usize) {
    work.run_block(phase, block);
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;

    /// A block that stores the product of its phase and block numbers.
    struct Product(AtomicUsize);

    impl Block for Product {
        #[inline(always)]
        fn run_block(&self, phase: usize, block: usize) {
            self.0.store(phase * block, Ordering::Relaxed);
        }
    }

    /// A block that panics.
    struct Panics;

    impl Block for Panics {
        fn run_block(&self, _phase: usize, _block: usize) {
            panic!("mid-block");
        }
    }

    #[test]
    fn a_cap_is_spelled_as_pwrs_spells_it() {
        assert_eq!(Level::cap_named("x86-64"), Some(Level::Base));
        assert_eq!(Level::cap_named("x86-64-v2"), Some(Level::V2));
        assert_eq!(Level::cap_named(" x86-64-v3 "), Some(Level::V3));
        assert_eq!(Level::cap_named("x86-64-v4"), Some(Level::V4));
        assert_eq!(Level::cap_named("native"), None);
        assert_eq!(Level::cap_named(""), None);
        assert_eq!(Level::cap_named("x86-64-v5"), None);
    }

    #[test]
    fn the_process_level_is_offered_and_every_offered_level_runs_work() {
        let offered = levels_offered();
        assert_eq!(offered.first(), Some(&Level::Base));
        assert!(offered.contains(&process_level()));
        for &level in &offered {
            let forced = with_level(level, for_new_job);
            assert_eq!(forced, level);
            let product = Product(AtomicUsize::new(0));
            run_at(level, &product, 6, 7);
            assert_eq!(product.0.load(Ordering::Relaxed), 42, "{level:?}");
            let sum = AtomicUsize::new(0);
            let adds = OneCopy(|phase, block| sum.store(phase + block, Ordering::Relaxed));
            run_at(level, &adds, 6, 7);
            assert_eq!(sum.load(Ordering::Relaxed), 13, "{level:?}");
        }
        assert_eq!(
            for_new_job(),
            process_level(),
            "the override ends with with_level"
        );
    }

    #[test]
    fn a_panic_in_the_work_reaches_the_caller_at_every_level() {
        for level in levels_offered() {
            let outcome = std::panic::catch_unwind(|| run_at(level, &Panics, 0, 0));
            match outcome {
                Ok(()) => panic!("{level:?}: the work's panic did not reach the caller"),
                Err(payload) => assert_eq!(payload.downcast_ref::<&str>(), Some(&"mid-block")),
            }
        }
    }
}
