//! Size suffixes on the parameters that take a byte size or a large
//! count.
//!
//! pwsh 7 binds a string such as '2MB' or '1.5GB' to a numeric parameter
//! by itself. Windows PowerShell 5.1 binds only the bare literal 2MB and
//! refuses the same text as a string, which is the form a size takes
//! when it comes from a variable, a CSV or a settings file. These
//! transforms make 5.1 bind what 7 binds.
//!
//! A transform runs before the binder on both editions, so on 7 it must
//! change nothing. It therefore reads only the forms it can read exactly
//! as 7 does: an optional sign, a decimal number with an optional
//! fraction and exponent, and one of KB, MB, GB, TB or PB in any case,
//! with white space around the whole. Every other value, whether a
//! number or a string, goes back unchanged for the binder to coerce or
//! refuse, so a form this does not read, a hex number with a suffix
//! among them, binds on 7 as before and is refused on 5.1 as before. The
//! multipliers are binary, as in PowerShell: 1KB is 1024.
//!
//! Each transform costs one boundary crossing whenever its parameter is
//! bound, whatever the value, which is why only these parameters carry
//! one.

use pwrs::prelude::*;
use pwrs::sys::PS_TYPE_STRING;

/// What a size-suffixed string reads as.
///
/// A whole number stays an Int64 while the product fits one, and
/// becomes a double past that or when the number has a fraction or an
/// exponent, which is what pwsh 7 hands its binder for the same text.
/// The binder then coerces it to the parameter's type, rounding a
/// fraction and refusing a value out of range, the same on both
/// editions.
#[derive(Debug, PartialEq)]
enum Size {
    Whole(i64),
    Real(f64),
}

/// The multiplier a two-letter suffix names.
fn multiplier(suffix: &str) -> Option<i64> {
    match suffix.to_ascii_lowercase().as_str() {
        "kb" => Some(1 << 10),
        "mb" => Some(1 << 20),
        "gb" => Some(1 << 30),
        "tb" => Some(1 << 40),
        "pb" => Some(1 << 50),
        _other => None,
    }
}

/// Whether every byte of the text is an ASCII digit. True for empty text.
fn all_digits(text: &str) -> bool {
    text.bytes().all(|b| b.is_ascii_digit())
}

/// The whole number the digits spell, times the scale, while that fits
/// an Int64. The digits are ASCII digits and at least one.
fn whole_times(digits: &str, scale: i64) -> Option<i64> {
    let mut n: i64 = 0;
    for b in digits.bytes() {
        n = n.checked_mul(10)?.checked_add(i64::from(b - b'0'))?;
    }
    n.checked_mul(scale)
}

/// Whether the text is a real literal: digits with a point, digits after
/// a leading point, or either of those or plain digits followed by an
/// exponent. No sign; the caller has taken it.
fn is_real(text: &str) -> bool {
    let (mantissa, exponent) = match text.find(['e', 'E']) {
        Some(at) => (&text[..at], Some(&text[at + 1..])),
        None => (text, None),
    };
    let (whole, fraction) = match mantissa.find('.') {
        Some(at) => (&mantissa[..at], Some(&mantissa[at + 1..])),
        None => (mantissa, None),
    };
    let has_digit = !whole.is_empty() || fraction.is_some_and(|f| !f.is_empty());
    let mantissa_ok = has_digit && all_digits(whole) && fraction.is_none_or(all_digits);
    let exponent_ok = match exponent {
        None => fraction.is_some(),
        Some(e) => {
            let unsigned = e.strip_prefix(['+', '-']).unwrap_or(e);
            !unsigned.is_empty() && all_digits(unsigned)
        }
    };
    mantissa_ok && exponent_ok
}

/// Reads a size-suffixed number, or None when the text is not one.
fn read_size(text: &str) -> Option<Size> {
    let t = text.trim();
    let split = t.len().checked_sub(2)?;
    if !t.is_char_boundary(split) {
        return None;
    }
    let (number, suffix) = t.split_at(split);
    let scale = multiplier(suffix)?;
    let (negative, unsigned) = match number.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, number.strip_prefix('+').unwrap_or(number)),
    };
    let signed = |v: f64| if negative { -v } else { v };
    if !unsigned.is_empty() && all_digits(unsigned) {
        return Some(match whole_times(unsigned, scale) {
            Some(v) => Size::Whole(if negative { -v } else { v }),
            None => Size::Real(signed(real(unsigned) * scale as f64)),
        });
    }
    if is_real(unsigned) {
        return Some(Size::Real(signed(real(unsigned) * scale as f64)));
    }
    None
}

/// The value of text that is_real or all_digits has already admitted.
fn real(text: &str) -> f64 {
    text.parse::<f64>()
        .expect("only text the grammar admits reaches here, and f64 parses all of it")
}

/// Reads a size-suffixed string into the number it stands for, and
/// hands every other value back unchanged.
pub(crate) fn size_suffix(value: &PsObject) -> PsResult<PsObject> {
    if value.is_null() || value.type_tag()? != PS_TYPE_STRING {
        return value.clone().into_ps();
    }
    let text = String::from_ps(value)?;
    match read_size(&text) {
        Some(Size::Whole(n)) => n.into_ps(),
        Some(Size::Real(v)) => v.into_ps(),
        None => value.clone().into_ps(),
    }
}

/// Declares one size-suffix transform per cmdlet and parameter.
macro_rules! size_suffix_on {
    ($($name:ident: $cmdlet:tt $parameter:tt;)*) => {
        $(
            #[transform(cmdlet = $cmdlet, parameter = $parameter)]
            pub(crate) fn $name(value: &PsObject) -> PsResult<PsObject> {
                size_suffix(value)
            }
        )*
    };
}

size_suffix_on! {
    ring_capacity: "New-FlynnelRing" "Capacity";
    spsc_ring_capacity: "New-FlynnelSpscRing" "Capacity";
    mpsc_ring_capacity: "New-FlynnelMpscRing" "Capacity";
    composed_mpsc_capacity: "New-FlynnelComposedMpsc" "Capacity";
    composed_mpmc_capacity: "New-FlynnelComposedMpmc" "Capacity";
    injector_capacity: "New-FlynnelInjector" "Capacity";
    notify_ring_capacity: "New-FlynnelNotifyRing" "Capacity";
    peer_config_slot_bytes: "New-FlynnelGpuPeerConfig" "SlotBytes";
    peer_config_slots_per_lane: "New-FlynnelGpuPeerConfig" "SlotsPerLane";
    peer_config_vram_block_bytes: "New-FlynnelGpuPeerConfig" "VramBlockBytes";
    plan_batch_size: "New-FlynnelPlan" "BatchSize";
    plan_shape_batch_size: "Update-FlynnelPlan" "ShapeBatchSize";
    hybrid_join_count: "Measure-FlynnelHybridJoin" "Count";
    hybrid_placement_count: "Measure-FlynnelHybridPlacement" "Count";
    hybrid_split_count: "Measure-FlynnelHybridSplit" "Count";
    hybrid_pipeline_count: "Measure-FlynnelHybridPipeline" "Count";
    race_any_count: "Measure-FlynnelRaceAny" "Count";
    explore_select_count: "Measure-FlynnelExploreSelect" "Count";
}

#[cfg(test)]
mod tests {
    use super::{Size, read_size};

    /// Every case here was bound by pwsh 7.6.6 to a [long] and a
    /// [double] parameter, and the expected value is what 7 produced.
    #[test]
    fn reads_what_pwsh_7_reads() {
        let cases: &[(&str, Size)] = &[
            ("2MB", Size::Whole(2_097_152)),
            ("2mb", Size::Whole(2_097_152)),
            ("2Mb", Size::Whole(2_097_152)),
            ("-2MB", Size::Whole(-2_097_152)),
            ("+2MB", Size::Whole(2_097_152)),
            (" 2MB", Size::Whole(2_097_152)),
            ("2MB ", Size::Whole(2_097_152)),
            ("2kb", Size::Whole(2048)),
            ("2TB", Size::Whole(2_199_023_255_552)),
            ("2PB", Size::Whole(2_251_799_813_685_248)),
            (".5MB", Size::Real(524_288.0)),
            ("1.KB", Size::Real(1024.0)),
            ("1.5GB", Size::Real(1_610_612_736.0)),
            ("1.1KB", Size::Real(1126.4)),
            ("2.5KB", Size::Real(2560.0)),
            ("1e3KB", Size::Real(1_024_000.0)),
            ("9223372036854775807KB", Size::Real(9_223_372_036_854_775_807.0 * 1024.0)),
        ];
        for (text, expected) in cases {
            assert_eq!(read_size(text).as_ref(), Some(expected), "reading {text:?}");
        }
    }

    /// Forms pwsh 7 refuses, and forms it reads but this does not, go
    /// back to the binder unchanged: the first stay refused on both
    /// editions and the second bind on 7 as they did.
    #[test]
    fn hands_back_what_it_does_not_read() {
        for text in [
            "2EB", "1,000KB", "2MBs", "MB", "2 MB", "0x10KB", "0x10", "3000000000", "1d", "2MBl",
            "1_000", "2e3", "", "KB", ".KB", "1eKB", "1e+KB", "e3KB", "1.2.3KB", "- 2MB", "2ΜΒ",
        ] {
            assert_eq!(read_size(text), None, "reading {text:?}");
        }
    }
}
