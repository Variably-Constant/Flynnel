//! Argument codec for the shared-memory dispatch backends.
//!
//! Encodes a [`KernelArg`] slice as a flat byte blob (1-byte tag +
//! little-endian scalar payload per arg) and the inverse decode.
//! Consumed by [`super::chase_lev_backend::SharedMemoryChaseLevBackend`]
//! when packing args into a [`super::chase_lev_mmf::RemoteJobSlot`].
//!
//! Per-slot inline-args capacity is
//! [`super::chase_lev_mmf::ARGS_INLINE_BYTES`] (= 48). Args larger
//! than that need a separate transport (a typed allocator + device
//! pointer rather than an inline blob).
//!
//! `KernelArg::HostSlice` is rejected because a 48-byte inline blob
//! cannot carry a meaningful host buffer; callers wanting cross-
//! process bulk transfer arrange their own shared region and pass a
//! `DevicePtr` offset into it.

use super::chase_lev_mmf::ARGS_INLINE_BYTES;
use crate::backend::{BackendError, KernelArg};

/// Maximum encoded-args length that still fits a single slot.
pub const MAX_ARGS_BYTES: usize = ARGS_INLINE_BYTES;

/// Tag byte that prefixes each encoded [`KernelArg`].
mod tag {
    pub const I32: u8 = 1;
    pub const I64: u8 = 2;
    pub const U32: u8 = 3;
    pub const U64: u8 = 4;
    pub const F32: u8 = 5;
    pub const F64: u8 = 6;
    pub const DEV_PTR: u8 = 7;
}

/// Encode a `KernelArg` slice into a flat byte blob. Returns
/// [`BackendError::NotSupported`] for the `HostSlice` variant.
pub fn encode_args(args: &[KernelArg<'_>]) -> Result<Vec<u8>, BackendError> {
    let mut out = Vec::with_capacity(args.len() * 9);
    for arg in args {
        match arg {
            KernelArg::I32(v) => {
                out.push(tag::I32);
                out.extend_from_slice(&v.to_le_bytes());
            }
            KernelArg::I64(v) => {
                out.push(tag::I64);
                out.extend_from_slice(&v.to_le_bytes());
            }
            KernelArg::U32(v) => {
                out.push(tag::U32);
                out.extend_from_slice(&v.to_le_bytes());
            }
            KernelArg::U64(v) => {
                out.push(tag::U64);
                out.extend_from_slice(&v.to_le_bytes());
            }
            KernelArg::F32(v) => {
                out.push(tag::F32);
                out.extend_from_slice(&v.to_le_bytes());
            }
            KernelArg::F64(v) => {
                out.push(tag::F64);
                out.extend_from_slice(&v.to_le_bytes());
            }
            KernelArg::DevicePtr(v) => {
                out.push(tag::DEV_PTR);
                out.extend_from_slice(&(*v as u64).to_le_bytes());
            }
            KernelArg::HostSlice(_) => {
                return Err(BackendError::NotSupported);
            }
        }
    }
    Ok(out)
}

/// Decoded `KernelArg`-equivalent; owns its bytes so it can outlive
/// the slot buffer the bytes were copied out of.
#[derive(Debug, Clone, PartialEq)]
pub enum DecodedArg {
    /// Decoded 32-bit signed integer.
    I32(i32),
    /// Decoded 64-bit signed integer.
    I64(i64),
    /// Decoded 32-bit unsigned integer.
    U32(u32),
    /// Decoded 64-bit unsigned integer.
    U64(u64),
    /// Decoded 32-bit float.
    F32(f32),
    /// Decoded 64-bit float.
    F64(f64),
    /// Decoded device-side pointer.
    DevicePtr(usize),
}

/// Inverse of [`encode_args`].
pub fn decode_args(blob: &[u8]) -> Result<Vec<DecodedArg>, BackendError> {
    let mut out = Vec::new();
    let mut i = 0usize;
    while i < blob.len() {
        let t = blob[i];
        i += 1;
        let need = match t {
            tag::I32 | tag::U32 | tag::F32 => 4,
            tag::I64 | tag::U64 | tag::F64 | tag::DEV_PTR => 8,
            other => {
                return Err(BackendError::Launch(format!(
                    "unknown KernelArg tag {other} in wire payload"
                )));
            }
        };
        if i + need > blob.len() {
            return Err(BackendError::Launch(
                "truncated KernelArg payload".to_string(),
            ));
        }
        let slice = &blob[i..i + need];
        i += need;
        out.push(match t {
            tag::I32 => DecodedArg::I32(i32::from_le_bytes(slice.try_into().unwrap())),
            tag::I64 => DecodedArg::I64(i64::from_le_bytes(slice.try_into().unwrap())),
            tag::U32 => DecodedArg::U32(u32::from_le_bytes(slice.try_into().unwrap())),
            tag::U64 => DecodedArg::U64(u64::from_le_bytes(slice.try_into().unwrap())),
            tag::F32 => DecodedArg::F32(f32::from_le_bytes(slice.try_into().unwrap())),
            tag::F64 => DecodedArg::F64(f64::from_le_bytes(slice.try_into().unwrap())),
            tag::DEV_PTR => {
                DecodedArg::DevicePtr(u64::from_le_bytes(slice.try_into().unwrap()) as usize)
            }
            _ => unreachable!(),
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn args_round_trip_for_every_scalar_variant() {
        let args = vec![
            KernelArg::I32(-7),
            KernelArg::I64(-9_000_000_000),
            KernelArg::U32(42),
            KernelArg::U64(0xDEAD_BEEF_CAFE_BABE),
            KernelArg::F32(1.5_f32),
            KernelArg::F64(123.456_789_f64),
            KernelArg::DevicePtr(0x1000_2000_3000),
        ];
        let blob = encode_args(&args).expect("encode");
        let back = decode_args(&blob).expect("decode");
        assert_eq!(back.len(), args.len());
        assert_eq!(back[0], DecodedArg::I32(-7));
        assert_eq!(back[1], DecodedArg::I64(-9_000_000_000));
        assert_eq!(back[2], DecodedArg::U32(42));
        assert_eq!(back[3], DecodedArg::U64(0xDEAD_BEEF_CAFE_BABE));
        assert_eq!(back[4], DecodedArg::F32(1.5_f32));
        assert_eq!(back[5], DecodedArg::F64(123.456_789_f64));
        assert_eq!(back[6], DecodedArg::DevicePtr(0x1000_2000_3000));
    }

    #[test]
    fn host_slice_arg_rejected() {
        let buf = [1u8, 2, 3];
        let args = vec![KernelArg::HostSlice(&buf)];
        match encode_args(&args) {
            Err(BackendError::NotSupported) => {}
            other => panic!("expected NotSupported, got {other:?}"),
        }
    }

    #[test]
    fn decode_rejects_unknown_tag() {
        let bad = vec![0xFFu8, 0, 0, 0, 0];
        let err = decode_args(&bad).expect_err("expected tag error");
        assert!(matches!(err, BackendError::Launch(_)));
    }

    #[test]
    fn args_blob_fits_chase_lev_slot() {
        // Even a slot full of u64 args should fit comfortably in the
        // 48-byte inline payload: 5 u64s = 5 * 9 = 45 bytes.
        let args: Vec<KernelArg<'_>> = vec![
            KernelArg::U64(1),
            KernelArg::U64(2),
            KernelArg::U64(3),
            KernelArg::U64(4),
            KernelArg::U64(5),
        ];
        let blob = encode_args(&args).expect("encode");
        assert!(blob.len() <= MAX_ARGS_BYTES);
    }
}
