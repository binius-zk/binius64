// Copyright 2024-2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use std::{
	arch::aarch64::*,
	ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not, Shl, Shr},
};

use binius_utils::{
	DeserializeBytes, FixedSizeSerializeBytes, SerializationError, SerializeBytes,
	bytes::{Buf, BufMut},
	serialization::{assert_enough_data_for, assert_enough_space_for},
};
use bytemuck::{Pod, Zeroable};
use rand::{distr::StandardUniform, prelude::*};
use seq_macro::seq;

use super::super::portable::packed_arithmetic::interleave_mask_even;
use crate::{
	BinaryField,
	divisible::{Divisible, impl_divisible_memcast, impl_divisible_self},
	packed_fields::primitive::PackedPrimitiveType,
	underlier::{SmallU, Underlier, impl_divisible_bitmask},
};

/// 128-bit value that is used for 128-bit SIMD operations
#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct M128(uint64x2_t);

impl M128 {
	pub const fn from_le_bytes(bytes: [u8; 16]) -> Self {
		Self(unsafe { std::mem::transmute::<u128, uint64x2_t>(u128::from_le_bytes(bytes)) })
	}

	pub const fn from_be_bytes(bytes: [u8; 16]) -> Self {
		Self(unsafe { std::mem::transmute::<u128, uint64x2_t>(u128::from_be_bytes(bytes)) })
	}

	pub const fn from_u128(value: u128) -> Self {
		Self(unsafe { std::mem::transmute::<u128, uint64x2_t>(value) })
	}
}

impl Default for M128 {
	fn default() -> Self {
		Self(unsafe { vdupq_n_u64(0) })
	}
}

impl PartialEq for M128 {
	fn eq(&self, other: &Self) -> bool {
		unsafe {
			let cmp = vceqq_u64(self.0, other.0);
			vminvq_u32(vreinterpretq_u32_u64(cmp)) == u32::MAX
		}
	}
}

impl Eq for M128 {}

impl PartialOrd for M128 {
	fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
		Some(self.cmp(other))
	}
}

impl Ord for M128 {
	fn cmp(&self, other: &Self) -> std::cmp::Ordering {
		u128::from(*self).cmp(&u128::from(*other))
	}
}

impl std::hash::Hash for M128 {
	fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
		u128::from(*self).hash(state);
	}
}

impl std::fmt::LowerHex for M128 {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		std::fmt::LowerHex::fmt(&u128::from(*self), f)
	}
}

unsafe impl Zeroable for M128 {}
unsafe impl Pod for M128 {}

impl From<M128> for u128 {
	fn from(value: M128) -> Self {
		unsafe { vreinterpretq_p128_u64(value.0) }
	}
}
impl From<M128> for uint8x16_t {
	fn from(value: M128) -> Self {
		unsafe { vreinterpretq_u8_u64(value.0) }
	}
}
impl From<M128> for uint16x8_t {
	fn from(value: M128) -> Self {
		unsafe { vreinterpretq_u16_u64(value.0) }
	}
}
impl From<M128> for uint32x4_t {
	fn from(value: M128) -> Self {
		unsafe { vreinterpretq_u32_u64(value.0) }
	}
}
impl From<M128> for uint64x2_t {
	fn from(value: M128) -> Self {
		value.0
	}
}
impl From<M128> for poly8x16_t {
	fn from(value: M128) -> Self {
		unsafe { vreinterpretq_p8_u64(value.0) }
	}
}
impl From<M128> for poly16x8_t {
	fn from(value: M128) -> Self {
		unsafe { vreinterpretq_p16_u64(value.0) }
	}
}
impl From<M128> for poly64x2_t {
	fn from(value: M128) -> Self {
		unsafe { vreinterpretq_p64_u64(value.0) }
	}
}

impl From<u128> for M128 {
	fn from(value: u128) -> Self {
		Self(unsafe { vreinterpretq_u64_p128(value) })
	}
}
impl From<u64> for M128 {
	fn from(value: u64) -> Self {
		Self::from(value as u128)
	}
}
impl From<u32> for M128 {
	fn from(value: u32) -> Self {
		Self::from(value as u128)
	}
}
impl From<u16> for M128 {
	fn from(value: u16) -> Self {
		Self::from(value as u128)
	}
}
impl From<u8> for M128 {
	fn from(value: u8) -> Self {
		Self::from(value as u128)
	}
}

impl<const N: usize> From<SmallU<N>> for M128 {
	fn from(value: SmallU<N>) -> Self {
		Self::from(value.val() as u128)
	}
}

impl From<uint8x16_t> for M128 {
	fn from(value: uint8x16_t) -> Self {
		Self(unsafe { vreinterpretq_u64_u8(value) })
	}
}
impl From<uint16x8_t> for M128 {
	fn from(value: uint16x8_t) -> Self {
		Self(unsafe { vreinterpretq_u64_u16(value) })
	}
}
impl From<uint32x4_t> for M128 {
	fn from(value: uint32x4_t) -> Self {
		Self(unsafe { vreinterpretq_u64_u32(value) })
	}
}
impl From<uint64x2_t> for M128 {
	fn from(value: uint64x2_t) -> Self {
		Self(value)
	}
}
impl From<poly8x16_t> for M128 {
	fn from(value: poly8x16_t) -> Self {
		Self(unsafe { vreinterpretq_u64_p8(value) })
	}
}
impl From<poly16x8_t> for M128 {
	fn from(value: poly16x8_t) -> Self {
		Self(unsafe { vreinterpretq_u64_p16(value) })
	}
}
impl From<poly64x2_t> for M128 {
	fn from(value: poly64x2_t) -> Self {
		Self(unsafe { vreinterpretq_u64_p64(value) })
	}
}

impl SerializeBytes for M128 {
	fn serialize(&self, mut write_buf: impl BufMut) -> Result<(), SerializationError> {
		assert_enough_space_for(&write_buf, std::mem::size_of::<Self>())?;

		write_buf.put_u128_le(u128::from(*self));

		Ok(())
	}
}

impl DeserializeBytes for M128 {
	fn deserialize(mut read_buf: impl Buf) -> Result<Self, SerializationError>
	where
		Self: Sized,
	{
		assert_enough_data_for(&read_buf, std::mem::size_of::<Self>())?;

		Ok(Self::from(read_buf.get_u128_le()))
	}
}

impl FixedSizeSerializeBytes for M128 {
	const BYTE_SIZE: usize = 16;
}

impl_divisible_bitmask!(M128, 1, 2, 4);

// Reflexive divisibility, needed when M128 is itself a field underlier (a width-1 packed field).
impl_divisible_self!(M128);

impl_divisible_memcast!(
	M128,
	u128 => |val| M128::from(val),
	u64 => |val| unsafe { M128(vdupq_n_u64(val)) },
	u32 => |val| unsafe { M128::from(vdupq_n_u32(val)) },
	u16 => |val| unsafe { M128::from(vdupq_n_u16(val)) },
	u8 => |val| unsafe { M128::from(vdupq_n_u8(val)) },
);

impl Not for M128 {
	type Output = Self;

	#[inline]
	fn not(self) -> Self::Output {
		unsafe { vmvnq_u8(self.into()).into() }
	}
}

impl BitAnd for M128 {
	type Output = Self;

	#[inline]
	fn bitand(self, rhs: Self) -> Self::Output {
		unsafe { vandq_u64(self.0, rhs.0).into() }
	}
}

impl BitAndAssign for M128 {
	fn bitand_assign(&mut self, rhs: Self) {
		*self = *self & rhs;
	}
}

impl BitOr for M128 {
	type Output = Self;

	#[inline]
	fn bitor(self, rhs: Self) -> Self::Output {
		unsafe { vorrq_u64(self.0, rhs.0).into() }
	}
}

impl BitOrAssign for M128 {
	fn bitor_assign(&mut self, rhs: Self) {
		*self = *self | rhs;
	}
}

impl BitXor for M128 {
	type Output = Self;

	#[inline]
	fn bitxor(self, rhs: Self) -> Self::Output {
		unsafe { veorq_u64(self.0, rhs.0).into() }
	}
}

impl BitXorAssign for M128 {
	fn bitxor_assign(&mut self, rhs: Self) {
		*self = *self ^ rhs;
	}
}

impl Shr<usize> for M128 {
	type Output = Self;

	#[inline]
	fn shr(self, rhs: usize) -> Self::Output {
		Self::from(u128::from(self) >> rhs)
	}
}

impl Shl<usize> for M128 {
	type Output = Self;

	#[inline]
	fn shl(self, rhs: usize) -> Self::Output {
		Self::from(u128::from(self) << rhs)
	}
}

impl Distribution<M128> for StandardUniform {
	fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> M128 {
		M128::from(rng.random::<u128>())
	}
}

impl std::fmt::Display for M128 {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		let data: u128 = (*self).into();
		write!(f, "{data:02X?}")
	}
}

impl std::fmt::Debug for M128 {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		write!(f, "M128({self})")
	}
}

impl Underlier for M128 {
	const LOG_BITS: usize = 7;
	const ZERO: Self = Self::from_u128(0);
	const ONE: Self = Self::from_u128(1);
	const ONES: Self = Self::from_u128(u128::MAX);

	#[inline]
	fn interleave(self, other: Self, log_block_len: usize) -> (Self, Self) {
		const MASKS: [M128; 3] = [
			M128::from_u128(interleave_mask_even!(u128, 0)),
			M128::from_u128(interleave_mask_even!(u128, 1)),
			M128::from_u128(interleave_mask_even!(u128, 2)),
		];

		unsafe {
			seq!(LOG_BLOCK_LEN in 0..=2 {
				if log_block_len == LOG_BLOCK_LEN {
					let (a, b) = (self.into(), other.into());
					let mask = MASKS[LOG_BLOCK_LEN].into();
					let t = vandq_u64(veorq_u64(vshrq_n_u64(a, 1 << LOG_BLOCK_LEN), b), mask);
					let c = veorq_u64(a, vshlq_n_u64(t, 1 << LOG_BLOCK_LEN));
					let d = veorq_u64(b, t);
					return (c.into(), d.into());
				}
			});
			match log_block_len {
				3 => {
					let (a, b) = (self.into(), other.into());
					let c = vtrn1q_u8(a, b);
					let d = vtrn2q_u8(a, b);
					(c.into(), d.into())
				}
				4 => {
					let (a, b) = (self.into(), other.into());
					let c = vtrn1q_u16(a, b);
					let d = vtrn2q_u16(a, b);
					(c.into(), d.into())
				}
				5 => {
					let (a, b) = (self.into(), other.into());
					let c = vtrn1q_u32(a, b);
					let d = vtrn2q_u32(a, b);
					(c.into(), d.into())
				}
				6 => {
					let (a, b) = (self.into(), other.into());
					let c = vtrn1q_u64(a, b);
					let d = vtrn2q_u64(a, b);
					(c.into(), d.into())
				}
				_ => panic!("Unsupported block length"),
			}
		}
	}

	#[inline]
	fn transpose(self, other: Self, log_block_len: usize) -> (Self, Self) {
		unsafe {
			match log_block_len {
				0..=3 => {
					let (a, b) = (self.into(), other.into());
					let (mut a, mut b) = (Self::from(vuzp1q_u8(a, b)), Self::from(vuzp2q_u8(a, b)));

					for log_block_len in (log_block_len..3).rev() {
						(a, b) = a.interleave(b, log_block_len);
					}

					(a, b)
				}
				4 => {
					let (a, b) = (self.into(), other.into());
					(vuzp1q_u16(a, b).into(), vuzp2q_u16(a, b).into())
				}
				5 => {
					let (a, b) = (self.into(), other.into());
					(vuzp1q_u32(a, b).into(), vuzp2q_u32(a, b).into())
				}
				6 => {
					let (a, b) = (self.into(), other.into());
					(vuzp1q_u64(a, b).into(), vuzp2q_u64(a, b).into())
				}
				_ => panic!("Unsupported block length"),
			}
		}
	}
}

impl<Scalar: BinaryField> From<u128> for PackedPrimitiveType<M128, Scalar> {
	fn from(value: u128) -> Self {
		Self::from(M128::from(value))
	}
}

impl<Scalar: BinaryField> From<PackedPrimitiveType<M128, Scalar>> for u128 {
	fn from(value: PackedPrimitiveType<M128, Scalar>) -> Self {
		value.to_underlier().into()
	}
}

#[cfg(test)]
mod tests {
	use binius_utils::bytes::BytesMut;

	use super::*;

	#[test]
	fn test_serialize_and_deserialize_m128() {
		let mut rng = StdRng::from_seed([0; 32]);

		let original_value = M128::from(rng.random::<u128>());

		let mut buf = BytesMut::new();
		original_value.serialize(&mut buf).unwrap();

		let deserialized_value = M128::deserialize(buf.freeze()).unwrap();

		assert_eq!(original_value, deserialized_value);
	}
}
