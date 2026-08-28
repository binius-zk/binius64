// Copyright 2024-2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use std::{
	arch::x86_64::*,
	mem,
	ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not, Shl, Shr},
};

use binius_utils::{
	DeserializeBytes, FixedSizeSerializeBytes, SerializationError, SerializeBytes,
	bytes::{Buf, BufMut},
	serialization::{assert_enough_data_for, assert_enough_space_for},
};
use bytemuck::{Pod, Zeroable};
use rand::{
	distr::{Distribution, StandardUniform},
	prelude::*,
};
use seq_macro::seq;

use crate::{
	BinaryField,
	divisible::{Divisible, impl_divisible_memcast, impl_divisible_self},
	packed_fields::primitive::PackedPrimitiveType,
	underlier::{SmallU, Underlier, impl_divisible_bitmask},
};

pub const fn m128i_from_u128(x: u128) -> __m128i {
	// Static assertion that u128 and __m128i have equal alignment
	let _: [(); align_of::<u128>()] = [(); align_of::<__m128i>()];
	unsafe { mem::transmute(x) }
}

/// 128-bit value that is used for 128-bit SIMD operations
#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct M128(pub(super) __m128i);

impl M128 {
	#[inline(always)]
	pub const fn from_u128(val: u128) -> Self {
		Self(m128i_from_u128(val))
	}
}

impl From<__m128i> for M128 {
	#[inline(always)]
	fn from(value: __m128i) -> Self {
		Self(value)
	}
}

impl From<u128> for M128 {
	fn from(value: u128) -> Self {
		Self(m128i_from_u128(value))
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

impl From<M128> for u128 {
	fn from(value: M128) -> Self {
		const {
			assert!(
				align_of::<u128>() == 16,
				"the store below needs a 16-byte aligned destination"
			);
		}
		let mut result = 0u128;
		unsafe {
			// Safety: u128 is 16-byte aligned, as the const assertion above checks.
			_mm_store_si128(&raw mut result as *mut __m128i, value.0);
		};
		result
	}
}

impl From<M128> for __m128i {
	#[inline(always)]
	fn from(value: M128) -> Self {
		value.0
	}
}

impl SerializeBytes for M128 {
	fn serialize(&self, mut write_buf: impl BufMut) -> Result<(), SerializationError> {
		assert_enough_space_for(&write_buf, std::mem::size_of::<Self>())?;

		let raw_value: u128 = (*self).into();

		write_buf.put_u128_le(raw_value);
		Ok(())
	}
}

impl DeserializeBytes for M128 {
	fn deserialize(mut read_buf: impl Buf) -> Result<Self, SerializationError>
	where
		Self: Sized,
	{
		assert_enough_data_for(&read_buf, std::mem::size_of::<Self>())?;

		let raw_value = read_buf.get_u128_le();

		Ok(Self::from(raw_value))
	}
}

impl FixedSizeSerializeBytes for M128 {
	const BYTE_SIZE: usize = 16;
}

impl_divisible_bitmask!(M128, 1, 2, 4);

impl Default for M128 {
	#[inline(always)]
	fn default() -> Self {
		Self(unsafe { _mm_setzero_si128() })
	}
}

impl BitAnd for M128 {
	type Output = Self;

	#[inline(always)]
	fn bitand(self, rhs: Self) -> Self::Output {
		Self(unsafe { _mm_and_si128(self.0, rhs.0) })
	}
}

impl BitAndAssign for M128 {
	#[inline(always)]
	fn bitand_assign(&mut self, rhs: Self) {
		*self = *self & rhs;
	}
}

impl BitOr for M128 {
	type Output = Self;

	#[inline(always)]
	fn bitor(self, rhs: Self) -> Self::Output {
		Self(unsafe { _mm_or_si128(self.0, rhs.0) })
	}
}

impl BitOrAssign for M128 {
	#[inline(always)]
	fn bitor_assign(&mut self, rhs: Self) {
		*self = *self | rhs;
	}
}

impl BitXor for M128 {
	type Output = Self;

	#[inline(always)]
	fn bitxor(self, rhs: Self) -> Self::Output {
		Self(unsafe { _mm_xor_si128(self.0, rhs.0) })
	}
}

impl BitXorAssign for M128 {
	#[inline(always)]
	fn bitxor_assign(&mut self, rhs: Self) {
		*self = *self ^ rhs;
	}
}

impl Not for M128 {
	type Output = Self;

	fn not(self) -> Self::Output {
		const ONES: M128 = M128::from_u128(u128::MAX);

		self ^ ONES
	}
}

/// `std::cmp::max` isn't const, so we need our own implementation
const fn max_i32(left: i32, right: i32) -> i32 {
	if left > right { left } else { right }
}

/// This solution shows 4X better performance.
/// We have to use macro because parameter `count` in _mm_slli_epi64/_mm_srli_epi64 should be passed
/// as constant and Rust currently doesn't allow passing expressions (`count - 64`) where variable
/// is a generic constant parameter. Source: <https://stackoverflow.com/questions/34478328/the-best-way-to-shift-a-m128i/34482688#34482688>
macro_rules! bitshift_128b {
	($val:expr, $shift:ident, $byte_shift:ident, $bit_shift_64:ident, $bit_shift_64_opposite:ident, $or:ident) => {
		unsafe {
			let carry = $byte_shift($val, 8);
			seq!(N in 64..128 {
				if $shift == N {
					return $bit_shift_64(
						carry,
						crate::arch::x86_64::m128::max_i32((N - 64) as i32, 0) as _,
					).into();
				}
			});
			seq!(N in 0..64 {
				if $shift == N {
					let carry = $bit_shift_64_opposite(
						carry,
						crate::arch::x86_64::m128::max_i32((64 - N) as i32, 0) as _,
					);

					let val = $bit_shift_64($val, N);
					return $or(val, carry).into();
				}
			});

			return Default::default()
		}
	};
}

impl Shr<usize> for M128 {
	type Output = Self;

	#[inline(always)]
	fn shr(self, rhs: usize) -> Self::Output {
		// This implementation is effective when `rhs` is known at compile-time.
		// In our code this is always the case.
		bitshift_128b!(self.0, rhs, _mm_bsrli_si128, _mm_srli_epi64, _mm_slli_epi64, _mm_or_si128)
	}
}

impl Shl<usize> for M128 {
	type Output = Self;

	#[inline(always)]
	fn shl(self, rhs: usize) -> Self::Output {
		// This implementation is effective when `rhs` is known at compile-time.
		// In our code this is always the case.
		bitshift_128b!(self.0, rhs, _mm_bslli_si128, _mm_slli_epi64, _mm_srli_epi64, _mm_or_si128);
	}
}

impl PartialEq for M128 {
	fn eq(&self, other: &Self) -> bool {
		unsafe {
			let neq = _mm_xor_si128(self.0, other.0);
			_mm_test_all_zeros(neq, neq) == 1
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

impl Distribution<M128> for StandardUniform {
	fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> M128 {
		M128(rng.random())
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
	const ZERO: Self = { Self::from_u128(0) };
	const ONE: Self = { Self::from_u128(1) };
	const ONES: Self = { Self::from_u128(u128::MAX) };

	#[inline(always)]
	fn interleave(self, other: Self, log_block_len: usize) -> (Self, Self) {
		unsafe {
			let (c, d) = interleave_bits(
				Into::<Self>::into(self).into(),
				Into::<Self>::into(other).into(),
				log_block_len,
			);
			(Self::from(c), Self::from(d))
		}
	}
}

unsafe impl Zeroable for M128 {}

unsafe impl Pod for M128 {}

unsafe impl Send for M128 {}

unsafe impl Sync for M128 {}

impl<Scalar: BinaryField> From<__m128i> for PackedPrimitiveType<M128, Scalar> {
	fn from(value: __m128i) -> Self {
		M128::from(value).into()
	}
}

impl<Scalar: BinaryField> From<u128> for PackedPrimitiveType<M128, Scalar> {
	fn from(value: u128) -> Self {
		M128::from(value).into()
	}
}

impl<Scalar: BinaryField> From<PackedPrimitiveType<M128, Scalar>> for __m128i {
	fn from(value: PackedPrimitiveType<M128, Scalar>) -> Self {
		value.to_underlier().into()
	}
}

#[inline]
unsafe fn interleave_bits(a: __m128i, b: __m128i, log_block_len: usize) -> (__m128i, __m128i) {
	match log_block_len {
		0 => unsafe {
			let mask = _mm_set1_epi8(0x55i8);
			interleave_bits_imm::<1>(a, b, mask)
		},
		1 => unsafe {
			let mask = _mm_set1_epi8(0x33i8);
			interleave_bits_imm::<2>(a, b, mask)
		},
		2 => unsafe {
			let mask = _mm_set1_epi8(0x0fi8);
			interleave_bits_imm::<4>(a, b, mask)
		},
		3 => unsafe {
			let shuffle = _mm_set_epi8(15, 13, 11, 9, 7, 5, 3, 1, 14, 12, 10, 8, 6, 4, 2, 0);
			let a = _mm_shuffle_epi8(a, shuffle);
			let b = _mm_shuffle_epi8(b, shuffle);
			let a_prime = _mm_unpacklo_epi8(a, b);
			let b_prime = _mm_unpackhi_epi8(a, b);
			(a_prime, b_prime)
		},
		4 => unsafe {
			let shuffle = _mm_set_epi8(15, 14, 11, 10, 7, 6, 3, 2, 13, 12, 9, 8, 5, 4, 1, 0);
			let a = _mm_shuffle_epi8(a, shuffle);
			let b = _mm_shuffle_epi8(b, shuffle);
			let a_prime = _mm_unpacklo_epi16(a, b);
			let b_prime = _mm_unpackhi_epi16(a, b);
			(a_prime, b_prime)
		},
		5 => unsafe {
			let shuffle = _mm_set_epi8(15, 14, 13, 12, 7, 6, 5, 4, 11, 10, 9, 8, 3, 2, 1, 0);
			let a = _mm_shuffle_epi8(a, shuffle);
			let b = _mm_shuffle_epi8(b, shuffle);
			let a_prime = _mm_unpacklo_epi32(a, b);
			let b_prime = _mm_unpackhi_epi32(a, b);
			(a_prime, b_prime)
		},
		6 => unsafe {
			let a_prime = _mm_unpacklo_epi64(a, b);
			let b_prime = _mm_unpackhi_epi64(a, b);
			(a_prime, b_prime)
		},
		_ => panic!("unsupported block length"),
	}
}

#[inline]
unsafe fn interleave_bits_imm<const BLOCK_LEN: i32>(
	a: __m128i,
	b: __m128i,
	mask: __m128i,
) -> (__m128i, __m128i) {
	unsafe {
		let t = _mm_and_si128(_mm_xor_si128(_mm_srli_epi64::<BLOCK_LEN>(a), b), mask);
		let a_prime = _mm_xor_si128(a, _mm_slli_epi64::<BLOCK_LEN>(t));
		let b_prime = _mm_xor_si128(b, t);
		(a_prime, b_prime)
	}
}

// Reflexive divisibility, needed when M128 is itself a field underlier (a width-1 packed field).
impl_divisible_self!(M128);

impl_divisible_memcast!(
	M128,
	u128 => |val| M128::from(val),
	u64 => |val| unsafe { M128(_mm_set1_epi64x(val as i64)) },
	u32 => |val| unsafe { M128(_mm_set1_epi32(val as i32)) },
	u16 => |val| unsafe { M128(_mm_set1_epi16(val as i16)) },
	u8 => |val| unsafe { M128(_mm_set1_epi8(val as i8)) },
);

#[cfg(test)]
mod tests {
	use binius_utils::bytes::BytesMut;
	use proptest::{arbitrary::any, proptest};
	use rand::prelude::*;

	use super::*;

	fn check_roundtrip<T>(val: M128)
	where
		T: From<M128>,
		M128: From<T>,
	{
		assert_eq!(M128::from(T::from(val)), val);
	}

	#[test]
	fn test_constants() {
		assert_eq!(M128::default(), M128::ZERO);
		assert_eq!(M128::from(0u128), M128::ZERO);
		assert_eq!(M128::from(1u128), M128::ONE);
	}

	fn get(value: M128, log_block_len: usize, index: usize) -> M128 {
		(value >> (index << log_block_len)) & M128::from(1u128 << log_block_len)
	}

	proptest! {
		#[test]
		fn test_conversion(a in any::<u128>()) {
			check_roundtrip::<u128>(a.into());
			check_roundtrip::<__m128i>(a.into());
		}

		#[test]
		fn test_binary_bit_operations(a in any::<u128>(), b in any::<u128>()) {
			assert_eq!(M128::from(a & b), M128::from(a) & M128::from(b));
			assert_eq!(M128::from(a | b), M128::from(a) | M128::from(b));
			assert_eq!(M128::from(a ^ b), M128::from(a) ^ M128::from(b));
		}

		#[test]
		fn test_negate(a in any::<u128>()) {
			assert_eq!(M128::from(!a), !M128::from(a));
		}

		#[test]
		fn test_shifts(a in any::<u128>(), b in 0..128usize) {
			assert_eq!(M128::from(a << b), M128::from(a) << b);
			assert_eq!(M128::from(a >> b), M128::from(a) >> b);
		}

		#[test]
		fn test_interleave_bits(a in any::<u128>(), b in any::<u128>(), height in 0usize..7) {
			let a = M128::from(a);
			let b = M128::from(b);

			let (c, d) = unsafe {interleave_bits(a.0, b.0, height)};
			let (c, d) = (M128::from(c), M128::from(d));

			for i in (0..128>>height).step_by(2) {
				assert_eq!(get(c, height, i), get(a, height, i));
				assert_eq!(get(c, height, i+1), get(b, height, i));
				assert_eq!(get(d, height, i), get(a, height, i+1));
				assert_eq!(get(d, height, i+1), get(b, height, i+1));
			}
		}
	}

	#[test]
	fn test_eq() {
		let a = M128::from(0u128);
		let b = M128::from(42u128);
		let c = M128::from(u128::MAX);

		assert_eq!(a, a);
		assert_eq!(b, b);
		assert_eq!(c, c);

		assert_ne!(a, b);
		assert_ne!(a, c);
		assert_ne!(b, c);
	}

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
