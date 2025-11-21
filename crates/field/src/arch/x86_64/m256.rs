// Copyright 2024-2025 Irreducible Inc.

use std::{
	arch::x86_64::*,
	mem::transmute,
	ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not, Shl, Shr},
};

use binius_utils::{
	DeserializeBytes, SerializationError, SerializeBytes,
	bytes::{Buf, BufMut},
	serialization::{assert_enough_data_for, assert_enough_space_for},
};
use bytemuck::{Pod, Zeroable, must_cast};
use cfg_if::cfg_if;
use rand::{
	Rng,
	distr::{Distribution, StandardUniform},
};
use seq_macro::seq;

use crate::{
	BinaryField,
	arch::{
		binary_utils::{as_array_mut, as_array_ref, make_func_to_i8},
		portable::{
			packed::{PackedPrimitiveType, impl_pack_scalar},
			packed_arithmetic::{
				UnderlierWithBitConstants, interleave_mask_even, interleave_mask_odd,
			},
		},
		x86_64::m128::bitshift_128b,
	},
	arithmetic_traits::Broadcast,
	underlier::{
		NumCast, SmallU, U1, U2, U4, UnderlierType, UnderlierWithBitOps, WithUnderlier,
		get_block_values, get_spread_bytes, impl_divisible, impl_iteration, spread_fallback,
		unpack_hi_128b_fallback, unpack_lo_128b_fallback,
	},
};

/// 256-bit value that is used for 256-bit SIMD operations
#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct M256(pub(super) __m256i);

impl M256 {
	pub const fn from_equal_u128s(val: u128) -> Self {
		unsafe { transmute([val, val]) }
	}
}

impl From<__m256i> for M256 {
	#[inline(always)]
	fn from(value: __m256i) -> Self {
		Self(value)
	}
}

impl From<[u128; 2]> for M256 {
	fn from(value: [u128; 2]) -> Self {
		Self(unsafe {
			_mm256_set_epi64x(
				(value[1] >> 64) as i64,
				value[1] as i64,
				(value[0] >> 64) as i64,
				value[0] as i64,
			)
		})
	}
}

impl From<u128> for M256 {
	fn from(value: u128) -> Self {
		Self::from([value, 0])
	}
}

impl From<u64> for M256 {
	fn from(value: u64) -> Self {
		Self::from(value as u128)
	}
}

impl From<u32> for M256 {
	fn from(value: u32) -> Self {
		Self::from(value as u128)
	}
}

impl From<u16> for M256 {
	fn from(value: u16) -> Self {
		Self::from(value as u128)
	}
}

impl From<u8> for M256 {
	fn from(value: u8) -> Self {
		Self::from(value as u128)
	}
}

impl<const N: usize> From<SmallU<N>> for M256 {
	fn from(value: SmallU<N>) -> Self {
		Self::from(value.val() as u128)
	}
}

impl From<M256> for [u128; 2] {
	fn from(value: M256) -> Self {
		let result: [u128; 2] = unsafe { transmute(value.0) };

		result
	}
}

impl From<M256> for __m256i {
	#[inline(always)]
	fn from(value: M256) -> Self {
		value.0
	}
}

impl SerializeBytes for M256 {
	fn serialize(&self, mut write_buf: impl BufMut) -> Result<(), SerializationError> {
		assert_enough_space_for(&write_buf, std::mem::size_of::<Self>())?;

		let raw_values: [u128; 2] = (*self).into();

		for &val in &raw_values {
			write_buf.put_u128_le(val);
		}

		Ok(())
	}
}

impl DeserializeBytes for M256 {
	fn deserialize(mut read_buf: impl Buf) -> Result<Self, SerializationError>
	where
		Self: Sized,
	{
		assert_enough_data_for(&read_buf, size_of::<Self>())?;

		let raw_values = [read_buf.get_u128_le(), read_buf.get_u128_le()];

		Ok(Self::from(raw_values))
	}
}

impl_divisible!(@pairs M256, M128, u128, u64, u32, u16, u8);
impl_pack_scalar!(M256);

impl<U: NumCast<u128>> NumCast<M256> for U {
	#[inline(always)]
	fn num_cast_from(val: M256) -> Self {
		let [low, _high] = val.into();
		Self::num_cast_from(low)
	}
}

impl Default for M256 {
	#[inline(always)]
	fn default() -> Self {
		Self(unsafe { _mm256_setzero_si256() })
	}
}

impl BitAnd for M256 {
	type Output = Self;

	#[inline(always)]
	fn bitand(self, rhs: Self) -> Self::Output {
		Self(unsafe { _mm256_and_si256(self.0, rhs.0) })
	}
}

impl BitAndAssign for M256 {
	#[inline(always)]
	fn bitand_assign(&mut self, rhs: Self) {
		*self = *self & rhs
	}
}

impl BitOr for M256 {
	type Output = Self;

	#[inline(always)]
	fn bitor(self, rhs: Self) -> Self::Output {
		Self(unsafe { _mm256_or_si256(self.0, rhs.0) })
	}
}

impl BitOrAssign for M256 {
	#[inline(always)]
	fn bitor_assign(&mut self, rhs: Self) {
		*self = *self | rhs
	}
}

impl BitXor for M256 {
	type Output = Self;

	#[inline(always)]
	fn bitxor(self, rhs: Self) -> Self::Output {
		Self(unsafe { _mm256_xor_si256(self.0, rhs.0) })
	}
}

impl BitXorAssign for M256 {
	#[inline(always)]
	fn bitxor_assign(&mut self, rhs: Self) {
		*self = *self ^ rhs;
	}
}

impl Not for M256 {
	type Output = Self;

	#[inline(always)]
	fn not(self) -> Self::Output {
		const ONES: __m256i = m256_from_u128s!(u128::MAX, u128::MAX,);

		self ^ Self(ONES)
	}
}

impl Shr<usize> for M256 {
	type Output = Self;

	/// TODO: this is inefficient implementation
	#[inline(always)]
	fn shr(self, rhs: usize) -> Self::Output {
		match rhs {
			rhs if rhs >= 256 => Self::ZERO,
			0 => self,
			rhs => {
				let [mut low, mut high]: [u128; 2] = self.into();
				if rhs >= 128 {
					low = high >> (rhs - 128);
					high = 0;
				} else {
					low = (low >> rhs) + (high << (128usize - rhs));
					high >>= rhs
				}
				[low, high].into()
			}
		}
	}
}
impl Shl<usize> for M256 {
	type Output = Self;

	/// TODO: this is inefficient implementation
	#[inline(always)]
	fn shl(self, rhs: usize) -> Self::Output {
		match rhs {
			rhs if rhs >= 256 => Self::ZERO,
			0 => self,
			rhs => {
				let [mut low, mut high]: [u128; 2] = self.into();
				if rhs >= 128 {
					high = low << (rhs - 128);
					low = 0;
				} else {
					high = (high << rhs) + (low >> (128usize - rhs));
					low <<= rhs
				}
				[low, high].into()
			}
		}
	}
}

impl PartialEq for M256 {
	#[inline(always)]
	fn eq(&self, other: &Self) -> bool {
		unsafe {
			let pcmp = _mm256_cmpeq_epi32(self.0, other.0);
			let bitmask = _mm256_movemask_epi8(pcmp) as u32;
			bitmask == 0xffffffff
		}
	}
}

impl Eq for M256 {}

impl PartialOrd for M256 {
	fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
		Some(self.cmp(other))
	}
}

impl Ord for M256 {
	fn cmp(&self, other: &Self) -> std::cmp::Ordering {
		<[u128; 2]>::from(*self).cmp(&<[u128; 2]>::from(*other))
	}
}

impl Distribution<M256> for StandardUniform {
	fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> M256 {
		let val: [u128; 2] = rng.random();
		val.into()
	}
}

impl std::fmt::Display for M256 {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		let data: [u128; 2] = (*self).into();
		write!(f, "{data:02X?}")
	}
}

impl std::fmt::Debug for M256 {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		write!(f, "M256({self})")
	}
}

#[repr(align(32))]
pub struct AlignedData(pub [u128; 2]);

macro_rules! m256_from_u128s {
    ($($values:expr,)+) => {{
        let aligned_data = $crate::arch::x86_64::m256::AlignedData([$($values,)*]);
        unsafe {* (aligned_data.0.as_ptr() as *const __m256i)}
    }};
}

pub(super) use m256_from_u128s;

use super::m128::M128;

impl UnderlierType for M256 {
	const LOG_BITS: usize = 8;
}

impl UnderlierWithBitOps for M256 {
	const ZERO: Self = { Self(m256_from_u128s!(0, 0,)) };
	const ONE: Self = { Self(m256_from_u128s!(1, 0,)) };
	const ONES: Self = { Self(m256_from_u128s!(u128::MAX, u128::MAX,)) };

	#[inline]
	fn fill_with_bit(val: u8) -> Self {
		Self(unsafe { _mm256_set1_epi8(val.wrapping_neg() as i8) })
	}

	#[inline(always)]
	fn from_fn<T>(mut f: impl FnMut(usize) -> T) -> Self
	where
		T: UnderlierType,
		Self: From<T>,
	{
		match T::BITS {
			1 | 2 | 4 => {
				let mut f = make_func_to_i8::<T, Self>(f);

				unsafe {
					_mm256_set_epi8(
						f(31),
						f(30),
						f(29),
						f(28),
						f(27),
						f(26),
						f(25),
						f(24),
						f(23),
						f(22),
						f(21),
						f(20),
						f(19),
						f(18),
						f(17),
						f(16),
						f(15),
						f(14),
						f(13),
						f(12),
						f(11),
						f(10),
						f(9),
						f(8),
						f(7),
						f(6),
						f(5),
						f(4),
						f(3),
						f(2),
						f(1),
						f(0),
					)
				}
			}
			8 => {
				let mut f = |i| u8::num_cast_from(Self::from(f(i))) as i8;
				unsafe {
					_mm256_set_epi8(
						f(31),
						f(30),
						f(29),
						f(28),
						f(27),
						f(26),
						f(25),
						f(24),
						f(23),
						f(22),
						f(21),
						f(20),
						f(19),
						f(18),
						f(17),
						f(16),
						f(15),
						f(14),
						f(13),
						f(12),
						f(11),
						f(10),
						f(9),
						f(8),
						f(7),
						f(6),
						f(5),
						f(4),
						f(3),
						f(2),
						f(1),
						f(0),
					)
				}
			}
			16 => {
				let mut f = |i| u16::num_cast_from(Self::from(f(i))) as i16;
				unsafe {
					_mm256_set_epi16(
						f(15),
						f(14),
						f(13),
						f(12),
						f(11),
						f(10),
						f(9),
						f(8),
						f(7),
						f(6),
						f(5),
						f(4),
						f(3),
						f(2),
						f(1),
						f(0),
					)
				}
			}
			32 => {
				let mut f = |i| u32::num_cast_from(Self::from(f(i))) as i32;
				unsafe { _mm256_set_epi32(f(7), f(6), f(5), f(4), f(3), f(2), f(1), f(0)) }
			}
			64 => {
				let mut f = |i| u64::num_cast_from(Self::from(f(i))) as i64;
				unsafe { _mm256_set_epi64x(f(3), f(2), f(1), f(0)) }
			}
			128 => {
				let mut f = |i| M128::from(u128::num_cast_from(Self::from(f(i)))).0;
				unsafe { _mm256_set_m128i(f(1), f(0)) }
			}
			_ => panic!("unsupported bit count"),
		}
		.into()
	}

	#[inline(always)]
	unsafe fn get_subvalue<T>(&self, i: usize) -> T
	where
		T: UnderlierType + NumCast<Self>,
	{
		match T::BITS {
			1 | 2 | 4 => {
				let elements_in_8 = 8 / T::BITS;
				let mut value_u8 = as_array_ref::<_, u8, 32, _>(self, |arr| unsafe {
					*arr.get_unchecked(i / elements_in_8)
				});

				let shift = (i % elements_in_8) * T::BITS;
				value_u8 >>= shift;

				T::from_underlier(T::num_cast_from(Self::from(value_u8)))
			}
			8 => {
				let value_u8 =
					as_array_ref::<_, u8, 32, _>(self, |arr| unsafe { *arr.get_unchecked(i) });
				T::from_underlier(T::num_cast_from(Self::from(value_u8)))
			}
			16 => {
				let value_u16 =
					as_array_ref::<_, u16, 16, _>(self, |arr| unsafe { *arr.get_unchecked(i) });
				T::from_underlier(T::num_cast_from(Self::from(value_u16)))
			}
			32 => {
				let value_u32 =
					as_array_ref::<_, u32, 8, _>(self, |arr| unsafe { *arr.get_unchecked(i) });
				T::from_underlier(T::num_cast_from(Self::from(value_u32)))
			}
			64 => {
				let value_u64 =
					as_array_ref::<_, u64, 4, _>(self, |arr| unsafe { *arr.get_unchecked(i) });
				T::from_underlier(T::num_cast_from(Self::from(value_u64)))
			}
			128 => {
				let value_u128 =
					as_array_ref::<_, u128, 2, _>(self, |arr| unsafe { *arr.get_unchecked(i) });
				T::from_underlier(T::num_cast_from(Self::from(value_u128)))
			}
			_ => panic!("unsupported bit count"),
		}
	}

	#[inline(always)]
	unsafe fn set_subvalue<T>(&mut self, i: usize, val: T)
	where
		T: UnderlierWithBitOps,
		Self: From<T>,
	{
		match T::BITS {
			1 | 2 | 4 => {
				let elements_in_8 = 8 / T::BITS;
				let mask = (1u8 << T::BITS) - 1;
				let shift = (i % elements_in_8) * T::BITS;
				let val = u8::num_cast_from(Self::from(val)) << shift;
				let mask = mask << shift;

				as_array_mut::<_, u8, 32>(self, |array| unsafe {
					let element = array.get_unchecked_mut(i / elements_in_8);
					*element &= !mask;
					*element |= val;
				});
			}
			8 => as_array_mut::<_, u8, 32>(self, |array| unsafe {
				*array.get_unchecked_mut(i) = u8::num_cast_from(Self::from(val));
			}),
			16 => as_array_mut::<_, u16, 16>(self, |array| unsafe {
				*array.get_unchecked_mut(i) = u16::num_cast_from(Self::from(val));
			}),
			32 => as_array_mut::<_, u32, 8>(self, |array| unsafe {
				*array.get_unchecked_mut(i) = u32::num_cast_from(Self::from(val));
			}),
			64 => as_array_mut::<_, u64, 4>(self, |array| unsafe {
				*array.get_unchecked_mut(i) = u64::num_cast_from(Self::from(val));
			}),
			128 => as_array_mut::<_, u128, 2>(self, |array| unsafe {
				*array.get_unchecked_mut(i) = u128::num_cast_from(Self::from(val));
			}),
			_ => panic!("unsupported bit count"),
		}
	}

	#[inline(always)]
	unsafe fn spread<T>(self, log_block_len: usize, block_idx: usize) -> Self
	where
		T: UnderlierWithBitOps + NumCast<Self>,
		Self: From<T>,
	{
		match T::LOG_BITS {
			0 => match log_block_len {
				0 => unsafe {
					let bits = get_block_values::<_, U1, 1>(self, block_idx)[0];
					Self::fill_with_bit(bits.val())
				},
				1 => unsafe {
					let bits = get_block_values::<_, U1, 2>(self, block_idx);
					let values: [u64; 2] = bits.map(|b| u64::fill_with_bit(b.val()));
					Self::from_fn::<u64>(|i| values[i / 2])
				},
				2 => unsafe {
					let bits = get_block_values::<_, U1, 4>(self, block_idx);
					Self::from_fn::<u64>(|i| u64::fill_with_bit(bits[i].val()))
				},
				3 => unsafe {
					let bits = get_block_values::<_, U1, 8>(self, block_idx);
					Self::from_fn::<u32>(|i| u32::fill_with_bit(bits[i].val()))
				},
				4 => unsafe {
					let bits = get_block_values::<_, U1, 16>(self, block_idx);
					Self::from_fn::<u16>(|i| u16::fill_with_bit(bits[i].val()))
				},
				5 => unsafe {
					let bits = get_block_values::<_, U1, 32>(self, block_idx);
					Self::from_fn::<u8>(|i| u8::fill_with_bit(bits[i].val()))
				},
				_ => unsafe { spread_fallback(self, log_block_len, block_idx) },
			},
			1 => match log_block_len {
				0 => unsafe {
					let byte = get_spread_bytes::<_, U2, 1>(self, block_idx)[0];

					_mm256_set1_epi8(byte as _).into()
				},
				1 => unsafe {
					let bytes = get_spread_bytes::<_, U2, 2>(self, block_idx);

					Self::from_fn::<u8>(|i| bytes[i / 16])
				},
				2 => unsafe {
					let bytes = get_spread_bytes::<_, U2, 4>(self, block_idx);

					Self::from_fn::<u8>(|i| bytes[i / 8])
				},
				3 => unsafe {
					let bytes = get_spread_bytes::<_, U2, 8>(self, block_idx);

					Self::from_fn::<u8>(|i| bytes[i / 4])
				},
				4 => unsafe {
					let bytes = get_spread_bytes::<_, U2, 16>(self, block_idx);

					Self::from_fn::<u8>(|i| bytes[i / 2])
				},
				5 => unsafe {
					let bytes = get_spread_bytes::<_, U2, 32>(self, block_idx);

					Self::from_fn::<u8>(|i| bytes[i])
				},
				_ => unsafe { spread_fallback(self, log_block_len, block_idx) },
			},
			2 => match log_block_len {
				0 => unsafe {
					let byte = get_spread_bytes::<_, U4, 1>(self, block_idx)[0];

					_mm256_set1_epi8(byte as _).into()
				},
				1 => unsafe {
					let bytes = get_spread_bytes::<_, U4, 2>(self, block_idx);

					Self::from_fn::<u8>(|i| bytes[i / 16])
				},
				2 => unsafe {
					let bytes = get_spread_bytes::<_, U4, 4>(self, block_idx);

					Self::from_fn::<u8>(|i| bytes[i / 8])
				},
				3 => unsafe {
					let bytes = get_spread_bytes::<_, U4, 8>(self, block_idx);

					Self::from_fn::<u8>(|i| bytes[i / 4])
				},
				4 => unsafe {
					let bytes = get_spread_bytes::<_, U4, 16>(self, block_idx);

					Self::from_fn::<u8>(|i| bytes[i / 2])
				},
				5 => unsafe {
					let bytes = get_spread_bytes::<_, U4, 32>(self, block_idx);

					Self::from_fn::<u8>(|i| bytes[i])
				},
				_ => unsafe { spread_fallback(self, log_block_len, block_idx) },
			},
			3 => {
				cfg_if! {
					if #[cfg(target_feature = "avx512f")] {
						match log_block_len {
							0 => unsafe {
								let byte = get_block_values::<_, u8, 1>(self, block_idx)[0];
								_mm256_set1_epi8(byte as _).into()
							}
							1 => unsafe {
								_mm256_permutexvar_epi8(
									LOG_B8_1[block_idx],
									self.0,
								).into()
							}
							2 => unsafe {
								_mm256_permutexvar_epi8(
									LOG_B8_2[block_idx],
									self.0,
								).into()
							}
							3 => unsafe {
								_mm256_permutexvar_epi8(
									LOG_B8_3[block_idx],
									self.0,
								).into()
							}
							4 => unsafe {
								_mm256_permutexvar_epi8(
									LOG_B8_4[block_idx],
									self.0,
								).into()
							}
							5 => self,
							_ => panic!("unsupported block length"),
						}
					} else {
						match log_block_len {
							0 => unsafe {
								let byte = get_block_values::<_, u8, 1>(self, block_idx)[0];
								_mm256_set1_epi8(byte as _).into()
							}
							1 => unsafe {
								let bytes = get_block_values::<_, u8, 2>(self, block_idx);
								Self::from_fn::<u8>(|i| bytes[i / 16])
							}
							2 => unsafe {
								let bytes = get_block_values::<_, u8, 4>(self, block_idx);
								Self::from_fn::<u8>(|i| bytes[i / 8])
							}
							3 => unsafe {
								let bytes = get_block_values::<_, u8, 8>(self, block_idx);
								Self::from_fn::<u8>(|i| bytes[i / 4])
							}
							4 => unsafe {
								let bytes = get_block_values::<_, u8, 16>(self, block_idx);
								Self::from_fn::<u8>(|i| bytes[i / 2])
							}
							5 => self,
							_ => panic!("unsupported block length"),
						}
					}
				}
			}
			4 => {
				cfg_if! {
					if #[cfg(target_feature = "avx512f")] {
						match log_block_len {
							0 => unsafe {
								let value = get_block_values::<_, u16, 1>(self, block_idx)[0];
								_mm256_set1_epi16(value as _).into()
							}
							1 => unsafe {
								_mm256_permutexvar_epi8(
									LOG_B16_1[block_idx],
									self.0,
								).into()
							}
							2 => unsafe {
								_mm256_permutexvar_epi8(
									LOG_B16_2[block_idx],
									self.0,
								).into()
							}
							3 => unsafe {
								_mm256_permutexvar_epi8(
									LOG_B16_3[block_idx],
									self.0,
								).into()
							}
							4 => self,
							_ => panic!("unsupported block length"),
						}
					} else {
						match log_block_len {
							0 => unsafe {
								let value = get_block_values::<_, u16, 1>(self, block_idx)[0];
								_mm256_set1_epi16(value as _).into()
							}
							1 => unsafe {
								let values = get_block_values::<_, u16, 2>(self, block_idx);
								Self::from_fn::<u16>(|i| values[i / 8])
							}
							2 => unsafe {
								let values = get_block_values::<_, u16, 4>(self, block_idx);
								Self::from_fn::<u16>(|i| values[i / 4])
							}
							3 => unsafe {
								let values = get_block_values::<_, u16, 8>(self, block_idx);
								Self::from_fn::<u16>(|i| values[i / 2])
							}
							4 => self,
							_ => panic!("unsupported block length"),
						}
					}
				}
			}
			5 => {
				cfg_if! {
					if #[cfg(target_feature = "avx512f")] {
						match log_block_len {
							0 => unsafe {
								let value = get_block_values::<_, u32, 1>(self, block_idx)[0];
								_mm256_set1_epi32(value as _).into()
							}
							1 => unsafe {
								_mm256_permutexvar_epi8(
									LOG_B32_1[block_idx],
									self.0,
								).into()
							}
							2 => unsafe {
								_mm256_permutexvar_epi8(
									LOG_B32_2[block_idx],
									self.0,
								).into()
							}
							3 => self,
							_ => panic!("unsupported block length"),
						}
					} else {
						match log_block_len {
							0 => unsafe {
								let value = get_block_values::<_, u32, 1>(self, block_idx)[0];
								_mm256_set1_epi32(value as _).into()
							}
							1 => unsafe {
								let values = get_block_values::<_, u32, 2>(self, block_idx);
								Self::from_fn::<u32>(|i| values[i / 4])
							}
							2 => unsafe {
								let values = get_block_values::<_, u32, 4>(self, block_idx);
								Self::from_fn::<u32>(|i| values[i / 2])
							}
							3 => self,
							_ => panic!("unsupported block length"),
						}
					}
				}
			}
			6 => match log_block_len {
				0 => unsafe {
					let value = get_block_values::<_, u64, 1>(self, block_idx)[0];
					_mm256_set1_epi64x(value as _).into()
				},
				1 => unsafe {
					cfg_if! {
						if #[cfg(target_feature = "avx512f")] {
							_mm256_permutexvar_epi8(
								LOG_B64_1[block_idx],
								self.0,
							).into()
						} else {
							let values = get_block_values::<_, u64, 2>(self, block_idx);
							Self::from_fn::<u64>(|i| values[i / 2])
						}
					}
				},
				2 => self,
				_ => panic!("unsupported block length"),
			},
			7 => match log_block_len {
				0 => unsafe {
					let value = get_block_values::<_, u128, 1>(self, block_idx)[0];
					Self::from_fn::<u128>(|_| value)
				},
				1 => self,
				_ => panic!("unsupported block length"),
			},
			_ => unsafe { spread_fallback(self, log_block_len, block_idx) },
		}
	}

	#[inline]
	fn shr_128b_lanes(self, rhs: usize) -> Self {
		// This implementation is effective when `rhs` is known at compile-time.
		// In our code this is always the case.
		bitshift_128b!(
			self.0,
			rhs,
			_mm256_bsrli_epi128,
			_mm256_srli_epi64,
			_mm256_slli_epi64,
			_mm256_or_si256
		)
	}

	#[inline]
	fn shl_128b_lanes(self, rhs: usize) -> Self {
		// This implementation is effective when `rhs` is known at compile-time.
		// In our code this is always the case.
		bitshift_128b!(
			self.0,
			rhs,
			_mm256_bslli_epi128,
			_mm256_slli_epi64,
			_mm256_srli_epi64,
			_mm256_or_si256
		);
	}

	#[inline]
	fn unpack_lo_128b_lanes(self, other: Self, log_block_len: usize) -> Self {
		match log_block_len {
			0..3 => unpack_lo_128b_fallback(self, other, log_block_len),
			3 => unsafe { _mm256_unpacklo_epi8(self.0, other.0).into() },
			4 => unsafe { _mm256_unpacklo_epi16(self.0, other.0).into() },
			5 => unsafe { _mm256_unpacklo_epi32(self.0, other.0).into() },
			6 => unsafe { _mm256_unpacklo_epi64(self.0, other.0).into() },
			_ => panic!("unsupported block length"),
		}
	}

	#[inline]
	fn unpack_hi_128b_lanes(self, other: Self, log_block_len: usize) -> Self {
		match log_block_len {
			0..3 => unpack_hi_128b_fallback(self, other, log_block_len),
			3 => unsafe { _mm256_unpackhi_epi8(self.0, other.0).into() },
			4 => unsafe { _mm256_unpackhi_epi16(self.0, other.0).into() },
			5 => unsafe { _mm256_unpackhi_epi32(self.0, other.0).into() },
			6 => unsafe { _mm256_unpackhi_epi64(self.0, other.0).into() },
			_ => panic!("unsupported block length"),
		}
	}
}

unsafe impl Zeroable for M256 {}

unsafe impl Pod for M256 {}

unsafe impl Send for M256 {}

unsafe impl Sync for M256 {}

impl<Scalar: BinaryField> From<__m256i> for PackedPrimitiveType<M256, Scalar> {
	fn from(value: __m256i) -> Self {
		Self::from(M256::from(value))
	}
}

impl<Scalar: BinaryField> From<[u128; 2]> for PackedPrimitiveType<M256, Scalar> {
	fn from(value: [u128; 2]) -> Self {
		Self::from(M256::from(value))
	}
}

impl<Scalar: BinaryField> From<PackedPrimitiveType<M256, Scalar>> for __m256i {
	fn from(value: PackedPrimitiveType<M256, Scalar>) -> Self {
		value.to_underlier().into()
	}
}

impl<Scalar: BinaryField> Broadcast<Scalar> for PackedPrimitiveType<M256, Scalar>
where
	u128: From<Scalar::Underlier>,
{
	fn broadcast(scalar: Scalar) -> Self {
		let tower_level = Scalar::N_BITS.ilog2() as usize;
		let mut value = u128::from(scalar.to_underlier());
		for n in tower_level..3 {
			value |= value << (1 << n);
		}

		match tower_level {
			0..=3 => unsafe { _mm256_broadcastb_epi8(must_cast(value)).into() },
			4 => unsafe { _mm256_broadcastw_epi16(must_cast(value)).into() },
			5 => unsafe { _mm256_broadcastd_epi32(must_cast(value)).into() },
			6 => unsafe { _mm256_broadcastq_epi64(must_cast(value)).into() },
			7 => [value, value].into(),
			_ => unreachable!(),
		}
	}
}

impl UnderlierWithBitConstants for M256 {
	const INTERLEAVE_EVEN_MASK: &'static [Self] = &[
		Self::from_equal_u128s(interleave_mask_even!(u128, 0)),
		Self::from_equal_u128s(interleave_mask_even!(u128, 1)),
		Self::from_equal_u128s(interleave_mask_even!(u128, 2)),
		Self::from_equal_u128s(interleave_mask_even!(u128, 3)),
		Self::from_equal_u128s(interleave_mask_even!(u128, 4)),
		Self::from_equal_u128s(interleave_mask_even!(u128, 5)),
		Self::from_equal_u128s(interleave_mask_even!(u128, 6)),
	];

	const INTERLEAVE_ODD_MASK: &'static [Self] = &[
		Self::from_equal_u128s(interleave_mask_odd!(u128, 0)),
		Self::from_equal_u128s(interleave_mask_odd!(u128, 1)),
		Self::from_equal_u128s(interleave_mask_odd!(u128, 2)),
		Self::from_equal_u128s(interleave_mask_odd!(u128, 3)),
		Self::from_equal_u128s(interleave_mask_odd!(u128, 4)),
		Self::from_equal_u128s(interleave_mask_odd!(u128, 5)),
		Self::from_equal_u128s(interleave_mask_odd!(u128, 6)),
	];

	fn interleave(self, other: Self, log_block_len: usize) -> (Self, Self) {
		let (a, b) = unsafe { interleave_bits(self.0, other.0, log_block_len) };
		(Self(a), Self(b))
	}

	fn transpose(mut self, mut other: Self, log_block_len: usize) -> (Self, Self) {
		let (a, b) = unsafe { transpose_bits(self.0, other.0, log_block_len) };
		self.0 = a;
		other.0 = b;
		(self, other)
	}
}

#[inline]
unsafe fn interleave_bits(a: __m256i, b: __m256i, log_block_len: usize) -> (__m256i, __m256i) {
	match log_block_len {
		0 => unsafe {
			let mask = _mm256_set1_epi8(0x55i8);
			interleave_bits_imm::<1>(a, b, mask)
		},
		1 => unsafe {
			let mask = _mm256_set1_epi8(0x33i8);
			interleave_bits_imm::<2>(a, b, mask)
		},
		2 => unsafe {
			let mask = _mm256_set1_epi8(0x0fi8);
			interleave_bits_imm::<4>(a, b, mask)
		},
		3 => unsafe {
			let shuffle = _mm256_set_epi8(
				15, 13, 11, 9, 7, 5, 3, 1, 14, 12, 10, 8, 6, 4, 2, 0, 15, 13, 11, 9, 7, 5, 3, 1,
				14, 12, 10, 8, 6, 4, 2, 0,
			);
			let a = _mm256_shuffle_epi8(a, shuffle);
			let b = _mm256_shuffle_epi8(b, shuffle);
			let a_prime = _mm256_unpacklo_epi8(a, b);
			let b_prime = _mm256_unpackhi_epi8(a, b);
			(a_prime, b_prime)
		},
		4 => unsafe {
			let shuffle = _mm256_set_epi8(
				15, 14, 11, 10, 7, 6, 3, 2, 13, 12, 9, 8, 5, 4, 1, 0, 15, 14, 11, 10, 7, 6, 3, 2,
				13, 12, 9, 8, 5, 4, 1, 0,
			);
			let a = _mm256_shuffle_epi8(a, shuffle);
			let b = _mm256_shuffle_epi8(b, shuffle);
			let a_prime = _mm256_unpacklo_epi16(a, b);
			let b_prime = _mm256_unpackhi_epi16(a, b);
			(a_prime, b_prime)
		},
		5 => unsafe {
			let shuffle = _mm256_set_epi8(
				15, 14, 13, 12, 7, 6, 5, 4, 11, 10, 9, 8, 3, 2, 1, 0, 15, 14, 13, 12, 7, 6, 5, 4,
				11, 10, 9, 8, 3, 2, 1, 0,
			);
			let a = _mm256_shuffle_epi8(a, shuffle);
			let b = _mm256_shuffle_epi8(b, shuffle);
			let a_prime = _mm256_unpacklo_epi32(a, b);
			let b_prime = _mm256_unpackhi_epi32(a, b);
			(a_prime, b_prime)
		},
		6 => unsafe {
			let a_prime = _mm256_unpacklo_epi64(a, b);
			let b_prime = _mm256_unpackhi_epi64(a, b);
			(a_prime, b_prime)
		},
		7 => unsafe {
			let a_prime = _mm256_permute2x128_si256(a, b, 0x20);
			let b_prime = _mm256_permute2x128_si256(a, b, 0x31);
			(a_prime, b_prime)
		},
		_ => panic!("unsupported block length"),
	}
}

#[inline]
unsafe fn transpose_bits(a: __m256i, b: __m256i, log_block_len: usize) -> (__m256i, __m256i) {
	match log_block_len {
		0..=3 => unsafe {
			let shuffle = _mm256_set_epi8(
				15, 13, 11, 9, 7, 5, 3, 1, 14, 12, 10, 8, 6, 4, 2, 0, 15, 13, 11, 9, 7, 5, 3, 1,
				14, 12, 10, 8, 6, 4, 2, 0,
			);
			let (mut a, mut b) = transpose_with_shuffle(a, b, shuffle);
			for log_block_len in (log_block_len..3).rev() {
				(a, b) = interleave_bits(a, b, log_block_len);
			}

			(a, b)
		},
		4 => unsafe {
			let shuffle = _mm256_set_epi8(
				15, 14, 11, 10, 7, 6, 3, 2, 13, 12, 9, 8, 5, 4, 1, 0, 15, 14, 11, 10, 7, 6, 3, 2,
				13, 12, 9, 8, 5, 4, 1, 0,
			);

			transpose_with_shuffle(a, b, shuffle)
		},
		5 => unsafe {
			let shuffle = _mm256_set_epi8(
				15, 14, 13, 12, 7, 6, 5, 4, 11, 10, 9, 8, 3, 2, 1, 0, 15, 14, 13, 12, 7, 6, 5, 4,
				11, 10, 9, 8, 3, 2, 1, 0,
			);

			transpose_with_shuffle(a, b, shuffle)
		},
		6 => unsafe {
			let (a, b) = (_mm256_unpacklo_epi64(a, b), _mm256_unpackhi_epi64(a, b));

			(_mm256_permute4x64_epi64(a, 0b11011000), _mm256_permute4x64_epi64(b, 0b11011000))
		},
		7 => unsafe {
			(_mm256_permute2x128_si256(a, b, 0x20), _mm256_permute2x128_si256(a, b, 0x31))
		},
		_ => panic!("unsupported block length"),
	}
}

#[inline(always)]
unsafe fn transpose_with_shuffle(a: __m256i, b: __m256i, shuffle: __m256i) -> (__m256i, __m256i) {
	unsafe {
		let a = _mm256_shuffle_epi8(a, shuffle);
		let b = _mm256_shuffle_epi8(b, shuffle);

		let (a, b) = (_mm256_unpacklo_epi64(a, b), _mm256_unpackhi_epi64(a, b));

		(_mm256_permute4x64_epi64(a, 0b11011000), _mm256_permute4x64_epi64(b, 0b11011000))
	}
}

#[inline]
unsafe fn interleave_bits_imm<const BLOCK_LEN: i32>(
	a: __m256i,
	b: __m256i,
	mask: __m256i,
) -> (__m256i, __m256i) {
	unsafe {
		let t = _mm256_and_si256(_mm256_xor_si256(_mm256_srli_epi64::<BLOCK_LEN>(a), b), mask);
		let a_prime = _mm256_xor_si256(a, _mm256_slli_epi64::<BLOCK_LEN>(t));
		let b_prime = _mm256_xor_si256(b, t);
		(a_prime, b_prime)
	}
}

cfg_if! {
	if #[cfg(target_feature = "avx512f")] {
		static LOG_B8_1: [__m256i; 16] = precompute_spread_mask::<16>(1, 3);
		static LOG_B8_2: [__m256i; 8] = precompute_spread_mask::<8>(2, 3);
		static LOG_B8_3: [__m256i; 4] = precompute_spread_mask::<4>(3, 3);
		static LOG_B8_4: [__m256i; 2] = precompute_spread_mask::<2>(4, 3);

		static LOG_B16_1: [__m256i; 8] = precompute_spread_mask::<8>(1, 4);
		static LOG_B16_2: [__m256i; 4] = precompute_spread_mask::<4>(2, 4);
		static LOG_B16_3: [__m256i; 2] = precompute_spread_mask::<2>(3, 4);

		static LOG_B32_1: [__m256i; 4] = precompute_spread_mask::<4>(1, 5);
		static LOG_B32_2: [__m256i; 2] = precompute_spread_mask::<2>(2, 5);

		static LOG_B64_1: [__m256i; 2] = precompute_spread_mask::<2>(1, 6);

		const fn precompute_spread_mask<const BLOCK_IDX_AMOUNT: usize>(
			log_block_len: usize,
			t_log_bits: usize,
		) -> [__m256i; BLOCK_IDX_AMOUNT] {
			let element_log_width = t_log_bits - 3;

			let element_width = 1 << element_log_width;

			let block_size = 1 << (log_block_len + element_log_width);
			let repeat = 1 << (5 - element_log_width - log_block_len);
			let mut masks = [[0u8; 32]; BLOCK_IDX_AMOUNT];

			let mut block_idx = 0;

			while block_idx < BLOCK_IDX_AMOUNT {
				let base = block_idx * block_size;
				let mut j = 0;
				while j < 32 {
					masks[block_idx][j] =
						(base + ((j / element_width) / repeat) * element_width + j % element_width) as u8;
					j += 1;
				}
				block_idx += 1;
			}
			let mut m256_masks = [m256_from_u128s!(0, 0,); BLOCK_IDX_AMOUNT];

			let mut block_idx = 0;

			while block_idx < BLOCK_IDX_AMOUNT {
				let mut u128s = [0; 2];
				let mut i = 0;
				while i < 2 {
					unsafe {
						u128s[i] = u128::from_le_bytes(
							*(masks[block_idx].as_ptr().add(16 * i) as *const [u8; 16]),
						);
					}
					i += 1;
				}
				m256_masks[block_idx] = m256_from_u128s!(u128s[0], u128s[1],);
				block_idx += 1;
			}

			m256_masks
		}
	}
}

impl_iteration!(M256,
	@strategy BitIterationStrategy, U1,
	@strategy FallbackStrategy, U2, U4,
	@strategy DivisibleStrategy, u8, u16, u32, u64, u128, M128, M256,
);

#[cfg(test)]
mod tests {
	use binius_utils::bytes::BytesMut;
	use proptest::{arbitrary::any, proptest};
	use rand::{SeedableRng, rngs::StdRng};

	use super::*;
	use crate::underlier::single_element_mask_bits;

	fn check_roundtrip<T>(val: M256)
	where
		T: From<M256>,
		M256: From<T>,
	{
		assert_eq!(M256::from(T::from(val)), val);
	}

	#[test]
	fn test_constants() {
		assert_eq!(M256::default(), M256::ZERO);
		assert_eq!(M256::from(0u128), M256::ZERO);
		assert_eq!(M256::from([1u128, 0u128]), M256::ONE);
	}

	#[derive(Default)]
	struct ByteData([u128; 2]);

	impl ByteData {
		const fn get_bit(&self, i: usize) -> u8 {
			if self.0[i / 128] & (1u128 << (i % 128)) == 0 {
				0
			} else {
				1
			}
		}

		fn set_bit(&mut self, i: usize, val: u8) {
			self.0[i / 128] &= !(1 << (i % 128));
			self.0[i / 128] |= (val as u128) << (i % 128);
		}
	}

	impl From<ByteData> for M256 {
		fn from(value: ByteData) -> Self {
			let vals: [u128; 2] = unsafe { std::mem::transmute(value) };
			vals.into()
		}
	}

	impl From<[u128; 2]> for ByteData {
		fn from(value: [u128; 2]) -> Self {
			unsafe { std::mem::transmute(value) }
		}
	}

	impl Shl<usize> for ByteData {
		type Output = Self;

		fn shl(self, rhs: usize) -> Self::Output {
			let mut result = Self::default();
			for i in 0..256 {
				if i >= rhs {
					result.set_bit(i, self.get_bit(i - rhs));
				}
			}

			result
		}
	}

	impl Shr<usize> for ByteData {
		type Output = Self;

		fn shr(self, rhs: usize) -> Self::Output {
			let mut result = Self::default();
			for i in 0..256 {
				if i + rhs < 256 {
					result.set_bit(i, self.get_bit(i + rhs));
				}
			}

			result
		}
	}

	fn get(value: M256, log_block_len: usize, index: usize) -> M256 {
		(value >> (index << log_block_len)) & single_element_mask_bits::<M256>(1 << log_block_len)
	}

	proptest! {
		#[test]
		#[allow(clippy::tuple_array_conversions)] // false positive
		fn test_conversion(a in any::<u128>(), b in any::<u128>()) {
			check_roundtrip::<[u128; 2]>([a, b].into());
			check_roundtrip::<__m256i>([a, b].into());
		}

		#[test]
		fn test_binary_bit_operations([a, b, c, d] in any::<[u128;4]>()) {
			assert_eq!(M256::from([a & b, c & d]), M256::from([a, c]) & M256::from([b, d]));
			assert_eq!(M256::from([a | b, c | d]), M256::from([a, c]) | M256::from([b, d]));
			assert_eq!(M256::from([a ^ b, c ^ d]), M256::from([a, c]) ^ M256::from([b, d]));
		}

		#[test]
		#[allow(clippy::tuple_array_conversions)] // false positive
		fn test_negate(a in any::<u128>(), b in any::<u128>()) {
			assert_eq!(M256::from([!a, ! b]), !M256::from([a, b]))
		}

		#[test]
		fn test_shifts(a in any::<[u128; 2]>(), rhs in 0..255usize) {
			assert_eq!(M256::from(a) << rhs, M256::from(ByteData::from(a) << rhs));
			assert_eq!(M256::from(a) >> rhs, M256::from(ByteData::from(a) >> rhs));
		}

		#[test]
		fn test_interleave_bits(a in any::<[u128; 2]>(), b in any::<[u128; 2]>(), height in 0usize..8) {
			let a = M256::from(a);
			let b = M256::from(b);
			let (c, d) = unsafe {interleave_bits(a.0, b.0, height)};
			let (c, d) = (M256::from(c), M256::from(d));

			let block_len = 1usize << height;
			for i in (0..256/block_len).step_by(2) {
				assert_eq!(get(c, height, i), get(a, height, i));
				assert_eq!(get(c, height, i+1), get(b, height, i));
				assert_eq!(get(d, height, i), get(a, height, i+1));
				assert_eq!(get(d, height, i+1), get(b, height, i+1));
			}
		}

		#[test]
		fn test_unpack_lo(a in any::<[u128; 2]>(), b in any::<[u128; 2]>(), height in 0usize..7) {
			let a = M256::from(a);
			let b = M256::from(b);

			let result = a.unpack_lo_128b_lanes(b, height);
			let half_block_count = 128>>(height + 1);
			for i in 0..half_block_count {
				assert_eq!(get(result, height, 2*i), get(a, height, i));
				assert_eq!(get(result, height, 2*i+1), get(b, height, i));
				assert_eq!(get(result, height, 2*(i + half_block_count)), get(a, height, 2 * half_block_count + i));
				assert_eq!(get(result, height, 2*(i + half_block_count)+1), get(b, height, 2 * half_block_count + i));
			}
		}

		#[test]
		fn test_unpack_hi(a in any::<[u128; 2]>(), b in any::<[u128; 2]>(), height in 0usize..7) {
			let a = M256::from(a);
			let b = M256::from(b);

			let result = a.unpack_hi_128b_lanes(b, height);
			let half_block_count = 128>>(height + 1);
			for i in 0..half_block_count {
				assert_eq!(get(result, height, 2*i), get(a, height, i + half_block_count));
				assert_eq!(get(result, height, 2*i+1), get(b, height, i + half_block_count));
				assert_eq!(get(result, height, 2*(half_block_count + i)), get(a, height, 3*half_block_count + i));
				assert_eq!(get(result, height, 2*(half_block_count + i) +1), get(b, height, 3*half_block_count + i));
			}
		}
	}

	#[test]
	fn test_fill_with_bit() {
		assert_eq!(M256::fill_with_bit(1), M256::from([u128::MAX, u128::MAX]));
		assert_eq!(M256::fill_with_bit(0), M256::from(0u128));
	}

	#[test]
	fn test_eq() {
		let a = M256::from(0u128);
		let b = M256::from(42u128);
		let c = M256::from(u128::MAX);
		let d = M256::from([u128::MAX, u128::MAX]);

		assert_eq!(a, a);
		assert_eq!(b, b);
		assert_eq!(c, c);
		assert_eq!(d, d);

		assert_ne!(a, b);
		assert_ne!(a, c);
		assert_ne!(a, d);
		assert_ne!(b, c);
		assert_ne!(b, d);
		assert_ne!(c, d);
	}

	#[test]
	fn test_serialize_and_deserialize_m256() {
		let mut rng = StdRng::from_seed([0; 32]);

		let original_value = M256::from([rng.random::<u128>(), rng.random::<u128>()]);

		let mut buf = BytesMut::new();
		original_value.serialize(&mut buf).unwrap();

		let deserialized_value = M256::deserialize(buf.freeze()).unwrap();

		assert_eq!(original_value, deserialized_value);
	}
}
