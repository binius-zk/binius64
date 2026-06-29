// Copyright 2024-2025 Irreducible Inc.

use super::{U1, U2, U4, underlier_type::UnderlierType};
use crate::Divisible;

/// Fallback implementation of `spread` method.
///
/// # Safety
/// `log_block_len + T::LOG_BITS` must be less than or equal to `U::LOG_BITS`.
/// `block_idx` must be less than `1 << (U::LOG_BITS - log_block_len)`.
pub(crate) unsafe fn spread_fallback<U, T>(value: U, log_block_len: usize, block_idx: usize) -> U
where
	U: UnderlierType + Divisible<T>,
	T: UnderlierType,
{
	debug_assert!(
		log_block_len + T::LOG_BITS <= U::LOG_BITS,
		"log_block_len: {}, U::BITS: {}, T::BITS: {}",
		log_block_len,
		U::BITS,
		T::BITS
	);
	debug_assert!(
		block_idx < 1 << (U::LOG_BITS - log_block_len),
		"block_idx: {}, U::BITS: {}, log_block_len: {}",
		block_idx,
		U::BITS,
		log_block_len
	);

	let mut result = U::ZERO;
	let block_offset = block_idx << log_block_len;
	let log_repeat = U::LOG_BITS - T::LOG_BITS - log_block_len;
	for i in 0..1 << log_block_len {
		// Safety: `block_offset + i < block_offset + block_len <= U::BITS / T::BITS`
		// (precondition), and `i << log_repeat < U::BITS / T::BITS` (by the loop bound and
		// log_repeat definition).
		unsafe {
			Divisible::<T>::set_unchecked(
				&mut result,
				i << log_repeat,
				Divisible::<T>::get_unchecked(&value, block_offset + i),
			);
		}
	}

	for i in 0..log_repeat {
		result |= result << (1 << (T::LOG_BITS + i));
	}

	result
}

#[cfg(test)]
#[allow(unused)]
pub(crate) fn single_element_mask_bits<T: UnderlierType>(bits_count: usize) -> T {
	use binius_utils::checked_arithmetics::checked_log_2;

	if bits_count == T::BITS {
		!T::ZERO
	} else {
		let mut result = T::ONE;
		for height in 0..checked_log_2(bits_count) {
			result |= result << (1 << height)
		}

		result
	}
}

/// Value that can be spread to a single u8
pub(crate) trait SpreadToByte {
	fn spread_to_byte(self) -> u8;
}

impl SpreadToByte for U1 {
	#[inline(always)]
	fn spread_to_byte(self) -> u8 {
		u8::fill_with_bit(self.val())
	}
}

impl SpreadToByte for U2 {
	#[inline(always)]
	fn spread_to_byte(self) -> u8 {
		let mut result = self.val();
		result |= result << 2;
		result |= result << 4;

		result
	}
}

impl SpreadToByte for U4 {
	#[inline(always)]
	fn spread_to_byte(self) -> u8 {
		let mut result = self.val();
		result |= result << 4;

		result
	}
}

/// A helper functions for implementing `UnderlierType::spread` for SIMD types.
///
/// # Safety
/// `log_block_len + T::LOG_BITS` must be less than or equal to `U::LOG_BITS`.
#[allow(unused)]
#[inline(always)]
pub(crate) unsafe fn get_block_values<U, T, const BLOCK_LEN: usize>(
	value: U,
	block_idx: usize,
) -> [T; BLOCK_LEN]
where
	U: UnderlierType + From<T> + Divisible<T>,
	T: UnderlierType,
{
	// Safety: `block_idx * BLOCK_LEN + i < U::BITS / T::BITS` by the function preconditions.
	std::array::from_fn(|i| unsafe {
		Divisible::<T>::get_unchecked(&value, block_idx * BLOCK_LEN + i)
	})
}

/// A helper functions for implementing `UnderlierType::spread` for SIMD types.
///
/// # Safety
/// `log_block_len + T::LOG_BITS` must be less than or equal to `U::LOG_BITS`.
#[allow(unused)]
#[inline(always)]
pub(crate) unsafe fn get_spread_bytes<U, T, const BLOCK_LEN: usize>(
	value: U,
	block_idx: usize,
) -> [u8; BLOCK_LEN]
where
	U: UnderlierType + From<T> + Divisible<T>,
	T: UnderlierType + SpreadToByte,
{
	unsafe { get_block_values::<U, T, BLOCK_LEN>(value, block_idx) }
		.map(SpreadToByte::spread_to_byte)
}

#[cfg(test)]
mod tests {
	use proptest::{arbitrary::any, bits, proptest};

	use super::{
		super::small_uint::{U1, U2, U4},
		*,
	};

	#[test]
	fn test_from_fn() {
		assert_eq!(u32::from_fn(|_| U1::new(0)), 0);
		assert_eq!(u32::from_fn(|i| U1::new((i % 2) as u8)), 0xaaaaaaaa);
		assert_eq!(u32::from_fn(|_| U1::new(1)), u32::MAX);

		assert_eq!(u32::from_fn(|_| U2::new(0)), 0);
		assert_eq!(u32::from_fn(|_| U2::new(1)), 0x55555555);
		assert_eq!(u32::from_fn(|_| U2::new(2)), 0xaaaaaaaa);
		assert_eq!(u32::from_fn(|_| U2::new(3)), u32::MAX);
		assert_eq!(u32::from_fn(|i| U2::new((i % 4) as u8)), 0xe4e4e4e4);

		assert_eq!(u32::from_fn(|_| U4::new(0)), 0);
		assert_eq!(u32::from_fn(|_| U4::new(1)), 0x11111111);
		assert_eq!(u32::from_fn(|_| U4::new(8)), 0x88888888);
		assert_eq!(u32::from_fn(|_| U4::new(31)), 0xffffffff);
		assert_eq!(u32::from_fn(|i| U4::new(i as u8)), 0x76543210);

		assert_eq!(u32::from_fn(|_| 0u8), 0);
		assert_eq!(u32::from_fn(|_| 0xabu8), 0xabababab);
		assert_eq!(u32::from_fn(|_| 255u8), 0xffffffff);
		assert_eq!(u32::from_fn(|i| i as u8), 0x03020100);
	}

	#[test]
	fn test_broadcast_subvalue() {
		assert_eq!(u32::broadcast_subvalue(U1::new(0)), 0);
		assert_eq!(u32::broadcast_subvalue(U1::new(1)), u32::MAX);

		assert_eq!(u32::broadcast_subvalue(U2::new(0)), 0);
		assert_eq!(u32::broadcast_subvalue(U2::new(1)), 0x55555555);
		assert_eq!(u32::broadcast_subvalue(U2::new(2)), 0xaaaaaaaa);
		assert_eq!(u32::broadcast_subvalue(U2::new(3)), u32::MAX);

		assert_eq!(u32::broadcast_subvalue(U4::new(0)), 0);
		assert_eq!(u32::broadcast_subvalue(U4::new(1)), 0x11111111);
		assert_eq!(u32::broadcast_subvalue(U4::new(8)), 0x88888888);
		assert_eq!(u32::broadcast_subvalue(U4::new(31)), 0xffffffff);

		assert_eq!(u32::broadcast_subvalue(0u8), 0);
		assert_eq!(u32::broadcast_subvalue(0xabu8), 0xabababab);
		assert_eq!(u32::broadcast_subvalue(255u8), 0xffffffff);
	}

	#[test]
	fn test_divisible_get_u32() {
		let value = 0xab12cd34u32;

		assert_eq!(Divisible::<U1>::get(&value, 0), U1::new(0));
		assert_eq!(Divisible::<U1>::get(&value, 1), U1::new(0));
		assert_eq!(Divisible::<U1>::get(&value, 2), U1::new(1));
		assert_eq!(Divisible::<U1>::get(&value, 31), U1::new(1));

		assert_eq!(Divisible::<U2>::get(&value, 0), U2::new(0));
		assert_eq!(Divisible::<U2>::get(&value, 1), U2::new(1));
		assert_eq!(Divisible::<U2>::get(&value, 2), U2::new(3));
		assert_eq!(Divisible::<U2>::get(&value, 15), U2::new(2));

		assert_eq!(Divisible::<U4>::get(&value, 0), U4::new(4));
		assert_eq!(Divisible::<U4>::get(&value, 1), U4::new(3));
		assert_eq!(Divisible::<U4>::get(&value, 2), U4::new(13));
		assert_eq!(Divisible::<U4>::get(&value, 7), U4::new(10));

		assert_eq!(Divisible::<u8>::get(&value, 0), 0x34u8);
		assert_eq!(Divisible::<u8>::get(&value, 1), 0xcdu8);
		assert_eq!(Divisible::<u8>::get(&value, 2), 0x12u8);
		assert_eq!(Divisible::<u8>::get(&value, 3), 0xabu8);
	}

	proptest! {
		#[test]
		fn test_divisible_set_1b(mut init_val in any::<u32>(), i in 0usize..31, val in bits::u8::masked(1)) {
			Divisible::<U1>::set(&mut init_val, i, U1::new(val));
			assert_eq!(Divisible::<U1>::get(&init_val, i), U1::new(val));
		}

		#[test]
		fn test_divisible_set_2b(mut init_val in any::<u32>(), i in 0usize..15, val in bits::u8::masked(3)) {
			Divisible::<U2>::set(&mut init_val, i, U2::new(val));
			assert_eq!(Divisible::<U2>::get(&init_val, i), U2::new(val));
		}

		#[test]
		fn test_divisible_set_4b(mut init_val in any::<u32>(), i in 0usize..7, val in bits::u8::masked(7)) {
			Divisible::<U4>::set(&mut init_val, i, U4::new(val));
			assert_eq!(Divisible::<U4>::get(&init_val, i), U4::new(val));
		}

		#[test]
		fn test_divisible_set_8b(mut init_val in any::<u32>(), i in 0usize..3, val in bits::u8::masked(15)) {
			Divisible::<u8>::set(&mut init_val, i, val);
			assert_eq!(Divisible::<u8>::get(&init_val, i), val);
		}
	}
}
