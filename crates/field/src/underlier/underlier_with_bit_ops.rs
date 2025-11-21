// Copyright 2024-2025 Irreducible Inc.

use std::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not, Shl, Shr};

use binius_utils::checked_arithmetics::{checked_int_div, checked_log_2};

use super::{
	U1, U2, U4,
	underlier_type::{NumCast, UnderlierType},
};

/// Underlier type that supports bit arithmetic.
pub trait UnderlierWithBitOps:
	UnderlierType
	+ BitAnd<Self, Output = Self>
	+ BitAndAssign<Self>
	+ BitOr<Self, Output = Self>
	+ BitOrAssign<Self>
	+ BitXor<Self, Output = Self>
	+ BitXorAssign<Self>
	+ Shr<usize, Output = Self>
	+ Shl<usize, Output = Self>
	+ Not<Output = Self>
{
	const ZERO: Self;
	const ONE: Self;
	const ONES: Self;

	/// Fill value with the given bit
	/// `val` must be 0 or 1.
	fn fill_with_bit(val: u8) -> Self;

	#[inline]
	fn from_fn<T>(mut f: impl FnMut(usize) -> T) -> Self
	where
		T: UnderlierType,
		Self: From<T>,
	{
		// This implementation is optimal for the case when `Self` us u8..u128.
		// For SIMD types/arrays specialization would be more performant.
		let mut result = Self::default();
		let width = checked_int_div(Self::BITS, T::BITS);
		for i in 0..width {
			result |= Self::from(f(i)) << (i * T::BITS);
		}

		result
	}

	/// Broadcast subvalue to fill `Self`.
	/// `Self::BITS/T::BITS` is supposed to be a power of 2.
	#[inline]
	fn broadcast_subvalue<T>(value: T) -> Self
	where
		T: UnderlierType,
		Self: From<T>,
	{
		// This implementation is optimal for the case when `Self` us u8..u128.
		// For SIMD types/arrays specialization would be more performant.
		let height = checked_log_2(checked_int_div(Self::BITS, T::BITS));
		let mut result = Self::from(value);
		for i in 0..height {
			result |= result << ((1 << i) * T::BITS);
		}

		result
	}

	/// Gets the subvalue from the given position.
	/// Function panics in case when index is out of range.
	///
	/// # Safety
	/// `i` must be less than `Self::BITS/T::BITS`.
	#[inline]
	unsafe fn get_subvalue<T>(&self, i: usize) -> T
	where
		T: UnderlierType + NumCast<Self>,
	{
		debug_assert!(
			i < checked_int_div(Self::BITS, T::BITS),
			"i: {} Self::BITS: {}, T::BITS: {}",
			i,
			Self::BITS,
			T::BITS
		);
		T::num_cast_from(*self >> (i * T::BITS))
	}

	/// Sets the subvalue in the given position.
	/// Function panics in case when index is out of range.
	///
	/// # Safety
	/// `i` must be less than `Self::BITS/T::BITS`.
	#[inline]
	unsafe fn set_subvalue<T>(&mut self, i: usize, val: T)
	where
		T: UnderlierWithBitOps,
		Self: From<T>,
	{
		debug_assert!(i < checked_int_div(Self::BITS, T::BITS));
		let mask = Self::from(single_element_mask::<T>());

		*self &= !(mask << (i * T::BITS));
		*self |= Self::from(val) << (i * T::BITS);
	}

	/// Spread takes a block of sub_elements of `T` type within the current value and
	/// repeats them to the full underlier width.
	///
	/// # Safety
	/// `log_block_len + T::LOG_BITS` must be less than or equal to `Self::LOG_BITS`.
	/// `block_idx` must be less than `1 << (Self::LOG_BITS - log_block_len)`.
	#[inline]
	unsafe fn spread<T>(self, log_block_len: usize, block_idx: usize) -> Self
	where
		T: UnderlierWithBitOps + NumCast<Self>,
		Self: From<T>,
	{
		unsafe { spread_fallback(self, log_block_len, block_idx) }
	}

	/// Left shift within 128-bit lanes.
	/// This can be more efficient than the full `Shl` implementation.
	fn shl_128b_lanes(self, shift: usize) -> Self;

	/// Right shift within 128-bit lanes.
	/// This can be more efficient than the full `Shr` implementation.
	fn shr_128b_lanes(self, shift: usize) -> Self;

	/// Unpacks `1 << log_block_len`-bit values from low parts of `self` and `other` within 128-bit
	/// lanes.
	///
	/// Example:
	///    self:  [a_0, a_1, a_2, a_3, a_4, a_5, a_6, a_7]
	///    other: [b_0, b_1, b_2, b_3, b_4, b_5, b_6, b_7]
	///    log_block_len: 1
	///
	///    result: [a_0, a_0, b_0, b_1, a_2, a_3, b_2, b_3]
	fn unpack_lo_128b_lanes(self, other: Self, log_block_len: usize) -> Self {
		unpack_lo_128b_fallback(self, other, log_block_len)
	}

	/// Unpacks `1 << log_block_len`-bit values from high parts of `self` and `other` within 128-bit
	/// lanes.
	///
	/// Example:
	///    self:  [a_0, a_1, a_2, a_3, a_4, a_5, a_6, a_7]
	///    other: [b_0, b_1, b_2, b_3, b_4, b_5, b_6, b_7]
	///    log_block_len: 1
	///
	///    result: [a_4, a_5, b_4, b_5, a_6, a_7, b_6, b_7]
	fn unpack_hi_128b_lanes(self, other: Self, log_block_len: usize) -> Self {
		unpack_hi_128b_fallback(self, other, log_block_len)
	}
}

/// Returns a bit mask for a single `T` element inside underlier type.
/// This function is completely optimized out by the compiler in release version
/// because all the values are known at compile time.
fn single_element_mask<T>() -> T
where
	T: UnderlierWithBitOps,
{
	single_element_mask_bits(T::BITS)
}

/// A helper function to apply unpack_lo/hi_128b_lanes for two values in an array
#[allow(dead_code)]
#[inline(always)]
pub(crate) fn pair_unpack_lo_hi_128b_lanes<U: UnderlierWithBitOps>(
	values: &mut impl AsMut<[U]>,
	i: usize,
	j: usize,
	log_block_len: usize,
) {
	let values = values.as_mut();

	(values[i], values[j]) = (
		values[i].unpack_lo_128b_lanes(values[j], log_block_len),
		values[i].unpack_hi_128b_lanes(values[j], log_block_len),
	);
}

/// Fallback implementation of `spread` method.
///
/// # Safety
/// `log_block_len + T::LOG_BITS` must be less than or equal to `U::LOG_BITS`.
/// `block_idx` must be less than `1 << (U::LOG_BITS - log_block_len)`.
pub(crate) unsafe fn spread_fallback<U, T>(value: U, log_block_len: usize, block_idx: usize) -> U
where
	U: UnderlierWithBitOps + From<T>,
	T: UnderlierWithBitOps + NumCast<U>,
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
		unsafe {
			result.set_subvalue(i << log_repeat, value.get_subvalue(block_offset + i));
		}
	}

	for i in 0..log_repeat {
		result |= result << (1 << (T::LOG_BITS + i));
	}

	result
}

#[inline(always)]
fn single_element_mask_bits_128b_lanes<T: UnderlierWithBitOps>(log_block_len: usize) -> T {
	let mut mask = single_element_mask_bits(1 << log_block_len);
	for i in 1..T::BITS / 128 {
		mask |= mask << (i * 128);
	}

	mask
}

pub(crate) fn unpack_lo_128b_fallback<T: UnderlierWithBitOps>(
	lhs: T,
	rhs: T,
	log_block_len: usize,
) -> T {
	assert!(log_block_len <= 6);

	let mask = single_element_mask_bits_128b_lanes::<T>(log_block_len);

	let mut result = T::ZERO;
	for i in 0..1 << (6 - log_block_len) {
		result |= ((lhs.shr_128b_lanes(i << log_block_len)) & mask)
			.shl_128b_lanes(i << (log_block_len + 1));
		result |= ((rhs.shr_128b_lanes(i << log_block_len)) & mask)
			.shl_128b_lanes((2 * i + 1) << log_block_len);
	}

	result
}

pub(crate) fn unpack_hi_128b_fallback<T: UnderlierWithBitOps>(
	lhs: T,
	rhs: T,
	log_block_len: usize,
) -> T {
	assert!(log_block_len <= 6);

	let mask = single_element_mask_bits_128b_lanes::<T>(log_block_len);
	let mut result = T::ZERO;
	for i in 0..1 << (6 - log_block_len) {
		result |= ((lhs.shr_128b_lanes(64 + (i << log_block_len))) & mask)
			.shl_128b_lanes(i << (log_block_len + 1));
		result |= ((rhs.shr_128b_lanes(64 + (i << log_block_len))) & mask)
			.shl_128b_lanes((2 * i + 1) << log_block_len);
	}

	result
}

pub(crate) fn single_element_mask_bits<T: UnderlierWithBitOps>(bits_count: usize) -> T {
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

/// A helper functions for implementing `UnderlierWithBitOps::spread_unchecked` for SIMD types.
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
	U: UnderlierWithBitOps + From<T>,
	T: UnderlierType + NumCast<U>,
{
	std::array::from_fn(|i| unsafe { value.get_subvalue::<T>(block_idx * BLOCK_LEN + i) })
}

/// A helper functions for implementing `UnderlierWithBitOps::spread_unchecked` for SIMD types.
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
	U: UnderlierWithBitOps + From<T>,
	T: UnderlierType + SpreadToByte + NumCast<U>,
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
	fn test_get_subvalue() {
		let value = 0xab12cd34u32;

		unsafe {
			assert_eq!(value.get_subvalue::<U1>(0), U1::new(0));
			assert_eq!(value.get_subvalue::<U1>(1), U1::new(0));
			assert_eq!(value.get_subvalue::<U1>(2), U1::new(1));
			assert_eq!(value.get_subvalue::<U1>(31), U1::new(1));

			assert_eq!(value.get_subvalue::<U2>(0), U2::new(0));
			assert_eq!(value.get_subvalue::<U2>(1), U2::new(1));
			assert_eq!(value.get_subvalue::<U2>(2), U2::new(3));
			assert_eq!(value.get_subvalue::<U2>(15), U2::new(2));

			assert_eq!(value.get_subvalue::<U4>(0), U4::new(4));
			assert_eq!(value.get_subvalue::<U4>(1), U4::new(3));
			assert_eq!(value.get_subvalue::<U4>(2), U4::new(13));
			assert_eq!(value.get_subvalue::<U4>(7), U4::new(10));

			assert_eq!(value.get_subvalue::<u8>(0), 0x34u8);
			assert_eq!(value.get_subvalue::<u8>(1), 0xcdu8);
			assert_eq!(value.get_subvalue::<u8>(2), 0x12u8);
			assert_eq!(value.get_subvalue::<u8>(3), 0xabu8);
		}
	}

	proptest! {
		#[test]
		fn test_set_subvalue_1b(mut init_val in any::<u32>(), i in 0usize..31, val in bits::u8::masked(1)) {
			unsafe {
				init_val.set_subvalue(i, U1::new(val));
				assert_eq!(init_val.get_subvalue::<U1>(i), U1::new(val));
			}
		}

		#[test]
		fn test_set_subvalue_2b(mut init_val in any::<u32>(), i in 0usize..15, val in bits::u8::masked(3)) {
			unsafe {
				init_val.set_subvalue(i, U2::new(val));
				assert_eq!(init_val.get_subvalue::<U2>(i), U2::new(val));
			}
		}

		#[test]
		fn test_set_subvalue_4b(mut init_val in any::<u32>(), i in 0usize..7, val in bits::u8::masked(7)) {
			unsafe {
				init_val.set_subvalue(i, U4::new(val));
				assert_eq!(init_val.get_subvalue::<U4>(i), U4::new(val));
			}
		}

		#[test]
		fn test_set_subvalue_8b(mut init_val in any::<u32>(), i in 0usize..3, val in bits::u8::masked(15)) {
			unsafe {
				init_val.set_subvalue(i, val);
				assert_eq!(init_val.get_subvalue::<u8>(i), val);
			}
		}
	}
}
