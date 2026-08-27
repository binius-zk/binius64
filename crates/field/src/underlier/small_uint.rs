// Copyright 2024-2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use std::{
	fmt::{Debug, Display, LowerHex},
	hash::{Hash, Hasher},
	mem::size_of,
	ops::{Not, Shl, Shr},
};

use binius_utils::{
	FixedSizeSerializeBytes, SerializationError, SerializeBytes,
	bytes::{Buf, BufMut},
	checked_arithmetics::checked_log_2,
	serialization::DeserializeBytes,
};
use bytemuck::{NoUninit, Zeroable};
use derive_more::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign};
use rand::{
	distr::{Distribution, StandardUniform},
	prelude::*,
};

use super::Underlier;
use crate::{
	arch::{interleave_mask_even, interleave_with_mask},
	divisible::{Divisible, impl_divisible_self, mapget},
};

/// Unsigned type with a size strictly less than 8 bits.
#[derive(
	Default,
	Zeroable,
	Clone,
	Copy,
	PartialEq,
	Eq,
	PartialOrd,
	Ord,
	BitAnd,
	BitAndAssign,
	BitOr,
	BitOrAssign,
	BitXor,
	BitXorAssign,
)]
#[repr(transparent)]
pub struct SmallU<const N: usize>(u8);

impl<const N: usize> SmallU<N> {
	const _CHECK_SIZE: () = {
		assert!(N < 8);
	};

	/// All bits set to one.
	pub const ONES: Self = Self((1u8 << N) - 1);

	#[inline(always)]
	pub const fn new(val: u8) -> Self {
		Self(val & Self::ONES.0)
	}

	#[inline(always)]
	pub const fn new_unchecked(val: u8) -> Self {
		Self(val)
	}

	#[inline(always)]
	pub const fn val(&self) -> u8 {
		self.0
	}
}

impl<const N: usize> Debug for SmallU<N> {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		Debug::fmt(&self.val(), f)
	}
}

impl<const N: usize> Display for SmallU<N> {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		Display::fmt(&self.val(), f)
	}
}

impl<const N: usize> LowerHex for SmallU<N> {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		LowerHex::fmt(&self.0, f)
	}
}
impl<const N: usize> Hash for SmallU<N> {
	#[inline]
	fn hash<H: Hasher>(&self, state: &mut H) {
		self.val().hash(state);
	}
}

impl<const N: usize> Distribution<SmallU<N>> for StandardUniform {
	fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> SmallU<N> {
		SmallU(rng.random_range(0..1u8 << N))
	}
}

impl<const N: usize> Shr<usize> for SmallU<N> {
	type Output = Self;

	#[inline(always)]
	fn shr(self, rhs: usize) -> Self::Output {
		Self(self.val() >> rhs)
	}
}

impl<const N: usize> Shl<usize> for SmallU<N> {
	type Output = Self;

	#[inline(always)]
	fn shl(self, rhs: usize) -> Self::Output {
		Self(self.val() << rhs) & Self::ONES
	}
}

impl<const N: usize> Not for SmallU<N> {
	type Output = Self;

	fn not(self) -> Self::Output {
		self ^ Self::ONES
	}
}

unsafe impl<const N: usize> NoUninit for SmallU<N> {}

impl Underlier for U1 {
	const LOG_BITS: usize = checked_log_2(1);

	const ZERO: Self = Self(0);
	const ONE: Self = Self(1);
	const ONES: Self = Self(1);

	fn interleave(self, _other: Self, _log_block_len: usize) -> (Self, Self) {
		panic!("interleave not supported for U1")
	}
}

impl Underlier for U2 {
	const LOG_BITS: usize = checked_log_2(2);

	const ZERO: Self = Self(0);
	const ONE: Self = Self(1);
	const ONES: Self = Self(0b11);

	fn interleave(self, other: Self, log_block_len: usize) -> (Self, Self) {
		const MASKS: &[U2] = &[U2::new(interleave_mask_even!(u8, 0))];
		interleave_with_mask(self, other, log_block_len, MASKS)
	}
}

impl Underlier for U4 {
	const LOG_BITS: usize = checked_log_2(4);

	const ZERO: Self = Self(0);
	const ONE: Self = Self(1);
	const ONES: Self = Self(0b1111);

	fn interleave(self, other: Self, log_block_len: usize) -> (Self, Self) {
		const MASKS: &[U4] = &[
			U4::new(interleave_mask_even!(u8, 0)),
			U4::new(interleave_mask_even!(u8, 1)),
		];
		interleave_with_mask(self, other, log_block_len, MASKS)
	}
}

impl<const N: usize> From<SmallU<N>> for u8 {
	#[inline(always)]
	fn from(value: SmallU<N>) -> Self {
		value.val()
	}
}

impl<const N: usize> From<SmallU<N>> for u16 {
	#[inline(always)]
	fn from(value: SmallU<N>) -> Self {
		u8::from(value) as _
	}
}

impl<const N: usize> From<SmallU<N>> for u32 {
	#[inline(always)]
	fn from(value: SmallU<N>) -> Self {
		u8::from(value) as _
	}
}

impl<const N: usize> From<SmallU<N>> for u64 {
	#[inline(always)]
	fn from(value: SmallU<N>) -> Self {
		u8::from(value) as _
	}
}

impl<const N: usize> From<SmallU<N>> for usize {
	#[inline(always)]
	fn from(value: SmallU<N>) -> Self {
		u8::from(value) as _
	}
}

impl<const N: usize> From<SmallU<N>> for u128 {
	#[inline(always)]
	fn from(value: SmallU<N>) -> Self {
		u8::from(value) as _
	}
}

impl From<SmallU<1>> for SmallU<2> {
	#[inline(always)]
	fn from(value: SmallU<1>) -> Self {
		Self(value.val())
	}
}

impl From<SmallU<1>> for SmallU<4> {
	#[inline(always)]
	fn from(value: SmallU<1>) -> Self {
		Self(value.val())
	}
}

impl From<SmallU<2>> for SmallU<4> {
	#[inline(always)]
	fn from(value: SmallU<2>) -> Self {
		Self(value.val())
	}
}

pub type U1 = SmallU<1>;
pub type U2 = SmallU<2>;
pub type U4 = SmallU<4>;

impl From<bool> for U1 {
	fn from(value: bool) -> Self {
		Self::new_unchecked(value as u8)
	}
}

impl From<U1> for bool {
	fn from(value: U1) -> Self {
		value == U1::ONE
	}
}

impl<const N: usize> SerializeBytes for SmallU<N> {
	fn serialize(&self, write_buf: impl BufMut) -> Result<(), SerializationError> {
		self.val().serialize(write_buf)
	}
}

impl<const N: usize> DeserializeBytes for SmallU<N> {
	fn deserialize(read_buf: impl Buf) -> Result<Self, SerializationError>
	where
		Self: Sized,
	{
		Ok(Self::new(DeserializeBytes::deserialize(read_buf)?))
	}
}

impl<const N: usize> FixedSizeSerializeBytes for SmallU<N> {
	const BYTE_SIZE: usize = 1;
}

/// Helper functions for Divisible implementations using bitmask operations on sub-byte elements.
///
/// These functions work on any type that implements `Divisible<u8>` by extracting
/// and modifying sub-byte elements through the byte interface.
pub mod bitmask {
	use super::{Divisible, SmallU};

	/// Get a sub-byte element at index (LSB-first ordering) without bounds checking.
	///
	/// # Safety
	///
	/// The caller must ensure that `index < Big::N` (over `SmallU<BITS>` elements).
	#[inline]
	pub unsafe fn get<Big, const BITS: usize>(value: &Big, index: usize) -> SmallU<BITS>
	where
		Big: Divisible<u8>,
	{
		let elems_per_byte = 8 / BITS;
		let byte_index = index / elems_per_byte;
		let sub_index = index % elems_per_byte;
		// Safety: `index < Big::N` over `SmallU<BITS>` implies `byte_index < Big::N` over `u8`.
		let byte = unsafe { Divisible::<u8>::get_unchecked(value, byte_index) };
		let shift = sub_index * BITS;
		SmallU::<BITS>::new(byte >> shift)
	}

	/// Set a sub-byte element at index (LSB-first ordering), returning modified value, without
	/// bounds checking.
	///
	/// # Safety
	///
	/// The caller must ensure that `index < Big::N` (over `SmallU<BITS>` elements).
	#[inline]
	pub unsafe fn set<Big, const BITS: usize>(
		mut value: Big,
		index: usize,
		val: SmallU<BITS>,
	) -> Big
	where
		Big: Divisible<u8>,
	{
		let elems_per_byte = 8 / BITS;
		let byte_index = index / elems_per_byte;
		let sub_index = index % elems_per_byte;
		// Safety: `index < Big::N` over `SmallU<BITS>` implies `byte_index < Big::N` over `u8`.
		let byte = unsafe { Divisible::<u8>::get_unchecked(&value, byte_index) };
		let shift = sub_index * BITS;
		let mask = (1u8 << BITS) - 1;
		let new_byte = (byte & !(mask << shift)) | (val.val() << shift);
		// Safety: `byte_index < Big::N` over `u8`, as above.
		unsafe { Divisible::<u8>::set_unchecked(&mut value, byte_index, new_byte) };
		value
	}
}

/// Iterator for dividing an underlier into sub-byte elements (ie. [`SmallU`]).
///
/// This iterator wraps a byte iterator and extracts sub-byte elements from each byte.
/// Generic over the byte iterator type `I`.
#[derive(Clone)]
pub struct SmallUDivisIter<I, const N: usize> {
	byte_iter: I,
	current_byte: Option<u8>,
	sub_idx: usize,
}

impl<I: Iterator<Item = u8>, const N: usize> SmallUDivisIter<I, N> {
	const ELEMS_PER_BYTE: usize = 8 / N;

	pub fn new(mut byte_iter: I) -> Self {
		let current_byte = byte_iter.next();
		Self {
			byte_iter,
			current_byte,
			sub_idx: 0,
		}
	}
}

impl<I: ExactSizeIterator<Item = u8>, const N: usize> Iterator for SmallUDivisIter<I, N> {
	type Item = SmallU<N>;

	#[inline]
	fn next(&mut self) -> Option<Self::Item> {
		let byte = self.current_byte?;
		let shift = self.sub_idx * N;
		let result = SmallU::<N>::new(byte >> shift);

		self.sub_idx += 1;
		if self.sub_idx >= Self::ELEMS_PER_BYTE {
			self.sub_idx = 0;
			self.current_byte = self.byte_iter.next();
		}

		Some(result)
	}

	#[inline]
	fn size_hint(&self) -> (usize, Option<usize>) {
		let remaining_in_current = if self.current_byte.is_some() {
			Self::ELEMS_PER_BYTE - self.sub_idx
		} else {
			0
		};
		let remaining_bytes = self.byte_iter.len();
		let total = remaining_in_current + remaining_bytes * Self::ELEMS_PER_BYTE;
		(total, Some(total))
	}
}

impl<I: ExactSizeIterator<Item = u8>, const N: usize> ExactSizeIterator for SmallUDivisIter<I, N> {}

/// Implements `Divisible` trait for SmallU types using bitmask operations.
///
/// This macro generates `Divisible<SmallU<BITS>>` implementations for a big type
/// by wrapping byte iteration with bitmasking to extract sub-byte elements.
macro_rules! impl_divisible_bitmask {
	// Special case for u8: operates directly on the byte without needing Divisible::<u8>
	(u8, $($bits:expr),+) => {
		$(
			impl $crate::divisible::Divisible<$crate::underlier::SmallU<$bits>> for u8 {
				const LOG_N: usize = (8usize / $bits).ilog2() as usize;

				#[inline]
				fn value_iter(value: Self) -> impl ExactSizeIterator<Item = $crate::underlier::SmallU<$bits>> + Send + Clone {
					$crate::underlier::SmallUDivisIter::new(std::iter::once(value))
				}

				#[inline]
				fn ref_iter(value: &Self) -> impl ExactSizeIterator<Item = $crate::underlier::SmallU<$bits>> + Send + Clone + '_ {
					$crate::underlier::SmallUDivisIter::new(std::iter::once(*value))
				}

				#[inline]
				fn slice_iter(slice: &[Self]) -> impl ExactSizeIterator<Item = $crate::underlier::SmallU<$bits>> + Send + Clone + '_ {
					$crate::underlier::SmallUDivisIter::new(slice.iter().copied())
				}

				#[inline]
				unsafe fn get_unchecked(&self, index: usize) -> $crate::underlier::SmallU<$bits> {
					let shift = index * $bits;
					$crate::underlier::SmallU::<$bits>::new(*self >> shift)
				}

				#[inline]
				unsafe fn set_unchecked(&mut self, index: usize, val: $crate::underlier::SmallU<$bits>) {
					let shift = index * $bits;
					let mask = (1u8 << $bits) - 1;
					*self = (*self & !(mask << shift)) | (val.val() << shift);
				}

				#[inline]
				fn broadcast(val: $crate::underlier::SmallU<$bits>) -> Self {
					if $bits == 1 {
						// For 1-bit values: 0 -> 0x00, 1 -> 0xFF
						val.val().wrapping_neg()
					} else {
						let mut result = val.val();
						// Self-replicate to fill the byte
						let mut current_bits = $bits;
						while current_bits < 8 {
							result |= result << current_bits;
							current_bits *= 2;
						}
						result
					}
				}

				#[inline]
				fn from_iter(iter: impl Iterator<Item = $crate::underlier::SmallU<$bits>>) -> Self {
					const N: usize = 8 / $bits;
					let mut result: Self = 0;
					for (i, val) in iter.take(N).enumerate() {
						$crate::divisible::Divisible::<$crate::underlier::SmallU<$bits>>::set(&mut result, i, val);
					}
					result
				}
			}
		)+
	};

	// General case for types larger than u8: wraps byte iteration
	($big:ty, $($bits:expr),+) => {
		$(
			impl $crate::divisible::Divisible<$crate::underlier::SmallU<$bits>> for $big {
				const LOG_N: usize = (8 * size_of::<$big>() / $bits).ilog2() as usize;

				#[inline]
				fn value_iter(value: Self) -> impl ExactSizeIterator<Item = $crate::underlier::SmallU<$bits>> + Send + Clone {
					$crate::underlier::SmallUDivisIter::new(
						$crate::divisible::Divisible::<u8>::value_iter(value)
					)
				}

				#[inline]
				fn ref_iter(value: &Self) -> impl ExactSizeIterator<Item = $crate::underlier::SmallU<$bits>> + Send + Clone + '_ {
					$crate::underlier::SmallUDivisIter::new(
						$crate::divisible::Divisible::<u8>::ref_iter(value)
					)
				}

				#[inline]
				fn slice_iter(slice: &[Self]) -> impl ExactSizeIterator<Item = $crate::underlier::SmallU<$bits>> + Send + Clone + '_ {
					$crate::underlier::SmallUDivisIter::new(
						$crate::divisible::Divisible::<u8>::slice_iter(slice)
					)
				}

				#[inline]
				unsafe fn get_unchecked(&self, index: usize) -> $crate::underlier::SmallU<$bits> {
					// Safety: the caller guarantees `index < Self::N`.
					unsafe { $crate::underlier::bitmask::get::<Self, $bits>(self, index) }
				}

				#[inline]
				unsafe fn set_unchecked(&mut self, index: usize, val: $crate::underlier::SmallU<$bits>) {
					// Safety: the caller guarantees `index < Self::N`.
					*self = unsafe { $crate::underlier::bitmask::set::<Self, $bits>(*self, index, val) };
				}

				#[inline]
				fn broadcast(val: $crate::underlier::SmallU<$bits>) -> Self {
					// First splat to u8, then splat the byte to fill Self
					let byte = $crate::divisible::Divisible::<$crate::underlier::SmallU<$bits>>::broadcast(val);
					$crate::divisible::Divisible::<u8>::broadcast(byte)
				}

				#[inline]
				fn from_iter(iter: impl Iterator<Item = $crate::underlier::SmallU<$bits>>) -> Self {
					const N: usize = 8 * size_of::<$big>() / $bits;
					let mut result: Self = bytemuck::Zeroable::zeroed();
					for (i, val) in iter.take(N).enumerate() {
						$crate::divisible::Divisible::<$crate::underlier::SmallU<$bits>>::set(&mut result, i, val);
					}
					result
				}
			}
		)+
	};
}

#[allow(unused)]
pub(crate) use impl_divisible_bitmask;

// Implement Divisible using bitmask for SmallU types
impl_divisible_bitmask!(u8, 1, 2, 4);
impl_divisible_bitmask!(u16, 1, 2, 4);
impl_divisible_bitmask!(u32, 1, 2, 4);
impl_divisible_bitmask!(u64, 1, 2, 4);
impl_divisible_bitmask!(u128, 1, 2, 4);

impl_divisible_self!(SmallU<1>, SmallU<2>, SmallU<4>);

// Divisible for SmallU types that subdivide into smaller SmallU types
impl Divisible<SmallU<1>> for SmallU<2> {
	const LOG_N: usize = 1;

	#[inline]
	fn value_iter(value: Self) -> impl ExactSizeIterator<Item = SmallU<1>> + Send + Clone {
		mapget::value_iter(value)
	}

	#[inline]
	fn ref_iter(value: &Self) -> impl ExactSizeIterator<Item = SmallU<1>> + Send + Clone + '_ {
		mapget::value_iter(*value)
	}

	#[inline]
	fn slice_iter(slice: &[Self]) -> impl ExactSizeIterator<Item = SmallU<1>> + Send + Clone + '_ {
		mapget::slice_iter(slice)
	}

	#[inline]
	unsafe fn get_unchecked(&self, index: usize) -> SmallU<1> {
		SmallU::<1>::new(self.val() >> index)
	}

	#[inline]
	unsafe fn set_unchecked(&mut self, index: usize, val: SmallU<1>) {
		let mask = 1u8 << index;
		*self = SmallU::<2>::new((self.val() & !mask) | (val.val() << index));
	}

	#[inline]
	fn broadcast(val: SmallU<1>) -> Self {
		// 0b0 -> 0b00, 0b1 -> 0b11
		let v = val.val();
		SmallU::<2>::new(v | (v << 1))
	}

	#[inline]
	fn from_iter(iter: impl Iterator<Item = SmallU<1>>) -> Self {
		iter.chain(std::iter::repeat(SmallU::<1>::new(0)))
			.take(2)
			.enumerate()
			.fold(SmallU::<2>::new(0), |mut acc, (i, val)| {
				acc.set(i, val);
				acc
			})
	}
}

impl Divisible<SmallU<1>> for SmallU<4> {
	const LOG_N: usize = 2;

	#[inline]
	fn value_iter(value: Self) -> impl ExactSizeIterator<Item = SmallU<1>> + Send + Clone {
		mapget::value_iter(value)
	}

	#[inline]
	fn ref_iter(value: &Self) -> impl ExactSizeIterator<Item = SmallU<1>> + Send + Clone + '_ {
		mapget::value_iter(*value)
	}

	#[inline]
	fn slice_iter(slice: &[Self]) -> impl ExactSizeIterator<Item = SmallU<1>> + Send + Clone + '_ {
		mapget::slice_iter(slice)
	}

	#[inline]
	unsafe fn get_unchecked(&self, index: usize) -> SmallU<1> {
		SmallU::<1>::new(self.val() >> index)
	}

	#[inline]
	unsafe fn set_unchecked(&mut self, index: usize, val: SmallU<1>) {
		let mask = 1u8 << index;
		*self = SmallU::<4>::new((self.val() & !mask) | (val.val() << index));
	}

	#[inline]
	fn broadcast(val: SmallU<1>) -> Self {
		// 0b0 -> 0b0000, 0b1 -> 0b1111
		let mut v = val.val();
		v |= v << 1;
		v |= v << 2;
		SmallU::<4>::new(v)
	}

	#[inline]
	fn from_iter(iter: impl Iterator<Item = SmallU<1>>) -> Self {
		iter.chain(std::iter::repeat(SmallU::<1>::new(0)))
			.take(4)
			.enumerate()
			.fold(SmallU::<4>::new(0), |mut acc, (i, val)| {
				acc.set(i, val);
				acc
			})
	}
}

impl Divisible<SmallU<2>> for SmallU<4> {
	const LOG_N: usize = 1;

	#[inline]
	fn value_iter(value: Self) -> impl ExactSizeIterator<Item = SmallU<2>> + Send + Clone {
		mapget::value_iter(value)
	}

	#[inline]
	fn ref_iter(value: &Self) -> impl ExactSizeIterator<Item = SmallU<2>> + Send + Clone + '_ {
		mapget::value_iter(*value)
	}

	#[inline]
	fn slice_iter(slice: &[Self]) -> impl ExactSizeIterator<Item = SmallU<2>> + Send + Clone + '_ {
		mapget::slice_iter(slice)
	}

	#[inline]
	unsafe fn get_unchecked(&self, index: usize) -> SmallU<2> {
		SmallU::<2>::new(self.val() >> (index * 2))
	}

	#[inline]
	unsafe fn set_unchecked(&mut self, index: usize, val: SmallU<2>) {
		let shift = index * 2;
		let mask = 0b11u8 << shift;
		*self = SmallU::<4>::new((self.val() & !mask) | (val.val() << shift));
	}

	#[inline]
	fn broadcast(val: SmallU<2>) -> Self {
		// 0bXX -> 0bXXXX
		let v = val.val();
		SmallU::<4>::new(v | (v << 2))
	}

	#[inline]
	fn from_iter(iter: impl Iterator<Item = SmallU<2>>) -> Self {
		iter.chain(std::iter::repeat(SmallU::<2>::new(0)))
			.take(2)
			.enumerate()
			.fold(SmallU::<4>::new(0), |mut acc, (i, val)| {
				acc.set(i, val);
				acc
			})
	}
}

#[cfg(test)]
impl<const N: usize> proptest::arbitrary::Arbitrary for SmallU<N> {
	type Parameters = ();
	type Strategy = proptest::strategy::BoxedStrategy<Self>;

	fn arbitrary_with(_args: Self::Parameters) -> Self::Strategy {
		use proptest::strategy::Strategy;

		(0u8..(1u8 << N)).prop_map(Self::new_unchecked).boxed()
	}
}

#[cfg(test)]
mod tests {
	use proptest::{arbitrary::any, proptest};

	use super::*;

	#[test]
	fn test_divisible_u8_u4() {
		let val: u8 = 0x34;

		// Test get - LSB first: nibbles
		assert_eq!(Divisible::<U4>::get(&val, 0), U4::new(0x4));
		assert_eq!(Divisible::<U4>::get(&val, 1), U4::new(0x3));

		// Test set
		let mut modified = val;
		Divisible::<U4>::set(&mut modified, 0, U4::new(0xF));
		assert_eq!(modified, 0x3F);
		let mut modified = val;
		Divisible::<U4>::set(&mut modified, 1, U4::new(0xA));
		assert_eq!(modified, 0xA4);

		// Test ref_iter
		let parts: Vec<U4> = Divisible::<U4>::ref_iter(&val).collect();
		assert_eq!(parts.len(), 2);
		assert_eq!(parts[0], U4::new(0x4));
		assert_eq!(parts[1], U4::new(0x3));

		// Test value_iter
		let parts: Vec<U4> = Divisible::<U4>::value_iter(val).collect();
		assert_eq!(parts.len(), 2);
		assert_eq!(parts[0], U4::new(0x4));
		assert_eq!(parts[1], U4::new(0x3));

		// Test slice_iter
		let vals = [0x34u8, 0x56u8];
		let parts: Vec<U4> = Divisible::<U4>::slice_iter(&vals).collect();
		assert_eq!(parts.len(), 4);
		assert_eq!(parts[0], U4::new(0x4));
		assert_eq!(parts[1], U4::new(0x3));
		assert_eq!(parts[2], U4::new(0x6));
		assert_eq!(parts[3], U4::new(0x5));
	}

	#[test]
	fn test_divisible_u16_u4() {
		let val: u16 = 0x1234;

		// Test get - LSB first: nibbles
		assert_eq!(Divisible::<U4>::get(&val, 0), U4::new(0x4));
		assert_eq!(Divisible::<U4>::get(&val, 1), U4::new(0x3));
		assert_eq!(Divisible::<U4>::get(&val, 2), U4::new(0x2));
		assert_eq!(Divisible::<U4>::get(&val, 3), U4::new(0x1));

		// Test set
		let mut modified = val;
		Divisible::<U4>::set(&mut modified, 1, U4::new(0xF));
		assert_eq!(modified, 0x12F4);

		// Test ref_iter
		let parts: Vec<U4> = Divisible::<U4>::ref_iter(&val).collect();
		assert_eq!(parts.len(), 4);
		assert_eq!(parts[0], U4::new(0x4));
		assert_eq!(parts[3], U4::new(0x1));
	}

	#[test]
	fn test_divisible_u16_u2() {
		// 0b1011_0010_1101_0011 = 0xB2D3
		let val: u16 = 0b1011001011010011;

		// Test get - LSB first: 2-bit chunks
		assert_eq!(Divisible::<U2>::get(&val, 0), U2::new(0b11)); // bits 0-1
		assert_eq!(Divisible::<U2>::get(&val, 1), U2::new(0b00)); // bits 2-3
		assert_eq!(Divisible::<U2>::get(&val, 7), U2::new(0b10)); // bits 14-15

		// Test ref_iter
		let parts: Vec<U2> = Divisible::<U2>::ref_iter(&val).collect();
		assert_eq!(parts.len(), 8);
		assert_eq!(parts[0], U2::new(0b11));
		assert_eq!(parts[7], U2::new(0b10));
	}

	#[test]
	fn test_divisible_u16_u1() {
		// 0b1010_1100_0011_0101 = 0xAC35
		let val: u16 = 0b1010110000110101;

		// Test get - LSB first: individual bits
		assert_eq!(Divisible::<U1>::get(&val, 0), U1::new(1)); // bit 0
		assert_eq!(Divisible::<U1>::get(&val, 1), U1::new(0)); // bit 1
		assert_eq!(Divisible::<U1>::get(&val, 15), U1::new(1)); // bit 15

		// Test set
		let mut modified = val;
		Divisible::<U1>::set(&mut modified, 0, U1::new(0));
		assert_eq!(modified, 0b1010110000110100);

		// Test ref_iter
		let parts: Vec<U1> = Divisible::<U1>::ref_iter(&val).collect();
		assert_eq!(parts.len(), 16);
		assert_eq!(parts[0], U1::new(1));
		assert_eq!(parts[15], U1::new(1));
	}

	#[test]
	fn test_divisible_u64_u4() {
		let val: u64 = 0x123456789ABCDEF0;

		// Test get - LSB first: nibbles
		assert_eq!(Divisible::<U4>::get(&val, 0), U4::new(0x0));
		assert_eq!(Divisible::<U4>::get(&val, 1), U4::new(0xF));
		assert_eq!(Divisible::<U4>::get(&val, 15), U4::new(0x1));

		// Iterating a u64 as nibbles yields 64 / 4 = 16 parts.
		assert_eq!(Divisible::<U4>::ref_iter(&val).count(), 16);
	}

	#[test]
	fn test_broadcast_u8_u4() {
		let result: u8 = Divisible::<U4>::broadcast(U4::new(0x5));
		assert_eq!(result, 0x55);
	}

	#[test]
	fn test_broadcast_u16_u4() {
		let result: u16 = Divisible::<U4>::broadcast(U4::new(0xA));
		assert_eq!(result, 0xAAAA);
	}

	#[test]
	fn test_broadcast_u8_u2() {
		let result: u8 = Divisible::<U2>::broadcast(U2::new(0b11));
		assert_eq!(result, 0xFF);
		let result: u8 = Divisible::<U2>::broadcast(U2::new(0b01));
		assert_eq!(result, 0x55);
	}

	#[test]
	fn test_broadcast_u8_u1() {
		let result: u8 = Divisible::<U1>::broadcast(U1::new(0));
		assert_eq!(result, 0x00);
		let result: u8 = Divisible::<U1>::broadcast(U1::new(1));
		assert_eq!(result, 0xFF);
	}

	#[test]
	fn test_broadcast_smallu2_from_smallu1() {
		let result: SmallU<2> = Divisible::<SmallU<1>>::broadcast(SmallU::<1>::new(0));
		assert_eq!(result.val(), 0b00);
		let result: SmallU<2> = Divisible::<SmallU<1>>::broadcast(SmallU::<1>::new(1));
		assert_eq!(result.val(), 0b11);
	}

	#[test]
	fn test_broadcast_smallu4_from_smallu1() {
		let result: SmallU<4> = Divisible::<SmallU<1>>::broadcast(SmallU::<1>::new(0));
		assert_eq!(result.val(), 0b0000);
		let result: SmallU<4> = Divisible::<SmallU<1>>::broadcast(SmallU::<1>::new(1));
		assert_eq!(result.val(), 0b1111);
	}

	#[test]
	fn test_broadcast_smallu4_from_smallu2() {
		let result: SmallU<4> = Divisible::<SmallU<2>>::broadcast(SmallU::<2>::new(0b10));
		assert_eq!(result.val(), 0b1010);
	}

	#[test]
	fn test_from_iter_smallu() {
		let result: u8 = Divisible::<U4>::from_iter([U4::new(0xA), U4::new(0xB)].into_iter());
		assert_eq!(result, 0xBA);
	}

	#[test]
	fn test_divisible_u32_smallu() {
		let val = 0xab12cd34u32;

		assert_eq!(Divisible::<U1>::get(&val, 0), U1::new(0));
		assert_eq!(Divisible::<U1>::get(&val, 1), U1::new(0));
		assert_eq!(Divisible::<U1>::get(&val, 2), U1::new(1));
		assert_eq!(Divisible::<U1>::get(&val, 31), U1::new(1));

		assert_eq!(Divisible::<U2>::get(&val, 0), U2::new(0));
		assert_eq!(Divisible::<U2>::get(&val, 1), U2::new(1));
		assert_eq!(Divisible::<U2>::get(&val, 2), U2::new(3));
		assert_eq!(Divisible::<U2>::get(&val, 15), U2::new(2));

		assert_eq!(Divisible::<U4>::get(&val, 0), U4::new(4));
		assert_eq!(Divisible::<U4>::get(&val, 1), U4::new(3));
		assert_eq!(Divisible::<U4>::get(&val, 2), U4::new(13));
		assert_eq!(Divisible::<U4>::get(&val, 7), U4::new(10));
	}

	proptest! {
		#[test]
		fn test_set_get_u32_u1(mut val in any::<u32>(), i in 0usize..32, elem in any::<U1>()) {
			Divisible::<U1>::set(&mut val, i, elem);
			assert_eq!(Divisible::<U1>::get(&val, i), elem);
		}

		#[test]
		fn test_set_get_u32_u2(mut val in any::<u32>(), i in 0usize..16, elem in any::<U2>()) {
			Divisible::<U2>::set(&mut val, i, elem);
			assert_eq!(Divisible::<U2>::get(&val, i), elem);
		}

		#[test]
		fn test_set_get_u32_u4(mut val in any::<u32>(), i in 0usize..8, elem in any::<U4>()) {
			Divisible::<U4>::set(&mut val, i, elem);
			assert_eq!(Divisible::<U4>::get(&val, i), elem);
		}
	}
}
