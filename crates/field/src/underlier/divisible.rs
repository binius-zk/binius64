// Copyright 2024-2025 Irreducible Inc.

use std::{
	mem::{align_of, size_of},
	slice::{self, from_raw_parts, from_raw_parts_mut},
};

/// Underlier value that can be split into a slice of smaller `U` values.
/// This trait is unsafe because it allows to reinterpret the memory of a type as a slice of another
/// type.
///
/// # Safety
/// Implementors must ensure that `&Self` can be safely bit-cast to `&[U; Self::WIDTH]` and
/// `&mut Self` can be safely bit-cast to `&mut [U; Self::WIDTH]`.
#[allow(dead_code)]
pub unsafe trait Divisible<U: UnderlierType>: UnderlierType {
	const WIDTH: usize = {
		assert!(size_of::<Self>().is_multiple_of(size_of::<U>()));
		assert!(align_of::<Self>() >= align_of::<U>());
		size_of::<Self>() / size_of::<U>()
	};

	/// This is actually `[U; Self::WIDTH]` but we can't use it as the default value in the trait
	/// definition without `generic_const_exprs` feature enabled.
	type Array: IntoIterator<Item = U, IntoIter: Send + Clone>;

	fn split_val(self) -> Self::Array;
	fn split_ref(&self) -> &[U];
	fn split_mut(&mut self) -> &mut [U];

	fn split_slice(values: &[Self]) -> &[U] {
		let ptr = values.as_ptr() as *const U;
		// Safety: if `&Self` can be reinterpreted as a sequence of `Self::WIDTH` elements of `U`
		// then `&[Self]` can be reinterpreted as a sequence of `Self::Width * values.len()`
		// elements of `U`.
		unsafe { from_raw_parts(ptr, values.len() * Self::WIDTH) }
	}

	fn split_slice_mut(values: &mut [Self]) -> &mut [U] {
		let ptr = values.as_mut_ptr() as *mut U;
		// Safety: if `&mut Self` can be reinterpreted as a sequence of `Self::WIDTH` elements of
		// `U` then `&mut [Self]` can be reinterpreted as a sequence of `Self::Width *
		// values.len()` elements of `U`.
		unsafe { from_raw_parts_mut(ptr, values.len() * Self::WIDTH) }
	}
}

unsafe impl<U: UnderlierType> Divisible<U> for U {
	type Array = [U; 1];

	fn split_val(self) -> Self::Array {
		[self]
	}

	fn split_ref(&self) -> &[U] {
		slice::from_ref(self)
	}

	fn split_mut(&mut self) -> &mut [U] {
		slice::from_mut(self)
	}
}

/// Divides an underlier type into smaller underliers in memory and iterates over them.
///
/// [`DivisIterable`] (say that 10 times, fast) provides iteration over the subdivisions of an
/// underlier type, guaranteeing that iteration proceeds from the least significant bits to the most
/// significant bits, regardless of the CPU architecture's endianness.
///
/// # Endianness Handling
///
/// To ensure consistent LSB-to-MSB iteration order across all platforms:
/// - On little-endian systems: elements are naturally ordered LSB-to-MSB in memory, so iteration
///   proceeds forward through the array
/// - On big-endian systems: elements are ordered MSB-to-LSB in memory, so iteration is reversed to
///   achieve LSB-to-MSB order
///
/// This abstraction allows code to work with subdivided underliers in a platform-independent way
/// while maintaining the invariant that the first element always represents the least significant
/// portion of the value.
pub trait DivisIterable<T>: Copy {
	/// The log2 of the number of `T` elements that fit in `Self`.
	const LOG_N: usize;

	/// The number of `T` elements that fit in `Self`.
	const N: usize = 1 << Self::LOG_N;

	/// Returns an iterator over subdivisions of this underlier value, ordered from LSB to MSB.
	fn value_iter(value: Self) -> impl ExactSizeIterator<Item = T> + Send + Clone;

	/// Returns an iterator over subdivisions of this underlier reference, ordered from LSB to MSB.
	fn ref_iter(value: &Self) -> impl ExactSizeIterator<Item = T> + Send + Clone + '_;

	/// Returns an iterator over subdivisions of a slice of underliers, ordered from LSB to MSB.
	fn slice_iter(slice: &[Self]) -> impl ExactSizeIterator<Item = T> + Send + Clone + '_;

	/// Get element at index (LSB-first ordering).
	///
	/// # Panics
	///
	/// Panics if `index >= Self::N`.
	fn get(self, index: usize) -> T;

	/// Set element at index (LSB-first ordering), returning modified value.
	///
	/// # Panics
	///
	/// Panics if `index >= Self::N`.
	fn set(self, index: usize, val: T) -> Self;
}

/// Helper functions for DivisIterable implementations using bytemuck memory casting.
///
/// These functions handle the endianness-aware iteration over subdivisions of an underlier type.
pub mod memcast {
	use bytemuck::Pod;

	/// Returns an iterator over subdivisions of a value, ordered from LSB to MSB.
	#[cfg(target_endian = "little")]
	#[inline]
	pub fn value_iter<Big, Small, const N: usize>(
		value: Big,
	) -> impl ExactSizeIterator<Item = Small> + Send + Clone
	where
		Big: Pod,
		Small: Pod + Send,
	{
		bytemuck::must_cast::<Big, [Small; N]>(value).into_iter()
	}

	/// Returns an iterator over subdivisions of a value, ordered from LSB to MSB.
	#[cfg(target_endian = "big")]
	#[inline]
	pub fn value_iter<Big, Small, const N: usize>(
		value: Big,
	) -> impl ExactSizeIterator<Item = Small> + Send + Clone
	where
		Big: Pod,
		Small: Pod + Send,
	{
		bytemuck::must_cast::<Big, [Small; N]>(value).into_iter().rev()
	}

	/// Returns an iterator over subdivisions of a reference, ordered from LSB to MSB.
	#[cfg(target_endian = "little")]
	#[inline]
	pub fn ref_iter<Big, Small, const N: usize>(
		value: &Big,
	) -> impl ExactSizeIterator<Item = Small> + Send + Clone + '_
	where
		Big: Pod,
		Small: Pod + Send + Sync,
	{
		bytemuck::must_cast_ref::<Big, [Small; N]>(value)
			.iter()
			.copied()
	}

	/// Returns an iterator over subdivisions of a reference, ordered from LSB to MSB.
	#[cfg(target_endian = "big")]
	#[inline]
	pub fn ref_iter<Big, Small, const N: usize>(
		value: &Big,
	) -> impl ExactSizeIterator<Item = Small> + Send + Clone + '_
	where
		Big: Pod,
		Small: Pod + Send + Sync,
	{
		bytemuck::must_cast_ref::<Big, [Small; N]>(value)
			.iter()
			.rev()
			.copied()
	}

	/// Returns an iterator over subdivisions of a slice, ordered from LSB to MSB.
	#[cfg(target_endian = "little")]
	#[inline]
	pub fn slice_iter<Big, Small>(
		slice: &[Big],
	) -> impl ExactSizeIterator<Item = Small> + Send + Clone + '_
	where
		Big: Pod,
		Small: Pod + Send + Sync,
	{
		bytemuck::must_cast_slice::<Big, Small>(slice)
			.iter()
			.copied()
	}

	/// Returns an iterator over subdivisions of a slice, ordered from LSB to MSB.
	///
	/// For big-endian: iterate through the raw slice, but for each element's
	/// subdivisions, reverse the index to maintain LSB-first ordering.
	#[cfg(target_endian = "big")]
	#[inline]
	pub fn slice_iter<Big, Small, const LOG_N: usize>(
		slice: &[Big],
	) -> impl ExactSizeIterator<Item = Small> + Send + Clone + '_
	where
		Big: Pod,
		Small: Pod + Send + Sync,
	{
		const N: usize = 1 << LOG_N;
		let raw_slice = bytemuck::must_cast_slice::<Big, Small>(slice);
		(0..raw_slice.len()).map(move |i| {
			let element_idx = i >> LOG_N;
			let sub_idx = i & (N - 1);
			let reversed_sub_idx = N - 1 - sub_idx;
			let raw_idx = element_idx * N + reversed_sub_idx;
			raw_slice[raw_idx]
		})
	}

	/// Get element at index (LSB-first ordering).
	#[cfg(target_endian = "little")]
	#[inline]
	pub fn get<Big, Small, const N: usize>(value: &Big, index: usize) -> Small
	where
		Big: Pod,
		Small: Pod,
	{
		bytemuck::must_cast_ref::<Big, [Small; N]>(value)[index]
	}

	/// Get element at index (LSB-first ordering).
	#[cfg(target_endian = "big")]
	#[inline]
	pub fn get<Big, Small, const N: usize>(value: &Big, index: usize) -> Small
	where
		Big: Pod,
		Small: Pod,
	{
		bytemuck::must_cast_ref::<Big, [Small; N]>(value)[N - 1 - index]
	}

	/// Set element at index (LSB-first ordering), returning modified value.
	#[cfg(target_endian = "little")]
	#[inline]
	pub fn set<Big, Small, const N: usize>(value: &Big, index: usize, val: Small) -> Big
	where
		Big: Pod,
		Small: Pod,
	{
		let mut arr = *bytemuck::must_cast_ref::<Big, [Small; N]>(value);
		arr[index] = val;
		bytemuck::must_cast(arr)
	}

	/// Set element at index (LSB-first ordering), returning modified value.
	#[cfg(target_endian = "big")]
	#[inline]
	pub fn set<Big, Small, const N: usize>(value: &Big, index: usize, val: Small) -> Big
	where
		Big: Pod,
		Small: Pod,
	{
		let mut arr = *bytemuck::must_cast_ref::<Big, [Small; N]>(value);
		arr[N - 1 - index] = val;
		bytemuck::must_cast(arr)
	}
}

/// Helper functions for DivisIterable implementations using bitmask operations on sub-byte elements.
///
/// These functions work on any type that implements `DivisIterable<u8>` by extracting
/// and modifying sub-byte elements through the byte interface.
pub mod bitmask {
	use super::{DivisIterable, SmallU};

	/// Get a sub-byte element at index (LSB-first ordering).
	#[inline]
	pub fn get<Big, const BITS: usize>(value: Big, index: usize) -> SmallU<BITS>
	where
		Big: DivisIterable<u8>,
	{
		let elems_per_byte = 8 / BITS;
		let byte_index = index / elems_per_byte;
		let sub_index = index % elems_per_byte;
		let byte = DivisIterable::<u8>::get(value, byte_index);
		let shift = sub_index * BITS;
		SmallU::<BITS>::new(byte >> shift)
	}

	/// Set a sub-byte element at index (LSB-first ordering), returning modified value.
	#[inline]
	pub fn set<Big, const BITS: usize>(value: Big, index: usize, val: SmallU<BITS>) -> Big
	where
		Big: DivisIterable<u8>,
	{
		let elems_per_byte = 8 / BITS;
		let byte_index = index / elems_per_byte;
		let sub_index = index % elems_per_byte;
		let byte = DivisIterable::<u8>::get(value, byte_index);
		let shift = sub_index * BITS;
		let mask = (1u8 << BITS) - 1;
		let new_byte = (byte & !(mask << shift)) | (val.val() << shift);
		DivisIterable::<u8>::set(value, byte_index, new_byte)
	}
}

/// Iterator for dividing an underlier into sub-byte elements (SmallU<N>).
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

/// Implements `Divisible` trait for a bigger type over smaller types using bytemuck casting.
///
/// This macro generates `Divisible` implementations for the big type and all smaller types,
/// as well as for `ScaledUnderlier<_, 2>` and `ScaledUnderlier<ScaledUnderlier<_, 2>, 2>` variants.
macro_rules! impl_divisible {
	(@pairs $name:ty,?) => {};
	(@pairs $bigger:ty, $smaller:ty) => {
		unsafe impl $crate::underlier::Divisible<$smaller> for $bigger {
			type Array = [$smaller; {size_of::<Self>() / size_of::<$smaller>()}];

			fn split_val(self) -> Self::Array {
				bytemuck::must_cast::<_, Self::Array>(self)
			}

			fn split_ref(&self) -> &[$smaller] {
				bytemuck::must_cast_ref::<_, [$smaller;{(<$bigger>::BITS as usize / <$smaller>::BITS as usize ) }]>(self)
			}

			fn split_mut(&mut self) -> &mut [$smaller] {
				bytemuck::must_cast_mut::<_, [$smaller;{(<$bigger>::BITS as usize / <$smaller>::BITS as usize ) }]>(self)
			}
		}

		unsafe impl $crate::underlier::Divisible<$smaller> for $crate::underlier::ScaledUnderlier<$bigger, 2> {
			type Array = [$smaller; {2 * size_of::<$bigger>() / size_of::<$smaller>()}];

			fn split_val(self) -> Self::Array {
				bytemuck::must_cast::<_, Self::Array>(self)
			}

			fn split_ref(&self) -> &[$smaller] {
				bytemuck::must_cast_ref::<_, [$smaller;{(2 * <$bigger>::BITS as usize / <$smaller>::BITS as usize ) }]>(&self.0)
			}

			fn split_mut(&mut self) -> &mut [$smaller] {
				bytemuck::must_cast_mut::<_, [$smaller;{(2 * <$bigger>::BITS as usize / <$smaller>::BITS as usize ) }]>(&mut self.0)
			}
		}

		unsafe impl $crate::underlier::Divisible<$smaller> for $crate::underlier::ScaledUnderlier<$crate::underlier::ScaledUnderlier<$bigger, 2>, 2> {
			type Array = [$smaller; {4 * size_of::<$bigger>() / size_of::<$smaller>()}];

			fn split_val(self) -> Self::Array {
				bytemuck::must_cast::<_, Self::Array>(self)
			}

			fn split_ref(&self) -> &[$smaller] {
				bytemuck::must_cast_ref::<_, [$smaller;{(4 * <$bigger>::BITS as usize / <$smaller>::BITS as usize ) }]>(&self.0)
			}

			fn split_mut(&mut self) -> &mut [$smaller] {
				bytemuck::must_cast_mut::<_, [$smaller;{(4 * <$bigger>::BITS as usize / <$smaller>::BITS as usize ) }]>(&mut self.0)
			}
		}
	};

	(@pairs $first:ty, $second:ty, $($tail:ty),*) => {
		impl_divisible!(@pairs $first, $second);
		impl_divisible!(@pairs $first, $($tail),*);
	};
	($_:ty) => {};
	($head:ty, $($tail:ty),*) => {
		impl_divisible!(@pairs $head, $($tail),*);
		impl_divisible!($($tail),*);
	}
}

#[allow(unused)]
pub(crate) use impl_divisible;

/// Implements `DivisIterable` trait using bytemuck memory casting.
///
/// This macro generates `DivisIterable` implementations for a big type over smaller types.
/// The implementations use the helper functions in the `memcast` module.
macro_rules! impl_divis_iterable_memcast {
	($big:ty, $($small:ty),+) => {
		$(
			impl $crate::underlier::DivisIterable<$small> for $big {
				const LOG_N: usize = (size_of::<$big>() / size_of::<$small>()).ilog2() as usize;

				#[inline]
				fn value_iter(value: Self) -> impl ExactSizeIterator<Item = $small> + Send + Clone {
					const N: usize = size_of::<$big>() / size_of::<$small>();
					$crate::underlier::memcast::value_iter::<$big, $small, N>(value)
				}

				#[inline]
				fn ref_iter(value: &Self) -> impl ExactSizeIterator<Item = $small> + Send + Clone + '_ {
					const N: usize = size_of::<$big>() / size_of::<$small>();
					$crate::underlier::memcast::ref_iter::<$big, $small, N>(value)
				}

				#[inline]
				#[cfg(target_endian = "little")]
				fn slice_iter(slice: &[Self]) -> impl ExactSizeIterator<Item = $small> + Send + Clone + '_ {
					$crate::underlier::memcast::slice_iter::<$big, $small>(slice)
				}

				#[inline]
				#[cfg(target_endian = "big")]
				fn slice_iter(slice: &[Self]) -> impl ExactSizeIterator<Item = $small> + Send + Clone + '_ {
					const LOG_N: usize = (size_of::<$big>() / size_of::<$small>()).ilog2() as usize;
					$crate::underlier::memcast::slice_iter::<$big, $small, LOG_N>(slice)
				}

				#[inline]
				fn get(self, index: usize) -> $small {
					const N: usize = size_of::<$big>() / size_of::<$small>();
					$crate::underlier::memcast::get::<$big, $small, N>(&self, index)
				}

				#[inline]
				fn set(self, index: usize, val: $small) -> Self {
					const N: usize = size_of::<$big>() / size_of::<$small>();
					$crate::underlier::memcast::set::<$big, $small, N>(&self, index, val)
				}
			}
		)+
	};
}

#[allow(unused)]
pub(crate) use impl_divis_iterable_memcast;

/// Implements `DivisIterable` trait for SmallU types using bitmask operations.
///
/// This macro generates `DivisIterable<SmallU<BITS>>` implementations for a big type
/// by wrapping byte iteration with bitmasking to extract sub-byte elements.
macro_rules! impl_divis_iterable_bitmask {
	// Special case for u8: operates directly on the byte without needing DivisIterable::<u8>
	(u8, $($bits:expr),+) => {
		$(
			impl $crate::underlier::DivisIterable<$crate::underlier::SmallU<$bits>> for u8 {
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
				fn get(self, index: usize) -> $crate::underlier::SmallU<$bits> {
					let shift = index * $bits;
					$crate::underlier::SmallU::<$bits>::new(self >> shift)
				}

				#[inline]
				fn set(self, index: usize, val: $crate::underlier::SmallU<$bits>) -> Self {
					let shift = index * $bits;
					let mask = (1u8 << $bits) - 1;
					(self & !(mask << shift)) | (val.val() << shift)
				}
			}
		)+
	};

	// General case for types larger than u8: wraps byte iteration
	($big:ty, $($bits:expr),+) => {
		$(
			impl $crate::underlier::DivisIterable<$crate::underlier::SmallU<$bits>> for $big {
				const LOG_N: usize = (8 * size_of::<$big>() / $bits).ilog2() as usize;

				#[inline]
				fn value_iter(value: Self) -> impl ExactSizeIterator<Item = $crate::underlier::SmallU<$bits>> + Send + Clone {
					$crate::underlier::SmallUDivisIter::new(
						$crate::underlier::DivisIterable::<u8>::value_iter(value)
					)
				}

				#[inline]
				fn ref_iter(value: &Self) -> impl ExactSizeIterator<Item = $crate::underlier::SmallU<$bits>> + Send + Clone + '_ {
					$crate::underlier::SmallUDivisIter::new(
						$crate::underlier::DivisIterable::<u8>::ref_iter(value)
					)
				}

				#[inline]
				fn slice_iter(slice: &[Self]) -> impl ExactSizeIterator<Item = $crate::underlier::SmallU<$bits>> + Send + Clone + '_ {
					$crate::underlier::SmallUDivisIter::new(
						$crate::underlier::DivisIterable::<u8>::slice_iter(slice)
					)
				}

				#[inline]
				fn get(self, index: usize) -> $crate::underlier::SmallU<$bits> {
					$crate::underlier::bitmask::get::<Self, $bits>(self, index)
				}

				#[inline]
				fn set(self, index: usize, val: $crate::underlier::SmallU<$bits>) -> Self {
					$crate::underlier::bitmask::set::<Self, $bits>(self, index, val)
				}
			}
		)+
	};
}

#[allow(unused)]
pub(crate) use impl_divis_iterable_bitmask;

use super::{UnderlierType, small_uint::SmallU};

// Implement Divisible trait for primitive types
impl_divisible!(u128, u64, u32, u16, u8);

// Implement DivisIterable using memcast for primitive types
impl_divis_iterable_memcast!(u128, u64, u32, u16, u8);
impl_divis_iterable_memcast!(u64, u32, u16, u8);
impl_divis_iterable_memcast!(u32, u16, u8);
impl_divis_iterable_memcast!(u16, u8);

// Implement DivisIterable using bitmask for SmallU types
impl_divis_iterable_bitmask!(u8, 1, 2, 4);
impl_divis_iterable_bitmask!(u16, 1, 2, 4);
impl_divis_iterable_bitmask!(u32, 1, 2, 4);
impl_divis_iterable_bitmask!(u64, 1, 2, 4);
impl_divis_iterable_bitmask!(u128, 1, 2, 4);

#[cfg(test)]
mod tests {
	use super::*;
	use crate::underlier::small_uint::{U1, U2, U4};

	#[test]
	fn test_divisiterable_u8_u4() {
		let val: u8 = 0x34;

		// Test get - LSB first: nibbles
		assert_eq!(DivisIterable::<U4>::get(val, 0), U4::new(0x4));
		assert_eq!(DivisIterable::<U4>::get(val, 1), U4::new(0x3));

		// Test set
		let modified = DivisIterable::<U4>::set(val, 0, U4::new(0xF));
		assert_eq!(modified, 0x3F);
		let modified = DivisIterable::<U4>::set(val, 1, U4::new(0xA));
		assert_eq!(modified, 0xA4);

		// Test ref_iter
		let parts: Vec<U4> = DivisIterable::<U4>::ref_iter(&val).collect();
		assert_eq!(parts.len(), 2);
		assert_eq!(parts[0], U4::new(0x4));
		assert_eq!(parts[1], U4::new(0x3));

		// Test value_iter
		let parts: Vec<U4> = DivisIterable::<U4>::value_iter(val).collect();
		assert_eq!(parts.len(), 2);
		assert_eq!(parts[0], U4::new(0x4));
		assert_eq!(parts[1], U4::new(0x3));

		// Test slice_iter
		let vals = [0x34u8, 0x56u8];
		let parts: Vec<U4> = DivisIterable::<U4>::slice_iter(&vals).collect();
		assert_eq!(parts.len(), 4);
		assert_eq!(parts[0], U4::new(0x4));
		assert_eq!(parts[1], U4::new(0x3));
		assert_eq!(parts[2], U4::new(0x6));
		assert_eq!(parts[3], U4::new(0x5));
	}

	#[test]
	fn test_divisiterable_u16_u4() {
		let val: u16 = 0x1234;

		// Test get - LSB first: nibbles
		assert_eq!(DivisIterable::<U4>::get(val, 0), U4::new(0x4));
		assert_eq!(DivisIterable::<U4>::get(val, 1), U4::new(0x3));
		assert_eq!(DivisIterable::<U4>::get(val, 2), U4::new(0x2));
		assert_eq!(DivisIterable::<U4>::get(val, 3), U4::new(0x1));

		// Test set
		let modified = DivisIterable::<U4>::set(val, 1, U4::new(0xF));
		assert_eq!(modified, 0x12F4);

		// Test ref_iter
		let parts: Vec<U4> = DivisIterable::<U4>::ref_iter(&val).collect();
		assert_eq!(parts.len(), 4);
		assert_eq!(parts[0], U4::new(0x4));
		assert_eq!(parts[3], U4::new(0x1));
	}

	#[test]
	fn test_divisiterable_u16_u2() {
		// 0b1011_0010_1101_0011 = 0xB2D3
		let val: u16 = 0b1011001011010011;

		// Test get - LSB first: 2-bit chunks
		assert_eq!(DivisIterable::<U2>::get(val, 0), U2::new(0b11)); // bits 0-1
		assert_eq!(DivisIterable::<U2>::get(val, 1), U2::new(0b00)); // bits 2-3
		assert_eq!(DivisIterable::<U2>::get(val, 7), U2::new(0b10)); // bits 14-15

		// Test ref_iter
		let parts: Vec<U2> = DivisIterable::<U2>::ref_iter(&val).collect();
		assert_eq!(parts.len(), 8);
		assert_eq!(parts[0], U2::new(0b11));
		assert_eq!(parts[7], U2::new(0b10));
	}

	#[test]
	fn test_divisiterable_u16_u1() {
		// 0b1010_1100_0011_0101 = 0xAC35
		let val: u16 = 0b1010110000110101;

		// Test get - LSB first: individual bits
		assert_eq!(DivisIterable::<U1>::get(val, 0), U1::new(1)); // bit 0
		assert_eq!(DivisIterable::<U1>::get(val, 1), U1::new(0)); // bit 1
		assert_eq!(DivisIterable::<U1>::get(val, 15), U1::new(1)); // bit 15

		// Test set
		let modified = DivisIterable::<U1>::set(val, 0, U1::new(0));
		assert_eq!(modified, 0b1010110000110100);

		// Test ref_iter
		let parts: Vec<U1> = DivisIterable::<U1>::ref_iter(&val).collect();
		assert_eq!(parts.len(), 16);
		assert_eq!(parts[0], U1::new(1));
		assert_eq!(parts[15], U1::new(1));
	}

	#[test]
	fn test_divisiterable_u64_u4() {
		let val: u64 = 0x123456789ABCDEF0;

		// Test get - LSB first: nibbles
		assert_eq!(DivisIterable::<U4>::get(val, 0), U4::new(0x0));
		assert_eq!(DivisIterable::<U4>::get(val, 1), U4::new(0xF));
		assert_eq!(DivisIterable::<U4>::get(val, 15), U4::new(0x1));

		// Test ref_iter
		let parts: Vec<U4> = DivisIterable::<U4>::ref_iter(&val).collect();
		assert_eq!(parts.len(), 16);
	}

	#[test]
	fn test_divisiterable_u32_u8_slice() {
		let vals: [u32; 2] = [0x04030201, 0x08070605];

		// Test slice_iter
		let parts: Vec<u8> = DivisIterable::<u8>::slice_iter(&vals).collect();
		assert_eq!(parts.len(), 8);
		// LSB-first ordering within each u32
		assert_eq!(parts[0], 0x01);
		assert_eq!(parts[1], 0x02);
		assert_eq!(parts[2], 0x03);
		assert_eq!(parts[3], 0x04);
		assert_eq!(parts[4], 0x05);
		assert_eq!(parts[5], 0x06);
		assert_eq!(parts[6], 0x07);
		assert_eq!(parts[7], 0x08);
	}
}
