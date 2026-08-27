// Copyright 2024-2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use std::mem::size_of;

/// Divides an underlier type into smaller underliers in memory and iterates over them.
///
/// [`Divisible`] provides iteration over the subdivisions of an underlier type, guaranteeing that
/// iteration proceeds from the least significant bits to the most significant bits, regardless of
/// the CPU architecture's endianness.
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
pub trait Divisible<T>: Sized {
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
	#[inline]
	fn get(&self, index: usize) -> T {
		assert!(index < Self::N, "index {index} out of bounds (N = {})", Self::N);
		// Safety: `index < Self::N` checked above.
		unsafe { self.get_unchecked(index) }
	}

	/// Set element at index (LSB-first ordering), in place.
	///
	/// # Panics
	///
	/// Panics if `index >= Self::N`.
	#[inline]
	fn set(&mut self, index: usize, val: T) {
		assert!(index < Self::N, "index {index} out of bounds (N = {})", Self::N);
		// Safety: `index < Self::N` checked above.
		unsafe { self.set_unchecked(index, val) };
	}

	/// Get element at index (LSB-first ordering) without bounds checking.
	///
	/// # Safety
	///
	/// The caller must ensure that `index < Self::N`.
	unsafe fn get_unchecked(&self, index: usize) -> T;

	/// Set element at index (LSB-first ordering) in place, without bounds checking.
	///
	/// # Safety
	///
	/// The caller must ensure that `index < Self::N`.
	unsafe fn set_unchecked(&mut self, index: usize, val: T);

	/// Create a value with `val` broadcast to all `N` positions.
	fn broadcast(val: T) -> Self;

	/// Construct a value from an iterator of elements.
	///
	/// Consumes at most `N` elements from the iterator. If the iterator
	/// yields fewer than `N` elements, remaining positions are filled with zeros.
	fn from_iter(iter: impl Iterator<Item = T>) -> Self;
}

/// Helper functions for Divisible implementations using bytemuck memory casting.
///
/// These functions handle the endianness-aware iteration over subdivisions of an underlier type.
pub mod memcast {
	use bytemuck::{Pod, Zeroable};

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
		bytemuck::must_cast::<Big, [Small; N]>(value)
			.into_iter()
			.rev()
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

	/// Get element at index (LSB-first ordering) without bounds checking.
	///
	/// # Safety
	///
	/// The caller must ensure that `index < N`.
	#[cfg(target_endian = "little")]
	#[inline]
	pub unsafe fn get<Big, Small, const N: usize>(value: &Big, index: usize) -> Small
	where
		Big: Pod,
		Small: Pod,
	{
		// Safety: the caller guarantees `index < N`.
		unsafe { *bytemuck::must_cast_ref::<Big, [Small; N]>(value).get_unchecked(index) }
	}

	/// Get element at index (LSB-first ordering) without bounds checking.
	///
	/// # Safety
	///
	/// The caller must ensure that `index < N`.
	#[cfg(target_endian = "big")]
	#[inline]
	pub unsafe fn get<Big, Small, const N: usize>(value: &Big, index: usize) -> Small
	where
		Big: Pod,
		Small: Pod,
	{
		// Safety: the caller guarantees `index < N`, so `N - 1 - index < N`.
		unsafe { *bytemuck::must_cast_ref::<Big, [Small; N]>(value).get_unchecked(N - 1 - index) }
	}

	/// Set element at index (LSB-first ordering), returning modified value, without bounds
	/// checking.
	///
	/// # Safety
	///
	/// The caller must ensure that `index < N`.
	#[cfg(target_endian = "little")]
	#[inline]
	pub unsafe fn set<Big, Small, const N: usize>(value: &Big, index: usize, val: Small) -> Big
	where
		Big: Pod,
		Small: Pod,
	{
		let mut arr = *bytemuck::must_cast_ref::<Big, [Small; N]>(value);
		// Safety: the caller guarantees `index < N`.
		unsafe {
			*arr.get_unchecked_mut(index) = val;
		}
		bytemuck::must_cast(arr)
	}

	/// Set element at index (LSB-first ordering), returning modified value, without bounds
	/// checking.
	///
	/// # Safety
	///
	/// The caller must ensure that `index < N`.
	#[cfg(target_endian = "big")]
	#[inline]
	pub unsafe fn set<Big, Small, const N: usize>(value: &Big, index: usize, val: Small) -> Big
	where
		Big: Pod,
		Small: Pod,
	{
		let mut arr = *bytemuck::must_cast_ref::<Big, [Small; N]>(value);
		// Safety: the caller guarantees `index < N`, so `N - 1 - index < N`.
		unsafe {
			*arr.get_unchecked_mut(N - 1 - index) = val;
		}
		bytemuck::must_cast(arr)
	}

	/// Broadcast a value to all positions.
	#[inline]
	pub fn broadcast<Big, Small, const N: usize>(val: Small) -> Big
	where
		Big: Pod,
		Small: Pod + Copy,
	{
		bytemuck::must_cast::<[Small; N], Big>([val; N])
	}

	/// Construct a value from an iterator of elements.
	#[cfg(target_endian = "little")]
	#[inline]
	pub fn from_iter<Big, Small, const N: usize>(iter: impl Iterator<Item = Small>) -> Big
	where
		Big: Pod,
		Small: Pod,
	{
		let mut arr: [Small; N] = Zeroable::zeroed();
		for (i, val) in iter.take(N).enumerate() {
			arr[i] = val;
		}
		bytemuck::must_cast(arr)
	}

	/// Construct a value from an iterator of elements.
	#[cfg(target_endian = "big")]
	#[inline]
	pub fn from_iter<Big, Small, const N: usize>(iter: impl Iterator<Item = Small>) -> Big
	where
		Big: Pod,
		Small: Pod,
	{
		let mut arr: [Small; N] = Zeroable::zeroed();
		for (i, val) in iter.take(N).enumerate() {
			arr[N - 1 - i] = val;
		}
		bytemuck::must_cast(arr)
	}
}

/// Helper functions for Divisible implementations using the get method.
///
/// These functions create iterators by mapping indices through `Divisible::get`,
/// useful for SIMD types where extract intrinsics provide efficient element access.
pub mod mapget {
	use binius_utils::iter::IterExtensions;

	use super::Divisible;

	/// Create an iterator over subdivisions by mapping get over indices.
	#[inline]
	pub fn value_iter<Big, Small>(value: Big) -> impl ExactSizeIterator<Item = Small> + Send + Clone
	where
		Big: Divisible<Small> + Send + Clone,
		Small: Send,
	{
		(0..Big::N).map_skippable(move |i| Divisible::<Small>::get(&value, i))
	}

	/// Create a slice iterator by computing global index and using get.
	#[inline]
	pub fn slice_iter<Big, Small>(
		slice: &[Big],
	) -> impl ExactSizeIterator<Item = Small> + Send + Clone + '_
	where
		Big: Divisible<Small> + Send + Sync,
		Small: Send,
	{
		let total = slice.len() * Big::N;
		(0..total).map_skippable(move |global_idx| {
			let elem_idx = global_idx / Big::N;
			let sub_idx = global_idx % Big::N;
			Divisible::<Small>::get(&slice[elem_idx], sub_idx)
		})
	}
}

/// Implements `Divisible` trait using bytemuck memory casting.
///
/// This macro generates `Divisible` implementations for a big type over smaller types.
/// The implementations use the helper functions in the `memcast` module.
macro_rules! impl_divisible_memcast {
	($big:ty, $($small:ty),+) => {
		$(
			impl $crate::divisible::Divisible<$small> for $big {
				const LOG_N: usize = (size_of::<$big>() / size_of::<$small>()).ilog2() as usize;

				#[inline]
				fn value_iter(value: Self) -> impl ExactSizeIterator<Item = $small> + Send + Clone {
					const N: usize = size_of::<$big>() / size_of::<$small>();
					$crate::divisible::memcast::value_iter::<$big, $small, N>(value)
				}

				#[inline]
				fn ref_iter(value: &Self) -> impl ExactSizeIterator<Item = $small> + Send + Clone + '_ {
					const N: usize = size_of::<$big>() / size_of::<$small>();
					$crate::divisible::memcast::ref_iter::<$big, $small, N>(value)
				}

				#[inline]
				#[cfg(target_endian = "little")]
				fn slice_iter(slice: &[Self]) -> impl ExactSizeIterator<Item = $small> + Send + Clone + '_ {
					$crate::divisible::memcast::slice_iter::<$big, $small>(slice)
				}

				#[inline]
				#[cfg(target_endian = "big")]
				fn slice_iter(slice: &[Self]) -> impl ExactSizeIterator<Item = $small> + Send + Clone + '_ {
					const LOG_N: usize = (size_of::<$big>() / size_of::<$small>()).ilog2() as usize;
					$crate::divisible::memcast::slice_iter::<$big, $small, LOG_N>(slice)
				}

				#[inline]
				unsafe fn get_unchecked(&self, index: usize) -> $small {
					const N: usize = size_of::<$big>() / size_of::<$small>();
					// Safety: the caller guarantees `index < Self::N == N`.
					unsafe { $crate::divisible::memcast::get::<$big, $small, N>(self, index) }
				}

				#[inline]
				unsafe fn set_unchecked(&mut self, index: usize, val: $small) {
					const N: usize = size_of::<$big>() / size_of::<$small>();
					// Safety: the caller guarantees `index < Self::N == N`.
					*self = unsafe { $crate::divisible::memcast::set::<$big, $small, N>(&*self, index, val) };
				}

				#[inline]
				fn broadcast(val: $small) -> Self {
					const N: usize = size_of::<$big>() / size_of::<$small>();
					$crate::divisible::memcast::broadcast::<$big, $small, N>(val)
				}

				#[inline]
				fn from_iter(iter: impl Iterator<Item = $small>) -> Self {
					const N: usize = size_of::<$big>() / size_of::<$small>();
					$crate::divisible::memcast::from_iter::<$big, $small, N>(iter)
				}
			}
		)+
	};
}

#[allow(unused)]
pub(crate) use impl_divisible_memcast;

// Implement Divisible using memcast for primitive types
impl_divisible_memcast!(u128, u64, u32, u16, u8);
impl_divisible_memcast!(u64, u32, u16, u8);
impl_divisible_memcast!(u32, u16, u8);
impl_divisible_memcast!(u16, u8);

/// Implements reflexive `Divisible<Self>` for a type (dividing into itself once).
macro_rules! impl_divisible_self {
	($($ty:ty),+) => {
		$(
			impl Divisible<$ty> for $ty {
				const LOG_N: usize = 0;

				#[inline]
				fn value_iter(value: Self) -> impl ExactSizeIterator<Item = $ty> + Send + Clone {
					std::iter::once(value)
				}

				#[inline]
				fn ref_iter(value: &Self) -> impl ExactSizeIterator<Item = $ty> + Send + Clone + '_ {
					std::iter::once(*value)
				}

				#[inline]
				fn slice_iter(slice: &[Self]) -> impl ExactSizeIterator<Item = $ty> + Send + Clone + '_ {
					slice.iter().copied()
				}

				#[inline]
				unsafe fn get_unchecked(&self, _index: usize) -> $ty {
					*self
				}

				#[inline]
				unsafe fn set_unchecked(&mut self, _index: usize, val: $ty) {
					*self = val;
				}

				#[inline]
				fn broadcast(val: $ty) -> Self {
					val
				}

				#[inline]
				fn from_iter(mut iter: impl Iterator<Item = $ty>) -> Self {
					iter.next().unwrap_or_else(bytemuck::Zeroable::zeroed)
				}
			}
		)+
	};
}

#[allow(unused)]
pub(crate) use impl_divisible_self;

impl_divisible_self!(u8, u16, u32, u64, u128);

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn test_divisible_u32_u8_slice() {
		let vals: [u32; 2] = [0x04030201, 0x08070605];

		// Test slice_iter
		let parts: Vec<u8> = Divisible::<u8>::slice_iter(&vals).collect();
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

	#[test]
	fn test_broadcast_u32_u8() {
		let result: u32 = Divisible::<u8>::broadcast(0xAB);
		assert_eq!(result, 0xABABABAB);
	}

	#[test]
	fn test_broadcast_u64_u16() {
		let result: u64 = Divisible::<u16>::broadcast(0x1234);
		assert_eq!(result, 0x1234123412341234);
	}

	#[test]
	fn test_broadcast_u128_u32() {
		let result: u128 = Divisible::<u32>::broadcast(0xDEADBEEF);
		assert_eq!(result, 0xDEADBEEFDEADBEEFDEADBEEFDEADBEEF);
	}

	#[test]
	fn test_broadcast_reflexive() {
		let result: u64 = Divisible::<u64>::broadcast(0x123456789ABCDEF0);
		assert_eq!(result, 0x123456789ABCDEF0);
	}

	#[test]
	fn test_from_iter_full() {
		let result: u32 = Divisible::<u8>::from_iter([0x01, 0x02, 0x03, 0x04].into_iter());
		assert_eq!(result, 0x04030201);
	}

	#[test]
	fn test_from_iter_partial() {
		// Only 2 elements, remaining should be 0
		let result: u32 = Divisible::<u8>::from_iter([0xAB, 0xCD].into_iter());
		assert_eq!(result, 0x0000CDAB);
	}

	#[test]
	fn test_from_iter_empty() {
		let result: u32 = Divisible::<u8>::from_iter(std::iter::empty());
		assert_eq!(result, 0);
	}

	#[test]
	fn test_from_iter_excess() {
		// More than N elements, only first 4 should be consumed
		let result: u32 =
			Divisible::<u8>::from_iter([0x01, 0x02, 0x03, 0x04, 0x05, 0x06].into_iter());
		assert_eq!(result, 0x04030201);
	}

	#[test]
	fn test_from_iter_u64_u16() {
		let result: u64 = Divisible::<u16>::from_iter([0x1234, 0x5678, 0x9ABC].into_iter());
		// Only 3 elements provided, 4th should be 0
		assert_eq!(result, 0x0000_9ABC_5678_1234);
	}
}
