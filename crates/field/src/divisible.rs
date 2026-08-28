// Copyright 2024-2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

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

	/// Set element at index (LSB-first ordering) in place, without bounds checking.
	///
	/// A single-element write stays a single-element store.
	///
	/// # Safety
	///
	/// The caller must ensure that `index < N`.
	#[cfg(target_endian = "little")]
	#[inline]
	pub unsafe fn set<Big, Small, const N: usize>(value: &mut Big, index: usize, val: Small)
	where
		Big: Pod,
		Small: Pod,
	{
		// Safety: the caller guarantees `index < N`.
		unsafe {
			*bytemuck::must_cast_mut::<Big, [Small; N]>(value).get_unchecked_mut(index) = val;
		}
	}

	/// Set element at index (LSB-first ordering) in place, without bounds checking.
	///
	/// A single-element write stays a single-element store.
	///
	/// # Safety
	///
	/// The caller must ensure that `index < N`.
	#[cfg(target_endian = "big")]
	#[inline]
	pub unsafe fn set<Big, Small, const N: usize>(value: &mut Big, index: usize, val: Small)
	where
		Big: Pod,
		Small: Pod,
	{
		// Safety: the caller guarantees `index < N`, so `N - 1 - index < N`.
		unsafe {
			*bytemuck::must_cast_mut::<Big, [Small; N]>(value).get_unchecked_mut(N - 1 - index) =
				val;
		}
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

/// Helper functions for iterating a subdivision by mapping over its indices.
///
/// Suits a subdivision whose element access is index arithmetic. Wrong for one whose access is a
/// lane extract, since at a run-time index that becomes an unpredictable branch per element.
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

/// Implements [`Divisible`] over each named subdivision by reinterpreting memory.
///
/// The plain form broadcasts through memory:
///
/// ```text
/// impl_divisible_memcast!(u128, u64, u32, u16, u8);
/// ```
///
/// The arrow form takes a broadcast instruction per subdivision:
///
/// ```text
/// impl_divisible_memcast!(M512, u64 => |val| unsafe { M512(_mm512_set1_epi64(val as i64)) });
/// ```
macro_rules! impl_divisible_memcast {
	// Each subdivision names the instruction that broadcasts it.
	($big:ty, $($small:ty => |$v:ident| $broadcast:expr),+ $(,)?) => {
		$(
			$crate::divisible::impl_divisible_memcast!(@impl $big, $small, |$v| $broadcast);
		)+
	};
	// Every subdivision broadcasts by a memory splat.
	($big:ty, $($small:ty),+ $(,)?) => {
		$(
			$crate::divisible::impl_divisible_memcast!(
				@impl $big, $small,
				|val| $crate::divisible::memcast::broadcast::<
					$big,
					$small,
					{ ::std::mem::size_of::<$big>() / ::std::mem::size_of::<$small>() },
				>(val)
			);
		)+
	};
	(@impl $big:ty, $small:ty, |$v:ident| $broadcast:expr) => {
		impl $crate::divisible::Divisible<$small> for $big {
			const LOG_N: usize =
				(::std::mem::size_of::<$big>() / ::std::mem::size_of::<$small>()).ilog2() as usize;

			#[inline]
			fn value_iter(value: Self) -> impl ExactSizeIterator<Item = $small> + Send + Clone {
				const N: usize = ::std::mem::size_of::<$big>() / ::std::mem::size_of::<$small>();
				$crate::divisible::memcast::value_iter::<$big, $small, N>(value)
			}

			#[inline]
			fn ref_iter(value: &Self) -> impl ExactSizeIterator<Item = $small> + Send + Clone + '_ {
				const N: usize = ::std::mem::size_of::<$big>() / ::std::mem::size_of::<$small>();
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
				const LOG_N: usize =
					(::std::mem::size_of::<$big>() / ::std::mem::size_of::<$small>()).ilog2() as usize;
				$crate::divisible::memcast::slice_iter::<$big, $small, LOG_N>(slice)
			}

			#[inline]
			unsafe fn get_unchecked(&self, index: usize) -> $small {
				const N: usize = ::std::mem::size_of::<$big>() / ::std::mem::size_of::<$small>();
				// Safety: the caller guarantees `index < Self::N == N`.
				unsafe { $crate::divisible::memcast::get::<$big, $small, N>(self, index) }
			}

			#[inline]
			unsafe fn set_unchecked(&mut self, index: usize, val: $small) {
				const N: usize = ::std::mem::size_of::<$big>() / ::std::mem::size_of::<$small>();
				// Safety: the caller guarantees `index < Self::N == N`.
				unsafe { $crate::divisible::memcast::set::<$big, $small, N>(self, index, val) };
			}

			#[inline]
			fn broadcast($v: $small) -> Self {
				$broadcast
			}

			#[inline]
			fn from_iter(iter: impl Iterator<Item = $small>) -> Self {
				const N: usize = ::std::mem::size_of::<$big>() / ::std::mem::size_of::<$small>();
				$crate::divisible::memcast::from_iter::<$big, $small, N>(iter)
			}
		}
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
	use proptest::{arbitrary::any, proptest};

	use super::*;

	#[test]
	fn test_divisible_u32_u8() {
		let val = 0xab12cd34u32;

		// Test get - LSB first: bytes
		assert_eq!(Divisible::<u8>::get(&val, 0), 0x34u8);
		assert_eq!(Divisible::<u8>::get(&val, 1), 0xcdu8);
		assert_eq!(Divisible::<u8>::get(&val, 2), 0x12u8);
		assert_eq!(Divisible::<u8>::get(&val, 3), 0xabu8);

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

	proptest! {
		#[test]
		fn test_set_get_u32_u8(mut val in any::<u32>(), i in 0usize..4, elem in any::<u8>()) {
			Divisible::<u8>::set(&mut val, i, elem);
			assert_eq!(Divisible::<u8>::get(&val, i), elem);
		}
	}
}

#[cfg(test)]
mod arch_tests {
	use std::fmt::Debug;

	use binius_utils::{SerializeBytes, bytes::BytesMut};
	use proptest::{arbitrary::any, proptest};

	use super::Divisible;
	use crate::arch::{M128, M256, M512};

	/// The byte subdivision is the value's little-endian byte string, which serialization states
	/// independently.
	fn check_byte_anchor<Big>(value: Big)
	where
		Big: Divisible<u8> + SerializeBytes + Copy,
	{
		let mut buf = BytesMut::new();
		value
			.serialize(&mut buf)
			.expect("BytesMut grows to fit the value");

		assert!(Big::value_iter(value).eq(buf));
	}

	/// Cutting one lane into bytes gives the same bytes as the whole value's matching window.
	fn check_refines<Big, Small>(value: Big)
	where
		Big: Divisible<u8> + Divisible<Small> + Copy,
		Small: Divisible<u8> + Copy,
	{
		let bytes_per_lane = <Small as Divisible<u8>>::N;

		for i in 0..<Big as Divisible<Small>>::N {
			let lane = Divisible::<Small>::get(&value, i);
			let window =
				(0..bytes_per_lane).map(|j| Divisible::<u8>::get(&value, i * bytes_per_lane + j));

			assert!(<Small as Divisible<u8>>::value_iter(lane).eq(window), "lane {i}");
		}
	}

	/// The iterators agree with element access, and rebuilding from them restores the value.
	fn check_iters<Big, Small>(value: Big, other: Big)
	where
		Big: Divisible<Small> + Copy + Eq + Debug,
		Small: Copy + Eq + Debug,
	{
		let by_index = (0..<Big as Divisible<Small>>::N)
			.map(|i| Divisible::<Small>::get(&value, i))
			.collect::<Vec<_>>();

		assert!(Big::value_iter(value).eq(by_index.iter().copied()));
		assert!(Big::ref_iter(&value).eq(by_index.iter().copied()));

		// Over a slice the subdivisions run element by element, in order.
		let slice = [value, other];
		assert!(Big::slice_iter(&slice).eq(Big::value_iter(value).chain(Big::value_iter(other))));

		assert_eq!(Big::from_iter(by_index.iter().copied()), value);
	}

	/// A broadcast lane reads back at every index.
	fn check_broadcast<Big, Small>(source: Big)
	where
		Big: Divisible<Small> + Copy,
		Small: Copy + Eq + Debug,
	{
		// Take the lane from a generated value, so no subdivision needs its own strategy.
		let lane = Divisible::<Small>::get(&source, 0);
		let value = <Big as Divisible<Small>>::broadcast(lane);

		for i in 0..<Big as Divisible<Small>>::N {
			assert_eq!(Divisible::<Small>::get(&value, i), lane, "index {i}");
		}
	}

	/// Writing one index leaves every other index alone.
	fn check_set<Big, Small>(value: Big, source: Big, index: usize)
	where
		Big: Divisible<Small> + Copy,
		Small: Copy + Eq + Debug,
	{
		let index = index % <Big as Divisible<Small>>::N;
		let lane = Divisible::<Small>::get(&source, 0);

		let mut updated = value;
		Divisible::<Small>::set(&mut updated, index, lane);

		assert_eq!(Divisible::<Small>::get(&updated, index), lane);
		for i in (0..<Big as Divisible<Small>>::N).filter(|&i| i != index) {
			assert_eq!(
				Divisible::<Small>::get(&updated, i),
				Divisible::<Small>::get(&value, i),
				"index {i}"
			);
		}
	}

	/// Runs every property at one subdivision width.
	fn check_width<Big, Small>(a: Big, b: Big, index: usize)
	where
		Big: Divisible<u8> + Divisible<Small> + Copy + Eq + Debug,
		Small: Divisible<u8> + Copy + Eq + Debug,
	{
		check_refines::<Big, Small>(a);
		check_iters::<Big, Small>(a, b);
		check_broadcast::<Big, Small>(b);
		check_set::<Big, Small>(a, b, index);
	}

	proptest! {
		// These resolve to the target's SIMD registers where it has them, the scaled fallbacks
		// otherwise.

		#[test]
		fn m128_subdivisions(a in any::<u128>(), b in any::<u128>(), index in any::<usize>()) {
			let (a, b) = (M128::from(a), M128::from(b));

			check_byte_anchor(a);
			check_width::<M128, u128>(a, b, index);
			check_width::<M128, u64>(a, b, index);
			check_width::<M128, u32>(a, b, index);
			check_width::<M128, u16>(a, b, index);
			check_width::<M128, u8>(a, b, index);
		}

		#[test]
		fn m256_subdivisions(
			a in any::<[u128; 2]>(),
			b in any::<[u128; 2]>(),
			index in any::<usize>(),
		) {
			let (a, b) = (M256::from(a), M256::from(b));

			check_byte_anchor(a);
			check_width::<M256, M128>(a, b, index);
			check_width::<M256, u128>(a, b, index);
			check_width::<M256, u64>(a, b, index);
			check_width::<M256, u32>(a, b, index);
			check_width::<M256, u16>(a, b, index);
			check_width::<M256, u8>(a, b, index);
		}

		#[test]
		fn m512_subdivisions(
			a in any::<[u128; 4]>(),
			b in any::<[u128; 4]>(),
			index in any::<usize>(),
		) {
			let (a, b) = (M512::from(a), M512::from(b));

			check_byte_anchor(a);
			check_width::<M512, M256>(a, b, index);
			check_width::<M512, M128>(a, b, index);
			check_width::<M512, u128>(a, b, index);
			check_width::<M512, u64>(a, b, index);
			check_width::<M512, u32>(a, b, index);
			check_width::<M512, u16>(a, b, index);
			check_width::<M512, u8>(a, b, index);
		}
	}
}
