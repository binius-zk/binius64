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
	/// The number of `T` elements that fit in `Self`.
	const N: usize;

	type Iter<'a>: ExactSizeIterator<Item = T>
	where
		Self: 'a,
		T: 'a;

	/// Returns an iterator over subdivisions of this underlier, ordered from LSB to MSB.
	///
	/// The iterator yields exactly [`Self::N`] elements.
	fn divide(&self) -> Self::Iter<'_>;

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

/// Iterator for dividing an underlier into sub-byte elements (SmallU<N>).
///
/// This iterator wraps a byte iterator and extracts sub-byte elements from each byte.
/// Generic over the byte iterator type `I`.
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

		#[cfg(target_endian = "little")]
		impl $crate::underlier::DivisIterable<$smaller> for $bigger {
			const N: usize = size_of::<$bigger>() / size_of::<$smaller>();

			type Iter<'a> = std::iter::Copied<std::slice::Iter<'a, $smaller>>;

			#[inline]
			fn divide(&self) -> Self::Iter<'_> {
				const N: usize = size_of::<$bigger>() / size_of::<$smaller>();
				::bytemuck::must_cast_ref::<Self, [$smaller; N]>(self).iter().copied()
			}

			#[inline]
			fn get(self, index: usize) -> $smaller {
				const N: usize = size_of::<$bigger>() / size_of::<$smaller>();
				::bytemuck::must_cast_ref::<Self, [$smaller; N]>(&self)[index]
			}

			#[inline]
			fn set(self, index: usize, val: $smaller) -> Self {
				const N: usize = size_of::<$bigger>() / size_of::<$smaller>();
				let mut arr = *::bytemuck::must_cast_ref::<Self, [$smaller; N]>(&self);
				arr[index] = val;
				::bytemuck::must_cast(arr)
			}
		}

		#[cfg(target_endian = "big")]
		impl $crate::underlier::DivisIterable<$smaller> for $bigger {
			const N: usize = size_of::<$bigger>() / size_of::<$smaller>();

			type Iter<'a> = std::iter::Copied<std::iter::Rev<std::slice::Iter<'a, $smaller>>>;

			#[inline]
			fn divide(&self) -> Self::Iter<'_> {
				const N: usize = size_of::<$bigger>() / size_of::<$smaller>();
				::bytemuck::must_cast_ref::<Self, [$smaller; N]>(self).iter().rev().copied()
			}

			#[inline]
			fn get(self, index: usize) -> $smaller {
				const N: usize = size_of::<$bigger>() / size_of::<$smaller>();
				::bytemuck::must_cast_ref::<Self, [$smaller; N]>(&self)[N - 1 - index]
			}

			#[inline]
			fn set(self, index: usize, val: $smaller) -> Self {
				const N: usize = size_of::<$bigger>() / size_of::<$smaller>();
				let mut arr = *::bytemuck::must_cast_ref::<Self, [$smaller; N]>(&self);
				arr[N - 1 - index] = val;
				::bytemuck::must_cast(arr)
			}
		}
    };

	// Small underlier implementation (SmallU<N>)
	// Uses direct shifting/masking on the bigger type
	(@small_pair $bigger:ty, $bits:expr) => {
		impl $crate::underlier::DivisIterable<$crate::underlier::SmallU<$bits>> for $bigger {
			const N: usize = <$bigger>::BITS as usize / $bits;

			type Iter<'a> = $crate::underlier::SmallUDivisIter<
				<$bigger as $crate::underlier::DivisIterable<u8>>::Iter<'a>,
				$bits
			>;

			#[inline]
			fn divide(&self) -> Self::Iter<'_> {
				$crate::underlier::SmallUDivisIter::new(
					$crate::underlier::DivisIterable::<u8>::divide(self)
				)
			}

			#[inline]
			fn get(self, index: usize) -> $crate::underlier::SmallU<$bits> {
				let shift = index * $bits;
				$crate::underlier::SmallU::<$bits>::new((self >> shift) as u8)
			}

			#[inline]
			fn set(self, index: usize, val: $crate::underlier::SmallU<$bits>) -> Self {
				let shift = index * $bits;
				let mask = ((1 as $bigger) << $bits) - 1;
				(self & !(mask << shift)) | ((val.val() as $bigger) << shift)
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

use super::{UnderlierType, small_uint::SmallU};

impl_divisible!(u128, u64, u32, u16, u8);

// Implement DivisIterable for small underliers (SmallU<N>)
// Special case for u8: operates directly on the byte
macro_rules! impl_divisible_u8_small {
	($bits:expr) => {
		impl DivisIterable<SmallU<$bits>> for u8 {
			const N: usize = 8 / $bits;

			type Iter<'a> = SmallUDivisIter<std::iter::Once<u8>, $bits>;

			#[inline]
			fn divide(&self) -> Self::Iter<'_> {
				SmallUDivisIter::new(std::iter::once(*self))
			}

			#[inline]
			fn get(self, index: usize) -> SmallU<$bits> {
				let shift = index * $bits;
				SmallU::<$bits>::new(self >> shift)
			}

			#[inline]
			fn set(self, index: usize, val: SmallU<$bits>) -> Self {
				let shift = index * $bits;
				let mask = (1u8 << $bits) - 1;
				(self & !(mask << shift)) | (val.val() << shift)
			}
		}
	};
}

impl_divisible_u8_small!(1);
impl_divisible_u8_small!(2);
impl_divisible_u8_small!(4);

impl_divisible!(@small_pair u16, 1);
impl_divisible!(@small_pair u16, 2);
impl_divisible!(@small_pair u16, 4);
impl_divisible!(@small_pair u32, 1);
impl_divisible!(@small_pair u32, 2);
impl_divisible!(@small_pair u32, 4);
impl_divisible!(@small_pair u64, 1);
impl_divisible!(@small_pair u64, 2);
impl_divisible!(@small_pair u64, 4);
impl_divisible!(@small_pair u128, 1);
impl_divisible!(@small_pair u128, 2);
impl_divisible!(@small_pair u128, 4);

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

		// Test divide iterator
		let parts: Vec<U4> = DivisIterable::<U4>::divide(&val).collect();
		assert_eq!(parts.len(), 2);
		assert_eq!(parts[0], U4::new(0x4));
		assert_eq!(parts[1], U4::new(0x3));
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

		// Test divide iterator
		let parts: Vec<U4> = DivisIterable::<U4>::divide(&val).collect();
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

		// Test divide iterator
		let parts: Vec<U2> = DivisIterable::<U2>::divide(&val).collect();
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

		// Test divide iterator
		let parts: Vec<U1> = DivisIterable::<U1>::divide(&val).collect();
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

		// Test divide iterator
		let parts: Vec<U4> = DivisIterable::<U4>::divide(&val).collect();
		assert_eq!(parts.len(), 16);
	}
}
