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
pub trait DivisIterable<T> {
	type Iter<'a>: ExactSizeIterator<Item = &'a T>
	where
		Self: 'a,
		T: 'a;

	/// Returns an iterator over subdivisions of this underlier, ordered from LSB to MSB.
	fn divide(&self) -> Self::Iter<'_>;
}

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
			type Iter<'a> = std::slice::Iter<'a, $smaller>;

			fn divide(&self) -> Self::Iter<'_> {
				const N: usize = size_of::<$bigger>() / size_of::<$smaller>();
				::bytemuck::must_cast_ref::<Self, [$smaller; N]>(self).iter()
			}
		}

		#[cfg(target_endian = "big")]
		impl $crate::underlier::DivisIterable<$smaller> for $bigger {
			type Iter<'a> = std::iter::Rev<std::slice::Iter<'a, u8>>;

			fn divide(&self) -> Self::Iter<'_> {
				const N: usize = size_of::<$bigger>() / size_of::<$smaller>();
				::bytemuck::must_cast_ref::<Self, [$smaller; N]>(self).iter().rev()
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

use super::UnderlierType;

impl_divisible!(u128, u64, u32, u16, u8);
