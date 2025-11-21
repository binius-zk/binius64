// Copyright 2024-2025 Irreducible Inc.

use core::slice;
use std::{
	mem::{align_of, size_of},
	slice::{from_raw_parts, from_raw_parts_mut},
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

use super::UnderlierType;

impl_divisible!(u128, u64, u32, u16, u8);
