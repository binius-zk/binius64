// Copyright 2024-2025 Irreducible Inc.

use crate::underlier::{NumCast, UnderlierType};

/// Helper function to convert `f` closure that returns a value 1-4 bits wide to a function that
/// returns i8.
#[allow(dead_code)]
#[inline]
pub(super) fn make_func_to_i8<T, U>(mut f: impl FnMut(usize) -> T) -> impl FnMut(usize) -> i8
where
	T: UnderlierType,
	U: From<T>,
	u8: NumCast<U>,
{
	move |i| {
		let elements_in_8 = 8 / T::BITS;
		let mut result = 0u8;
		for j in 0..elements_in_8 {
			result |= u8::num_cast_from(U::from(f(i * elements_in_8 + j))) << (j * T::BITS);
		}

		result as i8
	}
}
