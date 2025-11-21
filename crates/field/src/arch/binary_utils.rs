// Copyright 2024-2025 Irreducible Inc.

use bytemuck::{AnyBitPattern, NoUninit, must_cast_mut, must_cast_ref};

use crate::underlier::{NumCast, UnderlierType};

/// Execute function `f` with a reference to an array of length `N` casted from `val`.
#[allow(unused)]
pub(super) fn as_array_ref<T, U, const N: usize, R>(val: &T, f: impl FnOnce(&[U; N]) -> R) -> R
where
	T: NoUninit + AnyBitPattern,
	[U; N]: NoUninit + AnyBitPattern,
{
	let array = must_cast_ref(val);
	f(array)
}

/// Execute function `f` with a mutable reference to an array of length `N` casted from `val`.
#[allow(unused)]
pub(super) fn as_array_mut<T, U, const N: usize>(val: &mut T, f: impl FnOnce(&mut [U; N]))
where
	T: NoUninit + AnyBitPattern,
	[U; N]: NoUninit + AnyBitPattern,
{
	let array = must_cast_mut(val);
	f(array);
}

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
