// Copyright 2024-2025 Irreducible Inc.

use std::{
	array,
	ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not, Shl, Shr},
};

use binius_utils::checked_arithmetics::checked_log_2;
use bytemuck::{NoUninit, Pod, Zeroable, must_cast_mut, must_cast_ref};
use rand::{
	Rng,
	distr::{Distribution, StandardUniform},
};

use super::{Divisible, NumCast, UnderlierType, UnderlierWithBitOps};
use crate::Random;

/// A type that represents a pair of elements of the same underlier type.
/// We use it as an underlier for the `ScaledPackedField` type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(transparent)]
pub struct ScaledUnderlier<U, const N: usize>(pub [U; N]);

impl<U: Default, const N: usize> Default for ScaledUnderlier<U, N> {
	fn default() -> Self {
		Self(array::from_fn(|_| U::default()))
	}
}

impl<U: Random, const N: usize> Distribution<ScaledUnderlier<U, N>> for StandardUniform {
	fn sample<R: Rng + ?Sized>(&self, mut rng: &mut R) -> ScaledUnderlier<U, N> {
		ScaledUnderlier(array::from_fn(|_| U::random(&mut rng)))
	}
}

impl<U, const N: usize> From<ScaledUnderlier<U, N>> for [U; N] {
	fn from(val: ScaledUnderlier<U, N>) -> Self {
		val.0
	}
}

impl<T, U: From<T>, const N: usize> From<[T; N]> for ScaledUnderlier<U, N> {
	fn from(value: [T; N]) -> Self {
		Self(value.map(U::from))
	}
}

impl<T: Copy, U: From<[T; 2]>> From<[T; 4]> for ScaledUnderlier<U, 2> {
	fn from(value: [T; 4]) -> Self {
		Self([[value[0], value[1]], [value[2], value[3]]].map(Into::into))
	}
}

unsafe impl<U: Zeroable, const N: usize> Zeroable for ScaledUnderlier<U, N> {}

unsafe impl<U: Pod, const N: usize> Pod for ScaledUnderlier<U, N> {}

impl<U: UnderlierType + Pod, const N: usize> UnderlierType for ScaledUnderlier<U, N> {
	const LOG_BITS: usize = U::LOG_BITS + checked_log_2(N);
}

unsafe impl<U, const N: usize> Divisible<U> for ScaledUnderlier<U, N>
where
	Self: UnderlierType,
	U: UnderlierType,
{
	type Array = [U; N];

	#[inline]
	fn split_val(self) -> Self::Array {
		self.0
	}

	#[inline]
	fn split_ref(&self) -> &[U] {
		&self.0
	}

	#[inline]
	fn split_mut(&mut self) -> &mut [U] {
		&mut self.0
	}
}

unsafe impl<U> Divisible<U> for ScaledUnderlier<ScaledUnderlier<U, 2>, 2>
where
	Self: UnderlierType + NoUninit,
	U: UnderlierType + Pod,
{
	type Array = [U; 4];

	#[inline]
	fn split_val(self) -> Self::Array {
		bytemuck::must_cast(self)
	}

	#[inline]
	fn split_ref(&self) -> &[U] {
		must_cast_ref::<Self, [U; 4]>(self)
	}

	#[inline]
	fn split_mut(&mut self) -> &mut [U] {
		must_cast_mut::<Self, [U; 4]>(self)
	}
}

impl<U: BitAnd<Output = U> + Copy, const N: usize> BitAnd for ScaledUnderlier<U, N> {
	type Output = Self;

	fn bitand(self, rhs: Self) -> Self::Output {
		Self(array::from_fn(|i| self.0[i] & rhs.0[i]))
	}
}

impl<U: BitAndAssign + Copy, const N: usize> BitAndAssign for ScaledUnderlier<U, N> {
	fn bitand_assign(&mut self, rhs: Self) {
		for i in 0..N {
			self.0[i] &= rhs.0[i];
		}
	}
}

impl<U: BitOr<Output = U> + Copy, const N: usize> BitOr for ScaledUnderlier<U, N> {
	type Output = Self;

	fn bitor(self, rhs: Self) -> Self::Output {
		Self(array::from_fn(|i| self.0[i] | rhs.0[i]))
	}
}

impl<U: BitOrAssign + Copy, const N: usize> BitOrAssign for ScaledUnderlier<U, N> {
	fn bitor_assign(&mut self, rhs: Self) {
		for i in 0..N {
			self.0[i] |= rhs.0[i];
		}
	}
}

impl<U: BitXor<Output = U> + Copy, const N: usize> BitXor for ScaledUnderlier<U, N> {
	type Output = Self;

	fn bitxor(self, rhs: Self) -> Self::Output {
		Self(array::from_fn(|i| self.0[i] ^ rhs.0[i]))
	}
}

impl<U: BitXorAssign + Copy, const N: usize> BitXorAssign for ScaledUnderlier<U, N> {
	fn bitxor_assign(&mut self, rhs: Self) {
		for i in 0..N {
			self.0[i] ^= rhs.0[i];
		}
	}
}

impl<U: UnderlierWithBitOps, const N: usize> Shr<usize> for ScaledUnderlier<U, N> {
	type Output = Self;

	fn shr(self, rhs: usize) -> Self::Output {
		let mut result = Self::default();

		let shift_in_items = rhs / U::BITS;
		for i in 0..N.saturating_sub(shift_in_items.saturating_sub(1)) {
			if i + shift_in_items < N {
				result.0[i] |= self.0[i + shift_in_items] >> (rhs % U::BITS);
			}
			if i + shift_in_items + 1 < N && !rhs.is_multiple_of(U::BITS) {
				result.0[i] |= self.0[i + shift_in_items + 1] << (U::BITS - (rhs % U::BITS));
			}
		}

		result
	}
}

impl<U: UnderlierWithBitOps, const N: usize> Shl<usize> for ScaledUnderlier<U, N> {
	type Output = Self;

	fn shl(self, rhs: usize) -> Self::Output {
		let mut result = Self::default();

		let shift_in_items = rhs / U::BITS;
		for i in shift_in_items.saturating_sub(1)..N {
			if i >= shift_in_items {
				result.0[i] |= self.0[i - shift_in_items] << (rhs % U::BITS);
			}
			if i > shift_in_items && !rhs.is_multiple_of(U::BITS) {
				result.0[i] |= self.0[i - shift_in_items - 1] >> (U::BITS - (rhs % U::BITS));
			}
		}

		result
	}
}

impl<U: Not<Output = U>, const N: usize> Not for ScaledUnderlier<U, N> {
	type Output = Self;

	fn not(self) -> Self::Output {
		Self(self.0.map(U::not))
	}
}

impl<U, const N: usize> UnderlierWithBitOps for ScaledUnderlier<U, N>
where
	U: UnderlierWithBitOps + Pod + From<u8>,
	u8: NumCast<U>,
{
	const ZERO: Self = Self([U::ZERO; N]);
	const ONE: Self = {
		let mut arr = [U::ZERO; N];
		arr[0] = U::ONE;
		Self(arr)
	};
	const ONES: Self = Self([U::ONES; N]);

	#[inline]
	fn fill_with_bit(val: u8) -> Self {
		Self(array::from_fn(|_| U::fill_with_bit(val)))
	}

	#[inline]
	fn shl_128b_lanes(self, rhs: usize) -> Self {
		// We assume that the underlier type has at least 128 bits as the current implementation
		// is valid for this case only.
		// On practice, we don't use scaled underliers with underlier types that have less than 128
		// bits.
		assert!(U::BITS >= 128);

		Self(self.0.map(|x| x.shl_128b_lanes(rhs)))
	}

	#[inline]
	fn shr_128b_lanes(self, rhs: usize) -> Self {
		// We assume that the underlier type has at least 128 bits as the current implementation
		// is valid for this case only.
		// On practice, we don't use scaled underliers with underlier types that have less than 128
		// bits.
		assert!(U::BITS >= 128);

		Self(self.0.map(|x| x.shr_128b_lanes(rhs)))
	}

	#[inline]
	fn unpack_lo_128b_lanes(self, other: Self, log_block_len: usize) -> Self {
		// We assume that the underlier type has at least 128 bits as the current implementation
		// is valid for this case only.
		// On practice, we don't use scaled underliers with underlier types that have less than 128
		// bits.
		assert!(U::BITS >= 128);

		Self(array::from_fn(|i| self.0[i].unpack_lo_128b_lanes(other.0[i], log_block_len)))
	}

	#[inline]
	fn unpack_hi_128b_lanes(self, other: Self, log_block_len: usize) -> Self {
		// We assume that the underlier type has at least 128 bits as the current implementation
		// is valid for this case only.
		// On practice, we don't use scaled underliers with underlier types that have less than 128
		// bits.
		assert!(U::BITS >= 128);

		Self(array::from_fn(|i| self.0[i].unpack_hi_128b_lanes(other.0[i], log_block_len)))
	}
}

impl<U: UnderlierType, const N: usize> NumCast<ScaledUnderlier<U, N>> for u8
where
	Self: NumCast<U>,
{
	fn num_cast_from(val: ScaledUnderlier<U, N>) -> Self {
		Self::num_cast_from(val.0[0])
	}
}

impl<U, const N: usize> From<u8> for ScaledUnderlier<U, N>
where
	U: From<u8>,
{
	fn from(val: u8) -> Self {
		Self(array::from_fn(|_| U::from(val)))
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn test_shr() {
		let val = ScaledUnderlier::<u8, 4>([0, 1, 2, 3]);
		assert_eq!(
			val >> 1,
			ScaledUnderlier::<u8, 4>([0b10000000, 0b00000000, 0b10000001, 0b00000001])
		);
		assert_eq!(
			val >> 2,
			ScaledUnderlier::<u8, 4>([0b01000000, 0b10000000, 0b11000000, 0b00000000])
		);
		assert_eq!(
			val >> 8,
			ScaledUnderlier::<u8, 4>([0b00000001, 0b00000010, 0b00000011, 0b00000000])
		);
		assert_eq!(
			val >> 9,
			ScaledUnderlier::<u8, 4>([0b00000000, 0b10000001, 0b00000001, 0b00000000])
		);
	}

	#[test]
	fn test_shl() {
		let val = ScaledUnderlier::<u8, 4>([0, 1, 2, 3]);
		assert_eq!(val << 1, ScaledUnderlier::<u8, 4>([0, 2, 4, 6]));
		assert_eq!(val << 2, ScaledUnderlier::<u8, 4>([0, 4, 8, 12]));
		assert_eq!(val << 8, ScaledUnderlier::<u8, 4>([0, 0, 1, 2]));
		assert_eq!(val << 9, ScaledUnderlier::<u8, 4>([0, 0, 2, 4]));
	}
}
