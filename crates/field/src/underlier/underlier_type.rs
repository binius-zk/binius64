// Copyright 2024-2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use std::{
	fmt::Debug,
	ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not},
};

use bytemuck::{NoUninit, TransparentWrapper, Zeroable};

use super::U1;
use crate::{Divisible, Random};

/// Primitive integer underlying a binary field or packed binary field implementation.
/// Note that this type is not guaranteed to be POD, U1, U2 and U4 have some unused bits.
pub trait UnderlierType:
	Debug
	+ Default
	+ Eq
	+ Ord
	+ Copy
	+ Random
	+ NoUninit
	+ Zeroable
	+ Sized
	+ Send
	+ Sync
	+ 'static
	+ BitAnd<Self, Output = Self>
	+ BitAndAssign<Self>
	+ BitOr<Self, Output = Self>
	+ BitOrAssign<Self>
	+ BitXor<Self, Output = Self>
	+ BitXorAssign<Self>
	+ Not<Output = Self>
	+ Divisible<U1>
{
	/// Number of bits in value
	const LOG_BITS: usize;
	/// Number of bits used to represent a value.
	/// This may not be equal to the number of bits in a type instance.
	const BITS: usize = 1 << Self::LOG_BITS;

	const ZERO: Self;
	const ONE: Self;
	const ONES: Self;

	/// Interleave with the given bit size
	fn interleave(self, other: Self, log_block_len: usize) -> (Self, Self);

	/// Transpose with the given bit size
	fn transpose(mut self, mut other: Self, log_block_len: usize) -> (Self, Self) {
		assert!(log_block_len < Self::LOG_BITS);

		for log_block_len in (log_block_len..Self::LOG_BITS).rev() {
			(self, other) = self.interleave(other, log_block_len);
		}

		(self, other)
	}

	#[inline]
	fn from_fn<T>(f: impl FnMut(usize) -> T) -> Self
	where
		T: UnderlierType,
		Self: Divisible<T>,
	{
		Self::from_iter((0..<Self as Divisible<T>>::N).map(f))
	}
}

/// A type that is transparently backed by an underlier.
///
/// This trait is needed to make it possible getting the underlier type from already defined type.
/// Bidirectional `From` trait implementations are not enough, because they do not allow getting
/// underlier type in a generic code.
///
/// # Safety
/// `WithUnderlier` can be implemented for a type only if it's representation is a transparent
/// `Underlier`'s representation. That's allows us casting references of type and it's underlier in
/// both directions.
pub unsafe trait WithUnderlier:
	TransparentWrapper<Self::Underlier> + Sized + Zeroable + Copy + Send + Sync + 'static
{
	/// Underlier primitive type
	type Underlier: UnderlierType;

	/// Convert value to underlier.
	#[inline]
	fn to_underlier(self) -> Self::Underlier {
		Self::peel(self)
	}

	#[inline]
	fn to_underlier_ref(&self) -> &Self::Underlier {
		Self::peel_ref(self)
	}

	#[inline]
	fn to_underlier_ref_mut(&mut self) -> &mut Self::Underlier {
		Self::peel_mut(self)
	}

	#[inline]
	fn to_underliers_ref(val: &[Self]) -> &[Self::Underlier] {
		Self::peel_slice(val)
	}

	#[inline]
	fn to_underliers_ref_mut(val: &mut [Self]) -> &mut [Self::Underlier] {
		Self::peel_slice_mut(val)
	}

	#[inline]
	fn from_underlier(val: Self::Underlier) -> Self {
		Self::wrap(val)
	}

	#[inline]
	fn from_underlier_ref(val: &Self::Underlier) -> &Self {
		Self::wrap_ref(val)
	}

	#[inline]
	fn from_underlier_ref_mut(val: &mut Self::Underlier) -> &mut Self {
		Self::wrap_mut(val)
	}

	#[inline]
	fn from_underliers_ref(val: &[Self::Underlier]) -> &[Self] {
		Self::wrap_slice(val)
	}

	#[inline]
	fn from_underliers_ref_mut(val: &mut [Self::Underlier]) -> &mut [Self] {
		Self::wrap_slice_mut(val)
	}

	#[inline]
	fn mutate_underlier(self, f: impl FnOnce(Self::Underlier) -> Self::Underlier) -> Self {
		Self::from_underlier(f(self.to_underlier()))
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::underlier::{U2, U4};

	#[test]
	fn test_from_fn() {
		assert_eq!(u32::from_fn(|_| U1::new(0)), 0);
		assert_eq!(u32::from_fn(|i| U1::new((i % 2) as u8)), 0xaaaaaaaa);
		assert_eq!(u32::from_fn(|_| U1::new(1)), u32::MAX);

		assert_eq!(u32::from_fn(|_| U2::new(0)), 0);
		assert_eq!(u32::from_fn(|_| U2::new(1)), 0x55555555);
		assert_eq!(u32::from_fn(|_| U2::new(2)), 0xaaaaaaaa);
		assert_eq!(u32::from_fn(|_| U2::new(3)), u32::MAX);
		assert_eq!(u32::from_fn(|i| U2::new((i % 4) as u8)), 0xe4e4e4e4);

		assert_eq!(u32::from_fn(|_| U4::new(0)), 0);
		assert_eq!(u32::from_fn(|_| U4::new(1)), 0x11111111);
		assert_eq!(u32::from_fn(|_| U4::new(8)), 0x88888888);
		assert_eq!(u32::from_fn(|_| U4::new(31)), 0xffffffff);
		assert_eq!(u32::from_fn(|i| U4::new(i as u8)), 0x76543210);

		assert_eq!(u32::from_fn(|_| 0u8), 0);
		assert_eq!(u32::from_fn(|_| 0xabu8), 0xabababab);
		assert_eq!(u32::from_fn(|_| 255u8), 0xffffffff);
		assert_eq!(u32::from_fn(|i| i as u8), 0x03020100);
	}
}
