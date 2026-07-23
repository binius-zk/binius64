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

	/// Fill value with the given bit
	/// `val` must be 0 or 1.
	fn fill_with_bit(val: u8) -> Self {
		Self::broadcast_subvalue(U1::new(val))
	}

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

	/// Broadcast subvalue to fill `Self`.
	/// `Self::BITS/T::BITS` is supposed to be a power of 2.
	#[inline]
	fn broadcast_subvalue<T>(value: T) -> Self
	where
		T: UnderlierType,
		Self: Divisible<T>,
	{
		Divisible::<T>::broadcast(value)
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

/// A trait that represents potentially lossy numeric cast.
/// Is a drop-in replacement of `as _` in a generic code.
pub trait NumCast<From> {
	fn num_cast_from(val: From) -> Self;
}

impl<U: UnderlierType> NumCast<U> for U {
	#[inline(always)]
	fn num_cast_from(val: U) -> Self {
		val
	}
}
