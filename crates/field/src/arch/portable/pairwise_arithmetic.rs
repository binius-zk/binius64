// Copyright 2024-2025 Irreducible Inc.

use bytemuck::TransparentWrapper;

use crate::{
	arch::PairwiseStrategy,
	arithmetic_traits::{InvertOrZero, Square, TaggedInvertOrZero, TaggedSquare},
	packed::PackedField,
};

/// Pairwise multiplication wrapper. Apply the multiplication to each packed element independently.
#[repr(transparent)]
#[derive(TransparentWrapper)]
pub struct Pairwise<T>(T);

impl<PT: PackedField> std::ops::Mul for Pairwise<PT> {
	type Output = Self;

	#[inline]
	fn mul(self, rhs: Self) -> Self {
		let (a, b) = (Self::peel(self), Self::peel(rhs));
		Self::wrap(if PT::WIDTH == 1 {
			// fallback to be able to benchmark this strategy
			a * b
		} else {
			PT::from_fn(|i| a.get(i) * b.get(i))
		})
	}
}

impl<PT: PackedField> TaggedSquare<PairwiseStrategy> for PT
where
	PT::Scalar: Square,
{
	#[inline]
	fn square(self) -> Self {
		if PT::WIDTH == 1 {
			// fallback to be able to benchmark this strategy
			Square::square(self)
		} else {
			Self::from_fn(|i| Square::square(self.get(i)))
		}
	}
}

impl<PT: PackedField> TaggedInvertOrZero<PairwiseStrategy> for PT
where
	PT::Scalar: InvertOrZero,
{
	#[inline]
	fn invert_or_zero(self) -> Self {
		if PT::WIDTH == 1 {
			// fallback to be able to benchmark this strategy
			InvertOrZero::invert_or_zero(self)
		} else {
			Self::from_fn(|i| InvertOrZero::invert_or_zero(self.get(i)))
		}
	}
}
