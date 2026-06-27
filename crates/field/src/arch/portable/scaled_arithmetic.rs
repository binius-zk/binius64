// Copyright 2025-2026 The Binius Developers

use std::array;

use bytemuck::{Pod, TransparentWrapper};

use super::packed::PackedPrimitiveType;
use crate::{
	BinaryField,
	arithmetic_traits::{InvertOrZero, Square},
	underlier::{ScaledUnderlier, UnderlierType},
};

/// Wrapper for `ScaledUnderlier` multiplication that delegates to sub-underlier operations.
#[repr(transparent)]
#[derive(TransparentWrapper)]
pub struct Scaled<T>(T);

impl<U: UnderlierType + Pod, Scalar: BinaryField, const N: usize> std::ops::Mul
	for Scaled<PackedPrimitiveType<ScaledUnderlier<U, N>, Scalar>>
where
	PackedPrimitiveType<U, Scalar>: std::ops::Mul<Output = PackedPrimitiveType<U, Scalar>>,
{
	type Output = Self;

	fn mul(self, rhs: Self) -> Self {
		let (a, b) = (Self::peel(self), Self::peel(rhs));
		Self::wrap(PackedPrimitiveType::wrap(ScaledUnderlier(array::from_fn(|i| {
			let lhs_i = a.0.0[i];
			let rhs_i = b.0.0[i];
			PackedPrimitiveType::peel(
				PackedPrimitiveType::wrap(lhs_i) * PackedPrimitiveType::wrap(rhs_i),
			)
		}))))
	}
}

impl<U: UnderlierType + Pod, Scalar: BinaryField, const N: usize> Square
	for Scaled<PackedPrimitiveType<ScaledUnderlier<U, N>, Scalar>>
where
	PackedPrimitiveType<U, Scalar>: Square,
{
	fn square(self) -> Self {
		let val = Self::peel(self);
		Self::wrap(PackedPrimitiveType::wrap(ScaledUnderlier(val.0.0.map(|sub_underlier| {
			PackedPrimitiveType::peel(Square::square(PackedPrimitiveType::wrap(sub_underlier)))
		}))))
	}
}

impl<U: UnderlierType + Pod, Scalar: BinaryField, const N: usize> InvertOrZero
	for Scaled<PackedPrimitiveType<ScaledUnderlier<U, N>, Scalar>>
where
	PackedPrimitiveType<U, Scalar>: InvertOrZero,
{
	fn invert_or_zero(self) -> Self {
		let val = Self::peel(self);
		Self::wrap(PackedPrimitiveType::wrap(ScaledUnderlier(val.0.0.map(|sub_underlier| {
			PackedPrimitiveType::peel(InvertOrZero::invert_or_zero(PackedPrimitiveType::wrap(
				sub_underlier,
			)))
		}))))
	}
}
