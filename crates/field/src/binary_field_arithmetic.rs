// Copyright 2023-2025 Irreducible Inc.

use std::ops::Mul;

use super::{
	arithmetic_traits::{InvertOrZero, Square},
	binary_field::BinaryField1b,
};
use crate::PackedField;

macro_rules! impl_arithmetic_using_packed {
	($name:ident) => {
		impl InvertOrZero for $name {
			#[inline]
			fn invert_or_zero(self) -> Self {
				use $crate::packed_extension::PackedSubfield;

				$crate::binary_field_arithmetic::invert_or_zero_using_packed::<
					PackedSubfield<Self, Self>,
				>(self)
			}
		}

		impl ::core::ops::Mul<$name> for $name {
			type Output = $name;

			#[inline]
			fn mul(self, rhs: $name) -> $name {
				use $crate::packed_extension::PackedSubfield;

				$crate::tracing::trace_multiplication!($name);
				$crate::binary_field_arithmetic::multiple_using_packed::<PackedSubfield<Self, Self>>(
					self, rhs,
				)
			}
		}

		impl $crate::arithmetic_traits::Square for $name {
			#[inline]
			fn square(self) -> Self {
				use $crate::packed_extension::PackedSubfield;

				$crate::binary_field_arithmetic::square_using_packed::<PackedSubfield<Self, Self>>(
					self,
				)
			}
		}
	};
}

pub(crate) use impl_arithmetic_using_packed;

impl InvertOrZero for BinaryField1b {
	#[inline]
	fn invert_or_zero(self) -> Self {
		self
	}
}

#[allow(clippy::suspicious_arithmetic_impl)]
impl Mul<BinaryField1b> for BinaryField1b {
	type Output = Self;

	#[inline]
	fn mul(self, rhs: Self) -> Self::Output {
		crate::tracing::trace_multiplication!(BinaryField1b);
		Self(self.0 & rhs.0)
	}
}

impl Square for BinaryField1b {
	#[inline]
	fn square(self) -> Self {
		self
	}
}

/// For some architectures it may be faster to used SIM versions for packed fields than to use
/// portable single-element arithmetics. That's why we need these functions
#[inline]
pub(super) fn multiple_using_packed<P: PackedField>(lhs: P::Scalar, rhs: P::Scalar) -> P::Scalar {
	(P::set_single(lhs) * P::set_single(rhs)).get(0)
}

#[inline]
pub(super) fn square_using_packed<P: PackedField>(value: P::Scalar) -> P::Scalar {
	P::set_single(value).square().get(0)
}

#[inline]
pub(super) fn invert_or_zero_using_packed<P: PackedField>(value: P::Scalar) -> P::Scalar {
	P::set_single(value).invert_or_zero().get(0)
}
