// Copyright 2024-2025 Irreducible Inc.

use std::ops::{Add, AddAssign};

use crate::PackedField;

/// Value that can be multiplied by itself
pub trait Square {
	/// Returns the value multiplied by itself
	fn square(self) -> Self;
}

/// A packed field type that supports widening (unreduced) multiplication.
///
/// The multiply phase produces a [`Wide`](Self::Wide) value that can be accumulated via addition
/// (which is XOR in GF(2)) without overflow. A single [`reduce_wide`](Self::reduce_wide) call at
/// the end converts back to the packed field representation. This is useful for inner-product-style
/// computations where `3N + 2` CLMULs (Karatsuba widening) beats `6N` (full multiply per term).
pub trait WideningMul: PackedField {
	type Wide: Copy + Default + Send + Sync + Add<Output = Self::Wide> + AddAssign;

	fn widening_mul(a: Self, b: Self) -> Self::Wide;
	fn reduce_wide(wide: Self::Wide) -> Self;
}

macro_rules! impl_trivial_widening_mul {
	($name:ty) => {
		impl $crate::arithmetic_traits::WideningMul for $name {
			type Wide = Self;

			#[inline]
			fn widening_mul(a: Self, b: Self) -> Self {
				a * b
			}

			#[inline]
			fn reduce_wide(wide: Self) -> Self {
				wide
			}
		}
	};
}

pub(crate) use impl_trivial_widening_mul;

/// Value that can be inverted
pub trait InvertOrZero {
	/// Returns the inverted value or zero in case when `self` is zero
	fn invert_or_zero(self) -> Self;
}

/// Multiplication that is parameterized with some some strategy.
pub trait TaggedMul<Strategy> {
	fn mul(self, rhs: Self) -> Self;
}

macro_rules! impl_mul_with {
	($name:ident @ $strategy:ty) => {
		impl std::ops::Mul for $name {
			type Output = Self;

			#[inline]
			fn mul(self, rhs: Self) -> Self {
				$crate::tracing::trace_multiplication!($name);

				$crate::arithmetic_traits::TaggedMul::<$strategy>::mul(self, rhs)
			}
		}
	};
	($name:ty => $bigger:ty) => {
		impl std::ops::Mul for $name {
			type Output = Self;

			#[inline]
			fn mul(self, rhs: Self) -> Self {
				$crate::arch::portable::packed::mul_as_bigger_type::<_, $bigger>(self, rhs)
			}
		}
	};
}

pub(crate) use impl_mul_with;

/// Square operation that is parameterized with some some strategy.
pub trait TaggedSquare<Strategy> {
	fn square(self) -> Self;
}

macro_rules! impl_square_with {
	($name:ident @ $strategy:ty) => {
		impl $crate::arithmetic_traits::Square for $name {
			#[inline]
			fn square(self) -> Self {
				$crate::arithmetic_traits::TaggedSquare::<$strategy>::square(self)
			}
		}
	};
	($name:ty => $bigger:ty) => {
		impl $crate::arithmetic_traits::Square for $name {
			#[inline]
			fn square(self) -> Self {
				$crate::arch::portable::packed::square_as_bigger_type::<_, $bigger>(self)
			}
		}
	};
}

pub(crate) use impl_square_with;

/// Invert or zero operation that is parameterized with some some strategy.
pub trait TaggedInvertOrZero<Strategy> {
	fn invert_or_zero(self) -> Self;
}

macro_rules! impl_invert_with {
	($name:ident @ $strategy:ty) => {
		impl $crate::arithmetic_traits::InvertOrZero for $name {
			#[inline]
			fn invert_or_zero(self) -> Self {
				$crate::arithmetic_traits::TaggedInvertOrZero::<$strategy>::invert_or_zero(self)
			}
		}
	};
	($name:ty => $bigger:ty) => {
		impl $crate::arithmetic_traits::InvertOrZero for $name {
			#[inline]
			fn invert_or_zero(self) -> Self {
				$crate::arch::portable::packed::invert_as_bigger_type::<_, $bigger>(self)
			}
		}
	};
}

pub(crate) use impl_invert_with;
