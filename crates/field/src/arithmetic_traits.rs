// Copyright 2024-2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use std::ops::{Add, AddAssign, Sub, SubAssign};

/// Value that can be multiplied by itself
pub trait Square {
	/// Returns the value multiplied by itself
	fn square(self) -> Self;
}

/// A field type that supports widening (unreduced) multiplication.
///
/// The multiply phase produces an [`Output`](Self::Output) value that can be accumulated via
/// addition without overflow (XOR in characteristic 2). A single [`reduce`](Self::reduce) call at
/// the end converts back to the field representation. For `GF(2^128)` inner products this lets us
/// amortize the reduction across many products, which is a net win when reductions are comparable
/// in cost to the widening multiply itself.
///
/// `WideMul` is a parent trait of both [`Field`](crate::Field) and
/// [`PackedField`](crate::PackedField), so every field and packed field supports it (and each type
/// implements it directly, leaving room for specialized impls). Most types use the trivial
/// implementation — multiply eagerly, reduce to the identity — except the `GF(2^128)` scalar field
/// and its CLMUL-accelerated packings (x86_64 and AArch64), which defer the reduction by
/// accumulating an unreduced `WideGhashProduct`.
pub trait WideMul: Sized {
	type Output: Default
		+ Add<Output = Self::Output>
		+ AddAssign
		+ Sub<Output = Self::Output>
		+ SubAssign;

	fn wide_mul(a: Self, b: Self) -> Self::Output;
	fn reduce(wide: Self::Output) -> Self;
}

macro_rules! impl_trivial_wide_mul {
	($name:ty) => {
		impl $crate::arithmetic_traits::WideMul for $name {
			type Output = Self;

			#[inline]
			fn wide_mul(a: Self, b: Self) -> Self {
				a * b
			}

			#[inline]
			fn reduce(wide: Self) -> Self {
				wide
			}
		}
	};
}

pub(crate) use impl_trivial_wide_mul;

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
