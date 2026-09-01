// Copyright 2024-2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use std::{
	iter::Sum,
	ops::{Add, AddAssign, Sub, SubAssign},
};

/// Value that can be multiplied by itself
pub trait Square {
	/// Returns the value multiplied by itself
	fn square(self) -> Self;
}

/// Scales a value by `X`, the generator of the field's polynomial basis.
///
/// A one-bit shift plus a masked exclusive or, not a field multiply.
///
/// The scaling is `GF(2)`-linear, so it commutes with the modular reduction.
/// That is what lets one trait serve both a reduced element and an unreduced product:
///
/// ```text
///     reduce(mul_x(wide)) == mul_x(reduce(wide))
/// ```
///
/// Scaling an unreduced product folds the `X` of an irreducible polynomial into a reduction the
/// caller is going to pay for anyway.
pub trait MulX {
	/// Returns the value scaled by `X`.
	#[must_use]
	fn mul_x(self) -> Self;
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
		+ Clone
		+ Sum
		+ Add<Output = Self::Output>
		+ AddAssign
		+ Sub<Output = Self::Output>
		+ SubAssign;

	fn wide_mul(a: Self, b: Self) -> Self::Output;
	fn reduce(wide: Self::Output) -> Self;
}

/// Value that can be inverted
pub trait InvertOrZero {
	/// Returns the inverted value or zero in case when `self` is zero
	fn invert_or_zero(self) -> Self;

	/// Returns the multiplicative inverse.
	///
	/// ## Safety
	/// Requires that `self` is non-zero. Behavior is undefined otherwise.
	#[inline]
	unsafe fn invert(self) -> Self
	where
		Self: Sized,
	{
		self.invert_or_zero()
	}
}

// A strategy is captured as raw token-trees rather than a type fragment.
// A matched type fragment is opaque, so it cannot take the packed type as a generic argument.
macro_rules! impl_square_with {
	($name:ident @ $($strategy:tt)*) => {
		impl $crate::arithmetic_traits::Square for $name {
			#[inline]
			fn square(self) -> Self {
				<$($strategy)* <$name> as ::bytemuck::TransparentWrapper<$name>>::peel(
					$crate::arithmetic_traits::Square::square(
						<$($strategy)* <$name> as ::bytemuck::TransparentWrapper<$name>>::wrap(self),
					),
				)
			}
		}
	};
}

pub(crate) use impl_square_with;

macro_rules! impl_mul_x_with {
	($name:ident @ $($strategy:tt)*) => {
		impl $crate::arithmetic_traits::MulX for $name {
			#[inline]
			fn mul_x(self) -> Self {
				<$($strategy)* <$name> as ::bytemuck::TransparentWrapper<$name>>::peel(
					$crate::arithmetic_traits::MulX::mul_x(
						<$($strategy)* <$name> as ::bytemuck::TransparentWrapper<$name>>::wrap(self),
					),
				)
			}
		}
	};
}

pub(crate) use impl_mul_x_with;

macro_rules! impl_invert_with {
	($name:ident @ $($strategy:tt)*) => {
		impl $crate::arithmetic_traits::InvertOrZero for $name {
			#[inline]
			fn invert_or_zero(self) -> Self {
				<$($strategy)* <$name> as ::bytemuck::TransparentWrapper<$name>>::peel(
					$crate::arithmetic_traits::InvertOrZero::invert_or_zero(
						<$($strategy)* <$name> as ::bytemuck::TransparentWrapper<$name>>::wrap(self),
					),
				)
			}
		}
	};
}

pub(crate) use impl_invert_with;
