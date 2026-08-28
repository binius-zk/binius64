// Copyright 2024-2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use std::{
	iter::Sum,
	ops::{Add, AddAssign, Sub, SubAssign},
};

use crate::underlier::Underlier;

/// Value that can be multiplied by itself
pub trait Square {
	/// Returns the value multiplied by itself
	fn square(self) -> Self;
}

/// Scales a field element by `X`, the generator of its polynomial basis.
///
/// A one-bit shift plus a masked exclusive or, not a field multiply.
/// The scaling is `GF(2)`-linear, so it applies equally to an unreduced product.
pub trait MulX {
	/// Returns the element scaled by `X`.
	#[must_use]
	fn mul_x(self) -> Self;
}

/// An underlier whose 128-bit lanes each hold a GHASH element that can be scaled by `X`.
///
/// The per-architecture dispatch point behind [`MulX`] for GHASH.
pub trait GhashMulX: Underlier {
	/// Returns the value with every 128-bit lane scaled by `X`.
	#[must_use]
	fn ghash_mul_x(self) -> Self;
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

/// An unreduced widening product (a [`WideMul::Output`]) that can be scaled by the field element
/// `X` while still unreduced.
///
/// Scaling by `X` and the modular reduction are both `GF(2)`-linear, and they commute:
/// `reduce(wide.mul_x_wide()) == reduce(wide).mul_x()`. Doing the scaling on the unreduced product
/// lets an extension-field multiply fold the `X` of its irreducible polynomial into a product it is
/// going to reduce anyway, saving a reduction over scaling the reduced coordinate.
pub trait MulXWide {
	/// Returns the unreduced product scaled by `X`.
	fn mul_x_wide(self) -> Self;
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
