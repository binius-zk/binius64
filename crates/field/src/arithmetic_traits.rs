// Copyright 2024-2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use std::{
	fmt::Debug,
	iter::Sum,
	ops::{Add, AddAssign, Sub, SubAssign},
};

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
		+ Clone
		+ Sum
		+ Add<Output = Self::Output>
		+ AddAssign
		+ Sub<Output = Self::Output>
		+ SubAssign;

	fn wide_mul(a: Self, b: Self) -> Self::Output;
	fn reduce(wide: Self::Output) -> Self;
}

/// A field whose multiplier can be preprocessed once and then applied to many values.
///
/// Every hot loop in the prover multiplies by a loop constant.
///
/// A broadcast NTT twiddle and a broadcast sumcheck challenge are both that shape.
///
/// Work spent once on the multiplier then buys a cheaper multiply for every value it reaches.
///
/// This is a parent trait of both the scalar and the packed field traits, so every field has it.
///
/// Most fields use the trivial form: the prepared multiplier is the multiplier, and applying it is
/// ordinary multiplication.
///
/// The `GF(2^128)` field and its carry-less-multiply packings instead precompute the multiplier
/// scaled by `X^64`.
///
/// # Invariant
///
/// Preparing a multiplier and applying it must equal multiplying by it directly.
pub trait PreparedMul: Sized {
	/// The preprocessed form of a multiplier.
	type Prepared: Copy + Send + Sync + Debug;

	/// Preprocesses a multiplier for repeated use.
	fn prepare(self) -> Self::Prepared;

	/// Multiplies `self` by a preprocessed multiplier.
	fn mul_prepared(self, rhs: &Self::Prepared) -> Self;
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

// Wires the prepared multiply to a strategy wrapper, as squaring and inversion are wired.
// The wrapper's `Prepared` type is reused verbatim, so the strategy owns the representation.
macro_rules! impl_prepared_mul_with {
	($name:ident @ $($strategy:tt)*) => {
		impl $crate::arithmetic_traits::PreparedMul for $name {
			type Prepared =
				<$($strategy)* <$name> as $crate::arithmetic_traits::PreparedMul>::Prepared;

			#[inline]
			fn prepare(self) -> Self::Prepared {
				$crate::arithmetic_traits::PreparedMul::prepare(
					<$($strategy)* <$name> as ::bytemuck::TransparentWrapper<$name>>::wrap(self),
				)
			}

			#[inline]
			fn mul_prepared(self, rhs: &Self::Prepared) -> Self {
				<$($strategy)* <$name> as ::bytemuck::TransparentWrapper<$name>>::peel(
					$crate::arithmetic_traits::PreparedMul::mul_prepared(
						<$($strategy)* <$name> as ::bytemuck::TransparentWrapper<$name>>::wrap(self),
						rhs,
					),
				)
			}
		}
	};
}

pub(crate) use impl_prepared_mul_with;

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
