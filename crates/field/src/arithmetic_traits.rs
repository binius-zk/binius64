// Copyright 2024-2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use std::{
	iter::Sum,
	ops::{Add, AddAssign, Mul, Sub, SubAssign},
};

use bytemuck::TransparentWrapper;

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

#[derive(TransparentWrapper)]
#[repr(transparent)]
pub struct TrivialWideMul<T>(T);

impl<T> WideMul for TrivialWideMul<T>
where
	T: Default
		+ Clone
		+ Sum
		+ Add<Output = T>
		+ AddAssign
		+ Sub<Output = T>
		+ SubAssign
		+ Mul<Output = T>,
{
	type Output = T;

	#[inline]
	fn wide_mul(a: Self, b: Self) -> T {
		Self::peel(a) * Self::peel(b)
	}

	#[inline]
	fn reduce(wide: T) -> Self {
		Self::wrap(wide)
	}
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

// The `@ strategy` arm wires `$name`'s `Mul` to a strategy wrapper: a `TransparentWrapper` struct
// (e.g. `Gfni`, `MulFromWideMul`) that carries the actual algorithm. We wrap the inputs, run
// the wrapper's `Mul`, and peel the result. `$strategy` is captured as raw token-trees (not
// `:ty`/`:path`) because a matched type fragment is opaque and can't have `<$name>` appended to it.
macro_rules! impl_mul_with {
	($name:ident @ $($strategy:tt)*) => {
		impl std::ops::Mul for $name {
			type Output = Self;

			#[inline]
			fn mul(self, rhs: Self) -> Self {
				$crate::tracing::trace_multiplication!($name);

				<$($strategy)* <$name> as ::bytemuck::TransparentWrapper<$name>>::peel(
					<$($strategy)* <$name> as ::bytemuck::TransparentWrapper<$name>>::wrap(self)
						* <$($strategy)* <$name> as ::bytemuck::TransparentWrapper<$name>>::wrap(rhs),
				)
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
