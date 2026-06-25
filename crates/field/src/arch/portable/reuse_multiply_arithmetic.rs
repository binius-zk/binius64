// Copyright 2024-2025 Irreducible Inc.

use std::ops::Mul;

use bytemuck::TransparentWrapper;

use crate::arithmetic_traits::Square;

/// Square wrapper that reuses the type's own multiplication: `square(x) = x * x`.
#[repr(transparent)]
#[derive(TransparentWrapper)]
pub struct ReuseMultiply<T>(T);

impl<T> Square for ReuseMultiply<T>
where
	T: Mul<T, Output = T> + Copy,
{
	#[inline]
	fn square(self) -> Self {
		let val = Self::peel(self);
		Self::wrap(val * val)
	}
}
