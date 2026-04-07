// Copyright 2024-2025 Irreducible Inc.

use std::ops::Mul;

use crate::{arch::ReuseMultiplyStrategy, arithmetic_traits::TaggedSquare};

impl<T> TaggedSquare<ReuseMultiplyStrategy> for T
where
	T: Mul<Self, Output = Self> + Copy,
{
	fn square(self) -> Self {
		self * self
	}
}
