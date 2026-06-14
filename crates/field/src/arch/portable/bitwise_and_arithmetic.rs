// Copyright 2025-2026 The Binius Developers

use super::packed::PackedPrimitiveType;
use crate::{
	BinaryField1b,
	arch::BitwiseAndStrategy,
	arithmetic_traits::{TaggedInvertOrZero, TaggedMul, TaggedSquare},
	underlier::UnderlierType,
};

impl<U: UnderlierType> TaggedMul<BitwiseAndStrategy> for PackedPrimitiveType<U, BinaryField1b> {
	#[inline]
	fn mul(self, rhs: Self) -> Self {
		(self.0 & rhs.0).into()
	}
}

impl<U: UnderlierType> TaggedSquare<BitwiseAndStrategy> for PackedPrimitiveType<U, BinaryField1b> {
	#[inline]
	fn square(self) -> Self {
		self
	}
}

impl<U: UnderlierType> TaggedInvertOrZero<BitwiseAndStrategy>
	for PackedPrimitiveType<U, BinaryField1b>
{
	#[inline]
	fn invert_or_zero(self) -> Self {
		self
	}
}
