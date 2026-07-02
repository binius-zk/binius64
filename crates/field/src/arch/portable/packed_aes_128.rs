// Copyright 2024-2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use super::m128::M128;
use crate::{
	aes_field::AESTowerField8b,
	arch::{
		PackedAesArithmetic, PackedPrimitiveType,
		aes_arithmetic::{bytewise_invert_or_zero, bytewise_square, bytewise_wide_mul},
	},
};

impl PackedAesArithmetic for M128 {
	type WideProduct = PackedPrimitiveType<M128, AESTowerField8b>;

	#[inline]
	fn wide_mul(a: Self, b: Self) -> Self::WideProduct {
		bytewise_wide_mul(a, b)
	}

	#[inline]
	fn reduce(wide: Self::WideProduct) -> Self {
		wide.to_underlier()
	}

	#[inline]
	fn square(a: Self) -> Self {
		bytewise_square(a)
	}

	#[inline]
	fn invert_or_zero(a: Self) -> Self {
		bytewise_invert_or_zero(a)
	}
}
