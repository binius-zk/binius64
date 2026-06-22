// Copyright 2023-2025 Irreducible Inc.
// Copyright 2026 The Binius Developers
// Copyright (c) 2019-2023 RustCrypto Developers

//! ARMv8 `PMULL`-accelerated implementation of GHASH.
//!
//! Based on the optimized GHASH implementation using carryless multiplication
//! instructions available on ARMv8 processors with NEON support.

use super::m128::M128;
use crate::{
	BinaryField128bGhash,
	arch::portable::packed_macros::{portable_macros::*, *},
	arithmetic_traits::{
		TaggedInvertOrZero, TaggedMul, TaggedSquare, impl_invert_with, impl_mul_with,
		impl_square_with,
	},
};

/// Widening-multiply wrapper used by the GHASH packing: the reduction-deferring
/// `GhashClMulWideMul`.
pub type GhashWideMul<T> = super::arithmetic::ghash::GhashClMulWideMul<T>;

/// Strategy for aarch64 GHASH field arithmetic operations.
pub struct GhashStrategy;

// Define PackedBinaryGhash1x128b using the macro
define_packed_binary_field!(
	PackedBinaryGhash1x128b,
	BinaryField128bGhash,
	M128,
	(GhashStrategy),
	(GhashStrategy),
	(GhashStrategy),
	(GhashWideMul)
);

// Implement TaggedMul for GhashStrategy
impl TaggedMul<GhashStrategy> for PackedBinaryGhash1x128b {
	#[inline]
	fn mul(self, rhs: Self) -> Self {
		Self::from_underlier(super::arithmetic::ghash::mul_clmul(
			self.to_underlier(),
			rhs.to_underlier(),
		))
	}
}

// Implement TaggedSquare for GhashStrategy
impl TaggedSquare<GhashStrategy> for PackedBinaryGhash1x128b {
	#[inline]
	fn square(self) -> Self {
		Self::from_underlier(super::arithmetic::ghash::square_clmul(self.to_underlier()))
	}
}

// Implement TaggedInvertOrZero for GhashStrategy (Itoh-Tsujii — no CLMUL invert)
impl TaggedInvertOrZero<GhashStrategy> for PackedBinaryGhash1x128b {
	#[inline]
	fn invert_or_zero(self) -> Self {
		crate::arch::invert_b128(self)
	}
}
