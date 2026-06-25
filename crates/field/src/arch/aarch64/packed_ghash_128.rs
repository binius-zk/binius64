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
	arch::PackedPrimitiveType,
	arithmetic_traits::{TaggedInvertOrZero, TaggedSquare},
};

/// Widening-multiply wrapper used by the GHASH packing: the reduction-deferring
/// `GhashClMulWideMul`.
pub type GhashWideMul1x<T> = super::arithmetic::ghash::GhashClMulWideMul<T>;

/// Square strategy for the `PackedBinaryGhash1x128b` packing.
pub type GhashSquare1x = GhashStrategy;

/// Invert strategy for the `PackedBinaryGhash1x128b` packing.
pub type GhashInvert1x = GhashStrategy;

/// Strategy for aarch64 GHASH field arithmetic operations.
pub struct GhashStrategy;

// Implement TaggedSquare for GhashStrategy
impl TaggedSquare<GhashStrategy> for PackedPrimitiveType<M128, BinaryField128bGhash> {
	#[inline]
	fn square(self) -> Self {
		Self::from_underlier(super::arithmetic::ghash::square_clmul(self.to_underlier()))
	}
}

// Implement TaggedInvertOrZero for GhashStrategy (Itoh-Tsujii — no CLMUL invert)
impl TaggedInvertOrZero<GhashStrategy> for PackedPrimitiveType<M128, BinaryField128bGhash> {
	#[inline]
	fn invert_or_zero(self) -> Self {
		crate::arch::invert_b128(self)
	}
}
