// Copyright 2023-2025 Irreducible Inc.
// Copyright 2026 The Binius Developers
// Copyright (c) 2019-2023 RustCrypto Developers

//! ARMv8 `PMULL`-accelerated implementation of GHASH.
//!
//! Based on the optimized GHASH implementation using carryless multiplication
//! instructions available on ARMv8 processors with NEON support.

use bytemuck::TransparentWrapper;

use super::m128::M128;
use crate::{
	BinaryField128bGhash,
	arch::PackedPrimitiveType,
	arithmetic_traits::{InvertOrZero, Square},
};

/// Widening-multiply wrapper used by the GHASH packing: the reduction-deferring
/// `GhashClMulWideMul`.
pub type GhashWideMul1x<T> = super::arithmetic::ghash::GhashClMulWideMul<T>;

/// Square wrapper for the `PackedBinaryGhash1x128b` packing.
pub type GhashSquare1x<T> = Ghash<T>;

/// Invert wrapper for the `PackedBinaryGhash1x128b` packing.
pub type GhashInvert1x<T> = Ghash<T>;

/// Square wrapper for aarch64 GHASH field arithmetic.
#[repr(transparent)]
#[derive(TransparentWrapper)]
pub struct Ghash<T>(T);

impl Square for Ghash<PackedPrimitiveType<M128, BinaryField128bGhash>> {
	#[inline]
	fn square(self) -> Self {
		Self::wrap(PackedPrimitiveType::from_underlier(super::arithmetic::ghash::square_clmul(
			Self::peel(self).to_underlier(),
		)))
	}
}

// Implement InvertOrZero for the GHASH packing (Itoh-Tsujii — no CLMUL invert)
impl InvertOrZero for Ghash<PackedPrimitiveType<M128, BinaryField128bGhash>> {
	#[inline]
	fn invert_or_zero(self) -> Self {
		Self::wrap(crate::arch::invert_b128(Self::peel(self)))
	}
}
