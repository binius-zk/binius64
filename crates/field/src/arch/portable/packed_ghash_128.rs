// Copyright 2023-2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

//! Portable implementation of packed GHASH field operations.

use super::{
	m128::M128,
	univariate_mul_utils_128::{Underlier128bLanes, spread_bits_64},
};

/// Widening-multiply wrapper used by the `PackedBinaryGhash1x128b` packing.
pub type GhashWideMul1x<T> = super::arithmetic::ghash::GhashWideMul<T>;

/// Square wrapper for the `PackedBinaryGhash1x128b` packing: the shared software square.
pub type GhashSquare1x<T> = super::arithmetic::ghash::GhashSoftMul<T>;

/// Invert wrapper for the `PackedBinaryGhash1x128b` packing: the shared Itoh-Tsujii inversion.
pub type GhashInvert1x<T> = super::arithmetic::itoh_tsujii::GhashItohTsujii<T>;

// `M128` packs its GHASH 64-bit lanes the same way `u128` does — delegate through `u128`.
impl Underlier128bLanes for M128 {
	type U64 = u64;

	#[inline(always)]
	fn split_hi_lo_64(self) -> (u64, u64) {
		u128::from(self).split_hi_lo_64()
	}

	#[inline(always)]
	fn join_u64s(high: u64, low: u64) -> Self {
		Self::from(u128::join_u64s(high, low))
	}

	#[inline(always)]
	fn broadcast_64(val: u64) -> Self {
		Self::from(u128::broadcast_64(val))
	}

	#[inline(always)]
	fn spread_bits_128(self) -> (Self, Self) {
		let (hi, lo) = self.split_hi_lo_64();
		(Self::from(spread_bits_64(hi)), Self::from(spread_bits_64(lo)))
	}
}
