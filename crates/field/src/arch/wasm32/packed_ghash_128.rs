// Copyright 2023-2025 Irreducible Inc.
// Copyright 2026 The Binius Developers
// Copyright (c) 2019-2023 RustCrypto Developers

//! WASM32 implementation of GHASH.
//!
//! Based on the portable GHASH implementation adapted for WebAssembly
//! using WASM SIMD instructions where available.

use std::arch::wasm32::*;

use super::{super::portable::packed::PackedPrimitiveType, m128::M128};
use crate::{
	BinaryField128bGhash,
	arch::{
		PairwiseStrategy,
		portable::{
			packed_macros::impl_broadcast,
			univariate_mul_utils_128::{Underlier128bLanes, spread_bits_64},
		},
	},
	arithmetic_traits::impl_transformation_with_strategy,
};

/// Widening-multiply wrapper used by the GHASH packing: the reduction-deferring portable
/// [`GhashWideMul`](crate::arch::portable::arithmetic::ghash::GhashWideMul). The WASM SIMD `M128`
/// implements [`Underlier128bLanes`], so the portable schoolbook widening multiply applies.
pub type GhashWideMul1x<T> = crate::arch::portable::arithmetic::ghash::GhashWideMul<T>;

/// Square wrapper for the `PackedBinaryGhash1x128b` packing: the shared software square (the WASM
/// SIMD `M128` implements [`Underlier128bLanes`], so the portable bit-spread square applies).
pub type GhashSquare1x<T> = crate::arch::portable::arithmetic::ghash::GhashSoftMul<T>;

/// Invert wrapper for the `PackedBinaryGhash1x128b` packing: the shared Itoh-Tsujii inversion.
pub type GhashInvert1x<T> = crate::arch::portable::arithmetic::itoh_tsujii::GhashItohTsujii<T>;

// Define broadcast
impl_broadcast!(M128, BinaryField128bGhash);

impl Underlier128bLanes for M128 {
	type U64 = u64;

	#[inline(always)]
	fn split_hi_lo_64(self) -> (Self::U64, Self::U64) {
		(u64x2_extract_lane::<0>(self.0), u64x2_extract_lane::<1>(self.0))
	}

	#[inline(always)]
	fn join_u64s(high: Self::U64, low: Self::U64) -> Self {
		M128::from(u64x2(low, high))
	}

	#[inline(always)]
	fn broadcast_64(val: u64) -> Self {
		M128::from(u64x2_splat(val))
	}

	#[inline(always)]
	fn spread_bits_128(self) -> (Self, Self) {
		let (hi, lo) = self.split_hi_lo_64();

		(Self::from(spread_bits_64(hi)), Self::from(spread_bits_64(lo)))
	}
}

// Define linear transformations
impl_transformation_with_strategy!(
	PackedPrimitiveType<M128, BinaryField128bGhash>,
	PairwiseStrategy
);
