// Copyright 2023-2025 Irreducible Inc.
// Copyright 2026 The Binius Developers
// Copyright (c) 2019-2023 RustCrypto Developers

//! WASM32 implementation of GHASH.
//!
//! Based on the portable GHASH implementation adapted for WebAssembly
//! using WASM SIMD instructions where available.

use std::{arch::wasm32::*, ops::Mul};

use super::{super::portable::packed::PackedPrimitiveType, m128::M128};
use crate::{
	BinaryField128bGhash,
	arch::{
		PairwiseStrategy,
		portable::{
			arithmetic::ghash::ghash_mul,
			packed_macros::impl_broadcast,
			univariate_mul_utils_128::{Underlier128bLanes, spread_bits_64},
		},
	},
	arithmetic_traits::{InvertOrZero, Square, impl_transformation_with_strategy},
};

pub type PackedBinaryGhash1x128b = PackedPrimitiveType<M128, BinaryField128bGhash>;

/// Widening-multiply wrapper used by the GHASH packing. The WASM SIMD packing multiplies eagerly,
/// so this is the trivial (identity-reduce) wrapper.
pub type GhashWideMul<T> = crate::arithmetic_traits::TrivialWideMul<T>;

// Define broadcast
impl_broadcast!(M128, BinaryField128bGhash);

// Define multiply
impl Mul for PackedBinaryGhash1x128b {
	type Output = Self;

	#[inline]
	fn mul(self, rhs: Self) -> Self::Output {
		Self::from_underlier(ghash_mul(self.0, rhs.0))
	}
}

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

impl Square for PackedBinaryGhash1x128b {
	#[inline]
	fn square(self) -> Self {
		Self::from_underlier(crate::arch::portable::arithmetic::ghash::ghash_square(
			self.to_underlier(),
		))
	}
}

// Define invert
impl InvertOrZero for PackedBinaryGhash1x128b {
	#[inline]
	fn invert_or_zero(self) -> Self {
		crate::arch::invert_b128(self)
	}
}

// Implement WideMul (trivial: no CLMUL, just regular multiply)
crate::arithmetic_traits::impl_trivial_wide_mul!(PackedBinaryGhash1x128b);

// Define linear transformations
impl_transformation_with_strategy!(PackedBinaryGhash1x128b, PairwiseStrategy);
