// Copyright 2023-2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

//! Portable implementation of packed GHASH field operations.

pub use super::arithmetic::ghash::GhashWideMul;
use super::{
	arithmetic::{
		ghash::{ghash_mul, ghash_square},
		itoh_tsujii::invert_b128,
	},
	m128::M128,
	univariate_mul_utils_128::{Underlier128bLanes, spread_bits_64},
};
use crate::{
	arch::portable::packed_macros::{portable_macros::*, *},
	arithmetic_traits::{TaggedInvertOrZero, TaggedMul, TaggedSquare},
	ghash::BinaryField128bGhash,
};

/// Strategy for GHASH field arithmetic operations.
pub struct GhashStrategy;

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

define_packed_binary_field!(
	PackedBinaryGhash1x128b,
	BinaryField128bGhash,
	M128,
	(GhashStrategy),
	(GhashStrategy),
	(GhashStrategy),
	(GhashWideMul)
);

impl TaggedMul<GhashStrategy> for PackedBinaryGhash1x128b {
	#[inline]
	fn mul(self, rhs: Self) -> Self {
		ghash_mul(self.0, rhs.0).into()
	}
}

impl TaggedSquare<GhashStrategy> for PackedBinaryGhash1x128b {
	#[inline]
	fn square(self) -> Self {
		ghash_square(self.0).into()
	}
}

impl TaggedInvertOrZero<GhashStrategy> for PackedBinaryGhash1x128b {
	#[inline]
	fn invert_or_zero(self) -> Self {
		// This portable type's underlier is the portable `M128`, which on SIMD targets differs from
		// `BinaryField128bGhash`'s underlier, so it is not `Divisible<BinaryField128bGhash>`. As a
		// width-1 packing, bridge through the scalar (whose inverse is also Itoh-Tsujii).
		let scalar = BinaryField128bGhash::new(self.to_underlier().into());
		Self::from_underlier(M128::from_u128(u128::from(invert_b128(scalar))))
	}
}
