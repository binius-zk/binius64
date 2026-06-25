// Copyright 2023-2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

//! Portable implementation of packed GHASH field operations.

use bytemuck::TransparentWrapper;

use super::{
	arithmetic::{ghash::ghash_square, itoh_tsujii::invert_b128},
	m128::M128,
	univariate_mul_utils_128::{Underlier128bLanes, spread_bits_64},
};
use crate::{
	arch::PackedPrimitiveType,
	arithmetic_traits::{InvertOrZero, Square},
	ghash::BinaryField128bGhash,
};

/// Widening-multiply wrapper used by the `PackedBinaryGhash1x128b` packing.
pub type GhashWideMul1x<T> = super::arithmetic::ghash::GhashWideMul<T>;

/// Square wrapper for the `PackedBinaryGhash1x128b` packing.
pub type GhashSquare1x<T> = Ghash<T>;

/// Invert wrapper for the `PackedBinaryGhash1x128b` packing.
pub type GhashInvert1x<T> = Ghash<T>;

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

/// Square wrapper for portable GHASH field arithmetic.
#[repr(transparent)]
#[derive(TransparentWrapper)]
pub struct Ghash<T>(T);

impl Square for Ghash<PackedPrimitiveType<M128, BinaryField128bGhash>> {
	#[inline]
	fn square(self) -> Self {
		Self::wrap(ghash_square(Self::peel(self).0).into())
	}
}

impl InvertOrZero for Ghash<PackedPrimitiveType<M128, BinaryField128bGhash>> {
	#[inline]
	fn invert_or_zero(self) -> Self {
		// This portable type's underlier is the portable `M128`, which on SIMD targets differs from
		// `BinaryField128bGhash`'s underlier, so it is not `Divisible<BinaryField128bGhash>`. As a
		// width-1 packing, bridge through the scalar (whose inverse is also Itoh-Tsujii).
		let scalar = BinaryField128bGhash::new(Self::peel(self).to_underlier().into());
		Self::wrap(PackedPrimitiveType::from_underlier(M128::from_u128(u128::from(invert_b128(
			scalar,
		)))))
	}
}
