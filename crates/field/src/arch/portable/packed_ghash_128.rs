// Copyright 2023-2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

//! Portable implementation of packed GHASH field operations.

use super::{
	arithmetic::itoh_tsujii::invert_b128,
	m128::M128,
	univariate_mul_utils_128::{Underlier64bLanes, Underlier128bLanes, bmul64, spread_bits_64},
};
use crate::{
	arch::portable::packed_macros::{portable_macros::*, *},
	arithmetic_traits::{TaggedInvertOrZero, TaggedMul, TaggedSquare},
	ghash::BinaryField128bGhash,
};

/// Strategy for GHASH field arithmetic operations.
pub struct GhashStrategy;

/// Multiply two GHASH field elements using software implementation.
///
/// Method described at:
/// * <https://www.bearssl.org/constanttime.html#ghash-for-gcm>
/// * <https://crypto.stackexchange.com/questions/66448/how-does-bearssls-gcm-modular-reduction-work/66462#66462>
///
/// This code does not conform to the bit-endianness requirements of the GCM specification, but is
/// a valid GHASH field multiplication with the modified representation.
#[inline]
pub fn ghash_mul<U: Underlier128bLanes>(x: U, y: U) -> U {
	// Convert to U64x2 representation
	let (x1, x0) = U::split_hi_lo_64(x);
	let (y1, y0) = U::split_hi_lo_64(y);

	// Perform multiplication
	let x0r = x0.reverse_bits_64();
	let x1r = x1.reverse_bits_64();
	let x2 = x0 ^ x1;
	let x2r = x0r ^ x1r;

	let y0r = y0.reverse_bits_64();
	let y1r = y1.reverse_bits_64();
	let y2 = y0 ^ y1;
	let y2r = y0r ^ y1r;

	let z0 = bmul64(y0, x0);
	let z1 = bmul64(y1, x1);
	let mut z2 = bmul64(y2, x2);

	let mut z0h = bmul64(y0r, x0r);
	let mut z1h = bmul64(y1r, x1r);
	let mut z2h = bmul64(y2r, x2r);

	z2 ^= z0 ^ z1;
	z2h ^= z0h ^ z1h;
	z0h = z0h.reverse_bits_64().shr_64(1);
	z1h = z1h.reverse_bits_64().shr_64(1);
	z2h = z2h.reverse_bits_64().shr_64(1);

	let v0 = z0;
	let v1 = z0h ^ z2;
	let v2 = z1 ^ z2h;
	let v3 = z1h;

	reduce_64(v0, v1, v2, v3)
}

#[inline]
pub fn ghash_square<U: Underlier128bLanes>(x: U) -> U {
	// Squared value in the polynomial basis is just a value with bits interleaved with zeroes.
	let (hi, lo) = x.spread_bits_128();

	let (v3, v2) = hi.split_hi_lo_64();
	let (v1, v0) = lo.split_hi_lo_64();

	reduce_64(v0, v1, v2, v3)
}

/// Reduce a 256-bit value represented as four 64-bit values by the GHASH polynomial.
#[inline]
fn reduce_64<U: Underlier128bLanes>(
	mut v0: U::U64,
	mut v1: U::U64,
	mut v2: U::U64,
	v3: U::U64,
) -> U {
	// Reduce modulo X^64 + X^7 + X^2 + X + 1.
	v1 ^= v3 ^ v3.shl_64(1) ^ v3.shl_64(2) ^ v3.shl_64(7);
	v2 ^= v3.shr_64(63) ^ v3.shr_64(62) ^ v3.shr_64(57);
	v0 ^= v2 ^ v2.shl_64(1) ^ v2.shl_64(2) ^ v2.shl_64(7);
	v1 ^= v2.shr_64(63) ^ v2.shr_64(62) ^ v2.shr_64(57);

	// Convert back to 128-bit lanes
	U::join_u64s(v1, v0)
}

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
	(GhashStrategy)
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

crate::arithmetic_traits::impl_trivial_wide_mul!(PackedBinaryGhash1x128b);

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
