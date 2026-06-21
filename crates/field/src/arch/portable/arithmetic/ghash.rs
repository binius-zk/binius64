// Copyright 2023-2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

//! Portable (software) implementation of GHASH field multiplication.

// The widening-multiply wrapper is currently unused (the packed type uses `TrivialWideMul`), so
// allow dead code rather than annotating each item.
#![allow(dead_code)]

use bytemuck::TransparentWrapper;

use super::super::{
	m128::M128,
	univariate_mul_utils_128::{Underlier64bLanes, Underlier128bLanes, bmul64},
};
use crate::{BinaryField128bGhash as GhashB128, WideMul, arch::PackedPrimitiveType};

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

/// Widening-multiply wrapper for the portable GHASH packing.
///
/// The portable backend has no unreduced product representation, so this is an eager multiply:
/// [`wide_mul`](WideMul::wide_mul) computes the fully reduced product via [`ghash_mul`] and
/// [`reduce`](WideMul::reduce) is the identity.
#[repr(transparent)]
#[derive(bytemuck::TransparentWrapper)]
pub struct GhashWideMul<T>(T);

impl WideMul for GhashWideMul<PackedPrimitiveType<M128, GhashB128>> {
	type Output = PackedPrimitiveType<M128, GhashB128>;

	#[inline]
	fn wide_mul(a: Self, b: Self) -> Self::Output {
		let a = PackedPrimitiveType::peel(Self::peel(a));
		let b = PackedPrimitiveType::peel(Self::peel(b));
		PackedPrimitiveType::wrap(ghash_mul(a, b))
	}

	#[inline]
	fn reduce(wide: Self::Output) -> Self {
		Self::wrap(wide)
	}
}
