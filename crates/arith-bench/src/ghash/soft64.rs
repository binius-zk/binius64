// Copyright (c) 2019-2025 The RustCrypto Project Developers
//
// Permission is hereby granted, free of charge, to any
// person obtaining a copy of this software and associated
// documentation files (the "Software"), to deal in the
// Software without restriction, including without
// limitation the rights to use, copy, modify, merge,
// publish, distribute, sublicense, and/or sell copies of
// the Software, and to permit persons to whom the Software
// is furnished to do so, subject to the following
// conditions:
//
// The above copyright notice and this permission notice
// shall be included in all copies or substantial portions
// of the Software.

// Copyright 2026 The Binius Developers

//! Constant-time software implementation of GHASH for 64-bit architectures.
//!
//! This implementation is adapted from the RustCrypto/universal-hashes repository:
//! <https://github.com/RustCrypto/universal-hashes>
//!
//! Which in turn was adapted from BearSSL's `ghash_ctmul64.c`:
//! <https://bearssl.org/gitweb/?p=BearSSL;a=blob;f=src/hash/ghash_ctmul64.c;hb=4b6046412>
//!
//! Copyright (c) 2016 Thomas Pornin <pornin@bolet.org>
//!
//! Modified by Irreducible Inc. (2024-2025): Ported from C to Rust with
//! adaptations for the Binius field arithmetic framework.

use crate::arch::portable64::{U64x2, bmul64, bsqr64, rev64};

/// Widening (unreduced) GHASH multiply: the 256-bit schoolbook polynomial product of two GHASH
/// field elements, without the modular reduction.
///
/// Returns the four 64-bit limbs `[v0, v1, v2, v3]` of the product `v0 + v1·X^64 + v2·X^128 +
/// v3·X^192`. Because the reduction ([`reduce`]) is F2-linear, these limbs can be XOR-accumulated
/// across many products and reduced only once — an inner product of `n` terms costs one reduction
/// instead of `n`.
///
/// Method described at:
/// * <https://www.bearssl.org/constanttime.html#ghash-for-gcm>
/// * <https://crypto.stackexchange.com/questions/66448/how-does-bearssls-gcm-modular-reduction-work/66462#66462>
///
/// This code does not conform to the bit-endianness requirements of the GCM specification, but is
/// a valid GHASH field multiplication with the modified representation.
#[inline]
pub fn mul_wide(x: u128, y: u128) -> [u64; 4] {
	// Convert to U64x2 representation
	let U64x2(x0, x1) = U64x2::from(x);
	let U64x2(y0, y1) = U64x2::from(y);

	// Perform multiplication
	let x0r = rev64(x0);
	let x1r = rev64(x1);
	let x2 = x0 ^ x1;
	let x2r = x0r ^ x1r;

	let y0r = rev64(y0);
	let y1r = rev64(y1);
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
	z0h = rev64(z0h) >> 1;
	z1h = rev64(z1h) >> 1;
	z2h = rev64(z2h) >> 1;

	[z0, z0h ^ z2, z1 ^ z2h, z1h]
}

/// Reduce a 256-bit widening product, given as its four 64-bit limbs `[v0, v1, v2, v3]`, to a
/// single GHASH field element modulo `X^128 + X^7 + X^2 + X + 1`.
///
/// This is an F2-linear map, so `reduce(a) ^ reduce(b) == reduce([a0 ^ b0, ..])`: unreduced
/// products may be summed by XOR and reduced once at the end.
#[inline]
pub fn reduce([mut v0, mut v1, mut v2, v3]: [u64; 4]) -> u128 {
	v1 ^= v3 ^ (v3 << 1) ^ (v3 << 2) ^ (v3 << 7);
	v2 ^= (v3 >> 63) ^ (v3 >> 62) ^ (v3 >> 57);
	v0 ^= v2 ^ (v2 << 1) ^ (v2 << 2) ^ (v2 << 7);
	v1 ^= (v2 >> 63) ^ (v2 >> 62) ^ (v2 >> 57);

	// Convert back to u128
	U64x2(v0, v1).into()
}

/// Multiply two GHASH field elements using software implementation.
///
/// Composes the widening multiply [`mul_wide`] with the modular [`reduce`]; both are inlined.
#[inline]
pub fn mul(x: u128, y: u128) -> u128 {
	reduce(mul_wide(x, y))
}

/// Square a GHASH field element using software implementation.
///
/// Exploits the fact that squaring a GF(2) polynomial is a linear operation (all cross terms
/// vanish): the square of `a₀ + a₁X + a₂X² + ...` is `a₀ + a₁X² + a₂X⁴ + ...`, i.e.
/// bit-interleaving with zeros. This avoids the carry-less multiplications
/// needed by general [`mul`], replacing them with cheaper bit-shuffle operations.
pub fn square(x: u128) -> u128 {
	// Convert to U64x2 representation
	let U64x2(x0, x1) = U64x2::from(x);

	let x0l = x0 & 0x00000000FFFFFFFF;
	let x0h = x0 >> 32;
	let x1l = x1 & 0x00000000FFFFFFFF;
	let x1h = x1 >> 32;

	reduce([bsqr64(x0l), bsqr64(x0h), bsqr64(x1l), bsqr64(x1h)])
}

/// Multiply a GHASH field element by X^{-1}.
///
/// This is equivalent to `mul(x, INV_X)` but optimized: right-shift by 1 and conditionally XOR
/// with X^{-1} if the LSB was set.
pub const fn mul_inv_x(x: u128) -> u128 {
	let lsb = x & 1;
	let shifted = x >> 1;
	// If lsb is 1, XOR with INV_X; the mask is all-ones when lsb=1, all-zeros when lsb=0.
	shifted ^ (super::INV_X & (lsb.wrapping_neg()))
}

#[cfg(test)]
mod tests {
	use proptest::prelude::*;

	use super::*;
	use crate::{
		ghash::{INV_X, ONE},
		test_utils::multiplication_tests::{
			test_mul_associative, test_mul_commutative, test_mul_distributive,
			test_square_equals_mul,
		},
	};

	proptest! {
		#[test]
		fn test_ghash_soft64_mul_commutative(
			a in any::<u128>(),
			b in any::<u128>(),
		) {
			test_mul_commutative(a, b, mul, "GHASH");
		}

		#[test]
		fn test_ghash_soft64_mul_associative(
			a in any::<u128>(),
			b in any::<u128>(),
			c in any::<u128>(),
		) {
			test_mul_associative(a, b, c, mul, "GHASH");
		}

		#[test]
		fn test_ghash_soft64_mul_distributive(
			a in any::<u128>(),
			b in any::<u128>(),
			c in any::<u128>(),
		) {
			test_mul_distributive(a, b, c, mul, "GHASH");
		}

		#[test]
		fn test_ghash_soft64_mul_identity(
			a in any::<u128>(),
		) {
			prop_assert_eq!(mul(a, ONE), a, "ONE is not the multiplicative identity in GHASH");
		}

		#[test]
		fn test_ghash_soft64_mul_inv_x(
			a in any::<u128>(),
		) {
			prop_assert_eq!(mul_inv_x(a), mul(a, INV_X), "mul_inv_x does not match mul by INV_X");
		}

		#[test]
		fn test_ghash_soft64_square(
			a in any::<u128>(),
		) {
			test_square_equals_mul(a, mul, square, "GHASH");
		}

		// The reduction is F2-linear, so accumulating two unreduced products by XOR and reducing
		// once equals reducing each and summing.
		#[test]
		fn test_ghash_soft64_wide_mul_deferred_reduction(
			a1 in any::<u128>(), b1 in any::<u128>(),
			a2 in any::<u128>(), b2 in any::<u128>(),
		) {
			let [p0, p1, p2, p3] = mul_wide(a1, b1);
			let [q0, q1, q2, q3] = mul_wide(a2, b2);
			let acc = reduce([p0 ^ q0, p1 ^ q1, p2 ^ q2, p3 ^ q3]);
			prop_assert_eq!(acc, mul(a1, b1) ^ mul(a2, b2));
		}
	}
}
