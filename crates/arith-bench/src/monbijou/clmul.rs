// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers
//! CLMUL-based multiplication for the Monbijou field and its degree-2 extension.
//!
//! These implementations use carry-less multiplication (CLMUL) CPU instructions for efficient
//! field multiplication on modern x86_64 processors. The algorithm is optimized for SIMD
//! parallelism, processing multiple field elements simultaneously when using vector types like
//! __m128i or __m256i.

use crate::{PackedUnderlier, Underlier, underlier::OpsClmul};

/// Multiplies two elements in GF(2^64) using SIMD carry-less multiplication.
///
/// This function performs multiplication in the Monbijou field GF(2^64) defined by
/// the reduction polynomial X^64 + X^4 + X^3 + X + 1. The algorithm uses a two-stage
/// reduction process to efficiently handle the polynomial reduction after multiplication.
#[inline]
#[allow(dead_code)]
pub fn mul<U: Underlier + OpsClmul + PackedUnderlier<u64>>(a: U, b: U) -> U {
	// Step 1: Carry-less multiplication of 64-bit operands produces 128-bit results
	// For SIMD types, this processes multiple pairs in parallel
	let prod_0 = U::clmulepi64::<0x00>(a, b); // 128-bit pre-reduction product elements 0
	let prod_1 = U::clmulepi64::<0x11>(a, b); // 128-bit pre-reduction product elements 1
	reduce_pair(prod_0, prod_1)
}

/// Multiplies two elements in GF(2^128), represented as a degree-2 extension of GF(2^64).
///
/// This field is defined as GF(2)[X, Y] / (X^64 + X^4 + X^3 + X + 1) / (Y^2 + XY + 1).
#[inline]
pub fn mul_128b<U: Underlier + OpsClmul + PackedUnderlier<u64>>(x: U, y: U) -> U {
	// This is the bit representation of the lower-degree terms (X^4 + X^3 + X + 1)
	const POLY: u64 = 0x1B;
	let poly = <U as PackedUnderlier<u64>>::broadcast(POLY);

	// t0 = x.lo * y.lo
	let t0 = U::clmulepi64::<0x00>(x, y);
	// t2 = x.hi * y.hi
	let t2 = U::clmulepi64::<0x11>(x, y);

	// t1a = x.lo * y.hi
	let t1a = U::clmulepi64::<0x01>(x, y);
	// t1b = x.hi * y.lo
	let t1b = U::clmulepi64::<0x10>(x, y);
	// t1 = t1a + t1b (XOR in binary field)
	let t1 = U::xor(t1a, t1b);

	let mut t2_times_x = U::slli_epi64::<1>(t2);
	let t2_overflow_mask = U::movepi64_mask(t2);
	let t2_overflow_redc = U::and(poly, t2_overflow_mask);
	t2_times_x = U::xor(t2_overflow_redc, t2_times_x);

	let term0 = U::xor(t0, t2);
	let term1 = U::xor(t1, t2_times_x);

	reduce_pair(term0, term1)
}

/// Multiplies two elements of GF(2^128), the degree-2 extension of the Monbijou field, in
/// *sliced* representation.
///
/// Each element is given as `[U; 2]`: index 0 holds coefficient 0 and index 1 holds
/// coefficient 1, where each coefficient is an element (or SIMD pack of elements) of the base
/// field GF(2^64). This transposed layout keeps the two coefficients in separate registers, so
/// the multiplication is built from base-field carry-less multiplications and processes every
/// packed lane in parallel. It computes the same field product as [`mul_128b`], which
/// instead packs the two coefficients into the low and high halves of a single value.
///
/// The extension is GF(2)\[X, Y\] / (X^64 + X^4 + X^3 + X + 1) / (Y^2 + XY + 1), so `Y^2 = XY + 1`
/// and, writing `a = a0 + a1·Y` and `b = b0 + b1·Y`,
///
/// ```text
/// coeff 0 = a0·b0 + a1·b1
/// coeff 1 = a0·b1 + a1·b0 + X·(a1·b1)
/// ```
///
/// The base-field products are kept unreduced and combined into the two output coefficients, so
/// only a single `reduce_pair` is spent per coefficient (one reduction per element), matching
/// [`mul_128b`]. The cross term `a0·b1 + a1·b0` is recovered from Karatsuba as `t1 + t0 +
/// t2`.
#[inline]
pub fn mul_sliced_128b<U: Underlier + OpsClmul + PackedUnderlier<u64>>(
	x: [U; 2],
	y: [U; 2],
) -> [U; 2] {
	// The low-degree terms of the base-field reduction polynomial (X^4 + X^3 + X + 1), used to
	// multiply by X.
	const POLY: u64 = 0x1B;
	let poly = <U as PackedUnderlier<u64>>::broadcast(POLY);

	let [x0, x1] = x;
	let [y0, y1] = y;
	// Karatsuba middle inputs: (a0 + a1) and (b0 + b1).
	let xm = U::xor(x0, x1);
	let ym = U::xor(y0, y1);

	// Unreduced base-field products, one 128-bit product per lane (the `0x00`/`0x11` immediates
	// select the low/high halves of each 128-bit lane, as in `mul`).
	// t0 = a0·b0, t2 = a1·b1, t1 = (a0 + a1)·(b0 + b1).
	let t0_lo = U::clmulepi64::<0x00>(x0, y0);
	let t0_hi = U::clmulepi64::<0x11>(x0, y0);
	let t2_lo = U::clmulepi64::<0x00>(x1, y1);
	let t2_hi = U::clmulepi64::<0x11>(x1, y1);
	let t1_lo = U::clmulepi64::<0x00>(xm, ym);
	let t1_hi = U::clmulepi64::<0x11>(xm, ym);

	// X·(a1·b1) on the unreduced product, per lane: shift left by one and fold the overflow bit
	// (bit 63) back in with the reduction polynomial.
	let t2x_lo = U::xor(U::slli_epi64::<1>(t2_lo), U::and(poly, U::movepi64_mask(t2_lo)));
	let t2x_hi = U::xor(U::slli_epi64::<1>(t2_hi), U::and(poly, U::movepi64_mask(t2_hi)));

	// Assemble the two output coefficients while still unreduced, then reduce each once.
	let term0_lo = U::xor(t0_lo, t2_lo);
	let term0_hi = U::xor(t0_hi, t2_hi);
	let term1_lo = U::xor(U::xor(U::xor(t1_lo, t0_lo), t2_lo), t2x_lo);
	let term1_hi = U::xor(U::xor(U::xor(t1_hi, t0_hi), t2_hi), t2x_hi);

	let z0 = reduce_pair(term0_lo, term0_hi);
	let z1 = reduce_pair(term1_lo, term1_hi);

	[z0, z1]
}

#[inline]
fn reduce_pair<U: Underlier + OpsClmul + PackedUnderlier<u64>>(prod_0: U, prod_1: U) -> U {
	// The reduction polynomial X^64 + X^4 + X^3 + X + 1 is represented as 0x1B
	// This is the bit representation of the lower-degree terms (X^4 + X^3 + X + 1)
	const POLY: u64 = 0x1B;
	let poly = <U as PackedUnderlier<u64>>::broadcast(POLY);

	// Step 2: First reduction - multiply high 64 bits by reduction polynomial
	// This effectively computes: high_bits * (X^4 + X^3 + X + 1) mod X^128
	let first_reduction_0 = U::clmulepi64::<0x01>(prod_0, poly);
	let first_reduction_1 = U::clmulepi64::<0x01>(prod_1, poly);

	// Extract the low 64 bits from the original products and first reductions
	let prod_lo = U::unpacklo_epi64(prod_0, prod_1);
	let first_reduction_lo = U::unpacklo_epi64(first_reduction_0, first_reduction_1);
	let result = U::xor(prod_lo, first_reduction_lo);

	// Step 3: Second reduction - handle overflow from the first reduction
	// The first reduction can produce results up to 67 bits, so we need another reduction
	let second_reduction_0 = U::clmulepi64::<0x01>(first_reduction_0, poly);
	let second_reduction_1 = U::clmulepi64::<0x01>(first_reduction_1, poly);

	// Extract low 64 bits of the second reduction
	let second_reduction_lo = U::unpacklo_epi64(second_reduction_0, second_reduction_1);

	// Final result: XOR all three components together
	U::xor(result, second_reduction_lo)
}

#[cfg(test)]
#[cfg(all(
	target_arch = "x86_64",
	target_feature = "pclmulqdq",
	target_feature = "sse2"
))]
mod tests {
	use std::arch::x86_64::__m128i;

	use proptest::prelude::*;

	use super::{mul_128b, mul_sliced_128b};

	// A packed GF(2^128) element is a `u128` with coefficient 0 in the low 64 bits and coefficient
	// 1 in the high 64 bits, matching `__m128i`'s lane layout.
	//
	// Packs two packed elements into sliced form across the two `__m128i` lanes:
	//   x[0] = [e0 coeff 0, e1 coeff 0], x[1] = [e0 coeff 1, e1 coeff 1].
	fn to_sliced(e0: u128, e1: u128) -> [__m128i; 2] {
		let coeff0 = (e0 as u64 as u128) | ((e1 as u64 as u128) << 64);
		let coeff1 = ((e0 >> 64) as u64 as u128) | (((e1 >> 64) as u64 as u128) << 64);
		unsafe {
			[
				std::mem::transmute::<u128, __m128i>(coeff0),
				std::mem::transmute::<u128, __m128i>(coeff1),
			]
		}
	}

	// Recovers the two packed elements from sliced form.
	fn from_sliced(z: [__m128i; 2]) -> (u128, u128) {
		let coeff0 = unsafe { std::mem::transmute::<__m128i, u128>(z[0]) };
		let coeff1 = unsafe { std::mem::transmute::<__m128i, u128>(z[1]) };
		let e0 = (coeff0 as u64 as u128) | ((coeff1 as u64 as u128) << 64);
		let e1 = ((coeff0 >> 64) as u64 as u128) | (((coeff1 >> 64) as u64 as u128) << 64);
		(e0, e1)
	}

	fn packed_mul(a: u128, b: u128) -> u128 {
		unsafe {
			std::mem::transmute::<__m128i, u128>(mul_128b::<__m128i>(
				std::mem::transmute::<u128, __m128i>(a),
				std::mem::transmute::<u128, __m128i>(b),
			))
		}
	}

	proptest! {
		// The sliced multiplication computes the same field product as the packed one, for both
		// lanes packed into the `__m128i`.
		#[test]
		fn sliced_matches_packed_128b(
			a0 in any::<u128>(),
			a1 in any::<u128>(),
			b0 in any::<u128>(),
			b1 in any::<u128>(),
		) {
			// Lane 0 multiplies a0 by b0; lane 1 multiplies a1 by b1.
			let z = mul_sliced_128b::<__m128i>(to_sliced(a0, a1), to_sliced(b0, b1));
			let (z0, z1) = from_sliced(z);

			prop_assert_eq!(z0, packed_mul(a0, b0));
			prop_assert_eq!(z1, packed_mul(a1, b1));
		}
	}
}
