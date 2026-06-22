// Copyright 2026 The Binius Developers
//! Direct aarch64 `PMULL`-accelerated GHASH multiplication.
//!
//! GHASH elements are represented as `poly64x2_t` (a SIMD vector type, so the multiply stays in
//! NEON registers across call boundaries). The `PMULL` / `PMULL2` instructions (`vmull_p64` /
//! `vmull_high_p64`) drive the carryless multiplies, pairwise 128-bit XORs use `vaddq_p128`, and
//! three-way XORs go through [`xor3`] (`EOR3` when the `SHA3` extension is available). The
//! multiply is split into two phases:
//!
//! * `mul_wide`, which produces the unreduced product as three 128-bit limbs `[t0, t1, t2]`: `t0 =
//!   x0·y0` (low), `t2 = x1·y1` (high), and `t1` the middle (cross) term sitting at offset `X^64`,
//!   so the full product is `t0 + t1·X^64 + t2·X^128`.
//! * [`reduce`], which folds those three limbs back into a single GHASH element.
//!
//! Two `mul_wide` variants are provided so their cost can be compared: a schoolbook form using
//! four `PMULL`s and a Karatsuba form using three.

use core::arch::aarch64::*;

/// Low part of the reduction polynomial X^128 + X^7 + X^2 + X + 1, held in the high 64-bit lane
/// so it can feed `vmull_high_p64` directly.
const POLY: u128 = 0x87 << 64;

/// Three-way 128-bit XOR. Uses a single `EOR3` instruction when the `SHA3` extension is
/// available, otherwise two `vaddq_p128`s.
#[cfg(target_feature = "sha3")]
#[inline]
fn xor3(a: u128, b: u128, c: u128) -> u128 {
	unsafe {
		vreinterpretq_p128_u64(veor3q_u64(
			vreinterpretq_u64_p128(a),
			vreinterpretq_u64_p128(b),
			vreinterpretq_u64_p128(c),
		))
	}
}

#[cfg(not(target_feature = "sha3"))]
#[inline]
fn xor3(a: u128, b: u128, c: u128) -> u128 {
	unsafe { vaddq_p128(vaddq_p128(a, b), c) }
}

/// Schoolbook widening multiply: four `PMULL`s, no reduction.
///
/// Returns `[t0, t1, t2]` (low, middle, high) as described in the [module docs](self).
#[inline]
pub fn mul_wide_schoolbook(x: poly64x2_t, y: poly64x2_t) -> [u128; 3] {
	unsafe {
		// Cross term x0·y1 ⊕ x1·y0: swapping y's lanes makes PMULL (low×low) and PMULL2
		// (high×high) compute the opposite-index pairings without moving lanes to scalars.
		let y_swapped = vextq_p64::<1>(y, y);
		let cross_a = vmull_p64(vgetq_lane_p64::<0>(x), vgetq_lane_p64::<0>(y_swapped));
		let cross_b = vmull_high_p64(x, y_swapped);
		let t1 = vaddq_p128(cross_a, cross_b);

		let t0 = vmull_p64(vgetq_lane_p64::<0>(x), vgetq_lane_p64::<0>(y));
		let t2 = vmull_high_p64(x, y);

		[t0, t1, t2]
	}
}

/// Karatsuba widening multiply: three `PMULL`s, no reduction.
///
/// The middle term is `(x0 ⊕ x1)·(y0 ⊕ y1) ⊕ t0 ⊕ t2`, trading one carryless multiply for two
/// 128-bit XORs. Returns `[t0, t1, t2]` (low, middle, high).
#[inline]
pub fn mul_wide_karatsuba(x: poly64x2_t, y: poly64x2_t) -> [u128; 3] {
	unsafe {
		let t0 = vmull_p64(vgetq_lane_p64::<0>(x), vgetq_lane_p64::<0>(y));
		let t2 = vmull_high_p64(x, y);

		// Fold each operand's halves in-vector: lane 0 of `v ⊕ swap(v)` holds v0 ⊕ v1.
		let x_fold = vreinterpretq_p64_p128(vaddq_p128(
			vreinterpretq_p128_p64(x),
			vreinterpretq_p128_p64(vextq_p64::<1>(x, x)),
		));
		let y_fold = vreinterpretq_p64_p128(vaddq_p128(
			vreinterpretq_p128_p64(y),
			vreinterpretq_p128_p64(vextq_p64::<1>(y, y)),
		));
		let mid = vmull_p64(vgetq_lane_p64::<0>(x_fold), vgetq_lane_p64::<0>(y_fold));
		let t1 = xor3(mid, t0, t2);

		[t0, t1, t2]
	}
}

/// Folds the 128-bit limb `b`, positioned at offset `X^64`, into the accumulator `a`, returning
/// the 128-bit value congruent to `a + b·X^64` modulo the GHASH polynomial.
#[inline]
fn fold(a: u128, b: u128) -> u128 {
	unsafe {
		let bv = vreinterpretq_p64_p128(b);
		// b.lo·X^64 stays within the limb: move the low lane into the high lane, zero the low.
		let zero = vreinterpretq_p64_p128(0u128);
		let shifted = vreinterpretq_p128_p64(vextq_p64::<1>(zero, bv));
		// b.hi·X^128 ≡ b.hi·POLY since X^128 ≡ X^7 + X^2 + X + 1; the degree-7 POLY keeps this
		// within 128 bits, so no second fold is needed.
		let folded = vmull_high_p64(bv, vreinterpretq_p64_p128(POLY));
		xor3(a, shifted, folded)
	}
}

/// Reduces the wide product `[t0, t1, t2]` to a single GHASH element.
#[inline]
pub fn reduce([t0, t1, t2]: [u128; 3]) -> u128 {
	let t1 = fold(t1, t2);
	fold(t0, t1)
}

/// Multiply two GHASH elements using the schoolbook widening multiply.
#[inline]
pub fn mul_schoolbook(x: poly64x2_t, y: poly64x2_t) -> poly64x2_t {
	unsafe { vreinterpretq_p64_p128(reduce(mul_wide_schoolbook(x, y))) }
}

/// Multiply two GHASH elements using the Karatsuba widening multiply.
#[inline]
pub fn mul_karatsuba(x: poly64x2_t, y: poly64x2_t) -> poly64x2_t {
	unsafe { vreinterpretq_p64_p128(reduce(mul_wide_karatsuba(x, y))) }
}

#[cfg(test)]
mod tests {
	use proptest::prelude::*;

	use super::*;
	use crate::ghash::soft64;

	fn to_poly(x: u128) -> poly64x2_t {
		unsafe { vreinterpretq_p64_p128(x) }
	}

	fn from_poly(x: poly64x2_t) -> u128 {
		unsafe { vreinterpretq_p128_p64(x) }
	}

	proptest! {
		#[test]
		fn test_mul_schoolbook_matches_soft64(a in any::<u128>(), b in any::<u128>()) {
			prop_assert_eq!(from_poly(mul_schoolbook(to_poly(a), to_poly(b))), soft64::mul(a, b));
		}

		#[test]
		fn test_mul_karatsuba_matches_soft64(a in any::<u128>(), b in any::<u128>()) {
			prop_assert_eq!(from_poly(mul_karatsuba(to_poly(a), to_poly(b))), soft64::mul(a, b));
		}
	}
}
