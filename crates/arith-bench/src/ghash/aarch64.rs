// Copyright 2026 The Binius Developers
//! Direct aarch64 `PMULL`-accelerated GHASH multiplication.
//!
//! GHASH elements and the limbs of an unreduced product are represented as `poly64x2_t` (a SIMD
//! vector type, so the arithmetic stays in NEON registers across call boundaries — a `p128`/`u128`
//! limb would be legalized into general-purpose register pairs whenever it is XORed). The `PMULL` /
//! `PMULL2` instructions (`vmull_p64` / `vmull_high_p64`) drive the carryless multiplies, pairwise
//! XORs are `vaddq_p64` and three-way XORs go through `xor3` (`EOR3` when the `SHA3` extension is
//! available). The multiply is split into two phases:
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
/// available, otherwise two `EOR`s.
#[cfg(target_feature = "sha3")]
#[inline]
fn xor3(a: poly64x2_t, b: poly64x2_t, c: poly64x2_t) -> poly64x2_t {
	unsafe {
		vreinterpretq_p64_u64(veor3q_u64(
			vreinterpretq_u64_p64(a),
			vreinterpretq_u64_p64(b),
			vreinterpretq_u64_p64(c),
		))
	}
}

/// Three-way 128-bit XOR.
#[cfg(not(target_feature = "sha3"))]
#[inline]
fn xor3(a: poly64x2_t, b: poly64x2_t, c: poly64x2_t) -> poly64x2_t {
	unsafe { vaddq_p64(vaddq_p64(a, b), c) }
}

/// Schoolbook widening multiply: four `PMULL`s, no reduction.
///
/// Returns `[t0, t1, t2]` (low, middle, high) as described in the [module docs](self).
#[inline]
pub fn mul_wide_schoolbook(x: poly64x2_t, y: poly64x2_t) -> [poly64x2_t; 3] {
	unsafe {
		// Cross term x0·y1 ⊕ x1·y0: swapping y's lanes makes PMULL (low×low) and PMULL2
		// (high×high) compute the opposite-index pairings without moving lanes to scalars.
		let y_swapped = vextq_p64::<1>(y, y);
		let cross_a = vreinterpretq_p64_p128(vmull_p64(
			vgetq_lane_p64::<0>(x),
			vgetq_lane_p64::<0>(y_swapped),
		));
		let cross_b = vreinterpretq_p64_p128(vmull_high_p64(x, y_swapped));
		let t1 = vaddq_p64(cross_a, cross_b);

		let t0 = vreinterpretq_p64_p128(vmull_p64(vgetq_lane_p64::<0>(x), vgetq_lane_p64::<0>(y)));
		let t2 = vreinterpretq_p64_p128(vmull_high_p64(x, y));

		[t0, t1, t2]
	}
}

/// Karatsuba widening multiply: three `PMULL`s, no reduction.
///
/// The middle term is `(x0 ⊕ x1)·(y0 ⊕ y1) ⊕ t0 ⊕ t2`, trading one carryless multiply for two
/// 128-bit XORs. Returns `[t0, t1, t2]` (low, middle, high).
#[inline]
pub fn mul_wide_karatsuba(x: poly64x2_t, y: poly64x2_t) -> [poly64x2_t; 3] {
	unsafe {
		let t0 = vreinterpretq_p64_p128(vmull_p64(vgetq_lane_p64::<0>(x), vgetq_lane_p64::<0>(y)));
		let t2 = vreinterpretq_p64_p128(vmull_high_p64(x, y));

		// Fold each operand's halves in-vector: lane 0 of `v ⊕ swap(v)` holds v0 ⊕ v1.
		let x_fold = vaddq_p64(x, vextq_p64::<1>(x, x));
		let y_fold = vaddq_p64(y, vextq_p64::<1>(y, y));
		let mid = vreinterpretq_p64_p128(vmull_p64(
			vgetq_lane_p64::<0>(x_fold),
			vgetq_lane_p64::<0>(y_fold),
		));
		let t1 = xor3(mid, t0, t2);

		[t0, t1, t2]
	}
}

/// Folds the 128-bit limb `b`, positioned at offset `X^64`, into the accumulator `a`, returning
/// the 128-bit value congruent to `a + b·X^64` modulo the GHASH polynomial.
#[inline]
fn fold(a: poly64x2_t, b: poly64x2_t) -> poly64x2_t {
	unsafe {
		// b.lo·X^64 stays within the limb: move the low lane into the high lane, zero the low.
		let zero = vreinterpretq_p64_p128(0u128);
		let shifted = vextq_p64::<1>(zero, b);
		// b.hi·X^128 ≡ b.hi·POLY since X^128 ≡ X^7 + X^2 + X + 1; the degree-7 POLY keeps this
		// within 128 bits, so no second fold is needed.
		let folded = vreinterpretq_p64_p128(vmull_high_p64(b, vreinterpretq_p64_p128(POLY)));
		xor3(a, shifted, folded)
	}
}

/// Reduces the wide product `[t0, t1, t2]` to a single GHASH element.
#[inline]
pub fn reduce([t0, t1, t2]: [poly64x2_t; 3]) -> poly64x2_t {
	let t1 = fold(t1, t2);
	fold(t0, t1)
}

/// Multiply a GHASH element by X.
///
/// This is equivalent to `mul(x, X)` but optimized: a full 128-bit left shift by one, folding in
/// the reduction polynomial if bit 127 overflowed (`X^128 ≡ X^7 + X^2 + X + 1 = 0x87`).
#[inline]
pub fn mul_x(x: poly64x2_t) -> poly64x2_t {
	unsafe {
		let v = vreinterpretq_u64_p64(x);

		// Full 128-bit left shift by one: shift each lane, then carry bit 63 of the low lane into
		// bit 0 of the high lane. `vextq_u64::<1>(zero, w)` is a shift of `w` left by 8 bytes.
		let zero = vreinterpretq_u64_p128(0);
		let carry = vextq_u64::<1>(zero, vshrq_n_u64::<63>(v));
		let shifted = veorq_u64(vshlq_n_u64::<1>(v), carry);

		// All-ones in both lanes iff bit 127 was set: duplicate the high lane, then broadcast its
		// sign bit with an arithmetic shift.
		let msb_mask = vreinterpretq_u64_s64(vshrq_n_s64::<63>(vreinterpretq_s64_u64(
			vdupq_laneq_u64::<1>(v),
		)));

		// Conditionally fold in the reduction polynomial, which sits in the low lane.
		let poly = vreinterpretq_u64_p128(0x87);
		vreinterpretq_p64_u64(veorq_u64(shifted, vandq_u64(poly, msb_mask)))
	}
}

/// Multiply an unreduced widening product (the three 128-bit limbs `[t0, t1, t2]` from a
/// `mul_wide`, at weights `X^0, X^64, X^128`) by X: a one-bit left shift of the represented
/// 256-bit polynomial.
///
/// Requires each limb's bit 127 to be zero, which holds for any XOR-accumulation of `mul_wide`
/// outputs since each limb is then a sum of carry-less products of two 64-bit halves. The shift
/// therefore never overflows the three limbs — no reduction is needed here, keeping it deferred to
/// the single [`reduce`]. Because both this shift and [`reduce`] are F2-linear, the multiply-by-X
/// may be applied to XOR-accumulated products and reduced once.
///
/// The result does *not* satisfy that invariant (bit 126 of a limb shifts into bit 127), so this is
/// not to be applied twice; reduce in between.
#[inline]
pub fn mul_x_wide([t0, t1, t2]: [poly64x2_t; 3]) -> [poly64x2_t; 3] {
	unsafe {
		let v0 = vreinterpretq_u64_p64(t0);
		let v1 = vreinterpretq_u64_p64(t1);
		let v2 = vreinterpretq_u64_p64(t2);

		// Shift each 64-bit lane up by one; `srl` holds the bit shifted out of the top of each
		// lane, which belongs 64 bit positions higher — i.e. in the lanes of the next limb, since
		// consecutive limbs overlap by 64 bits.
		let sll0 = vreinterpretq_p64_u64(vshlq_n_u64::<1>(v0));
		let sll1 = vreinterpretq_p64_u64(vshlq_n_u64::<1>(v1));
		let sll2 = vreinterpretq_p64_u64(vshlq_n_u64::<1>(v2));

		let srl0 = vreinterpretq_p64_u64(vshrq_n_u64::<63>(v0));
		let srl1 = vreinterpretq_p64_u64(vshrq_n_u64::<63>(v1));
		let srl2 = vreinterpretq_p64_u64(vshrq_n_u64::<63>(v2));

		// `t2` has no higher limb to carry into: its low lane's carry belongs in its own high lane,
		// where rotating the lanes puts it. The rotation also wraps the high lane's carry — bit 255
		// of the product — back into the low lane, which is harmless precisely because `t2`'s bit
		// 127, and hence that carry, is zero.
		let carry2 = vextq_p64::<1>(srl2, srl2);

		[sll0, vaddq_p64(sll1, srl0), xor3(sll2, srl1, carry2)]
	}
}

/// Square a GHASH element.
///
/// Squaring is a carry-less multiply of `x` by itself, so the cross term `x0·y1 ⊕ x1·y0` cancels:
/// only the two corner products remain, and the middle limb of the wide product is zero.
#[inline]
pub fn square(x: poly64x2_t) -> poly64x2_t {
	unsafe {
		let t0 = vreinterpretq_p64_p128(vmull_p64(vgetq_lane_p64::<0>(x), vgetq_lane_p64::<0>(x)));
		let t2 = vreinterpretq_p64_p128(vmull_high_p64(x, x));
		reduce([t0, vreinterpretq_p64_p128(0), t2])
	}
}

/// Multiply two GHASH elements using the schoolbook widening multiply.
#[inline]
pub fn mul_schoolbook(x: poly64x2_t, y: poly64x2_t) -> poly64x2_t {
	reduce(mul_wide_schoolbook(x, y))
}

/// Multiply two GHASH elements using the Karatsuba widening multiply.
#[inline]
pub fn mul_karatsuba(x: poly64x2_t, y: poly64x2_t) -> poly64x2_t {
	reduce(mul_wide_karatsuba(x, y))
}

#[cfg(test)]
mod tests {
	use proptest::prelude::*;

	use super::*;
	use crate::{Underlier, ghash::soft64};

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

		#[test]
		fn test_square_matches_soft64(a in any::<u128>()) {
			prop_assert_eq!(from_poly(square(to_poly(a))), soft64::square(a));
		}

		#[test]
		fn test_mul_x_matches_soft64(a in any::<u128>()) {
			prop_assert_eq!(from_poly(mul_x(to_poly(a))), soft64::mul_x(a));
		}

		// `mul_x_wide` on an unreduced product then `reduce` equals `mul_x` after `reduce`: both
		// paths compute X times the represented field element (the multiply-by-X commutes with the
		// F2-linear reduction).
		#[test]
		fn test_mul_x_wide_matches_soft64(a in any::<u128>(), b in any::<u128>()) {
			let wide = mul_wide_karatsuba(to_poly(a), to_poly(b));
			prop_assert_eq!(from_poly(reduce(mul_x_wide(wide))), soft64::mul_x(soft64::mul(a, b)));
		}

		// The reduction is F2-linear, so accumulating two unreduced products by XOR and reducing
		// once equals reducing each and summing.
		#[test]
		fn test_wide_deferred_reduction(
			a1 in any::<u128>(), b1 in any::<u128>(),
			a2 in any::<u128>(), b2 in any::<u128>(),
		) {
			let p = mul_wide_karatsuba(to_poly(a1), to_poly(b1));
			let q = mul_wide_schoolbook(to_poly(a2), to_poly(b2));
			let acc = reduce(<[poly64x2_t; 3]>::xor(p, q));
			prop_assert_eq!(from_poly(acc), soft64::mul(a1, b1) ^ soft64::mul(a2, b2));
		}
	}
}
