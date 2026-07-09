// Copyright 2026 The Binius Developers
//! Portable (non-CLMUL) multiplication for the Monbijou field and its degree-2 extension.
//!
//! These are software implementations of the algorithms in the [parent module](super), built on
//! the portable carry-less multiply primitives in `crate::arch::portable64` rather than on CLMUL or
//! SIMD intrinsics. They operate on scalar `u64`/`u128` values, one field element at a time.

use crate::{
	Underlier,
	arch::portable64::{bmul64, rev64},
};

/// Widening (unreduced) Monbijou multiply: the 128-bit carry-less product of two 64-bit polynomials
/// as `[low, high]` 64-bit limbs, without the modular reduction.
///
/// `bmul64` gives the low limb directly. The high limb comes from multiplying the bit-reversed
/// operands and reversing the result: reversing a product of two degree-`<64` polynomials leaves it
/// shifted by one bit, which the final shift removes. Because [`reduce`] is F2-linear, these limbs
/// can be XOR-accumulated across many products and reduced only once — an inner product of `n`
/// terms costs one reduction instead of `n`.
#[inline]
pub fn mul_wide(x: u64, y: u64) -> [u64; 2] {
	let lo = bmul64(x, y);
	let hi = rev64(bmul64(rev64(x), rev64(y))) >> 1;
	[lo, hi]
}

/// Reduces a 128-bit carry-less product, given as `[low, high]` limbs, modulo the Monbijou
/// polynomial X^64 + X^4 + X^3 + X + 1.
///
/// The high limb holds the coefficients of X^64..X^127. Folding it down is a carry-less multiply by
/// X^64 ≡ `0x1B` (`= 1 + X + X^3 + X^4`). The left shifts drop the bits that spill past X^63; the
/// matching right shifts collect those as the coefficients of X^64..X^67, which fold in once more.
/// This is an F2-linear map, so unreduced products may be summed by XOR and reduced once at the
/// end.
#[inline]
pub const fn reduce([lo, hi]: [u64; 2]) -> u64 {
	// The bits of hi·0x1B that spill past X^63 (from the <<1, <<3, <<4 terms): coefficients
	// X^64..X^67.
	let spill = (hi >> 63) ^ (hi >> 61) ^ (hi >> 60);
	let lo = lo ^ hi ^ (hi << 1) ^ (hi << 3) ^ (hi << 4);
	// spill < 2^4, so folding it back in (spill·X^64 ≡ spill·0x1B) no longer spills past X^63.
	lo ^ spill ^ (spill << 1) ^ (spill << 3) ^ (spill << 4)
}

/// Multiplies two elements of the base field GF(2^64), the Monbijou field.
///
/// Composes the widening multiply [`mul_wide`] with the modular [`reduce`]; both are inlined.
#[inline]
pub fn mul(x: u64, y: u64) -> u64 {
	reduce(mul_wide(x, y))
}

/// Multiplies two elements of GF(2^128), the degree-2 extension of the Monbijou field.
///
/// The element `a0 + a1·Y` is packed into a `u128` with `a0` in the low 64 bits and `a1` in the
/// high 64 bits, matching [`super::mul_128b_clmul`]. The extension is `Y^2 = XY + 1`, so for
/// `a = a0 + a1·Y` and `b = b0 + b1·Y`:
///
/// ```text
/// coeff 0 = a0·b0 + a1·b1
/// coeff 1 = a0·b1 + a1·b0 + X·(a1·b1)
/// ```
///
/// with Karatsuba recovering the cross term `a0·b1 + a1·b0` as `t1 + t0 + t2`. The three base-field
/// products are kept unreduced and the multiply-by-X is a plain shift of the unreduced `a1·b1`, so
/// only the two output coefficients need reducing — two reductions instead of three, matching the
/// CLMUL path's deferred reduction.
#[inline]
pub fn mul_128b(x: u128, y: u128) -> u128 {
	let (x0, x1) = (x as u64, (x >> 64) as u64);
	let (y0, y1) = (y as u64, (y >> 64) as u64);

	// Unreduced 128-bit carry-less products, as [low, high] limbs.
	let [t0l, t0h] = mul_wide(x0, y0); // a0·b0
	let [t2l, t2h] = mul_wide(x1, y1); // a1·b1
	let [t1l, t1h] = mul_wide(x0 ^ x1, y0 ^ y1); // (a0 + a1)·(b0 + b1)

	// coeff 0 = a0·b0 + a1·b1; reduction is linear, so summing before reducing gives one reduction.
	let z0 = reduce([t0l ^ t2l, t0h ^ t2h]);

	// coeff 1 = (a0·b1 + a1·b0) + X·(a1·b1). Karatsuba gives the cross term as t1 + t0 + t2;
	// X·(a1·b1) is the unreduced a1·b1 shifted up one bit (its degree is ≤ 126, so the shift stays
	// in 128 bits).
	let cross_lo = t1l ^ t0l ^ t2l;
	let cross_hi = t1h ^ t0h ^ t2h;
	let xt2_lo = t2l << 1;
	let xt2_hi = (t2h << 1) | (t2l >> 63);
	let z1 = reduce([cross_lo ^ xt2_lo, cross_hi ^ xt2_hi]);

	(z0 as u128) | ((z1 as u128) << 64)
}

/// Widening (unreduced) degree-3 Monbijou multiply: the six raw base-field Karatsuba products
/// `[m0, m1, m2, m01, m02, m12]` of two GF(2^192) elements `[a0, a1, a2]`, `[b0, b1, b2]`, where
/// `m0 = a0·b0`, `m1 = a1·b1`, `m2 = a2·b2`, and the sum products `m01 = (a0+a1)(b0+b1)`,
/// `m02 = (a0+a2)(b0+b2)`, `m12 = (a1+a2)(b1+b2)`.
///
/// No combination, scaling, or reduction is done here — those are all F2-linear and deferred to
/// [`reduce_192b`], so an inner product XOR-accumulates the six products and reduces once.
#[inline]
pub fn mul_wide_192b(a: [u64; 3], b: [u64; 3]) -> [[u64; 2]; 6] {
	let [a0, a1, a2] = a;
	let [b0, b1, b2] = b;
	[
		mul_wide(a0, b0),
		mul_wide(a1, b1),
		mul_wide(a2, b2),
		mul_wide(a0 ^ a1, b0 ^ b1),
		mul_wide(a0 ^ a2, b0 ^ b2),
		mul_wide(a1 ^ a2, b1 ^ b2),
	]
}

/// Reduce the six raw products from [`mul_wide_192b`] into a GF(2^192) element.
///
/// The tower is GF(2)\[X, Y\] / (X^64 + X^4 + X^3 + X + 1) / (Y^3 + X), so `Y^3 = X`. Karatsuba
/// gives the degree-≤4 product coefficients `c0..c4` (`c0 = m0`, `c1 = m01+m0+m1`, `c2 =
/// m02+m0+m1+m2`, `c3 = m12+m1+m2`, `c4 = m2`); folding `Y^3 = X`, `Y^4 = X·Y` gives `z0 = c0 +
/// X·c3`, `z1 = c1 + X·c4`, `z2 = c2`. The multiply-by-X (a 1-bit left shift of the unreduced
/// `[low, high]` product, whose degree ≤ 126 leaves room) is done here, on the unreduced values, so
/// only the three output coefficients are reduced.
#[inline]
pub fn reduce_192b([m0, m1, m2, m01, m02, m12]: [[u64; 2]; 6]) -> [u64; 3] {
	// `[u64; 2]` is itself an `Underlier` (blanket array impl), so `Underlier::xor` adds the
	// unreduced product pairs component-wise.
	let c0 = m0;
	let c1 = Underlier::xor(Underlier::xor(m01, m0), m1);
	let c2 = Underlier::xor(Underlier::xor(Underlier::xor(m02, m0), m1), m2);
	let c3 = Underlier::xor(Underlier::xor(m12, m1), m2);
	let c4 = m2;

	let z0 = reduce(Underlier::xor(c0, mul_x_wide(c3)));
	let z1 = reduce(Underlier::xor(c1, mul_x_wide(c4)));
	let z2 = reduce(c2);
	[z0, z1, z2]
}

/// Multiply an unreduced base-field product pair `[low, high]` by X: a one-bit left shift of the
/// 128-bit value (whose degree ≤ 126 leaves room, so no reduction is needed here).
#[inline]
const fn mul_x_wide([lo, hi]: [u64; 2]) -> [u64; 2] {
	[lo << 1, (hi << 1) | (lo >> 63)]
}

/// Multiplies two elements of GF(2^192), the degree-3 extension of the Monbijou field.
///
/// Composes [`mul_wide_192b`] with [`reduce_192b`]; both are inlined.
#[inline]
pub fn mul_192b(a: [u64; 3], b: [u64; 3]) -> [u64; 3] {
	reduce_192b(mul_wide_192b(a, b))
}

#[cfg(test)]
mod tests {
	use proptest::prelude::*;

	use super::{mul, mul_128b, mul_192b, mul_wide, reduce};
	use crate::{
		monbijou::MONBIJOU_128B_ONE,
		test_utils::multiplication_tests::{
			test_mul_associative, test_mul_commutative, test_mul_distributive,
		},
	};

	proptest! {
		// The 64-bit base field: portable field-axiom checks.
		#[test]
		fn base_mul_commutative(a in any::<u64>(), b in any::<u64>()) {
			prop_assert_eq!(mul(a, b), mul(b, a));
		}

		#[test]
		fn base_mul_identity(a in any::<u64>()) {
			prop_assert_eq!(mul(a, 0x01), a);
		}

		#[test]
		fn base_mul_distributive(a in any::<u64>(), b in any::<u64>(), c in any::<u64>()) {
			prop_assert_eq!(mul(a, b ^ c), mul(a, b) ^ mul(a, c));
		}

		// The reduction is F2-linear, so accumulating two unreduced products by XOR and reducing
		// once equals reducing each and summing.
		#[test]
		fn base_wide_mul_deferred_reduction(
			a1 in any::<u64>(), b1 in any::<u64>(),
			a2 in any::<u64>(), b2 in any::<u64>(),
		) {
			let [p0, p1] = mul_wide(a1, b1);
			let [q0, q1] = mul_wide(a2, b2);
			prop_assert_eq!(reduce([p0 ^ q0, p1 ^ q1]), mul(a1, b1) ^ mul(a2, b2));
		}

		// The 128-bit extension field: reuse the generic field-axiom helpers (u128 is an Underlier).
		#[test]
		fn ext_mul_commutative(a in any::<u128>(), b in any::<u128>()) {
			test_mul_commutative(a, b, mul_128b, "Monbijou 128b");
		}

		#[test]
		fn ext_mul_associative(a in any::<u128>(), b in any::<u128>(), c in any::<u128>()) {
			test_mul_associative(a, b, c, mul_128b, "Monbijou 128b");
		}

		#[test]
		fn ext_mul_distributive(a in any::<u128>(), b in any::<u128>(), c in any::<u128>()) {
			test_mul_distributive(a, b, c, mul_128b, "Monbijou 128b");
		}

		#[test]
		fn ext_mul_identity(a in any::<u128>()) {
			prop_assert_eq!(mul_128b(a, MONBIJOU_128B_ONE), a);
		}

		// The degree-3 extension GF(2^192): field axioms ([u64; 3] is an Underlier via the blanket
		// array impl, so the generic helpers apply).
		#[test]
		fn ext192_mul_commutative(a in any::<[u64; 3]>(), b in any::<[u64; 3]>()) {
			test_mul_commutative(a, b, mul_192b, "Monbijou 192b");
		}

		#[test]
		fn ext192_mul_associative(
			a in any::<[u64; 3]>(), b in any::<[u64; 3]>(), c in any::<[u64; 3]>(),
		) {
			test_mul_associative(a, b, c, mul_192b, "Monbijou 192b");
		}

		#[test]
		fn ext192_mul_distributive(
			a in any::<[u64; 3]>(), b in any::<[u64; 3]>(), c in any::<[u64; 3]>(),
		) {
			test_mul_distributive(a, b, c, mul_192b, "Monbijou 192b");
		}

		#[test]
		fn ext192_mul_identity(a in any::<[u64; 3]>()) {
			prop_assert_eq!(mul_192b(a, [0x01, 0, 0]), a);
		}
	}

	// Equivalence to the trusted CLMUL implementations, on hardware that has them.
	#[cfg(all(
		target_arch = "x86_64",
		target_feature = "pclmulqdq",
		target_feature = "sse2"
	))]
	mod clmul_equivalence {
		use std::arch::x86_64::__m128i;

		use proptest::prelude::*;

		use super::super::{mul, mul_128b};
		use crate::monbijou::{mul_128b_clmul, mul_clmul};

		proptest! {
			// The base-field soft64 mul matches `mul_clmul` (compared in lane 0 of an `__m128i`).
			#[test]
			fn base_matches_clmul(x in any::<u64>(), y in any::<u64>()) {
				let a = unsafe { std::mem::transmute::<u128, __m128i>(x as u128) };
				let b = unsafe { std::mem::transmute::<u128, __m128i>(y as u128) };
				let clmul = unsafe { std::mem::transmute::<__m128i, u128>(mul_clmul::<__m128i>(a, b)) };
				prop_assert_eq!(mul(x, y), clmul as u64);
			}

			// The extension soft64 mul matches the packed `mul_128b_clmul`.
			#[test]
			fn ext_matches_clmul(a in any::<u128>(), b in any::<u128>()) {
				let clmul = unsafe {
					std::mem::transmute::<__m128i, u128>(mul_128b_clmul::<__m128i>(
						std::mem::transmute::<u128, __m128i>(a),
						std::mem::transmute::<u128, __m128i>(b),
					))
				};
				prop_assert_eq!(mul_128b(a, b), clmul);
			}
		}
	}
}
