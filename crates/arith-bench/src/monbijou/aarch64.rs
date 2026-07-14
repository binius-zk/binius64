// Copyright 2026 The Binius Developers
//! Direct aarch64 `PMULL`-accelerated multiplication for the Monbijou field and its extensions.
//!
//! Monbijou elements are represented with the `poly64x2_t` underlier (a SIMD vector, so the
//! multiply stays in NEON registers across call boundaries), matching the [`ghash::aarch64`]
//! module. The `PMULL` / `PMULL2` instructions (`vmull_p64` / `vmull_high_p64`) drive the carryless
//! multiplies. In contrast to GHASH, the Monbijou reduction polynomial X^64 + X^4 + X^3 + X + 1 has
//! only low-degree tail terms (`0x1B`), so the reduction is done with plain shifts and XORs rather
//! than further carryless multiplies — the same algorithm as [`super::soft64`], vectorized across
//! the two 64-bit lanes.
//!
//! [`ghash::aarch64`]: crate::ghash::aarch64

use core::arch::aarch64::*;

use crate::Underlier;

/// A widening (unreduced) base-field product held as two 64-bit lanes: `.0` is the low limb and
/// `.1` the high limb, with each lane carrying an independent GF(2^64) product.
type Wide = (uint64x2_t, uint64x2_t);

/// Widening (unreduced) 2-lane Monbijou multiply: the two 128-bit carryless products of the paired
/// lanes, transposed into `(low, high)` limbs so lane `i` holds `[x_i·y_i].lo` / `[x_i·y_i].hi`.
///
/// Because `reduce` is F2-linear, these limbs can be XOR-accumulated across many products and
/// reduced only once — an inner product of `n` terms costs one reduction instead of `n`.
#[inline]
fn mul_wide(x: poly64x2_t, y: poly64x2_t) -> Wide {
	unsafe {
		let p0 = vreinterpretq_u64_p128(vmull_p64(vgetq_lane_p64::<0>(x), vgetq_lane_p64::<0>(y)));
		let p1 = vreinterpretq_u64_p128(vmull_high_p64(x, y));
		// Transpose the two `[lo, hi]` products into `([lo0, lo1], [hi0, hi1])`.
		(vzip1q_u64(p0, p1), vzip2q_u64(p0, p1))
	}
}

/// Reduce a widening product `(lo, hi)` to a packed base-field element, modulo
/// X^64 + X^4 + X^3 + X + 1, operating on both lanes in parallel.
///
/// The high limb holds the coefficients of X^64..X^127. Folding it down multiplies by X^64 ≡
/// `0x1B` (`= 1 + X + X^3 + X^4`); the left shifts drop the bits past X^63 and the matching right
/// shifts collect those (coefficients X^64..X^67), which fold in once more. This is an F2-linear
/// map, so unreduced products may be summed by XOR and reduced once at the end.
#[inline]
fn reduce((lo, hi): Wide) -> poly64x2_t {
	unsafe {
		// The bits of hi·0x1B that spill past X^63 (from the <<1, <<3, <<4 terms): X^64..X^67.
		let spill = veorq_u64(
			veorq_u64(vshrq_n_u64::<63>(hi), vshrq_n_u64::<61>(hi)),
			vshrq_n_u64::<60>(hi),
		);
		let lo = veorq_u64(
			lo,
			veorq_u64(
				veorq_u64(hi, vshlq_n_u64::<1>(hi)),
				veorq_u64(vshlq_n_u64::<3>(hi), vshlq_n_u64::<4>(hi)),
			),
		);
		// spill < 2^4, so folding it back in (spill·X^64 ≡ spill·0x1B) no longer spills past X^63.
		let folded_spill = veorq_u64(
			veorq_u64(spill, vshlq_n_u64::<1>(spill)),
			veorq_u64(vshlq_n_u64::<3>(spill), vshlq_n_u64::<4>(spill)),
		);
		vreinterpretq_p64_u64(veorq_u64(lo, folded_spill))
	}
}

/// Component-wise XOR of two widening products.
#[inline]
fn xor_wide(a: Wide, b: Wide) -> Wide {
	unsafe { (veorq_u64(a.0, b.0), veorq_u64(a.1, b.1)) }
}

/// Multiply an unreduced product by X: a one-bit left shift of each lane's 128-bit `[lo, hi]` value
/// (whose degree ≤ 126 leaves room, so no reduction is needed here).
#[inline]
fn mul_x_wide((lo, hi): Wide) -> Wide {
	unsafe {
		let new_hi = veorq_u64(vshlq_n_u64::<1>(hi), vshrq_n_u64::<63>(lo));
		(vshlq_n_u64::<1>(lo), new_hi)
	}
}

/// Multiplies two elements of the base field GF(2^64), the Monbijou field, for each of the two
/// packed 64-bit lanes independently.
#[inline]
pub fn mul(x: poly64x2_t, y: poly64x2_t) -> poly64x2_t {
	reduce(mul_wide(x, y))
}

/// Multiplies two elements of GF(2^128), the degree-2 extension of the Monbijou field, in the
/// *packed* representation (coefficient 0 in the low lane, coefficient 1 in the high lane).
///
/// This field is GF(2)\[X, Y\] / (X^64 + X^4 + X^3 + X + 1) / (Y^2 + XY + 1), so `Y^2 = XY + 1`. A
/// Karatsuba split gives the three raw base products `t0 = x0·y0`, `t2 = x1·y1`, and `t1 =
/// (x0+x1)·(y0+y1)`; the reduced coefficients are `t0 + t2` and `(t1 + t0 + t2) + X·t2`. The two
/// coefficient products are reduced together in the two lanes of one `reduce`.
#[inline]
pub fn mul_128b(x: poly64x2_t, y: poly64x2_t) -> poly64x2_t {
	unsafe {
		let x0 = vgetq_lane_p64::<0>(x);
		let x1 = vgetq_lane_p64::<1>(x);
		let y0 = vgetq_lane_p64::<0>(y);
		let y1 = vgetq_lane_p64::<1>(y);

		let t0 = vmull_p64(x0, y0);
		let t2 = vmull_high_p64(x, y); // x1·y1
		let t1 = vmull_p64(x0 ^ x1, y0 ^ y1);

		// coeff 0 = t0 + t2; coeff 1 = (t1 + t0 + t2) + X·t2 (Karatsuba recovers the cross term
		// x0·y1 + x1·y0 as t1 + t0 + t2). The multiply-by-X on the unreduced t2 (degree ≤ 126) is a
		// plain 128-bit left shift.
		let term0 = t0 ^ t2;
		let term1 = t1 ^ t0 ^ t2 ^ (t2 << 1);

		// Reduce coeff 0 in lane 0 and coeff 1 in lane 1 with a single 2-lane reduction.
		let term0 = vreinterpretq_u64_p128(term0);
		let term1 = vreinterpretq_u64_p128(term1);
		reduce((vzip1q_u64(term0, term1), vzip2q_u64(term0, term1)))
	}
}

/// Multiplies two elements of GF(2^128), the degree-2 extension of the Monbijou field, in the
/// *sliced* representation: each element is `[poly64x2_t; 2]` with coefficient `i` in index `i`,
/// and the two 64-bit lanes carry two independent elements processed in parallel.
///
/// Same field product as [`mul_128b`], with the coefficients kept in separate registers. Composes a
/// Karatsuba widening multiply with the deferred `reduce`.
#[inline]
pub fn mul_sliced_128b(x: [poly64x2_t; 2], y: [poly64x2_t; 2]) -> [poly64x2_t; 2] {
	let [x0, x1] = x;
	let [y0, y1] = y;
	let t0 = mul_wide(x0, y0);
	let t1 = mul_wide(Underlier::xor(x0, x1), Underlier::xor(y0, y1));
	let t2 = mul_wide(x1, y1);

	let term0 = xor_wide(t0, t2);
	let term1 = xor_wide(xor_wide(xor_wide(t1, t0), t2), mul_x_wide(t2));
	[reduce(term0), reduce(term1)]
}

/// Multiplies two elements of GF(2^192), the degree-3 extension of the Monbijou field, in the
/// *sliced* representation: each element is `[poly64x2_t; 3]` with coefficient `i` in index `i`,
/// and the two 64-bit lanes carry two independent elements processed in parallel.
///
/// The tower is GF(2)\[X, Y\] / (X^64 + X^4 + X^3 + X + 1) / (Y^3 + X), so `Y^3 = X`; see
/// [`super::soft64::mul_192b`] for the algebra. Karatsuba gives the degree-≤4 coefficients
/// `c0..c4`; folding `Y^3 = X`, `Y^4 = X·Y` gives `z0 = c0 + X·c3`, `z1 = c1 + X·c4`, `z2 = c2`.
#[inline]
pub fn mul_sliced_192b(x: [poly64x2_t; 3], y: [poly64x2_t; 3]) -> [poly64x2_t; 3] {
	let [x0, x1, x2] = x;
	let [y0, y1, y2] = y;
	let m0 = mul_wide(x0, y0);
	let m1 = mul_wide(x1, y1);
	let m2 = mul_wide(x2, y2);
	let m01 = mul_wide(Underlier::xor(x0, x1), Underlier::xor(y0, y1));
	let m02 = mul_wide(Underlier::xor(x0, x2), Underlier::xor(y0, y2));
	let m12 = mul_wide(Underlier::xor(x1, x2), Underlier::xor(y1, y2));

	// Product coefficients (unreduced): c0 = m0, c4 = m2, and the Karatsuba cross terms
	//   c1 = m01 + m0 + m1,  c2 = m02 + m0 + m1 + m2,  c3 = m12 + m1 + m2.
	let c0 = m0;
	let c1 = xor_wide(xor_wide(m01, m0), m1);
	let c2 = xor_wide(xor_wide(xor_wide(m02, m0), m1), m2);
	let c3 = xor_wide(xor_wide(m12, m1), m2);
	let c4 = m2;

	let z0 = reduce(xor_wide(c0, mul_x_wide(c3)));
	let z1 = reduce(xor_wide(c1, mul_x_wide(c4)));
	let z2 = reduce(c2);
	[z0, z1, z2]
}

#[cfg(test)]
mod tests {
	use proptest::prelude::*;

	use super::*;
	use crate::monbijou::soft64;

	fn to_poly(x: u128) -> poly64x2_t {
		unsafe { vreinterpretq_p64_p128(x) }
	}

	fn from_poly(x: poly64x2_t) -> u128 {
		unsafe { vreinterpretq_p128_p64(x) }
	}

	// Packs two GF(2^128) elements into sliced form across the two lanes: index i holds
	// [e0 coeff i, e1 coeff i].
	fn to_sliced_128b(e0: u128, e1: u128) -> [poly64x2_t; 2] {
		let coeff0 = (e0 as u64 as u128) | ((e1 as u64 as u128) << 64);
		let coeff1 = ((e0 >> 64) as u64 as u128) | (((e1 >> 64) as u64 as u128) << 64);
		[to_poly(coeff0), to_poly(coeff1)]
	}

	fn from_sliced_128b(z: [poly64x2_t; 2]) -> (u128, u128) {
		let coeff0 = from_poly(z[0]);
		let coeff1 = from_poly(z[1]);
		let e0 = (coeff0 as u64 as u128) | ((coeff1 as u64 as u128) << 64);
		let e1 = ((coeff0 >> 64) as u64 as u128) | (((coeff1 >> 64) as u64 as u128) << 64);
		(e0, e1)
	}

	proptest! {
		// The 2-lane base multiply agrees with the soft64 reference in both lanes.
		#[test]
		fn base_mul_matches_soft64(x0 in any::<u64>(), x1 in any::<u64>(), y0 in any::<u64>(), y1 in any::<u64>()) {
			let x = to_poly((x0 as u128) | ((x1 as u128) << 64));
			let y = to_poly((y0 as u128) | ((y1 as u128) << 64));
			let z = from_poly(mul(x, y));
			prop_assert_eq!(z as u64, soft64::mul(x0, y0));
			prop_assert_eq!((z >> 64) as u64, soft64::mul(x1, y1));
		}

		// The packed 128b multiply agrees with the soft64 reference.
		#[test]
		fn packed_128b_matches_soft64(a in any::<u128>(), b in any::<u128>()) {
			let z = from_poly(mul_128b(to_poly(a), to_poly(b)));
			prop_assert_eq!(z, soft64::mul_128b(a, b));
		}

		// The sliced 128b multiply agrees with the soft64 reference, for both lanes.
		#[test]
		fn sliced_128b_matches_soft64(a0 in any::<u128>(), a1 in any::<u128>(), b0 in any::<u128>(), b1 in any::<u128>()) {
			let (z0, z1) = from_sliced_128b(mul_sliced_128b(to_sliced_128b(a0, a1), to_sliced_128b(b0, b1)));
			prop_assert_eq!(z0, soft64::mul_128b(a0, b0));
			prop_assert_eq!(z1, soft64::mul_128b(a1, b1));
		}

		// The sliced 192b multiply agrees with the soft64 reference, for both lanes.
		#[test]
		fn sliced_192b_matches_soft64(xa in any::<[u64; 3]>(), xb in any::<[u64; 3]>(), ya in any::<[u64; 3]>(), yb in any::<[u64; 3]>()) {
			// Coefficient i is [lane0 = e0[i], lane1 = e1[i]] packed into a poly64x2_t.
			let to_sliced = |e0: [u64; 3], e1: [u64; 3]| -> [poly64x2_t; 3] {
				std::array::from_fn(|i| to_poly((e0[i] as u128) | ((e1[i] as u128) << 64)))
			};
			let lane = |z: [poly64x2_t; 3], lane: u32| -> [u64; 3] {
				std::array::from_fn(|i| (from_poly(z[i]) >> (64 * lane)) as u64)
			};
			let z = mul_sliced_192b(to_sliced(xa, xb), to_sliced(ya, yb));
			prop_assert_eq!(lane(z, 0), soft64::mul_192b(xa, ya));
			prop_assert_eq!(lane(z, 1), soft64::mul_192b(xb, yb));
		}
	}
}
