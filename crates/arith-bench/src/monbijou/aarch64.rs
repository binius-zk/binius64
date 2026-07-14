// Copyright 2026 The Binius Developers
//! Direct aarch64 `PMULL`-accelerated multiplication for the Monbijou field and its extensions.
//!
//! Monbijou elements are represented with the `poly64x2_t` underlier (a SIMD vector, so the
//! multiply stays in NEON registers across call boundaries), matching the [`ghash::aarch64`]
//! module. The `PMULL` / `PMULL2` instructions (`vmull_p64` / `vmull_high_p64`) drive both the
//! carryless multiplies and the modular reduction: the high limb is folded down by carrylessly
//! multiplying it with the reduction polynomial's tail `0x1B` (`X^64 ≡ X^4 + X^3 + X + 1`), the
//! same strategy the generic [`OpsClmul`](crate::underlier::OpsClmul) path uses. Everything stays
//! in NEON registers, and the two 64-bit lanes are reduced in parallel.
//!
//! Each multiply is split into a widening half (`mul_wide*`, producing the unreduced products) and
//! a reduction (`reduce*`). Both halves are exposed because the reduction is F2-linear, so an inner
//! product may XOR-accumulate the widening products and reduce once at the very end instead of per
//! term — the same delayed-reduction structure as the `x86_64` module.
//!
//! [`ghash::aarch64`]: crate::ghash::aarch64

use core::arch::aarch64::*;

use crate::Underlier;

/// A widening (unreduced) base-field product held as two 64-bit lanes: `[0]` is the low limb and
/// `[1]` the high limb, with each lane carrying an independent GF(2^64) product. `[uint64x2_t; 2]`
/// is itself an [`Underlier`], so widening products XOR-accumulate lane- and limb-wise.
type Wide = [uint64x2_t; 2];

/// The reduction polynomial's tail: X^64 ≡ X^4 + X^3 + X + 1 = `0x1B`.
const POLY: u64 = 0x1B;

/// Widening (unreduced) 2-lane Monbijou multiply: the two 128-bit carryless products of the paired
/// lanes, transposed into `[low, high]` limbs so lane `i` holds `[x_i·y_i].lo` / `[x_i·y_i].hi`.
///
/// Because [`reduce`] is F2-linear, these limbs can be XOR-accumulated across many products and
/// reduced only once — an inner product of `n` terms costs one reduction instead of `n`.
#[inline]
pub fn mul_wide(x: poly64x2_t, y: poly64x2_t) -> Wide {
	unsafe {
		let p0 = vreinterpretq_u64_p128(vmull_p64(vgetq_lane_p64::<0>(x), vgetq_lane_p64::<0>(y)));
		let p1 = vreinterpretq_u64_p128(vmull_high_p64(x, y));
		// Transpose the two `[lo, hi]` products into `[[lo0, lo1], [hi0, hi1]]`.
		[vzip1q_u64(p0, p1), vzip2q_u64(p0, p1)]
	}
}

/// Reduce a widening product `[lo, hi]` to a packed base-field element, modulo
/// X^64 + X^4 + X^3 + X + 1, folding both lanes in parallel with `PMULL`.
///
/// The high limb holds the coefficients of X^64..X^127, so `hi·X^64 ≡ hi·0x1B` folds it down. That
/// carryless product is up to 69 bits, so its own overflow past X^63 is folded once more by the
/// same `·0x1B`. Both folds are `PMULL`s (`vmull_p64` / `vmull_high_p64`), and the low limbs of the
/// two lanes' products are gathered with `vzip1q`. This is an F2-linear map, so unreduced products
/// may be summed by XOR and reduced once at the end.
#[inline]
pub fn reduce([lo, hi]: Wide) -> poly64x2_t {
	unsafe {
		let poly = vreinterpretq_p64_u64(vdupq_n_u64(POLY));
		let hi = vreinterpretq_p64_u64(hi);
		// First fold: fr_i = hi_i · 0x1B (lane 0 via PMULL, lane 1 via PMULL2).
		let fr0 = vreinterpretq_u64_p128(vmull_p64(vgetq_lane_p64::<0>(hi), POLY));
		let fr1 = vreinterpretq_u64_p128(vmull_high_p64(hi, poly));
		let acc = veorq_u64(lo, vzip1q_u64(fr0, fr1));
		// Second fold: the bits of fr_i past X^63 (its high limb) folded by ·0x1B; the result fits
		// in the low 64 bits, so only those are gathered.
		let fr0 = vreinterpretq_p64_u64(fr0);
		let fr1 = vreinterpretq_p64_u64(fr1);
		let sr0 = vreinterpretq_u64_p128(vmull_p64(vgetq_lane_p64::<1>(fr0), POLY));
		let sr1 = vreinterpretq_u64_p128(vmull_p64(vgetq_lane_p64::<1>(fr1), POLY));
		vreinterpretq_p64_u64(veorq_u64(acc, vzip1q_u64(sr0, sr1)))
	}
}

/// Multiply an unreduced product by X: a one-bit left shift of each lane's 128-bit `[lo, hi]` value
/// (whose degree ≤ 126 leaves room, so no reduction is needed here).
#[inline]
fn mul_x_wide([lo, hi]: Wide) -> Wide {
	unsafe {
		let new_hi = veorq_u64(vshlq_n_u64::<1>(hi), vshrq_n_u64::<63>(lo));
		[vshlq_n_u64::<1>(lo), new_hi]
	}
}

/// Multiply a single unreduced product `[lo, hi]` (packed into one register, low limb in lane 0) by
/// X: a one-bit left shift of the whole 128-bit value, carrying lane 0's top bit into lane 1. The
/// product's degree ≤ 126 leaves room, so no reduction is needed here.
#[inline]
fn mul_x_packed(p: uint64x2_t) -> uint64x2_t {
	unsafe {
		// `[0, lo >> 63]`: the bit shifted out of lane 0 lands in lane 1.
		let carry = vextq_u64(vdupq_n_u64(0), vshrq_n_u64::<63>(p), 1);
		veorq_u64(vshlq_n_u64::<1>(p), carry)
	}
}

/// Multiplies two elements of the base field GF(2^64), the Monbijou field, for each of the two
/// packed 64-bit lanes independently. Composes [`mul_wide`] with [`reduce`].
#[inline]
pub fn mul(x: poly64x2_t, y: poly64x2_t) -> poly64x2_t {
	reduce(mul_wide(x, y))
}

/// Widening (unreduced) degree-2 Monbijou multiply in the *packed* representation: the three raw
/// base products `[t0, t1, t2] = [x0·y0, (x0+x1)·(y0+y1), x1·y1]`, each a 128-bit carryless product
/// occupying one register as `[lo, hi]`.
///
/// No combination or reduction happens here — those are F2-linear and deferred to [`reduce_128b`],
/// so an inner product XOR-accumulates the three products and reduces once.
#[inline]
pub fn mul_wide_128b(x: poly64x2_t, y: poly64x2_t) -> [uint64x2_t; 3] {
	unsafe {
		let x0 = vgetq_lane_p64::<0>(x);
		let x1 = vgetq_lane_p64::<1>(x);
		let y0 = vgetq_lane_p64::<0>(y);
		let y1 = vgetq_lane_p64::<1>(y);
		let t0 = vreinterpretq_u64_p128(vmull_p64(x0, y0));
		let t2 = vreinterpretq_u64_p128(vmull_high_p64(x, y)); // x1·y1
		let t1 = vreinterpretq_u64_p128(vmull_p64(x0 ^ x1, y0 ^ y1));
		[t0, t1, t2]
	}
}

/// Reduce the three raw products from [`mul_wide_128b`] into a packed GF(2^128) element.
///
/// The extension is `Y^2 = XY + 1`, so `coeff 0 = t0 + t2` and `coeff 1 = (t1 + t0 + t2) + X·t2`
/// (Karatsuba recovers the cross term `x0·y1 + x1·y0` as `t1 + t0 + t2`). The multiply-by-X on the
/// unreduced `t2` (degree ≤ 126) is a plain 128-bit left shift; all combining stays in NEON
/// registers. The two coefficient products are reduced together in the two lanes of one [`reduce`].
#[inline]
pub fn reduce_128b([t0, t1, t2]: [uint64x2_t; 3]) -> poly64x2_t {
	unsafe {
		let term0 = veorq_u64(t0, t2);
		let term1 = veorq_u64(veorq_u64(veorq_u64(t1, t0), t2), mul_x_packed(t2));
		// Transpose the two `[lo, hi]` coefficient products into `[[lo0, lo1], [hi0, hi1]]`.
		reduce([vzip1q_u64(term0, term1), vzip2q_u64(term0, term1)])
	}
}

/// Multiplies two elements of GF(2^128), the degree-2 extension of the Monbijou field, in the
/// *packed* representation (coefficient 0 in the low lane, coefficient 1 in the high lane).
/// Composes [`mul_wide_128b`] with [`reduce_128b`].
#[inline]
pub fn mul_128b(x: poly64x2_t, y: poly64x2_t) -> poly64x2_t {
	reduce_128b(mul_wide_128b(x, y))
}

/// Widening (unreduced) degree-2 Monbijou multiply in the *sliced* representation: the three raw
/// Karatsuba products `[t0, t1, t2] = [a0·b0, (a0+a1)·(b0+b1), a1·b1]` (each a `Wide` limb pair)
/// of two GF(2^128) elements `[poly64x2_t; 2]`.
///
/// Each element keeps its coefficients in separate registers, and the two 64-bit lanes carry two
/// independent elements. Deferred to [`reduce_sliced_128b`], so an inner product XOR-accumulates
/// the three products and reduces once.
#[inline]
pub fn mul_wide_sliced_128b(x: [poly64x2_t; 2], y: [poly64x2_t; 2]) -> [Wide; 3] {
	let [x0, x1] = x;
	let [y0, y1] = y;
	[
		mul_wide(x0, y0),
		mul_wide(Underlier::xor(x0, x1), Underlier::xor(y0, y1)),
		mul_wide(x1, y1),
	]
}

/// Reduce the three raw products from [`mul_wide_sliced_128b`] into a GF(2^128) element
/// `[poly64x2_t; 2]`.
///
/// The extension is `Y^2 = XY + 1`, so `coeff 0 = t0 + t2` and `coeff 1 = (t1 + t0 + t2) + X·t2`.
#[inline]
pub fn reduce_sliced_128b([t0, t1, t2]: [Wide; 3]) -> [poly64x2_t; 2] {
	let term0 = Underlier::xor(t0, t2);
	let term1 = Underlier::xor(Underlier::xor(Underlier::xor(t1, t0), t2), mul_x_wide(t2));
	[reduce(term0), reduce(term1)]
}

/// Multiplies two elements of GF(2^128), the degree-2 extension of the Monbijou field, in the
/// *sliced* representation. Same field product as [`mul_128b`], with the coefficients kept in
/// separate registers. Composes [`mul_wide_sliced_128b`] with [`reduce_sliced_128b`].
#[inline]
pub fn mul_sliced_128b(x: [poly64x2_t; 2], y: [poly64x2_t; 2]) -> [poly64x2_t; 2] {
	reduce_sliced_128b(mul_wide_sliced_128b(x, y))
}

/// Widening (unreduced) degree-3 Monbijou multiply in the *sliced* representation: the six raw
/// Karatsuba products `[m0, m1, m2, m01, m02, m12]` (each a `Wide` limb pair) of two GF(2^192)
/// elements `[poly64x2_t; 3]`.
///
/// Deferred to [`reduce_sliced_192b`], so an inner product XOR-accumulates the six products and
/// reduces once.
#[inline]
pub fn mul_wide_sliced_192b(x: [poly64x2_t; 3], y: [poly64x2_t; 3]) -> [Wide; 6] {
	let [x0, x1, x2] = x;
	let [y0, y1, y2] = y;
	[
		mul_wide(x0, y0),
		mul_wide(x1, y1),
		mul_wide(x2, y2),
		mul_wide(Underlier::xor(x0, x1), Underlier::xor(y0, y1)),
		mul_wide(Underlier::xor(x0, x2), Underlier::xor(y0, y2)),
		mul_wide(Underlier::xor(x1, x2), Underlier::xor(y1, y2)),
	]
}

/// Reduce the six raw products from [`mul_wide_sliced_192b`] into a GF(2^192) element
/// `[poly64x2_t; 3]`.
///
/// The tower is GF(2)\[X, Y\] / (X^64 + X^4 + X^3 + X + 1) / (Y^3 + X), so `Y^3 = X`; see
/// [`super::soft64::mul_192b`] for the algebra. Karatsuba gives the degree-≤4 coefficients
/// `c0..c4`; folding `Y^3 = X`, `Y^4 = X·Y` gives `z0 = c0 + X·c3`, `z1 = c1 + X·c4`, `z2 = c2`.
#[inline]
pub fn reduce_sliced_192b([m0, m1, m2, m01, m02, m12]: [Wide; 6]) -> [poly64x2_t; 3] {
	// Product coefficients (unreduced): c0 = m0, c4 = m2, and the Karatsuba cross terms
	//   c1 = m01 + m0 + m1,  c2 = m02 + m0 + m1 + m2,  c3 = m12 + m1 + m2.
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

/// Multiplies two elements of GF(2^192), the degree-3 extension of the Monbijou field, in the
/// *sliced* representation. Composes [`mul_wide_sliced_192b`] with [`reduce_sliced_192b`].
#[inline]
pub fn mul_sliced_192b(x: [poly64x2_t; 3], y: [poly64x2_t; 3]) -> [poly64x2_t; 3] {
	reduce_sliced_192b(mul_wide_sliced_192b(x, y))
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

		// Delayed reduction is valid: XOR-accumulating two widening products and reducing once equals
		// summing the two reduced products, for each widening variant.
		#[test]
		fn wide_deferred_reduction_base(
			a0 in any::<u64>(), a1 in any::<u64>(), b0 in any::<u64>(), b1 in any::<u64>(),
			c0 in any::<u64>(), c1 in any::<u64>(), d0 in any::<u64>(), d1 in any::<u64>(),
		) {
			let a = to_poly((a0 as u128) | ((a1 as u128) << 64));
			let b = to_poly((b0 as u128) | ((b1 as u128) << 64));
			let c = to_poly((c0 as u128) | ((c1 as u128) << 64));
			let d = to_poly((d0 as u128) | ((d1 as u128) << 64));
			let acc = <Wide as Underlier>::xor(mul_wide(a, b), mul_wide(c, d));
			let z = from_poly(reduce(acc));
			prop_assert_eq!(z as u64, soft64::mul(a0, b0) ^ soft64::mul(c0, d0));
			prop_assert_eq!((z >> 64) as u64, soft64::mul(a1, b1) ^ soft64::mul(c1, d1));
		}

		#[test]
		fn wide_deferred_reduction_packed_128b(
			a in any::<u128>(), b in any::<u128>(), c in any::<u128>(), d in any::<u128>(),
		) {
			let acc = <[uint64x2_t; 3] as Underlier>::xor(
				mul_wide_128b(to_poly(a), to_poly(b)),
				mul_wide_128b(to_poly(c), to_poly(d)),
			);
			prop_assert_eq!(from_poly(reduce_128b(acc)), soft64::mul_128b(a, b) ^ soft64::mul_128b(c, d));
		}

		#[test]
		fn wide_deferred_reduction_sliced_128b(
			a0 in any::<u128>(), a1 in any::<u128>(), b0 in any::<u128>(), b1 in any::<u128>(),
			c0 in any::<u128>(), c1 in any::<u128>(), d0 in any::<u128>(), d1 in any::<u128>(),
		) {
			let w1 = mul_wide_sliced_128b(to_sliced_128b(a0, a1), to_sliced_128b(b0, b1));
			let w2 = mul_wide_sliced_128b(to_sliced_128b(c0, c1), to_sliced_128b(d0, d1));
			let (z0, z1) = from_sliced_128b(reduce_sliced_128b(<[Wide; 3] as Underlier>::xor(w1, w2)));
			prop_assert_eq!(z0, soft64::mul_128b(a0, b0) ^ soft64::mul_128b(c0, d0));
			prop_assert_eq!(z1, soft64::mul_128b(a1, b1) ^ soft64::mul_128b(c1, d1));
		}
	}
}
