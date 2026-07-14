// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers
//! CLMUL-based multiplication for the Monbijou field and its degree-2 extension.
//!
//! These implementations use carry-less multiplication (CLMUL) CPU instructions for efficient
//! field multiplication on modern x86_64 processors. The algorithm is optimized for SIMD
//! parallelism, processing multiple field elements simultaneously when using vector types like
//! __m128i or __m256i.

use crate::{PackedUnderlier, Underlier, underlier::OpsClmul};

/// Widening (unreduced) Monbijou multiply: the two 128-bit carry-less products `[prod_0, prod_1]`,
/// without the modular reduction.
///
/// The `0x00`/`0x11` immediates select the low·low and high·high halves of each 128-bit SIMD lane,
/// i.e. the two independent base elements packed per lane. Because [`reduce`] is F2-linear, these
/// products can be XOR-accumulated across many products and reduced only once — an inner product of
/// `n` terms costs one reduction instead of `n`.
#[inline]
pub fn mul_wide<U: Underlier + OpsClmul + PackedUnderlier<u64>>(a: U, b: U) -> [U; 2] {
	let prod_0 = U::clmulepi64::<0x00>(a, b); // 128-bit pre-reduction product elements 0
	let prod_1 = U::clmulepi64::<0x11>(a, b); // 128-bit pre-reduction product elements 1
	[prod_0, prod_1]
}

/// Multiplies two elements in GF(2^64) using SIMD carry-less multiplication.
///
/// This function performs multiplication in the Monbijou field GF(2^64) defined by
/// the reduction polynomial X^64 + X^4 + X^3 + X + 1. Composes the widening multiply [`mul_wide`]
/// with the two-stage [`reduce`]; both are inlined.
#[inline]
#[allow(dead_code)]
pub fn mul<U: Underlier + OpsClmul + PackedUnderlier<u64>>(a: U, b: U) -> U {
	reduce(mul_wide(a, b))
}

/// Widening (unreduced) degree-2 Monbijou multiply in the *packed* representation: the three raw
/// base-field products `[t0, t1, t2] = [x.lo·y.lo, x.lo·y.hi + x.hi·y.lo, x.hi·y.hi]`, each a
/// 128-bit carry-less product occupying one packed lane.
///
/// No combination, scaling, or reduction happens here — those are all F2-linear and deferred to
/// [`reduce_128b`], so an inner product XOR-accumulates the three products and reduces once.
#[inline]
pub fn mul_wide_128b<U: Underlier + OpsClmul + PackedUnderlier<u64>>(x: U, y: U) -> [U; 3] {
	let t0 = U::clmulepi64::<0x00>(x, y); // x.lo · y.lo
	let t2 = U::clmulepi64::<0x11>(x, y); // x.hi · y.hi
	// t1 = x.lo·y.hi + x.hi·y.lo (XOR in the binary field).
	let t1 = U::xor(U::clmulepi64::<0x01>(x, y), U::clmulepi64::<0x10>(x, y));
	[t0, t1, t2]
}

/// Reduce the three raw products from [`mul_wide_128b`] into a packed GF(2^128) element.
///
/// The extension is `Y^2 = XY + 1`, so `coeff 0 = t0 + t2` and `coeff 1 = (t1 + t0 + t2) + X·t2`
/// (Karatsuba recovers the cross term `x.lo·y.hi + x.hi·y.lo` as `t1 + t0 + t2`). The
/// multiply-by-X on the unreduced `t2` is done here, so [`reduce`] folds the two coefficients into
/// the low/high halves once.
#[inline]
pub fn reduce_128b<U: Underlier + OpsClmul + PackedUnderlier<u64>>([t0, t1, t2]: [U; 3]) -> U {
	let term0 = U::xor(t0, t2);
	let term1 = U::xor(t1, mul_x(t2));
	reduce([term0, term1])
}

/// Multiplies two elements in GF(2^128), the degree-2 extension of GF(2^64), in the *packed*
/// representation (coefficient 0 in the low half of each lane, coefficient 1 in the high half).
///
/// This field is defined as GF(2)\[X, Y\] / (X^64 + X^4 + X^3 + X + 1) / (Y^2 + XY + 1). Composes
/// [`mul_wide_128b`] with [`reduce_128b`].
#[inline]
pub fn mul_128b<U: Underlier + OpsClmul + PackedUnderlier<u64>>(x: U, y: U) -> U {
	reduce_128b(mul_wide_128b(x, y))
}

/// Widening (unreduced) degree-2 Monbijou multiply in *sliced* representation: the three raw
/// base-field Karatsuba products `[t0, t1, t2] = [a0·b0, (a0+a1)·(b0+b1), a1·b1]` (each a `[U; 2]`
/// lane pair, as from [`mul_wide`]) of two GF(2^128) elements `[U; 2]`.
///
/// Each element is `[U; 2]`: index i holds coefficient i (a base-field element or SIMD pack), so
/// the two coefficients live in separate registers. No combination, scaling, or reduction happens
/// here — all F2-linear and deferred to [`reduce_sliced_128b`], so an inner product
/// XOR-accumulates the three products and reduces once.
#[inline]
pub fn mul_wide_sliced_128b<U: Underlier + OpsClmul + PackedUnderlier<u64>>(
	x: [U; 2],
	y: [U; 2],
) -> [[U; 2]; 3] {
	let [x0, x1] = x;
	let [y0, y1] = y;
	[
		mul_wide(x0, y0),
		mul_wide(U::xor(x0, x1), U::xor(y0, y1)),
		mul_wide(x1, y1),
	]
}

/// Reduce the three raw products from [`mul_wide_sliced_128b`] into a GF(2^128) element `[U; 2]`.
///
/// The extension is GF(2)\[X, Y\] / (X^64 + X^4 + X^3 + X + 1) / (Y^2 + XY + 1), so `Y^2 = XY + 1`
/// and, writing `a = a0 + a1·Y` and `b = b0 + b1·Y`,
///
/// ```text
/// coeff 0 = a0·b0 + a1·b1             = t0 + t2
/// coeff 1 = a0·b1 + a1·b0 + X·(a1·b1) = (t1 + t0 + t2) + X·t2
/// ```
///
/// with Karatsuba recovering the cross term `a0·b1 + a1·b0` as `t1 + t0 + t2`. The combinations and
/// the multiply-by-X on the unreduced `t2` happen here, so each output coefficient is reduced once
/// by [`reduce`].
#[inline]
pub fn reduce_sliced_128b<U: Underlier + OpsClmul + PackedUnderlier<u64>>(
	[t0, t1, t2]: [[U; 2]; 3],
) -> [U; 2] {
	// `[U; 2]` is itself an `Underlier` (blanket array impl), so `Underlier::xor` adds the
	// unreduced product pairs lane-wise.
	let term0 = Underlier::xor(t0, t2);
	let term1 = Underlier::xor(Underlier::xor(Underlier::xor(t1, t0), t2), mul_x_wide(t2));
	[reduce(term0), reduce(term1)]
}

/// Multiplies two elements of GF(2^128), the degree-2 extension of the Monbijou field, in
/// *sliced* representation.
///
/// This transposed layout keeps the two coefficients in separate registers and processes every
/// packed lane in parallel; it computes the same field product as [`mul_128b`], which instead
/// packs the two coefficients into the low and high halves of a single value. Composes
/// [`mul_wide_sliced_128b`] with [`reduce_sliced_128b`].
#[inline]
pub fn mul_sliced_128b<U: Underlier + OpsClmul + PackedUnderlier<u64>>(
	x: [U; 2],
	y: [U; 2],
) -> [U; 2] {
	reduce_sliced_128b(mul_wide_sliced_128b(x, y))
}

/// Widening (unreduced) degree-3 Monbijou multiply in *sliced* representation: the six raw
/// base-field Karatsuba products `[m0, m1, m2, m01, m02, m12]` (each a `[U; 2]` lane pair, as from
/// [`mul_wide`]) of two GF(2^192) elements `[U; 3]`.
///
/// Each element is `[U; 3]`: index i holds coefficient i (a base-field element or SIMD pack), so
/// the three coefficients live in separate registers. No combination, scaling, or reduction happens
/// here — all F2-linear and deferred to [`reduce_sliced_192b`], so an inner product
/// XOR-accumulates the six products and reduces once.
#[inline]
pub fn mul_wide_sliced_192b<U: Underlier + OpsClmul + PackedUnderlier<u64>>(
	x: [U; 3],
	y: [U; 3],
) -> [[U; 2]; 6] {
	let [x0, x1, x2] = x;
	let [y0, y1, y2] = y;
	[
		mul_wide(x0, y0),
		mul_wide(x1, y1),
		mul_wide(x2, y2),
		mul_wide(U::xor(x0, x1), U::xor(y0, y1)),
		mul_wide(U::xor(x0, x2), U::xor(y0, y2)),
		mul_wide(U::xor(x1, x2), U::xor(y1, y2)),
	]
}

/// Reduce the six raw products from [`mul_wide_sliced_192b`] into a GF(2^192) element `[U; 3]`.
///
/// The tower is GF(2)\[X, Y\] / (X^64 + X^4 + X^3 + X + 1) / (Y^3 + X), so `Y^3 = X`; see
/// [`super::soft64::mul_192b`] for the algebra. Karatsuba gives the degree-≤4 coefficients
/// `c0..c4`; folding `Y^3 = X`, `Y^4 = X·Y` gives `z0 = c0 + X·c3`, `z1 = c1 + X·c4`, `z2 = c2`.
/// The combinations and the multiply-by-X (per lane: shift the unreduced product left by one and
/// fold the overflow bit back in with the polynomial) happen here, so each output coefficient is
/// reduced once by [`reduce`].
#[inline]
pub fn reduce_sliced_192b<U: Underlier + OpsClmul + PackedUnderlier<u64>>(
	[m0, m1, m2, m01, m02, m12]: [[U; 2]; 6],
) -> [U; 3] {
	// Product coefficients (unreduced): c0 = m0, c4 = m2, and the Karatsuba cross terms
	//   c1 = m01 + m0 + m1,  c2 = m02 + m0 + m1 + m2,  c3 = m12 + m1 + m2.
	// `[U; 2]` is itself an `Underlier` (blanket array impl), so `Underlier::xor` adds the
	// unreduced product pairs lane-wise.
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

/// Multiply an unreduced base-field product by X, per lane: shift each 64-bit lane left by one and
/// fold the overflow bit (bit 63) back in with the low terms of the reduction polynomial
/// (X^4 + X^3 + X + 1).
#[inline]
fn mul_x<U: Underlier + OpsClmul + PackedUnderlier<u64>>(p: U) -> U {
	const POLY: u64 = 0x1B;
	let poly = <U as PackedUnderlier<u64>>::broadcast(POLY);
	U::xor(U::slli_epi64::<1>(p), U::and(poly, U::movepi64_mask(p)))
}

/// Multiply an unreduced product pair `[lo, hi]` by X, applying [`mul_x`] to each limb.
#[inline]
fn mul_x_wide<U: Underlier + OpsClmul + PackedUnderlier<u64>>([lo, hi]: [U; 2]) -> [U; 2] {
	[mul_x(lo), mul_x(hi)]
}

/// Multiplies two elements of GF(2^192), the degree-3 extension of the Monbijou field, in *sliced*
/// representation. Composes [`mul_wide_sliced_192b`] with [`reduce_sliced_192b`].
#[inline]
pub fn mul_sliced_192b<U: Underlier + OpsClmul + PackedUnderlier<u64>>(
	x: [U; 3],
	y: [U; 3],
) -> [U; 3] {
	reduce_sliced_192b(mul_wide_sliced_192b(x, y))
}

/// Reduce a Monbijou widening product `[prod_0, prod_1]` (two 128-bit carry-less products) to a
/// single packed base-field element, modulo X^64 + X^4 + X^3 + X + 1.
///
/// Two CLMUL folds by the low-degree terms `0x1B`; each lane's low 64 bits are packed into the
/// output via `unpacklo_epi64`. This is an F2-linear map, so unreduced products may be summed by
/// XOR and reduced once at the end.
#[inline]
pub fn reduce<U: Underlier + OpsClmul + PackedUnderlier<u64>>([prod_0, prod_1]: [U; 2]) -> U {
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

	use super::{
		mul, mul_128b, mul_sliced_128b, mul_sliced_192b, mul_wide, mul_wide_128b,
		mul_wide_sliced_128b, reduce, reduce_128b, reduce_sliced_128b,
	};
	use crate::{Underlier, monbijou::soft64};

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

		// The 128b packed reduction is F2-linear: accumulating two widening products by XOR and
		// reducing once equals reducing each (a field multiply) and summing (field addition is XOR).
		#[test]
		fn packed_wide_mul_deferred_reduction_128b(
			a1 in any::<u128>(), b1 in any::<u128>(),
			a2 in any::<u128>(), b2 in any::<u128>(),
		) {
			let to = |v: u128| unsafe { std::mem::transmute::<u128, __m128i>(v) };
			let from = |v: __m128i| unsafe { std::mem::transmute::<__m128i, u128>(v) };
			let acc = Underlier::xor(
				mul_wide_128b::<__m128i>(to(a1), to(b1)),
				mul_wide_128b::<__m128i>(to(a2), to(b2)),
			);
			prop_assert_eq!(from(reduce_128b(acc)), packed_mul(a1, b1) ^ packed_mul(a2, b2));
		}

		// Same deferred-reduction check for the sliced 128b path, with both lanes carrying a
		// distinct pair of terms of an inner product.
		#[test]
		fn sliced_wide_mul_deferred_reduction_128b(
			a1 in any::<u128>(), a2 in any::<u128>(),
			b1 in any::<u128>(), b2 in any::<u128>(),
			c1 in any::<u128>(), c2 in any::<u128>(),
			d1 in any::<u128>(), d2 in any::<u128>(),
		) {
			// Term 1 = a·b, term 2 = c·d; lane 0 holds the first element of each pair, lane 1 the
			// second.
			let w1 = mul_wide_sliced_128b::<__m128i>(to_sliced(a1, a2), to_sliced(b1, b2));
			let w2 = mul_wide_sliced_128b::<__m128i>(to_sliced(c1, c2), to_sliced(d1, d2));
			let (z0, z1) = from_sliced(reduce_sliced_128b(Underlier::xor(w1, w2)));
			prop_assert_eq!(z0, packed_mul(a1, b1) ^ packed_mul(c1, d1));
			prop_assert_eq!(z1, packed_mul(a2, b2) ^ packed_mul(c2, d2));
		}

		// The CLMUL base-field mul agrees with the soft64 reference (compared in lane 0).
		#[test]
		fn base_mul_matches_soft64(x in any::<u64>(), y in any::<u64>()) {
			let to = |v: u64| unsafe { std::mem::transmute::<u128, __m128i>(v as u128) };
			let got = unsafe { std::mem::transmute::<__m128i, u128>(mul::<__m128i>(to(x), to(y))) };
			prop_assert_eq!(got as u64, soft64::mul(x, y));
		}

		// The reduction is F2-linear: accumulating two unreduced products by XOR and reducing once
		// equals reducing each and summing (lane 0).
		#[test]
		fn base_wide_mul_deferred_reduction(
			x1 in any::<u64>(), y1 in any::<u64>(),
			x2 in any::<u64>(), y2 in any::<u64>(),
		) {
			let to = |v: u64| unsafe { std::mem::transmute::<u128, __m128i>(v as u128) };
			let from = |v: __m128i| unsafe { std::mem::transmute::<__m128i, u128>(v) } as u64;
			let [p0, p1] = mul_wide::<__m128i>(to(x1), to(y1));
			let [q0, q1] = mul_wide::<__m128i>(to(x2), to(y2));
			let acc = reduce([Underlier::xor(p0, q0), Underlier::xor(p1, q1)]);
			prop_assert_eq!(from(acc), soft64::mul(x1, y1) ^ soft64::mul(x2, y2));
		}

		// The degree-3 sliced multiply agrees with the soft64 reference in both packed lanes.
		#[test]
		fn sliced_matches_soft64_192b(
			xa in any::<[u64; 3]>(), xb in any::<[u64; 3]>(),
			ya in any::<[u64; 3]>(), yb in any::<[u64; 3]>(),
		) {
			// Coefficient i is [lane0 = e0[i], lane1 = e1[i]] packed into an __m128i.
			let to_sliced = |e0: [u64; 3], e1: [u64; 3]| -> [__m128i; 3] {
				std::array::from_fn(|i| unsafe {
					std::mem::transmute::<u128, __m128i>((e0[i] as u128) | ((e1[i] as u128) << 64))
				})
			};
			let lane = |z: [__m128i; 3], lane: u32| -> [u64; 3] {
				std::array::from_fn(|i| {
					let c = unsafe { std::mem::transmute::<__m128i, u128>(z[i]) };
					(c >> (64 * lane)) as u64
				})
			};

			let z = mul_sliced_192b::<__m128i>(to_sliced(xa, xb), to_sliced(ya, yb));
			prop_assert_eq!(lane(z, 0), soft64::mul_192b(xa, ya));
			prop_assert_eq!(lane(z, 1), soft64::mul_192b(xb, yb));
		}
	}
}
