// Copyright 2025 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Modified by Irreducible Inc. (2025): Translated from C++ to Rust
// Original: lib/gf2k/sysdep.h from google/longfellow-zk
//
// Copyright 2026 The Binius Developers

//! Hardware-accelerated GHASH multiplication using CLMUL instructions.
//!
//! This implementation is derived from:
//! <https://github.com/google/longfellow-zk/blob/main/lib/gf2k/sysdep.h>

use std::array;

use crate::{PackedUnderlier, Underlier, underlier::OpsClmul};

/// Multiply a packed GHASH field element by X^{-1} using SIMD operations.
///
/// This is equivalent to `mul(x, broadcast(INV_X))` but optimized: per 128-bit lane, right-shift
/// by 1 and conditionally XOR with X^{-1} if the LSB was set.
#[inline]
pub fn mul_inv_x<U: Underlier + OpsClmul + PackedUnderlier<u128>>(x: U) -> U {
	let inv_x = <U as PackedUnderlier<u128>>::broadcast(super::INV_X);

	// Put bit 0 of each 64-bit lane into bit 63.
	let lsb_at_top = U::slli_epi64::<63>(x);

	// Right-shift each 64-bit lane by 1.
	let shifted = U::srli_epi64::<1>(x);

	// Carry bit 0 of the high qword into bit 63 of the low qword within each 128-bit lane.
	// unpackhi gives us [hi_a, hi_b] from two inputs; with zero as second arg, this moves
	// the high qword to the low position and zeros the high.
	let carry = U::unpackhi_epi64(lsb_at_top, U::ZERO);
	let shifted = U::xor(shifted, carry);

	// Build a mask from the original LSB of each 128-bit element (bit 0 of the low qword).
	// Duplicate the low qword's lsb_at_top into both lanes, then broadcast via movepi64_mask.
	let lsb_mask = U::movepi64_mask(U::unpacklo_epi64(lsb_at_top, lsb_at_top));

	// Conditionally XOR with INV_X.
	U::xor(shifted, U::and(inv_x, lsb_mask))
}

/// Widening (unreduced) CLMUL GHASH multiply: the schoolbook product as three 128-bit limbs
/// `[t0, t1, t2]`, without the modular reduction.
///
/// Per 128-bit lane, `t0 = x.lo·y.lo` (low), `t1 = x.lo·y.hi ⊕ x.hi·y.lo` (middle, at offset
/// `X^64`), and `t2 = x.hi·y.hi` (high, at offset `X^128`), so the product is
/// `t0 + t1·X^64 + t2·X^128`. Because the reduction ([`reduce`]) is F2-linear, these limbs can be
/// XOR-accumulated across many products and reduced only once — an inner product of `n` terms
/// costs one reduction instead of `n`.
#[inline]
pub fn mul_wide<U: Underlier + OpsClmul + PackedUnderlier<u128>>(x: U, y: U) -> [U; 3] {
	// t0 = x.lo * y.lo
	let t0 = U::clmulepi64::<0x00>(x, y);
	// t1 = x.lo * y.hi + x.hi * y.lo (XOR in binary field)
	let t1 = U::xor(U::clmulepi64::<0x01>(x, y), U::clmulepi64::<0x10>(x, y));
	// t2 = x.hi * y.hi
	let t2 = U::clmulepi64::<0x11>(x, y);

	[t0, t1, t2]
}

/// Reduce the wide product `[t0, t1, t2]` to a single GHASH field element.
///
/// Folds the high limb into the middle, then the middle into the low, via `gf2_128_reduce`. Each
/// fold is F2-linear, so `reduce` is F2-linear in the limbs: unreduced products may be summed by
/// XOR and reduced once at the end.
#[inline]
pub fn reduce<U: Underlier + OpsClmul + PackedUnderlier<u128>>([t0, t1, t2]: [U; 3]) -> U {
	let t1 = gf2_128_reduce(t1, t2);
	gf2_128_reduce(t0, t1)
}

/// Multiply two GHASH field elements using CLMUL instructions.
///
/// Composes the widening multiply [`mul_wide`] with the modular [`reduce`]; both are inlined.
#[inline]
pub fn mul<U: Underlier + OpsClmul + PackedUnderlier<u128>>(x: U, y: U) -> U {
	reduce(mul_wide(x, y))
}

/// Multiply two GHASH field elements using CLMUL instructions.
#[inline]
pub fn square<U: Underlier + OpsClmul + PackedUnderlier<u128>>(x: U) -> U {
	// t2 = x.hi * y.hi
	let t2 = U::clmulepi64::<0x11>(x, x);
	// Reduce t1 and t2
	let t1 = gf2_128_shift_reduce(t2);
	// t0 = x.lo * y.lo
	let mut t0 = U::clmulepi64::<0x00>(x, x);
	// Final reduction
	t0 = gf2_128_reduce(t0, t1);

	t0
}

/// Performs reduction step: returns t0 + x^64 * t1
#[inline]
fn gf2_128_reduce<U: Underlier + OpsClmul + PackedUnderlier<u128>>(mut t0: U, t1: U) -> U {
	// The reduction polynomial x^128 + x^7 + x^2 + x + 1 is represented as 0x87
	const POLY: u128 = 0x87;
	let poly = <U as PackedUnderlier<u128>>::broadcast(POLY);

	// t0 = t0 XOR (t1 << 64)
	// In SIMD, left shift by 64 bits is shifting by 8 bytes
	t0 = U::xor(t0, U::slli_si128::<8>(t1));

	// t0 = t0 XOR clmul(t1, poly, 0x01)
	// This multiplies the high 64 bits of t1 with the low 64 bits of poly
	t0 = U::xor(t0, U::clmulepi64::<0x01>(t1, poly));

	t0
}

/// Performs reduction step: returns x^64 * t1
#[inline]
fn gf2_128_shift_reduce<U: Underlier + OpsClmul + PackedUnderlier<u128>>(t1: U) -> U {
	// The reduction polynomial x^128 + x^7 + x^2 + x + 1 is represented as 0x87
	const POLY: u128 = 0x87;
	let poly = <U as PackedUnderlier<u128>>::broadcast(POLY);

	// t0 = t1 << 64
	// In SIMD, left shift by 64 bits is shifting by 8 bytes
	let mut t0 = U::slli_si128::<8>(t1);

	// t0 = t0 XOR clmul(t1, poly, 0x01)
	// This multiplies the high 64 bits of t1 with the low 64 bits of poly
	t0 = U::xor(t0, U::clmulepi64::<0x01>(t1, poly));

	t0
}

/// The two raw 64×64 carry-less products of paired slice registers `u`, `v`, kept in the layout
/// CLMUL emits them — one full 128-bit product per packed element, **not** transposed.
///
/// In the *sliced* representation each register packs one 64-bit slice per 64-bit lane, so per
/// 128-bit lane `clmulepi64::<0x00>` multiplies the low-qword element and `::<0x11>` the high-qword
/// element. Leaving the products un-transposed lets an inner product XOR-accumulate them cheaply;
/// the transpose is deferred to [`reduce_sliced`].
#[inline]
fn clmul_pair<U: Underlier + OpsClmul>(u: U, v: U) -> [U; 2] {
	[U::clmulepi64::<0x00>(u, v), U::clmulepi64::<0x11>(u, v)]
}

/// Widening (unreduced) *sliced* schoolbook GHASH multiply: the three raw carry-less product pairs
/// `[p00, cross, p11]` of the 256-bit product of `x` and `y`, at weights `X^0`, `X^64`, `X^128`.
///
/// Each element is `[U; 2] = [low64, high64]`; each returned pair is a `[U; 2]` of raw
/// (un-transposed) CLMUL products, one per packed element (see `clmul_pair`). Schoolbook forms
/// four 64×64 products and sums the two cross terms. No transpose or reduction happens here — both
/// are F2-linear and deferred to [`reduce_sliced`], so an inner product XOR-accumulates
/// the three pairs and reduces once.
#[inline]
pub fn mul_wide_sliced_schoolbook<U: Underlier + OpsClmul>(
	[x0, x1]: [U; 2],
	[y0, y1]: [U; 2],
) -> [[U; 2]; 3] {
	let p00 = clmul_pair(x0, y0); // x.lo · y.lo
	let cross = Underlier::xor(clmul_pair(x0, y1), clmul_pair(x1, y0)); // x.lo·y.hi + x.hi·y.lo
	let p11 = clmul_pair(x1, y1); // x.hi · y.hi
	[p00, cross, p11]
}

/// Widening (unreduced) *sliced* Karatsuba GHASH multiply: the three raw carry-less product pairs
/// `[p00, pm, p11]`, where `pm = (x.lo+x.hi)·(y.lo+y.hi)` is the single extra Karatsuba product
/// that replaces the two schoolbook cross products.
///
/// Unlike [`mul_wide_sliced_schoolbook`] the middle entry is the *raw* Karatsuba product, **not**
/// the cross term: the recombination `cross = pm - p00 - p11` is deferred to
/// [`reduce_sliced_karatsuba`], so — like the corner products — `pm` XOR-accumulates across an
/// inner product and is recombined only once.
#[inline]
pub fn mul_wide_sliced_karatsuba<U: Underlier + OpsClmul>(
	[x0, x1]: [U; 2],
	[y0, y1]: [U; 2],
) -> [[U; 2]; 3] {
	let p00 = clmul_pair(x0, y0); // x.lo · y.lo
	let p11 = clmul_pair(x1, y1); // x.hi · y.hi
	// (x.lo+x.hi)·(y.lo+y.hi) = x.lo·y.lo + (x.lo·y.hi + x.hi·y.lo) + x.hi·y.hi
	let pm = clmul_pair(U::xor(x0, x1), U::xor(y0, y1));
	[p00, pm, p11]
}

/// Reduce a *sliced* schoolbook widening product (the three raw product pairs `[t0, t1, t2]` from
/// [`mul_wide_sliced_schoolbook`], at weights `X^0, X^64, X^128`) to a GHASH field element
/// `[U; 2] = [low64, high64]`, modulo `X^128 + X^7 + X^2 + X + 1`.
///
/// Each `clmul_pair` entry is a full un-transposed 128-bit product. The two fold steps use the
/// same CLMUL-by-`0x87` reduction as the packed [`reduce`], but applied to the products in place:
/// `clmulepi64::<0x01>` folds each product's high 64 bits (weight `X^192`/`X^128`) into the lower
/// limbs, and `unpacklo`/`unpackhi_epi64` recombine the two elements' overlapping halves only where
/// needed, so the result lands directly in the sliced `[low64, high64]` layout without a separate
/// transpose pass. Being F2-linear, unreduced products may be XOR-summed and reduced once.
#[inline]
pub fn reduce_sliced<U: Underlier + OpsClmul + PackedUnderlier<u128>>(
	[t0, t1, t2]: [[U; 2]; 3],
) -> [U; 2] {
	// The reduction polynomial x^128 + x^7 + x^2 + x + 1 is represented as 0x87
	const POLY: u128 = 0x87;
	let poly = <U as PackedUnderlier<u128>>::broadcast(POLY);

	let t2_hi_times_poly = [
		U::clmulepi64::<0x01>(t2[0], poly),
		U::clmulepi64::<0x01>(t2[1], poly),
	];
	let t2_lo = U::unpacklo_epi64(t2[0], t2[1]);

	let t1_prime = array::from_fn::<_, 2, _>(|i| U::xor(t1[i], t2_hi_times_poly[i]));
	let t1_prime_lo = U::unpacklo_epi64(t1_prime[0], t1_prime[1]);
	let t1_prime_hi = U::xor(U::unpackhi_epi64(t1_prime[0], t1_prime[1]), t2_lo);

	let t1_hi_times_poly = [
		U::clmulepi64::<0x00>(t1_prime_hi, poly),
		U::clmulepi64::<0x01>(t1_prime_hi, poly),
	];

	let t0_prime = array::from_fn::<_, 2, _>(|i| U::xor(t0[i], t1_hi_times_poly[i]));
	let t0_prime_lo = U::unpacklo_epi64(t0_prime[0], t0_prime[1]);
	let t0_prime_hi = U::xor(U::unpackhi_epi64(t0_prime[0], t0_prime[1]), t1_prime_lo);

	[t0_prime_lo, t0_prime_hi]
}

/// Reduce a *sliced* Karatsuba widening product (the three raw product pairs `[p00, pm, p11]` from
/// [`mul_wide_sliced_karatsuba`]) to a GHASH field element `[U; 2] = [low64, high64]`.
///
/// Recovers the middle cross term `cross = pm - p00 - p11` (subtraction is XOR in characteristic 2)
/// and delegates to [`reduce_sliced`].
#[inline]
pub fn reduce_sliced_karatsuba<U: Underlier + OpsClmul + PackedUnderlier<u128>>(
	[p00, pm, p11]: [[U; 2]; 3],
) -> [U; 2] {
	let cross = Underlier::xor(Underlier::xor(pm, p00), p11);
	reduce_sliced([p00, cross, p11])
}

/// Multiply two *sliced* GHASH field elements with the schoolbook widening multiply.
///
/// Each element is `[U; 2] = [low64, high64]`. Composes [`mul_wide_sliced_schoolbook`] with
/// [`reduce_sliced`]; both are inlined.
#[inline]
pub fn mul_sliced_schoolbook<U: Underlier + OpsClmul + PackedUnderlier<u128>>(
	x: [U; 2],
	y: [U; 2],
) -> [U; 2] {
	reduce_sliced(mul_wide_sliced_schoolbook(x, y))
}

/// Multiply two *sliced* GHASH field elements with the Karatsuba widening multiply.
///
/// Each element is `[U; 2] = [low64, high64]`. Composes [`mul_wide_sliced_karatsuba`] with
/// [`reduce_sliced_karatsuba`]; both are inlined.
#[inline]
pub fn mul_sliced_karatsuba<U: Underlier + OpsClmul + PackedUnderlier<u128>>(
	x: [U; 2],
	y: [U; 2],
) -> [U; 2] {
	reduce_sliced_karatsuba(mul_wide_sliced_karatsuba(x, y))
}

#[cfg(all(
	test,
	target_arch = "x86_64",
	target_feature = "pclmulqdq",
	target_feature = "sse2"
))]
mod tests {
	use std::arch::x86_64::__m128i;

	use proptest::prelude::*;

	use super::*;
	use crate::ghash::soft64;

	fn to_u(x: u128) -> __m128i {
		<__m128i as PackedUnderlier<u128>>::broadcast(x)
	}

	fn from_u(x: __m128i) -> u128 {
		<__m128i as PackedUnderlier<u128>>::get(x, 0)
	}

	// Packs two GHASH elements into sliced form across the two 64-bit lanes of `[__m128i; 2]`:
	// register 0 holds both elements' low 64 bits, register 1 holds both elements' high 64 bits.
	fn to_sliced(e0: u128, e1: u128) -> [__m128i; 2] {
		let low = (e0 as u64 as u128) | ((e1 as u64 as u128) << 64);
		let high = ((e0 >> 64) as u64 as u128) | (((e1 >> 64) as u64 as u128) << 64);
		unsafe {
			[
				std::mem::transmute::<u128, __m128i>(low),
				std::mem::transmute::<u128, __m128i>(high),
			]
		}
	}

	// Recovers the two GHASH elements from sliced form.
	fn from_sliced(z: [__m128i; 2]) -> (u128, u128) {
		let low = unsafe { std::mem::transmute::<__m128i, u128>(z[0]) };
		let high = unsafe { std::mem::transmute::<__m128i, u128>(z[1]) };
		let e0 = (low as u64 as u128) | ((high as u64 as u128) << 64);
		let e1 = ((low >> 64) as u64 as u128) | (((high >> 64) as u64 as u128) << 64);
		(e0, e1)
	}

	proptest! {
		// The sliced schoolbook multiply agrees with the reference software multiply, per lane.
		#[test]
		fn test_sliced_schoolbook_matches_soft64(
			a0 in any::<u128>(), a1 in any::<u128>(),
			b0 in any::<u128>(), b1 in any::<u128>(),
		) {
			let z = mul_sliced_schoolbook::<__m128i>(to_sliced(a0, a1), to_sliced(b0, b1));
			let (z0, z1) = from_sliced(z);
			prop_assert_eq!(z0, soft64::mul(a0, b0));
			prop_assert_eq!(z1, soft64::mul(a1, b1));
		}

		// The sliced Karatsuba multiply agrees with the reference software multiply, per lane.
		#[test]
		fn test_sliced_karatsuba_matches_soft64(
			a0 in any::<u128>(), a1 in any::<u128>(),
			b0 in any::<u128>(), b1 in any::<u128>(),
		) {
			let z = mul_sliced_karatsuba::<__m128i>(to_sliced(a0, a1), to_sliced(b0, b1));
			let (z0, z1) = from_sliced(z);
			prop_assert_eq!(z0, soft64::mul(a0, b0));
			prop_assert_eq!(z1, soft64::mul(a1, b1));
		}

		// The schoolbook and Karatsuba sliced multiplies produce identical results.
		#[test]
		fn test_sliced_schoolbook_karatsuba_agree(
			a0 in any::<u128>(), a1 in any::<u128>(),
			b0 in any::<u128>(), b1 in any::<u128>(),
		) {
			let x = to_sliced(a0, a1);
			let y = to_sliced(b0, b1);
			let s = from_sliced(mul_sliced_schoolbook::<__m128i>(x, y));
			let k = from_sliced(mul_sliced_karatsuba::<__m128i>(x, y));
			prop_assert_eq!(s, k);
		}

		// The sliced reduction is F2-linear, so accumulating two unreduced products by XOR and
		// reducing once equals reducing each and summing.
		#[test]
		fn test_sliced_wide_deferred_reduction(
			a0 in any::<u128>(), a1 in any::<u128>(),
			b0 in any::<u128>(), b1 in any::<u128>(),
			c0 in any::<u128>(), c1 in any::<u128>(),
			d0 in any::<u128>(), d1 in any::<u128>(),
		) {
			let p = mul_wide_sliced_schoolbook::<__m128i>(to_sliced(a0, a1), to_sliced(b0, b1));
			let q = mul_wide_sliced_schoolbook::<__m128i>(to_sliced(c0, c1), to_sliced(d0, d1));
			let acc = reduce_sliced::<__m128i>(Underlier::xor(p, q));
			let (z0, z1) = from_sliced(acc);
			prop_assert_eq!(z0, soft64::mul(a0, b0) ^ soft64::mul(c0, d0));
			prop_assert_eq!(z1, soft64::mul(a1, b1) ^ soft64::mul(c1, d1));
		}
	}

	proptest! {
		// The widening multiply followed by reduction agrees with the reference software multiply.
		#[test]
		fn test_clmul_mul_matches_soft64(a in any::<u128>(), b in any::<u128>()) {
			prop_assert_eq!(from_u(mul(to_u(a), to_u(b))), soft64::mul(a, b));
		}

		// The reduction is F2-linear, so accumulating two unreduced products by XOR and reducing
		// once equals reducing each and summing.
		#[test]
		fn test_clmul_wide_mul_deferred_reduction(
			a1 in any::<u128>(), b1 in any::<u128>(),
			a2 in any::<u128>(), b2 in any::<u128>(),
		) {
			let [p0, p1, p2] = mul_wide(to_u(a1), to_u(b1));
			let [q0, q1, q2] = mul_wide(to_u(a2), to_u(b2));
			let acc = reduce([
				Underlier::xor(p0, q0),
				Underlier::xor(p1, q1),
				Underlier::xor(p2, q2),
			]);
			prop_assert_eq!(from_u(acc), soft64::mul(a1, b1) ^ soft64::mul(a2, b2));
		}

	}
}
