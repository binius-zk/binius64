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
