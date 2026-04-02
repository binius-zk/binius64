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
	let carry = U::unpackhi_epi64(lsb_at_top, U::zero());
	let shifted = U::xor(shifted, carry);

	// Build a mask from the original LSB of each 128-bit element (bit 0 of the low qword).
	// Duplicate the low qword's lsb_at_top into both lanes, then broadcast via movepi64_mask.
	let lsb_mask = U::movepi64_mask(U::unpacklo_epi64(lsb_at_top, lsb_at_top));

	// Conditionally XOR with INV_X.
	U::xor(shifted, U::and(inv_x, lsb_mask))
}

/// Multiply two GHASH field elements using CLMUL instructions.
#[inline]
pub fn mul<U: Underlier + OpsClmul + PackedUnderlier<u128>>(x: U, y: U) -> U {
	// Based on the C++ reference implementation
	// The algorithm performs polynomial multiplication followed by reduction

	// t1a = x.lo * y.hi
	let t1a = U::clmulepi64::<0x01>(x, y);
	// t1b = x.hi * y.lo
	let t1b = U::clmulepi64::<0x10>(x, y);
	// t1 = t1a + t1b (XOR in binary field)
	let mut t1 = U::xor(t1a, t1b);
	// t2 = x.hi * y.hi
	let t2 = U::clmulepi64::<0x11>(x, y);
	// Reduce t1 and t2
	t1 = gf2_128_reduce(t1, t2);
	// t0 = x.lo * y.lo
	let mut t0 = U::clmulepi64::<0x00>(x, y);
	// Final reduction
	t0 = gf2_128_reduce(t0, t1);

	t0
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
