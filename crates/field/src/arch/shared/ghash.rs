// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

// Items in this module are conditionally used by architecture-specific GHASH backends. On
// targets where none of those backends are compiled (e.g. wasm32 without simd128), the entire
// module is unused, so allow dead code here rather than sprinkling attributes on every item.
#![allow(dead_code)]

use std::ops::{Add, AddAssign, Sub, SubAssign};

use crate::{Divisible, underlier::UnderlierWithBitOps};

/// Trait for underliers that support CLMUL operations which are needed for the
/// GHASH multiplication algorithm.
pub trait ClMulUnderlier: UnderlierWithBitOps + Divisible<u128> {
	/// Performs CLMUL operation on two 64-bit values that are selected from 128-bit lanes
	/// by the bytes of the IMM8 parameter.
	fn clmulepi64<const IMM8: i32>(a: Self, b: Self) -> Self;

	/// For each 128-bit lane, shifts the lower 64 bits to the upper 64 bits and zeroes the lower
	/// 64-bit.
	fn move_64_to_hi(a: Self) -> Self;
}

#[inline]
pub fn mul_clmul<U: ClMulUnderlier>(x: U, y: U) -> U {
	// Based on the C++ reference implementation
	// The algorithm performs polynomial multiplication followed by reduction

	// t1a = x.lo * y.hi
	let t1a = U::clmulepi64::<0x01>(x, y);

	// t1b = x.hi * y.lo
	let t1b = U::clmulepi64::<0x10>(x, y);

	// t1 = t1a + t1b (XOR in binary field)
	let mut t1 = t1a ^ t1b;

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

/// The version of the multiplication for optimized suqare operation.
#[inline]
pub fn square_clmul<U: ClMulUnderlier>(x: U) -> U {
	// t1 from the previous function is always zero for squaring
	// t2 = x.hi * x.hi
	let t2 = U::clmulepi64::<0x11>(x, x);

	// Calculate t1 * x^64
	let t1 = gf2_128_shift_reduce(t2);

	// t0 = x.lo * x.lo
	let mut t0 = U::clmulepi64::<0x00>(x, x);

	// Final reduction
	t0 = gf2_128_reduce(t0, t1);

	t0
}

// The reduction polynomial x^128 + x^7 + x^2 + x + 1 is represented as 0x87
const POLY: u128 = 0x87;

/// Performs reduction step: returns t0 + x^64 * t1
#[inline]
fn gf2_128_reduce<U: ClMulUnderlier>(mut t0: U, t1: U) -> U {
	let poly = <U as UnderlierWithBitOps>::broadcast_subvalue(POLY);

	// t0 = t0 XOR (t1 << 64)
	// In SIMD, left shift by 64 bits is shifting by 8 bytes
	t0 ^= U::move_64_to_hi(t1);

	// t0 = t0 XOR clmul(t1, poly, 0x01)
	// This multiplies the high 64 bits of t1 with the low 64 bits of poly
	t0 ^= U::clmulepi64::<0x01>(t1, poly);

	t0
}

/// Returns a `x^64 * t` after reduction.
fn gf2_128_shift_reduce<U: ClMulUnderlier>(t: U) -> U {
	let poly = <U as UnderlierWithBitOps>::broadcast_subvalue(POLY);
	let mut result = U::move_64_to_hi(t);

	result ^= U::clmulepi64::<0x01>(t, poly);

	result
}

/// An unreduced product of two `GF(2^128)` elements, stored as three 128-bit limbs
/// `(lo, hi, mid)` where `mid = cross_a XOR cross_b`. Values of this type can be summed by XOR
/// and reduced once at the end via [`reduce`](WideGhashProduct::reduce).
///
/// Uses the "schoolbook" form: 4 independent CLMULs for the multiply and 2 reduction CLMULs per
/// reduce.
#[derive(Clone, Copy, Default, Debug)]
pub struct WideGhashProduct<U: ClMulUnderlier> {
	lo: U,
	hi: U,
	mid: U,
}

impl<U: ClMulUnderlier> WideGhashProduct<U> {
	/// Widening multiply with 4 independent CLMULs, no reduction.
	#[inline]
	pub fn wide_mul(x: U, y: U) -> Self {
		let lo = U::clmulepi64::<0x00>(x, y);
		let hi = U::clmulepi64::<0x11>(x, y);
		let cross_a = U::clmulepi64::<0x01>(x, y);
		let cross_b = U::clmulepi64::<0x10>(x, y);
		Self {
			lo,
			hi,
			mid: cross_a ^ cross_b,
		}
	}

	/// Reduce the accumulated wide product to a single GF(2^128) element.
	/// Costs 2 CLMULs (the reduction steps).
	#[inline]
	pub fn reduce(self) -> U {
		let t1 = gf2_128_reduce(self.mid, self.hi);
		gf2_128_reduce(self.lo, t1)
	}
}

impl<U: ClMulUnderlier> Add for WideGhashProduct<U> {
	type Output = Self;

	#[inline]
	fn add(self, rhs: Self) -> Self {
		Self {
			lo: self.lo ^ rhs.lo,
			hi: self.hi ^ rhs.hi,
			mid: self.mid ^ rhs.mid,
		}
	}
}

impl<U: ClMulUnderlier> AddAssign for WideGhashProduct<U> {
	#[inline]
	fn add_assign(&mut self, rhs: Self) {
		self.lo ^= rhs.lo;
		self.hi ^= rhs.hi;
		self.mid ^= rhs.mid;
	}
}

// In characteristic 2, subtraction is identical to addition (XOR).
impl<U: ClMulUnderlier> Sub for WideGhashProduct<U> {
	type Output = Self;

	#[inline]
	fn sub(self, rhs: Self) -> Self {
		Self {
			lo: self.lo ^ rhs.lo,
			hi: self.hi ^ rhs.hi,
			mid: self.mid ^ rhs.mid,
		}
	}
}

impl<U: ClMulUnderlier> SubAssign for WideGhashProduct<U> {
	#[inline]
	fn sub_assign(&mut self, rhs: Self) {
		self.lo ^= rhs.lo;
		self.hi ^= rhs.hi;
		self.mid ^= rhs.mid;
	}
}

#[cfg(test)]
mod tests {
	use rand::{SeedableRng, rngs::StdRng};

	use crate::{Random, WideMul, arch::OptimalPackedB128};

	/// Stress-test accumulation of many widening products. Correctness / linearity for each
	/// individual packed width is covered by the proptest suite in `packed_ghash.rs`.
	#[test]
	fn test_wide_mul_accumulation() {
		type P = OptimalPackedB128;

		let mut rng = StdRng::seed_from_u64(999);
		let n = 64;

		let a_vals: Vec<P> = (0..n).map(|_| P::random(&mut rng)).collect();
		let b_vals: Vec<P> = (0..n).map(|_| P::random(&mut rng)).collect();

		let wide_sum = a_vals
			.iter()
			.zip(b_vals.iter())
			.map(|(&a, &b)| P::wide_mul(a, b))
			.fold(<P as WideMul>::Output::default(), |acc, w| acc + w);
		let reduced = P::reduce(wide_sum);

		let direct_sum: P = a_vals
			.iter()
			.zip(b_vals.iter())
			.map(|(&a, &b)| a * b)
			.fold(P::default(), |acc, p| acc + p);

		assert_eq!(reduced, direct_sum);
	}
}
