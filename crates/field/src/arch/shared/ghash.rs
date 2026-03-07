// Copyright 2025 Irreducible Inc.

use std::ops::{Add, AddAssign};

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

	/// For each 128-bit lane, XORs the high and low 64-bit halves. The result is placed in
	/// the low 64 bits of each lane; the high 64 bits are unspecified.
	fn xor_halves(a: Self) -> Self;
}

#[inline]
#[allow(dead_code)]
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
#[allow(dead_code)]
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

/// An unreduced product of two GF(2^128) elements, stored in Karatsuba form
/// as three 128-bit limbs (lo, hi, mid). Accumulate via XOR (which is free in
/// GF(2)), then call [`reduce`](WideGhashProduct::reduce) once at the end.
#[derive(Clone, Copy, Default)]
pub struct WideGhashProduct<U: ClMulUnderlier> {
	lo: U,
	hi: U,
	mid: U,
}

impl<U: ClMulUnderlier> WideGhashProduct<U> {
	/// Karatsuba widening multiply: 3 CLMULs, no reduction.
	#[inline]
	pub fn widening_mul(x: U, y: U) -> Self {
		let lo = U::clmulepi64::<0x00>(x, y);
		let hi = U::clmulepi64::<0x11>(x, y);
		let mid = U::clmulepi64::<0x00>(U::xor_halves(x), U::xor_halves(y));
		Self { lo, hi, mid }
	}

	/// Reduce the accumulated wide product to a single GF(2^128) element.
	/// Costs 2 CLMULs (the reduction steps).
	#[inline]
	pub fn reduce(self) -> U {
		let cross = self.mid ^ self.lo ^ self.hi;
		let t1 = gf2_128_reduce(cross, self.hi);
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

#[cfg(test)]
mod tests {
	use crate::{
		BinaryField128bGhash, PackedField, Random, WideningMul,
		arch::{OptimalPackedB128, packed_ghash_128::PackedBinaryGhash1x128b},
	};

	fn test_widening_mul_correctness<P: PackedField<Scalar = BinaryField128bGhash> + WideningMul>(
	) {
		use rand::{SeedableRng, rngs::StdRng};

		let mut rng = StdRng::seed_from_u64(42);
		for _ in 0..100 {
			let a = P::random(&mut rng);
			let b = P::random(&mut rng);

			let wide = P::widening_mul(a, b);
			let reduced = P::reduce_wide(wide);
			let direct = a * b;

			assert_eq!(
				reduced, direct,
				"reduce(widening_mul(a, b)) must equal a * b"
			);
		}
	}

	fn test_widening_mul_linearity<P: PackedField<Scalar = BinaryField128bGhash> + WideningMul>()
	{
		use rand::{SeedableRng, rngs::StdRng};

		let mut rng = StdRng::seed_from_u64(123);
		for _ in 0..100 {
			let a1 = P::random(&mut rng);
			let b1 = P::random(&mut rng);
			let a2 = P::random(&mut rng);
			let b2 = P::random(&mut rng);

			let wide1 = P::widening_mul(a1, b1);
			let wide2 = P::widening_mul(a2, b2);
			let sum_reduced = P::reduce_wide(wide1 + wide2);
			let direct_sum = a1 * b1 + a2 * b2;

			assert_eq!(
				sum_reduced, direct_sum,
				"reduce(wide1 + wide2) must equal a1*b1 + a2*b2"
			);
		}
	}

	#[test]
	fn test_widening_mul_correctness_1x128() {
		test_widening_mul_correctness::<PackedBinaryGhash1x128b>();
	}

	#[test]
	fn test_widening_mul_linearity_1x128() {
		test_widening_mul_linearity::<PackedBinaryGhash1x128b>();
	}

	#[test]
	fn test_widening_mul_correctness_optimal() {
		test_widening_mul_correctness::<OptimalPackedB128>();
	}

	#[test]
	fn test_widening_mul_linearity_optimal() {
		test_widening_mul_linearity::<OptimalPackedB128>();
	}

	#[test]
	fn test_widening_mul_accumulation() {
		use rand::{SeedableRng, rngs::StdRng};

		type P = OptimalPackedB128;

		let mut rng = StdRng::seed_from_u64(999);
		let n = 64;

		let a_vals: Vec<P> = (0..n).map(|_| P::random(&mut rng)).collect();
		let b_vals: Vec<P> = (0..n).map(|_| P::random(&mut rng)).collect();

		let wide_sum = a_vals
			.iter()
			.zip(b_vals.iter())
			.map(|(&a, &b)| P::widening_mul(a, b))
			.fold(<P as WideningMul>::Wide::default(), |acc, w| acc + w);
		let reduced = P::reduce_wide(wide_sum);

		let direct_sum: P = a_vals
			.iter()
			.zip(b_vals.iter())
			.map(|(&a, &b)| a * b)
			.fold(P::default(), |acc, p| acc + p);

		assert_eq!(
			reduced, direct_sum,
			"Accumulated widening inner product must equal direct inner product"
		);
	}
}
