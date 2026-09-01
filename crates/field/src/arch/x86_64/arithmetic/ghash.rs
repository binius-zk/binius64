// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

// Items in this module are conditionally used by the x86_64 GHASH backends. Depending on which
// CLMUL target features are enabled, some items may be unused, so allow dead code here rather
// than sprinkling attributes on every item.
#![allow(dead_code)]

use std::{
	iter::Sum,
	ops::{Add, AddAssign, Sub, SubAssign},
};

use bytemuck::TransparentWrapper;

use crate::{
	Divisible, Ghash128b as GhashB128, WideMul,
	arch::portable::arithmetic::ghash::POLY,
	arithmetic_traits::{MulX, Square},
	packed_fields::primitive::PackedPrimitiveType,
	underlier::Underlier,
};

/// Trait for underliers whose 128-bit lanes each hold a GHASH element, carrying the per-lane bit
/// operations the GHASH arithmetic needs.
///
/// None of these is a multiply, so an implementation needs only the base SIMD instruction set for
/// its width.
pub trait GhashLanes: Underlier + Divisible<u128> {
	/// For each 128-bit lane, shifts the lower 64 bits to the upper 64 bits and zeroes the lower
	/// 64-bit.
	fn move_64_to_hi(a: Self) -> Self;

	/// For each 64-bit lane, shifts the value left by one bit.
	fn shl_1_epi64(a: Self) -> Self;

	/// For each 64-bit lane, shifts the value right by 63 bits, leaving the lane's top bit as its
	/// low bit.
	fn shr_63_epi64(a: Self) -> Self;

	/// For each 128-bit lane, returns all ones when bit 127 is set and all zeros otherwise.
	fn broadcast_bit_127(a: Self) -> Self;
}

/// Trait for underliers that support CLMUL operations which are needed for the
/// GHASH multiplication algorithm.
///
/// On x86_64 this abstracts over the `M128`/`M256`/`M512` wrappers for `__m128i`/`__m256i`/
/// `__m512i`, so the same algorithm code drives PCLMULQDQ and VPCLMULQDQ.
pub trait ClMulUnderlier: GhashLanes {
	/// Performs CLMUL operation on two 64-bit values that are selected from 128-bit lanes
	/// by the bytes of the IMM8 parameter.
	fn clmulepi64<const IMM8: i32>(a: Self, b: Self) -> Self;
}

/// Scaling wrapper for the GHASH packings whose width has a vector shift.
///
/// One implementation covers every register width, since the sequence needs no carry-less multiply.
#[repr(transparent)]
#[derive(TransparentWrapper)]
pub struct GhashMulX<T>(T);

impl<U: GhashLanes> MulX for GhashMulX<PackedPrimitiveType<U, GhashB128>> {
	#[inline]
	fn mul_x(self) -> Self {
		Self::wrap(PackedPrimitiveType::wrap(mul_x(PackedPrimitiveType::peel(Self::peel(self)))))
	}
}

/// Scales every 128-bit GHASH lane by `X`.
#[inline]
pub fn mul_x<U: GhashLanes>(x: U) -> U {
	// No instruction shifts a whole 128-bit lane by one bit, so build the shift from the two
	// 64-bit halves. The bit leaving the low half belongs at position 64, which is where moving it
	// up a half-lane puts it. The bit leaving the high half is the coefficient of X^128 and falls
	// off the top.
	let shifted = U::shl_1_epi64(x) ^ U::move_64_to_hi(U::shr_63_epi64(x));

	// That is the term the modulus rewrites: X^128 = 0x87, folded back in where bit 127 was set.
	let overflow = U::broadcast_bit_127(x) & <U as Divisible<u128>>::broadcast(POLY);

	shifted ^ overflow
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

/// Square wrapper for the full-width GHASH square via CLMUL, available for any [`ClMulUnderlier`] —
/// this single impl covers `M256`/`M512` (and `M128`) whenever the corresponding CLMUL target
/// feature is present.
#[repr(transparent)]
#[derive(TransparentWrapper)]
pub struct GhashClMul<T>(T);

impl<U: ClMulUnderlier> Square for GhashClMul<PackedPrimitiveType<U, GhashB128>> {
	#[inline]
	fn square(self) -> Self {
		Self::wrap(PackedPrimitiveType::from_underlier(square_clmul(
			Self::peel(self).to_underlier(),
		)))
	}
}

/// Performs reduction step: returns t0 + x^64 * t1
#[inline]
fn gf2_128_reduce<U: ClMulUnderlier>(mut t0: U, t1: U) -> U {
	let poly = <U as Divisible<u128>>::broadcast(POLY);

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
	let poly = <U as Divisible<u128>>::broadcast(POLY);
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

impl<U: ClMulUnderlier> MulX for WideGhashProduct<U> {
	/// Shifts the represented 256-bit polynomial `lo + mid·X^64 + hi·X^128` left by one bit.
	///
	/// Each 64-bit lane shifts up by one; the bit leaving the top of a lane belongs 64 bit
	/// positions higher, which — since consecutive limbs overlap by 64 bits — is the corresponding
	/// lane of the next limb. `hi` has no limb above it, so its low lane's carry moves to its own
	/// high lane.
	///
	/// Every limb is a carry-less product of 64-bit halves, so it has degree at most 126, and
	/// XOR-accumulating such products preserves that. Bit 127 of `hi` is therefore clear and
	/// nothing is shifted out of the top.
	#[inline]
	fn mul_x(self) -> Self {
		let (shl_lo, shl_mid, shl_hi) =
			(U::shl_1_epi64(self.lo), U::shl_1_epi64(self.mid), U::shl_1_epi64(self.hi));
		let (carry_lo, carry_mid, carry_hi) =
			(U::shr_63_epi64(self.lo), U::shr_63_epi64(self.mid), U::shr_63_epi64(self.hi));

		Self {
			lo: shl_lo,
			mid: shl_mid ^ carry_lo,
			hi: shl_hi ^ carry_mid ^ U::move_64_to_hi(carry_hi),
		}
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

impl<U: ClMulUnderlier> Sum for WideGhashProduct<U> {
	#[inline]
	fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
		iter.fold(Self::default(), |acc, x| acc + x)
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

#[repr(transparent)]
#[derive(bytemuck::TransparentWrapper)]
pub struct GhashClMulWideMul<T>(T);

impl<U: ClMulUnderlier> WideMul for GhashClMulWideMul<PackedPrimitiveType<U, GhashB128>> {
	type Output = WideGhashProduct<U>;

	fn wide_mul(a: Self, b: Self) -> Self::Output {
		WideGhashProduct::wide_mul(
			PackedPrimitiveType::peel(Self::peel(a)),
			PackedPrimitiveType::peel(Self::peel(b)),
		)
	}

	fn reduce(wide: Self::Output) -> Self {
		Self::wrap(PackedPrimitiveType::wrap(wide.reduce()))
	}
}

#[cfg(test)]
mod tests {
	use rand::{Rng, SeedableRng, rngs::StdRng};

	use super::{ClMulUnderlier, GhashLanes, POLY, WideGhashProduct, mul_x};
	use crate::{
		Divisible, Random, WideMul,
		arch::{OptimalPackedB128, portable::arithmetic::ghash::ghash_mul_x},
		arithmetic_traits::MulX,
	};

	/// Scaling by X commutes with the reduction: scaling the unreduced product matches multiplying
	/// the reduced product by X (the field element 2) in every 128-bit lane.
	#[allow(dead_code)]
	fn check_mul_x_wide<U: ClMulUnderlier>(mut rng: impl Rng) {
		let x = U::broadcast(2);

		for _ in 0..64 {
			let wide = WideGhashProduct::wide_mul(U::random(&mut rng), U::random(&mut rng));

			assert_eq!(
				wide.mul_x().reduce(),
				WideGhashProduct::wide_mul(wide.reduce(), x).reduce()
			);
		}
	}

	// Covers every CLMUL underlier width the target supports, since the scaling impl is shared
	// but its per-lane shift is not.
	#[cfg(target_feature = "pclmulqdq")]
	#[test]
	fn mul_x_wide_commutes_with_reduce() {
		let mut rng = StdRng::seed_from_u64(0);

		check_mul_x_wide::<crate::arch::x86_64::m128::M128>(&mut rng);
		#[cfg(all(target_feature = "avx2", target_feature = "vpclmulqdq"))]
		check_mul_x_wide::<crate::arch::x86_64::m256::M256>(&mut rng);
		#[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
		check_mul_x_wide::<crate::arch::x86_64::m512::M512>(&mut rng);
	}

	/// The vector sequence agrees with the scalar reference in every 128-bit lane.
	#[allow(dead_code)]
	fn check_mul_x<U: GhashLanes>(mut rng: impl Rng) {
		// Values that exercise the carry between the 64-bit halves and the fold of the modulus.
		const BOUNDARY: [u128; 8] = [
			0,
			1,
			2,
			POLY,
			1 << 127,
			(1 << 127) | 1,
			(1 << 127) | (1 << 63),
			u128::MAX,
		];

		// Give neighbouring lanes different boundary values, so a sequence that let one lane's
		// carry leak into the next would fail here and not only on random input.
		let mut cases = (0..BOUNDARY.len())
			.map(|i| Divisible::<u128>::from_iter(BOUNDARY.iter().copied().cycle().skip(i)))
			.collect::<Vec<U>>();
		cases.extend((0..64).map(|_| U::random(&mut rng)));

		for u in cases {
			let scaled = mul_x(u);
			let expected = Divisible::<u128>::value_iter(u).map(ghash_mul_x);

			for (i, (lane, want)) in Divisible::<u128>::value_iter(scaled)
				.zip(expected)
				.enumerate()
			{
				assert_eq!(lane, want, "lane {i} of {u:?}");
			}
		}
	}

	// Covers every width whose lane operations the target provides, since the algorithm is shared
	// but the shifts and the bit-127 mask underneath it are written per width.
	#[cfg(target_feature = "sse2")]
	#[test]
	fn mul_x_matches_the_scalar_reference() {
		let mut rng = StdRng::seed_from_u64(0);

		check_mul_x::<crate::arch::x86_64::m128::M128>(&mut rng);
		#[cfg(target_feature = "avx2")]
		check_mul_x::<crate::arch::x86_64::m256::M256>(&mut rng);
		#[cfg(target_feature = "avx512f")]
		check_mul_x::<crate::arch::x86_64::m512::M512>(&mut rng);
	}

	/// Stress-test accumulation of many widening products. Correctness / linearity for each
	/// individual packed width is covered by the proptest suite in `packed_fields::ghash`.
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
