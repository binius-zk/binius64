// Copyright 2023-2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

//! Portable (software) implementation of GHASH field multiplication.

use std::{
	iter::Sum,
	ops::{Add, AddAssign, Sub, SubAssign},
};

use bytemuck::TransparentWrapper;

use super::super::univariate_mul_utils_128::{Underlier64bLanes, Underlier128bLanes, bmul64};
use crate::{
	BinaryField128bGhash as GhashB128, WideMul,
	arch::PackedPrimitiveType,
	arithmetic_traits::{MulXWide, Square},
};

/// Multiply two GHASH field elements using software implementation.
///
/// Method described at:
/// * <https://www.bearssl.org/constanttime.html#ghash-for-gcm>
/// * <https://crypto.stackexchange.com/questions/66448/how-does-bearssls-gcm-modular-reduction-work/66462#66462>
///
/// This code does not conform to the bit-endianness requirements of the GCM specification, but is
/// a valid GHASH field multiplication with the modified representation.
#[inline]
pub fn ghash_mul<U: Underlier128bLanes>(x: U, y: U) -> U {
	ghash_wide_mul(x, y).reduce()
}

/// Widening multiply: the schoolbook polynomial product of two GHASH field elements, without the
/// modular reduction. The unreduced result can be accumulated by XOR and reduced once at the end
/// via [`WideGhashProduct::reduce`].
#[inline]
pub fn ghash_wide_mul<U: Underlier128bLanes>(x: U, y: U) -> WideGhashProduct<U> {
	// Convert to U64x2 representation
	let (x1, x0) = U::split_hi_lo_64(x);
	let (y1, y0) = U::split_hi_lo_64(y);

	// Perform multiplication
	let x0r = x0.reverse_bits_64();
	let x1r = x1.reverse_bits_64();
	let x2 = x0 ^ x1;
	let x2r = x0r ^ x1r;

	let y0r = y0.reverse_bits_64();
	let y1r = y1.reverse_bits_64();
	let y2 = y0 ^ y1;
	let y2r = y0r ^ y1r;

	let z0 = bmul64(y0, x0);
	let z1 = bmul64(y1, x1);
	let mut z2 = bmul64(y2, x2);

	let mut z0h = bmul64(y0r, x0r);
	let mut z1h = bmul64(y1r, x1r);
	let mut z2h = bmul64(y2r, x2r);

	z2 ^= z0 ^ z1;
	z2h ^= z0h ^ z1h;
	z0h = z0h.reverse_bits_64().shr_64(1);
	z1h = z1h.reverse_bits_64().shr_64(1);
	z2h = z2h.reverse_bits_64().shr_64(1);

	WideGhashProduct {
		v0: z0,
		v1: z0h ^ z2,
		v2: z1 ^ z2h,
		v3: z1h,
	}
}

#[inline]
pub fn ghash_square<U: Underlier128bLanes>(x: U) -> U {
	// Squared value in the polynomial basis is just a value with bits interleaved with zeroes.
	let (hi, lo) = x.spread_bits_128();

	let (v3, v2) = hi.split_hi_lo_64();
	let (v1, v0) = lo.split_hi_lo_64();

	reduce_64(v0, v1, v2, v3)
}

/// Reduce a 256-bit value represented as four 64-bit values by the GHASH polynomial.
#[inline]
fn reduce_64<U: Underlier128bLanes>(
	mut v0: U::U64,
	mut v1: U::U64,
	mut v2: U::U64,
	v3: U::U64,
) -> U {
	// Reduce modulo X^64 + X^7 + X^2 + X + 1.
	v1 ^= v3 ^ v3.shl_64(1) ^ v3.shl_64(2) ^ v3.shl_64(7);
	v2 ^= v3.shr_64(63) ^ v3.shr_64(62) ^ v3.shr_64(57);
	v0 ^= v2 ^ v2.shl_64(1) ^ v2.shl_64(2) ^ v2.shl_64(7);
	v1 ^= v2.shr_64(63) ^ v2.shr_64(62) ^ v2.shr_64(57);

	// Convert back to 128-bit lanes
	U::join_u64s(v1, v0)
}

/// An unreduced GHASH product, stored as the four 64-bit limbs `(v0, v1, v2, v3)` of the 256-bit
/// schoolbook product. Values of this type can be summed by XOR and reduced once at the end via
/// [`reduce`](WideGhashProduct::reduce).
#[derive(Clone, Copy, Default, Debug)]
pub struct WideGhashProduct<U: Underlier128bLanes> {
	v0: U::U64,
	v1: U::U64,
	v2: U::U64,
	v3: U::U64,
}

impl<U: Underlier128bLanes> WideGhashProduct<U> {
	/// Reduce the accumulated wide product to a single GF(2^128) element.
	#[inline]
	pub fn reduce(self) -> U {
		reduce_64(self.v0, self.v1, self.v2, self.v3)
	}
}

impl<U: Underlier128bLanes> MulXWide for WideGhashProduct<U> {
	/// Shifts the 256-bit schoolbook product left by one bit, carrying between the four 64-bit
	/// limbs.
	///
	/// The product of two 128-bit polynomials has degree at most 254, and XOR-accumulating such
	/// products preserves that, so the top bit of `v3` is always clear and nothing is shifted out.
	#[inline]
	fn mul_x_wide(self) -> Self {
		Self {
			v0: self.v0.shl_64(1),
			v1: self.v1.shl_64(1) ^ self.v0.shr_64(63),
			v2: self.v2.shl_64(1) ^ self.v1.shr_64(63),
			v3: self.v3.shl_64(1) ^ self.v2.shr_64(63),
		}
	}
}

impl<U: Underlier128bLanes> Add for WideGhashProduct<U> {
	type Output = Self;

	#[inline]
	fn add(self, rhs: Self) -> Self {
		Self {
			v0: self.v0 ^ rhs.v0,
			v1: self.v1 ^ rhs.v1,
			v2: self.v2 ^ rhs.v2,
			v3: self.v3 ^ rhs.v3,
		}
	}
}

impl<U: Underlier128bLanes> AddAssign for WideGhashProduct<U> {
	#[inline]
	fn add_assign(&mut self, rhs: Self) {
		self.v0 ^= rhs.v0;
		self.v1 ^= rhs.v1;
		self.v2 ^= rhs.v2;
		self.v3 ^= rhs.v3;
	}
}

impl<U: Underlier128bLanes> Sum for WideGhashProduct<U> {
	#[inline]
	fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
		iter.fold(Self::default(), |acc, x| acc + x)
	}
}

// In characteristic 2, subtraction is identical to addition (XOR).
impl<U: Underlier128bLanes> Sub for WideGhashProduct<U> {
	type Output = Self;

	#[inline]
	fn sub(self, rhs: Self) -> Self {
		Self {
			v0: self.v0 ^ rhs.v0,
			v1: self.v1 ^ rhs.v1,
			v2: self.v2 ^ rhs.v2,
			v3: self.v3 ^ rhs.v3,
		}
	}
}

impl<U: Underlier128bLanes> SubAssign for WideGhashProduct<U> {
	#[inline]
	fn sub_assign(&mut self, rhs: Self) {
		self.v0 ^= rhs.v0;
		self.v1 ^= rhs.v1;
		self.v2 ^= rhs.v2;
		self.v3 ^= rhs.v3;
	}
}

/// Widening-multiply wrapper for the portable GHASH packing.
///
/// [`wide_mul`](WideMul::wide_mul) computes the unreduced schoolbook product via
/// [`ghash_wide_mul`], and [`reduce`](WideMul::reduce) performs the GHASH modular reduction. This
/// defers the reduction so a sum of products is reduced only once.
#[repr(transparent)]
#[derive(bytemuck::TransparentWrapper)]
pub struct GhashWideMul<T>(T);

impl<U: Underlier128bLanes> WideMul for GhashWideMul<PackedPrimitiveType<U, GhashB128>> {
	type Output = WideGhashProduct<U>;

	#[inline]
	fn wide_mul(a: Self, b: Self) -> Self::Output {
		let a = PackedPrimitiveType::peel(Self::peel(a));
		let b = PackedPrimitiveType::peel(Self::peel(b));
		ghash_wide_mul(a, b)
	}

	#[inline]
	fn reduce(wide: Self::Output) -> Self {
		Self::wrap(PackedPrimitiveType::wrap(wide.reduce()))
	}
}

/// Square strategy wrapper for the software GHASH implementation.
///
/// Shared by the portable and wasm32 packings and used by the x86_64 packing when CLMUL is
/// unavailable. Squares via the bit-spread [`ghash_square`], which interleaves the input bits with
/// zeroes and reduces — no carryless multiply required.
#[repr(transparent)]
#[derive(TransparentWrapper)]
pub struct GhashSoftMul<T>(T);

impl<U: Underlier128bLanes> Square for GhashSoftMul<PackedPrimitiveType<U, GhashB128>> {
	#[inline]
	fn square(self) -> Self {
		Self::wrap(PackedPrimitiveType::wrap(ghash_square(PackedPrimitiveType::peel(Self::peel(
			self,
		)))))
	}
}

#[cfg(test)]
mod tests {
	use proptest::{prelude::any, proptest};

	use super::{super::super::m128::M128, MulXWide, ghash_mul, ghash_wide_mul};

	// Exercises the deferred wide-mul building blocks (`ghash_wide_mul` + `WideGhashProduct`) that
	// `GhashWideMul` wraps, directly on the portable `M128`. This runs on every host, whereas the
	// portable `PackedBinaryGhash1x128b` is only a usable `PackedField` on targets where it is the
	// re-exported b128 type (covered there by the proptests in `packed_ghash.rs`).
	proptest! {
		// The split must agree with the fused multiply: wide-multiply then reduce == ghash_mul.
		#[test]
		fn wide_mul_then_reduce_matches_ghash_mul(a in any::<u128>(), b in any::<u128>()) {
			let (a, b) = (M128::from(a), M128::from(b));
			assert_eq!(ghash_wide_mul(a, b).reduce(), ghash_mul(a, b));
		}

		// Accumulate two unreduced products and reduce once.
		#[test]
		fn wide_mul_deferred_accumulation(
			a1 in any::<u128>(), b1 in any::<u128>(),
			a2 in any::<u128>(), b2 in any::<u128>(),
		) {
			let (a1, b1) = (M128::from(a1), M128::from(b1));
			let (a2, b2) = (M128::from(a2), M128::from(b2));
			let acc = ghash_wide_mul(a1, b1) + ghash_wide_mul(a2, b2);
			assert_eq!(acc.reduce(), ghash_mul(a1, b1) ^ ghash_mul(a2, b2));
		}

		// Scaling by X commutes with the reduction: scaling the unreduced product matches
		// multiplying the reduced product by X (the field element 2).
		#[test]
		fn mul_x_wide_commutes_with_reduce(a in any::<u128>(), b in any::<u128>()) {
			let (a, b) = (M128::from(a), M128::from(b));
			let wide = ghash_wide_mul(a, b);
			assert_eq!(wide.mul_x_wide().reduce(), ghash_mul(wide.reduce(), M128::from(2u128)));
		}
	}
}
