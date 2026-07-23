// Copyright 2023-2025 Irreducible Inc.
// Copyright 2026 The Binius Developers
// Copyright (c) 2019-2023 RustCrypto Developers

//! ARMv8 `PMULL`-accelerated GHASH arithmetic.
//!
//! Unlike the x86_64 backend, aarch64 only ever operates on a single 128-bit lane (`M128`), so
//! this module calls the `PMULL` intrinsics directly rather than abstracting over an underlier
//! trait.

use core::arch::aarch64::*;
use std::{
	iter::Sum,
	ops::{Add, AddAssign, Sub, SubAssign},
};

use bytemuck::TransparentWrapper;

use super::super::m128::M128;
use crate::{
	BinaryField128bGhash as GhashB128, WideMul,
	arch::PackedPrimitiveType,
	arithmetic_traits::{MulXWide, Square},
};

// The reduction polynomial x^128 + x^7 + x^2 + x + 1 is represented as 0x87.
const POLY: u128 = 0x87;

/// Carryless multiply of two 64-bit lanes selected from the 128-bit inputs by the bytes of
/// `IMM8`, matching the semantics of x86_64's `clmulepi64`.
#[inline]
fn pmull<const IMM8: i32>(a: M128, b: M128) -> M128 {
	let a_u64x2: uint64x2_t = a.into();
	let b_u64x2: uint64x2_t = b.into();

	let result = match IMM8 {
		0x00 => unsafe { vmull_p64(vgetq_lane_u64(a_u64x2, 0), vgetq_lane_u64(b_u64x2, 0)) },
		0x11 => unsafe { vmull_p64(vgetq_lane_u64(a_u64x2, 1), vgetq_lane_u64(b_u64x2, 1)) },
		0x10 => unsafe { vmull_p64(vgetq_lane_u64(a_u64x2, 0), vgetq_lane_u64(b_u64x2, 1)) },
		0x01 => unsafe { vmull_p64(vgetq_lane_u64(a_u64x2, 1), vgetq_lane_u64(b_u64x2, 0)) },
		_ => panic!("Unsupported IMM8 value for clmulepi64"),
	};

	unsafe { std::mem::transmute::<u128, uint64x2_t>(result) }.into()
}

/// Shifts the lower 64 bits to the upper 64 bits and zeroes the lower 64 bits.
#[inline]
fn move_64_to_hi(a: M128) -> M128 {
	let a_bytes: uint8x16_t = a.into();
	// Shift left by 8 bytes
	unsafe {
		let zero = vdupq_n_u8(0);
		vextq_u8::<8>(zero, a_bytes).into()
	}
}

/// Performs reduction step: returns t0 + x^64 * t1
#[inline]
fn gf2_128_reduce(mut t0: M128, t1: M128) -> M128 {
	let poly = M128::from_u128(POLY);

	t0 ^= move_64_to_hi(t1);
	t0 ^= pmull::<0x01>(t1, poly);

	t0
}

/// Returns `x^64 * t` after reduction.
#[inline]
fn gf2_128_shift_reduce(t: M128) -> M128 {
	let poly = M128::from_u128(POLY);
	let mut result = move_64_to_hi(t);

	result ^= pmull::<0x01>(t, poly);

	result
}

/// The version of the multiplication optimized for the square operation.
#[inline]
pub fn square_clmul(x: M128) -> M128 {
	// t1 from the multiply is always zero for squaring; t2 = x.hi * x.hi
	let t2 = pmull::<0x11>(x, x);
	// Calculate t1 * x^64
	let t1 = gf2_128_shift_reduce(t2);
	// t0 = x.lo * x.lo
	let mut t0 = pmull::<0x00>(x, x);
	// Final reduction
	t0 = gf2_128_reduce(t0, t1);

	t0
}

/// Square strategy wrapper for the aarch64 GHASH packing: the PMULL-accelerated carryless-multiply
/// square via [`square_clmul`]. Mirrors the x86_64 `GhashClMul`, specialized to the single 128-bit
/// `M128` lane that aarch64 NEON provides.
#[repr(transparent)]
#[derive(TransparentWrapper)]
pub struct GhashClMul<T>(T);

impl Square for GhashClMul<PackedPrimitiveType<M128, GhashB128>> {
	#[inline]
	fn square(self) -> Self {
		Self::wrap(PackedPrimitiveType::from_underlier(square_clmul(
			Self::peel(self).to_underlier(),
		)))
	}
}

/// An unreduced product of two `GF(2^128)` elements, stored as three 128-bit limbs
/// `(lo, hi, mid)` where `mid = cross_a XOR cross_b`. Values of this type can be summed by XOR
/// and reduced once at the end via [`reduce`](WideGhashProduct::reduce).
///
/// Uses the "schoolbook" form: 4 independent CLMULs for the multiply and 2 reduction CLMULs per
/// reduce.
#[derive(Clone, Copy, Default, Debug)]
pub struct WideGhashProduct {
	lo: M128,
	hi: M128,
	mid: M128,
}

impl WideGhashProduct {
	/// Widening multiply with 4 independent CLMULs, no reduction.
	#[inline]
	pub fn wide_mul(x: M128, y: M128) -> Self {
		let lo = pmull::<0x00>(x, y);
		let hi = pmull::<0x11>(x, y);
		let cross_a = pmull::<0x01>(x, y);
		let cross_b = pmull::<0x10>(x, y);
		Self {
			lo,
			hi,
			mid: cross_a ^ cross_b,
		}
	}

	/// Reduce the accumulated wide product to a single GF(2^128) element.
	/// Costs 2 CLMULs (the reduction steps).
	#[inline]
	pub fn reduce(self) -> M128 {
		let t1 = gf2_128_reduce(self.mid, self.hi);
		gf2_128_reduce(self.lo, t1)
	}
}

impl MulXWide for WideGhashProduct {
	/// Shifts the represented 256-bit polynomial `lo + mid·X^64 + hi·X^128` left by one bit.
	///
	/// Each 64-bit lane shifts up by one; the bit leaving the top of a lane belongs 64 bit
	/// positions higher, which — since consecutive limbs overlap by 64 bits — is the corresponding
	/// lane of the next limb. `hi` has no limb above it, so its low lane's carry goes to its own
	/// high lane, where rotating the lanes puts it.
	///
	/// Every limb is a carry-less product of 64-bit halves, so it has degree at most 126, and
	/// XOR-accumulating such products preserves that. Bit 127 of `hi` is therefore clear, and the
	/// bit the rotation wraps back into the low lane is zero.
	#[inline]
	fn mul_x_wide(self) -> Self {
		let (v0, v1, v2): (uint64x2_t, uint64x2_t, uint64x2_t) =
			(self.lo.into(), self.mid.into(), self.hi.into());

		unsafe {
			let (sll0, sll1, sll2) =
				(vshlq_n_u64::<1>(v0), vshlq_n_u64::<1>(v1), vshlq_n_u64::<1>(v2));
			let (srl0, srl1, srl2) =
				(vshrq_n_u64::<63>(v0), vshrq_n_u64::<63>(v1), vshrq_n_u64::<63>(v2));

			Self {
				lo: sll0.into(),
				mid: veorq_u64(sll1, srl0).into(),
				hi: veorq_u64(veorq_u64(sll2, srl1), vextq_u64::<1>(srl2, srl2)).into(),
			}
		}
	}
}

impl Add for WideGhashProduct {
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

impl AddAssign for WideGhashProduct {
	#[inline]
	fn add_assign(&mut self, rhs: Self) {
		self.lo ^= rhs.lo;
		self.hi ^= rhs.hi;
		self.mid ^= rhs.mid;
	}
}

impl Sum for WideGhashProduct {
	#[inline]
	fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
		iter.fold(Self::default(), |acc, x| acc + x)
	}
}

// In characteristic 2, subtraction is identical to addition (XOR).
impl Sub for WideGhashProduct {
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

impl SubAssign for WideGhashProduct {
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

impl WideMul for GhashClMulWideMul<PackedPrimitiveType<M128, GhashB128>> {
	type Output = WideGhashProduct;

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
	use proptest::{prelude::any, proptest};

	use super::{M128, MulXWide, WideGhashProduct};

	proptest! {
		// Scaling by X commutes with the reduction: scaling the unreduced product matches
		// multiplying the reduced product by X (the field element 2).
		#[test]
		fn mul_x_wide_commutes_with_reduce(a in any::<u128>(), b in any::<u128>()) {
			let wide = WideGhashProduct::wide_mul(M128::from_u128(a), M128::from_u128(b));
			let mul_x = WideGhashProduct::wide_mul(wide.reduce(), M128::from_u128(2)).reduce();

			assert_eq!(wide.mul_x_wide().reduce(), mul_x);
		}
	}
}
