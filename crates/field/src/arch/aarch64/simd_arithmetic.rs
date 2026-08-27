// Copyright 2024-2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use std::{
	arch::aarch64::*,
	iter::Sum,
	ops::{Add, AddAssign, Sub, SubAssign},
};

use bytemuck::TransparentWrapper;

use super::m128::M128;
use crate::{
	arch::portable::packed::PackedPrimitiveType, arithmetic_traits::WideMul,
	fields::rijndael::Rijndael8b,
};

#[inline]
pub fn packed_aes_16x8b_invert_or_zero(x: M128) -> M128 {
	lookup_16x8b(AES_INVERT_OR_ZERO_LOOKUP_TABLE, x)
}

#[inline]
pub fn packed_aes_16x8b_square(x: M128) -> M128 {
	// Freshman's dream: in GF(2^n), (a0 + a1*x + ... + a7*x^7)^2
	// = a0 + a1*x^2 + a2*x^4 + ... + a7*x^14
	// This is just bit-spreading (interleave with zeros), then reducing mod the AES polynomial.
	//
	// Split each byte into low nibble (bits 3..0) and high nibble (bits 7..4):
	// - Low nibble spread gives degree < 8, no reduction needed
	// - High nibble maps to x^8..x^14 terms, reduced via a 16-entry table
	unsafe {
		let x: uint8x16_t = x.into();

		// Nibble-to-spread table: maps 4-bit nibble abcd -> 0a0b0c0d (8-bit spread)
		let spread_tbl = vld1q_u8(
			[
				0x00, 0x01, 0x04, 0x05, 0x10, 0x11, 0x14, 0x15, 0x40, 0x41, 0x44, 0x45, 0x50, 0x51,
				0x54, 0x55,
			]
			.as_ptr(),
		);

		// Reduction table: maps high nibble (a7,a6,a5,a4) to the reduction of
		// a4*x^8 + a5*x^10 + a6*x^12 + a7*x^14 mod (x^8 + x^4 + x^3 + x + 1)
		let reduce_tbl = vld1q_u8(
			[
				0x00, 0x1B, 0x6C, 0x77, 0xAB, 0xB0, 0xC7, 0xDC, 0x9A, 0x81, 0xF6, 0xED, 0x31, 0x2A,
				0x5D, 0x46,
			]
			.as_ptr(),
		);

		let lo = vandq_u8(x, vdupq_n_u8(0x0F));

		let hi = vshrq_n_u8(x, 4);

		let spread_lo = vqtbl1q_u8(spread_tbl, lo);

		let reduced_hi = vqtbl1q_u8(reduce_tbl, hi);

		veorq_u8(spread_lo, reduced_hi).into()
	}
}

/// The unreduced product of two [`PackedAESBinaryField16x8b`](PackedPrimitiveType) values: the 16
/// per-byte carryless products, split into their low bytes (`lo`, which need no reduction) and high
/// bytes (`hi`, the overflow above `x^7` still to be folded down). Both the carryless multiply and
/// the GF(2^8) reduction are linear, so products accumulate by XOR and reduce once at the end via
/// [`packed_aes_16x8b_reduce`].
#[derive(Clone, Copy, Debug)]
pub struct WideAes16x8bProduct {
	c0: poly16x8_t,
	c1: poly16x8_t,
}

impl Default for WideAes16x8bProduct {
	fn default() -> Self {
		unsafe {
			Self {
				c0: vdupq_n_p16(0),
				c1: vdupq_n_p16(0),
			}
		}
	}
}

impl Add for WideAes16x8bProduct {
	type Output = Self;

	#[inline]
	fn add(mut self, rhs: Self) -> Self {
		self += rhs;
		self
	}
}

impl AddAssign for WideAes16x8bProduct {
	#[inline]
	fn add_assign(&mut self, rhs: Self) {
		unsafe {
			self.c0 = vaddq_p16(self.c0, rhs.c0);
			self.c1 = vaddq_p16(self.c1, rhs.c1);
		}
	}
}

// In characteristic 2, subtraction is identical to addition (XOR).
impl Sub for WideAes16x8bProduct {
	type Output = Self;

	#[inline]
	fn sub(mut self, rhs: Self) -> Self {
		self -= rhs;
		self
	}
}

impl SubAssign for WideAes16x8bProduct {
	#[inline]
	fn sub_assign(&mut self, rhs: Self) {
		unsafe {
			self.c0 = vaddq_p16(self.c0, rhs.c0);
			self.c1 = vaddq_p16(self.c1, rhs.c1);
		}
	}
}

impl Sum for WideAes16x8bProduct {
	#[inline]
	fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
		iter.fold(Self::default(), |acc, x| acc + x)
	}
}

/// Widening (unreduced) GF(2^8) multiply of the packed bytes via `vmull_p8`, deferring reduction.
/// See <https://doc.rust-lang.org/beta/core/arch/x86_64/fn._mm_gf2p8mul_epi8.html>.
#[inline]
pub fn packed_aes_16x8b_wide_mul(a: M128, b: M128) -> WideAes16x8bProduct {
	unsafe {
		let a = vreinterpretq_p8_p128(a.into());
		let b = vreinterpretq_p8_p128(b.into());
		let c0 = vmull_p8(vget_low_p8(a), vget_low_p8(b));
		let c1 = vmull_p8(vget_high_p8(a), vget_high_p8(b));

		WideAes16x8bProduct { c0, c1 }
	}
}

/// Reduces an accumulated [`WideAes16x8bProduct`] back to the packed GF(2^8) bytes.
///
/// # Overview
///
/// Each 16-bit lane holds a carryless product, or an XOR-accumulation of such products.
/// The low byte carries degrees 0..7 and passes through unchanged.
/// The high byte carries degrees 8..15 and folds down mod p(x) = x^8 + x^4 + x^3 + x + 1.
///
/// # Nibble tables
///
/// The fold is F_2-linear in the high byte's 8 bits.
/// Splitting the byte into two 4-bit nibbles turns the fold into two 16-entry lookups:
///
/// ```text
///     RED_LO[b3 b2 b1 b0] = b0*(x^8  mod p) + b1*(x^9  mod p) + b2*(x^10 mod p) + b3*(x^11 mod p)
///     RED_HI[b3 b2 b1 b0] = b0*(x^12 mod p) + b1*(x^13 mod p) + b2*(x^14 mod p) + b3*(x^15 mod p)
/// ```
///
/// Two byte-shuffle lookups replace a two-stage Barrett fold of four carryless multiplies.
/// M-series cores issue four byte shuffles per cycle but only two carryless multiplies.
/// The packed squaring above reduces its high nibble through the same table scheme.
#[inline]
pub fn packed_aes_16x8b_reduce(wide: WideAes16x8bProduct) -> M128 {
	unsafe {
		let WideAes16x8bProduct { c0, c1 } = wide;

		// Deinterleave the 16 per-byte products into their low and high bytes.
		//
		//     c0 bytes: [ l0 h0 l1 h1 ... l7  h7  ]
		//     c1 bytes: [ l8 h8 l9 h9 ... l15 h15 ]
		//     cl      = [ l0 l1 ... l15 ]  (degrees 0..7, kept)
		//     ch      = [ h0 h1 ... h15 ]  (degrees 8..15, folded below)
		let c0 = vreinterpretq_u8_p16(c0);
		let c1 = vreinterpretq_u8_p16(c1);

		let cl = vuzp1q_u8(c0, c1);
		let ch = vuzp2q_u8(c0, c1);

		// Reductions of x^8, x^9, x^10, x^11 mod p(x): 0x1B, 0x36, 0x6C, 0xD8.
		// Entry n is the XOR of the values selected by n's set bits.
		let red_lo = vld1q_u8(
			[
				0x00, 0x1B, 0x36, 0x2D, 0x6C, 0x77, 0x5A, 0x41, 0xD8, 0xC3, 0xEE, 0xF5, 0xB4, 0xAF,
				0x82, 0x99,
			]
			.as_ptr(),
		);

		// Reductions of x^12, x^13, x^14, x^15 mod p(x): 0xAB, 0x4D, 0x9A, 0x2F.
		// Entry n is the XOR of the values selected by n's set bits.
		let red_hi = vld1q_u8(
			[
				0x00, 0xAB, 0x4D, 0xE6, 0x9A, 0x31, 0xD7, 0x7C, 0x2F, 0x84, 0x62, 0xC9, 0xB5, 0x1E,
				0xF8, 0x53,
			]
			.as_ptr(),
		);

		// Split each high byte into its two table indices.
		let lo_nibble = vandq_u8(ch, vdupq_n_u8(0x0F));
		let hi_nibble = vshrq_n_u8(ch, 4);

		// Fold the whole high byte below x^8: one lookup per nibble, XOR-combined.
		let folded = veorq_u8(vqtbl1q_u8(red_lo, lo_nibble), vqtbl1q_u8(red_hi, hi_nibble));

		// Reduced lane = low byte + folded high byte.
		vreinterpretq_p128_u8(veorq_u8(cl, folded)).into()
	}
}

/// Widening-multiply wrapper for the aarch64 `vmull_p8` AES packing, mirroring the GHASH
/// [`GhashClMulWideMul`](super::arithmetic::ghash::GhashClMulWideMul) pattern: `wide_mul` runs the
/// carryless multiply (deferring reduction) and `reduce` folds the high bytes back down. The packed
/// field forwards its `WideMul` to this via the `define_packed_binary_field!` macro.
#[repr(transparent)]
#[derive(TransparentWrapper)]
pub struct VmullWideMul<T>(T);

impl WideMul for VmullWideMul<PackedPrimitiveType<M128, Rijndael8b>> {
	type Output = WideAes16x8bProduct;

	#[inline]
	fn wide_mul(a: Self, b: Self) -> Self::Output {
		packed_aes_16x8b_wide_mul(Self::peel(a).to_underlier(), Self::peel(b).to_underlier())
	}

	#[inline]
	fn reduce(wide: Self::Output) -> Self {
		Self::wrap(PackedPrimitiveType::from_underlier(packed_aes_16x8b_reduce(wide)))
	}
}

#[inline]
pub fn lookup_16x8b(table: [u8; 256], x: M128) -> M128 {
	unsafe {
		let table: [uint8x16x4_t; 4] = std::mem::transmute(table);
		let x = x.into();
		let y0 = vqtbl4q_u8(table[0], x);
		let y1 = vqtbl4q_u8(table[1], veorq_u8(x, vdupq_n_u8(0x40)));
		let y2 = vqtbl4q_u8(table[2], veorq_u8(x, vdupq_n_u8(0x80)));
		let y3 = vqtbl4q_u8(table[3], veorq_u8(x, vdupq_n_u8(0xC0)));
		veorq_u8(veorq_u8(y0, y1), veorq_u8(y2, y3)).into()
	}
}

pub const AES_INVERT_OR_ZERO_LOOKUP_TABLE: [u8; 256] = [
	0x00, 0x01, 0x8D, 0xF6, 0xCB, 0x52, 0x7B, 0xD1, 0xE8, 0x4F, 0x29, 0xC0, 0xB0, 0xE1, 0xE5, 0xC7,
	0x74, 0xB4, 0xAA, 0x4B, 0x99, 0x2B, 0x60, 0x5F, 0x58, 0x3F, 0xFD, 0xCC, 0xFF, 0x40, 0xEE, 0xB2,
	0x3A, 0x6E, 0x5A, 0xF1, 0x55, 0x4D, 0xA8, 0xC9, 0xC1, 0x0A, 0x98, 0x15, 0x30, 0x44, 0xA2, 0xC2,
	0x2C, 0x45, 0x92, 0x6C, 0xF3, 0x39, 0x66, 0x42, 0xF2, 0x35, 0x20, 0x6F, 0x77, 0xBB, 0x59, 0x19,
	0x1D, 0xFE, 0x37, 0x67, 0x2D, 0x31, 0xF5, 0x69, 0xA7, 0x64, 0xAB, 0x13, 0x54, 0x25, 0xE9, 0x09,
	0xED, 0x5C, 0x05, 0xCA, 0x4C, 0x24, 0x87, 0xBF, 0x18, 0x3E, 0x22, 0xF0, 0x51, 0xEC, 0x61, 0x17,
	0x16, 0x5E, 0xAF, 0xD3, 0x49, 0xA6, 0x36, 0x43, 0xF4, 0x47, 0x91, 0xDF, 0x33, 0x93, 0x21, 0x3B,
	0x79, 0xB7, 0x97, 0x85, 0x10, 0xB5, 0xBA, 0x3C, 0xB6, 0x70, 0xD0, 0x06, 0xA1, 0xFA, 0x81, 0x82,
	0x83, 0x7E, 0x7F, 0x80, 0x96, 0x73, 0xBE, 0x56, 0x9B, 0x9E, 0x95, 0xD9, 0xF7, 0x02, 0xB9, 0xA4,
	0xDE, 0x6A, 0x32, 0x6D, 0xD8, 0x8A, 0x84, 0x72, 0x2A, 0x14, 0x9F, 0x88, 0xF9, 0xDC, 0x89, 0x9A,
	0xFB, 0x7C, 0x2E, 0xC3, 0x8F, 0xB8, 0x65, 0x48, 0x26, 0xC8, 0x12, 0x4A, 0xCE, 0xE7, 0xD2, 0x62,
	0x0C, 0xE0, 0x1F, 0xEF, 0x11, 0x75, 0x78, 0x71, 0xA5, 0x8E, 0x76, 0x3D, 0xBD, 0xBC, 0x86, 0x57,
	0x0B, 0x28, 0x2F, 0xA3, 0xDA, 0xD4, 0xE4, 0x0F, 0xA9, 0x27, 0x53, 0x04, 0x1B, 0xFC, 0xAC, 0xE6,
	0x7A, 0x07, 0xAE, 0x63, 0xC5, 0xDB, 0xE2, 0xEA, 0x94, 0x8B, 0xC4, 0xD5, 0x9D, 0xF8, 0x90, 0x6B,
	0xB1, 0x0D, 0xD6, 0xEB, 0xC6, 0x0E, 0xCF, 0xAD, 0x08, 0x4E, 0xD7, 0xE3, 0x5D, 0x50, 0x1E, 0xB3,
	0x5B, 0x23, 0x38, 0x34, 0x68, 0x46, 0x03, 0x8C, 0xDD, 0x9C, 0x7D, 0xA0, 0xCD, 0x1A, 0x41, 0x1C,
];
