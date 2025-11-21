// Copyright 2024-2025 Irreducible Inc.

use std::arch::aarch64::*;

use seq_macro::seq;

use super::m128::M128;
use crate::{
	BinaryField, TowerField,
	arch::{
		SimdStrategy,
		portable::packed_arithmetic::{
			PackedTowerField, TowerConstants, UnderlierWithBitConstants,
		},
	},
	arithmetic_traits::{
		MulAlpha, Square, TaggedInvertOrZero, TaggedMul, TaggedMulAlpha, TaggedSquare,
	},
	underlier::WithUnderlier,
};

#[inline]
pub fn packed_aes_16x8b_invert_or_zero(x: M128) -> M128 {
	lookup_16x8b(AES_INVERT_OR_ZERO_LOOKUP_TABLE, x)
}

#[inline]
pub fn packed_aes_16x8b_mul_alpha(x: M128) -> M128 {
	// 0xD3 corresponds to 0x10 after isomorphism from BinaryField8b to AESField
	packed_aes_16x8b_multiply(x, M128::from_le_bytes([0xD3; 16]))
}

#[inline]
pub fn packed_aes_16x8b_multiply(a: M128, b: M128) -> M128 {
	//! Performs a multiplication in GF(2^8) on the packed bytes.
	//! See <https://doc.rust-lang.org/beta/core/arch/x86_64/fn._mm_gf2p8mul_epi8.html>
	unsafe {
		let a = vreinterpretq_p8_p128(a.into());
		let b = vreinterpretq_p8_p128(b.into());
		let c0 = vreinterpretq_p8_p16(vmull_p8(vget_low_p8(a), vget_low_p8(b)));
		let c1 = vreinterpretq_p8_p16(vmull_p8(vget_high_p8(a), vget_high_p8(b)));

		// Reduces the 16-bit output of a carryless multiplication to 8 bits using equation 22 in
		// https://www.intel.com/content/dam/develop/external/us/en/documents/clmul-wp-rev-2-02-2014-04-20.pdf

		// Since q+(x) doesn't fit into 8 bits, we right shift the polynomial (divide by x) and
		// correct for this later. This works because q+(x) is divisible by x/the last polynomial
		// bit is 0. q+(x)/x = (x^8 + x^4 + x^3 + x)/x = 0b100011010 >> 1 = 0b10001101 = 0x8d
		const QPLUS_RSH1: poly8x8_t = unsafe { std::mem::transmute(0x8d8d8d8d8d8d8d8d_u64) };

		// q*(x) = x^4 + x^3 + x + 1 = 0b00011011 = 0x1b
		const QSTAR: poly8x8_t = unsafe { std::mem::transmute(0x1b1b1b1b1b1b1b1b_u64) };

		let cl = vuzp1q_p8(c0, c1);
		let ch = vuzp2q_p8(c0, c1);

		let tmp0 = vmull_p8(vget_low_p8(ch), QPLUS_RSH1);
		let tmp1 = vmull_p8(vget_high_p8(ch), QPLUS_RSH1);

		// Correct for q+(x) having been divided by x
		let tmp0 = vreinterpretq_p8_u16(vshlq_n_u16(vreinterpretq_u16_p16(tmp0), 1));
		let tmp1 = vreinterpretq_p8_u16(vshlq_n_u16(vreinterpretq_u16_p16(tmp1), 1));

		let tmp_hi = vuzp2q_p8(tmp0, tmp1);
		let tmp0 = vreinterpretq_p8_p16(vmull_p8(vget_low_p8(tmp_hi), QSTAR));
		let tmp1 = vreinterpretq_p8_p16(vmull_p8(vget_high_p8(tmp_hi), QSTAR));
		let tmp_lo = vuzp1q_p8(tmp0, tmp1);

		vreinterpretq_p128_p8(vaddq_p8(cl, tmp_lo)).into()
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

impl<PT> TaggedMul<SimdStrategy> for PT
where
	PT: PackedTowerField<Underlier = M128>,
	PT::DirectSubfield: TowerConstants<M128> + BinaryField,
{
	#[inline]
	fn mul(self, rhs: Self) -> Self {
		let alphas = PT::DirectSubfield::ALPHAS_ODD;
		let odd_mask = M128::INTERLEAVE_ODD_MASK[PT::DirectSubfield::TOWER_LEVEL];
		let a = self.as_packed_subfield();
		let b = rhs.as_packed_subfield();
		let p1 = (a * b).to_underlier();
		let (lo, hi) =
			M128::interleave(a.to_underlier(), b.to_underlier(), PT::DirectSubfield::TOWER_LEVEL);
		let (lhs, rhs) =
			M128::interleave(lo ^ hi, alphas ^ (p1 & odd_mask), PT::DirectSubfield::TOWER_LEVEL);
		let p2 = (PT::PackedDirectSubfield::from_underlier(lhs)
			* PT::PackedDirectSubfield::from_underlier(rhs))
		.to_underlier();
		let q1 = p1 ^ flip_even_odd::<PT::DirectSubfield>(p1);
		let q2 = p2 ^ shift_left::<PT::DirectSubfield>(p2);
		Self::from_underlier(q1 ^ (q2 & odd_mask))
	}
}

impl<PT> TaggedMulAlpha<SimdStrategy> for PT
where
	PT: PackedTowerField<Underlier = M128>,
	PT::PackedDirectSubfield: MulAlpha,
{
	#[inline]
	fn mul_alpha(self) -> Self {
		let a0_a1 = self.as_packed_subfield();
		let a0alpha_a1alpha: M128 = a0_a1.mul_alpha().to_underlier();
		let a1_a0 = flip_even_odd::<PT::DirectSubfield>(a0_a1.to_underlier());
		Self::from_underlier(blend_odd_even::<PT::DirectSubfield>(a1_a0 ^ a0alpha_a1alpha, a1_a0))
	}
}

impl<PT> TaggedSquare<SimdStrategy> for PT
where
	PT: PackedTowerField<Underlier = M128>,
	PT::PackedDirectSubfield: MulAlpha + Square,
{
	#[inline]
	fn square(self) -> Self {
		let a0_a1 = self.as_packed_subfield();
		let a0sq_a1sq = Square::square(a0_a1);
		let a1sq_a0sq = flip_even_odd::<PT::DirectSubfield>(a0sq_a1sq.to_underlier());
		let a0sq_plus_a1sq = a0sq_a1sq.to_underlier() ^ a1sq_a0sq;
		let a1_mul_alpha = a0sq_a1sq.mul_alpha();
		Self::from_underlier(blend_odd_even::<PT::DirectSubfield>(
			a1_mul_alpha.to_underlier(),
			a0sq_plus_a1sq,
		))
	}
}

impl<PT> TaggedInvertOrZero<SimdStrategy> for PT
where
	PT: PackedTowerField<Underlier = M128>,
	PT::PackedDirectSubfield: MulAlpha + Square,
{
	#[inline]
	fn invert_or_zero(self) -> Self {
		let a0_a1 = self.as_packed_subfield();
		let a1_a0 = a0_a1.mutate_underlier(flip_even_odd::<PT::DirectSubfield>);
		let a1alpha = a1_a0.mul_alpha();
		let a0_plus_a1alpha = a0_a1 + a1alpha;
		let a1sq_a0sq = Square::square(a1_a0);
		let delta = a1sq_a0sq + (a0_plus_a1alpha * a0_a1);
		let deltainv = delta.invert_or_zero();
		let deltainv_deltainv = deltainv.mutate_underlier(duplicate_odd::<PT::DirectSubfield>);
		let delta_multiplier = a0_a1.mutate_underlier(|a0_a1| {
			blend_odd_even::<PT::DirectSubfield>(a0_a1, a0_plus_a1alpha.to_underlier())
		});
		PT::from_packed_subfield(deltainv_deltainv * delta_multiplier)
	}
}

#[inline]
fn duplicate_odd<F: TowerField>(x: M128) -> M128 {
	match F::TOWER_LEVEL {
		0..=2 => {
			let t = x & M128::INTERLEAVE_ODD_MASK[F::TOWER_LEVEL];
			t | shift_right::<F>(t)
		}
		3 => x.shuffle_u8([1, 1, 3, 3, 5, 5, 7, 7, 9, 9, 11, 11, 13, 13, 15, 15]),
		4 => x.shuffle_u8([2, 3, 2, 3, 6, 7, 6, 7, 10, 11, 10, 11, 14, 15, 14, 15]),
		5 => x.shuffle_u8([4, 5, 6, 7, 4, 5, 6, 7, 12, 13, 14, 15, 12, 13, 14, 15]),
		6 => x.shuffle_u8([8, 9, 10, 11, 12, 13, 14, 15, 8, 9, 10, 11, 12, 13, 14, 15]),
		_ => panic!("Unsupported tower level"),
	}
}

#[inline]
fn flip_even_odd<F: TowerField>(x: M128) -> M128 {
	match F::TOWER_LEVEL {
		0..=2 => {
			let m1 = M128::INTERLEAVE_ODD_MASK[F::TOWER_LEVEL];
			let m2 = M128::INTERLEAVE_EVEN_MASK[F::TOWER_LEVEL];
			shift_right::<F>(x & m1) | shift_left::<F>(x & m2)
		}
		3 => x.shuffle_u8([1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14]),
		4 => x.shuffle_u8([2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9, 14, 15, 12, 13]),
		5 => x.shuffle_u8([4, 5, 6, 7, 0, 1, 2, 3, 12, 13, 14, 15, 8, 9, 10, 11]),
		6 => x.shuffle_u8([8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7]),
		_ => panic!("Unsupported tower level"),
	}
}

#[inline]
fn blend_odd_even<F: TowerField>(x: M128, y: M128) -> M128 {
	let m1 = M128::INTERLEAVE_ODD_MASK[F::TOWER_LEVEL];
	let m2 = M128::INTERLEAVE_EVEN_MASK[F::TOWER_LEVEL];
	(x & m1) | (y & m2)
}

#[inline]
fn shift_left<F: TowerField>(x: M128) -> M128 {
	let tower_level = F::TOWER_LEVEL;
	seq!(TOWER_LEVEL in 0..=5 {
		if tower_level == TOWER_LEVEL {
			return unsafe { vshlq_n_u64(x.into(), 1 << TOWER_LEVEL).into() };
		}
	});
	if tower_level == 6 {
		return unsafe { vcombine_u64(vcreate_u64(0), vget_low_u64(x.into())).into() };
	}
	panic!("Unsupported tower level {tower_level}");
}

#[inline]
fn shift_right<F: TowerField>(x: M128) -> M128 {
	let tower_level = F::TOWER_LEVEL;
	seq!(TOWER_LEVEL in 0..=5 {
		if tower_level == TOWER_LEVEL {
			return unsafe { vshrq_n_u64(x.into(), 1 << TOWER_LEVEL).into() };
		}
	});
	if tower_level == 6 {
		return unsafe { vcombine_u64(vget_high_u64(x.into()), vcreate_u64(0)).into() };
	}
	panic!("Unsupported tower level {tower_level}");
}
