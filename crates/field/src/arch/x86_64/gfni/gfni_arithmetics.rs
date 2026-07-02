// Copyright 2024-2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use crate::arch::x86_64::simd::simd_arithmetic::TowerSimdType;

/// Affine map taking a GFNI inversion back to the `AESTowerField8b` basis.
///
/// `gf2p8affineinv` inverts in the GFNI (AES SBox) basis.
/// Composing with this identity permutation yields the raw field inverse.
#[rustfmt::skip]
pub const IDENTITY_MAP: i64 = u64::from_le_bytes([
	0b10000000,
	0b01000000,
	0b00100000,
	0b00010000,
	0b00001000,
	0b00000100,
	0b00000010,
	0b00000001,
]) as i64;

/// x86 GFNI byte operations, available on any SIMD register (`M128`/`M256`/`M512`) with `gfni`.
pub(crate) trait GfniType: Copy + TowerSimdType {
	#[allow(unused)]
	fn gf2p8affine_epi64_epi8(x: Self, a: Self) -> Self;
	fn gf2p8mul_epi8(a: Self, b: Self) -> Self;
	fn gf2p8affineinv_epi64_epi8(x: Self, a: Self) -> Self;
}

/// GFNI byte multiply: `gf2p8mul` produces the reduced GF(2^8) product in one instruction.
#[inline(always)]
pub(crate) fn gfni_mul<U: GfniType>(a: U, b: U) -> U {
	U::gf2p8mul_epi8(a, b)
}

/// GFNI byte inverse, mapping zero to zero.
///
/// One `gf2p8affineinv`: invert in the GFNI basis, then map back with [`IDENTITY_MAP`].
#[inline(always)]
pub(crate) fn gfni_invert_or_zero<U: GfniType>(a: U) -> U {
	U::gf2p8affineinv_epi64_epi8(a, U::set_epi_64(IDENTITY_MAP))
}
