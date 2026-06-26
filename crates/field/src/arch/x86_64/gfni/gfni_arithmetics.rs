// Copyright 2024-2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use bytemuck::TransparentWrapper;

use crate::{
	AESTowerField8b,
	arch::{portable::packed::PackedPrimitiveType, x86_64::simd::simd_arithmetic::TowerSimdType},
	arithmetic_traits::{InvertOrZero, WideMul},
	underlier::UnderlierType,
};

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

pub(super) trait GfniType: Copy + TowerSimdType {
	#[allow(unused)]
	fn gf2p8affine_epi64_epi8(x: Self, a: Self) -> Self;
	fn gf2p8mul_epi8(a: Self, b: Self) -> Self;
	fn gf2p8affineinv_epi64_epi8(x: Self, a: Self) -> Self;
}

/// GFNI multiplication wrapper for AES packings: `gf2p8mul` produces the reduced byte directly.
#[repr(transparent)]
#[derive(TransparentWrapper)]
pub struct Gfni<T>(T);

impl<U: GfniType + UnderlierType> std::ops::Mul for Gfni<PackedPrimitiveType<U, AESTowerField8b>> {
	type Output = Self;

	#[inline(always)]
	fn mul(self, rhs: Self) -> Self {
		let (a, b) = (Self::peel(self), Self::peel(rhs));
		Self::wrap(U::gf2p8mul_epi8(a.0, b.0).into())
	}
}

/// GFNI widening multiply for AES packings: `gf2p8mul` already produces the reduced byte, so the
/// wide product is `Self` and `reduce` is the identity. The single-instruction multiply covers
/// `M128`/`M256`/`M512` (any [`GfniType`]).
#[repr(transparent)]
#[derive(TransparentWrapper)]
pub struct GfniWideMul<T>(T);

impl<U: GfniType + UnderlierType> WideMul for GfniWideMul<PackedPrimitiveType<U, AESTowerField8b>> {
	type Output = PackedPrimitiveType<U, AESTowerField8b>;

	#[inline(always)]
	fn wide_mul(a: Self, b: Self) -> Self::Output {
		let a = Self::peel(a);
		let b = Self::peel(b);
		U::gf2p8mul_epi8(a.0, b.0).into()
	}

	#[inline(always)]
	fn reduce(wide: Self::Output) -> Self {
		Self::wrap(wide)
	}
}

impl<U: GfniType + UnderlierType> InvertOrZero for Gfni<PackedPrimitiveType<U, AESTowerField8b>> {
	#[inline(always)]
	fn invert_or_zero(self) -> Self {
		let val_gfni = Self::peel(self).to_underlier();

		// Calculate inversion and linear transformation to the original field with a single
		// instruction
		let transform_after = U::set_epi_64(IDENTITY_MAP);
		let inv_gfni = U::gf2p8affineinv_epi64_epi8(val_gfni, transform_after);

		Self::wrap(inv_gfni.into())
	}
}
