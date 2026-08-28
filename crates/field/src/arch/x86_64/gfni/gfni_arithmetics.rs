// Copyright 2024-2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use bytemuck::TransparentWrapper;

use crate::{
	Divisible, Rijndael8b,
	arithmetic_traits::{InvertOrZero, WideMul},
	packed_fields::primitive::PackedPrimitiveType,
	underlier::Underlier,
};

/// The 8x8 identity matrix over `GF(2)`, in the byte encoding `vgf2p8affineinvqb` expects.
///
/// The instruction reads its matrix bytes from the high output bit down.
/// So the identity descends from `0x80` rather than ascending from `0x01`.
///
/// A zero byte inverts to zero, so no special case is needed.
#[rustfmt::skip]
const IDENTITY_MAP: u64 = u64::from_le_bytes([
	0b10000000,
	0b01000000,
	0b00100000,
	0b00010000,
	0b00001000,
	0b00000100,
	0b00000010,
	0b00000001,
]);

/// SIMD underlier exposing the two GFNI byte instructions the AES packings need.
pub(super) trait GfniType: Underlier {
	fn gf2p8mul_epi8(a: Self, b: Self) -> Self;
	fn gf2p8affineinv_epi64_epi8(x: Self, a: Self) -> Self;
}

/// GFNI multiplication wrapper for AES packings: `gf2p8mul` produces the reduced byte directly.
#[repr(transparent)]
#[derive(TransparentWrapper)]
pub struct Gfni<T>(T);

impl<U: GfniType + Underlier> std::ops::Mul for Gfni<PackedPrimitiveType<U, Rijndael8b>> {
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

impl<U: GfniType + Underlier> WideMul for GfniWideMul<PackedPrimitiveType<U, Rijndael8b>> {
	type Output = PackedPrimitiveType<U, Rijndael8b>;

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

impl<U: GfniType + Divisible<u64>> InvertOrZero for Gfni<PackedPrimitiveType<U, Rijndael8b>> {
	#[inline(always)]
	fn invert_or_zero(self) -> Self {
		let val_gfni = Self::peel(self).to_underlier();

		// One instruction inverts every byte: the identity matrix leaves `inv(b)` untransformed.
		let identity_map = <U as Divisible<u64>>::broadcast(IDENTITY_MAP);
		let inv_gfni = U::gf2p8affineinv_epi64_epi8(val_gfni, identity_map);

		Self::wrap(inv_gfni.into())
	}
}
