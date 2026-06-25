// Copyright 2024-2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

//! PCLMULQDQ-accelerated implementation of GHASH for x86_64.
//!
//! This module provides optimized GHASH multiplication using the PCLMULQDQ instruction
//! available on modern x86_64 processors. The implementation follows the algorithm
//! described in the GHASH specification with polynomial x^128 + x^7 + x^2 + x + 1.

use bytemuck::TransparentWrapper;
use cfg_if::cfg_if;

use super::m128::M128;
#[cfg(not(target_feature = "pclmulqdq"))]
use crate::arch::portable::univariate_mul_utils_128::{Underlier128bLanes, spread_bits_64};
// Used by the CLMUL-accelerated `ClMulUnderlier` impl and the `GhashWideMul` alias below.
#[cfg(target_feature = "pclmulqdq")]
use crate::arch::x86_64::arithmetic::ghash;
use crate::{
	BinaryField128bGhash,
	arch::PackedPrimitiveType,
	arithmetic_traits::{InvertOrZero, Square},
};

/// Widening-multiply wrapper used by the GHASH packing: the reduction-deferring
/// [`GhashClMulWideMul`](ghash::GhashClMulWideMul) when PCLMULQDQ is available, otherwise the
/// portable [`GhashWideMul`](crate::arch::portable::arithmetic::ghash::GhashWideMul) which also
/// defers reduction for deferred-reduction sum-of-products.
#[cfg(target_feature = "pclmulqdq")]
pub type GhashWideMul1x<T> = ghash::GhashClMulWideMul<T>;
#[cfg(not(target_feature = "pclmulqdq"))]
pub type GhashWideMul1x<T> = crate::arch::portable::arithmetic::ghash::GhashWideMul<T>;

/// Square wrapper for the `PackedBinaryGhash1x128b` packing.
pub type GhashSquare1x<T> = Ghash<T>;

/// Invert wrapper for the `PackedBinaryGhash1x128b` packing.
pub type GhashInvert1x<T> = Ghash<T>;

/// `Underlier128bLanes` for x86_64 `M128` — required for the portable `GhashWideMul` fallback.
///
/// Delegates through `u128` (SSE2 load/store) since this path is only active on targets without
/// PCLMULQDQ, where SIMD lane extraction intrinsics are not necessarily available.
#[cfg(not(target_feature = "pclmulqdq"))]
impl Underlier128bLanes for M128 {
	type U64 = u64;

	#[inline(always)]
	fn split_hi_lo_64(self) -> (u64, u64) {
		u128::from(self).split_hi_lo_64()
	}

	#[inline(always)]
	fn join_u64s(high: u64, low: u64) -> Self {
		Self::from(u128::join_u64s(high, low))
	}

	#[inline(always)]
	fn broadcast_64(val: u64) -> Self {
		Self::from(u128::broadcast_64(val))
	}

	#[inline(always)]
	fn spread_bits_128(self) -> (Self, Self) {
		let (hi, lo) = self.split_hi_lo_64();
		(Self::from(spread_bits_64(hi)), Self::from(spread_bits_64(lo)))
	}
}

#[cfg(target_feature = "pclmulqdq")]
impl ghash::ClMulUnderlier for M128 {
	#[inline]
	fn clmulepi64<const IMM8: i32>(a: Self, b: Self) -> Self {
		unsafe { std::arch::x86_64::_mm_clmulepi64_si128::<IMM8>(a.into(), b.into()) }.into()
	}

	#[inline]
	fn move_64_to_hi(a: Self) -> Self {
		unsafe { std::arch::x86_64::_mm_slli_si128::<8>(a.into()) }.into()
	}
}

/// Square wrapper for x86_64 GHASH field arithmetic.
#[repr(transparent)]
#[derive(TransparentWrapper)]
pub struct Ghash<T>(T);

// Implement Square for the GHASH packing
cfg_if! {
	if #[cfg(target_feature = "pclmulqdq")] {
		impl Square for Ghash<PackedPrimitiveType<M128, BinaryField128bGhash>> {
			#[inline]
			fn square(self) -> Self {
				Self::wrap(PackedPrimitiveType::from_underlier(
					crate::arch::x86_64::arithmetic::ghash::square_clmul(
						Self::peel(self).to_underlier(),
					),
				))
			}
		}
	} else {
		impl Square for Ghash<PackedPrimitiveType<M128, BinaryField128bGhash>> {
			#[inline]
			fn square(self) -> Self {
				use super::super::portable::arithmetic::ghash::ghash_square;

				Self::wrap(PackedPrimitiveType::from_underlier(ghash_square(
					Self::peel(self).to_underlier(),
				)))
			}
		}
	}
}

// Implement InvertOrZero for the GHASH packing (Itoh-Tsujii — no CLMUL invert)
impl InvertOrZero for Ghash<PackedPrimitiveType<M128, BinaryField128bGhash>> {
	#[inline]
	fn invert_or_zero(self) -> Self {
		Self::wrap(crate::arch::invert_b128(Self::peel(self)))
	}
}
