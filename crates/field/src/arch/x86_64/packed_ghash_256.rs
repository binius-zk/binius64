// Copyright 2024-2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

//! VPCLMULQDQ-accelerated implementation of GHASH for x86_64 AVX2.
//!
//! This module provides optimized GHASH multiplication using the VPCLMULQDQ instruction
//! available on modern x86_64 processors with AVX2 support. The implementation follows
//! the algorithm described in the GHASH specification with polynomial x^128 + x^7 + x^2 + x + 1.

// Used by the `GhashWideMul` and `GhashSquareStrategy` fallbacks when VPCLMULQDQ is
// unavailable.
#[cfg(not(target_feature = "vpclmulqdq"))]
use crate::arch::{portable::scaled_arithmetic::Scaled2xWideMul, x86_64::m128::M128};
use crate::{
	BinaryField128bGhash,
	arch::{PackedPrimitiveType, x86_64::m256::M256},
	arithmetic_traits::TaggedInvertOrZero,
};

/// Widening-multiply wrapper used by the GHASH packing: the reduction-deferring vectorized
/// `GhashClMulWideMul` when VPCLMULQDQ is available, otherwise the per-lane `ScaledWideMul`
/// (which still defers reduction, applying the width-1 GHASH `WideMul` to each 128-bit lane).
#[cfg(target_feature = "vpclmulqdq")]
pub type GhashWideMul2x<T> = crate::arch::x86_64::arithmetic::ghash::GhashClMulWideMul<T>;
#[cfg(not(target_feature = "vpclmulqdq"))]
pub type GhashWideMul2x<T> = Scaled2xWideMul<T>;

/// Square wrapper for the GHASH packing: a full-width CLMUL square when VPCLMULQDQ is available,
/// otherwise divide into 128-bit lanes and square each (the 1×128b GHASH square uses PCLMULQDQ).
#[cfg(target_feature = "vpclmulqdq")]
pub type GhashSquare2x<T> = crate::arch::x86_64::arithmetic::ghash::GhashClMulSquare<T>;
#[cfg(not(target_feature = "vpclmulqdq"))]
pub type GhashSquare2x<T> = crate::arch::Divide<M128, T>;

/// Invert strategy for the `PackedBinaryGhash2x128b` packing.
pub type GhashInvert2x = Ghash256Strategy;

#[cfg(target_feature = "vpclmulqdq")]
mod vpclmulqdq {
	use super::*;
	use crate::arch::x86_64::arithmetic::ghash::ClMulUnderlier;

	impl ClMulUnderlier for M256 {
		#[inline]
		fn clmulepi64<const IMM8: i32>(a: Self, b: Self) -> Self {
			unsafe { std::arch::x86_64::_mm256_clmulepi64_epi128::<IMM8>(a.into(), b.into()) }
				.into()
		}

		#[inline]
		fn move_64_to_hi(a: Self) -> Self {
			unsafe { std::arch::x86_64::_mm256_slli_si256::<8>(a.into()) }.into()
		}
	}
}

/// Strategy for x86_64 AVX2 GHASH field arithmetic operations.
pub struct Ghash256Strategy;

// Implement TaggedInvertOrZero for Ghash256Strategy (Itoh-Tsujii over the full 256-bit vector)
impl TaggedInvertOrZero<Ghash256Strategy> for PackedPrimitiveType<M256, BinaryField128bGhash> {
	#[inline]
	fn invert_or_zero(self) -> Self {
		crate::arch::invert_b128(self)
	}
}
