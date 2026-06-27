// Copyright 2024-2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

//! VPCLMULQDQ-accelerated implementation of GHASH for x86_64 AVX2.
//!
//! This module provides optimized GHASH multiplication using the VPCLMULQDQ instruction
//! available on modern x86_64 processors with AVX2 support. The implementation follows
//! the algorithm described in the GHASH specification with polynomial x^128 + x^7 + x^2 + x + 1.

// Used by the `GhashWideMul2x` and `GhashSquare2x` fallbacks when VPCLMULQDQ is unavailable.
#[cfg(not(target_feature = "vpclmulqdq"))]
use crate::arch::{Divide, x86_64::m128::M128};

/// Widening-multiply wrapper used by the GHASH packing: the reduction-deferring vectorized
/// `GhashClMulWideMul` when VPCLMULQDQ is available, otherwise divide into two `M128` lanes and
/// apply the width-1 GHASH `WideMul` to each, still deferring reduction per lane.
#[cfg(target_feature = "vpclmulqdq")]
pub type GhashWideMul2x<T> = crate::arch::x86_64::arithmetic::ghash::GhashClMulWideMul<T>;
#[cfg(not(target_feature = "vpclmulqdq"))]
pub type GhashWideMul2x<T> = Divide<M128, T, 2>;

/// Square wrapper for the GHASH packing: a full-width CLMUL square ([`GhashClMul`]) when VPCLMULQDQ
/// is available, otherwise divide into 128-bit lanes and square each (the 1×128b GHASH square uses
/// PCLMULQDQ).
///
/// [`GhashClMul`]: crate::arch::x86_64::arithmetic::ghash::GhashClMul
#[cfg(target_feature = "vpclmulqdq")]
pub type GhashSquare2x<T> = crate::arch::x86_64::arithmetic::ghash::GhashClMul<T>;
#[cfg(not(target_feature = "vpclmulqdq"))]
pub type GhashSquare2x<T> = Divide<M128, T, 2>;

/// Invert wrapper for the `PackedBinaryGhash2x128b` packing: the shared Itoh-Tsujii inversion
/// applied across the full 256-bit vector.
pub type GhashInvert2x<T> = crate::arch::portable::arithmetic::itoh_tsujii::GhashItohTsujii<T>;

#[cfg(target_feature = "vpclmulqdq")]
mod vpclmulqdq {
	use crate::arch::x86_64::{arithmetic::ghash::ClMulUnderlier, m256::M256};

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
