// Copyright 2024-2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

//! VPCLMULQDQ-accelerated implementation of GHASH for x86_64 AVX-512.
//!
//! This module provides optimized GHASH multiplication using the VPCLMULQDQ instruction
//! available on modern x86_64 processors with AVX-512 support. The implementation follows
//! the algorithm described in the GHASH specification with polynomial x^128 + x^7 + x^2 + x + 1.

// Used by the CLMUL-accelerated `ClMulUnderlier` impl, the `GhashClMulWideMul`, and the
// `GhashClMul` square aliases below.
#[cfg(all(target_feature = "vpclmulqdq", target_feature = "avx512f"))]
use super::m512::M512;
#[cfg(all(target_feature = "vpclmulqdq", target_feature = "avx512f"))]
use crate::arch::x86_64::arithmetic::ghash;
// Used by the `GhashWideMul4x` and `GhashSquare4x` fallbacks when VPCLMULQDQ is unavailable.
#[cfg(not(all(target_feature = "vpclmulqdq", target_feature = "avx512f")))]
use crate::arch::{Divide, x86_64::m128::M128};

/// Widening-multiply wrapper used by the GHASH packing: the reduction-deferring vectorized
/// [`GhashClMulWideMul`](ghash::GhashClMulWideMul) when VPCLMULQDQ + AVX-512 are available,
/// otherwise divide into four `M128` lanes and apply the width-1 GHASH `WideMul` to each, still
/// deferring reduction per lane.
#[cfg(all(target_feature = "vpclmulqdq", target_feature = "avx512f"))]
pub type GhashWideMul4x<T> = ghash::GhashClMulWideMul<T>;
#[cfg(not(all(target_feature = "vpclmulqdq", target_feature = "avx512f")))]
pub type GhashWideMul4x<T> = Divide<M128, T, 4>;

/// Square wrapper for the GHASH packing: a full-width CLMUL square
/// ([`GhashClMul`](ghash::GhashClMul)) when VPCLMULQDQ + AVX-512 are available, otherwise divide
/// into 128-bit lanes and square each (the 1×128b GHASH square uses PCLMULQDQ).
#[cfg(all(target_feature = "vpclmulqdq", target_feature = "avx512f"))]
pub type GhashSquare4x<T> = ghash::GhashClMul<T>;
#[cfg(not(all(target_feature = "vpclmulqdq", target_feature = "avx512f")))]
pub type GhashSquare4x<T> = Divide<M128, T, 4>;

/// Invert wrapper for the `PackedBinaryGhash4x128b` packing: the shared Itoh-Tsujii inversion
/// applied across the full 512-bit vector.
pub type GhashInvert4x<T> = crate::arch::portable::arithmetic::itoh_tsujii::GhashItohTsujii<T>;

#[cfg(all(target_feature = "vpclmulqdq", target_feature = "avx512f"))]
impl ghash::ClMulUnderlier for M512 {
	#[inline]
	fn clmulepi64<const IMM8: i32>(a: Self, b: Self) -> Self {
		unsafe { std::arch::x86_64::_mm512_clmulepi64_epi128::<IMM8>(a.into(), b.into()) }.into()
	}

	#[inline]
	fn move_64_to_hi(a: Self) -> Self {
		unsafe {
			std::arch::x86_64::_mm512_unpacklo_epi64(
				std::arch::x86_64::_mm512_setzero_si512(),
				a.into(),
			)
		}
		.into()
	}

	#[inline]
	fn shl_1_epi64(a: Self) -> Self {
		unsafe { std::arch::x86_64::_mm512_slli_epi64::<1>(a.into()) }.into()
	}

	#[inline]
	fn shr_63_epi64(a: Self) -> Self {
		unsafe { std::arch::x86_64::_mm512_srli_epi64::<63>(a.into()) }.into()
	}
}
