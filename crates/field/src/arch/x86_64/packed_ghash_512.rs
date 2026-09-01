// Copyright 2024-2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

//! VPCLMULQDQ-accelerated implementation of GHASH for x86_64 AVX-512.
//!
//! This module provides optimized GHASH multiplication using the VPCLMULQDQ instruction
//! available on modern x86_64 processors with AVX-512 support. The implementation follows
//! the algorithm described in the GHASH specification with polynomial x^128 + x^7 + x^2 + x + 1.
//!
//! Every lane operation below acts within a 128-bit lane, so one body serves all four GHASH lanes.

use super::m512::M512;
use crate::arch::x86_64::arithmetic::ghash::{self, GhashLanes};
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

/// Invert wrapper for the `PackedGhash4x128b` packing: the shared Itoh-Tsujii inversion
/// applied across the full 512-bit vector.
pub type GhashInvert4x<T> = crate::arch::portable::arithmetic::itoh_tsujii::GhashItohTsujii<T>;

impl GhashLanes for M512 {
	#[inline]
	fn move_64_to_hi(a: Self) -> Self {
		// Interleaving the low halves of zero and `a` puts each lane's low half in its high half.
		// Shifting a whole 128-bit lane by 8 bytes would need AVX512BW, which this does not.
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

	#[inline]
	fn broadcast_bit_127(a: Self) -> Self {
		// Bit 127 is the sign bit of each lane's top 32-bit word.
		// Copying that word over the lane, then shifting each word right by 31 as a signed value,
		// leaves every bit of the lane equal to it.
		unsafe {
			let top_word = std::arch::x86_64::_mm512_shuffle_epi32::<0xff>(a.into());
			std::arch::x86_64::_mm512_srai_epi32::<31>(top_word)
		}
		.into()
	}
}

/// Scaling wrapper for the `PackedGhash4x128b` packing: the vector sequence, which needs
/// only AVX-512F.
pub type GhashMulX4x<T> = ghash::GhashMulX<T>;

#[cfg(all(target_feature = "vpclmulqdq", target_feature = "avx512f"))]
impl ghash::ClMulUnderlier for M512 {
	#[inline]
	fn clmulepi64<const IMM8: i32>(a: Self, b: Self) -> Self {
		// Safety: the `vpclmulqdq` + `avx512f` gate on this impl is what the intrinsic requires.
		unsafe { std::arch::x86_64::_mm512_clmulepi64_epi128::<IMM8>(a.into(), b.into()) }.into()
	}
}
