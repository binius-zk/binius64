// Copyright 2024-2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

//! VPCLMULQDQ-accelerated implementation of GHASH for x86_64 AVX-512.
//!
//! This module provides optimized GHASH multiplication using the VPCLMULQDQ instruction
//! available on modern x86_64 processors with AVX-512 support. The implementation follows
//! the algorithm described in the GHASH specification with polynomial x^128 + x^7 + x^2 + x + 1.

use super::m512::M512;
// Used by the CLMUL-accelerated `ClMulUnderlier` impl and the `GhashWideMul` alias below.
#[cfg(all(target_feature = "vpclmulqdq", target_feature = "avx512f"))]
use crate::arch::x86_64::arithmetic::ghash;
// Used by the `GhashWideMul` and `GhashSquareStrategy` fallbacks when VPCLMULQDQ is
// unavailable.
#[cfg(not(all(target_feature = "vpclmulqdq", target_feature = "avx512f")))]
use crate::arch::{portable::arithmetic::ghash_scaled::Scaled4xGhashWideMul, x86_64::m128::M128};
use crate::{
	BinaryField128bGhash,
	arch::{
		portable::packed_macros::{portable_macros::*, *},
		strategies::MulFromWideMul,
	},
	arithmetic_traits::{TaggedInvertOrZero, impl_invert_with, impl_mul_with, impl_square_with},
};

/// Widening-multiply wrapper used by the GHASH packing: the reduction-deferring vectorized
/// [`GhashClMulWideMul`](ghash::GhashClMulWideMul) when VPCLMULQDQ + AVX-512 are available,
/// otherwise the per-lane [`ScaledGhashWideMul`] (still deferring, applying the width-1 GHASH
/// `WideMul` to each 128-bit lane).
#[cfg(all(target_feature = "vpclmulqdq", target_feature = "avx512f"))]
pub type GhashWideMul<T> = ghash::GhashClMulWideMul<T>;
#[cfg(not(all(target_feature = "vpclmulqdq", target_feature = "avx512f")))]
pub type GhashWideMul<T> = Scaled4xGhashWideMul<T>;

/// Square strategy for the GHASH packing: a full-width CLMUL square when VPCLMULQDQ + AVX-512 are
/// available, otherwise divide into 128-bit lanes and square each (the 1×128b GHASH square uses
/// PCLMULQDQ).
#[cfg(all(target_feature = "vpclmulqdq", target_feature = "avx512f"))]
pub type GhashSquareStrategy = ghash::GhashClMulSquareStrategy;
#[cfg(not(all(target_feature = "vpclmulqdq", target_feature = "avx512f")))]
pub type GhashSquareStrategy = crate::arch::DivideStrategy<M128>;

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
}

/// Strategy for x86_64 AVX-512 GHASH field arithmetic operations.
pub struct Ghash512Strategy;

// Define PackedBinaryGhash4x128b using the macro
define_packed_binary_field!(
	PackedBinaryGhash4x128b,
	BinaryField128bGhash,
	M512,
	(MulFromWideMul),
	(GhashSquareStrategy),
	(Ghash512Strategy),
	(GhashWideMul)
);

// Implement TaggedInvertOrZero for Ghash512Strategy (Itoh-Tsujii over the full 512-bit vector)
impl TaggedInvertOrZero<Ghash512Strategy> for PackedBinaryGhash4x128b {
	#[inline]
	fn invert_or_zero(self) -> Self {
		crate::arch::invert_b128(self)
	}
}
