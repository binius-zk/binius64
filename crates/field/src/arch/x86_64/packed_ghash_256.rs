// Copyright 2024-2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

//! VPCLMULQDQ-accelerated implementation of GHASH for x86_64 AVX2.
//!
//! This module provides optimized GHASH multiplication using the VPCLMULQDQ instruction
//! available on modern x86_64 processors with AVX2 support. The implementation follows
//! the algorithm described in the GHASH specification with polynomial x^128 + x^7 + x^2 + x + 1.

use cfg_if::cfg_if;

use crate::{
	BinaryField128bGhash,
	arch::{
		portable::packed_macros::{portable_macros::*, *},
		strategies::GhashMulStrategy,
		x86_64::m256::M256,
	},
	arithmetic_traits::{
		TaggedInvertOrZero, TaggedSquare, impl_invert_with, impl_mul_with, impl_square_with,
	},
};
// Only used by the element-wise square fallback when VPCLMULQDQ is unavailable.
#[cfg(not(target_feature = "vpclmulqdq"))]
use crate::{
	arch::{
		portable::arithmetic::ghash_scaled::Scaled2xGhashWideMul,
		x86_64::{m128::M128, packed_ghash_128::PackedBinaryGhash1x128b},
	},
	underlier::Divisible,
};

/// Widening-multiply wrapper used by the GHASH packing: the reduction-deferring vectorized
/// `GhashClMulWideMul` when VPCLMULQDQ is available, otherwise the per-lane `ScaledGhashWideMul`
/// (which still defers reduction, applying the width-1 GHASH `WideMul` to each 128-bit lane).
#[cfg(target_feature = "vpclmulqdq")]
pub type GhashWideMul<T> = crate::arch::x86_64::arithmetic::ghash::GhashClMulWideMul<T>;
#[cfg(not(target_feature = "vpclmulqdq"))]
pub type GhashWideMul<T> = Scaled2xGhashWideMul<T>;

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

// Define PackedBinaryGhash2x128b using the macro
define_packed_binary_field!(
	PackedBinaryGhash2x128b,
	BinaryField128bGhash,
	M256,
	(GhashMulStrategy),
	(Ghash256Strategy),
	(Ghash256Strategy),
	(GhashWideMul)
);

// Implement TaggedSquare for Ghash256Strategy
cfg_if! {
	if #[cfg(target_feature = "vpclmulqdq")] {
		impl TaggedSquare<Ghash256Strategy> for PackedBinaryGhash2x128b {
			#[inline]
			fn square(self) -> Self {
				Self::from_underlier(crate::arch::x86_64::arithmetic::ghash::square_clmul(self.to_underlier()))
			}
		}
	} else {
		impl TaggedSquare<Ghash256Strategy> for PackedBinaryGhash2x128b {
			#[inline]
			fn square(self) -> Self {
				let mut result_underlier = self.to_underlier();
				unsafe {
					let self_0 = Divisible::<M128>::get_unchecked(&self.to_underlier(), 0);
					let self_1 = Divisible::<M128>::get_unchecked(&self.to_underlier(), 1);

					let result_0 = crate::arithmetic_traits::Square::square(PackedBinaryGhash1x128b::from(self_0));
					let result_1 = crate::arithmetic_traits::Square::square(PackedBinaryGhash1x128b::from(self_1));

					Divisible::<M128>::set_unchecked(&mut result_underlier, 0, result_0.to_underlier());
					Divisible::<M128>::set_unchecked(&mut result_underlier, 1, result_1.to_underlier());
				}

				Self::from_underlier(result_underlier)
			}
		}
	}
}

// Implement TaggedInvertOrZero for Ghash256Strategy (Itoh-Tsujii over the full 256-bit vector)
impl TaggedInvertOrZero<Ghash256Strategy> for PackedBinaryGhash2x128b {
	#[inline]
	fn invert_or_zero(self) -> Self {
		crate::arch::invert_b128(self)
	}
}
