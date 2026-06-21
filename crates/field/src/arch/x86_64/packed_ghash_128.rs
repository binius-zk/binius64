// Copyright 2024-2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

//! PCLMULQDQ-accelerated implementation of GHASH for x86_64.
//!
//! This module provides optimized GHASH multiplication using the PCLMULQDQ instruction
//! available on modern x86_64 processors. The implementation follows the algorithm
//! described in the GHASH specification with polynomial x^128 + x^7 + x^2 + x + 1.

use cfg_if::cfg_if;

use super::m128::M128;
// Used by the CLMUL-accelerated `ClMulUnderlier` impl and the `GhashWideMul` alias below.
#[cfg(target_feature = "pclmulqdq")]
use crate::arch::x86_64::arithmetic::ghash;
use crate::{
	BinaryField128bGhash,
	arch::portable::packed_macros::{portable_macros::*, *},
	arithmetic_traits::{
		TaggedInvertOrZero, TaggedMul, TaggedSquare, impl_invert_with, impl_mul_with,
		impl_square_with,
	},
};

/// Widening-multiply wrapper used by the GHASH packing: the reduction-deferring
/// [`GhashClMulWideMul`](ghash::GhashClMulWideMul) when PCLMULQDQ is available, otherwise an eager
/// [`TrivialWideMul`].
#[cfg(target_feature = "pclmulqdq")]
pub type GhashWideMul<T> = ghash::GhashClMulWideMul<T>;
#[cfg(not(target_feature = "pclmulqdq"))]
pub type GhashWideMul<T> = TrivialWideMul<T>;

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

/// Strategy for x86_64 GHASH field arithmetic operations.
pub struct GhashStrategy;

// Define PackedBinaryGhash1x128b using the macro
define_packed_binary_field!(
	PackedBinaryGhash1x128b,
	BinaryField128bGhash,
	M128,
	(GhashStrategy),
	(GhashStrategy),
	(GhashStrategy),
	(GhashWideMul)
);

// Implement TaggedMul for GhashStrategy
cfg_if! {
	if #[cfg(target_feature = "pclmulqdq")] {
		impl TaggedMul<GhashStrategy> for PackedBinaryGhash1x128b {
			#[inline]
			fn mul(self, rhs: Self) -> Self {
				Self::from_underlier(crate::arch::x86_64::arithmetic::ghash::mul_clmul(
					self.to_underlier(),
					rhs.to_underlier(),
				))
			}
		}
	} else {
		impl TaggedMul<GhashStrategy> for PackedBinaryGhash1x128b {
			#[inline]
			fn mul(self, rhs: Self) -> Self {
				use super::super::portable::arithmetic::ghash::ghash_mul;

				let product = ghash_mul(u128::from(self.to_underlier()), u128::from(rhs.to_underlier()));
				Self::from_underlier(M128::from(product))
			}
		}
	}
}

// Implement TaggedSquare for GhashStrategy
cfg_if! {
	if #[cfg(target_feature = "pclmulqdq")] {
		impl TaggedSquare<GhashStrategy> for PackedBinaryGhash1x128b {
			#[inline]
			fn square(self) -> Self {
				Self::from_underlier(crate::arch::x86_64::arithmetic::ghash::square_clmul(
					self.to_underlier(),
				))
			}
		}
	} else {
		impl TaggedSquare<GhashStrategy> for PackedBinaryGhash1x128b {
			#[inline]
			fn square(self) -> Self {
				use super::super::portable::arithmetic::ghash::ghash_square;

				Self::from_underlier(M128::from(ghash_square(u128::from(self.to_underlier()))))
			}
		}
	}
}

// Implement TaggedInvertOrZero for GhashStrategy (Itoh-Tsujii — no CLMUL invert)
impl TaggedInvertOrZero<GhashStrategy> for PackedBinaryGhash1x128b {
	#[inline]
	fn invert_or_zero(self) -> Self {
		crate::arch::invert_b128(self)
	}
}
