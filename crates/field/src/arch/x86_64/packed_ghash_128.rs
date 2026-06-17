// Copyright 2024-2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

//! PCLMULQDQ-accelerated implementation of GHASH for x86_64.
//!
//! This module provides optimized GHASH multiplication using the PCLMULQDQ instruction
//! available on modern x86_64 processors. The implementation follows the algorithm
//! described in the GHASH specification with polynomial x^128 + x^7 + x^2 + x + 1.

use cfg_if::cfg_if;

use super::m128::M128;
use crate::{
	BinaryField128bGhash,
	arch::portable::packed_macros::{portable_macros::*, *},
	arithmetic_traits::{
		TaggedInvertOrZero, TaggedMul, TaggedSquare, impl_invert_with, impl_mul_with,
		impl_square_with,
	},
};
// Only used by the CLMUL-accelerated `ClMulUnderlier` and `WideMul` impls below.
#[cfg(target_feature = "pclmulqdq")]
use crate::{arch::shared::ghash, arithmetic_traits::WideMul};

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
	(None)
);

// Implement TaggedMul for GhashStrategy
cfg_if! {
	if #[cfg(target_feature = "pclmulqdq")] {
		impl TaggedMul<GhashStrategy> for PackedBinaryGhash1x128b {
			#[inline]
			fn mul(self, rhs: Self) -> Self {
				Self::from_underlier(crate::arch::shared::ghash::mul_clmul(
					self.to_underlier(),
					rhs.to_underlier(),
				))
			}
		}
	} else {
		impl TaggedMul<GhashStrategy> for PackedBinaryGhash1x128b {
			#[inline]
			fn mul(self, rhs: Self) -> Self {
				use super::super::portable::packed_ghash_128::ghash_mul;

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
				Self::from_underlier(crate::arch::shared::ghash::square_clmul(
					self.to_underlier(),
				))
			}
		}
	} else {
		impl TaggedSquare<GhashStrategy> for PackedBinaryGhash1x128b {
			#[inline]
			fn square(self) -> Self {
				use super::super::portable::packed_ghash_128::ghash_square;

				Self::from_underlier(M128::from(ghash_square(u128::from(self.to_underlier()))))
			}
		}
	}
}

// Implement WideMul
cfg_if! {
	if #[cfg(target_feature = "pclmulqdq")] {
		impl WideMul for PackedBinaryGhash1x128b {
			type Output = ghash::WideGhashProduct<M128>;

			#[inline]
			fn wide_mul(a: Self, b: Self) -> Self::Output {
				ghash::WideGhashProduct::wide_mul(a.to_underlier(), b.to_underlier())
			}

			#[inline]
			fn reduce(wide: Self::Output) -> Self {
				Self::from_underlier(wide.reduce())
			}
		}
	} else {
		crate::arithmetic_traits::impl_trivial_wide_mul!(PackedBinaryGhash1x128b);
	}
}

// Implement TaggedInvertOrZero for GhashStrategy (software fallback — no CLMUL invert)
impl TaggedInvertOrZero<GhashStrategy> for PackedBinaryGhash1x128b {
	fn invert_or_zero(self) -> Self {
		use crate::{
			Divisible, arch::portable::packed_ghash_128::ghash_invert_or_zero, packed::PackedField,
		};

		Self::set_single(ghash_invert_or_zero(self.get(0)))
	}
}
