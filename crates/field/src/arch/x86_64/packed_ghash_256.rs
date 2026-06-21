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
		x86_64::m256::M256,
	},
	arithmetic_traits::{
		TaggedInvertOrZero, TaggedMul, TaggedSquare, impl_invert_with, impl_mul_with,
		impl_square_with,
	},
};
// Only used by the element-wise fallback when VPCLMULQDQ is unavailable.
#[cfg(not(target_feature = "vpclmulqdq"))]
use crate::{
	arch::x86_64::{m128::M128, packed_ghash_128::PackedBinaryGhash1x128b},
	underlier::Divisible,
};

/// Widening-multiply wrapper used by the GHASH packing: the reduction-deferring
/// `GhashClMulWideMul` when VPCLMULQDQ is available, otherwise an eager `TrivialWideMul`.
#[cfg(target_feature = "vpclmulqdq")]
pub type GhashWideMul<T> = crate::arch::x86_64::arithmetic::ghash::GhashClMulWideMul<T>;
#[cfg(not(target_feature = "vpclmulqdq"))]
pub type GhashWideMul<T> = TrivialWideMul<T>;

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
	(Ghash256Strategy),
	(Ghash256Strategy),
	(Ghash256Strategy),
	(GhashWideMul)
);

// Implement TaggedMul for Ghash256Strategy
cfg_if! {
	if #[cfg(target_feature = "vpclmulqdq")] {
		impl TaggedMul<Ghash256Strategy> for PackedBinaryGhash2x128b {
			#[inline]
			fn mul(self, rhs: Self) -> Self {
				Self::from_underlier(crate::arch::x86_64::arithmetic::ghash::mul_clmul(
					self.to_underlier(),
					rhs.to_underlier(),
				))
			}
		}
	} else {
		impl TaggedMul<Ghash256Strategy> for PackedBinaryGhash2x128b {
			#[inline]
			fn mul(self, rhs: Self) -> Self {
				// Fallback: perform scalar multiplication on each 128-bit element
				let mut result_underlier = self.to_underlier();
				unsafe {
					let self_0 = Divisible::<M128>::get_unchecked(&self.to_underlier(), 0);
					let self_1 = Divisible::<M128>::get_unchecked(&self.to_underlier(), 1);
					let rhs_0 = Divisible::<M128>::get_unchecked(&rhs.to_underlier(), 0);
					let rhs_1 = Divisible::<M128>::get_unchecked(&rhs.to_underlier(), 1);

					let result_0 = std::ops::Mul::mul(
						PackedBinaryGhash1x128b::from(self_0),
						PackedBinaryGhash1x128b::from(rhs_0),
					);
					let result_1 = std::ops::Mul::mul(
						PackedBinaryGhash1x128b::from(self_1),
						PackedBinaryGhash1x128b::from(rhs_1),
					);

					Divisible::<M128>::set_unchecked(&mut result_underlier, 0, result_0.to_underlier());
					Divisible::<M128>::set_unchecked(&mut result_underlier, 1, result_1.to_underlier());
				}

				Self::from_underlier(result_underlier)
			}
		}
	}
}

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
