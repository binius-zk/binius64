// Copyright 2024-2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

//! VPCLMULQDQ-accelerated implementation of GHASH for x86_64 AVX-512.
//!
//! This module provides optimized GHASH multiplication using the VPCLMULQDQ instruction
//! available on modern x86_64 processors with AVX-512 support. The implementation follows
//! the algorithm described in the GHASH specification with polynomial x^128 + x^7 + x^2 + x + 1.

use cfg_if::cfg_if;

use super::m512::M512;
use crate::{
	BinaryField128bGhash,
	arch::portable::packed_macros::{portable_macros::*, *},
	arithmetic_traits::{
		InvertOrZero, TaggedInvertOrZero, TaggedMul, TaggedSquare, impl_invert_with, impl_mul_with,
		impl_square_with,
	},
	underlier::Divisible,
};
// Only used by the CLMUL-accelerated `ClMulUnderlier` and `WideMul` impls below.
#[cfg(all(target_feature = "vpclmulqdq", target_feature = "avx512f"))]
use crate::{arch::shared::ghash, arithmetic_traits::WideMul};

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
	(Ghash512Strategy),
	(Ghash512Strategy),
	(Ghash512Strategy),
	(None)
);

// Implement TaggedMul for Ghash512Strategy
cfg_if! {
	if #[cfg(all(target_feature = "vpclmulqdq", target_feature = "avx512f"))] {
		impl TaggedMul<Ghash512Strategy> for PackedBinaryGhash4x128b {
			#[inline]
			fn mul(self, rhs: Self) -> Self {
				Self::from_underlier(crate::arch::shared::ghash::mul_clmul(
					self.to_underlier(),
					rhs.to_underlier(),
				))
			}
		}
	} else {
		impl TaggedMul<Ghash512Strategy> for PackedBinaryGhash4x128b {
			#[inline]
			fn mul(self, rhs: Self) -> Self {
				// Fallback: perform scalar multiplication on each 128-bit element
				let mut result_underlier = self.to_underlier();
				unsafe {
					let self_0 = Divisible::<u128>::get_unchecked(&self.to_underlier(), 0);
					let self_1 = Divisible::<u128>::get_unchecked(&self.to_underlier(), 1);
					let self_2 = Divisible::<u128>::get_unchecked(&self.to_underlier(), 2);
					let self_3 = Divisible::<u128>::get_unchecked(&self.to_underlier(), 3);
					let rhs_0 = Divisible::<u128>::get_unchecked(&rhs.to_underlier(), 0);
					let rhs_1 = Divisible::<u128>::get_unchecked(&rhs.to_underlier(), 1);
					let rhs_2 = Divisible::<u128>::get_unchecked(&rhs.to_underlier(), 2);
					let rhs_3 = Divisible::<u128>::get_unchecked(&rhs.to_underlier(), 3);

					// Use the portable scalar multiplication for each element
					use super::super::portable::packed_ghash_128::PackedBinaryGhash1x128b as PortablePackedBinaryGhash1x128b;
					let result_0 = std::ops::Mul::mul(
						PortablePackedBinaryGhash1x128b::from(self_0),
						PortablePackedBinaryGhash1x128b::from(rhs_0),
					);
					let result_1 = std::ops::Mul::mul(
						PortablePackedBinaryGhash1x128b::from(self_1),
						PortablePackedBinaryGhash1x128b::from(rhs_1),
					);
					let result_2 = std::ops::Mul::mul(
						PortablePackedBinaryGhash1x128b::from(self_2),
						PortablePackedBinaryGhash1x128b::from(rhs_2),
					);
					let result_3 = std::ops::Mul::mul(
						PortablePackedBinaryGhash1x128b::from(self_3),
						PortablePackedBinaryGhash1x128b::from(rhs_3),
					);

					Divisible::<u128>::set_unchecked(&mut result_underlier, 0, result_0.to_underlier());
					Divisible::<u128>::set_unchecked(&mut result_underlier, 1, result_1.to_underlier());
					Divisible::<u128>::set_unchecked(&mut result_underlier, 2, result_2.to_underlier());
					Divisible::<u128>::set_unchecked(&mut result_underlier, 3, result_3.to_underlier());
				}

				Self::from_underlier(result_underlier)
			}
		}
	}
}

// Implement TaggedSquare for Ghash512Strategy
cfg_if! {
	if #[cfg(all(target_feature = "vpclmulqdq", target_feature = "avx512f"))] {
		impl TaggedSquare<Ghash512Strategy> for PackedBinaryGhash4x128b {
			#[inline]
			fn square(self) -> Self {
				Self::from_underlier(crate::arch::shared::ghash::square_clmul(
					self.to_underlier(),
				))
			}
		}
	} else {
		// Potentially we could use an optimized square implementation here with a scaled underlier.
		// But this case (an architecture with AVX512 but without VPCLMULQDQ) is pretty rare.
		impl_square_with!(PackedBinaryGhash4x128b @ crate::arch::ReuseMultiplyStrategy);
	}
}

// Implement WideMul
cfg_if! {
	if #[cfg(all(target_feature = "vpclmulqdq", target_feature = "avx512f"))] {
		impl WideMul for PackedBinaryGhash4x128b {
			type Output = ghash::WideGhashProduct<M512>;

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
		crate::arithmetic_traits::impl_trivial_wide_mul!(PackedBinaryGhash4x128b);
	}
}

// Implement TaggedInvertOrZero for Ghash512Strategy (always uses element-wise fallback)
impl TaggedInvertOrZero<Ghash512Strategy> for PackedBinaryGhash4x128b {
	fn invert_or_zero(self) -> Self {
		Self::from_iter(Self::value_iter(self).map(InvertOrZero::invert_or_zero))
	}
}
