// Copyright 2024-2025 Irreducible Inc.

//! VPCLMULQDQ-accelerated implementation of GHASH for x86_64 AVX2.
//!
//! This module provides optimized GHASH multiplication using the VPCLMULQDQ instruction
//! available on modern x86_64 processors with AVX2 support. The implementation follows
//! the algorithm described in the GHASH specification with polynomial x^128 + x^7 + x^2 + x + 1.

use std::ops::Mul;

use cfg_if::cfg_if;

use super::{super::portable::packed::PackedPrimitiveType, m256::M256};
use crate::{
	BinaryField128bGhash,
	arch::portable::packed_macros::impl_serialize_deserialize_for_packed_binary_field,
	arithmetic_traits::InvertOrZero, packed::PackedField, underlier::UnderlierWithBitOps,
};

#[cfg(all(target_feature = "vpclmulqdq", target_feature = "avx2"))]
impl crate::arch::shared::ghash::ClMulUnderlier for M256 {
	#[inline]
	fn clmulepi64<const IMM8: i32>(a: Self, b: Self) -> Self {
		unsafe { std::arch::x86_64::_mm256_clmulepi64_epi128::<IMM8>(a.into(), b.into()) }.into()
	}

	#[inline]
	fn move_64_to_hi(a: Self) -> Self {
		unsafe { std::arch::x86_64::_mm256_slli_si256::<8>(a.into()) }.into()
	}
}

pub type PackedBinaryGhash2x128b = PackedPrimitiveType<M256, BinaryField128bGhash>;

// Define multiply using if_cfg!
cfg_if! {
	if #[cfg(all(target_feature = "vpclmulqdq", target_feature = "avx2"))] {
		impl Mul for PackedBinaryGhash2x128b {
			type Output = Self;

			#[inline]
			fn mul(self, rhs: Self) -> Self::Output {
				crate::tracing::trace_multiplication!(PackedBinaryGhash2x128b);

				Self::from_underlier(crate::arch::shared::ghash::mul_clmul(
					self.to_underlier(),
					rhs.to_underlier(),
				))
			}
		}
	} else {
		impl Mul for PackedBinaryGhash2x128b {
			type Output = Self;

			#[inline]
			fn mul(self, rhs: Self) -> Self::Output {
				crate::tracing::trace_multiplication!(PackedBinaryGhash2x128b);

				// Fallback: perform scalar multiplication on each 128-bit element
				let mut result_underlier = self.to_underlier();
				unsafe {
					let self_0 = self.to_underlier().get_subvalue::<u128>(0);
					let self_1 = self.to_underlier().get_subvalue::<u128>(1);
					let rhs_0 = rhs.to_underlier().get_subvalue::<u128>(0);
					let rhs_1 = rhs.to_underlier().get_subvalue::<u128>(1);

					// Use the portable scalar multiplication for each element
					use super::super::portable::packed_ghash_128::PackedBinaryGhash1x128b as PortablePackedBinaryGhash1x128b;
					let result_0 = Mul::mul(
						PortablePackedBinaryGhash1x128b::from(self_0),
						PortablePackedBinaryGhash1x128b::from(rhs_0),
					);
					let result_1 = Mul::mul(
						PortablePackedBinaryGhash1x128b::from(self_1),
						PortablePackedBinaryGhash1x128b::from(rhs_1),
					);

					result_underlier.set_subvalue(0, result_0.to_underlier());
					result_underlier.set_subvalue(1, result_1.to_underlier());
				}

				Self::from_underlier(result_underlier)
			}
		}
	}
}

// Define square
cfg_if! {
	if #[cfg(all(target_feature = "vpclmulqdq", target_feature = "avx2"))] {
		impl crate::arithmetic_traits::Square for PackedBinaryGhash2x128b {
			#[inline]
			fn square(self) -> Self {
				Self::from_underlier(crate::arch::shared::ghash::square_clmul(
					self.to_underlier(),
				))
			}
		}
	} else {
		// Potentially we could  use an optimized square implementation here with a scaled underlier.
		// But this case (an architecture with AVX2 but without VPCLMULQDQ) is pretty rare, doesn't worth spending time on it.
		crate::arithmetic_traits::impl_square_with!(PackedBinaryGhash2x128b @ crate::arch::ReuseMultiplyStrategy);
	}
}

// Define invert
impl InvertOrZero for PackedBinaryGhash2x128b {
	fn invert_or_zero(self) -> Self {
		// Fallback: perform scalar invert on each 128-bit element
		let mut result_underlier = self.to_underlier();
		unsafe {
			let self_0 = self.to_underlier().get_subvalue::<u128>(0);
			let self_1 = self.to_underlier().get_subvalue::<u128>(1);

			// Use the portable scalar invert for each element
			use super::super::portable::packed_ghash_128::PackedBinaryGhash1x128b as PortablePackedBinaryGhash1x128b;
			let result_0 =
				PackedField::invert_or_zero(PortablePackedBinaryGhash1x128b::from(self_0));
			let result_1 =
				PackedField::invert_or_zero(PortablePackedBinaryGhash1x128b::from(self_1));

			result_underlier.set_subvalue(0, result_0.to_underlier());
			result_underlier.set_subvalue(1, result_1.to_underlier());
		}

		Self::from_underlier(result_underlier)
	}
}

// Define linear transformations
cfg_if! {
	if #[cfg(all(target_feature = "gfni", target_feature = "avx2"))] {
		use crate::arch::x86_64::gfni::gfni_arithmetics::impl_transformation_with_gfni_nxn;
		impl_transformation_with_gfni_nxn!(PackedBinaryGhash2x128b, 16);
	} else {
		crate::arithmetic_traits::impl_transformation_with_strategy!(
			PackedBinaryGhash2x128b,
			crate::arch::SimdStrategy
		);
	}
}

// Define (de)serialize
impl_serialize_deserialize_for_packed_binary_field!(PackedBinaryGhash2x128b);
