// Copyright 2024-2025 Irreducible Inc.

//! VPCLMULQDQ-accelerated implementation of GHASH for x86_64 AVX2.
//!
//! This module provides optimized GHASH multiplication using the VPCLMULQDQ instruction
//! available on modern x86_64 processors with AVX2 support. The implementation follows
//! the algorithm described in the GHASH specification with polynomial x^128 + x^7 + x^2 + x + 1.

use std::ops::Mul;

use crate::{
	BinaryField128bGhash,
	arch::{
		portable::{
			packed::PackedPrimitiveType,
			packed_macros::{impl_broadcast, impl_serialize_deserialize_for_packed_binary_field},
		},
		x86_64::{m128::M128, m256::M256, packed_ghash_128::PackedBinaryGhash1x128b},
	},
	arithmetic_traits::{InvertOrZero, Square},
	underlier::UnderlierWithBitOps,
};

pub type PackedBinaryGhash2x128b = PackedPrimitiveType<M256, BinaryField128bGhash>;

// Define broadcast
impl_broadcast!(M256, BinaryField128bGhash);

#[cfg(target_feature = "vpclmulqdq")]
mod vpclmulqdq {
	use super::*;
	use crate::arch::shared::ghash::ClMulUnderlier;

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

	impl Square for PackedBinaryGhash2x128b {
		#[inline]
		fn square(self) -> Self {
			Self::from_underlier(crate::arch::shared::ghash::square_clmul(self.to_underlier()))
		}
	}
}

#[cfg(not(target_feature = "vpclmulqdq"))]
mod no_vpclmuldqd_impls {
	use super::*;

	impl Mul for PackedBinaryGhash2x128b {
		type Output = Self;

		#[inline]
		fn mul(self, rhs: Self) -> Self::Output {
			crate::tracing::trace_multiplication!(PackedBinaryGhash2x128b);

			// Fallback: perform scalar multiplication on each 128-bit element
			let mut result_underlier = self.to_underlier();
			unsafe {
				let self_0 = self.to_underlier().get_subvalue::<M128>(0);
				let self_1 = self.to_underlier().get_subvalue::<M128>(1);
				let rhs_0 = rhs.to_underlier().get_subvalue::<M128>(0);
				let rhs_1 = rhs.to_underlier().get_subvalue::<M128>(1);

				let result_0 = Mul::mul(
					PackedBinaryGhash1x128b::from(self_0),
					PackedBinaryGhash1x128b::from(rhs_0),
				);
				let result_1 = Mul::mul(
					PackedBinaryGhash1x128b::from(self_1),
					PackedBinaryGhash1x128b::from(rhs_1),
				);

				result_underlier.set_subvalue(0, result_0.to_underlier());
				result_underlier.set_subvalue(1, result_1.to_underlier());
			}

			Self::from_underlier(result_underlier)
		}
	}

	impl Square for PackedBinaryGhash2x128b {
		#[inline]
		fn square(self) -> Self {
			let mut result_underlier = self.to_underlier();
			unsafe {
				let self_0 = self.to_underlier().get_subvalue::<M128>(0);
				let self_1 = self.to_underlier().get_subvalue::<M128>(1);

				let result_0 = Square::square(PackedBinaryGhash1x128b::from(self_0));
				let result_1 = Square::square(PackedBinaryGhash1x128b::from(self_1));

				result_underlier.set_subvalue(0, result_0.to_underlier());
				result_underlier.set_subvalue(1, result_1.to_underlier());
			}

			Self::from_underlier(result_underlier)
		}
	}
}

impl InvertOrZero for PackedBinaryGhash2x128b {
	fn invert_or_zero(self) -> Self {
		let mut result_underlier = self.to_underlier();
		unsafe {
			let self_0 = self.to_underlier().get_subvalue::<M128>(0);
			let self_1 = self.to_underlier().get_subvalue::<M128>(1);

			// Use the portable scalar invert for each element
			let result_0 = InvertOrZero::invert_or_zero(PackedBinaryGhash1x128b::from(self_0));
			let result_1 = InvertOrZero::invert_or_zero(PackedBinaryGhash1x128b::from(self_1));

			result_underlier.set_subvalue(0, result_0.to_underlier());
			result_underlier.set_subvalue(1, result_1.to_underlier());
		}

		Self::from_underlier(result_underlier)
	}
}

// Define (de)serialize
impl_serialize_deserialize_for_packed_binary_field!(PackedBinaryGhash2x128b);
