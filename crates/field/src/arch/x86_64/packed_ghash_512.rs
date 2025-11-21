// Copyright 2024-2025 Irreducible Inc.

//! VPCLMULQDQ-accelerated implementation of GHASH for x86_64 AVX-512.
//!
//! This module provides optimized GHASH multiplication using the VPCLMULQDQ instruction
//! available on modern x86_64 processors with AVX-512 support. The implementation follows
//! the algorithm described in the GHASH specification with polynomial x^128 + x^7 + x^2 + x + 1.

use std::ops::Mul;

use cfg_if::cfg_if;

use super::{super::portable::packed::PackedPrimitiveType, m512::M512};
use crate::{
	BinaryField128bGhash,
	arch::portable::packed_macros::impl_serialize_deserialize_for_packed_binary_field,
	arithmetic_traits::InvertOrZero, packed::PackedField, underlier::UnderlierWithBitOps,
};

#[cfg(all(target_feature = "vpclmulqdq", target_feature = "avx512f"))]
impl crate::arch::shared::ghash::ClMulUnderlier for M512 {
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

pub type PackedBinaryGhash4x128b = PackedPrimitiveType<M512, BinaryField128bGhash>;

// Define multiply
cfg_if! {
	if #[cfg(all(target_feature = "vpclmulqdq", target_feature = "avx512f"))] {
		impl Mul for PackedBinaryGhash4x128b {
			type Output = Self;

			#[inline]
			fn mul(self, rhs: Self) -> Self::Output {
				crate::tracing::trace_multiplication!(PackedBinaryGhash4x128b);

				Self::from_underlier(crate::arch::shared::ghash::mul_clmul(
					self.to_underlier(),
					rhs.to_underlier(),
				))
			}
		}
	} else {
		impl Mul for PackedBinaryGhash4x128b {
			type Output = Self;

			#[inline]
			fn mul(self, rhs: Self) -> Self::Output {
				crate::tracing::trace_multiplication!(PackedBinaryGhash4x128b);

				// Fallback: perform scalar multiplication on each 128-bit element
				let mut result_underlier = self.to_underlier();
				unsafe {
					let self_0 = self.to_underlier().get_subvalue::<u128>(0);
					let self_1 = self.to_underlier().get_subvalue::<u128>(1);
					let self_2 = self.to_underlier().get_subvalue::<u128>(2);
					let self_3 = self.to_underlier().get_subvalue::<u128>(3);
					let rhs_0 = rhs.to_underlier().get_subvalue::<u128>(0);
					let rhs_1 = rhs.to_underlier().get_subvalue::<u128>(1);
					let rhs_2 = rhs.to_underlier().get_subvalue::<u128>(2);
					let rhs_3 = rhs.to_underlier().get_subvalue::<u128>(3);

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
					let result_2 = Mul::mul(
						PortablePackedBinaryGhash1x128b::from(self_2),
						PortablePackedBinaryGhash1x128b::from(rhs_2),
					);
					let result_3 = Mul::mul(
						PortablePackedBinaryGhash1x128b::from(self_3),
						PortablePackedBinaryGhash1x128b::from(rhs_3),
					);

					result_underlier.set_subvalue(0, result_0.to_underlier());
					result_underlier.set_subvalue(1, result_1.to_underlier());
					result_underlier.set_subvalue(2, result_2.to_underlier());
					result_underlier.set_subvalue(3, result_3.to_underlier());
				}

				Self::from_underlier(result_underlier)
			}
		}
	}
}

// Define square
cfg_if! {
	if #[cfg(all(target_feature = "vpclmulqdq", target_feature = "avx512f"))] {
		impl crate::arithmetic_traits::Square for PackedBinaryGhash4x128b {
			#[inline]
			fn square(self) -> Self {
				Self::from_underlier(crate::arch::shared::ghash::square_clmul(
					self.to_underlier(),
				))
			}
		}
	} else {
		// Potentially we could  use an optimized square implementation here with a scaled underlier.
		// But this case (an architecture with AVX512 but without VPCLMULQDQ) is pretty rare, doesn't worth spending time on it.
		crate::arithmetic_traits::impl_square_with!(PackedBinaryGhash4x128b @ crate::arch::ReuseMultiplyStrategy);
	}
}

// Define invert
impl InvertOrZero for PackedBinaryGhash4x128b {
	fn invert_or_zero(self) -> Self {
		// Fallback: perform scalar invert on each 128-bit element
		let mut result_underlier = self.to_underlier();
		unsafe {
			let self_0 = self.to_underlier().get_subvalue::<u128>(0);
			let self_1 = self.to_underlier().get_subvalue::<u128>(1);
			let self_2 = self.to_underlier().get_subvalue::<u128>(2);
			let self_3 = self.to_underlier().get_subvalue::<u128>(3);

			// Use the portable scalar invert for each element
			use super::super::portable::packed_ghash_128::PackedBinaryGhash1x128b as PortablePackedBinaryGhash1x128b;
			let result_0 =
				PackedField::invert_or_zero(PortablePackedBinaryGhash1x128b::from(self_0));
			let result_1 =
				PackedField::invert_or_zero(PortablePackedBinaryGhash1x128b::from(self_1));
			let result_2 =
				PackedField::invert_or_zero(PortablePackedBinaryGhash1x128b::from(self_2));
			let result_3 =
				PackedField::invert_or_zero(PortablePackedBinaryGhash1x128b::from(self_3));

			result_underlier.set_subvalue(0, result_0.to_underlier());
			result_underlier.set_subvalue(1, result_1.to_underlier());
			result_underlier.set_subvalue(2, result_2.to_underlier());
			result_underlier.set_subvalue(3, result_3.to_underlier());
		}

		Self::from_underlier(result_underlier)
	}
}

// Define linear transformations
cfg_if! {
	if #[cfg(target_feature = "gfni")] {
		use crate::arch::x86_64::gfni::gfni_arithmetics::impl_transformation_with_gfni_nxn;
		impl_transformation_with_gfni_nxn!(PackedBinaryGhash4x128b, 16);
	} else {
		crate::arithmetic_traits::impl_transformation_with_strategy!(
			PackedBinaryGhash4x128b,
			crate::arch::SimdStrategy
		);
	}
}

// Define (de)serialize
impl_serialize_deserialize_for_packed_binary_field!(PackedBinaryGhash4x128b);
