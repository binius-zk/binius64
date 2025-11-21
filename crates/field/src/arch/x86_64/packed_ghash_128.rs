// Copyright 2024-2025 Irreducible Inc.

//! PCLMULQDQ-accelerated implementation of GHASH for x86_64.
//!
//! This module provides optimized GHASH multiplication using the PCLMULQDQ instruction
//! available on modern x86_64 processors. The implementation follows the algorithm
//! described in the GHASH specification with polynomial x^128 + x^7 + x^2 + x + 1.

use std::ops::Mul;

use cfg_if::cfg_if;

use super::{super::portable::packed::PackedPrimitiveType, m128::M128};
use crate::{
	BinaryField128bGhash,
	arch::portable::packed_macros::impl_serialize_deserialize_for_packed_binary_field,
	arithmetic_traits::{InvertOrZero, Square},
	packed::PackedField,
};

#[cfg(target_feature = "pclmulqdq")]
impl crate::arch::shared::ghash::ClMulUnderlier for M128 {
	#[inline]
	fn clmulepi64<const IMM8: i32>(a: Self, b: Self) -> Self {
		unsafe { std::arch::x86_64::_mm_clmulepi64_si128::<IMM8>(a.into(), b.into()) }.into()
	}

	#[inline]
	fn move_64_to_hi(a: Self) -> Self {
		unsafe { std::arch::x86_64::_mm_slli_si128::<8>(a.into()) }.into()
	}
}

pub type PackedBinaryGhash1x128b = PackedPrimitiveType<M128, BinaryField128bGhash>;

// Define multiply
cfg_if! {
	if #[cfg(target_feature = "pclmulqdq")] {
		impl Mul for PackedBinaryGhash1x128b {
			type Output = Self;

			#[inline]
			fn mul(self, rhs: Self) -> Self::Output {
				crate::tracing::trace_multiplication!(PackedBinaryGhash1x128b);

				Self::from_underlier(crate::arch::shared::ghash::mul_clmul(
					self.to_underlier(),
					rhs.to_underlier(),
				))
			}
		}

		impl Square for PackedBinaryGhash1x128b {
			#[inline]
			fn square(self) -> Self {
				Self::from_underlier(crate::arch::shared::ghash::square_clmul(
					self.to_underlier(),
				))
			}
		}

	} else {
		impl Mul for PackedBinaryGhash1x128b {
			type Output = Self;

			#[inline]
			fn mul(self, rhs: Self) -> Self::Output {
				use super::super::portable::packed_ghash_128::PackedBinaryGhash1x128b as PortablePackedBinaryGhash1x128b;

				crate::tracing::trace_multiplication!(PackedBinaryGhash1x128b);

				let portable_lhs = PortablePackedBinaryGhash1x128b::from(u128::from(self.to_underlier()));
				let portable_rhs = PortablePackedBinaryGhash1x128b::from(u128::from(rhs.to_underlier()));

				Self::from_underlier(Mul::mul(portable_lhs, portable_rhs).to_underlier().into())
			}
		}

		impl Square for PackedBinaryGhash1x128b {
			#[inline]
			fn square(self) -> Self {
				use super::super::portable::packed_ghash_128::PackedBinaryGhash1x128b as PortablePackedBinaryGhash1x128b;

				let portable_val = PortablePackedBinaryGhash1x128b::from(u128::from(self.to_underlier()));

				Self::from_underlier(Square::square(portable_val).to_underlier().into())
			}
		}
	}
}

// Define invert
impl InvertOrZero for PackedBinaryGhash1x128b {
	fn invert_or_zero(self) -> Self {
		let portable = super::super::portable::packed_ghash_128::PackedBinaryGhash1x128b::from(
			u128::from(self.to_underlier()),
		);

		Self::from_underlier(PackedField::invert_or_zero(portable).to_underlier().into())
	}
}

cfg_if! {
	if #[cfg(target_feature = "gfni")] {
		use crate::arch::x86_64::gfni::gfni_arithmetics::impl_transformation_with_gfni_nxn;
		impl_transformation_with_gfni_nxn!(PackedBinaryGhash1x128b, 16);
	} else {
		crate::arithmetic_traits::impl_transformation_with_strategy!(
			PackedBinaryGhash1x128b,
			crate::arch::SimdStrategy
		);
	}
}

// Define (de)serialize
impl_serialize_deserialize_for_packed_binary_field!(PackedBinaryGhash1x128b);
