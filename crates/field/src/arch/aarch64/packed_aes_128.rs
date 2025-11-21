// Copyright 2024-2025 Irreducible Inc.

use std::ops::Mul;

use super::{
	m128::M128,
	simd_arithmetic::{
		packed_aes_16x8b_invert_or_zero, packed_aes_16x8b_mul_alpha, packed_aes_16x8b_multiply,
	},
};
use crate::{
	aes_field::AESTowerField8b,
	arch::portable::{packed::PackedPrimitiveType, packed_arithmetic::impl_tower_constants},
	arithmetic_traits::{InvertOrZero, MulAlpha, Square, impl_transformation_with_strategy},
	underlier::WithUnderlier,
};

pub type PackedAESBinaryField16x8b = PackedPrimitiveType<M128, AESTowerField8b>;
impl_tower_constants!(AESTowerField8b, M128, {
	M128::from_u128(0x00d300d300d300d300d300d300d300d3)
});
impl Mul for PackedAESBinaryField16x8b {
	type Output = Self;

	fn mul(self, rhs: Self) -> Self {
		crate::tracing::trace_multiplication!(PackedAESBinaryField16x8b);

		self.mutate_underlier(|underlier| packed_aes_16x8b_multiply(underlier, rhs.to_underlier()))
	}
}
impl Square for PackedAESBinaryField16x8b {
	fn square(self) -> Self {
		self * self
	}
}
impl InvertOrZero for PackedAESBinaryField16x8b {
	fn invert_or_zero(self) -> Self {
		self.mutate_underlier(packed_aes_16x8b_invert_or_zero)
	}
}
impl MulAlpha for PackedAESBinaryField16x8b {
	fn mul_alpha(self) -> Self {
		self.mutate_underlier(packed_aes_16x8b_mul_alpha)
	}
}
impl_transformation_with_strategy!(PackedAESBinaryField16x8b, crate::arch::PackedStrategy);
