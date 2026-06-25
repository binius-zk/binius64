// Copyright 2024-2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use super::{
	m128::M128,
	simd_arithmetic::{VmullWideMul, packed_aes_16x8b_invert_or_zero, packed_aes_16x8b_square},
};
use crate::{
	aes_field::AESTowerField8b,
	arch::PackedPrimitiveType,
	arithmetic_traits::{TaggedInvertOrZero, TaggedSquare},
	underlier::WithUnderlier,
};

/// Widening-multiply wrapper used by the AES packing: the `vmull_p8`-backed `VmullWideMul`.
pub type AesWideMul16x<T> = VmullWideMul<T>;

/// Square strategy for the `PackedAESBinaryField16x8b` packing.
pub type AesSquare16x = AesStrategy;

/// Invert strategy for the `PackedAESBinaryField16x8b` packing.
pub type AesInvert16x = AesStrategy;

/// Strategy for aarch64 AES square/invert, both backed by `vqtbl` lookup tables.
pub struct AesStrategy;

impl TaggedSquare<AesStrategy> for PackedPrimitiveType<M128, AESTowerField8b> {
	#[inline]
	fn square(self) -> Self {
		self.mutate_underlier(packed_aes_16x8b_square)
	}
}

impl TaggedInvertOrZero<AesStrategy> for PackedPrimitiveType<M128, AESTowerField8b> {
	#[inline]
	fn invert_or_zero(self) -> Self {
		self.mutate_underlier(packed_aes_16x8b_invert_or_zero)
	}
}

#[cfg(test)]
mod tests {
	use proptest::prelude::*;

	use crate::{Divisible, arithmetic_traits::Square};

	proptest! {
		#[test]
		fn test_square_equals_self_mul_self(a_val in any::<u128>()) {
			let a = crate::PackedAESBinaryField16x8b::from_underlier(a_val.into());

			let squared = Square::square(a);

			for i in 0..crate::PackedAESBinaryField16x8b::WIDTH {
				assert_eq!(squared.get(i), a.get(i) * a.get(i));
			}
		}
	}
}
