// Copyright 2024-2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use bytemuck::TransparentWrapper;

use super::{
	m128::M128,
	simd_arithmetic::{VmullWideMul, packed_aes_16x8b_invert_or_zero, packed_aes_16x8b_square},
};
use crate::{
	aes_field::AESTowerField8b,
	arch::PackedPrimitiveType,
	arithmetic_traits::{InvertOrZero, Square},
	underlier::WithUnderlier,
};

/// Widening-multiply wrapper used by the AES packing: the `vmull_p8`-backed `VmullWideMul`.
pub type AesWideMul16x<T> = VmullWideMul<T>;

/// Square wrapper for the `PackedAESBinaryField16x8b` packing.
pub type AesSquare16x<T> = NeonTableLookupArithmetic<T>;

/// Invert wrapper for the `PackedAESBinaryField16x8b` packing.
pub type AesInvert16x<T> = NeonTableLookupArithmetic<T>;

/// Square and invert strategy wrapper for aarch64 AES, backed by `vqtbl` table lookups over the
/// 16-byte `M128` vector.
#[repr(transparent)]
#[derive(TransparentWrapper)]
pub struct NeonTableLookupArithmetic<T>(T);

impl Square for NeonTableLookupArithmetic<PackedPrimitiveType<M128, AESTowerField8b>> {
	#[inline]
	fn square(self) -> Self {
		Self::wrap(Self::peel(self).mutate_underlier(packed_aes_16x8b_square))
	}
}

impl InvertOrZero for NeonTableLookupArithmetic<PackedPrimitiveType<M128, AESTowerField8b>> {
	#[inline]
	fn invert_or_zero(self) -> Self {
		Self::wrap(Self::peel(self).mutate_underlier(packed_aes_16x8b_invert_or_zero))
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
