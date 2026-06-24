// Copyright 2024-2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use super::{
	m128::M128,
	simd_arithmetic::{VmullWideMul, packed_aes_16x8b_invert_or_zero, packed_aes_16x8b_square},
};
use crate::{
	aes_field::AESTowerField8b,
	arch::{
		portable::packed_macros::{portable_macros::*, *},
		strategies::MulFromWideMul,
	},
	arithmetic_traits::{
		TaggedInvertOrZero, TaggedSquare, impl_invert_with, impl_mul_with, impl_square_with,
	},
	underlier::WithUnderlier,
};

/// Strategy for aarch64 AES square/invert, both backed by `vqtbl` lookup tables.
pub struct AesStrategy;

// Define PackedAESBinaryField16x8b using the macro. `Mul` is derived from `WideMul` (supplied by
// the `VmullWideMul` wrapper, whose `vmull_p8` multiply already produces the reduced byte); square
// and invert use lookup tables via `AesStrategy`.
define_packed_binary_field!(
	PackedAESBinaryField16x8b,
	AESTowerField8b,
	M128,
	(MulFromWideMul),
	(AesStrategy),
	(AesStrategy),
	(VmullWideMul)
);

impl TaggedSquare<AesStrategy> for PackedAESBinaryField16x8b {
	#[inline]
	fn square(self) -> Self {
		self.mutate_underlier(packed_aes_16x8b_square)
	}
}

impl TaggedInvertOrZero<AesStrategy> for PackedAESBinaryField16x8b {
	#[inline]
	fn invert_or_zero(self) -> Self {
		self.mutate_underlier(packed_aes_16x8b_invert_or_zero)
	}
}

#[cfg(test)]
mod tests {
	use proptest::prelude::*;

	use super::*;
	use crate::{Divisible, arithmetic_traits::Square};

	proptest! {
		#[test]
		fn test_square_equals_self_mul_self(a_val in any::<u128>()) {
			let a = PackedAESBinaryField16x8b::from_underlier(a_val.into());

			let squared = Square::square(a);

			for i in 0..PackedAESBinaryField16x8b::WIDTH {
				assert_eq!(squared.get(i), a.get(i) * a.get(i));
			}
		}
	}
}
