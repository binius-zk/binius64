// Copyright 2024-2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use super::{
	m128::M128,
	simd_arithmetic::{
		WideAes16x8bProduct, packed_aes_16x8b_invert_or_zero, packed_aes_16x8b_reduce,
		packed_aes_16x8b_square, packed_aes_16x8b_wide_mul,
	},
};
use crate::arch::PackedAesArithmetic;

impl PackedAesArithmetic for M128 {
	type WideProduct = WideAes16x8bProduct;

	#[inline]
	fn wide_mul(a: Self, b: Self) -> Self::WideProduct {
		packed_aes_16x8b_wide_mul(a, b)
	}

	#[inline]
	fn reduce(wide: Self::WideProduct) -> Self {
		packed_aes_16x8b_reduce(wide)
	}

	#[inline]
	fn square(a: Self) -> Self {
		packed_aes_16x8b_square(a)
	}

	#[inline]
	fn invert_or_zero(a: Self) -> Self {
		packed_aes_16x8b_invert_or_zero(a)
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
