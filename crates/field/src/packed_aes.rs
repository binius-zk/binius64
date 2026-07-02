// Copyright 2024-2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use crate::{
	aes_field::AESTowerField8b,
	arch::{M128, M256, M512, PackedPrimitiveType},
};

// Arithmetic is provided by the blanket impls on `PackedPrimitiveType<U, AESTowerField8b>`.

/// A single `AESTowerField8b` byte (a degenerate width-1 packing).
pub type PackedAESBinaryField1x8b = PackedPrimitiveType<u8, AESTowerField8b>;

/// 16 `AESTowerField8b` bytes in a 128-bit underlier.
pub type PackedAESBinaryField16x8b = PackedPrimitiveType<M128, AESTowerField8b>;

/// 32 `AESTowerField8b` bytes in a 256-bit underlier.
pub type PackedAESBinaryField32x8b = PackedPrimitiveType<M256, AESTowerField8b>;

/// 64 `AESTowerField8b` bytes in a 512-bit underlier.
pub type PackedAESBinaryField64x8b = PackedPrimitiveType<M512, AESTowerField8b>;

#[cfg(test)]
mod test_utils {
	/// Test if `mult_func` operation is a valid multiply operation on the given values for
	/// all possible packed fields defined on 8-512 bits.
	macro_rules! define_multiply_tests {
		($mult_func:path, $constraint:ty) => {
			$crate::packed_binary_field::test_utils::define_check_packed_mul!(
				$mult_func,
				$constraint
			);

			proptest! {
				#[test]
				fn test_mul_packed_8(a_val in any::<u8>(), b_val in any::<u8>()) {
					TestMult::<$crate::PackedAESBinaryField1x8b>::test_mul(
						a_val.into(),
						b_val.into(),
					);
				}

				#[test]
				fn test_mul_packed_128(a_val in any::<u128>(), b_val in any::<u128>()) {
					TestMult::<$crate::PackedAESBinaryField16x8b>::test_mul(
						a_val.into(),
						b_val.into(),
					);
				}

				#[test]
				fn test_mul_packed_256(a_val in any::<[u128; 2]>(), b_val in any::<[u128; 2]>()) {
					TestMult::<$crate::PackedAESBinaryField32x8b>::test_mul(
						a_val.into(),
						b_val.into(),
					);
				}

				#[test]
				fn test_mul_packed_512(a_val in any::<[u128; 4]>(), b_val in any::<[u128; 4]>()) {
					TestMult::<$crate::PackedAESBinaryField64x8b>::test_mul(
						a_val.into(),
						b_val.into(),
					);
				}
			}
		};
	}

	/// Test if `square_func` operation is a valid square operation on the given value for
	/// all possible packed fields.
	macro_rules! define_square_tests {
		($square_func:path, $constraint:ident) => {
			$crate::packed_binary_field::test_utils::define_check_packed_square!(
				$square_func,
				$constraint
			);

			proptest! {
				#[test]
				fn test_square_packed_8(a_val in any::<u8>()) {
					TestSquare::<$crate::PackedAESBinaryField1x8b>::test_square(a_val.into());
				}

				#[test]
				fn test_square_packed_128(a_val in any::<u128>()) {
					TestSquare::<$crate::PackedAESBinaryField16x8b>::test_square(a_val.into());
				}

				#[test]
				fn test_square_packed_256(a_val in any::<[u128; 2]>()) {
					TestSquare::<$crate::PackedAESBinaryField32x8b>::test_square(a_val.into());
				}

				#[test]
				fn test_square_packed_512(a_val in any::<[u128; 4]>()) {
					TestSquare::<$crate::PackedAESBinaryField64x8b>::test_square(a_val.into());
				}
			}
		};
	}

	/// Test if `invert_func` operation is a valid invert operation on the given value for
	/// all possible packed fields.
	macro_rules! define_invert_tests {
		($invert_func:path, $constraint:ident) => {
			$crate::packed_binary_field::test_utils::define_check_packed_inverse!(
				$invert_func,
				$constraint
			);

			proptest! {
				#[test]
				fn test_invert_packed_8(a_val in any::<u8>()) {
					TestSquare::<$crate::PackedAESBinaryField1x8b>::test_invert(a_val.into());
				}

				#[test]
				fn test_invert_packed_128(a_val in any::<u128>()) {
					TestInvert::<$crate::PackedAESBinaryField16x8b>::test_invert(a_val.into());
				}

				#[test]
				fn test_invert_packed_256(a_val in any::<[u128; 2]>()) {
					TestInvert::<$crate::PackedAESBinaryField32x8b>::test_invert(a_val.into());
				}

				#[test]
				fn test_invert_packed_512(a_val in any::<[u128; 4]>()) {
					TestInvert::<$crate::PackedAESBinaryField64x8b>::test_invert(a_val.into());
				}
			}
		};
	}

	pub(crate) use define_invert_tests;
	pub(crate) use define_multiply_tests;
	pub(crate) use define_square_tests;
}

#[cfg(test)]
mod tests {
	use std::ops::Mul;

	use proptest::prelude::*;

	use super::test_utils::{define_invert_tests, define_multiply_tests, define_square_tests};
	use crate::{
		PackedField,
		arithmetic_traits::{InvertOrZero, Square},
	};

	define_multiply_tests!(Mul::mul, PackedField);

	define_square_tests!(Square::square, PackedField);

	define_invert_tests!(InvertOrZero::invert_or_zero, PackedField);
}
