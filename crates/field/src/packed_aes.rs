// Copyright 2024-2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use crate::{
	aes_field::AESTowerField8b,
	arch::{
		AesInvert1x, AesInvert16x, AesInvert32x, AesInvert64x, AesSquare1x, AesSquare16x,
		AesSquare32x, AesSquare64x, AesWideMul1x, AesWideMul16x, AesWideMul32x, AesWideMul64x,
		M128, M256, M512, MulFromWideMul,
		portable::packed_macros::{portable_macros::*, *},
	},
};

define_packed_binary_field!(
	PackedAESBinaryField1x8b,
	AESTowerField8b,
	u8,
	(MulFromWideMul),
	(AesSquare1x),
	(AesInvert1x),
	(AesWideMul1x)
);
define_packed_binary_field!(
	PackedAESBinaryField16x8b,
	AESTowerField8b,
	M128,
	(MulFromWideMul),
	(AesSquare16x),
	(AesInvert16x),
	(AesWideMul16x)
);
define_packed_binary_field!(
	PackedAESBinaryField32x8b,
	AESTowerField8b,
	M256,
	(MulFromWideMul),
	(AesSquare32x),
	(AesInvert32x),
	(AesWideMul32x)
);
define_packed_binary_field!(
	PackedAESBinaryField64x8b,
	AESTowerField8b,
	M512,
	(MulFromWideMul),
	(AesSquare64x),
	(AesInvert64x),
	(AesWideMul64x)
);

#[cfg(test)]
mod tests {
	use super::*;
	use crate::{WideMul, packed_binary_field::test_utils::packed_field_tests};

	packed_field_tests!(aes_1x8b, PackedAESBinaryField1x8b);
	packed_field_tests!(aes_16x8b, PackedAESBinaryField16x8b);
	packed_field_tests!(aes_32x8b, PackedAESBinaryField32x8b);
	packed_field_tests!(aes_64x8b, PackedAESBinaryField64x8b);

	#[test]
	fn test_wide_mul_exhaustive_scalar_pairs() {
		// The scalar field has only 2^8 elements, so every product admits an exhaustive check.
		// Each byte pair is broadcast across the 128-bit packing and multiplied deferred.
		//
		//     reduce(wide_mul(a, b)) must equal the scalar product in every lane.
		//
		// The scalar multiply is an independent oracle: it runs the tower-field log/exp tables,
		// not the packed widening path under test.
		for a in 0..=u8::MAX {
			for b in 0..=u8::MAX {
				let expected = crate::AESTowerField8b::new(a) * crate::AESTowerField8b::new(b);

				let a_packed =
					crate::PackedAESBinaryField16x8b::broadcast(crate::AESTowerField8b::new(a));
				let b_packed =
					crate::PackedAESBinaryField16x8b::broadcast(crate::AESTowerField8b::new(b));
				let reduced = crate::PackedAESBinaryField16x8b::reduce(
					crate::PackedAESBinaryField16x8b::wide_mul(a_packed, b_packed),
				);

				assert_eq!(
					reduced,
					crate::PackedAESBinaryField16x8b::broadcast(expected),
					"a={a:#04x} b={b:#04x}"
				);
			}
		}
	}
}
