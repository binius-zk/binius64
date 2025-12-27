// Copyright 2024-2025 Irreducible Inc.

use super::{m256::M256, packed_macros::*};
use crate::arch::portable::{packed::PackedPrimitiveType, packed_macros::*};

define_packed_binary_fields!(
	underlier: M256,
	packed_fields: [
		packed_field {
			name: PackedBinaryField256x1b,
			scalar: BinaryField1b,
			mul:       (BitwiseAndStrategy),
			square:    (BitwiseAndStrategy),
			invert:    (BitwiseAndStrategy),
			mul_alpha: (BitwiseAndStrategy),
			transform: (SimdStrategy),
		},
	]
);
