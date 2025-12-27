// Copyright 2024-2025 Irreducible Inc.

use super::{super::portable::packed::PackedPrimitiveType, m128::M128, packed_macros::*};
use crate::arch::portable::packed_macros::*;

define_packed_binary_fields!(
	underlier: M128,
	packed_fields: [
		packed_field {
			name: PackedBinaryField128x1b,
			scalar: BinaryField1b,
			mul: (BitwiseAndStrategy),
			square: (BitwiseAndStrategy),
			invert: (BitwiseAndStrategy),
			mul_alpha: (BitwiseAndStrategy),
			transform: (PackedStrategy),
		},
	]
);
