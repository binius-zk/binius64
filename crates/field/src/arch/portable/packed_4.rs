// Copyright 2024-2025 Irreducible Inc.

use super::packed::PackedPrimitiveType;
use crate::{
	arch::portable::packed_macros::{portable_macros::*, *},
	underlier::{U4, UnderlierType},
};

define_packed_binary_fields!(
	underlier: U4,
	packed_fields: [
		packed_field {
			name: PackedBinaryField4x1b,
			scalar: BinaryField1b,
			alpha_idx: _,
			mul: (None),
			square: (None),
			invert: (None),
			mul_alpha: (None),
			transform: (PackedStrategy),
		},
	]
);

// Define operations for height 0
impl_ops_for_zero_height!(PackedBinaryField4x1b);
