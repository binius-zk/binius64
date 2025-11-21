// Copyright 2024-2025 Irreducible Inc.

use super::{
	super::portable::{
		packed::PackedPrimitiveType,
		packed_arithmetic::{alphas, impl_tower_constants},
	},
	m128::M128,
	packed_macros::*,
};
use crate::{
	arch::portable::packed_macros::*, arithmetic_traits::impl_transformation_with_strategy,
};

define_packed_binary_fields!(
	underlier: M128,
	packed_fields: [
		packed_field {
			name: PackedBinaryField128x1b,
			scalar: BinaryField1b,
			alpha_idx: 0,
			mul: (None),
			square: (None),
			invert: (None),
			mul_alpha: (None),
			transform: (PackedStrategy),
		},
	]
);
