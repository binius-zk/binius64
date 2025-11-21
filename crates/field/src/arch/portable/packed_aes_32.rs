// Copyright 2024-2025 Irreducible Inc.

use cfg_if::cfg_if;

use super::{
	packed::PackedPrimitiveType, packed_arithmetic::impl_tower_constants,
	packed_macros::impl_broadcast,
};
use crate::{
	AESTowerField8b,
	arch::portable::packed_macros::{portable_macros::*, *},
	arithmetic_traits::{
		impl_invert_with, impl_mul_alpha_with, impl_mul_with, impl_square_with,
		impl_transformation_with_strategy,
	},
};

define_packed_binary_fields!(
	underlier: u32,
	packed_fields: [
		packed_field {
			name: PackedAESBinaryField4x8b,
			scalar: AESTowerField8b,
			alpha_idx: _,
			mul:       (if gfni_x86 PackedAESBinaryField16x8b else PairwiseTableStrategy),
			square:    (PairwiseTableStrategy),
			invert:    (if gfni_x86 PackedAESBinaryField16x8b else PairwiseTableStrategy),
			mul_alpha: (PairwiseTableStrategy),
			transform: (PackedStrategy),
		},
	]
);

impl_tower_constants!(AESTowerField8b, u32, 0x00d300d3);
