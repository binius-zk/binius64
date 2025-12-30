// Copyright 2024-2025 Irreducible Inc.

use super::m512::M512;
use crate::{
	arch::{
		BitwiseAndStrategy, SimdStrategy,
		portable::{
			packed::PackedPrimitiveType,
			packed_arithmetic::{alphas, impl_tower_constants},
			packed_macros::{portable_macros::*, *},
		},
	},
	arithmetic_traits::impl_transformation_with_strategy,
};

define_packed_binary_fields!(
	underlier: M512,
	packed_fields: [
		packed_field {
			name: PackedBinaryField512x1b,
			scalar: BinaryField1b,
			mul:       (BitwiseAndStrategy),
			square:    (BitwiseAndStrategy),
			invert:    (BitwiseAndStrategy),
			mul_alpha: (BitwiseAndStrategy),
			transform: (SimdStrategy),
		},
	]
);
