// Copyright 2024-2025 Irreducible Inc.

use super::{packed::PackedPrimitiveType, packed_arithmetic::TowerConstants};
use crate::{
	BinaryField1b,
	arch::portable::packed_macros::{portable_macros::*, *},
	arithmetic_traits::impl_transformation_with_strategy,
	underlier::{U2, UnderlierType},
};

define_packed_binary_fields!(
	underlier: U2,
	packed_fields: [
		packed_field {
			name: PackedBinaryField2x1b,
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
impl_ops_for_zero_height!(PackedBinaryField2x1b);

// Define constants
impl TowerConstants<U2> for BinaryField1b {
	const ALPHAS_ODD: U2 = U2::new(<Self as TowerConstants<u8>>::ALPHAS_ODD);
}
