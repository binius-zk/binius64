// Copyright 2024-2025 Irreducible Inc.

use super::{
	packed_256::M256,
	packed_macros::{portable_macros::*, *},
};
use crate::arch::strategies::ScaledStrategy;

define_packed_binary_fields!(
	underlier: M256,
	packed_fields: [
		packed_field {
			name: PackedBinaryGhash2x128b,
			scalar: BinaryField128bGhash,
			mul:       (ScaledStrategy),
			square:    (ScaledStrategy),
			invert:    (ScaledStrategy),
			transform: (None),
		},
	]
);

crate::arithmetic_traits::impl_trivial_widening_mul!(PackedBinaryGhash2x128b);
