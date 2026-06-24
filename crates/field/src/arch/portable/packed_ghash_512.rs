// Copyright 2024-2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use super::{
	m512::M512,
	packed_macros::{portable_macros::*, *},
	scaled_arithmetic::Scaled4xWideMul,
};
use crate::arch::strategies::{MulFromWideMul, ScaledStrategy};

define_packed_binary_fields!(
	underlier: M512,
	packed_fields: [
		packed_field {
			name: PackedBinaryGhash4x128b,
			scalar: BinaryField128bGhash,
			mul:       (MulFromWideMul),
			square:    (ScaledStrategy),
			invert:    (ScaledStrategy),
			wide_mul: (Scaled4xWideMul),
		},
	]
);
