// Copyright 2024-2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use super::{
	arithmetic::ghash_scaled::Scaled2xGhashWideMul,
	m256::M256,
	packed_macros::{portable_macros::*, *},
};
use crate::arch::strategies::{MulFromWideMul, ScaledStrategy};

define_packed_binary_fields!(
	underlier: M256,
	packed_fields: [
		packed_field {
			name: PackedBinaryGhash2x128b,
			scalar: BinaryField128bGhash,
			mul:       (MulFromWideMul),
			square:    (ScaledStrategy),
			invert:    (ScaledStrategy),
			wide_mul: (Scaled2xGhashWideMul),
		},
	]
);
