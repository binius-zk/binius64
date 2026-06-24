// Copyright 2024-2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use super::{
	m256::M256,
	packed_macros::{portable_macros::*, *},
	scaled_arithmetic::Scaled2xWideMul,
};
use crate::arch::{MulFromWideMul, strategies::ScaledStrategy};

define_packed_binary_fields!(
	underlier: M256,
	packed_fields: [
		packed_field {
			name: PackedAESBinaryField32x8b,
			scalar: AESTowerField8b,
			mul:       (MulFromWideMul),
			square:    (ScaledStrategy),
			invert:    (ScaledStrategy),
			wide_mul: (Scaled2xWideMul),
		},
	]
);
