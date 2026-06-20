// Copyright 2024-2025 Irreducible Inc.

use super::m256::M256;
use crate::arch::{
	BitwiseAndStrategy,
	portable::packed_macros::{portable_macros::*, *},
};

pub const fn m256_from_u128s(lo: u128, high: u128) -> M256 {
	M256::from_u128s(lo, high)
}

define_packed_binary_fields!(
	underlier: M256,
	packed_fields: [
		packed_field {
			name: PackedBinaryField256x1b,
			scalar: BinaryField1b,
			mul:       (BitwiseAndStrategy),
			square:    (BitwiseAndStrategy),
			invert:    (BitwiseAndStrategy),
			wide_mul: (TrivialWideMul),
		},
	]
);
