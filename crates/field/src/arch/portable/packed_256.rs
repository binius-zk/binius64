// Copyright 2023-2025 Irreducible Inc.

use crate::{
	arch::{
		M128,
		portable::packed_macros::{portable_macros::*, *},
		strategies::ScaledStrategy,
	},
	underlier::ScaledUnderlier,
};

pub type M256 = ScaledUnderlier<M128, 2>;

pub const fn m256_from_u128s(lo: u128, hi: u128) -> M256 {
	ScaledUnderlier([M128::from_u128(lo), M128::from_u128(hi)])
}

define_packed_binary_fields!(
	underlier: M256,
	packed_fields: [
		packed_field {
			name: PackedBinaryField256x1b,
			scalar: BinaryField1b,
			mul:       (ScaledStrategy),
			square:    (ScaledStrategy),
			invert:    (ScaledStrategy),
		},
	]
);
