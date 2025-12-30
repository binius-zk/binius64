// Copyright 2023-2025 Irreducible Inc.

use super::{packed::PackedPrimitiveType, packed_scaled::packed_scaled_field};
use crate::{
	BinaryField1b, PackedBinaryField128x1b, PackedField,
	arch::{
		M128,
		portable::packed_macros::{portable_macros::*, *},
		strategies::ScaledStrategy,
	},
	underlier::ScaledUnderlier,
};

packed_scaled_field!(PPackedBinaryField256x1b = [PackedBinaryField128x1b; 2]);

pub type M256 = ScaledUnderlier<M128, 2>;

define_packed_binary_fields!(
	underlier: M256,
	packed_fields: [
		packed_field {
			name: PackedBinaryField256x1b,
			scalar: BinaryField1b,
			mul:       (ScaledStrategy),
			square:    (ScaledStrategy),
			invert:    (ScaledStrategy),
			mul_alpha: (ScaledStrategy),
			transform: (ScaledStrategy),
		},
	]
);

fn blah(x: PackedBinaryField256x1b) -> BinaryField1b {
	<PackedBinaryField256x1b as PackedField>::get(&x, 0)
}
