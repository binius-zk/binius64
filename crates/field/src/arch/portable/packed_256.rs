// Copyright 2023-2025 Irreducible Inc.

use super::packed_scaled::packed_scaled_field;
use crate::{PackedBinaryField128x1b, PackedField, BinaryField1b};

use super::packed::PackedPrimitiveType;
use crate::{
	arch::portable::packed_macros::{portable_macros::*, *},
	underlier::ScaledUnderlier,
};

packed_scaled_field!(PackedBinaryField256x1b = [PackedBinaryField128x1b; 2]);

pub type M256 = ScaledUnderlier<u128, 2>;

/*
define_packed_binary_fields!(
	underlier: M256,
	packed_fields: [
		packed_field {
			name: PPackedBinaryField256x1b,
			scalar: BinaryField1b,
			mul:       (ScaledStrategy),
			square:    (ScaledStrategy),
			invert:    (ScaledStrategy),
			mul_alpha: (ScaledStrategy),
			transform: (ScaledStrategy),
		},
	]
);

fn blah(x: PPackedBinaryField256x1b) -> BinaryField1b {
	<PPackedBinaryField256x1b as PackedField>::get(&x, 0)
}
*/
