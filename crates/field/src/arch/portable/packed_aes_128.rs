// Copyright 2024-2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use super::m128::M128;
use crate::{
	arch::{
		PairwiseTableStrategy,
		portable::packed_macros::{portable_macros::*, *},
	},
	arithmetic_traits::{impl_invert_with, impl_mul_with, impl_square_with},
};

define_packed_binary_fields!(
	underlier: M128,
	packed_fields: [
		packed_field {
			name: PackedAESBinaryField16x8b,
			scalar: AESTowerField8b,
			mul: (PairwiseTableStrategy),
			square: (PairwiseTableStrategy),
			invert: (PairwiseTableStrategy),
			wide_mul: (TrivialWideMul),
		},
	]
);
