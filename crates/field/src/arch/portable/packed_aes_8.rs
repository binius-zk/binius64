// Copyright 2024-2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use crate::{
	arch::{
		MulFromWideMul, PairwiseTableStrategy,
		portable::{
			packed_macros::{portable_macros::*, *},
			pairwise_table_arithmetic::AesLookupWideMul,
		},
	},
	arithmetic_traits::{impl_invert_with, impl_mul_with, impl_square_with},
};

define_packed_binary_fields!(
	underlier: u8,
	packed_fields: [
		packed_field {
			name: PackedAESBinaryField1x8b,
			scalar: AESTowerField8b,
			mul: (MulFromWideMul),
			square: (PairwiseTableStrategy),
			invert: (PairwiseTableStrategy),
			wide_mul: (AesLookupWideMul),
		},
	]
);
