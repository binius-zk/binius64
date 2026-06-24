// Copyright 2024-2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use cfg_if::cfg_if;

use super::m128::M128;
use crate::{
	arch::{
		MulFromWideMul,
		portable::packed_macros::{portable_macros::*, *},
	},
	arithmetic_traits::{impl_invert_with, impl_mul_with, impl_square_with},
};

// `Mul` is now `reduce(wide_mul)`; the widening multiply is the GFNI single-instruction multiply
// when available, otherwise the per-byte elementwise lookup (matching the old
// `PairwiseTableStrategy`).
#[cfg(all(
	target_arch = "x86_64",
	target_feature = "sse2",
	target_feature = "gfni"
))]
pub type AesWideMul128<T> = super::gfni::gfni_arithmetics::GfniWideMul<T>;
#[cfg(not(all(
	target_arch = "x86_64",
	target_feature = "sse2",
	target_feature = "gfni"
)))]
pub type AesWideMul128<T> = crate::arch::ElementwiseWideMul<T>;

define_packed_binary_fields!(
	underlier: M128,
	packed_fields: [
		packed_field {
			name: PackedAESBinaryField16x8b,
			scalar: AESTowerField8b,
			mul:       (MulFromWideMul),
			square:    (if gfni ReuseMultiplyStrategy else PairwiseTableStrategy),
			invert:    (if gfni GfniStrategy else PairwiseTableStrategy),
			wide_mul: (AesWideMul128),
		},
	]
);
