// Copyright 2023-2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use crate::{
	arch::M128,
	underlier::{Divisible, ScaledUnderlier, impl_divisible_self},
};

pub type M256 = ScaledUnderlier<M128, 2>;

// Reflexive `Divisible<Self>`, needed by the width-one `PackedPrimitiveType<M256, _>` packing whose
// scalar (e.g. `GhashSq256b`) is itself `M256`-backed. `M256` is a `ScaledUnderlier` alias here, so
// this is the portable/aarch64 counterpart to the native `impl_divisible_self!(M256)` on x86_64.
impl_divisible_self!(M256);

pub const fn m256_from_u128s(lo: u128, hi: u128) -> M256 {
	ScaledUnderlier([M128::from_u128(lo), M128::from_u128(hi)])
}
