// Copyright 2023-2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use crate::{
	arch::M128,
	underlier::{Divisible, ScaledUnderlier, impl_divisible_self},
};

pub type M256 = ScaledUnderlier<M128, 2>;

// A 256-bit register divides into one 256-bit scalar such as `GhashSq256b`.
// The generic `ScaledUnderlier` impl only covers strictly smaller sub-underliers.
impl_divisible_self!(M256);

pub const fn m256_from_u128s(lo: u128, hi: u128) -> M256 {
	ScaledUnderlier([M128::from_u128(lo), M128::from_u128(hi)])
}
