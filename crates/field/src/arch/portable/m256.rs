// Copyright 2023-2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use crate::{arch::M128, underlier::ScaledUnderlier};

pub type M256 = ScaledUnderlier<M128, 2>;

pub const fn m256_from_u128s(lo: u128, hi: u128) -> M256 {
	ScaledUnderlier([M128::from_u128(lo), M128::from_u128(hi)])
}
