// Copyright 2026 The Binius Developers
//! GHASH² sliced multiplication using aarch64 CLMUL (PMULL) instructions.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::uint64x2_t;

use crate::{PackedUnderlier, Underlier, ghash, underlier::OpsClmul};

/// Multiply packed GHASH² elements in sliced representation using CLMUL arithmetic.
#[inline]
pub fn mul_sliced<U: Underlier + OpsClmul + PackedUnderlier<u128>>(
	x: [U; 2],
	y: [U; 2],
) -> [U; 2] {
	super::sliced::mul_sliced(x, y, ghash::clmul::mul, ghash::clmul::mul_inv_x)
}

/// Multiply packed GHASH² elements in sliced representation using NEON PMULL.
#[cfg(all(target_arch = "aarch64", target_feature = "neon", target_feature = "aes"))]
#[inline]
pub fn mul_sliced_uint64x2(
	x: [uint64x2_t; 2],
	y: [uint64x2_t; 2],
) -> [uint64x2_t; 2] {
	mul_sliced(x, y)
}
