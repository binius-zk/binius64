// Copyright 2026 The Binius Developers
//! GHASH² sliced multiplication and squaring using aarch64 CLMUL (PMULL) instructions.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::uint64x2_t;

use crate::{PackedUnderlier, Underlier, ghash, underlier::OpsClmul};

/// Multiply packed GHASH² elements in sliced representation using CLMUL arithmetic.
#[inline]
pub fn mul_sliced<U: Underlier + OpsClmul + PackedUnderlier<u128>>(x: [U; 2], y: [U; 2]) -> [U; 2] {
	super::sliced::mul_sliced(x, y, ghash::clmul::mul, ghash::clmul::mul_inv_x)
}

/// Square packed GHASH² elements in sliced representation using CLMUL arithmetic.
#[inline]
pub fn square_sliced<U: Underlier + OpsClmul + PackedUnderlier<u128>>(x: [U; 2]) -> [U; 2] {
	super::sliced::square_sliced(x, ghash::clmul::square, ghash::clmul::mul_inv_x)
}

/// Multiply a GHASH² element stored as a pair of NEON registers.
#[cfg(all(
	target_arch = "aarch64",
	target_feature = "neon",
	target_feature = "aes"
))]
#[inline]
pub fn mul_neon(x: [uint64x2_t; 2], y: [uint64x2_t; 2]) -> [uint64x2_t; 2] {
	mul_sliced(x, y)
}

/// Square a GHASH² element stored as a pair of NEON registers.
#[cfg(all(
	target_arch = "aarch64",
	target_feature = "neon",
	target_feature = "aes"
))]
#[inline]
pub fn square_neon(x: [uint64x2_t; 2]) -> [uint64x2_t; 2] {
	square_sliced(x)
}
