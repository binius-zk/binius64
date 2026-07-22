// Copyright 2026 The Binius Developers
//! GHASH² sliced multiplication and squaring using aarch64 CLMUL (PMULL) instructions.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::uint64x2_t;

use crate::{PackedUnderlier, Underlier, ghash, underlier::OpsClmul};

/// Multiply packed GHASH² elements in sliced representation using CLMUL arithmetic.
#[inline]
pub fn mul_sliced<U: Underlier + OpsClmul + PackedUnderlier<u128>>(x: [U; 2], y: [U; 2]) -> [U; 2] {
	super::sliced::mul_sliced(
		x,
		y,
		ghash::clmul::mul_wide,
		ghash::clmul::reduce,
		ghash::clmul::mul_x_wide,
	)
}

/// Widening (unreduced) multiply of packed GHASH² elements in sliced representation using CLMUL
/// arithmetic. Returns the three raw GHASH products (`[U; 3]` each); see
/// [`super::sliced::mul_wide_sliced`] and reduce with [`reduce_sliced`].
#[inline]
pub fn mul_wide_sliced<U: Underlier + OpsClmul + PackedUnderlier<u128>>(
	x: [U; 2],
	y: [U; 2],
) -> [[U; 3]; 3] {
	super::sliced::mul_wide_sliced(x, y, ghash::clmul::mul_wide)
}

/// Reduce the three raw products from [`mul_wide_sliced`] into a GHASH² element using CLMUL
/// arithmetic; see [`super::sliced::reduce_sliced`].
#[inline]
pub fn reduce_sliced<U: Underlier + OpsClmul + PackedUnderlier<u128>>(t: [[U; 3]; 3]) -> [U; 2] {
	super::sliced::reduce_sliced(t, ghash::clmul::reduce, ghash::clmul::mul_x_wide)
}

/// Square packed GHASH² elements in sliced representation using CLMUL arithmetic.
#[inline]
pub fn square_sliced<U: Underlier + OpsClmul + PackedUnderlier<u128>>(x: [U; 2]) -> [U; 2] {
	super::sliced::square_sliced(x, ghash::clmul::square, ghash::clmul::mul_x)
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
