// Copyright 2026 The Binius Developers
//! GHASH² sliced multiplication using x86_64 CLMUL instructions.

#[cfg(target_arch = "x86_64")]
#[allow(unused_imports)]
use std::arch::x86_64::*;

use crate::{PackedUnderlier, Underlier, ghash, underlier::OpsClmul};

/// Multiply packed GHASH² elements in sliced representation using CLMUL arithmetic.
#[inline]
pub fn mul_sliced<U: Underlier + OpsClmul + PackedUnderlier<u128>>(x: [U; 2], y: [U; 2]) -> [U; 2] {
	super::sliced::mul_sliced(x, y, ghash::clmul::mul, ghash::clmul::mul_inv_x)
}

/// Multiply packed GHASH² elements in sliced representation using 128-bit CLMUL.
#[cfg(all(
	target_arch = "x86_64",
	target_feature = "pclmulqdq",
	target_feature = "sse2"
))]
#[inline]
pub fn mul_sliced_m128i(x: [__m128i; 2], y: [__m128i; 2]) -> [__m128i; 2] {
	mul_sliced(x, y)
}

/// Multiply packed GHASH² elements in sliced representation using 256-bit CLMUL.
#[cfg(all(
	target_arch = "x86_64",
	target_feature = "vpclmulqdq",
	target_feature = "avx2",
	target_feature = "sse2"
))]
#[inline]
pub fn mul_sliced_m256i(x: [__m256i; 2], y: [__m256i; 2]) -> [__m256i; 2] {
	mul_sliced(x, y)
}
