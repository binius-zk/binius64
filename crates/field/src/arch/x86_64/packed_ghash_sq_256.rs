// Copyright 2026 The Binius Developers

//! x86_64 strategy selection for the `PackedGhashSq1x256b` packing.

/// Widening-multiply wrapper used by the `PackedGhashSq1x256b` packing: batch the Karatsuba
/// diagonal into a single 256-bit carry-less multiply when VPCLMULQDQ is available, otherwise the
/// sliced Karatsuba multiply, which trades that batching for one fewer GHASH reduction.
///
/// AVX2 is part of the condition because without it `PackedBinaryGhash2x128b` is backed by the
/// scaled `M256`, whose widening multiply is two independent 128-bit multiplies — exactly what the
/// batching is meant to avoid.
#[cfg(all(target_feature = "vpclmulqdq", target_feature = "avx2"))]
pub type GhashSqWideMul1x<T> = super::arithmetic::ghash_sq::GhashSqHybridWideMul<T>;
#[cfg(not(all(target_feature = "vpclmulqdq", target_feature = "avx2")))]
pub type GhashSqWideMul1x<T> = crate::arch::portable::arithmetic::ghash_sq::GhashSqSlicedWideMul<T>;
