// Copyright 2026 The Binius Developers

//! Portable strategy selection for the `PackedGhashSq1x256b` packing.

/// Widening-multiply wrapper used by the `PackedGhashSq1x256b` packing: the sliced Karatsuba
/// multiply, which defers the multiply-by-`X` into one of its two GHASH reductions.
pub type GhashSqWideMul1x<T> = super::arithmetic::ghash_sq::GhashSqSlicedWideMul<T>;
