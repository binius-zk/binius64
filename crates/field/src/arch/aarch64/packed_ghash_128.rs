// Copyright 2023-2025 Irreducible Inc.
// Copyright 2026 The Binius Developers
// Copyright (c) 2019-2023 RustCrypto Developers

//! ARMv8 `PMULL`-accelerated implementation of GHASH.
//!
//! Based on the optimized GHASH implementation using carryless multiplication
//! instructions available on ARMv8 processors with NEON support.

/// Widening-multiply wrapper used by the GHASH packing: the reduction-deferring
/// `GhashClMulWideMul`.
pub type GhashWideMul1x<T> = super::arithmetic::ghash::GhashClMulWideMul<T>;

/// Square wrapper for the `PackedBinaryGhash1x128b` packing: the PMULL carryless-multiply square.
pub type GhashSquare1x<T> = super::arithmetic::ghash::GhashClMul<T>;

/// Invert wrapper for the `PackedBinaryGhash1x128b` packing: the shared Itoh-Tsujii inversion
/// (there is no CLMUL inverse).
pub type GhashInvert1x<T> = crate::arch::portable::arithmetic::itoh_tsujii::GhashItohTsujii<T>;
