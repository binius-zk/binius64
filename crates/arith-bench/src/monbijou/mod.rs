// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers
//! Arithmetic for the Monbijou field, GF(2)\[X\] / (X^64 + X^4 + X^3 + X + 1).
//!
//! This module implements arithmetic in the GF(2^64) binary field defined by the
//! reduction polynomial X^64 + X^4 + X^3 + X + 1, which is used in the ISO 3309
//! standard for CRC-64 error detection.
//!
//! The `x86_64` submodule provides implementations using x86_64 carry-less multiplication (CLMUL)
//! instructions via the [`OpsClmul`](crate::underlier::OpsClmul) trait, optimized for SIMD
//! parallelism across vector types like __m128i or __m256i. The `aarch64` submodule provides direct
//! aarch64 PMULL implementations on the `poly64x2_t` underlier. The [`soft64`] submodule provides
//! portable implementations that use no CLMUL or SIMD intrinsics.

#[cfg(target_arch = "aarch64")]
pub mod aarch64;
pub mod soft64;
#[cfg(target_arch = "x86_64")]
pub mod x86_64;

/// The multiplicative identity in the Monbijou field
///
/// In this field, the standard representation of 1 is simply 0x01
pub const MONBIJOU_ONE: u64 = 0x01;

/// The multiplicative identity in the Monbijou 128-bit extension field
///
/// In the degree-2 extension GF(2^128), the standard representation of 1 is simply 0x01
pub const MONBIJOU_128B_ONE: u128 = 0x01;
