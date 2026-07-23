// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers
//! Arithmetic for the GHASH field, GF(2)\[X\] / (X^128 + X^7 + X^2 + X + 1).

#[cfg(target_arch = "aarch64")]
pub mod aarch64;
pub mod bit_sliced;
pub mod clmul;
pub mod soft64;

/// The multiplicative identity in GHASH
///
/// In GHASH, the standard representation of 1 is simply 0x01
pub const ONE: u128 = 0x01;

/// The multiplicative inverse of X in GHASH.
///
/// X^{-1} = X^127 + X^6 + X + 1 modulo X^128 + X^7 + X^2 + X + 1.
pub const INV_X: u128 = 0x80000000000000000000000000000043;

/// The field generator X in GHASH.
///
/// In the polynomial-basis representation used here (bit `i` is the coefficient of `X^i`), `X` is
/// simply `X^1`.
pub const X: u128 = 0x02;

// Re-export mul_clmul for backward compatibility
#[allow(unused_imports)]
pub use clmul::{mul as mul_clmul, square as square_clmul};
