// Copyright 2026 The Binius Developers
//! Arithmetic for the GHASH² field, a degree-2 extension of GHASH.
//!
//! This module implements multiplication in a GF(2^256) binary field defined as a quadratic
//! extension of the GHASH field. Elements are pairs (a, b) representing a + bY, where a and b are
//! elements of the GHASH field GF(2)\[X\] / (X^128 + X^7 + X^2 + X + 1).
//!
//! The extension is defined by the irreducible polynomial Y^2 + Y + X^{-1} over the GHASH field.

pub mod aarch64;
pub mod sliced;
pub mod soft64;
pub mod x86_64;
