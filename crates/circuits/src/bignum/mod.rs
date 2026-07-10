// Copyright 2025 Irreducible Inc.
//! Arbitrary-precision bignum arithmetic for circuits.
//!
//! This module provides operations on big integers represented as vectors of `Wire` elements,
//! where each `Wire` represents a 64-bit limb. The representation uses little-endian ordering,
//! meaning the least significant limb is at index 0.

mod addsub;
mod big_uint_divide;
mod big_uint_mod_pow;
mod biguint;
mod cmp;
mod mod_divide;
mod mod_inverse;
mod mul;
mod prime_field;
mod reduce;

#[cfg(test)]
mod tests;

pub use addsub::{add, sub};
pub use big_uint_divide::BigUintDivideHint;
pub use big_uint_mod_pow::BigUintModPowHint;
pub use biguint::{BigUint, assert_eq, select};
pub use cmp::{biguint_eq, biguint_lt};
pub use mod_divide::ModDivideHint;
pub use mod_inverse::ModInverseHint;
pub use mul::{karatsuba_mul, optimal_mul, optimal_sqr, textbook_mul, textbook_square};
pub use prime_field::PseudoMersennePrimeField;
pub use reduce::{ModReduce, PseudoMersenneModReduce};
