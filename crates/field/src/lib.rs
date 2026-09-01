// Copyright 2023-2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

#![warn(rustdoc::missing_crate_level_docs)]

//! Binary tower field implementations for use in Binius.
//!
//! This library implements binary tower field arithmetic. The canonical binary field tower
//! construction is specified in [DP23], section 2.3. This is a family of binary fields with
//! extension degree $2^{\iota}$ for any tower height $\iota$. Mathematically, we label these sets
//! $T_{\iota}$.
//!
//! [DP23]: https://eprint.iacr.org/2023/1784

pub mod arch;
pub mod arithmetic_traits;
pub mod binary_field;
mod divisible;
pub mod extension;
pub mod field;
pub mod fields;
pub mod linear_transformation;
mod macros;
mod maskable;
pub mod packed;
pub mod packed_extension;
pub mod packed_fields;
mod random;
#[cfg(test)]
mod tests;
pub mod transpose;
mod underlier;
pub mod util;

pub use arithmetic_traits::WideMul;
pub use binary_field::*;
pub use divisible::Divisible;
pub use extension::*;
pub use field::{Field, FieldOps};
pub use fields::{ghash::*, ghash_sq::*, rijndael::*};
pub use maskable::Maskable;
pub use packed::PackedField;
pub use packed_extension::*;
pub use packed_fields::{sliced::SlicedPackedField, *};
pub use random::Random;
pub use transpose::{transpose_square_blocks, transpose_square_blocks_array};
pub use underlier::{Underlier, UnderlierView};
