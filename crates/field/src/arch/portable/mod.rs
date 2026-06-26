// Copyright 2023-2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

pub(crate) mod packed;
pub(crate) mod packed_macros;

pub mod m128;
pub mod m256;
pub mod m512;

pub use m128::M128;
pub use m256::{M256, m256_from_u128s};
pub use m512::M512;

pub mod arithmetic;

pub mod packed_aes_128;
pub mod packed_aes_256;
pub mod packed_aes_512;
pub mod packed_aes_8;

pub mod packed_ghash_128;
pub mod packed_ghash_256;
pub mod packed_ghash_512;

pub(crate) mod univariate_mul_utils_128;

pub(crate) mod packed_arithmetic;
pub(super) mod pairwise_table_arithmetic;
pub(super) mod reuse_multiply_arithmetic;
pub(super) mod scaled_arithmetic;
