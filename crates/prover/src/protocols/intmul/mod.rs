// Copyright 2025 Irreducible Inc.

mod error;
pub mod prove;
mod transpose_bits;
pub mod witness;

pub use error::Error;
pub use prove::prove;

#[cfg(test)]
mod tests;
