// Copyright 2025 Irreducible Inc.

mod error;
pub mod prove;
pub mod witness;

pub use error::Error;
pub use prove::prove;

#[cfg(test)]
mod tests;
