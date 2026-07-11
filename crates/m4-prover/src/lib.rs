// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

//! Witness table and prover for the data-parallel Binius64 M4 proof system.

mod bitand;
mod prove;
mod reduction;
mod shift;
#[cfg(test)]
mod test_utils;
mod value_table;

pub use bitand::BatchAndCheckWitness;
pub use prove::Prover;
pub use reduction::{ReductionProverOutput, prove_reduction};
pub use value_table::{BatchWitnessFiller, ValueTable};
