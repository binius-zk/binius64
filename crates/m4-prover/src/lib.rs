// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

//! Witness table and prover for the data-parallel Binius64 M4 proof system.

mod prove;
mod shift;
#[cfg(test)]
mod test_utils;
mod value_table;
mod witness;

pub use prove::{IOPProver, Prover};
pub use value_table::{BatchWitnessFiller, ValueTable};
pub use witness::build_operation_columns;
