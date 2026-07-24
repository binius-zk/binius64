// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

//! Witness table and prover for the data-parallel Binius64 M4 proof system.

mod operand_witness;
mod prove;
mod shift;
#[cfg(test)]
mod test_utils;
mod value_table;

pub use operand_witness::build_operation_columns;
pub use prove::{IOPProver, Prover};
pub use value_table::{BatchWitnessFiller, ValueTable};
