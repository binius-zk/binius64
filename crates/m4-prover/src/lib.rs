// Copyright 2025 Irreducible Inc.

//! Witness table and prover for the data-parallel Binius64 M4 proof system.

mod prove;
mod value_table;

pub use prove::Prover;
pub use value_table::{PopulateInstanceError, ValueTable};
