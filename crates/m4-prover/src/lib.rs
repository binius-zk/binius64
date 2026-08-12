// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

//! Prover for the data-parallel Binius64 M4 proof system.
//!
//! The witness it proves over is a [`ValueTable`](binius_core::ValueTable), which
//! [`binius_core`] defines and [`binius_frontend`] populates. This crate commits one and reduces
//! the constraint system over it.

mod composite;
mod prove;
mod shift;
#[cfg(test)]
mod test_utils;
mod value_table;
mod witness;

pub use composite::{IOPProverM4, ProverM4};
pub use prove::{IOPProver, Prover};
pub use value_table::{commit_layout, pack_table};
pub use witness::OperandColumns;
