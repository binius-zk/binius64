// Copyright 2026 The Binius Developers

//! The projections of a populated [`ValueTable`](crate::ValueTable) that the reductions consume.
//!
//! The table stores raw words, wire-major; no reduction reads it in that form.
//!
//! - [`build_operation_columns`] — one word column per operand, for the operation checks.
//! - [`FoldedWitness`] — the witness with its instance axis collapsed, for the shift.
//! - [`operand_rho_multilinear`] — one column reduced to one element per instance.
//!
//! A word column is constraint-major, so one constraint's instances are contiguous:
//!
//! ```text
//! row = local_constraint * n_instances + instance
//! ```

mod instance_fold;
mod operand_columns;
mod operand_rho;

pub use instance_fold::{FoldedWitness, FoldedWord};
pub use operand_columns::build_operation_columns;
pub use operand_rho::operand_rho_multilinear;
