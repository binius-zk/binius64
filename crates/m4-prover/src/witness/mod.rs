// Copyright 2026 The Binius Developers

//! The projections of a populated [`ValueTable`](crate::ValueTable) that the reductions consume.
//!
//! The table stores raw words, wire-major; no reduction reads it in that form.
//!
//! - [`OperandColumns`] — one word column per operand, for the operation checks.
//! - [`FoldedWitness`] — the witness with its instance axis collapsed, for the shift.
//!
//! A word column is constraint-major, so one constraint's instances are contiguous:
//!
//! ```text
//! row = local_constraint * n_instances + instance
//! ```

mod instance_fold;
mod operand_columns;

pub use instance_fold::{FoldedWitness, FoldedWord};
pub use operand_columns::OperandColumns;
