// Copyright 2025 Irreducible Inc.

//! Witness table and commitment for the data-parallel Binius64 M4 proof system.

mod commit;
mod value_table;

pub use commit::BatchCommitLayout;
pub use value_table::{PopulateInstanceError, ValueTable};
