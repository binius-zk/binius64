// Copyright 2025 Irreducible Inc.

//! Commitment shape and verifier for the data-parallel Binius64 M4 proof system.

mod bitand;
mod commit;
mod reduction;
mod verify;

pub use bitand::verify_bitand_reduction;
pub use commit::BatchCommitLayout;
pub use reduction::{ReductionVerifierOutput, verify_reduction};
pub use verify::Verifier;
