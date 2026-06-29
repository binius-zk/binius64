// Copyright 2025 Irreducible Inc.

//! Commitment shape and verifier for the data-parallel Binius64 M4 proof system.

mod commit;
mod verify;

pub use commit::BatchCommitLayout;
pub use verify::Verifier;
