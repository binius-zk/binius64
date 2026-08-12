// Copyright 2025 Irreducible Inc.

//! Commitment shape and verifier for the data-parallel Binius64 M4 proof system.

mod commit;
mod composite;
mod verify;

pub use commit::BatchCommitLayout;
pub use composite::{IOPVerifierM4, VerifierM4};
pub use verify::{IOPVerifier, Verifier};
