// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers
//! Core datatypes common to prover and verifier of Binius64.
//!
//! Most imporantly it hosts the definition of a [`ConstraintSystem`].

#![warn(rustdoc::missing_crate_level_docs)]

pub mod constraint_system;
pub mod error;
pub mod word;

pub use constraint_system::*;
pub use error::{
	ConstraintSystemError, ConstraintViolation, VerificationError, VerificationM4Error,
};
pub use word::Word;
