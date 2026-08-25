// Copyright 2026 The Binius Developers

//! Parameter selection and proof-size accounting for the Ligerito polynomial commitment scheme.
//!
//! Ligerito ([NA25]) is Ligero's interleaved matrix commitment, recursed.
//! One *level* of that recursion does four things:
//!
//! - reshape its message into `2^log_lanes` interleaved lanes of `2^log_msg_cols` columns;
//! - Reed–Solomon encode every lane at the level's own rate;
//! - Merkle-commit one leaf per codeword position, holding that position across all lanes;
//! - open `n_queries` rows, then fold the matrix by the level's `log_lanes` sumcheck challenges.
//!
//! The folded matrix becomes the next level's message, committed at a *strictly lower rate*.
//! That is what makes the deep levels cheap, because a lower rate needs fewer queries.
//! After the last level the residual matrix is sent in the clear.
//!
//! This module holds:
//!
//! - [`LigeritoParams`] and its invariants;
//! - [`InducedBasis`], the weight vector a level's opened rows put on its message;
//! - [`LigeritoVerifier`], the ladder of committed levels down to a cleartext residual;
//! - [`channel::LigeritoVerifierChannel`], the ladder behind the IOP verifier channel trait;
//! - [`LigeritoParams::proof_size`], the byte-exact estimate;
//! - [`LigeritoParams::optimal_ladder`], the search that minimizes it subject to a security target.
//!
//! The soundness regimes and the error terms live in [`crate::soundness`], because they describe
//! Reed-Solomon proximity testing rather than Ligerito.
//! `PROXIMITY_GAPS.md` at the repository root records which of them this field can support.
//!
//! [NA25]: <https://eprint.iacr.org/2025/1187>

pub mod channel;
mod common;
pub mod compiler;
mod error;
mod induced_basis;
mod opening;
mod size_estimation;

pub use common::*;
pub use error::{Error, VerificationError};
pub use induced_basis::InducedBasis;
pub use opening::LigeritoVerifier;
pub use size_estimation::{MAX_LOG_INV_RATE, MAX_LOG_LANES};
