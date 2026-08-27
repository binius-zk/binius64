// Copyright 2026 The Binius Developers

//! Parameter selection and proof-size accounting for the WHIR polynomial commitment scheme.
//!
//! WHIR ([ACFY24]) is Ligero's interleaved matrix commitment, recursed.
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
//! - [`WHIRParams`] and its invariants;
//! - [`InducedBasis`], the weight vector a level's opened rows put on its message;
//! - [`CommittedOracle`], one committed message as level 0 sees it;
//! - [`WHIRVerifier`], the ladder of committed levels down to a cleartext residual;
//! - [`channel::WHIRVerifierChannel`], the ladder behind the IOP verifier channel trait;
//! - [`WHIRParams::proof_size`], the byte-exact estimate;
//! - [`WHIRParams::verifier_cost`], what checking that proof costs, level by level;
//! - [`WHIRParams::optimal_ladder`], the search that minimizes it subject to a security target.
//!
//! The soundness regimes and the error terms live in [`crate::soundness`], because they describe
//! Reed-Solomon proximity testing rather than WHIR.
//! `PROXIMITY_GAPS.md` at the repository root records which of them this field can support.
//!
//! # Naming
//!
//! Two papers describe this recursion, and the one implemented here is the earlier one.
//!
//! - WHIR ([ACFY24]) fixes the code to Reed-Solomon, which is the only code committed with here.
//! - Ligerito ([NA25]) is the same recursion over any linear code with cheaply evaluable generator
//!   rows, and it published later.
//! - The interleaving both of them use is older than either, and standard from FRI.
//!
//! Ligerito's own discussion section notes that the two are structurally similar.
//! Nothing here is generic over the code, so the credit belongs to the Reed-Solomon paper.
//! Ligerito is still cited wherever a concrete choice follows its exposition rather than WHIR's.
//!
//! [ACFY24]: <https://eprint.iacr.org/2024/1586>
//! [NA25]: <https://eprint.iacr.org/2025/1187>

pub mod channel;
mod committed_oracle;
mod common;
pub mod compiler;
mod error;
mod induced_basis;
mod opening;
mod size_estimation;
mod verifier_cost;

pub use committed_oracle::CommittedOracle;
pub use common::*;
pub use error::{Error, VerificationError};
pub use induced_basis::InducedBasis;
pub use opening::WHIRVerifier;
pub use size_estimation::{MAX_LOG_INV_RATE, MAX_LOG_LANES};
pub use verifier_cost::VerifierCost;
