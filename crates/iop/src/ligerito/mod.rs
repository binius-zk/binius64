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
//! - [`CommittedOracle`], one committed message as level 0 sees it;
//! - [`LigeritoVerifier`], the ladder of committed levels down to a cleartext residual;
//! - [`channel::LigeritoVerifierChannel`], the ladder behind the IOP verifier channel trait;
//! - [`LigeritoParams::proof_size`], the byte-exact estimate;
//! - [`LigeritoParams::verifier_cost`], what checking that proof costs, level by level;
//! - [`LadderCost`], the two prices a ladder shape pays, in bytes and in encoding work;
//! - [`LadderObjective`], the exchange rate that turns those two prices into one;
//! - [`LadderSearch`], the search for the best ladder under that exchange rate.
//!
//! The soundness regimes and the error terms live in [`crate::soundness`], because they describe
//! Reed-Solomon proximity testing rather than Ligerito.
//! `PROXIMITY_GAPS.md` at the repository root records which of them this field can support.
//!
//! [NA25]: <https://eprint.iacr.org/2025/1187>

pub mod channel;
mod committed_oracle;
mod common;
pub mod compiler;
mod error;
mod induced_basis;
mod ladder_cost;
mod ladder_objective;
mod ladder_search;
mod opening;
mod size_estimation;
mod verifier_cost;

pub use committed_oracle::CommittedOracle;
pub use common::*;
pub use error::{Error, VerificationError};
pub use induced_basis::InducedBasis;
pub use ladder_cost::LadderCost;
pub use ladder_objective::LadderObjective;
pub use ladder_search::{LadderSearch, MAX_LOG_INV_RATE, MAX_LOG_LANES};
pub use opening::LigeritoVerifier;
pub use verifier_cost::VerifierCost;
