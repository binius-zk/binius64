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
//! This module contains **no protocol code**.
//! It holds only:
//!
//! - the parameter type and its invariants;
//! - the two soundness regimes the query counts may be derived in;
//! - the byte-exact proof-size estimate;
//! - the search that picks a rate ladder minimizing it.
//!
//! [NA25]: <https://eprint.iacr.org/2025/1187>

mod common;
mod size_estimation;

pub use common::*;
pub use size_estimation::{MAX_LOG_INV_RATE, MAX_LOG_LANES, optimal_ladder, proof_size};
