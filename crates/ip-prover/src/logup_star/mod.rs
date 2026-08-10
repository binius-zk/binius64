// Copyright 2026 The Binius Developers

//! Prover for the logUp* indexed-lookup reduction of knowledge.
//!
//! This is the prover counterpart of the verifier in [`binius_ip::logup_star`].
//! See that module for the protocol, its soundness, and the index embedding.
//!
//! logUp* proves an indexed lookup `(I^* T)[i] = T[index[i]]`, for one or more lookers reading one
//! or more tables (batched by a random linear combination over the looker numerators).
//! - It never commits the looked-up vectors `I_j^* T`, which would have `2^n` entries each.
//! - Instead it commits each table's pushforward `Y_t`, which has only `2^m_t` entries.
//! - This rests on the duality `(I^* T)(r) = <I^* T, eq_r> = <T, I_* eq_r> = <T, Y>`.
//!
//! # What this prover does
//!
//! Given the tables `T_t`, the index columns, the evaluation points `r_j`, and the claims `e_j`,
//! it:
//!
//! 1. samples the looker batching challenge `gamma` and builds the numerators and pushforwards,
//! 2. samples one logUp challenge `c_t` per table,
//! 3. builds one fractional-addition circuit per looker and per table, and one top circuit summing
//!    their root fractions, then sends that sum's denominator alone — its numerator is zero exactly
//!    when the lookup identities hold,
//! 4. runs the whole thing as one batched GKR down to the leaves,
//! 5. proves one batched sumcheck closing every table's pushforward and product claims.
//!
//! The result is the same [`LogupOutput`] the verifier returns.
//! It holds reduced evaluation claims on each `T_t`, on each `Y_t`, and on the index multilinears.
//! The caller verifies those claims separately.

mod prove;
mod pushforward;
pub mod witness;

pub use binius_ip::logup_star::LogupOutput;

pub use self::prove::{Looker, TableLookup, prove, prove_reduction};
