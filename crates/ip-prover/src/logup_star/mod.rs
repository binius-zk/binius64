// Copyright 2026 The Binius Developers

//! Prover for the logUp* indexed-lookup reduction of knowledge.
//!
//! This is the prover counterpart of the verifier in [`binius_ip::logup_star`].
//! See that module for the protocol, its soundness, and the index embedding.
//!
//! logUp* proves an indexed lookup `(I^* T)[i] = T[index[i]]`, for one or more lookers sharing
//! the table (batched by a random linear combination over the looker numerators).
//! - It never commits the looked-up vectors `I_j^* T`, which would have `2^n` entries each.
//! - Instead it commits the combined pushforward `Y`, which has only `2^m` entries.
//! - This rests on the duality `(I^* T)(r) = <I^* T, eq_r> = <T, I_* eq_r> = <T, Y>`.
//!
//! # What this prover does
//!
//! Given the table `T`, the index column, the evaluation point `r`, and the claim `e`, it:
//!
//! 1. samples the logUp challenge `c`,
//! 2. builds the looker and table fractional-addition circuits and sends their root fractions,
//! 3. runs the looker-side GKR to the index leaf claim,
//! 4. runs the first `m-1` table-side GKR layers to the layer-1 claim,
//! 5. proves the batched final layer, fusing the last GKR layer with the product check.
//!
//! The result is the same [`LogupOutput`] the verifier returns.
//! It holds reduced evaluation claims on `T`, on `Y`, and on the index multilinear.
//! The caller verifies those three claims separately.

mod final_layer;
mod prove;
pub mod witness;

pub use binius_ip::logup_star::LogupOutput;

pub use self::prove::{Looker, prove, prove_reduction};
