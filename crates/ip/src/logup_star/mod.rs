// Copyright 2026 The Binius Developers

//! Verifier for the logUp* indexed-lookup reduction of knowledge.
//!
//! logUp* proves an indexed lookup `(I^* T)[i] = T[I[i]]`.
//! Unlike classic logUp, it never commits the looked-up vector `I^* T`.
//! See [Soukhanov25] for the construction.
//!
//! [Soukhanov25]: <https://eprint.iacr.org/2025/946>
//!
//! # What is being proved
//!
//! The caller holds a claim about the looked-up vector at a point:
//!
//! ```text
//!     (I^* T)(r) = e
//! ```
//!
//! The symbols are:
//!
//! - `T`: the table multilinear, with `m` variables (`2^m` entries).
//! - `I`: the index multilinear, with `n` variables (`2^n` entries).
//! - `r`: the `n`-coordinate evaluation point.
//! - `e`: the claimed evaluation.
//!
//! The reduction turns this one claim into three separate evaluation claims:
//!
//! - one on the table `T`,
//! - one on the pushforward `Y`,
//! - one on the index `I`.
//!
//! The caller verifies those three claims, which is out of scope here.
//!
//! # The pushforward trick
//!
//! Let `X = eq_r` be the equality-indicator multilinear at the point `r`.
//! Pullback and pushforward are dual under the inner product, which gives:
//!
//! ```text
//!     (I^* T)(r) = <I^* T, eq_r> = <T, I_* eq_r> = <T, Y>
//! ```
//!
//! Here `Y = I_* eq_r` is the pushforward of `eq_r` along `I`.
//! `Y` has only `2^m` entries, which is cheap.
//! The avoided vector `I^* T` has `2^n` entries, which is expensive when `n` is large.
//!
//! # The two checks
//!
//! First, pushforward correctness, via a logarithmic-derivative (logUp) identity for a random `c`:
//!
//! ```text
//!     sum_{i in B_n} eq_r(i) / (c - I(i)) = sum_{j in B_m} Y(j) / (c - j)
//! ```
//!
//! - Each side is a sum of fractions.
//! - A fractional-addition GKR circuit collapses each side to a single root fraction.
//! - Equality of the two sums is the cross-multiplication of the two root fractions.
//!
//! See [`crate::fracaddcheck`] for the GKR circuit.
//!
//! Second, the product claim `<T, Y> = e`, proved by a product sumcheck over the `m`-variable cube.
//!
//! # Batching the last GKR layer with the product sumcheck
//!
//! The table-side GKR circuit ends in an evaluation of `Y`.
//! The product sumcheck also ends in an evaluation of `Y`.
//! Run naively, these are two distinct evaluations at two distinct points.
//!
//! Both reductions share the same final step:
//!
//! - they split the leaf multilinears on the highest variable into two halves,
//! - they combine the halves over the same `m-1` low variables,
//! - they finish with one line-fold over the highest variable.
//!
//! So both can run as one `(m-1)`-variable sumcheck followed by one shared line-fold.
//! That yields a single evaluation point, collapsing the two `Y` evaluations into one.
//!
//! # Soundness
//!
//! - The logUp identity for a random `c` catches a wrong `Y` except with probability `(n + m) /
//!   |F|`.
//! - This is Lemma 2 of [Soukhanov25]: the identity holds only when `Y = I_* eq_r`.
//! - The two GKR circuits and the batched sumcheck add the usual sumcheck soundness error.
//! - The cross-multiplication of the root fractions assumes both root denominators are nonzero.
//! - A root denominator is a product of factors `c - I(i)` or `c - j`.
//! - That product is nonzero except with probability `(n + m) / |F|` over the random `c`.
//!
//! # Index embedding
//!
//! Table positions `j` in `0..2^m` and committed index values `I[i]` live in the same domain.
//! A position is embedded into `F` through the `GF(2)`-linear basis:
//!
//! ```text
//!     iota(j) = sum_{t : bit t of j is set} basis(t)
//! ```
//!
//! The table-side denominator multilinear is therefore `J(x) = sum_t basis(t) * x_t`.
//! The verifier evaluates it by itself.
//! This matches the index encoding used elsewhere in the Spartan verifier.

mod error;
mod final_layer;
mod output;
mod verify;

pub use self::{
	error::{Error, VerificationError},
	output::LogupOutput,
	verify::verify,
};
