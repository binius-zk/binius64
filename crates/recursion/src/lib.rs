// Copyright 2026 The Binius Developers

//! Building a Binius64 circuit that verifies a Binius64 proof.
//!
//! A verifier written against the channel traits does not name a concrete field element or word
//! type: it reads [`SymbolicElem`]s and [`SymbolicWord`]s off a channel and asks the channel to do
//! what it cannot express itself. Running such a verifier against [`Binius64BuilderChannel`]
//! therefore produces, instead of an accept or a reject, a circuit that records what the verifier
//! did.
//!
//! ```text
//!   verifier + builder channel  ->  circuit
//!   verifier + filler channel   ->  its witness
//! ```
//!
//! The same verifier drives both. No shape in the protocol depends on a received value — every
//! length comes from the FRI parameters and the oracle specs — so the two runs reach the same
//! operations in the same order, and one cursor pairs what the second saw with the wires the first
//! allocated.
//!
//! # Two channels
//!
//! [`Binius64BuilderChannel`] is a skeleton, and not sound.
//! Values that ought to be derived are left as circuit inputs for a replay to supply:
//!
//! - **The Fiat-Shamir state.** `sample` and `sample_bits` return free wires rather than the
//!   challenger's output, so nothing ties a challenge to the transcript that produced it.
//! - **The Merkle openings.** `recv_openings` and `recv_committed_vector` return free wires rather
//!   than values checked against a commitment root.
//!
//! What it does constrain is the verifier's arithmetic alone:
//!
//! - the sumcheck folding, and the eq-indicator and Lagrange evaluations
//! - the monster multilinear, and every `assert_zero` along the way
//!
//! So a circuit built there accepts proofs it should reject.
//! It is for measuring that arithmetic, not for proving anything.
//!
//! [`merkle_channel`] closes both holes.
//! It drives the [`challenger`] and [`merkle`] gadgets from those same four methods:
//!
//! ```text
//!   skeleton:        proof -> replay -> wires the circuit could not derive
//!   merkle_channel:  proof -> wires, and every other value is a gate output
//! ```
//!
//! An unsatisfied circuit there means a rejected proof, which the skeleton cannot say.

pub mod challenger;
mod channel;
mod filler;
mod hints;
pub mod merkle;
pub mod merkle_channel;
mod shared;
pub mod symbolic;

pub use channel::{Binius64BuilderChannel, Commitment, Recorded};
pub use filler::WitnessFillerChannel;
pub use symbolic::{SymbolicElem, SymbolicWord};
