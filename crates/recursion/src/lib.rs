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
//! The statement a verifier reads comes back as wires.
//! [`bind_public`](Binius64BuilderChannel::bind_public) ties chosen ones to public inputs.
//! That is what lets whoever checks an outer proof see which statement was verified.
//!
//! # Status: a skeleton, and not sound
//!
//! One gadget is still missing, so a value that ought to be derived is left as a circuit input:
//!
//! - **The Fiat-Shamir state.** `sample` and `sample_bits` return free wires rather than the
//!   challenger's output, so nothing ties a challenge to the transcript that produced it.
//!
//! What *is* constrained is the verifier's arithmetic: the sumcheck folding, the eq-indicator and
//! Lagrange evaluations, the monster multilinear, every `assert_zero` along the way, and both field
//! gadgets that used to be bare hints.
//!
//! The Merkle commitments are constrained too, by [`merkle`].
//! An opened leaf is hashed and climbed to a decommitted layer the root fixes.
//! A committed vector has its tree rebuilt over it.
//! Both keep their values as circuit inputs, since those values are proof data.
//!
//! A circuit built here still accepts proofs it should reject: a prover picks its own challenges.
//! Driving [`challenger`] from `sample` and `sample_bits` is what removes the bullet above.

pub mod challenger;
mod channel;
mod filler;
mod hints;
pub mod merkle;
mod shared;
pub mod symbolic;

pub use channel::{Binius64BuilderChannel, Commitment, Recorded};
pub use filler::WitnessFillerChannel;
pub use symbolic::{SymbolicElem, SymbolicWord};
