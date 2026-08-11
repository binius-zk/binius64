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
//! # Status: a skeleton, and not sound
//!
//! The gadgets a real recursive verifier needs are not written. In their place, values that ought
//! to be derived are left as circuit inputs for the replay to supply:
//!
//! - **The Fiat-Shamir state.** `sample` and `sample_bits` return free wires rather than the
//!   challenger's output, so nothing ties a challenge to the transcript that produced it.
//! - **The Merkle openings.** `recv_openings` and `recv_committed_vector` return free wires rather
//!   than values checked against a commitment root.
//! - **Two field gadgets.** `invert_or_zero` and `square_transpose` are hints, so they carry the
//!   right value but nothing pins them to their inputs.
//!
//! What *is* constrained is the verifier's arithmetic: the sumcheck folding, the eq-indicator and
//! Lagrange evaluations, the monster multilinear, and every `assert_zero` along the way.
//!
//! So a circuit built here accepts proofs it should reject. It is useful for measuring that
//! arithmetic and for keeping the pipeline honest while the gadgets are written, not for proving
//! anything. Each gadget that lands removes entries from the recorded input list; with all of them
//! in place the only input left is the proof itself.

mod channel;
mod filler;
mod hints;
mod shared;
pub mod symbolic;

pub use channel::{Binius64BuilderChannel, Commitment, Recorded};
pub use filler::WitnessFillerChannel;
pub use symbolic::{SymbolicElem, SymbolicWord};
