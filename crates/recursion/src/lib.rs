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
//! # Status: nothing the channel hands out is free
//!
//! Every value a verifier reads here is derived in-circuit, or bound to something that is:
//!
//! - the verifier's arithmetic, down to every `assert_zero` along the way
//! - the two field gadgets that used to be bare hints
//! - the Merkle commitments, by [`merkle`]
//! - the Fiat-Shamir state, by [`challenger`]
//!
//! An opened leaf is hashed and climbed to a decommitted layer the root fixes.
//! A committed vector has its tree rebuilt over it.
//! Both keep their values as circuit inputs, since those values are proof data.
//!
//! Every byte the native challenger observes is absorbed here too.
//! A challenge is therefore derived from the transcript rather than supplied.
//! A query index is derived the same way, and masked to the width it was asked for.
//!
//! What a replay still fills is the proof itself, and the statement being verified.
//! Both are given rather than derived, which is why they are the inputs that remain.

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
