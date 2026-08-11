// Copyright 2026 The Binius Developers

//! Building a Binius64 circuit that verifies a Binius64 proof.
//!
//! A verifier written against the channel traits does not name a concrete field element or word
//! type: it reads [`Elem`]s and [`Word`]s off a channel and asks the channel to do the operations
//! it cannot express itself. Running such a verifier against [`Binius64BuilderChannel`] therefore
//! produces, instead of an accept/reject, a circuit whose satisfying assignments are exactly the
//! proofs the verifier would accept.
//!
//! # Where the values live
//!
//! Everything the prover sends becomes a witness wire, and everything derived from it becomes a
//! gate output. The circuit's only inputs are the proof byte stream and the statement, so filling
//! a witness means writing the proof into the recorded wires and letting the compiled evaluator
//! derive the rest — there is no second pass that replays the verifier.
//!
//! # Status
//!
//! The channel structure is in place. The gadgets it drives are not yet written, and every call
//! site that needs one is marked `todo!()` or noted as leaving its values unconstrained. Those are
//! tracked separately: the Fiat-Shamir challenger and the SHA-256 Merkle verification. Until they
//! land, a circuit this channel builds is **not sound** — it constrains the arithmetic but not the
//! transcript or the openings.

mod channel;
mod elem;
mod word;

pub use channel::Binius64BuilderChannel;
pub use elem::Elem;
pub use word::Word;
