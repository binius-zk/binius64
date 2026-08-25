// Copyright 2026 The Binius Developers

//! Proof-of-work grinding, as a capability a verifier channel may carry.
//!
//! A grind is a nonce the prover had to search for and the verifier re-checks cheaply.
//! It taxes re-rolling the challenge that follows it, and that tax is the whole point.
//! Grinding is the one lever that moves a proximity bound no number of queries can reach.
//! [`Grinding`](crate::soundness::Grinding) turns it into bits a security budget can count.

use binius_transcript::Error;

/// A verifier channel that can check a proof of work its prover paid into the transcript.
///
/// Grinding acts on the Fiat-Shamir state rather than on anything committed.
/// So it is a trait of its own, asked for alongside the channel bound a protocol already needs.
/// A channel with no way to express a proof of work is then rejected at the type level.
///
/// # Contract
///
/// A difficulty of zero is not a grind: it leaves the proof tape and the challenger untouched.
/// So a protocol configured to grind nothing writes exactly the transcript it wrote before.
/// That rule lives here rather than at the call sites, where the two sides could drift apart.
/// A grind is sound only when prover and verifier apply it at the same point in the transcript.
pub trait GrindingVerifierChannel {
	/// Checks the proof of work of `bits` difficulty standing at this point in the transcript.
	///
	/// ## Errors
	///
	/// Returns [`Error::InsufficientWork`] when the nonce the prover sent does not meet `bits`.
	/// Returns a deserialization error when the proof carries no nonce here.
	///
	/// ## Preconditions
	///
	/// * `bits` is at most [`MAX_GRINDING_BITS`](binius_transcript::MAX_GRINDING_BITS).
	fn verify_grind(&mut self, bits: usize) -> Result<(), Error>;
}
