// Copyright 2026 The Binius Developers

//! Proof-of-work grinding, as a capability a verifier channel may carry.
//!
//! A grind is a nonce the prover searched for and the verifier checks.
//! It taxes re-rolling whatever challenge comes next.
//! That is the one lever that moves a proximity bound no query count can reach.
//! [`crate::soundness::Grinding`] is where the bits it buys enter a security budget.

use binius_transcript::Error;

/// A verifier channel that can check a proof of work its prover paid into the transcript.
///
/// This sits apart from the traits a protocol reads messages through.
/// Grinding is a property of the Fiat-Shamir state rather than of what is committed.
/// So a protocol that grinds asks for this alongside its ordinary channel bound.
/// A channel that cannot express a proof of work then cannot be handed to it at all.
///
/// # Contract
///
/// A difficulty of zero is not a grind.
/// It must leave both the proof tape and the challenger untouched.
/// A protocol configured to grind nothing therefore writes the transcript an ungrinding one does.
/// That rule lives here rather than at the call sites.
/// Prover and verifier have to apply it in the same places, and neither can do so alone.
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
	/// * `bits` is at most [`binius_transcript::MAX_GRINDING_BITS`].
	fn verify_grind(&mut self, bits: usize) -> Result<(), Error>;
}
