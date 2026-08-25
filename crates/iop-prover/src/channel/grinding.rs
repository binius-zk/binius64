// Copyright 2026 The Binius Developers

//! Proof-of-work grinding, as a capability a prover channel may carry.
//!
//! The counterpart of `binius_iop::channel::grinding::GrindingVerifierChannel`.
//! That trait states the contract both sides obey.

/// A prover channel that can pay a proof of work into the transcript.
///
/// The mirror of `binius_iop::channel::grinding::GrindingVerifierChannel`.
/// It is a trait of its own for the same reason that one is.
/// Grinding is a property of the Fiat-Shamir state rather than of what is committed.
///
/// # Contract
///
/// A difficulty of zero is not a grind, and must leave tape and challenger untouched.
/// The verifier's trait documentation says why that rule lives in the channel.
pub trait GrindingProverChannel {
	/// Searches for a nonce meeting `bits` of difficulty and writes it, returning the nonce.
	///
	/// The expected cost is `2^bits` challenger trials, paid in wall clock.
	///
	/// ## Preconditions
	///
	/// * `bits` is at most [`binius_transcript::MAX_GRINDING_BITS`].
	fn grind(&mut self, bits: usize) -> u64;
}
