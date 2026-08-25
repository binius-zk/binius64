// Copyright 2026 The Binius Developers

//! Proof-of-work grinding, as a capability a prover channel may carry.
//!
//! The counterpart of
//! [`GrindingVerifierChannel`](binius_iop::channel::grinding::GrindingVerifierChannel). That trait
//! states the contract both sides obey, and why it is a capability not a method.

/// A prover channel that can pay a proof of work into the transcript.
///
/// The mirror of
/// [`GrindingVerifierChannel`](binius_iop::channel::grinding::GrindingVerifierChannel).
/// It is a trait of its own for the same reason: grinding acts on the Fiat-Shamir state alone.
///
/// # Contract
///
/// A difficulty of zero is not a grind, and must leave tape and challenger untouched.
/// The verifier's trait documents why that rule belongs to the channel rather than the call sites.
pub trait GrindingProverChannel {
	/// Searches for a nonce meeting `bits` of difficulty and writes it, returning the nonce.
	///
	/// The expected cost is `2^bits` challenger trials, paid in wall clock.
	///
	/// ## Preconditions
	///
	/// * `bits` is at most [`MAX_GRINDING_BITS`](binius_transcript::MAX_GRINDING_BITS).
	fn grind(&mut self, bits: usize) -> u64;
}
