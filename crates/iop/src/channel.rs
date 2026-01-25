// Copyright 2026 The Binius Developers

//! Channel abstraction for interactive oracle protocol (IOP) verifiers.
//!
//! An IOP extends the public-coin interactive protocol model with oracle access: the prover can
//! commit to oracles (e.g., Merkle trees) that the verifier can query at specific positions. This
//! module provides the [`IOPVerifierChannel`] trait that models the verifier's view of such an
//! interaction.
//!
//! The trait extends [`IPVerifierChannel`] with additional methods for:
//! - Receiving oracle commitments from the prover
//! - Querying oracle positions and receiving opening proofs
//!
//! This abstraction allows protocol implementations to be generic over the underlying
//! communication and commitment mechanisms.

use binius_field::Field;
use binius_ip::{MultilinearRationalEvalClaim, channel::IPVerifierChannel};

use crate::basefold;

/// Error type for IOP verifier channel operations.
#[derive(Debug, thiserror::Error)]
pub enum Error {
	#[error("proof is empty")]
	ProofEmpty,
	#[error("BaseFold verification failed: {0}")]
	BaseFold(#[from] basefold::Error),
}

/// Specification for an oracle to be committed in the IOP.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OracleSpec {
	/// Log2 of the message length (number of field elements).
	pub log_msg_len: usize,
	/// Whether this oracle uses zero-knowledge blinding.
	pub is_zk: bool,
}

/// Channel for IOP verifiers that extends the IP verifier channel with oracle operations.
///
/// In an IOP, the verifier can:
/// 1. Receive field elements from the prover via `recv_*` methods (inherited)
/// 2. Sample random challenges via `sample` (inherited)
/// 3. Receive oracle commitments from the prover
/// 4. Query oracles at specific positions and verify opening proofs
///
/// # Contract
///
/// The caller must call `recv_oracle()` exactly `remaining_oracle_specs().len()` times before
/// calling `finish()`. The oracles must be received in order and match their specifications.
pub trait IOPVerifierChannel<F: Field>: IPVerifierChannel<F> {
	type Oracle: Clone;

	/// Returns the specifications for the remaining oracles to be received.
	///
	/// This slice shrinks as oracles are received via `recv_oracle()`.
	fn remaining_oracle_specs(&self) -> &[OracleSpec];

	/// Receives an oracle commitment from the prover.
	///
	/// # Preconditions
	///
	/// `remaining_oracle_specs()` must be non-empty.
	fn recv_oracle(&mut self) -> Result<Self::Oracle, Error>;

	/// Finishes the protocol by opening all oracle relations.
	///
	/// Each oracle relation is a pair of (oracle, claimed_eval) where claimed_eval is the
	/// claimed evaluation of the oracle's polynomial multiplied by a transparent polynomial.
	///
	/// Returns a vector of [`MultilinearRationalEvalClaim`]s, one per oracle relation. Each claim
	/// contains:
	/// - `eval_numerator`: the final sumcheck value
	/// - `eval_denominator`: the FRI evaluation of the committed polynomial
	/// - `point`: the evaluation point derived from sumcheck challenges
	///
	/// The caller must verify: `eval_numerator == eval_denominator * transparent_poly_eval(point)`
	///
	/// # Preconditions
	///
	/// * `remaining_oracle_specs()` must be empty (all oracles received).
	/// * All oracle handles in `oracle_relations` must be valid handles returned by
	///   `recv_oracle()`.
	fn finish(
		self,
		oracle_relations: &[(Self::Oracle, F)],
	) -> Result<Vec<MultilinearRationalEvalClaim<F>>, Error>;
}
