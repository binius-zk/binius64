// Copyright 2026 The Binius Developers

//! Channel abstraction for interactive oracle protocol (IOP) verifiers.

pub mod naive;
pub mod oracle_setup;
pub mod size_tracking;

use binius_field::Field;
use binius_ip::channel::IPVerifierChannel;

use crate::basefold;

/// Error type for IOP verifier channel operations.
#[derive(Debug, thiserror::Error)]
pub enum Error {
	#[error("proof is empty")]
	ProofEmpty,
	#[error("BaseFold verification failed: {0}")]
	BaseFold(#[from] basefold::Error),
	#[error("IP channel error: {0}")]
	IPChannel(#[from] binius_ip::channel::Error),
	#[error("sumcheck error: {0}")]
	Sumcheck(#[from] binius_ip::sumcheck::Error),
	#[error("Merkle channel error: {0}")]
	Merkle(#[from] crate::merkle_channel::Error),
}

/// Specification for an oracle to be committed in the IOP.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OracleSpec {
	/// Log2 of the message length (number of field elements).
	pub log_msg_len: usize,
	/// Whether the oracle is committed with zero-knowledge (hiding) masking.
	///
	/// ZK oracles interleave the message with a fresh mask and are folded by a shared masking
	/// challenge γ in the batched BaseFold opening; non-ZK oracles are committed without a mask.
	pub is_zk: bool,
}

impl OracleSpec {
	/// A non-ZK (unmasked) oracle of the given message length.
	pub const fn new(log_msg_len: usize) -> Self {
		Self {
			log_msg_len,
			is_zk: false,
		}
	}

	/// A ZK (masked, hiding) oracle of the given message length.
	pub const fn new_zk(log_msg_len: usize) -> Self {
		Self {
			log_msg_len,
			is_zk: true,
		}
	}
}

/// A boxed closure that evaluates a transparent MLE at a given point.
///
/// The closure is `'static` and owns every value it reads, sharing large data via `Rc`/`Arc`.
/// A channel that defers the opening can therefore store it and evaluate it later.
pub type TransparentEvalFn<Elem> = Box<dyn Fn(&[Elem]) -> Elem + 'static>;

/// An oracle linear relation specifying an inner product claim between a committed oracle
/// polynomial and a transparent polynomial.
///
/// The claim asserts that `<oracle_poly, transparent_poly> = claim`, where `transparent_poly` is
/// the multilinear extension defined by the `transparent` closure evaluated at the challenge point
/// sampled during the protocol.
pub struct OracleLinearRelation<Oracle, Elem> {
	/// The oracle handle for the committed polynomial.
	pub oracle: Oracle,
	/// A closure that evaluates the transparent MLE at a given point.
	///
	/// The closure receives the challenge point (sampled during `verify_oracle_relations`) and
	/// returns the evaluation of the transparent polynomial's MLE at that point.
	pub transparent: TransparentEvalFn<Elem>,
	/// The claimed inner product of the oracle polynomial and the transparent polynomial.
	pub claim: Elem,
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
/// calling `verify_oracle_relations()`. The oracles must be received in order and match their
/// specifications.
pub trait IOPVerifierChannel<F: Field>: IPVerifierChannel<F, Elem: 'static> {
	type Oracle: Clone;

	/// Returns the specifications for the remaining oracles to be received.
	///
	/// This slice shrinks as oracles are received via `recv_oracle()`.
	fn remaining_oracle_specs(&self) -> &[OracleSpec];

	/// Receives an oracle commitment from the prover.
	///
	/// The caller describes the oracle being received: `log_msg_len` is the log2 of the message
	/// length, and `is_witness_dependent` is whether the oracle's contents depend on the witness.
	/// These let a channel record the oracle's [`OracleSpec`] rather than requiring the specs to be
	/// supplied up front. The resulting oracle is zero-knowledge iff the channel is configured for
	/// ZK *and* the oracle is witness-dependent — a non-witness-dependent oracle (e.g. a
	/// pre-indexed commitment to the wiring matrix for succinctness, a planned feature) is never
	/// masked.
	fn recv_oracle(
		&mut self,
		log_msg_len: usize,
		is_witness_dependent: bool,
	) -> Result<Self::Oracle, Error>;

	/// Queues oracle linear relations to be opened.
	///
	/// Implementations may either verify the relations immediately, or queue them and defer the
	/// actual opening (masking + sumcheck + FRI) to `finish()`. Either way, each
	/// relation asserts that `<oracle_poly, transparent_poly> = claim`.
	///
	/// The transparent closures are `'static` and own their captures.
	/// An implementation that defers the opening can store the relations and evaluate them later.
	///
	/// # Preconditions
	///
	/// * All oracle handles in `oracle_relations` must be valid handles returned by
	///   `recv_oracle()`.
	fn verify_oracle_relations(
		&mut self,
		oracle_relations: impl IntoIterator<Item = OracleLinearRelation<Self::Oracle, Self::Elem>>,
	) -> Result<(), Error>;
}
