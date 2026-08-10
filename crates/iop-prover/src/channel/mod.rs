// Copyright 2026 The Binius Developers

//! Channel abstraction for interactive oracle protocol (IOP) provers.

pub mod naive;

use binius_compute::Allocator;
use binius_field::PackedField;
use binius_iop::channel::OracleSpec;
use binius_ip_prover::channel::IPProverChannel;
use binius_math::{FieldSlice, FieldVec};

/// One transparent multilinear paired with the inner product it is claimed to have.
///
/// The transparent multilinear is public data both parties can evaluate.
/// The claim is the value the prover asserts for `<message, transparent>`.
pub struct TransparentClaim<P: PackedField, A: Allocator> {
	/// The transparent multilinear `t`, over the oracle's own variable count.
	pub transparent: FieldVec<P, A>,
	/// The claimed inner product `<pi, t>`.
	pub claim: P::Scalar,
}

/// One committed oracle opened against one or more transparent multilinears.
///
/// A committed message may be claimed at several points in the same proof.
/// Each point contributes one [`TransparentClaim`], and they open together.
///
/// The message is held once per oracle, not once per claim.
/// So opening a large trace at several points never copies the trace.
pub struct OracleOpening<O, P: PackedField, A: Allocator> {
	/// The handle of the oracle being opened.
	pub oracle: O,
	/// The committed multilinear `pi`, the buffer this oracle was committed with.
	pub message: FieldVec<P, A>,
	/// The claims on this oracle, in the order the verifier queues them.
	///
	/// Must hold at least one claim.
	pub claims: Vec<TransparentClaim<P, A>>,
}

impl<O, P: PackedField, A: Allocator> OracleOpening<O, P, A> {
	/// An opening with a single claim, the common case.
	pub fn single(
		oracle: O,
		message: FieldVec<P, A>,
		transparent: FieldVec<P, A>,
		claim: P::Scalar,
	) -> Self {
		Self {
			oracle,
			message,
			claims: vec![TransparentClaim { transparent, claim }],
		}
	}
}

/// Channel for IOP provers that extends the IP prover channel with oracle operations.
///
/// In an IOP, the prover can:
/// 1. Send field elements to the verifier via `send_*` methods (inherited)
/// 2. Sample random challenges via `sample` (inherited)
/// 3. Commit oracles to the verifier
/// 4. Respond to oracle queries with opening proofs
///
/// # Contract
///
/// The caller must call `send_oracle()` exactly `remaining_oracle_specs().len()` times before
/// calling `prove_oracle_relations()`. Each oracle buffer must match the corresponding
/// specification.
pub trait IOPProverChannel<P: PackedField, A: Allocator>: IPProverChannel<P::Scalar> {
	type Oracle: Clone;

	/// Returns the specifications for the remaining oracles to be committed.
	///
	/// This slice shrinks as oracles are committed via `send_oracle()`.
	fn remaining_oracle_specs(&self) -> &[OracleSpec];

	/// Commits an oracle to the verifier.
	///
	/// # Preconditions
	///
	/// * `remaining_oracle_specs()` must be non-empty.
	/// * `buffer.log_len()` must match the expected length from the next oracle spec.
	fn send_oracle(&mut self, buffer: FieldSlice<P>) -> Self::Oracle;

	/// Generates opening proofs for all oracle linear relations.
	///
	/// One [`OracleOpening`] carries a committed message and every transparent it opens against.
	/// The caller supplies the message, so the channel never stores it.
	///
	/// The channel owns each message and transparent until the opening runs.
	/// Both are drawn from the caller's allocator `A`, so a pooled buffer stays pooled throughout.
	///
	/// The verifier queues a flat relation list and groups it by oracle.
	/// Two orders must agree between the two sides:
	///
	/// ```text
	/// across oracles  -> the order each oracle is first opened in
	/// within one      -> the order of that oracle's claims
	/// ```
	///
	/// # Preconditions
	///
	/// * `remaining_oracle_specs()` must be empty (all oracles committed).
	/// * All oracle handles in `openings` must be valid handles returned by `send_oracle()`.
	/// * Each `message` must match the buffer previously committed via `send_oracle()`.
	/// * Every committed oracle must appear in exactly one opening, with at least one claim.
	fn prove_oracle_relations(
		&mut self,
		openings: impl IntoIterator<Item = OracleOpening<Self::Oracle, P, A>>,
	);
}
