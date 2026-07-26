// Copyright 2026 The Binius Developers

//! Channel abstraction for interactive oracle protocol (IOP) provers.

pub mod naive;

use binius_compute::Allocator;
use binius_field::PackedField;
use binius_iop::channel::OracleSpec;
use binius_ip_prover::channel::IPProverChannel;
use binius_math::{FieldBuffer, FieldSlice, FieldVec};

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
	/// Each item is `(oracle, message, transparent_poly, eval_claim)` where `message` is
	/// the same buffer that was passed to `send_oracle()` for this oracle. Callers provide
	/// the message here so the channel does not need to store it internally.
	///
	/// The channel owns each message until the opening runs, so the message is drawn from the
	/// caller's allocator `A` — a pooled message stays pooled all the way through the opening.
	/// The transparent multilinear is `Vec`-backed: every producer of one builds it from a
	/// `binius-math` tensor expansion that takes no allocator.
	///
	/// # Preconditions
	///
	/// * `remaining_oracle_specs()` must be empty (all oracles committed).
	/// * All oracle handles in `oracle_relations` must be valid handles returned by
	///   `send_oracle()`.
	/// * Each `message` must match the buffer previously committed via `send_oracle()`.
	fn prove_oracle_relations(
		&mut self,
		oracle_relations: impl IntoIterator<
			Item = (Self::Oracle, FieldVec<P, A>, FieldBuffer<P>, P::Scalar),
		>,
	);
}
