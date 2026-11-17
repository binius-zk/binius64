// Copyright 2026 The Binius Developers

//! Channel abstraction for interactive oracle protocol (IOP) provers.

pub mod grinding;
pub mod naive;

use binius_compute::Allocator;
use binius_field::PackedField;
use binius_iop::channel::OracleSpec;
use binius_ip_prover::channel::IPProverChannel;
use binius_math::{FieldBuffer, FieldSlice};

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
/// calling `prove_oracle_relation()`. Each oracle buffer must match the corresponding
/// specification.
pub trait IOPProverChannel<P: PackedField, A: Allocator>: IPProverChannel<P::Scalar> {
	type Oracle: Clone;

	/// Returns the specifications for the remaining oracles to be committed.
	///
	/// This slice shrinks as oracles are committed via `send_oracle()`.
	fn remaining_oracle_specs(&self) -> &[OracleSpec];

	/// Commits an oracle to the verifier.
	///
	/// The channel reads the first [`OracleSpec::len`] entries of `oracle` and keeps its own copy;
	/// the caller's buffer stays the caller's. The channel decides what occupies the
	/// `2^log_msg_len - len` entries above the content, and no query can see them.
	///
	/// # Preconditions
	///
	/// * `remaining_oracle_specs()` must be non-empty.
	/// * `oracle.log_len()` must equal [`OracleSpec::log_msg_len`] of the next oracle spec.
	fn send_oracle(&mut self, oracle: FieldSlice<'_, P>) -> Self::Oracle;

	/// Generates an opening proof for one linear relation over committed oracles.
	///
	/// The relation asserts that `Σ_i <oracle_i, transparent_i> = claim` over its operands. An
	/// oracle may appear in any number of relations, and more than once in one.
	///
	/// Each transparent must vanish on every index at or above its oracle's [`OracleSpec::len`],
	/// so the relation does not depend on the channel's padding.
	///
	/// The channel owns the transparents until the opening runs, so they are drawn from the
	/// caller's allocator `A` — a pooled buffer stays pooled all the way through the opening.
	///
	/// # Preconditions
	///
	/// * `remaining_oracle_specs()` must be empty (all oracles committed).
	/// * Every operand's oracle must be a valid handle returned by `send_oracle()`.
	/// * Every operand's `transparent.log_len()` must match its oracle's message length.
	/// * The claim must already be bound to the transcript, since the coefficient that batches the
	///   queued relations is drawn only after the queue closes.
	fn prove_multi_oracle_relation(
		&mut self,
		operands: impl IntoIterator<Item = (Self::Oracle, FieldBuffer<P, A::Vec<P>>)>,
		claim: P::Scalar,
	);

	/// Generates an opening proof for one oracle linear relation `<oracle, transparent> = claim`.
	///
	/// This is [`Self::prove_multi_oracle_relation`] with a single operand.
	fn prove_oracle_relation(
		&mut self,
		oracle: Self::Oracle,
		transparent: FieldBuffer<P, A::Vec<P>>,
		claim: P::Scalar,
	) {
		self.prove_multi_oracle_relation([(oracle, transparent)], claim);
	}
}
