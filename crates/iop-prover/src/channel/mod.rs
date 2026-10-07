// Copyright 2026 The Binius Developers

//! Channel abstraction for interactive oracle protocol (IOP) provers.

pub mod grinding;
pub mod merge;
pub mod naive;

use binius_compute::Allocator;
use binius_field::PackedField;
use binius_iop::channel::OracleSpec;
use binius_ip_prover::channel::IPProverChannel;
use binius_math::{FieldVec, StructuredBuffer};

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
/// specification. The channel owns each oracle buffer from the moment it is sent. A caller that
/// needs a buffer back borrows it with `take_oracle()` and hands it back with `return_oracle()`
/// before the opening.
pub trait IOPProverChannel<P: PackedField, A: Allocator>: IPProverChannel<P::Scalar> {
	type Oracle: Clone;

	/// Returns the specifications for the remaining oracles to be committed.
	///
	/// This slice shrinks as oracles are committed via `send_oracle()`.
	fn remaining_oracle_specs(&self) -> &[OracleSpec];

	/// Commits an oracle to the verifier, taking ownership of its buffer.
	///
	/// # Preconditions
	///
	/// * `remaining_oracle_specs()` must be non-empty.
	/// * `buffer.log_len()` must match the expected length from the next oracle spec.
	///
	/// Only the first [`OracleSpec::len`] entries of `buffer` are the prover's content. The rest is
	/// padding the channel may overwrite arbitrarily.
	fn send_oracle(&mut self, buffer: FieldVec<P, A>) -> Self::Oracle;

	/// Lends a committed oracle's buffer back to the caller.
	///
	/// The buffer is the data as committed, including anything the channel wrote into it.
	///
	/// # Panics
	///
	/// Panics if the oracle's interaction round is still open, or if its buffer is already out.
	fn take_oracle(&mut self, oracle: Self::Oracle) -> FieldVec<P, A>;

	/// Hands a buffer obtained from [`Self::take_oracle`] back to the channel.
	///
	/// # Preconditions
	///
	/// * `buffer` must be the one [`Self::take_oracle`] returned for `oracle`, unchanged.
	/// * Every taken buffer must be returned before the opening runs.
	fn return_oracle(&mut self, oracle: Self::Oracle, buffer: FieldVec<P, A>);

	/// Generates an opening proof for one oracle linear relation.
	///
	/// The relation asserts that `<oracle_poly, transparent> = claim`. An oracle may carry any
	/// number of relations.
	///
	/// The transparent may be zero outside one aligned block, and say so through its structure. A
	/// channel that understands the structure skips the zeros; any other materializes it.
	///
	/// The channel owns the transparent multilinear until the opening runs, so it is drawn from
	/// the caller's allocator `A` — a pooled buffer stays pooled all the way through the opening.
	///
	/// # Preconditions
	///
	/// * `remaining_oracle_specs()` must be empty (all oracles committed).
	/// * `oracle` must be a valid handle returned by `send_oracle()`.
	/// * `transparent.log_len()` must match the oracle's message length.
	/// * The claim must already be bound to the transcript, since the coefficient that batches the
	///   queued relations is drawn only after the queue closes.
	fn prove_oracle_relation(
		&mut self,
		oracle: Self::Oracle,
		transparent: StructuredBuffer<P, A::Vec<P>>,
		claim: P::Scalar,
	);
}
