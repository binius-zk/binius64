// Copyright 2026 The Binius Developers

//! Channel abstraction for interactive oracle protocol (IOP) provers.

pub mod merge;
pub mod naive;

use binius_compute::Allocator;
use binius_field::PackedField;
use binius_iop::channel::OracleSpec;
use binius_ip_prover::channel::IPProverChannel;
use binius_math::{FieldSlice, FieldVec};

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
/// specification. Every committed oracle must be handed back to the channel exactly once with
/// `finalize_oracle()`.
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

	/// Generates an opening proof for one oracle linear relation.
	///
	/// The relation asserts that `<oracle_poly, transparent> = claim`. An oracle may carry any
	/// number of relations.
	///
	/// The channel owns the transparent multilinear until the opening runs, so it is drawn from
	/// the caller's allocator `A` — a pooled buffer stays pooled all the way through the opening.
	///
	/// # Preconditions
	///
	/// * `remaining_oracle_specs()` must be empty (all oracles committed).
	/// * `oracle` must be a valid handle returned by `send_oracle()`.
	/// * `transparent.log_len()` must match the oracle's message length.
	fn prove_oracle_relation(
		&mut self,
		oracle: Self::Oracle,
		transparent: FieldVec<P, A>,
		claim: P::Scalar,
	);

	/// Gives ownership of the oracle buffer to the channel.
	///
	/// The [`Self::send_oracle`] method takes a borrowed reference to an oracle buffer and returns
	/// a handle to it. In order to prove the oracle relations without unnecessarily cloning the
	/// buffer, some channel implementations require ownership of the buffer.
	///
	/// # Preconditions
	///
	/// * `oracle` must be a valid handle returned by `send_oracle()`, not already finalized.
	/// * `buffer` must equal the buffer previously committed via `send_oracle()`.
	fn finalize_oracle(&mut self, oracle: Self::Oracle, buffer: FieldVec<P, A>);
}
