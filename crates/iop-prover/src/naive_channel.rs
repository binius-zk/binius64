// Copyright 2026 The Binius Developers

//! Naive implementation of IOP prover channel for testing.
//!
//! This module provides [`NaiveProverChannel`], a simple implementation of [`IOPProverChannel`]
//! that writes full polynomial data to the transcript instead of using FRI commitments.
//! This is intended for unit testing of protocols without the overhead of BaseFold/FRI.

use binius_field::{Field, PackedField};
use binius_iop::channel::OracleSpec;
use binius_ip_prover::channel::IPProverChannel;
use binius_math::{FieldBuffer, FieldSlice, inner_product::inner_product_buffers};
use binius_transcript::{
	ProverTranscript,
	fiat_shamir::{CanSample, Challenger},
};

use crate::channel::IOPProverChannel;

/// Oracle handle returned by [`NaiveProverChannel::send_oracle`].
#[derive(Debug, Clone, Copy)]
pub struct NaiveOracle {
	index: usize,
}

/// Stored data for a committed oracle.
struct StoredOracleData<P: PackedField> {
	/// The original polynomial buffer.
	/// For ZK oracles, this is `witness || mask` (size 2^(n_vars+1)).
	/// For non-ZK oracles, this is just the polynomial (size 2^n_vars).
	buffer: FieldBuffer<P>,
	/// Whether this oracle uses ZK blinding.
	is_zk: bool,
}

/// A naive prover channel that writes full polynomial data to the transcript.
///
/// This channel wraps a [`ProverTranscript`] and provides oracle operations by writing
/// the entire polynomial coefficients to the transcript. This is useful for testing
/// protocols without the complexity of FRI/BaseFold.
///
/// # Type Parameters
///
/// - `F`: The field type
/// - `P`: The packed field type with `Scalar = F`
/// - `Challenger_`: The Fiat-Shamir challenger
pub struct NaiveProverChannel<'a, F, P, Challenger_>
where
	F: Field,
	P: PackedField<Scalar = F>,
	Challenger_: Challenger,
{
	/// Prover transcript for Fiat-Shamir (borrowed).
	transcript: &'a mut ProverTranscript<Challenger_>,
	/// Oracle specifications.
	oracle_specs: Vec<OracleSpec>,
	/// Stored oracle data for each committed oracle.
	stored_oracles: Vec<StoredOracleData<P>>,
	/// Next oracle index.
	next_oracle_index: usize,
}

impl<'a, F, P, Challenger_> NaiveProverChannel<'a, F, P, Challenger_>
where
	F: Field,
	P: PackedField<Scalar = F>,
	Challenger_: Challenger,
{
	/// Creates a new naive prover channel.
	///
	/// # Arguments
	///
	/// * `transcript` - The prover transcript for Fiat-Shamir (borrowed mutably)
	/// * `oracle_specs` - Specifications for each oracle to be committed
	pub fn new(
		transcript: &'a mut ProverTranscript<Challenger_>,
		oracle_specs: Vec<OracleSpec>,
	) -> Self {
		Self {
			transcript,
			oracle_specs,
			stored_oracles: Vec::new(),
			next_oracle_index: 0,
		}
	}

	/// Returns a reference to the underlying transcript.
	pub fn transcript(&self) -> &ProverTranscript<Challenger_> {
		self.transcript
	}
}

impl<F, P, Challenger_> IPProverChannel<F> for NaiveProverChannel<'_, F, P, Challenger_>
where
	F: Field,
	P: PackedField<Scalar = F>,
	Challenger_: Challenger,
{
	fn send_one(&mut self, elem: F) {
		self.transcript.message().write_scalar(elem);
	}

	fn send_many(&mut self, elems: &[F]) {
		self.transcript.message().write_scalar_slice(elems);
	}

	fn sample(&mut self) -> F {
		CanSample::sample(&mut self.transcript)
	}
}

impl<F, P, Challenger_> IOPProverChannel<P> for NaiveProverChannel<'_, F, P, Challenger_>
where
	F: Field,
	P: PackedField<Scalar = F>,
	Challenger_: Challenger,
{
	type Oracle = NaiveOracle;

	fn remaining_oracle_specs(&self) -> &[OracleSpec] {
		&self.oracle_specs[self.next_oracle_index..]
	}

	fn send_oracle(&mut self, buffer: FieldSlice<P>) -> Self::Oracle {
		assert!(
			!self.remaining_oracle_specs().is_empty(),
			"send_oracle called but no remaining oracle specs"
		);

		let index = self.next_oracle_index;

		// Extract spec data before any mutable borrows
		let spec_log_msg_len = self.oracle_specs[index].log_msg_len;
		let spec_is_zk = self.oracle_specs[index].is_zk;

		// Validate buffer length matches spec
		let expected_log_len = if spec_is_zk {
			spec_log_msg_len + 1
		} else {
			spec_log_msg_len
		};
		assert_eq!(
			buffer.log_len(),
			expected_log_len,
			"oracle buffer log_len mismatch: expected {expected_log_len}, got {}",
			buffer.log_len()
		);

		// Write all polynomial coefficients to the transcript.
		// This is the "naive" commitment - just send all the data.
		self.transcript
			.message()
			.write_scalar_iter(buffer.iter_scalars());

		// Store the buffer for use in finish()
		let stored_buffer =
			FieldBuffer::new(buffer.log_len(), buffer.as_ref().to_vec().into_boxed_slice());
		self.stored_oracles.push(StoredOracleData {
			buffer: stored_buffer,
			is_zk: spec_is_zk,
		});

		self.next_oracle_index += 1;

		NaiveOracle { index }
	}

	fn finish(mut self, oracle_relations: &[(Self::Oracle, FieldBuffer<P>, P::Scalar)]) {
		assert!(
			self.remaining_oracle_specs().is_empty(),
			"finish called but {} oracle specs remaining",
			self.remaining_oracle_specs().len()
		);

		// For the naive channel, we write the transparent polynomial to the transcript
		// so the verifier can read it and verify the inner product directly.
		for (oracle, transparent_poly, eval_claim) in oracle_relations {
			let index = oracle.index;
			assert!(index < self.stored_oracles.len(), "oracle index {index} out of bounds");

			let log_msg_len = self.oracle_specs[index].log_msg_len;
			let is_zk = self.stored_oracles[index].is_zk;

			// Write the transparent polynomial to the transcript
			self.transcript
				.message()
				.write_scalar_iter(transparent_poly.iter_scalars());

			// Sample evaluation point challenges (verifier will sample the same)
			let _point: Vec<F> = CanSample::sample_vec(&mut self.transcript, log_msg_len);

			// Debug assertion: prover should provide consistent eval claims
			let stored = &self.stored_oracles[index];
			let witness_poly = if is_zk {
				stored.buffer.split_half_ref().0
			} else {
				stored.buffer.to_ref()
			};
			let actual_eval: F = inner_product_buffers(&witness_poly, transparent_poly);
			debug_assert_eq!(
				actual_eval, *eval_claim,
				"NaiveProverChannel: eval_claim mismatch for oracle {index}"
			);
		}
	}
}
