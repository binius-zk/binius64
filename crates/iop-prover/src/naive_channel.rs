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

/// A naive prover channel that writes full polynomial data to the transcript.
///
/// This channel wraps a [`ProverTranscript`] and provides oracle operations by writing
/// the entire polynomial coefficients to the transcript. This is useful for testing
/// protocols without the complexity of FRI/BaseFold.
///
/// # Type Parameters
///
/// - `F`: The field type
/// - `Challenger_`: The Fiat-Shamir challenger
pub struct NaiveProverChannel<'a, F, Challenger_>
where
	F: Field,
	Challenger_: Challenger,
{
	/// Prover transcript for Fiat-Shamir (borrowed).
	transcript: &'a mut ProverTranscript<Challenger_>,
	/// Oracle specifications.
	oracle_specs: Vec<OracleSpec>,
	/// Number of oracles committed so far.
	n_committed: usize,
	/// Next oracle index.
	next_oracle_index: usize,
	_f: std::marker::PhantomData<F>,
}

impl<'a, F, Challenger_> NaiveProverChannel<'a, F, Challenger_>
where
	F: Field,
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
			n_committed: 0,
			next_oracle_index: 0,
			_f: std::marker::PhantomData,
		}
	}

	/// Returns a reference to the underlying transcript.
	pub fn transcript(&self) -> &ProverTranscript<Challenger_> {
		self.transcript
	}

	/// Consumes the channel, asserting all oracle specs have been consumed.
	pub fn finish(self) {
		let n_remaining = self.oracle_specs.len() - self.next_oracle_index;
		assert!(n_remaining == 0, "finish called but {n_remaining} oracle specs remaining",);
	}
}

impl<F, Challenger_> IPProverChannel<F> for NaiveProverChannel<'_, F, Challenger_>
where
	F: Field,
	Challenger_: Challenger,
{
	fn send_one(&mut self, elem: F) {
		self.transcript.message().write_scalar(elem);
	}

	fn send_many(&mut self, elems: &[F]) {
		self.transcript.message().write_scalar_slice(elems);
	}

	fn observe_one(&mut self, val: F) {
		self.transcript.observe().write_scalar(val);
	}

	fn observe_many(&mut self, vals: &[F]) {
		self.transcript.observe().write_scalar_slice(vals);
	}

	fn sample(&mut self) -> F {
		CanSample::sample(&mut self.transcript)
	}
}

impl<F, P, Challenger_> IOPProverChannel<P> for NaiveProverChannel<'_, F, Challenger_>
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
		let index = self.next_oracle_index;
		assert!(
			index < self.oracle_specs.len(),
			"send_oracle called but no remaining oracle specs"
		);

		// Validate buffer length matches spec
		let spec_log_msg_len = self.oracle_specs[index].log_msg_len;
		assert_eq!(
			buffer.log_len(),
			spec_log_msg_len,
			"oracle buffer log_len mismatch: expected {spec_log_msg_len}, got {}",
			buffer.log_len()
		);

		// Write all polynomial coefficients to the transcript.
		// This is the "naive" commitment - just send all the data.
		self.transcript
			.message()
			.write_scalar_iter(buffer.iter_scalars());

		self.n_committed += 1;
		self.next_oracle_index += 1;

		NaiveOracle { index }
	}

	fn prove_oracle_relations(
		&mut self,
		oracle_relations: impl IntoIterator<
			Item = (Self::Oracle, FieldBuffer<P>, FieldBuffer<P>, P::Scalar),
		>,
	) {
		// For the naive channel, we write the transparent polynomial to the transcript
		// so the verifier can read it and verify the inner product directly.
		for (oracle, message, transparent_poly, eval_claim) in oracle_relations {
			let index = oracle.index;
			assert!(index < self.n_committed, "oracle index {index} out of bounds");

			let log_msg_len = self.oracle_specs[index].log_msg_len;
			assert_eq!(
				message.log_len(),
				log_msg_len,
				"oracle message log_len mismatch: expected {log_msg_len}, got {}",
				message.log_len()
			);

			// Write the transparent polynomial to the transcript
			self.transcript
				.message()
				.write_scalar_iter(transparent_poly.iter_scalars());

			// Sample evaluation point challenges (verifier will sample the same)
			let _point: Vec<F> = CanSample::sample_vec(&mut self.transcript, log_msg_len);

			// Debug assertion: prover should provide consistent eval claims
			let actual_eval: F = inner_product_buffers(&message, &transparent_poly);
			debug_assert_eq!(
				actual_eval, eval_claim,
				"NaiveProverChannel: eval_claim mismatch for oracle {index}"
			);
		}
	}
}
