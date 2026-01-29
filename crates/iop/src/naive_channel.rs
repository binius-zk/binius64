// Copyright 2026 The Binius Developers

//! Naive implementation of IOP verifier channel for testing.
//!
//! This module provides [`NaiveVerifierChannel`], a simple implementation of [`IOPVerifierChannel`]
//! that reads full polynomial data from the transcript instead of verifying FRI commitments.
//! This is intended for unit testing of protocols without the overhead of BaseFold/FRI.

use binius_field::Field;
use binius_ip::{MultilinearRationalEvalClaim, channel::IPVerifierChannel};
use binius_math::{
	FieldBuffer, inner_product::inner_product_buffers, multilinear::evaluate::evaluate_inplace,
};
use binius_transcript::{
	VerifierTranscript,
	fiat_shamir::{CanSample, Challenger},
};

use crate::channel::{Error, IOPVerifierChannel, OracleSpec};

/// Oracle handle returned by [`NaiveVerifierChannel::recv_oracle`].
#[derive(Debug, Clone, Copy)]
pub struct NaiveOracle {
	index: usize,
}

/// A naive verifier channel that reads full polynomial data from the transcript.
///
/// This channel wraps a [`VerifierTranscript`] and provides oracle operations by reading
/// the entire polynomial coefficients from the transcript. This is useful for testing
/// protocols without the complexity of FRI/BaseFold.
///
/// # Type Parameters
///
/// - `F`: The field type
/// - `Challenger_`: The Fiat-Shamir challenger
pub struct NaiveVerifierChannel<'a, F, Challenger_>
where
	F: Field,
	Challenger_: Challenger,
{
	/// Verifier transcript for Fiat-Shamir (borrowed).
	transcript: &'a mut VerifierTranscript<Challenger_>,
	/// Oracle specifications (borrowed).
	oracle_specs: &'a [OracleSpec],
	/// Stored polynomial buffers read from the transcript.
	/// For ZK oracles, this stores the combined polynomial (witness || mask).
	/// For non-ZK oracles, this stores the full polynomial.
	stored_polynomials: Vec<FieldBuffer<F>>,
	/// Next oracle index.
	next_oracle_index: usize,
}

impl<'a, F, Challenger_> NaiveVerifierChannel<'a, F, Challenger_>
where
	F: Field,
	Challenger_: Challenger,
{
	/// Creates a new naive verifier channel.
	///
	/// # Arguments
	///
	/// * `transcript` - The verifier transcript for Fiat-Shamir (borrowed mutably)
	/// * `oracle_specs` - Specifications for each oracle to be received (borrowed)
	pub fn new(
		transcript: &'a mut VerifierTranscript<Challenger_>,
		oracle_specs: &'a [OracleSpec],
	) -> Self {
		Self {
			transcript,
			oracle_specs,
			stored_polynomials: Vec::new(),
			next_oracle_index: 0,
		}
	}

	/// Returns a reference to the underlying transcript.
	pub fn transcript(&self) -> &VerifierTranscript<Challenger_> {
		self.transcript
	}
}

impl<F, Challenger_> IPVerifierChannel<F> for NaiveVerifierChannel<'_, F, Challenger_>
where
	F: Field,
	Challenger_: Challenger,
{
	fn recv_one(&mut self) -> Result<F, binius_ip::channel::Error> {
		self.transcript
			.message()
			.read_scalar()
			.map_err(|_| binius_ip::channel::Error::ProofEmpty)
	}

	fn recv_many(&mut self, n: usize) -> Result<Vec<F>, binius_ip::channel::Error> {
		self.transcript
			.message()
			.read_scalar_slice(n)
			.map_err(|_| binius_ip::channel::Error::ProofEmpty)
	}

	fn recv_array<const N: usize>(&mut self) -> Result<[F; N], binius_ip::channel::Error> {
		self.transcript
			.message()
			.read()
			.map_err(|_| binius_ip::channel::Error::ProofEmpty)
	}

	fn sample(&mut self) -> F {
		CanSample::sample(&mut self.transcript)
	}
}

impl<F, Challenger_> IOPVerifierChannel<F> for NaiveVerifierChannel<'_, F, Challenger_>
where
	F: Field,
	Challenger_: Challenger,
{
	type Oracle = NaiveOracle;

	fn remaining_oracle_specs(&self) -> &[OracleSpec] {
		&self.oracle_specs[self.next_oracle_index..]
	}

	fn recv_oracle(&mut self) -> Result<Self::Oracle, Error> {
		assert!(
			!self.remaining_oracle_specs().is_empty(),
			"recv_oracle called but no remaining oracle specs"
		);

		let index = self.next_oracle_index;
		let spec = &self.oracle_specs[index];

		// For ZK oracles, the prover sends witness || mask (2^(n+1) elements).
		// For non-ZK oracles, the prover sends the full polynomial (2^n elements).
		let buffer_len = if spec.is_zk {
			1 << (spec.log_msg_len + 1)
		} else {
			1 << spec.log_msg_len
		};

		// Read all polynomial coefficients from the transcript
		let values = self
			.transcript
			.message()
			.read_scalar_slice(buffer_len)
			.map_err(|_| Error::ProofEmpty)?;

		self.stored_polynomials
			.push(FieldBuffer::from_values(&values));
		self.next_oracle_index += 1;

		Ok(NaiveOracle { index })
	}

	fn finish(
		mut self,
		oracle_relations: &[(Self::Oracle, F)],
	) -> Result<Vec<MultilinearRationalEvalClaim<F>>, Error> {
		assert!(
			self.remaining_oracle_specs().is_empty(),
			"finish called but {} oracle specs remaining",
			self.remaining_oracle_specs().len()
		);

		let mut claims = Vec::with_capacity(oracle_relations.len());

		for (oracle, eval_claim) in oracle_relations {
			let index = oracle.index;
			assert!(index < self.stored_polynomials.len(), "oracle index {index} out of bounds");

			// Extract spec data before mutable borrow of transcript
			let log_msg_len = self.oracle_specs[index].log_msg_len;
			let is_zk = self.oracle_specs[index].is_zk;

			// Read the transparent polynomial from the transcript (prover wrote it in finish)
			let transparent_len = 1 << log_msg_len;
			let transparent_values = self
				.transcript
				.message()
				.read_scalar_slice(transparent_len)
				.map_err(|_| Error::ProofEmpty)?;
			let transparent_poly = FieldBuffer::from_values(&transparent_values);

			// Verify the inner product claim directly
			let stored_poly = &self.stored_polynomials[index];
			let witness_poly = if is_zk {
				stored_poly.split_half_ref().0
			} else {
				stored_poly.to_ref()
			};
			let actual_inner_product: F = inner_product_buffers(&witness_poly, &transparent_poly);

			assert_eq!(
				actual_inner_product, *eval_claim,
				"NaiveVerifierChannel: inner product verification failed"
			);

			// Sample evaluation point challenges (same as prover sampled)
			let point: Vec<F> = CanSample::sample_vec(&mut self.transcript, log_msg_len);

			// Compute transparent polynomial evaluation at the sampled point
			let transparent_eval = evaluate_inplace(transparent_poly, &point);

			// Return a claim that trivially passes:
			// eval_numerator == eval_denominator * transparent_poly_eval(point)
			// transparent_eval == 1 * transparent_eval
			claims.push(MultilinearRationalEvalClaim {
				eval_numerator: transparent_eval,
				eval_denominator: F::ONE,
				point,
			});
		}

		Ok(claims)
	}
}
