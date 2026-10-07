// Copyright 2026 The Binius Developers

//! Naive [`IOPProverChannel`] for testing: writes full polynomials instead of using FRI.

use binius_compute::GlobalAllocator;
use binius_field::{Field, PackedField};
use binius_iop::channel::OracleSpec;
use binius_ip_prover::channel::IPProverChannel;
use binius_math::{FieldBuffer, StructuredBuffer};
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
	/// The scalars of every committed oracle, by index, and `None` while the caller has it out.
	///
	/// Stored unpacked because the channel is not generic over the packing; a take repacks them.
	messages: Vec<Option<Vec<F>>>,
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
	pub const fn new(
		transcript: &'a mut ProverTranscript<Challenger_>,
		oracle_specs: Vec<OracleSpec>,
	) -> Self {
		Self {
			transcript,
			oracle_specs,
			messages: Vec::new(),
			next_oracle_index: 0,
			_f: std::marker::PhantomData,
		}
	}

	/// Returns a reference to the underlying transcript.
	pub const fn transcript(&self) -> &ProverTranscript<Challenger_> {
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

// Pinned to [`GlobalAllocator`], whose `Vec<P>` is a plain `Vec<P>`: this channel writes every
// buffer straight to the transcript and never draws from a pool, so being generic over the
// allocator would only force call sites to name one.
impl<F, P, Challenger_> IOPProverChannel<P, GlobalAllocator>
	for NaiveProverChannel<'_, F, Challenger_>
where
	F: Field,
	P: PackedField<Scalar = F>,
	Challenger_: Challenger,
{
	type Oracle = NaiveOracle;

	fn remaining_oracle_specs(&self) -> &[OracleSpec] {
		&self.oracle_specs[self.next_oracle_index..]
	}

	fn send_oracle(&mut self, buffer: FieldBuffer<P>) -> Self::Oracle {
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

		self.messages.push(Some(buffer.iter_scalars().collect()));
		self.next_oracle_index += 1;

		NaiveOracle { index }
	}

	fn prove_oracle_relation(
		&mut self,
		oracle: Self::Oracle,
		transparent: StructuredBuffer<P, Vec<P>>,
		_claim: P::Scalar,
	) {
		// For the naive channel, we write the transparent polynomial to the transcript so the
		// verifier can read it and check the inner product against the message it already read.
		let index = oracle.index;
		assert!(index < self.messages.len(), "oracle index {index} out of bounds");

		let log_msg_len = self.oracle_specs[index].log_msg_len;
		assert_eq!(
			transparent.log_len(),
			log_msg_len,
			"transparent log_len mismatch: expected {log_msg_len}, got {}",
			transparent.log_len()
		);

		// Write the transparent polynomial to the transcript
		let transparent = transparent.materialize(&GlobalAllocator);
		self.transcript
			.message()
			.write_scalar_iter(transparent.iter_scalars());

		// Sample evaluation point challenges (verifier will sample the same)
		let _point: Vec<F> = CanSample::sample_vec(&mut self.transcript, log_msg_len);
	}

	fn take_oracle(&mut self, oracle: Self::Oracle) -> FieldBuffer<P> {
		let values = self.messages[oracle.index]
			.take()
			.unwrap_or_else(|| panic!("oracle {} is already taken", oracle.index));
		FieldBuffer::from_values(&values)
	}

	fn return_oracle(&mut self, oracle: Self::Oracle, buffer: FieldBuffer<P>) {
		let message = &mut self.messages[oracle.index];
		assert!(message.is_none(), "oracle {} was not taken", oracle.index);
		*message = Some(buffer.iter_scalars().collect());
	}
}
