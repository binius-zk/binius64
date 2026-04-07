// Copyright 2026 The Binius Developers

//! ZK-wrapped verifier channel that delegates to a BaseFold ZK channel and an outer IOP verifier.
//!
//! [`ZKWrappedVerifierChannel`] wraps a [`BaseFoldZKVerifierChannel`] and an [`IOPVerifier`],
//! recording all channel values as outer public inputs. In [`finish()`], it prepends constants,
//! pads to the required public size, and runs the outer verifier against the inner channel.
//!
//! [`finish()`]: ZKWrappedVerifierChannel::finish

use binius_field::BinaryField;
use binius_iop::{
	basefold_zk_channel::{BaseFoldZKOracle, BaseFoldZKVerifierChannel},
	channel::{IOPVerifierChannel, OracleLinearRelation, OracleSpec},
	merkle_tree::MerkleTreeScheme,
};
use binius_ip::channel::IPVerifierChannel;
use binius_transcript::fiat_shamir::Challenger;
use binius_utils::DeserializeBytes;

use crate::{Error, IOPVerifier};

/// A verifier channel that wraps a [`BaseFoldZKVerifierChannel`] and an [`IOPVerifier`].
///
/// All values received, sampled, and observed through this channel are recorded as public inputs
/// to the outer constraint system. When [`finish()`](Self::finish) is called, the outer verifier
/// is run against the inner channel to verify the proof.
pub struct ZKWrappedVerifierChannel<'a, F, MTScheme, Challenger_>
where
	F: BinaryField,
	MTScheme: MerkleTreeScheme<F>,
	Challenger_: Challenger,
{
	inner_channel: BaseFoldZKVerifierChannel<'a, F, MTScheme, Challenger_>,
	outer_verifier: &'a IOPVerifier<F>,
	public_values: Vec<F>,
	n_outer_oracles: usize,
}

impl<'a, F, MTScheme, Challenger_> ZKWrappedVerifierChannel<'a, F, MTScheme, Challenger_>
where
	F: BinaryField,
	MTScheme: MerkleTreeScheme<F, Digest: DeserializeBytes>,
	Challenger_: Challenger,
{
	/// Creates a new ZK-wrapped verifier channel.
	///
	/// The outer verifier's oracle specs must be a suffix of the inner channel's oracle specs.
	///
	/// # Panics
	///
	/// Panics if the outer oracle specs are not a suffix of the channel's oracle specs.
	pub fn new(
		inner_channel: BaseFoldZKVerifierChannel<'a, F, MTScheme, Challenger_>,
		outer_verifier: &'a IOPVerifier<F>,
	) -> Self {
		let outer_oracle_specs = outer_verifier.oracle_specs();
		let channel_oracle_specs = inner_channel.remaining_oracle_specs();

		let n_outer = outer_oracle_specs.len();
		let n_total = channel_oracle_specs.len();
		assert!(
			n_outer <= n_total,
			"outer oracle specs ({n_outer}) exceed channel oracle specs ({n_total})"
		);

		let suffix = &channel_oracle_specs[n_total - n_outer..];
		assert_eq!(
			suffix, &outer_oracle_specs,
			"outer oracle specs must be a suffix of channel oracle specs"
		);

		let outer_public_size = 1 << outer_verifier.constraint_system().log_public();
		Self {
			inner_channel,
			outer_verifier,
			public_values: Vec::with_capacity(outer_public_size),
			n_outer_oracles: n_outer,
		}
	}

	/// Consumes the channel and runs the outer verifier.
	///
	/// Prepends the outer constraint system's constants to the recorded public values, pads to
	/// the required public size, and runs [`IOPVerifier::verify`] against the inner channel.
	///
	/// Returns the full public input vector on success.
	pub fn finish(mut self) -> Result<Vec<F>, Error> {
		let outer_cs = self.outer_verifier.constraint_system();
		let public_size = 1 << outer_cs.log_public();

		let mut public = outer_cs.constants().to_vec();
		public.append(&mut self.public_values);
		public.resize(public_size, F::ZERO);

		// IOPVerifier::verify takes Vec<Channel::Elem>, not &[F].
		self.outer_verifier
			.verify(public.clone(), &mut self.inner_channel)?;
		Ok(public)
	}
}

impl<F, MTScheme, Challenger_> IPVerifierChannel<F>
	for ZKWrappedVerifierChannel<'_, F, MTScheme, Challenger_>
where
	F: BinaryField,
	MTScheme: MerkleTreeScheme<F, Digest: DeserializeBytes>,
	Challenger_: Challenger,
{
	type Elem = F;

	fn recv_one(&mut self) -> Result<F, binius_ip::channel::Error> {
		let val = self.inner_channel.recv_one()?;
		self.public_values.push(val);
		Ok(val)
	}

	fn recv_many(&mut self, n: usize) -> Result<Vec<F>, binius_ip::channel::Error> {
		let vals = self.inner_channel.recv_many(n)?;
		self.public_values.extend_from_slice(&vals);
		Ok(vals)
	}

	fn recv_array<const N: usize>(&mut self) -> Result<[F; N], binius_ip::channel::Error> {
		let vals = self.inner_channel.recv_array::<N>()?;
		self.public_values.extend_from_slice(&vals);
		Ok(vals)
	}

	fn sample(&mut self) -> F {
		let val = self.inner_channel.sample();
		self.public_values.push(val);
		val
	}

	fn observe_one(&mut self, val: F) -> F {
		let elem = self.inner_channel.observe_one(val);
		self.public_values.push(elem);
		elem
	}

	fn observe_many(&mut self, vals: &[F]) -> Vec<F> {
		let elems = self.inner_channel.observe_many(vals);
		self.public_values.extend_from_slice(&elems);
		elems
	}

	fn assert_zero(&mut self, _val: F) -> Result<(), binius_ip::channel::Error> {
		// No-op: inner assertions are checked by the outer verifier.
		Ok(())
	}
}

impl<F, MTScheme, Challenger_> IOPVerifierChannel<F>
	for ZKWrappedVerifierChannel<'_, F, MTScheme, Challenger_>
where
	F: BinaryField,
	MTScheme: MerkleTreeScheme<F, Digest: DeserializeBytes>,
	Challenger_: Challenger,
{
	type Oracle = BaseFoldZKOracle;

	fn remaining_oracle_specs(&self) -> &[OracleSpec] {
		let all = self.inner_channel.remaining_oracle_specs();
		let n_remaining_inner = all.len() - self.n_outer_oracles;
		&all[..n_remaining_inner]
	}

	fn recv_oracle(&mut self) -> Result<Self::Oracle, binius_iop::channel::Error> {
		assert!(
			!self.remaining_oracle_specs().is_empty(),
			"recv_oracle called but no remaining inner oracle specs"
		);
		self.inner_channel.recv_oracle()
	}

	fn verify_oracle_relations<'b>(
		&mut self,
		oracle_relations: impl IntoIterator<Item = OracleLinearRelation<'b, Self::Oracle, Self::Elem>>,
	) -> Result<(), binius_iop::channel::Error> {
		self.inner_channel.verify_oracle_relations(oracle_relations)
	}
}
