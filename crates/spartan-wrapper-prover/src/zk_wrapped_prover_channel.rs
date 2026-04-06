// Copyright 2026 The Binius Developers

//! ZK-wrapped prover channel that runs an inner Spartan proof and then proves the outer
//! wrapper constraint system.
//!
//! [`ZKWrappedProverChannel`] wraps a [`BaseFoldZKProverChannel`] and records all channel values.
//! On `send_*`/`sample`/`observe_*`, it delegates to the inner BaseFoldZK channel and records
//! each value. After the inner proof is run, [`finish`] replays the recorded interaction through
//! a [`ReplayChannel`] to fill the outer witness, then runs the outer IOP prover.
//!
//! [`BaseFoldZKProverChannel`]: binius_iop_prover::basefold_zk_channel::BaseFoldZKProverChannel
//! [`ReplayChannel`]: binius_spartan_wrapper::ReplayChannel
//! [`finish`]: ZKWrappedProverChannel::finish

use binius_field::{BinaryField128bGhash as B128, PackedExtension, PackedField};
use binius_iop::{channel::OracleSpec, merkle_tree::MerkleTreeScheme};
use binius_iop_prover::{
	basefold_zk_channel::{BaseFoldZKOracle, BaseFoldZKProverChannel},
	channel::IOPProverChannel,
	merkle_tree::MerkleTreeProver,
};
use binius_ip::channel::IPVerifierChannel;
use binius_ip_prover::channel::IPProverChannel;
use binius_math::{FieldBuffer, FieldSlice, ntt::AdditiveNTT};
use binius_spartan_frontend::constraint_system::WitnessLayout;
use binius_spartan_prover::IOPProver;
use binius_spartan_verifier::IOPVerifier;
use binius_spartan_wrapper::ReplayChannel;
use binius_transcript::fiat_shamir::Challenger;
use binius_utils::SerializeBytes;
use rand::CryptoRng;

/// A prover channel that wraps a [`BaseFoldZKProverChannel`] and an outer Spartan IOP prover.
///
/// This channel records all channel values. On
/// `send_*`/`sample`/`observe_*`, it delegates to the inner BaseFoldZK channel and records each
/// value. After the inner proof is run through this channel, call
/// [`finish`](Self::finish) to replay the interaction through a [`ReplayChannel`], fill the outer
/// witness, and generate the outer proof.
pub struct ZKWrappedProverChannel<'a, P, NTT, MTProver, Challenger_>
where
	P: PackedField<Scalar = B128>,
	NTT: AdditiveNTT<Field = B128> + Sync,
	MTProver: MerkleTreeProver<B128>,
	Challenger_: Challenger,
{
	inner_channel: BaseFoldZKProverChannel<'a, B128, P, NTT, MTProver, Challenger_>,
	outer_prover: &'a IOPProver,
	inner_verifier: &'a IOPVerifier,
	outer_layout: &'a WitnessLayout<B128>,
	interaction: Vec<B128>,
	n_outer_oracles: usize,
}

impl<'a, P, NTT, MTScheme, MTProver, Challenger_>
	ZKWrappedProverChannel<'a, P, NTT, MTProver, Challenger_>
where
	P: PackedField<Scalar = B128> + PackedExtension<B128>,
	NTT: AdditiveNTT<Field = B128> + Sync,
	MTScheme: MerkleTreeScheme<B128, Digest: SerializeBytes>,
	MTProver: MerkleTreeProver<B128, Scheme = MTScheme>,
	Challenger_: Challenger,
{
	/// Creates a new ZK-wrapped prover channel.
	///
	/// # Arguments
	///
	/// * `inner_channel` - The BaseFold ZK channel with oracle specs for both inner and outer
	///   proofs
	/// * `outer_prover` - The IOP prover for the outer (wrapper) constraint system
	/// * `inner_verifier` - The IOP verifier for the inner constraint system (used for replay)
	/// * `outer_layout` - The witness layout for the outer constraint system
	pub fn new(
		inner_channel: BaseFoldZKProverChannel<'a, B128, P, NTT, MTProver, Challenger_>,
		outer_prover: &'a IOPProver,
		inner_verifier: &'a IOPVerifier,
		outer_layout: &'a WitnessLayout<B128>,
	) -> Self {
		let outer_oracle_specs =
			IOPVerifier::new(outer_prover.constraint_system().clone()).oracle_specs();
		let all_specs = inner_channel.remaining_oracle_specs();
		let n_outer = outer_oracle_specs.len();
		assert!(
			all_specs.len() >= n_outer,
			"outer oracle specs ({n_outer}) exceed channel oracle specs ({})",
			all_specs.len(),
		);
		assert_eq!(
			&all_specs[all_specs.len() - n_outer..],
			&outer_oracle_specs,
			"outer oracle specs must be a suffix of channel oracle specs",
		);

		Self {
			inner_channel,
			outer_prover,
			inner_verifier,
			outer_layout,
			interaction: Vec::new(),
			n_outer_oracles: n_outer,
		}
	}

	/// Consumes the channel and runs the outer proof.
	///
	/// This should be called after the inner proof has been run through this channel
	/// (via [`IOPProver::prove`]). It:
	/// 1. Replays the recorded interaction through a [`ReplayChannel`] to fill the outer witness
	/// 2. Validates and generates the outer IOP proof
	///
	/// [`ReplayChannel`]: binius_spartan_wrapper::ReplayChannel
	pub fn finish(self, rng: impl CryptoRng) -> Result<(), binius_spartan_prover::Error> {
		let Self {
			inner_channel,
			outer_prover,
			inner_verifier,
			outer_layout,
			interaction,
			..
		} = self;

		// Extract inner public values from the initial events.
		let inner_cs = inner_verifier.constraint_system();
		let inner_public_size = 1 << inner_cs.log_public();
		let public: Vec<B128> = interaction[..inner_public_size].to_vec();

		// Replay the inner verification through the outer witness generator.
		// First observe the public input (mirrors the prover-side observe_many).
		let mut replay_channel = ReplayChannel::new(outer_layout, interaction);
		let inner_public_elems = replay_channel.observe_many(&public);

		// Run the inner verification to fill private wires.
		inner_verifier
			.verify(inner_public_elems, &mut replay_channel)
			.expect("replay verification should not fail");
		let witness = replay_channel
			.finish()
			.expect("outer witness generation should not fail");

		// Validate and generate the outer proof.
		let outer_cs = outer_prover.constraint_system();
		outer_cs.validate(&witness);
		outer_prover.prove::<B128, P, _>(&witness, rng, inner_channel)?;
		Ok(())
	}
}

impl<P, NTT, MTScheme, MTProver, Challenger_> IPProverChannel<B128>
	for &mut ZKWrappedProverChannel<'_, P, NTT, MTProver, Challenger_>
where
	P: PackedField<Scalar = B128> + PackedExtension<B128>,
	NTT: AdditiveNTT<Field = B128> + Sync,
	MTScheme: MerkleTreeScheme<B128, Digest: SerializeBytes>,
	MTProver: MerkleTreeProver<B128, Scheme = MTScheme>,
	Challenger_: Challenger,
{
	fn send_one(&mut self, elem: B128) {
		self.inner_channel.send_one(elem);
		self.interaction.push(elem);
	}

	fn send_many(&mut self, elems: &[B128]) {
		self.inner_channel.send_many(elems);
		self.interaction.extend_from_slice(elems);
	}

	fn observe_one(&mut self, val: B128) {
		self.inner_channel.observe_one(val);
		self.interaction.push(val);
	}

	fn observe_many(&mut self, vals: &[B128]) {
		self.inner_channel.observe_many(vals);
		self.interaction.extend_from_slice(vals);
	}

	fn sample(&mut self) -> B128 {
		let val = self.inner_channel.sample();
		self.interaction.push(val);
		val
	}
}

impl<P, NTT, MTScheme, MTProver, Challenger_> IOPProverChannel<P>
	for &mut ZKWrappedProverChannel<'_, P, NTT, MTProver, Challenger_>
where
	P: PackedField<Scalar = B128> + PackedExtension<B128>,
	NTT: AdditiveNTT<Field = B128> + Sync,
	MTScheme: MerkleTreeScheme<B128, Digest: SerializeBytes>,
	MTProver: MerkleTreeProver<B128, Scheme = MTScheme>,
	Challenger_: Challenger,
{
	type Oracle = BaseFoldZKOracle;

	fn remaining_oracle_specs(&self) -> &[OracleSpec] {
		let remaining = self.inner_channel.remaining_oracle_specs();
		let n_inner_remaining = remaining.len() - self.n_outer_oracles;
		&remaining[..n_inner_remaining]
	}

	fn send_oracle(&mut self, buffer: FieldSlice<P>) -> Self::Oracle {
		assert!(
			!self.remaining_oracle_specs().is_empty(),
			"send_oracle called but no inner oracle specs remaining"
		);
		self.inner_channel.send_oracle(buffer)
	}

	fn prove_oracle_relations(
		&mut self,
		oracle_relations: impl IntoIterator<Item = (Self::Oracle, FieldBuffer<P>, P::Scalar)>,
	) {
		self.inner_channel.prove_oracle_relations(oracle_relations)
	}
}
