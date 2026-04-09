// Copyright 2026 The Binius Developers

//! ZK-wrapped prover channel that runs an inner proof and then proves the outer
//! wrapper constraint system.
//!
//! [`ZKWrappedProverChannel`] wraps a [`BaseFoldZKProverChannel`] and records all channel values.
//! On `send_*`/`sample`/`observe_*`, it delegates to the inner BaseFoldZK channel and records
//! each value. After the inner proof is run, [`finish`] replays the recorded interaction through
//! a caller-provided closure to fill the outer witness, then runs the outer IOP prover.
//!
//! [`BaseFoldZKProverChannel`]: binius_iop_prover::basefold_zk_channel::BaseFoldZKProverChannel
//! [`finish`]: ZKWrappedProverChannel::finish

use binius_field::{BinaryField, PackedExtension, PackedField};
use binius_iop::{channel::OracleSpec, merkle_tree::MerkleTreeScheme};
use binius_iop_prover::{
	basefold_zk_channel::{BaseFoldZKOracle, BaseFoldZKProverChannel},
	channel::IOPProverChannel,
	merkle_tree::MerkleTreeProver,
};
use binius_ip_prover::channel::IPProverChannel;
use binius_math::{FieldBuffer, FieldSlice, ntt::AdditiveNTT};
use binius_spartan_frontend::constraint_system::WitnessLayout;
use binius_spartan_verifier::{IOPVerifier, wrapper::ReplayChannel};
use binius_transcript::fiat_shamir::Challenger;
use binius_utils::SerializeBytes;
use rand::CryptoRng;

use crate::IOPProver;

/// A prover channel that wraps a [`BaseFoldZKProverChannel`] and an outer Spartan IOP prover.
///
/// This channel records all channel values. On
/// `send_*`/`sample`/`observe_*`, it delegates to the inner BaseFoldZK channel and records each
/// value. After the inner proof is run through this channel, call
/// [`finish`](Self::finish) to replay the interaction, fill the outer witness, and generate the
/// outer proof.
///
/// The `ReplayFn` closure is called during [`finish`](Self::finish) with a [`ReplayChannel`] to
/// replay the inner verification and fill the outer witness. This allows the channel to be generic
/// over different inner verification protocols.
pub struct ZKWrappedProverChannel<'a, P, NTT, MTProver, Challenger_, ReplayFn>
where
	P: PackedField<Scalar: BinaryField>,
	NTT: AdditiveNTT<Field = P::Scalar> + Sync,
	MTProver: MerkleTreeProver<P::Scalar>,
	Challenger_: Challenger,
{
	inner_channel: BaseFoldZKProverChannel<'a, P::Scalar, P, NTT, MTProver, Challenger_>,
	outer_prover: &'a IOPProver<P::Scalar>,
	outer_layout: &'a WitnessLayout<P::Scalar>,
	replay_fn: ReplayFn,
	interaction: Vec<P::Scalar>,
	n_outer_oracles: usize,
}

impl<'a, F, P, NTT, MTScheme, MTProver, Challenger_, ReplayFn>
	ZKWrappedProverChannel<'a, P, NTT, MTProver, Challenger_, ReplayFn>
where
	F: BinaryField,
	P: PackedField<Scalar = F> + PackedExtension<F>,
	NTT: AdditiveNTT<Field = F> + Sync,
	MTScheme: MerkleTreeScheme<F, Digest: SerializeBytes>,
	MTProver: MerkleTreeProver<F, Scheme = MTScheme>,
	Challenger_: Challenger,
{
	/// Creates a new ZK-wrapped prover channel.
	///
	/// # Arguments
	///
	/// * `inner_channel` - The BaseFold ZK channel with oracle specs for both inner and outer
	///   proofs
	/// * `outer_prover` - The IOP prover for the outer (wrapper) constraint system
	/// * `outer_layout` - The witness layout for the outer constraint system
	/// * `replay_fn` - Closure called during [`finish`](Self::finish) with a [`ReplayChannel`] to
	///   replay the inner verification and fill the outer witness
	pub fn new(
		inner_channel: BaseFoldZKProverChannel<'a, F, P, NTT, MTProver, Challenger_>,
		outer_prover: &'a IOPProver<F>,
		outer_layout: &'a WitnessLayout<F>,
		replay_fn: ReplayFn,
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
			outer_layout,
			replay_fn,
			interaction: Vec::new(),
			n_outer_oracles: n_outer,
		}
	}

	/// Consumes the channel and runs the outer proof.
	///
	/// This should be called after the inner proof has been run through this channel.
	/// It:
	/// 1. Creates a [`ReplayChannel`] from the recorded interaction
	/// 2. Calls the `replay_fn` closure to replay the inner verification and fill the outer witness
	/// 3. Validates and generates the outer IOP proof
	pub fn finish(self, rng: impl CryptoRng) -> Result<(), crate::Error>
	where
		ReplayFn: FnOnce(&mut ReplayChannel<'_, F>),
	{
		let Self {
			inner_channel,
			outer_prover,
			outer_layout,
			replay_fn,
			interaction,
			..
		} = self;

		// Replay the inner verification through the outer witness generator.
		let mut replay_channel = ReplayChannel::new(outer_layout, interaction);
		replay_fn(&mut replay_channel);
		let witness = replay_channel
			.finish()
			.expect("outer witness generation should not fail");

		// Validate and generate the outer proof.
		let outer_cs = outer_prover.constraint_system();
		outer_cs.validate(&witness);
		outer_prover.prove::<P, _>(&witness, rng, inner_channel)?;
		Ok(())
	}
}

impl<F, P, NTT, MTScheme, MTProver, Challenger_, ReplayFn> IPProverChannel<F>
	for &mut ZKWrappedProverChannel<'_, P, NTT, MTProver, Challenger_, ReplayFn>
where
	F: BinaryField,
	P: PackedField<Scalar = F> + PackedExtension<F>,
	NTT: AdditiveNTT<Field = F> + Sync,
	MTScheme: MerkleTreeScheme<F, Digest: SerializeBytes>,
	MTProver: MerkleTreeProver<F, Scheme = MTScheme>,
	Challenger_: Challenger,
{
	fn send_one(&mut self, elem: F) {
		self.inner_channel.send_one(elem);
		self.interaction.push(elem);
	}

	fn send_many(&mut self, elems: &[F]) {
		self.inner_channel.send_many(elems);
		self.interaction.extend_from_slice(elems);
	}

	fn observe_one(&mut self, val: F) {
		self.inner_channel.observe_one(val);
		self.interaction.push(val);
	}

	fn observe_many(&mut self, vals: &[F]) {
		self.inner_channel.observe_many(vals);
		self.interaction.extend_from_slice(vals);
	}

	fn sample(&mut self) -> F {
		let val = self.inner_channel.sample();
		self.interaction.push(val);
		val
	}
}

impl<F, P, NTT, MTScheme, MTProver, Challenger_, ReplayFn> IOPProverChannel<P>
	for &mut ZKWrappedProverChannel<'_, P, NTT, MTProver, Challenger_, ReplayFn>
where
	F: BinaryField,
	P: PackedField<Scalar = F> + PackedExtension<F>,
	NTT: AdditiveNTT<Field = F> + Sync,
	MTScheme: MerkleTreeScheme<F, Digest: SerializeBytes>,
	MTProver: MerkleTreeProver<F, Scheme = MTScheme>,
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
