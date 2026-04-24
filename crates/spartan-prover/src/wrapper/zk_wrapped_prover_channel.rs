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

use std::iter::repeat_with;

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

use crate::{Error, IOPProver, pack_and_blind_witness};

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
	keys: Vec<P::Scalar>,
	next_key_idx: usize,
	interaction: Vec<P::Scalar>,
	/// Handle to the outer precommit oracle committed at construction time. The buffer
	/// (`precommit_packed`) is purely random — it is the one-time-pad encryption key for the
	/// outer encrypted transcript (to be wired up in a follow-up; for now the outer circuit has
	/// no precommit wires that reference it).
	precommit_oracle: BaseFoldZKOracle,
	precommit_packed: FieldBuffer<P>,
	/// Number of outer oracles still to be committed on `inner_channel` during `finish` (the
	/// outer prover's non-precommit oracles — private and mask).
	n_outer_suffix_oracles: usize,
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
	/// Commits the outer prover's precommit oracle on the inner channel as part of construction:
	/// a random [`FieldBuffer<P>`] the size of the outer precommit oracle segment is sent to
	/// the channel and kept for use in [`Self::finish`]. This random buffer is the one-time-pad
	/// encryption key for the (future) outer encrypted transcript.
	///
	/// The inner channel's oracle specs are expected to be laid out as
	/// `[outer_precommit, inner..., outer_private, outer_mask]`.
	///
	/// # Arguments
	///
	/// * `inner_channel` - The BaseFold ZK channel with oracle specs for both inner and outer
	///   proofs
	/// * `outer_prover` - The IOP prover for the outer (wrapper) constraint system
	/// * `outer_layout` - The witness layout for the outer constraint system
	/// * `rng` - RNG used to generate the random precommit buffer (the future OTP key)
	/// * `replay_fn` - Closure called during [`finish`](Self::finish) with a [`ReplayChannel`] to
	///   replay the inner verification and fill the outer witness
	pub fn new(
		mut inner_channel: BaseFoldZKProverChannel<'a, F, P, NTT, MTProver, Challenger_>,
		outer_prover: &'a IOPProver<F>,
		outer_layout: &'a WitnessLayout<F>,
		mut rng: impl CryptoRng,
		replay_fn: ReplayFn,
	) -> Self {
		let outer_oracle_specs =
			IOPVerifier::new(outer_prover.constraint_system().clone()).oracle_specs();
		let all_specs = inner_channel.remaining_oracle_specs();
		let n_outer = outer_oracle_specs.len();
		assert!(
			n_outer >= 1 && all_specs.len() >= n_outer,
			"outer oracle specs ({n_outer}) exceed channel oracle specs ({}) or are empty",
			all_specs.len(),
		);
		assert_eq!(
			all_specs[0], outer_oracle_specs[0],
			"outer precommit oracle spec must be the first spec on the channel",
		);
		let suffix_len = n_outer - 1;
		assert_eq!(
			&all_specs[all_specs.len() - suffix_len..],
			&outer_oracle_specs[1..],
			"outer private/mask oracle specs must be the final suffix of channel specs",
		);

		// Commit random OTP keys as the outer precommit oracle. Each key encrypts one
		// element sent by the inner prover through this wrapped channel; the outer CS
		// (built symbolically from the inner verifier) contains a matching precommit wire per
		// key that the outer proof uses to decrypt.
		let cs = outer_prover.constraint_system();
		let keys = repeat_with(|| F::random(&mut rng))
			.take(cs.n_precommit() as usize)
			.collect::<Vec<F>>();
		// The precommit segment has no dummy mul-constraint blinding (see
		// ConstraintSystemPadded::new) — mirror that when packing.
		let precommit_blinding = binius_spartan_frontend::constraint_system::BlindingInfo {
			n_dummy_wires: cs.blinding_info().n_dummy_wires,
			n_dummy_constraints: 0,
		};
		let precommit_packed = pack_and_blind_witness::<_, P>(
			cs.log_precommit() as usize,
			&keys,
			cs.n_precommit() as usize,
			&precommit_blinding,
			&mut rng,
		);
		let precommit_oracle = inner_channel.send_oracle(precommit_packed.to_ref());

		Self {
			inner_channel,
			outer_prover,
			outer_layout,
			replay_fn,
			keys,
			next_key_idx: 0,
			interaction: Vec::new(),
			precommit_oracle,
			precommit_packed,
			n_outer_suffix_oracles: suffix_len,
		}
	}

	fn next_key(&mut self) -> F {
		let key = self.keys[self.next_key_idx];
		self.next_key_idx += 1;
		key
	}

	/// Consumes the channel and runs the outer proof.
	///
	/// This should be called after the inner proof has been run through this channel.
	/// It:
	/// 1. Creates a [`ReplayChannel`] from the recorded interaction
	/// 2. Calls the `replay_fn` closure to replay the inner verification and fill the outer witness
	/// 3. Validates and generates the outer IOP proof
	pub fn finish(self, rng: impl CryptoRng) -> Result<(), Error>
	where
		ReplayFn: FnOnce(&mut ReplayChannel<'_, F>),
	{
		let Self {
			inner_channel,
			outer_prover,
			outer_layout,
			replay_fn,
			keys,
			interaction,
			precommit_oracle,
			precommit_packed,
			..
		} = self;

		// Replay the inner verification through the outer witness generator.
		let mut replay_channel = ReplayChannel::new(outer_layout, keys, interaction);
		replay_fn(&mut replay_channel);
		let witness = replay_channel
			.finish()
			.expect("outer witness generation should not fail");

		// Validate and generate the outer proof.
		let outer_cs = outer_prover.constraint_system();
		outer_cs.validate(&witness);
		outer_prover.prove::<P, _>(
			witness,
			precommit_oracle,
			precommit_packed,
			rng,
			inner_channel,
		)?;
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
		let key = self.next_key();
		// Encrypt the element with the OTP key before sending. Record the encrypted value in
		// `interaction` — that's what the outer witness's inout wires hold (and what the replay
		// side adds the key back to in order to recover the plaintext for the inner verifier).
		let encrypted = elem + key;
		self.inner_channel.send_one(encrypted);
		self.interaction.push(encrypted);
	}

	fn observe_one(&mut self, val: F) {
		self.inner_channel.observe_one(val);
		self.interaction.push(val);
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
		let n_inner_remaining = remaining.len() - self.n_outer_suffix_oracles;
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
		oracle_relations: impl IntoIterator<
			Item = (Self::Oracle, FieldBuffer<P>, FieldBuffer<P>, P::Scalar),
		>,
	) {
		self.inner_channel.prove_oracle_relations(oracle_relations)
	}
}
