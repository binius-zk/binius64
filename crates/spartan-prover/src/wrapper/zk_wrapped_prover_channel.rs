// Copyright 2026 The Binius Developers

//! ZK-wrapped prover channel that runs an inner proof and then proves the outer
//! wrapper constraint system.
//!
//! [`ZKWrappedProverChannel`] wraps a [`BaseFoldZKProverChannel`] and an internal
//! [`WitnessGenerator`]. On `send_*`/`sample`/`observe_*`, it delegates to the inner BaseFoldZK
//! channel and writes each value (and the one-time-pad keys) directly into the outer witness,
//! allocating the inout and precommit wires in the symbolic build's order. After the inner proof is
//! run, [`finish`] replays the recorded derived-computation gates over that partially-filled
//! witness to fill the derived and private wires, then runs the outer IOP prover.
//!
//! [`BaseFoldZKProverChannel`]: binius_iop_prover::basefold_zk_channel::BaseFoldZKProverChannel
//! [`WitnessGenerator`]: binius_spartan_frontend::circuit_builder::WitnessGenerator
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
use binius_spartan_frontend::{
	circuit_builder::{WireAllocator, WitnessGenerator},
	constraint_system::{BlindingInfo, WireKind, WitnessLayout},
	gate::GateSequence,
};
use binius_spartan_verifier::IOPVerifier;
use binius_transcript::fiat_shamir::Challenger;
use binius_utils::SerializeBytes;
use rand::CryptoRng;

use crate::{Error, IOPProver, pack_and_blind_witness};

/// A prover channel that wraps a [`BaseFoldZKProverChannel`] and an outer Spartan IOP prover.
///
/// On `send_*`/`sample`/`observe_*`, it delegates to the inner BaseFoldZK channel and writes each
/// value (and the one-time-pad keys) directly into the outer witness on its own internal
/// [`WitnessGenerator`] — allocating the inout and precommit wires in the same order the symbolic
/// build did. After the inner proof is run through this channel, call [`finish`](Self::finish) with
/// the recorded gate sequence to replay the derived-computation gates over that partially-filled
/// witness (substituting for a re-execution of the inner verifier), then generate the outer proof.
pub struct ZKWrappedProverChannel<'a, P, NTT, MTProver, Challenger_>
where
	P: PackedField<Scalar: BinaryField>,
	NTT: AdditiveNTT<Field = P::Scalar> + Sync,
	MTProver: MerkleTreeProver<P::Scalar>,
	Challenger_: Challenger,
{
	inner_channel: BaseFoldZKProverChannel<'a, P::Scalar, P, NTT, MTProver, Challenger_>,
	outer_prover: &'a IOPProver<P::Scalar>,
	/// The outer witness under construction: inout (transcript) and precommit (key) wires are
	/// written here directly as the inner proof runs; [`finish`](Self::finish) then replays the
	/// recorded gates to fill the derived and private wires.
	witness_gen: WitnessGenerator<'a, P::Scalar>,
	/// Allocators for the inout and precommit segments, advanced in the same order as the symbolic
	/// [`IronSpartanBuilderChannel`](binius_spartan_verifier::wrapper::IronSpartanBuilderChannel)
	/// so the wire ids align with the outer layout (and hence the recorded gates).
	inout_alloc: WireAllocator,
	precommit_alloc: WireAllocator,
	keys: Vec<P::Scalar>,
	next_key_idx: usize,
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

impl<'a, F, P, NTT, MTScheme, MTProver, Challenger_>
	ZKWrappedProverChannel<'a, P, NTT, MTProver, Challenger_>
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
	pub fn new(
		mut inner_channel: BaseFoldZKProverChannel<'a, F, P, NTT, MTProver, Challenger_>,
		outer_prover: &'a IOPProver<F>,
		outer_layout: &'a WitnessLayout<F>,
		rng: impl CryptoRng,
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

		let (keys, precommit_oracle, precommit_packed) = {
			let _scope = tracing::debug_span!("Commit Transcript Mask").entered();
			Self::commit_transcript_mask(&mut inner_channel, outer_prover, rng)
		};

		Self {
			inner_channel,
			outer_prover,
			witness_gen: WitnessGenerator::new(outer_layout),
			inout_alloc: WireAllocator::new(WireKind::InOut),
			precommit_alloc: WireAllocator::new(WireKind::Precommit),
			keys,
			next_key_idx: 0,
			precommit_oracle,
			precommit_packed,
			n_outer_suffix_oracles: suffix_len,
		}
	}

	/// Commits random OTP keys as the outer precommit oracle. Each key encrypts one element sent by
	/// the inner prover through this wrapped channel; the outer CS (built symbolically from the
	/// inner verifier) contains a matching precommit wire per key that the outer proof uses to
	/// decrypt.
	fn commit_transcript_mask(
		inner_channel: &mut BaseFoldZKProverChannel<'a, F, P, NTT, MTProver, Challenger_>,
		outer_prover: &IOPProver<F>,
		mut rng: impl CryptoRng,
	) -> (Vec<F>, BaseFoldZKOracle, FieldBuffer<P>) {
		let cs = outer_prover.constraint_system();
		let keys = repeat_with(|| F::random(&mut rng))
			.take(cs.n_precommit() as usize)
			.collect::<Vec<F>>();
		// The precommit segment has no dummy mul-constraint blinding (see
		// ConstraintSystemPadded::new) — mirror that when packing.
		let precommit_blinding = BlindingInfo {
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
		(keys, precommit_oracle, precommit_packed)
	}

	fn next_key(&mut self) -> F {
		let key = self.keys[self.next_key_idx];
		self.next_key_idx += 1;
		key
	}

	/// Consumes the channel and runs the outer proof.
	///
	/// This should be called after the inner proof has been run through this channel. By then the
	/// inout (transcript) and precommit (OTP key) wires are already written into the internal
	/// witness; this replays the recorded derived-computation gates (`gate_seq`) over that
	/// partially-filled witness to fill the derived and private wires, then validates and generates
	/// the outer IOP proof.
	///
	/// `gate_seq` is the sequence recorded during the symbolic build of the outer constraint system
	/// (see [`GateRecordingConstraintBuilder`](binius_spartan_frontend::gate::GateRecordingConstraintBuilder)); it holds only
	/// the inner verifier's derived arithmetic (`Add`/`Mul`/`Generic`).
	pub fn finish(self, gate_seq: &GateSequence<F>, rng: impl CryptoRng) -> Result<(), Error> {
		let _ = tracing::debug_span!("Proving ZK wrapper proof").entered();

		let Self {
			mut inner_channel,
			outer_prover,
			mut witness_gen,
			precommit_oracle,
			precommit_packed,
			..
		} = self;

		// The inout and precommit wires were written into `witness_gen` as the inner proof ran;
		// replay the recorded gates to fill the derived and private wires, then build the witness.
		let witness = {
			let _ = tracing::debug_span!("Generating ZK wrapper witness").entered();
			witness_gen.apply_gates(gate_seq);
			witness_gen
				.build()
				.expect("apply_gates records no assertions, so witness generation cannot fail")
		};

		// Validate and generate the outer proof.
		let outer_cs = outer_prover.constraint_system();
		outer_cs.validate(&witness);
		outer_prover.prove::<P, _>(
			witness,
			precommit_oracle,
			precommit_packed,
			rng,
			&mut inner_channel,
		)?;
		// Both the inner and outer proofs queued their oracle relations onto `inner_channel`; run
		// the single combined opening over all committed oracles now.
		inner_channel.finish();
		Ok(())
	}
}

impl<F, P, NTT, MTScheme, MTProver, Challenger_> IPProverChannel<F>
	for &mut ZKWrappedProverChannel<'_, P, NTT, MTProver, Challenger_>
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
		// Encrypt the element with the OTP key before sending. The inout wire holds the encrypted
		// value and the paired precommit wire holds the key, mirroring the symbolic `recv_one`
		// (`inout - key`); the recorded gates add the key back to recover the plaintext for the
		// inner verifier. Allocate inout then precommit, matching the symbolic allocation order.
		let encrypted = elem + key;
		self.inner_channel.send_one(encrypted);
		let inout = self.inout_alloc.alloc();
		self.witness_gen.write_inout(inout, encrypted);
		let precommit = self.precommit_alloc.alloc();
		self.witness_gen.write_precommit(precommit, key);
	}

	fn observe_one(&mut self, val: F) {
		self.inner_channel.observe_one(val);
		let inout = self.inout_alloc.alloc();
		self.witness_gen.write_inout(inout, val);
	}

	fn sample(&mut self) -> F {
		let val = self.inner_channel.sample();
		let inout = self.inout_alloc.alloc();
		self.witness_gen.write_inout(inout, val);
		val
	}
}

impl<F, P, NTT, MTScheme, MTProver, Challenger_> IOPProverChannel<P>
	for &mut ZKWrappedProverChannel<'_, P, NTT, MTProver, Challenger_>
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
		let oracle_relations = oracle_relations.into_iter().collect::<Vec<_>>();

		// For each oracle opening, the prover sends the decrypted evaluation. The outer verifier
		// checks in the circuit equality of this value with the expected expression over encrypted
		// values. The decrypted claim is an inout wire (no paired key), matching the symbolic
		// `verify_oracle_relations`.
		for (_, _, _, claim) in &oracle_relations {
			self.inner_channel.send_one(*claim);
			let inout = self.inout_alloc.alloc();
			self.witness_gen.write_inout(inout, *claim);
		}

		self.inner_channel.prove_oracle_relations(oracle_relations)
	}
}
