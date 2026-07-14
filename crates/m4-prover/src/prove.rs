// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use binius_core::{constraint_system::ConstraintSystem, word::Word};
use binius_field::PackedField;
use binius_hash::StdHashSuite;
use binius_iop_prover::{basefold_compiler::BaseFoldProverCompiler, channel::IOPProverChannel};
use binius_m4_verifier::Verifier;
use binius_math::ntt::{NeighborsLastMultiThread, domain_context::GenericPreExpanded};
use binius_prover::{
	protocols::shift::{KeyCollection, build_key_collection},
	ring_switch::{self, RingSwitchOutput},
};
use binius_transcript::{ProverTranscript, fiat_shamir::Challenger};
use binius_verifier::config::B128;

use crate::{
	ValueTable,
	reduction::{ReductionProverOutput, prove_reduction},
};

/// The multithreaded additive NTT used to encode the committed codeword.
type ProverNtt = NeighborsLastMultiThread<GenericPreExpanded<B128>>;

/// Proves the data-parallel M4 statement for a batch of `2^log_instances` circuit instances.
///
/// One-time setup builds the shift keys and the BaseFold prover, reusing the verifier's parameters.
/// A later proving call commits a witness table and proves it satisfies every AND constraint.
pub struct Prover<P>
where
	P: PackedField<Scalar = B128>,
{
	/// The prepared single-instance constraint system shared by every instance.
	cs: ConstraintSystem,
	/// The shift keys for the constraint system, built once and reused across proofs.
	key_collection: KeyCollection,
	/// The precomputed BaseFold prover, holding the NTT and the FRI parameters.
	basefold_compiler: BaseFoldProverCompiler<P, ProverNtt>,
}

impl<P> Prover<P>
where
	P: PackedField<Scalar = B128>,
{
	/// Builds the prover from a verifier, inheriting its constraint system and FRI parameters.
	///
	/// The prover encodes the codeword with the multithreaded NTT, spread across the cores.
	/// Reusing the verifier's compiler keeps both sides on one set of FRI parameters.
	pub fn setup(verifier: &Verifier) -> Self {
		// Reuse the verifier's evaluation domain so both sides agree on the code.
		let domain_context =
			GenericPreExpanded::generate_from_subspace(verifier.iop_compiler().max_subspace());

		// Spread the NTT across the available cores.
		let log_num_shares = binius_utils::rayon::current_num_threads().ilog2() as usize;
		let ntt = NeighborsLastMultiThread::new(domain_context, log_num_shares);

		// Inherit the verifier's oracle specs and FRI parameters verbatim.
		let basefold_compiler =
			BaseFoldProverCompiler::from_verifier_compiler(verifier.iop_compiler(), ntt);

		// Build the shift keys once from the shared constraint system.
		let key_collection = build_key_collection(verifier.constraint_system());

		Self {
			cs: verifier.constraint_system().clone(),
			key_collection,
			basefold_compiler,
		}
	}

	/// Proves that every instance in the batch satisfies the constraint system.
	///
	/// The flow composes the commitment, the reduction, and the opening on one transcript:
	/// - Pack the table into one B128 multilinear and commit it as the trace oracle.
	/// - Run the AND-check and shift reduction to a claim about the instance-folded witness.
	/// - Ring-switch that claim onto the committed trace and open it.
	///
	/// The trace commits before the reduction draws its challenges.
	/// So Fiat-Shamir binds every challenge to the committed data.
	///
	/// The reduction ends with a claim about the witness folded over instances at `r_rho`.
	/// The trace's bit index is `[bit | instance | wire]`.
	/// Evaluating its instance coordinates at `r_rho` performs that fold.
	/// So the ring-switch opens the trace at `r_j || r_rho || r_y`, matching the reduced claim.
	///
	/// The trace oracle is not ZK, so the channel masks nothing and needs no randomness.
	///
	/// # Panics
	///
	/// Panics if the constraint system has any MUL constraints.
	pub fn prove<Challenger_>(
		&self,
		table: &ValueTable,
		transcript: &mut ProverTranscript<Challenger_>,
	) where
		Challenger_: Challenger,
	{
		let mut channel = self
			.basefold_compiler
			.create_channel_without_zk_from_transcript::<StdHashSuite, Challenger_, _>(transcript);

		// Pack the 2-D table into one multilinear and commit it as the trace oracle.
		let trace_packed = {
			let _scope = tracing::debug_span!("Prepare trace").entered();
			table.pack::<P>()
		};
		let trace_oracle = {
			let _scope = tracing::debug_span!("Commit trace").entered();
			channel.send_oracle(trace_packed.to_ref())
		};

		// Reduce the AND constraints and shift to one folded-witness claim.
		// Every challenge is drawn now, after the commitment.
		let ReductionProverOutput {
			r_rho,
			witness_claim,
		} = prove_reduction::<P, _>(&self.cs, &self.key_collection, table, &mut channel);

		let RingSwitchOutput {
			rs_eq_ind,
			sumcheck_claim,
		} = {
			let _scope = tracing::debug_span!("Ring-switching reduction").entered();

			// Split the shift's final point `r_j || r_y || r_segment` into its three parts.
			// The bit index `r_j` is the low coordinates addressing a bit within a 64-bit word.
			// The segment selector `r_segment` is the last coordinate, choosing public or hidden
			// words. The hidden-only trace drops it.
			// The word index `r_y` is everything in between.
			let challenges = &witness_claim.challenges;
			let r_j = &challenges[..Word::LOG_BITS];
			let r_y = &challenges[Word::LOG_BITS..challenges.len() - 1];

			// Ring-switch the reduced claim onto the committed trace.
			// The point is `r_j || r_rho || r_y`.
			// Its instance coordinates fold the trace at `r_rho`.
			let trace_point = [r_j, r_rho.as_slice(), r_y].concat();
			ring_switch::prove(&trace_packed, &trace_point, &mut channel)
		};

		// Queue the trace opening against the ring-switch's transparent multilinear.
		// The final call runs the single combined FRI opening and writes it to the transcript.
		channel.prove_oracle_relations([(trace_oracle, trace_packed, rs_eq_ind, sumcheck_claim)]);

		let _scope = tracing::debug_span!("PCS opening").entered();
		channel.finish();
	}
}

#[cfg(test)]
mod tests {
	use std::array;

	use assert_matches::assert_matches;
	use binius_field::PackedBinaryGhash1x128b;
	use binius_iop::{
		basefold::{Error as BaseFoldError, VerificationError as BaseFoldVerificationError},
		channel::Error as IOPChannelError,
		fri::VerificationError as FriVerificationError,
		merkle_tree::VerificationError as MerkleVerificationError,
	};
	use binius_transcript::VerifierTranscript;
	use binius_verifier::{Error, config::StdChallenger};
	use rand::prelude::*;

	use super::*;
	use crate::test_utils::{N_INPUT_WORDS, crc64_circuit, populate_crc64_witness};

	type P = PackedBinaryGhash1x128b;

	// Builds a batch of `2^log_instances` CRC-64 instances with random input words.
	fn setup_batch(log_instances: usize, seed: u64) -> (ConstraintSystem, ValueTable) {
		let c = crc64_circuit();
		let n_instances = 1usize << log_instances;
		let mut rng = StdRng::seed_from_u64(seed);
		let inputs: Vec<[u64; N_INPUT_WORDS]> = (0..n_instances)
			.map(|_| array::from_fn(|_| rng.random()))
			.collect();
		let table = populate_crc64_witness(&c, &inputs);

		let mut cs = c.circuit.constraint_system().clone();
		cs.validate_and_prepare().unwrap();
		(cs, table)
	}

	// The prover and verifier run the whole protocol on one transcript.
	// A faithful proof over 64 instances verifies and leaves no trailing data.
	#[test]
	fn protocol_round_trips() {
		let log_instances = 6;
		let (cs, table) = setup_batch(log_instances, 0);

		// Setup once: the verifier fixes the shape and FRI parameters.
		// The prover inherits them.
		let verifier = Verifier::setup(&cs, log_instances, 1);
		let prover = Prover::<P>::setup(&verifier);

		// Prover: commit, reduce, and open on a fresh transcript.
		let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
		prover.prove(&table, &mut prover_transcript);

		// Verifier: replay the same transcript end to end.
		let mut verifier_transcript = prover_transcript.into_verifier();
		verifier
			.verify(&mut verifier_transcript)
			.expect("a faithful proof verifies");
		verifier_transcript
			.finalize()
			.expect("no trailing proof data");
	}

	// Tampering with the trace opening breaks the final FRI check.
	#[test]
	fn tampered_opening_is_rejected() {
		let log_instances = 6;
		let (cs, table) = setup_batch(log_instances, 1);

		let verifier = Verifier::setup(&cs, log_instances, 1);
		let prover = Prover::<P>::setup(&verifier);

		// Produce a faithful proof, then collect its bytes.
		let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
		prover.prove(&table, &mut prover_transcript);
		let mut proof = prover_transcript.finalize();

		// Flip one bit in the last byte, which lands in a FRI query's Merkle opening.
		// The opening no longer matches the committed root, so BaseFold verification rejects it.
		let last = proof.len() - 1;
		proof[last] ^= 1;

		let mut verifier_transcript = VerifierTranscript::new(StdChallenger::default(), proof);
		let err = verifier.verify(&mut verifier_transcript).unwrap_err();
		assert_matches!(
			err,
			Error::IOPChannel(IOPChannelError::BaseFold(BaseFoldError::Verification(
				BaseFoldVerificationError::FRI(FriVerificationError::MerkleError(
					MerkleVerificationError::InvalidProof
				))
			)))
		);
	}
}
