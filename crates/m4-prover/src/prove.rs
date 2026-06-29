// Copyright 2025 Irreducible Inc.

use binius_field::PackedField;
use binius_hash::StdHashSuite;
use binius_iop_prover::{
	basefold_compiler::BaseFoldProverCompiler, channel::IOPProverChannel,
	merkle_tree::prover::BinaryMerkleTreeProver,
};
use binius_ip_prover::channel::IPProverChannel;
use binius_m4_verifier::{BatchCommitLayout, Verifier};
use binius_math::{
	inner_product::inner_product_buffers,
	multilinear::eq::eq_ind_partial_eval,
	ntt::{NeighborsLastMultiThread, domain_context::GenericPreExpanded},
};
use binius_transcript::{ProverTranscript, fiat_shamir::Challenger};
use binius_verifier::config::B128;
use rand::CryptoRng;

use crate::ValueTable;

/// The multithreaded additive NTT used to encode the committed codeword.
type ProverNtt = NeighborsLastMultiThread<GenericPreExpanded<B128>>;

/// The Merkle prover used to commit the codeword.
type ProverMerkle = BinaryMerkleTreeProver<B128, StdHashSuite>;

/// Proves a batch-witness commitment opened at a verifier-chosen random point.
///
/// Setup builds the BaseFold prover once, reusing the verifier's FRI parameters.
/// A later call commits a witness table and opens it at a transcript-chosen point.
pub struct Prover<P>
where
	P: PackedField<Scalar = B128>,
{
	/// The committed-multilinear shape of the batch, shared with the verifier.
	layout: BatchCommitLayout,
	/// The precomputed BaseFold prover, holding the NTT and the Merkle prover.
	basefold_compiler: BaseFoldProverCompiler<P, ProverNtt, ProverMerkle>,
}

impl<P> Prover<P>
where
	P: PackedField<Scalar = B128>,
{
	/// Builds the prover from a verifier, inheriting its FRI parameters.
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

		let merkle_prover = ProverMerkle::new();

		// Inherit the verifier's oracle specs and FRI parameters verbatim.
		let basefold_compiler = BaseFoldProverCompiler::from_verifier_compiler(
			verifier.iop_compiler(),
			ntt,
			merkle_prover,
		);

		Self {
			layout: *verifier.layout(),
			basefold_compiler,
		}
	}

	/// Commits the batch witness and proves its evaluation at a verifier-chosen random point.
	///
	/// The flow is the standard polynomial-commitment open:
	/// - Pack the table into one B128 multilinear and commit it as the trace oracle.
	/// - Draw a random point from the transcript.
	/// - Open the commitment at that point.
	///
	/// The point is drawn after the commitment.
	/// So Fiat-Shamir binds it to the committed data.
	///
	/// This opens the packed B128 multilinear directly.
	/// It does not ring-switch down to the bit witness.
	/// That step belongs with the later reductions.
	///
	/// The `rng` seeds the ZK channel's internal mask generation.
	///
	/// # Returns
	///
	/// The random point and the evaluation the prover commits to at it.
	pub fn prove<Challenger_>(
		&self,
		table: &ValueTable,
		rng: impl CryptoRng,
		transcript: &mut ProverTranscript<Challenger_>,
	) -> (Vec<B128>, B128)
	where
		Challenger_: Challenger,
	{
		let mut channel = self.basefold_compiler.create_channel(transcript, rng);

		// Pack the 2-D table into one multilinear and commit it as the trace oracle.
		let packed = table.pack::<P>();
		let oracle = channel.send_oracle(packed.to_ref());

		// Sample the evaluation point only now.
		// This makes it depend on the commitment.
		let point = channel.sample_many(self.layout.log_witness_elems);

		// The claimed evaluation is the inner product of the committed values with eq(point, .).
		//
		//     committed(point) = sum_w committed[w] * eq(point, w) = <committed, eq_ind(point)>
		let eq = eq_ind_partial_eval::<P>(&point);
		let eval = inner_product_buffers(&packed, &eq);

		// Send the claim, then open the commitment to prove it.
		// The verifier cannot recompute the claim, since the point depends on the commitment.
		channel.send_one(eval);

		// The ZK channel only queues the relation here.
		// `finish` runs the single combined FRI opening and writes it to the transcript.
		channel.prove_oracle_relations([(oracle, packed, eq, eval)]);
		channel.finish();

		(point, eval)
	}
}

#[cfg(test)]
mod tests {
	use assert_matches::assert_matches;
	use binius_core::word::Word;
	use binius_field::PackedBinaryGhash1x128b;
	use binius_frontend::{Circuit, CircuitBuilder, Wire};
	use binius_iop::{
		basefold::{Error as BaseFoldError, VerificationError as BaseFoldVerificationError},
		fri::VerificationError as FriVerificationError,
		merkle_tree::VerificationError as MerkleVerificationError,
	};
	use binius_math::multilinear::evaluate::evaluate;
	use binius_transcript::VerifierTranscript;
	use binius_verifier::config::StdChallenger;
	use proptest::prelude::*;
	use rand::{SeedableRng, rngs::StdRng};

	use super::*;

	type P = PackedBinaryGhash1x128b;

	// A circuit asserting `z == x & y` over three public words.
	// Satisfiable for an instance exactly when it sets z = x & y.
	struct AndCircuit {
		circuit: Circuit,
		x: Wire,
		y: Wire,
		z: Wire,
	}

	fn and_circuit() -> AndCircuit {
		let builder = CircuitBuilder::new();
		let x = builder.add_inout();
		let y = builder.add_inout();
		let z = builder.add_inout();
		let and = builder.band(x, y);
		builder.assert_eq("z_eq_x_and_y", and, z);
		AndCircuit {
			circuit: builder.build(),
			x,
			y,
			z,
		}
	}

	// Populate one instance per `(x, y)` pair; the instance count is the pair count.
	fn populate_table(c: &AndCircuit, inputs: &[(u64, u64)]) -> ValueTable {
		let log_instances = inputs.len().ilog2() as usize;
		ValueTable::populate(&c.circuit, log_instances, |i, w| {
			let (x, y) = inputs[i];
			w[c.x] = Word(x);
			w[c.y] = Word(y);
			w[c.z] = Word(x & y);
		})
		.unwrap()
	}

	proptest! {
		// Round-trip: a commitment opened at the transcript point verifies, and the value the
		// opening proves is the committed multilinear evaluated at that same point.
		//
		//     prover  : commit, draw point, open at point, claim e
		//     verifier: commit', draw point' (== point), read e, check the opening
		//
		// The inputs vary the witness, so each case commits to different data.
		#[test]
		fn commit_open_round_trips(inputs in prop::collection::vec((any::<u64>(), any::<u64>()), 4)) {
			let c = and_circuit();
			let table = populate_table(&c, &inputs);

			// Setup once: the verifier fixes the shape and FRI parameters; the prover inherits them.
			let verifier = Verifier::setup(c.circuit.constraint_system(), 2, 1);
			let prover = Prover::<P>::setup(&verifier);

			// Prover: commit and open on a fresh transcript.
			let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
			let rng = StdRng::seed_from_u64(0);
			let (point, eval) = prover.prove(&table, rng, &mut prover_transcript);

			// Verifier: replay the same transcript and check the opening.
			let mut verifier_transcript = prover_transcript.into_verifier();
			let (v_point, v_eval) = verifier
				.verify(&mut verifier_transcript)
				.expect("a faithful opening verifies");
			verifier_transcript.finalize().expect("no trailing proof data");

			// Both sides drew the same Fiat-Shamir point and agree on the opened value.
			prop_assert_eq!(&v_point, &point);
			prop_assert_eq!(v_eval, eval);

			// The opened value is the committed multilinear evaluated at the point.
			let direct = evaluate(&table.pack::<P>(), &point);
			prop_assert_eq!(eval, direct);
		}
	}

	#[test]
	fn tampered_proof_is_rejected() {
		let c = and_circuit();
		let table = populate_table(&c, &[(1, 2), (3, 4), (5, 6), (7, 8)]);

		let verifier = Verifier::setup(c.circuit.constraint_system(), 2, 1);
		let prover = Prover::<P>::setup(&verifier);

		// Produce a faithful proof, then collect its bytes.
		let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
		let rng = StdRng::seed_from_u64(0);
		let _ = prover.prove(&table, rng, &mut prover_transcript);
		let mut proof = prover_transcript.finalize();

		// Flip one bit deep in the proof; any change to the committed data must break the opening.
		let mid = proof.len() / 2;
		proof[mid] ^= 1;

		// The flipped byte lands in a FRI query's Merkle opening.
		// The opening no longer matches the committed root, so BaseFold verification rejects it.
		let mut verifier_transcript = VerifierTranscript::new(StdChallenger::default(), proof);
		let err = verifier.verify(&mut verifier_transcript).unwrap_err();
		assert_matches!(
			err,
			binius_iop::channel::Error::BaseFold(BaseFoldError::Verification(
				BaseFoldVerificationError::FRI(FriVerificationError::MerkleError(
					MerkleVerificationError::InvalidProof
				))
			))
		);
	}
}
