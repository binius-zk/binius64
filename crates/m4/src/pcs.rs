// Copyright 2025 Irreducible Inc.

use binius_field::PackedField;
use binius_hash::StdHashSuite;
use binius_iop::{
	basefold_compiler::BaseFoldVerifierCompiler,
	channel::{IOPVerifierChannel, OracleLinearRelation, OracleSpec},
	fri::{MinProofSizeStrategy, calculate_n_test_queries},
	merkle_tree::BinaryMerkleTreeScheme,
};
use binius_iop_prover::{
	basefold_compiler::BaseFoldProverCompiler, channel::IOPProverChannel,
	merkle_tree::prover::BinaryMerkleTreeProver,
};
use binius_ip::channel::IPVerifierChannel;
use binius_ip_prover::channel::IPProverChannel;
use binius_math::{
	BinarySubspace,
	inner_product::inner_product_buffers,
	multilinear::eq::{eq_ind, eq_ind_partial_eval},
	ntt::{NeighborsLastSingleThread, domain_context::GenericOnTheFly},
};
use binius_transcript::{ProverTranscript, VerifierTranscript, fiat_shamir::Challenger};
use binius_verifier::config::B128;

use crate::{commit::BatchCommitLayout, value_table::ValueTable};

/// The target soundness, in bits.
///
/// This matches the Binius64 verifier's target.
/// It only feeds the FRI query count.
/// It never changes what gets committed.
const SECURITY_BITS: usize = 96;

/// The polynomial-commitment configuration shared by the batch prover and verifier.
///
/// Both sides build the committed-oracle shape and the FRI query count from the same inputs.
/// Deriving them identically is what lets the verifier replay the prover's commitment.
#[derive(Clone, Copy, Debug)]
pub struct BatchPcsParams {
	/// The committed-multilinear shape of the batch.
	layout: BatchCommitLayout,
	/// The base-2 logarithm of the inverse Reed-Solomon rate.
	log_inv_rate: usize,
	/// The number of FRI query repetitions, set for the target soundness.
	n_test_queries: usize,
}

impl BatchPcsParams {
	/// Builds the configuration for a batch of the given shape at the given code rate.
	///
	/// # Arguments
	///
	/// - `layout`: the committed-multilinear shape of the batch.
	/// - `log_inv_rate`: the base-2 logarithm of the inverse Reed-Solomon rate.
	pub fn new(layout: BatchCommitLayout, log_inv_rate: usize) -> Self {
		// The query count is fixed by the rate and the soundness target.
		let n_test_queries = calculate_n_test_queries(SECURITY_BITS, log_inv_rate);
		Self {
			layout,
			log_inv_rate,
			n_test_queries,
		}
	}

	/// The committed-multilinear shape of the batch.
	pub fn layout(&self) -> BatchCommitLayout {
		self.layout
	}

	/// The single committed oracle: the packed batch witness.
	fn oracle_specs(&self) -> Vec<OracleSpec> {
		vec![self.layout.oracle_spec()]
	}

	/// The base-2 logarithm of the Reed-Solomon codeword length.
	///
	/// This is the message length plus the inverse-rate logarithm.
	/// The NTT evaluation domain must span at least this many points.
	fn log_code_len(&self) -> usize {
		self.layout.log_witness_elems + self.log_inv_rate
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
	/// # Returns
	///
	/// The random point and the evaluation the prover commits to at it.
	pub fn prove_evaluation<P, Challenger_>(
		&self,
		table: &ValueTable,
		transcript: &mut ProverTranscript<Challenger_>,
	) -> (Vec<B128>, B128)
	where
		P: PackedField<Scalar = B128>,
		Challenger_: Challenger,
	{
		// Build the BaseFold prover over an evaluation domain large enough for the codeword.
		let subspace = BinarySubspace::<B128>::with_dim(self.log_code_len());
		let domain_context = GenericOnTheFly::generate_from_subspace(&subspace);
		let ntt = NeighborsLastSingleThread::new(domain_context);
		let merkle_prover = BinaryMerkleTreeProver::<B128, StdHashSuite>::new();
		let compiler = BaseFoldProverCompiler::<P, _, _>::new(
			ntt,
			merkle_prover,
			self.oracle_specs(),
			self.log_inv_rate,
			self.n_test_queries,
		);
		let mut channel = compiler.create_channel(transcript);

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
		channel.prove_oracle_relations([(oracle, packed, eq, eval)]);

		(point, eval)
	}

	/// Verifies a batch-witness commitment opened at a verifier-chosen random point.
	///
	/// The verifier mirrors the prover.
	/// It receives the commitment, redraws the same point, reads the claim, and checks the opening.
	///
	/// # Returns
	///
	/// The random point and the evaluation the opening proves at it.
	///
	/// # Errors
	///
	/// Returns an error if the commitment opening does not verify.
	pub fn verify_evaluation<Challenger_>(
		&self,
		transcript: &mut VerifierTranscript<Challenger_>,
	) -> Result<(Vec<B128>, B128), binius_iop::channel::Error>
	where
		Challenger_: Challenger,
	{
		// Build the matching verifier over the same oracle shape and query count.
		let merkle_scheme = BinaryMerkleTreeScheme::<B128, StdHashSuite>::new();
		let compiler = BaseFoldVerifierCompiler::new(
			merkle_scheme,
			self.oracle_specs(),
			self.log_inv_rate,
			self.n_test_queries,
			&MinProofSizeStrategy,
		);
		let mut channel = compiler.create_channel(transcript);

		// Receive the commitment, then redraw the same point and read the claimed evaluation.
		let oracle = channel.recv_oracle()?;
		let point = channel.sample_many(self.layout.log_witness_elems);
		let eval = channel.recv_one()?;

		// The committed multilinear opened at `point` must equal `eval`.
		//
		// The transparent multilinear is eq(point, .).
		// BaseFold reduces to a challenge point `pt`, where this transparent evaluates to eq(point,
		// pt).
		let point_at_pt = point.clone();
		channel.verify_oracle_relations([OracleLinearRelation {
			oracle,
			transparent: Box::new(move |pt: &[B128]| eq_ind(&point_at_pt, pt)),
			claim: eval,
		}])?;

		Ok((point, eval))
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
		channel::Error as ChannelError,
		fri::VerificationError as FriVerificationError,
		merkle_tree::VerificationError as MerkleVerificationError,
	};
	use binius_math::{FieldBuffer, multilinear::evaluate::evaluate};
	use binius_transcript::ProverTranscript;
	use binius_verifier::config::StdChallenger;
	use proptest::prelude::*;

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
		// Round-trip: a commitment opened at the transcript point verifies.
		// The value it proves is the committed multilinear evaluated at that same point.
		//
		//     prover  : commit, draw point, open at point, claim e
		//     verifier: commit', draw point' (== point), read e, check the opening
		//
		// The inputs vary the witness, so each case commits to different data.
		#[test]
		fn commit_open_round_trips(inputs in prop::collection::vec((any::<u64>(), any::<u64>()), 4)) {
			let c = and_circuit();
			let table = populate_table(&c, &inputs);
			let params = BatchPcsParams::new(table.commit_layout(), 1);

			// Prover: commit and open on a fresh transcript.
			let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
			let (point, eval) = params.prove_evaluation::<P, _>(&table, &mut prover_transcript);

			// Verifier: replay the same transcript and check the opening.
			let mut verifier_transcript = prover_transcript.into_verifier();
			let (v_point, v_eval) = params
				.verify_evaluation(&mut verifier_transcript)
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
		let params = BatchPcsParams::new(table.commit_layout(), 1);

		// Produce a faithful proof, then collect its bytes.
		let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
		let _ = params.prove_evaluation::<P, _>(&table, &mut prover_transcript);
		let mut proof = prover_transcript.finalize();

		// Flip one bit deep in the proof; any change to the committed data must break the opening.
		let mid = proof.len() / 2;
		proof[mid] ^= 1;

		// The verifier must reject the corrupted proof rather than accept a false opening.
		let mut verifier_transcript = VerifierTranscript::new(StdChallenger::default(), proof);
		let err = params
			.verify_evaluation(&mut verifier_transcript)
			.unwrap_err();

		// The flipped byte lands in a FRI query's Merkle opening.
		// The opening no longer matches the committed root, so BaseFold verification rejects it.
		assert_matches!(
			err,
			ChannelError::BaseFold(BaseFoldError::Verification(BaseFoldVerificationError::FRI(
				FriVerificationError::MerkleError(MerkleVerificationError::InvalidProof)
			)))
		);
	}

	#[test]
	fn packed_multilinear_has_expected_dimension() {
		let c = and_circuit();
		let table = populate_table(&c, &[(1, 2), (3, 4), (5, 6), (7, 8)]);

		// The committed multilinear has one variable per committed element.
		let packed: FieldBuffer<P> = table.pack::<P>();
		assert_eq!(packed.log_len(), table.commit_layout().log_witness_elems);
	}
}
