// Copyright 2025 Irreducible Inc.

use binius_core::constraint_system::ConstraintSystem;
use binius_hash::StdHashSuite;
use binius_iop::{
	basefold_compiler::BaseFoldVerifierCompiler,
	channel::{IOPVerifierChannel, OracleLinearRelation, OracleSpec},
	fri::{ConstantArityStrategy, calculate_n_test_queries},
	merkle_tree::BinaryMerkleTreeScheme,
};
use binius_ip::channel::IPVerifierChannel;
use binius_math::multilinear::eq::eq_ind;
use binius_transcript::{VerifierTranscript, fiat_shamir::Challenger};
use binius_verifier::config::B128;

use crate::commit::BatchCommitLayout;

/// The target soundness, in bits.
///
/// This matches the Binius64 verifier's target.
/// It only sets the FRI query count.
const SECURITY_BITS: usize = 96;

/// The Merkle commitment scheme over the committed field.
type Scheme = BinaryMerkleTreeScheme<B128, StdHashSuite>;

/// Verifies a batch-witness commitment opened at a verifier-chosen random point.
///
/// Setup fixes the committed-oracle shape and the BaseFold parameters once.
/// A later opening proof is then checked against that fixed setup.
///
/// The prover is built from this verifier, so both sides share one set of FRI parameters.
pub struct Verifier {
	/// The committed-multilinear shape of the batch.
	layout: BatchCommitLayout,
	/// The precomputed BaseFold verifier, holding the Merkle scheme and FRI parameters.
	iop_compiler: BaseFoldVerifierCompiler<B128, Scheme>,
}

impl Verifier {
	/// Builds the verifier for `2^log_instances` instances of one circuit at the given code rate.
	///
	/// # Arguments
	///
	/// - `cs`: the single-instance constraint system shared by every instance.
	/// - `log_instances`: base-2 logarithm of the instance count.
	/// - `log_inv_rate`: base-2 logarithm of the inverse Reed-Solomon rate.
	pub fn setup(cs: &ConstraintSystem, log_instances: usize, log_inv_rate: usize) -> Self {
		// The committed shape follows from one instance's length and the instance count.
		let layout = BatchCommitLayout::for_constraint_system(cs, log_instances);

		// One oracle: the packed batch witness, committed without zero-knowledge.
		// ZK-ness is a higher-level choice that M4 does not make here.
		let oracle_specs = vec![OracleSpec::new(layout.log_witness_elems)];

		// Pick the proof-size-optimal FRI fold arity for this codeword length.
		let log_code_len = layout.log_witness_elems + log_inv_rate;
		let merkle_scheme = Scheme::new();
		let fri_arity =
			ConstantArityStrategy::with_optimal_arity::<B128, _>(&merkle_scheme, log_code_len)
				.arity;

		// The query count is fixed by the rate and the soundness target.
		let n_test_queries = calculate_n_test_queries(SECURITY_BITS, log_inv_rate);

		let iop_compiler = BaseFoldVerifierCompiler::new(
			merkle_scheme,
			oracle_specs,
			log_inv_rate,
			n_test_queries,
			&ConstantArityStrategy::new(fri_arity),
		);

		Self {
			layout,
			iop_compiler,
		}
	}

	/// The committed-multilinear shape this verifier expects.
	pub fn layout(&self) -> &BatchCommitLayout {
		&self.layout
	}

	/// The precomputed BaseFold verifier compiler.
	///
	/// The prover reuses it so both sides share one set of FRI parameters.
	pub fn iop_compiler(&self) -> &BaseFoldVerifierCompiler<B128, Scheme> {
		&self.iop_compiler
	}

	/// Verifies a batch-witness commitment opened at a verifier-chosen random point.
	///
	/// The verifier receives the commitment, redraws the same point, reads the claim, and checks
	/// the opening.
	///
	/// # Returns
	///
	/// The random point and the evaluation the opening proves at it.
	///
	/// # Errors
	///
	/// Returns an error if the commitment opening does not verify.
	pub fn verify<Challenger_>(
		&self,
		transcript: &mut VerifierTranscript<Challenger_>,
	) -> Result<(Vec<B128>, B128), binius_iop::channel::Error>
	where
		Challenger_: Challenger,
	{
		let mut channel = self.iop_compiler.create_channel(transcript);

		// Receive the commitment, redraw the same point, and read the claimed evaluation.
		let oracle = channel.recv_oracle()?;
		let point = channel.sample_many(self.layout.log_witness_elems);
		let eval = channel.recv_one()?;

		// The committed multilinear opened at `point` must equal `eval`.
		//
		// The transparent multilinear is eq(point, .).
		// BaseFold reduces to a challenge point `pt`, where this transparent evaluates to eq(point,
		// pt).
		let point_at_pt = point.clone();

		// The channel only queues the relation here.
		// `finish` runs the single combined FRI opening check.
		channel.verify_oracle_relations([OracleLinearRelation {
			oracle,
			transparent: Box::new(move |pt: &[B128]| eq_ind(&point_at_pt, pt)),
			claim: eval,
		}])?;
		channel.finish()?;

		Ok((point, eval))
	}
}
