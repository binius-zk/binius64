// Copyright 2025 Irreducible Inc.

use binius_field::{BinaryField, PackedField};
use binius_math::{FieldBuffer, ntt::AdditiveNTT};
use binius_transcript::{
	ProverTranscript,
	fiat_shamir::{CanSample, Challenger},
};
use binius_utils::SerializeBytes;
use binius_verifier::{merkle_tree::MerkleTreeScheme, protocols::sumcheck::RoundCoeffs};

use crate::{
	Error,
	fri::{FRIFoldProver, FoldRoundOutput},
	merkle_tree::MerkleTreeProver,
	protocols::sumcheck::{
		bivariate_product::BivariateProductSumcheckProver, common::SumcheckProver,
	},
};

/// Prover for the BaseFold protocol.
///
/// The [BaseFold] protocol is a sumcheck-PIOP to IP compiler, used in the [DP24] polynomial
/// commitment scheme. The verifier module [`binius_verifier::protocols::basefold`] provides a
/// description of the protocol.
///
/// This struct exposes a round-by-round interface for one instance of the interactive protocol.
///
/// [BaseFold]: <https://link.springer.com/chapter/10.1007/978-3-031-68403-6_5>
/// [DP24]: <https://eprint.iacr.org/2024/504>
pub struct BaseFoldProver<'a, F, P, NTT, MerkleProver>
where
	F: BinaryField,
	P: PackedField<Scalar = F>,
	NTT: AdditiveNTT<Field = F> + Sync,
	MerkleProver: MerkleTreeProver<F>,
{
	sumcheck_prover: BivariateProductSumcheckProver<P>,
	fri_folder: FRIFoldProver<'a, F, P, NTT, MerkleProver>,
}

impl<'a, F, P, NTT, MerkleScheme, MerkleProver> BaseFoldProver<'a, F, P, NTT, MerkleProver>
where
	F: BinaryField,
	P: PackedField<Scalar = F>,
	NTT: AdditiveNTT<Field = F> + Sync,
	MerkleScheme: MerkleTreeScheme<F, Digest: SerializeBytes>,
	MerkleProver: MerkleTreeProver<F, Scheme = MerkleScheme>,
{
	/// Constructs a new prover.
	///
	/// ## Arguments
	///
	/// * `multilinear` - the multilinear polynomial
	/// * `transparent_multilinear` - the transparent multilinear polynomial
	/// * `claim` - the claim
	/// * `fri_folder` - the FRI fold prover
	///
	/// ## Pre-conditions
	///  * the multilinear has already been committed to using FRI
	///  * the length of the multilinear and transparent_multilinear are equal
	pub fn new(
		multilinear: FieldBuffer<P>,
		transparent_multilinear: FieldBuffer<P>,
		claim: F,
		fri_folder: FRIFoldProver<'a, F, P, NTT, MerkleProver>,
	) -> Self {
		assert_eq!(multilinear.log_len(), transparent_multilinear.log_len());
		assert_eq!(multilinear.log_len(), fri_folder.n_rounds());

		let sumcheck_prover =
			BivariateProductSumcheckProver::new([multilinear, transparent_multilinear], claim)
				.expect("precondition: multilinear.log_len() == transparent_multilinear.log_len()");

		Self {
			sumcheck_prover,
			fri_folder,
		}
	}

	/// Executes the sumcheck round, producing a round message.
	///
	/// ## Pre-conditions
	///  * the sumcheck has already been initialized
	///
	/// ## Returns
	///  * the sumcheck round message
	///  * the FRI fold round output
	fn execute(
		&mut self,
	) -> Result<(RoundCoeffs<F>, FoldRoundOutput<MerkleScheme::Digest>), Error> {
		let [round_coeffs] = self
			.sumcheck_prover
			.execute()?
			.try_into()
			.expect("sumcheck_prover proves only one multivariate");
		let commitment = self.fri_folder.execute_fold_round()?;
		Ok((round_coeffs, commitment))
	}

	/// Folds both the sumcheck multilinear and its codeword.
	///
	/// ## Arguments
	/// * `challenge` - a challenge sampled from the transcript
	fn fold(&mut self, challenge: F) -> Result<(), Error> {
		self.sumcheck_prover.fold(challenge)?;
		self.fri_folder.receive_challenge(challenge);
		Ok(())
	}

	/// Runs the protocol to completion.
	///
	/// ## Arguments
	/// * `transcript` - the prover's view of the proof transcript
	///
	/// ## Returns
	///  * the FRI fold round output
	pub fn prove<T: Challenger>(
		mut self,
		transcript: &mut ProverTranscript<T>,
	) -> Result<(), Error> {
		let _scope = tracing::debug_span!("Basefold").entered();

		let n_vars = self.sumcheck_prover.n_vars();
		for _ in 0..n_vars {
			let (round_coeffs, commitment) = self.execute()?;
			transcript
				.message()
				.write_scalar_slice(round_coeffs.truncate().coeffs());
			if let FoldRoundOutput::Commitment(commitment) = commitment {
				transcript.message().write(&commitment);
			}

			let challenge = transcript.sample();
			self.fold(challenge)?;
		}
		self.finish(transcript)?;

		Ok(())
	}

	/// Finalizes the transcript by proving FRI queries.
	///
	/// ## Arguments
	/// * `prover_challenger` - the prover's mutable transcript
	fn finish<T: Challenger>(mut self, transcript: &mut ProverTranscript<T>) -> Result<(), Error> {
		let commitment = self.fri_folder.execute_fold_round()?;
		if let FoldRoundOutput::Commitment(commitment) = commitment {
			transcript.message().write(&commitment);
		}

		self.fri_folder.finish_proof(transcript)?;
		Ok(())
	}
}

#[cfg(test)]
mod test {
	use anyhow::{Result, bail};
	use binius_field::{
		BinaryField, PackedBinaryGhash1x128b, PackedBinaryGhash2x128b, PackedBinaryGhash4x128b,
		PackedExtension, PackedField,
	};
	use binius_math::{
		BinarySubspace, FieldBuffer,
		inner_product::inner_product_buffers,
		multilinear::eq::eq_ind_partial_eval,
		ntt::{NeighborsLastSingleThread, domain_context::GenericOnTheFly},
		test_utils::{random_field_buffer, random_scalars},
	};
	use binius_transcript::ProverTranscript;
	use binius_verifier::{
		config::StdChallenger,
		fri::{ConstantArityStrategy, FRIParams, calculate_n_test_queries},
		hash::{StdCompression, StdDigest},
		protocols::basefold,
	};
	use rand::{SeedableRng, rngs::StdRng};

	use super::BaseFoldProver;
	use crate::{
		fri::{self, CommitOutput, FRIFoldProver},
		hash::parallel_compression::ParallelCompressionAdaptor,
		merkle_tree::prover::BinaryMerkleTreeProver,
	};

	pub const LOG_INV_RATE: usize = 1;
	pub const SECURITY_BITS: usize = 32;

	fn run_basefold_prove_and_verify<F, P>(
		multilinear: FieldBuffer<P>,
		evaluation_point: Vec<F>,
		evaluation_claim: F,
	) -> Result<()>
	where
		F: BinaryField,
		P: PackedField<Scalar = F> + PackedExtension<F>,
	{
		let eval_point_eq = eq_ind_partial_eval::<P>(&evaluation_point);

		let merkle_prover = BinaryMerkleTreeProver::<F, StdDigest, _>::new(
			ParallelCompressionAdaptor::new(StdCompression::default()),
		);

		let subspace = BinarySubspace::with_dim(multilinear.log_len() + LOG_INV_RATE).unwrap();
		let domain_context = GenericOnTheFly::generate_from_subspace(&subspace);
		let ntt = NeighborsLastSingleThread::new(domain_context);

		let n_test_queries = calculate_n_test_queries(SECURITY_BITS, LOG_INV_RATE);
		let fri_params = FRIParams::with_strategy(
			&ntt,
			merkle_prover.scheme(),
			multilinear.log_len(),
			None,
			LOG_INV_RATE,
			n_test_queries,
			&ConstantArityStrategy::new(2),
		)?;

		let CommitOutput {
			commitment: codeword_commitment,
			committed: codeword_committed,
			codeword,
		} = fri::commit_interleaved(&fri_params, &ntt, &merkle_prover, multilinear.to_ref())?;

		let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
		prover_transcript.message().write(&codeword_commitment);

		let fri_folder =
			FRIFoldProver::new(&fri_params, &ntt, &merkle_prover, codeword, &codeword_committed)?;

		let prover = BaseFoldProver::new(multilinear, eval_point_eq, evaluation_claim, fri_folder);
		prover.prove(&mut prover_transcript)?;

		let mut verifier_transcript = prover_transcript.into_verifier();

		let retrieved_codeword_commitment = verifier_transcript.message().read()?;

		let basefold::ReducedOutput {
			final_fri_value,
			final_sumcheck_value,
			challenges,
		} = basefold::verify(
			&fri_params,
			merkle_prover.scheme(),
			retrieved_codeword_commitment,
			evaluation_claim,
			&mut verifier_transcript,
		)?;

		if !basefold::sumcheck_fri_consistency(
			final_fri_value,
			final_sumcheck_value,
			&evaluation_point,
			challenges,
		) {
			bail!("Sumcheck and FRI are inconsistent");
		}

		Ok(())
	}

	fn test_setup<F, P>(n_vars: usize) -> (FieldBuffer<P>, Vec<F>, F)
	where
		F: BinaryField,
		P: PackedField<Scalar = F>,
	{
		let mut rng = StdRng::from_seed([0; 32]);

		let multilinear = random_field_buffer::<P>(&mut rng, n_vars);
		let evaluation_point = random_scalars::<F>(&mut rng, n_vars);

		let eval_point_eq = eq_ind_partial_eval(&evaluation_point);
		let evaluation_claim = inner_product_buffers(&multilinear, &eval_point_eq);

		(multilinear, evaluation_point, evaluation_claim)
	}

	fn dubiously_modify_claim<F, P>(claim: &mut F)
	where
		F: BinaryField,
		P: PackedField<Scalar = F>,
	{
		*claim += P::Scalar::ONE
	}

	#[test]
	fn test_basefold_valid_proof() {
		type P = PackedBinaryGhash1x128b;

		let n_vars = 8;
		let (multilinear, evaluation_point, evaluation_claim) = test_setup::<_, P>(n_vars);

		run_basefold_prove_and_verify::<_, P>(multilinear, evaluation_point, evaluation_claim)
			.unwrap();
	}

	#[test]
	fn test_basefold_invalid_proof() {
		type P = PackedBinaryGhash1x128b;

		let n_vars = 8;
		let (multilinear, evaluation_point, mut evaluation_claim) = test_setup::<_, P>(n_vars);

		dubiously_modify_claim::<_, P>(&mut evaluation_claim);
		let result =
			run_basefold_prove_and_verify::<_, P>(multilinear, evaluation_point, evaluation_claim);
		assert!(result.is_err());
	}

	#[test]
	fn test_basefold_valid_packing_width_2() {
		type P = PackedBinaryGhash2x128b;

		let n_vars = 8;
		let (multilinear, evaluation_point, evaluation_claim) = test_setup::<_, P>(n_vars);

		run_basefold_prove_and_verify::<_, P>(multilinear, evaluation_point, evaluation_claim)
			.unwrap();
	}

	#[test]
	fn test_basefold_valid_packing_width_4() {
		type P = PackedBinaryGhash4x128b;

		let n_vars = 8;
		let (multilinear, evaluation_point, evaluation_claim) = test_setup::<_, P>(n_vars);

		run_basefold_prove_and_verify::<_, P>(multilinear, evaluation_point, evaluation_claim)
			.unwrap();
	}
}
