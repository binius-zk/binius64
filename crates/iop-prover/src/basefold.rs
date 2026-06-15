// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use binius_field::{BinaryField, PackedField};
use binius_iop::merkle_tree::MerkleTreeScheme;
use binius_ip::{mlecheck, sumcheck::RoundCoeffs};
use binius_ip_prover::sumcheck::{
	bivariate_product::BivariateProductSumcheckProver, common::SumcheckProver,
	multilinear_eval::MultilinearEvalProver,
};
use binius_math::{FieldBuffer, ntt::AdditiveNTT};
use binius_transcript::{
	ProverTranscript,
	fiat_shamir::{CanSample, Challenger},
};
use binius_utils::SerializeBytes;

use crate::{
	fri::{FRIFoldProver, FoldRoundOutput},
	merkle_tree::MerkleTreeProver,
};

#[derive(Debug, thiserror::Error)]
pub enum Error {
	#[error("sumcheck error: {0}")]
	Sumcheck(#[from] binius_ip_prover::sumcheck::Error),
}

/// Prover for the BaseFold protocol.
///
/// The [BaseFold] protocol is a sumcheck-PIOP to IP compiler, used in the [DP24] polynomial
/// commitment scheme. The verifier module [`binius_iop::basefold`] provides a
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
		assert_eq!(multilinear.log_len(), fri_folder.n_rounds_remaining());

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
		let commitment = self.fri_folder.execute_fold_round();
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
		self.finish(transcript);

		Ok(())
	}

	/// Finalizes the transcript by proving FRI queries.
	///
	/// ## Arguments
	/// * `prover_challenger` - the prover's mutable transcript
	fn finish<T: Challenger>(mut self, transcript: &mut ProverTranscript<T>) {
		let commitment = self.fri_folder.execute_fold_round();
		if let FoldRoundOutput::Commitment(commitment) = commitment {
			transcript.message().write(&commitment);
		}

		self.fri_folder.finish_proof(transcript);
	}
}

/// Proves a *combined* multilinear evaluation claim `𝛑(eval_point) = eval_claim` by interleaving a
/// single [`MultilinearEvalProver`] MLE-check with a single combined FRI over the
/// piecewise-concatenated oracle of the Batched ZK BaseFold construction (whitepaper §7.2 /
/// §sec:batched-basefold Step 2).
///
/// A prior batched sumcheck reduced the `k` masked opening claims to per-oracle point-evaluation
/// claims `π_i'(ρ_i) = α_i` at a shared point `r ∈ K^𝐧` (`𝐧 = max_i n_i`). The caller has collapsed
/// the oracle-index variables up front at sampled batching challenges `r'` into a single combined
/// multilinear `𝛑(X) = Σ_i e[i]·π_i^↑(X)`, `e = eq_ind_partial_eval(r')` (passed as `witness`),
/// with target `s' = 𝛑(r)`. Here we run the degree-1 MLE-check on `𝛑` against `r`, interleaved with
/// the FRI codeword built (via [`FRIFoldProver::new_batch`]) from the `k` committed interleaved
/// `[π_i ‖ ω_i]` codewords.
///
/// ## Arguments
///
/// * `witness` - the combined oracle multilinear `𝛑` with `log_len = 𝐧`
/// * `eval_point` - the point `r` with `len = 𝐧`, in low-to-high variable order
/// * `eval_claim` - the combined target `s' = 𝛑(r)`
/// * `batch_challenge` - the masking challenge `γ`; folds each interleaved `[π_i ‖ ω_i]` codeword
///   down to the codeword of `π_i'` in the FRI inner (unbatch) round
/// * `outer_challenges` - the batching challenges `r'` (`len = log_n_oracles`); combine the `k`
///   lifted codewords in the FRI outer (oracle-combine) rounds
/// * `fri_folder` - the combined FRI fold prover, with `n_rounds == 𝐧 + 1 + log_n_oracles`
/// * `transcript` - the prover transcript
///
/// The final FRI value equals the final MLE-check value `𝛑(r)` (see
/// [`binius_iop::basefold::mlecheck_fri_consistency`]).
#[allow(clippy::too_many_arguments)]
pub fn prove_mlecheck_basefold_zk_batch<'a, F, P, NTT, MerkleScheme, MerkleProver, Challenger_>(
	witness: FieldBuffer<P>,
	eval_point: &[F],
	eval_claim: F,
	batch_challenge: F,
	outer_challenges: &[F],
	mut fri_folder: FRIFoldProver<'a, F, P, NTT, MerkleProver>,
	transcript: &mut ProverTranscript<Challenger_>,
) -> Result<(), Error>
where
	F: BinaryField,
	P: PackedField<Scalar = F>,
	NTT: AdditiveNTT<Field = F> + Sync,
	MerkleScheme: MerkleTreeScheme<F, Digest: SerializeBytes>,
	MerkleProver: MerkleTreeProver<F, Scheme = MerkleScheme>,
	Challenger_: Challenger,
{
	let _scope = tracing::debug_span!("Basefold MLE-check ZK (batched)").entered();

	let n_vars = witness.log_len();
	assert_eq!(eval_point.len(), n_vars);
	// The FRI folder has one inner (unbatch) round, `log_n_oracles` outer (oracle-combine) rounds,
	// and `𝐧` standard fold rounds.
	assert_eq!(n_vars + 1 + outer_challenges.len(), fri_folder.n_rounds());

	// Inner (unbatch) round: fold every interleaved (π_i ‖ ω_i) codeword at the masking challenge.
	fri_folder.receive_challenge(batch_challenge);
	// Outer rounds: combine the k lifted codewords at the batching challenges r'. These carry no
	// sumcheck round-polynomial; the folder applies them lazily with γ at the first commit round.
	for &outer_challenge in outer_challenges {
		fri_folder.receive_challenge(outer_challenge);
	}

	let mut sumcheck = MultilinearEvalProver::new(witness, eval_point, eval_claim)?;
	for _ in 0..n_vars {
		let mut round_coeffs_vec = sumcheck.execute()?;
		let round_coeffs = round_coeffs_vec
			.pop()
			.expect("MultilinearEvalProver proves exactly one claim");
		let commitment = fri_folder.execute_fold_round();

		transcript
			.message()
			.write_scalar_slice(mlecheck::RoundProof::truncate(round_coeffs).coeffs());
		if let FoldRoundOutput::Commitment(commitment) = commitment {
			transcript.message().write(&commitment);
		}

		let challenge = transcript.sample();
		sumcheck.fold(challenge)?;
		fri_folder.receive_challenge(challenge);
	}

	let commitment = fri_folder.execute_fold_round();
	if let FoldRoundOutput::Commitment(commitment) = commitment {
		transcript.message().write(&commitment);
	}
	fri_folder.finish_proof(transcript);

	Ok(())
}

#[cfg(test)]
mod test {
	use anyhow::{Result, bail};
	use binius_field::{
		BinaryField, PackedBinaryGhash1x128b, PackedBinaryGhash2x128b, PackedBinaryGhash4x128b,
		PackedExtension, PackedField,
	};
	use binius_hash::{StdDigest, StdHashSuite};
	use binius_iop::{
		basefold as verifier_basefold, channel::OracleSpec, fri::ConstantArityStrategy,
	};
	use binius_math::{
		BinarySubspace, FieldBuffer,
		inner_product::inner_product_buffers,
		line::extrapolate_line_packed,
		multilinear::eq::eq_ind_partial_eval,
		ntt::{AdditiveNTT, NeighborsLastSingleThread, domain_context::GenericOnTheFly},
		test_utils::{random_field_buffer, random_scalars},
	};
	use binius_transcript::{
		ProverTranscript,
		fiat_shamir::{CanSample, HasherChallenger},
	};
	use binius_utils::rayon::prelude::*;
	use rand::{SeedableRng, rngs::StdRng};

	use super::{BaseFoldProver, prove_mlecheck_basefold_zk_batch};
	use crate::{
		fri::{self, CommitMaskedOutput, CommitOutput, FRIFoldProver},
		merkle_tree::prover::BinaryMerkleTreeProver,
	};

	type StdChallenger = HasherChallenger<StdDigest>;

	pub const LOG_INV_RATE: usize = 1;
	pub const SECURITY_BITS: usize = 32;

	fn calculate_n_test_queries(security_bits: usize, log_inv_rate: usize) -> usize {
		security_bits.div_ceil(log_inv_rate)
	}

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

		let merkle_prover = BinaryMerkleTreeProver::<F, StdHashSuite>::new();

		let subspace = BinarySubspace::with_dim(multilinear.log_len() + LOG_INV_RATE);
		let domain_context = GenericOnTheFly::generate_from_subspace(&subspace);
		let ntt = NeighborsLastSingleThread::new(domain_context);

		let n_test_queries = calculate_n_test_queries(SECURITY_BITS, LOG_INV_RATE);
		let fri_params = binius_iop::fri::FRIParams::with_strategy(
			ntt.domain_context(),
			merkle_prover.scheme(),
			multilinear.log_len(),
			None,
			LOG_INV_RATE,
			n_test_queries,
			&ConstantArityStrategy::new(2),
		);

		let CommitOutput {
			commitment: codeword_commitment,
			committed: codeword_committed,
			codeword,
		} = fri::commit_interleaved(&fri_params, 0, &ntt, &merkle_prover, multilinear.to_ref());

		let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
		prover_transcript.message().write(&codeword_commitment);

		let fri_folder =
			FRIFoldProver::new(&fri_params, &ntt, &merkle_prover, codeword, &codeword_committed);

		let prover = BaseFoldProver::new(multilinear, eval_point_eq, evaluation_claim, fri_folder);
		prover.prove(&mut prover_transcript)?;

		let mut verifier_transcript = prover_transcript.into_verifier();

		let retrieved_codeword_commitment = verifier_transcript.message().read()?;

		let verifier_basefold::ReducedOutput {
			final_fri_value,
			final_sumcheck_value,
			challenges,
		} = verifier_basefold::verify(
			&fri_params,
			merkle_prover.scheme(),
			retrieved_codeword_commitment,
			evaluation_claim,
			&mut verifier_transcript,
		)?;

		if !verifier_basefold::sumcheck_fri_consistency(
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

		let witness = random_field_buffer::<P>(&mut rng, n_vars);
		let evaluation_point = random_scalars::<F>(&mut rng, n_vars);

		let eval_point_eq = eq_ind_partial_eval(&evaluation_point);
		let evaluation_claim = inner_product_buffers(&witness, &eval_point_eq);

		(witness, evaluation_point, evaluation_claim)
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

	/// Drives [`prove_mlecheck_basefold_zk_batch`] against
	/// [`binius_iop::basefold::verify_mlecheck_basefold_zk_batch`] for a single oracle (`k = 1`, no
	/// outer rounds): commits the interleaved (π ‖ ω) codeword, samples the masking challenge γ,
	/// forms π' = (1-γ)π + γω, and proves/verifies the point-evaluation claim π'(ρ) via the
	/// combined FRI path. If `tamper`, the claim is corrupted and verification must fail. (The
	/// multi-oracle path is exercised end-to-end by the channel tests.)
	fn run_mlecheck_basefold_zk_prove_and_verify<F, P>(
		witness: FieldBuffer<P>,
		evaluation_point: Vec<F>,
		tamper: bool,
	) -> Result<()>
	where
		F: BinaryField,
		P: PackedField<Scalar = F> + PackedExtension<F>,
	{
		let n_vars = evaluation_point.len();
		assert_eq!(witness.log_len(), n_vars);

		let merkle_prover = BinaryMerkleTreeProver::<F, StdHashSuite>::new();

		let subspace = BinarySubspace::with_dim(n_vars + 1 + LOG_INV_RATE);
		let domain_context = GenericOnTheFly::generate_from_subspace(&subspace);
		let ntt = NeighborsLastSingleThread::new(domain_context);

		// For a single oracle the combined opening params (`optimal_for_batch`) also satisfy
		// `commit_masked`'s preconditions (`log_batch_size() == 1`, `rs_code().log_dim() ==
		// n_vars`).
		let (fri_params, _) = binius_iop::fri::FRIParams::optimal_for_batch(
			ntt.domain_context(),
			merkle_prover.scheme(),
			&[OracleSpec::new_zk(n_vars)],
			LOG_INV_RATE,
			32,
		);

		// Commit the interleaved (witness ‖ mask), generating the mask internally.
		let mut commit_rng = StdRng::seed_from_u64(7);
		let CommitMaskedOutput {
			commitment: codeword_commitment,
			committed: codeword_committed,
			codeword,
			mask,
		} = fri::commit_masked(
			&fri_params,
			0,
			&ntt,
			&merkle_prover,
			witness.to_ref(),
			&mut commit_rng,
		);

		let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
		prover_transcript.message().write(&codeword_commitment);

		// Sample the masking challenge γ and form π' = (1-γ)·witness + γ·mask.
		let batch_challenge: F = prover_transcript.sample();
		let mut witness_prime = witness.clone();
		let gamma_broadcast = P::broadcast(batch_challenge);
		(witness_prime.as_mut(), mask.as_ref())
			.into_par_iter()
			.for_each(|(w, &m)| {
				*w = extrapolate_line_packed(*w, m, gamma_broadcast);
			});

		let eval_point_eq = eq_ind_partial_eval::<P>(&evaluation_point);
		let mut eval_claim = inner_product_buffers(&witness_prime, &eval_point_eq);
		if tamper {
			eval_claim += F::ONE;
		}

		let fri_folder = FRIFoldProver::new_batch(
			&fri_params,
			&ntt,
			&merkle_prover,
			vec![(codeword, &codeword_committed)],
		);
		prove_mlecheck_basefold_zk_batch(
			witness_prime,
			&evaluation_point,
			eval_claim,
			batch_challenge,
			&[],
			fri_folder,
			&mut prover_transcript,
		)?;

		let mut verifier_transcript = prover_transcript.into_verifier();
		let retrieved_commitment = verifier_transcript.message().read()?;
		let batch_challenge_v: F = verifier_transcript.sample();

		let verifier_basefold::ReducedOutput {
			final_fri_value,
			final_sumcheck_value,
			..
		} = verifier_basefold::verify_mlecheck_basefold_zk_batch(
			&fri_params,
			merkle_prover.scheme(),
			&[retrieved_commitment],
			eval_claim,
			&evaluation_point,
			batch_challenge_v,
			&[],
			&mut verifier_transcript,
		)?;

		if !verifier_basefold::mlecheck_fri_consistency(final_fri_value, final_sumcheck_value) {
			bail!("MLE-check and FRI are inconsistent");
		}

		Ok(())
	}

	#[test]
	fn test_mlecheck_basefold_zk_valid_proof() {
		type P = PackedBinaryGhash1x128b;

		let n_vars = 8;
		let mut rng = StdRng::seed_from_u64(0);
		let witness = random_field_buffer::<P>(&mut rng, n_vars);
		let evaluation_point = random_scalars(&mut rng, n_vars);

		run_mlecheck_basefold_zk_prove_and_verify::<_, P>(witness, evaluation_point, false)
			.unwrap();
	}

	#[test]
	fn test_mlecheck_basefold_zk_invalid_proof() {
		type P = PackedBinaryGhash1x128b;

		let n_vars = 8;
		let mut rng = StdRng::seed_from_u64(0);
		let witness = random_field_buffer::<P>(&mut rng, n_vars);
		let evaluation_point = random_scalars(&mut rng, n_vars);

		let result =
			run_mlecheck_basefold_zk_prove_and_verify::<_, P>(witness, evaluation_point, true);
		assert!(result.is_err());
	}
}
