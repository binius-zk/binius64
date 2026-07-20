// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

//! The core BaseFold opening protocol on the prover side.

use binius_field::{BinaryField, PackedField};
use binius_ip::mlecheck;
use binius_ip_prover::sumcheck::{common::SumcheckProver, multilinear_eval::MultilinearEvalProver};
use binius_math::{FieldBuffer, ntt::AdditiveNTT};

use crate::{fri::FRIFoldProver, merkle_channel::MerkleIPProverChannel};

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
/// * `channel` - the Merkle channel carrying all prover interaction: round coefficients,
///   challenges, commitments, and query openings
///
/// The final FRI value equals the final MLE-check value `𝛑(r)` (see
/// [`binius_iop::basefold::mlecheck_fri_consistency`]).
pub fn prove_mlecheck_basefold<F, P, NTT, Channel>(
	witness: FieldBuffer<P>,
	eval_point: &[F],
	eval_claim: F,
	batch_challenge: Option<F>,
	outer_challenges: &[F],
	mut fri_folder: FRIFoldProver<'_, F, P, NTT, Channel::Commitment>,
	channel: &mut Channel,
) where
	F: BinaryField,
	P: PackedField<Scalar = F>,
	NTT: AdditiveNTT<Field = F> + Sync,
	Channel: MerkleIPProverChannel<F>,
{
	let _scope = tracing::debug_span!("Basefold MLE-check ZK (batched)").entered();

	let n_vars = witness.log_len();
	assert_eq!(eval_point.len(), n_vars);
	// The FRI folder has `n_inner` inner (unbatch) rounds — one for the shared mask challenge γ
	// when any ZK oracle is present, none otherwise — `log_n_oracles` outer (oracle-combine)
	// rounds, and `𝐧` standard fold rounds.
	let n_inner = usize::from(batch_challenge.is_some());
	assert_eq!(n_vars + n_inner + outer_challenges.len(), fri_folder.n_rounds());

	// Inner (unbatch) round: fold every interleaved (π_i ‖ ω_i) ZK codeword at the masking
	// challenge.
	if let Some(gamma) = batch_challenge {
		fri_folder.receive_challenge(gamma);
	}
	// Outer rounds: combine the k lifted codewords at the batching challenges r'. These carry no
	// sumcheck round-polynomial; the folder applies them lazily at the first commit round. The
	// FirstFold consumes its accumulated challenges as `[early ++ outer ++ later]`, so feeding the
	// outer challenges here (right after γ, before any standard round) lands them in the outer
	// window; the leading standard rounds then supply each non-ZK oracle's later batch fold.
	for &outer_challenge in outer_challenges {
		fri_folder.receive_challenge(outer_challenge);
	}

	let mut sumcheck = MultilinearEvalProver::new(witness, eval_point, eval_claim);
	for _ in 0..n_vars {
		let mut round_coeffs_vec = sumcheck.execute();
		let round_coeffs = round_coeffs_vec
			.pop()
			.expect("MultilinearEvalProver proves exactly one claim");

		// Send the round coefficients first: on a commitment round, `execute_fold_round` commits
		// the folded codeword and writes its root, which must land after the coefficients.
		channel.send_many(mlecheck::RoundProof::truncate(round_coeffs).coeffs());
		fri_folder.execute_fold_round(channel);

		let challenge = channel.sample();
		sumcheck.fold(challenge);
		fri_folder.receive_challenge(challenge);
	}

	fri_folder.execute_fold_round(channel);
	fri_folder.finish_proof(channel);
}

#[cfg(test)]
mod test {
	use anyhow::{Result, bail};
	use binius_field::{BinaryField, PackedBinaryGhash1x128b, PackedField};
	use binius_hash::{StdDigest, StdHashSuite};
	use binius_iop::{
		basefold as verifier_basefold,
		channel::OracleSpec,
		merkle_channel::{MerkleIPVerifierChannel, VerifierMerkleTranscriptChannel},
	};
	use binius_ip::channel::IPVerifierChannel;
	use binius_ip_prover::channel::IPProverChannel;
	use binius_math::{
		BinarySubspace, FieldBuffer,
		inner_product::inner_product_buffers,
		line::extrapolate_line_packed,
		multilinear::eq::eq_ind_partial_eval,
		ntt::{AdditiveNTT, NeighborsLastSingleThread, domain_context::GenericOnTheFly},
		test_utils::{random_field_buffer, random_scalars},
	};
	use binius_transcript::{ProverTranscript, fiat_shamir::HasherChallenger};
	use binius_utils::rayon::prelude::*;
	use rand::{SeedableRng, rngs::StdRng};

	use super::prove_mlecheck_basefold;
	use crate::{
		fri::{self, FRIFoldProver, MaskedCodeword},
		merkle_channel::{MerkleIPProverChannel, ProverMerkleTranscriptChannel},
		merkle_tree::prover::BinaryMerkleTreeProver,
	};

	type StdChallenger = HasherChallenger<StdDigest>;

	pub const LOG_INV_RATE: usize = 1;

	/// Drives [`prove_mlecheck_basefold`] against
	/// [`binius_iop::basefold::verify_mlecheck_basefold`] for a single oracle (`k = 1`, no
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
		P: PackedField<Scalar = F>,
	{
		let n_vars = evaluation_point.len();
		assert_eq!(witness.log_len(), n_vars);

		let merkle_prover = BinaryMerkleTreeProver::<F, StdHashSuite>::new();

		let subspace = BinarySubspace::with_dim(n_vars + 1 + LOG_INV_RATE);
		let domain_context = GenericOnTheFly::generate_from_subspace(&subspace);
		let ntt = NeighborsLastSingleThread::new(domain_context);

		// For a single oracle the combined opening params (`optimal_for_batch`) also satisfy
		// `encode_masked`'s preconditions (`log_batch_size() == 1`, `rs_code().log_dim() ==
		// n_vars`).
		let (fri_params, _) = binius_iop::fri::FRIParams::optimal_for_batch(
			ntt.domain_context(),
			merkle_prover.scheme(),
			&[OracleSpec::new_zk(n_vars)],
			LOG_INV_RATE,
			32,
		);

		// Encode the interleaved (witness ‖ mask), generating the mask internally, and commit the
		// codeword over the Merkle channel.
		let mut commit_rng = StdRng::seed_from_u64(7);
		let MaskedCodeword { codeword, mask } =
			fri::encode_masked(&fri_params, 0, &ntt, witness.to_ref(), &mut commit_rng);

		let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
		let mut prover_channel =
			ProverMerkleTranscriptChannel::<_, StdChallenger, _, StdHashSuite>::with_merkle_prover(
				&mut prover_transcript,
				merkle_prover,
			);
		let codeword_commitment = prover_channel.send_merkle_commitment(codeword.to_ref(), 2);

		// Sample the masking challenge γ and form π' = (1-γ)·witness + γ·mask.
		let batch_challenge: F = IPProverChannel::sample(&mut prover_channel);
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

		let fri_folder =
			FRIFoldProver::new_batch(&fri_params, &ntt, vec![(codeword, codeword_commitment)]);
		prove_mlecheck_basefold(
			witness_prime,
			&evaluation_point,
			eval_claim,
			Some(batch_challenge),
			&[],
			fri_folder,
			&mut prover_channel,
		);
		drop(prover_channel);

		let mut verifier_transcript = prover_transcript.into_verifier();
		let mut verifier_channel =
			VerifierMerkleTranscriptChannel::<_, StdChallenger, _, StdHashSuite>::new(
				&mut verifier_transcript,
			);
		// The committed codeword has one interleaved (π ‖ ω) coset of 2 scalars per leaf.
		let retrieved_commitment =
			verifier_channel.recv_merkle_commitment(2, n_vars + LOG_INV_RATE)?;
		let batch_challenge_v: F = IPVerifierChannel::sample(&mut verifier_channel);

		let verifier_basefold::ReducedOutput {
			final_fri_value,
			final_sumcheck_value,
			..
		} = verifier_basefold::verify_mlecheck_basefold(
			&fri_params,
			&[retrieved_commitment],
			eval_claim,
			&evaluation_point,
			Some(batch_challenge_v),
			&[],
			&mut verifier_channel,
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
