// Copyright 2024-2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use std::{iter, vec};

use binius_field::{
	BinaryField, BinaryField128bGhash as B128, Field, PackedBinaryGhash1x128b, PackedField,
};
use binius_hash::{StdDigest, StdHashSuite};
use binius_iop::fri::{
	self, FRIFoldVerifier, FRIParams, PartialOracleSpec, verify::FRIQueryVerifier,
};
use binius_math::{
	BinarySubspace, ReedSolomonCode,
	multilinear::{eq::eq_ind_partial_eval_scalars, evaluate::evaluate},
	ntt::{AdditiveNTT, NeighborsLastSingleThread, domain_context::GenericOnTheFly},
	test_utils::{Packed128b, random_field_buffer},
};
use binius_transcript::{
	ProverTranscript,
	fiat_shamir::{CanSample, HasherChallenger},
};
use rand::prelude::*;

use super::{CommitOutput, FRIFoldProver, FoldRoundOutput, commit_interleaved};
use crate::merkle_tree::prover::BinaryMerkleTreeProver;

type StdChallenger = HasherChallenger<StdDigest>;

fn test_commit_prove_verify_success<F, P>(
	log_dimension: usize,
	log_inv_rate: usize,
	log_batch_size: usize,
	arities: &[usize],
) where
	F: BinaryField,
	P: PackedField<Scalar = F>,
{
	let mut rng = StdRng::seed_from_u64(0);

	let merkle_prover = BinaryMerkleTreeProver::<_, StdHashSuite>::new();

	let committed_rs_code = ReedSolomonCode::<F>::new(log_dimension, log_inv_rate);

	let n_test_queries = 3;
	let params =
		FRIParams::new(committed_rs_code, log_batch_size, arities.to_vec(), n_test_queries);

	let subspace = BinarySubspace::with_dim(params.rs_code().log_len());
	let domain_context = GenericOnTheFly::generate_from_subspace(&subspace);
	let ntt = NeighborsLastSingleThread::new(domain_context);

	let n_round_commitments = arities.len();

	// Generate a random message
	let msg = random_field_buffer::<P>(&mut rng, params.log_msg_len());

	// Prover commits the message
	let CommitOutput {
		commitment: mut codeword_commitment,
		committed: codeword_committed,
		codeword,
	} = commit_interleaved(&params, &ntt, &merkle_prover, msg.to_ref());

	// Run the prover to generate the proximity proof
	let mut round_prover =
		FRIFoldProver::new(&params, &ntt, &merkle_prover, codeword, &codeword_committed);

	let mut prover_challenger = ProverTranscript::new(StdChallenger::default());
	prover_challenger.message().write(&codeword_commitment);

	// Note: The prover does an initial fold round before receiving any challenges
	// This is round 0, which won't produce a commitment when log_batch_size > 0
	let fold_round_output = round_prover.execute_fold_round();
	if let FoldRoundOutput::Commitment(round_commitment) = fold_round_output {
		prover_challenger.message().write(&round_commitment);
	}

	for _i in 0..params.n_fold_rounds() {
		let challenge = prover_challenger.sample();
		round_prover.receive_challenge(challenge);

		let fold_round_output = round_prover.execute_fold_round();
		if let FoldRoundOutput::Commitment(round_commitment) = fold_round_output {
			prover_challenger.message().write(&round_commitment);
		}
	}

	round_prover.finish_proof(&mut prover_challenger);
	// Now run the verifier
	let mut verifier_challenger = prover_challenger.into_verifier();
	codeword_commitment = verifier_challenger.message().read().unwrap();
	let mut verifier_challenges = Vec::with_capacity(params.n_fold_rounds());

	assert_eq!(params.fold_arities().len(), n_round_commitments);

	// The prover executes fold rounds starting from round 0, then receives challenges and continues
	// We need to match this pattern in the verifier
	let mut fri_fold_verifier = FRIFoldVerifier::new(&params);

	// Process initial round (before any challenges) - round 0
	fri_fold_verifier
		.process_round(&mut verifier_challenger.message())
		.unwrap();

	// Process remaining rounds with challenges
	for _ in 0..params.n_fold_rounds() {
		verifier_challenges.push(verifier_challenger.sample());
		fri_fold_verifier
			.process_round(&mut verifier_challenger.message())
			.unwrap();
	}

	let round_commitments = fri_fold_verifier.finalize();

	assert_eq!(verifier_challenges.len(), params.n_fold_rounds());

	let verifier = FRIQueryVerifier::new(
		&params,
		merkle_prover.scheme(),
		&codeword_commitment,
		&round_commitments,
		&verifier_challenges,
	);

	// The verifier checks the terminal codeword against its commitment internally (the terminal
	// codeword is sent in full at the end of the query phase).
	//
	// check c == t(r'_0, ..., r'_{\ell-1})
	// note that the prover is claiming that the final_message is [c]
	let mut eval_point = verifier_challenges.clone();
	eval_point.reverse();
	let computed_eval = evaluate(&msg, &eval_point);

	let final_fri_value = verifier.verify(&mut verifier_challenger).unwrap();
	assert_eq!(computed_eval, final_fri_value);
}

#[test]
fn test_commit_prove_verify_success_128b_full() {
	// This tests the case where we have a round commitment for every round
	let log_dimension = 8;
	let log_final_dimension = 1;
	let log_inv_rate = 2;
	let arities = vec![1; log_dimension - log_final_dimension];

	// TODO: Make this test pass with non-trivial packing width
	test_commit_prove_verify_success::<B128, PackedBinaryGhash1x128b>(
		log_dimension,
		log_inv_rate,
		0,
		&arities,
	);
}

#[test]
fn test_commit_prove_verify_success_128b_higher_arity() {
	let log_dimension = 8;
	let log_inv_rate = 2;
	let arities = [3, 2, 1];

	// TODO: Make this test pass with non-trivial packing width
	test_commit_prove_verify_success::<B128, PackedBinaryGhash1x128b>(
		log_dimension,
		log_inv_rate,
		0,
		&arities,
	);
}

#[test]
fn test_commit_prove_verify_success_128b_interleaved() {
	let log_dimension = 6;
	let log_inv_rate = 2;
	let log_batch_size = 2;
	let arities = [3, 2, 1];

	test_commit_prove_verify_success::<B128, Packed128b>(
		log_dimension,
		log_inv_rate,
		log_batch_size,
		&arities,
	);
}

#[test]
fn test_commit_prove_verify_success_128b_interleaved_packed() {
	let log_dimension = 6;
	let log_inv_rate = 2;
	let log_batch_size = 2;
	let arities = [3, 2, 1];

	test_commit_prove_verify_success::<B128, Packed128b>(
		log_dimension,
		log_inv_rate,
		log_batch_size,
		&arities,
	);
}

#[test]
fn test_commit_prove_verify_success_without_folding() {
	let log_dimension = 4;
	let log_inv_rate = 2;
	let log_batch_size = 2;

	test_commit_prove_verify_success::<B128, Packed128b>(
		log_dimension,
		log_inv_rate,
		log_batch_size,
		&[],
	);
}

/// Full FRI round trip that batches several initial oracles in the first fold.
///
/// The three input oracles share the same Reed-Solomon code (dimension and inverse rate) but have
/// differing batch sizes (1, 1, 2). The prover commits each interleaved codeword separately, folds
/// and combines them into a single first-round codeword via [`FRIFoldProver::new_batch`], and the
/// verifier reconstructs the same combined oracle via [`FRIQueryVerifier::new_batch`].
#[test]
fn test_commit_prove_verify_batched_multi_oracle() {
	type F = B128;
	type P = PackedBinaryGhash1x128b;

	let mut rng = StdRng::seed_from_u64(0);

	let log_dim = 8;
	let log_inv_rate = 2;
	let log_batch_sizes = [1usize, 1, 2];
	let n_test_queries = 3;

	let merkle_prover = BinaryMerkleTreeProver::<F, StdHashSuite>::new();

	// The reduced Reed-Solomon code is shared by every input oracle, so the domain only needs to
	// cover its length.
	let subspace = BinarySubspace::with_dim(log_dim + log_inv_rate);
	let domain_context = GenericOnTheFly::generate_from_subspace(&subspace);
	let ntt = NeighborsLastSingleThread::new(domain_context);

	// Each oracle has RS dimension `log_dim`, hence message length `log_dim + log_batch_size`.
	// Fixing every batch size forces the reduced (first-round) dimension to `log_dim`.
	let oracle_specs = log_batch_sizes
		.iter()
		.map(|&log_batch_size| PartialOracleSpec {
			log_msg_len: log_dim + log_batch_size,
			log_batch_size: Some(log_batch_size),
		})
		.collect::<Vec<_>>();
	let (params, _proof_size) = FRIParams::<F>::optimal_for_batch(
		ntt.domain_context(),
		merkle_prover.scheme(),
		&oracle_specs,
		log_inv_rate,
		n_test_queries,
	);
	assert_eq!(params.rs_code().log_dim(), log_dim);

	// Commit each input oracle's interleaved codeword separately.
	let mut messages = Vec::new();
	let mut commitments = Vec::new();
	let mut committeds = Vec::new();
	let mut codewords = Vec::new();
	for &log_batch_size in &log_batch_sizes {
		let oracle_params =
			FRIParams::new(params.rs_code().clone(), log_batch_size, vec![], n_test_queries);
		let msg = random_field_buffer::<P>(&mut rng, log_dim + log_batch_size);
		let CommitOutput {
			commitment,
			committed,
			codeword,
		} = commit_interleaved(&oracle_params, &ntt, &merkle_prover, msg.to_ref());
		messages.push(msg);
		commitments.push(commitment);
		committeds.push(committed);
		codewords.push(codeword);
	}

	// Run the prover: write the per-oracle codeword commitments, then fold.
	let committed_codewords = iter::zip(codewords, &committeds).collect::<Vec<_>>();
	let mut round_prover =
		FRIFoldProver::new_batch(&params, &ntt, &merkle_prover, committed_codewords);

	let mut prover_challenger = ProverTranscript::new(StdChallenger::default());
	for commitment in &commitments {
		prover_challenger.message().write(commitment);
	}

	let fold_round_output = round_prover.execute_fold_round();
	if let FoldRoundOutput::Commitment(round_commitment) = fold_round_output {
		prover_challenger.message().write(&round_commitment);
	}
	for _ in 0..params.n_fold_rounds() {
		let challenge = prover_challenger.sample();
		round_prover.receive_challenge(challenge);

		let fold_round_output = round_prover.execute_fold_round();
		if let FoldRoundOutput::Commitment(round_commitment) = fold_round_output {
			prover_challenger.message().write(&round_commitment);
		}
	}
	round_prover.finish_proof(&mut prover_challenger);

	// Run the verifier.
	let mut verifier_challenger = prover_challenger.into_verifier();
	let read_commitments = commitments
		.iter()
		.map(|_| verifier_challenger.message().read().unwrap())
		.collect::<Vec<_>>();

	let mut verifier_challenges = Vec::with_capacity(params.n_fold_rounds());
	let mut fri_fold_verifier = FRIFoldVerifier::new(&params);
	fri_fold_verifier
		.process_round(&mut verifier_challenger.message())
		.unwrap();
	for _ in 0..params.n_fold_rounds() {
		verifier_challenges.push(verifier_challenger.sample());
		fri_fold_verifier
			.process_round(&mut verifier_challenger.message())
			.unwrap();
	}
	let round_commitments = fri_fold_verifier.finalize();

	let verifier = FRIQueryVerifier::new_batch(
		&params,
		merkle_prover.scheme(),
		&read_commitments,
		&round_commitments,
		&verifier_challenges,
	);
	let final_value = verifier.verify(&mut verifier_challenger).unwrap();

	// The first fold reduces oracle `i` by its inner challenges (the last `log_batch_size_i` of the
	// first `max_log_batch_size` challenges) and combines the oracles with the outer-challenge
	// tensor. The remaining (tail) challenges fold the shared reduced codeword. So the final value
	// is   sum_i outer_tensor[i] * evaluate(msg_i, reversed(inner_i ++ tail)).
	let max_log_batch_size = log_batch_sizes.iter().copied().max().unwrap();
	let first_fold_arity = params.log_batch_size();
	let inner = &verifier_challenges[..max_log_batch_size];
	let outer = &verifier_challenges[max_log_batch_size..first_fold_arity];
	let tail = &verifier_challenges[first_fold_arity..];
	let outer_tensor = eq_ind_partial_eval_scalars::<F>(outer);

	let mut expected = F::ZERO;
	for (i, (msg, &log_batch_size)) in iter::zip(&messages, &log_batch_sizes).enumerate() {
		let inner_i = &inner[max_log_batch_size - log_batch_size..];
		let mut eval_point = inner_i.iter().chain(tail).copied().collect::<Vec<_>>();
		eval_point.reverse();
		expected += outer_tensor[i] * evaluate(msg, &eval_point);
	}
	assert_eq!(final_value, expected);
}

/// Full FRI round trip that batches initial oracles of **differing Reed-Solomon dimension**,
/// forcing oracle padding (Lifted FRI).
///
/// The three input oracles have RS dimensions `[6, 8, 4]` with fixed batch sizes `[1, 1, 2]`, so
/// the reduced (first-round) dimension is `max(6, 8, 4) = 8`. The dimension-6 and dimension-4
/// oracles are smaller than the reduced code and must be lifted (their folded codewords duplicated
/// `2^2` and `2^4` times) before they combine into the first-round codeword; the dimension-8 oracle
/// needs no lifting (`eta = 0`), exercising both paths in one batch. Every per-oracle code is built
/// from the same `domain_context` so the smaller subspaces are prefixes of the reduced one, as the
/// duplication identity requires.
#[test]
fn test_commit_prove_verify_lifted_multi_oracle() {
	type F = B128;
	type P = PackedBinaryGhash1x128b;

	let mut rng = StdRng::seed_from_u64(0);

	let oracle_log_dims = [6usize, 8, 4];
	let log_batch_sizes = [1usize, 1, 2];
	let log_inv_rate = 2;
	let n_test_queries = 3;
	// The reduced (first-round) code dimension is the largest oracle dimension.
	let reduced_log_dim = oracle_log_dims.iter().copied().max().unwrap();

	let merkle_prover = BinaryMerkleTreeProver::<F, StdHashSuite>::new();

	// A single shared domain context covers the reduced code; every per-oracle (smaller) code is
	// derived from it so their subspaces are nested prefixes of the reduced subspace.
	let subspace = BinarySubspace::with_dim(reduced_log_dim + log_inv_rate);
	let domain_context = GenericOnTheFly::generate_from_subspace(&subspace);
	let ntt = NeighborsLastSingleThread::new(domain_context);

	// Each oracle has message length `log_dim + log_batch_size`, with its batch size fixed so the
	// reduced dimension is forced to `reduced_log_dim`.
	let oracle_specs = iter::zip(oracle_log_dims, log_batch_sizes)
		.map(|(log_dim, log_batch_size)| PartialOracleSpec {
			log_msg_len: log_dim + log_batch_size,
			log_batch_size: Some(log_batch_size),
		})
		.collect::<Vec<_>>();
	let (params, _proof_size) = FRIParams::<F>::optimal_for_batch(
		ntt.domain_context(),
		merkle_prover.scheme(),
		&oracle_specs,
		log_inv_rate,
		n_test_queries,
	);
	assert_eq!(params.rs_code().log_dim(), reduced_log_dim);

	// Commit each input oracle's interleaved codeword separately, using its own smaller code built
	// from the shared domain context.
	let mut messages = Vec::new();
	let mut commitments = Vec::new();
	let mut committeds = Vec::new();
	let mut codewords = Vec::new();
	for (&log_dim, &log_batch_size) in iter::zip(&oracle_log_dims, &log_batch_sizes) {
		let rs_code = ReedSolomonCode::with_domain_context_subspace(
			ntt.domain_context(),
			log_dim,
			log_inv_rate,
		);
		let oracle_params = FRIParams::new(rs_code, log_batch_size, vec![], n_test_queries);
		let msg = random_field_buffer::<P>(&mut rng, log_dim + log_batch_size);
		let CommitOutput {
			commitment,
			committed,
			codeword,
		} = commit_interleaved(&oracle_params, &ntt, &merkle_prover, msg.to_ref());
		messages.push(msg);
		commitments.push(commitment);
		committeds.push(committed);
		codewords.push(codeword);
	}

	// Run the prover: write the per-oracle codeword commitments, then fold.
	let committed_codewords = iter::zip(codewords, &committeds).collect::<Vec<_>>();
	let mut round_prover =
		FRIFoldProver::new_batch(&params, &ntt, &merkle_prover, committed_codewords);

	let mut prover_challenger = ProverTranscript::new(StdChallenger::default());
	for commitment in &commitments {
		prover_challenger.message().write(commitment);
	}

	let fold_round_output = round_prover.execute_fold_round();
	if let FoldRoundOutput::Commitment(round_commitment) = fold_round_output {
		prover_challenger.message().write(&round_commitment);
	}
	for _ in 0..params.n_fold_rounds() {
		let challenge = prover_challenger.sample();
		round_prover.receive_challenge(challenge);

		let fold_round_output = round_prover.execute_fold_round();
		if let FoldRoundOutput::Commitment(round_commitment) = fold_round_output {
			prover_challenger.message().write(&round_commitment);
		}
	}
	round_prover.finish_proof(&mut prover_challenger);

	// Run the verifier.
	let mut verifier_challenger = prover_challenger.into_verifier();
	let read_commitments = commitments
		.iter()
		.map(|_| verifier_challenger.message().read().unwrap())
		.collect::<Vec<_>>();

	let mut verifier_challenges = Vec::with_capacity(params.n_fold_rounds());
	let mut fri_fold_verifier = FRIFoldVerifier::new(&params);
	fri_fold_verifier
		.process_round(&mut verifier_challenger.message())
		.unwrap();
	for _ in 0..params.n_fold_rounds() {
		verifier_challenges.push(verifier_challenger.sample());
		fri_fold_verifier
			.process_round(&mut verifier_challenger.message())
			.unwrap();
	}
	let round_commitments = fri_fold_verifier.finalize();

	let verifier = FRIQueryVerifier::new_batch(
		&params,
		merkle_prover.scheme(),
		&read_commitments,
		&round_commitments,
		&verifier_challenges,
	);
	let final_value = verifier.verify(&mut verifier_challenger).unwrap();

	// The first fold reduces oracle `i` by its inner challenges (the last `log_batch_size_i` of the
	// first `max_log_batch_size` challenges) and combines the oracles with the outer-challenge
	// tensor. The remaining `reduced_log_dim` tail challenges fold the combined codeword.
	//
	// Lifting oracle `i` embeds its codeword as `Enc_M(ZeroPadMSB_eta(m_i'))`, whose multilinear
	// extension factors as `m_i'(low vars) * prod_j (1 - Y_j)` over the `eta_i = reduced_log_dim -
	// log_dim_i` padded high variables. Folding `Enc_M(X)` evaluates `X` at the reversed tail
	// challenges, so the padded high variables are bound by the first `eta_i` tail challenges
	// (contributing the factor `prod (1 - tail_k)`), and the surviving `log_dim_i` tail challenges
	// bind the real message. Hence the final value is
	//   sum_i outer_tensor[i] * prod_{k<eta_i}(1 - tail_k)
	//                          * evaluate(msg_i, reversed(inner_i ++ tail[eta_i..])).
	let max_log_batch_size = log_batch_sizes.iter().copied().max().unwrap();
	let first_fold_arity = params.log_batch_size();
	let inner = &verifier_challenges[..max_log_batch_size];
	let outer = &verifier_challenges[max_log_batch_size..first_fold_arity];
	let tail = &verifier_challenges[first_fold_arity..];
	let outer_tensor = eq_ind_partial_eval_scalars::<F>(outer);

	let mut expected = F::ZERO;
	for (i, ((msg, &log_batch_size), &log_dim)) in
		iter::zip(iter::zip(&messages, &log_batch_sizes), &oracle_log_dims).enumerate()
	{
		let eta = reduced_log_dim - log_dim;
		// The lift (oracle padding) factor from the zero-padded high message variables.
		let pad_factor = tail[..eta].iter().map(|&t| F::ONE - t).product::<F>();
		let inner_i = &inner[max_log_batch_size - log_batch_size..];
		let mut eval_point = inner_i
			.iter()
			.chain(&tail[eta..])
			.copied()
			.collect::<Vec<_>>();
		eval_point.reverse();
		expected += outer_tensor[i] * pad_factor * evaluate(msg, &eval_point);
	}
	assert_eq!(final_value, expected);
}

/// Runs the FRI prover and returns the proof bytes along with the FRI params and Merkle scheme,
/// so the caller can check the estimated proof size.
fn generate_fri_proof<F, P>(
	log_dimension: usize,
	log_inv_rate: usize,
	log_batch_size: usize,
	arities: &[usize],
) -> (Vec<u8>, FRIParams<F>, binius_iop::merkle_tree::BinaryMerkleTreeScheme<F, StdHashSuite>)
where
	F: BinaryField,
	P: PackedField<Scalar = F>,
{
	let mut rng = StdRng::seed_from_u64(0);

	let merkle_prover = BinaryMerkleTreeProver::<_, StdHashSuite>::new();

	let committed_rs_code = ReedSolomonCode::<F>::new(log_dimension, log_inv_rate);

	let n_test_queries = 3;
	let params =
		FRIParams::new(committed_rs_code, log_batch_size, arities.to_vec(), n_test_queries);

	let subspace = BinarySubspace::with_dim(params.rs_code().log_len());
	let domain_context = GenericOnTheFly::generate_from_subspace(&subspace);
	let ntt = NeighborsLastSingleThread::new(domain_context);

	let msg = random_field_buffer::<P>(&mut rng, params.log_msg_len());

	let CommitOutput {
		commitment: codeword_commitment,
		committed: codeword_committed,
		codeword,
	} = commit_interleaved(&params, &ntt, &merkle_prover, msg.to_ref());

	let mut round_prover =
		FRIFoldProver::new(&params, &ntt, &merkle_prover, codeword, &codeword_committed);

	let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
	prover_transcript.message().write(&codeword_commitment);

	let fold_round_output = round_prover.execute_fold_round();
	if let FoldRoundOutput::Commitment(round_commitment) = fold_round_output {
		prover_transcript.message().write(&round_commitment);
	}

	for _ in 0..params.n_fold_rounds() {
		let challenge = prover_transcript.sample();
		round_prover.receive_challenge(challenge);

		let fold_round_output = round_prover.execute_fold_round();
		if let FoldRoundOutput::Commitment(round_commitment) = fold_round_output {
			prover_transcript.message().write(&round_commitment);
		}
	}

	round_prover.finish_proof(&mut prover_transcript);

	let scheme = merkle_prover.scheme().clone();
	let proof_bytes = prover_transcript.finalize();
	(proof_bytes, params, scheme)
}

#[test]
fn test_proof_size_higher_arity() {
	let (proof_bytes, params, scheme) =
		generate_fri_proof::<B128, PackedBinaryGhash1x128b>(8, 2, 0, &[3, 2, 1]);
	assert_eq!(proof_bytes.len(), fri::proof_size(&params, &scheme));
}

#[test]
fn test_proof_size_interleaved() {
	let (proof_bytes, params, scheme) = generate_fri_proof::<B128, Packed128b>(6, 2, 2, &[3, 2, 1]);
	assert_eq!(proof_bytes.len(), fri::proof_size(&params, &scheme));
}

#[test]
fn test_proof_size_no_folding() {
	let (proof_bytes, params, scheme) = generate_fri_proof::<B128, Packed128b>(4, 2, 2, &[]);
	assert_eq!(proof_bytes.len(), fri::proof_size(&params, &scheme));
}
