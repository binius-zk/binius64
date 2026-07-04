// Copyright 2024-2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use std::{iter, vec};

use binius_field::{
	BinaryField, BinaryField128bGhash as B128, Field, PackedBinaryGhash1x128b, PackedField,
};
use binius_hash::{StdDigest, StdHashSuite};
use binius_iop::{
	fri::{self, CodewordSpec, FRIFoldVerifier, FRIParams, verify::FRIQueryVerifier},
	merkle_channel::{MerkleIPVerifierChannel, VerifierMerkleTranscriptChannel},
};
use binius_ip::channel::IPVerifierChannel;
use binius_ip_prover::channel::IPProverChannel;
use binius_math::{
	BinarySubspace, ReedSolomonCode,
	multilinear::{eq::eq_ind_partial_eval_scalars, evaluate::evaluate},
	ntt::{AdditiveNTT, NeighborsLastSingleThread, domain_context::GenericOnTheFly},
	test_utils::{Packed128b, random_field_buffer},
};
use binius_transcript::{ProverTranscript, fiat_shamir::HasherChallenger};
use binius_utils::checked_arithmetics::log2_ceil_usize;
use rand::prelude::*;

use super::{FRIFoldProver, encode_interleaved};
use crate::merkle_channel::{MerkleIPProverChannel, ProverMerkleTranscriptChannel};

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

	// Prover encodes the message and commits the codeword over a Merkle channel.
	let codeword = encode_interleaved(&params, 0, &ntt, msg.to_ref());

	let mut prover_challenger = ProverTranscript::new(StdChallenger::default());
	let mut prover_channel =
		ProverMerkleTranscriptChannel::<_, StdChallenger, _, StdHashSuite>::new(
			&mut prover_challenger,
		);
	let codeword_commitment =
		prover_channel.send_merkle_commitment(codeword.to_ref(), 1 << log_batch_size);

	// Run the prover to generate the proximity proof
	let mut round_prover = FRIFoldProver::new(&params, &ntt, codeword, codeword_commitment);

	// Note: The prover does an initial fold round before receiving any challenges
	// This is round 0, which won't produce a commitment when log_batch_size > 0
	round_prover.execute_fold_round(&mut prover_channel);

	for _i in 0..params.n_fold_rounds() {
		let challenge: F = IPProverChannel::sample(&mut prover_channel);
		round_prover.receive_challenge(challenge);
		round_prover.execute_fold_round(&mut prover_channel);
	}

	round_prover.finish_proof(&mut prover_channel);
	drop(prover_channel);
	// Now run the verifier, receiving commitments and openings over a Merkle channel.
	let mut verifier_challenger = prover_challenger.into_verifier();
	let mut channel = VerifierMerkleTranscriptChannel::<_, StdChallenger, _, StdHashSuite>::new(
		&mut verifier_challenger,
	);
	// The committed codeword's Merkle tree has one interleaved coset per leaf.
	let codeword_commitment = channel
		.recv_merkle_commitment(1 << log_batch_size, log_dimension + log_inv_rate)
		.unwrap();
	let mut verifier_challenges = Vec::with_capacity(params.n_fold_rounds());

	assert_eq!(params.fold_arities().len(), n_round_commitments);

	// The prover executes fold rounds starting from round 0, then receives challenges and continues
	// We need to match this pattern in the verifier
	let mut fri_fold_verifier = FRIFoldVerifier::new(&params);

	// Process initial round (before any challenges) - round 0
	fri_fold_verifier.process_round(&mut channel).unwrap();

	// Process remaining rounds with challenges
	for _ in 0..params.n_fold_rounds() {
		verifier_challenges.push(IPVerifierChannel::<F>::sample(&mut channel));
		fri_fold_verifier.process_round(&mut channel).unwrap();
	}

	let round_commitments = fri_fold_verifier.finalize();

	assert_eq!(verifier_challenges.len(), params.n_fold_rounds());

	let verifier = FRIQueryVerifier::new(
		&params,
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

	let final_fri_value = verifier.verify(&mut channel).unwrap();
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

	// The reduced Reed-Solomon code is shared by every input oracle, so the domain only needs to
	// cover its length.
	let subspace = BinarySubspace::with_dim(log_dim + log_inv_rate);
	let domain_context = GenericOnTheFly::generate_from_subspace(&subspace);
	let ntt = NeighborsLastSingleThread::new(domain_context);

	// Each oracle has RS dimension `log_dim` (no lifting), so every oracle sits at the reduced
	// (first-round) dimension `log_dim`. These are all non-ZK oracles, so their batch folds are
	// "later" challenges (sampled after the outer oracle-combine challenges); oracle `i` folds
	// with the `log_batch_size_i`-length suffix of the later group.
	let oracle_specs = log_batch_sizes
		.iter()
		.map(|&log_batch_size| CodewordSpec {
			log_lift: 0,
			log_early_batch_size: 0,
			log_later_batch_size: log_batch_size,
		})
		.collect::<Vec<_>>();
	let rs_code =
		ReedSolomonCode::with_domain_context_subspace(ntt.domain_context(), log_dim, log_inv_rate);
	let params = FRIParams::<F>::new_batch(rs_code, oracle_specs, vec![], n_test_queries);
	assert_eq!(params.rs_code().log_dim(), log_dim);

	// Encode each input oracle's interleaved codeword separately and commit it over the channel.
	let mut prover_challenger = ProverTranscript::new(StdChallenger::default());
	let mut prover_channel =
		ProverMerkleTranscriptChannel::<_, StdChallenger, _, StdHashSuite>::new(
			&mut prover_challenger,
		);
	let mut messages = Vec::new();
	let mut committed_codewords = Vec::new();
	for &log_batch_size in &log_batch_sizes {
		let oracle_params =
			FRIParams::new(params.rs_code().clone(), log_batch_size, vec![], n_test_queries);
		let msg = random_field_buffer::<P>(&mut rng, log_dim + log_batch_size);
		let codeword = encode_interleaved(&oracle_params, 0, &ntt, msg.to_ref());
		let commitment =
			prover_channel.send_merkle_commitment(codeword.to_ref(), 1 << log_batch_size);
		messages.push(msg);
		committed_codewords.push((codeword, commitment));
	}

	// Run the prover to fold the committed codewords.
	let mut round_prover = FRIFoldProver::new_batch(&params, &ntt, committed_codewords);

	round_prover.execute_fold_round(&mut prover_channel);
	for _ in 0..params.n_fold_rounds() {
		let challenge: F = IPProverChannel::sample(&mut prover_channel);
		round_prover.receive_challenge(challenge);
		round_prover.execute_fold_round(&mut prover_channel);
	}
	round_prover.finish_proof(&mut prover_channel);
	drop(prover_channel);

	// Run the verifier, receiving commitments and openings over a Merkle channel.
	let mut verifier_challenger = prover_challenger.into_verifier();
	let mut channel = VerifierMerkleTranscriptChannel::<_, StdChallenger, _, StdHashSuite>::new(
		&mut verifier_challenger,
	);
	// Each input oracle's Merkle tree has one interleaved coset per leaf and depth equal to its
	// own (possibly lifted) codeword dimension plus the inverse rate.
	let read_commitments = params
		.input_oracles()
		.iter()
		.map(|spec| {
			let depth = (params.rs_code().log_dim() - spec.log_lift) + log_inv_rate;
			channel
				.recv_merkle_commitment(1 << spec.log_batch_size(), depth)
				.unwrap()
		})
		.collect::<Vec<_>>();

	let mut verifier_challenges = Vec::with_capacity(params.n_fold_rounds());
	let mut fri_fold_verifier = FRIFoldVerifier::new(&params);
	fri_fold_verifier.process_round(&mut channel).unwrap();
	for _ in 0..params.n_fold_rounds() {
		verifier_challenges.push(IPVerifierChannel::<F>::sample(&mut channel));
		fri_fold_verifier.process_round(&mut channel).unwrap();
	}
	let round_commitments = fri_fold_verifier.finalize();

	let verifier = FRIQueryVerifier::new_batch(
		&params,
		&read_commitments,
		&round_commitments,
		&verifier_challenges,
	);
	let final_value = verifier.verify(&mut channel).unwrap();

	// The first-fold challenge slice is `[early ++ outer ++ later]`. Here every oracle is non-ZK,
	// so `max_early = 0` and the slice is `[outer ++ later]`. Oracle `i` folds with the
	// `log_batch_size_i`-length suffix of the later group; the remaining (tail) challenges fold the
	// shared reduced codeword. So the final value is
	//   sum_i outer_tensor[i] * evaluate(msg_i, reversed(later_i ++ tail)).
	let max_later = log_batch_sizes.iter().copied().max().unwrap();
	let log_n_oracles = log2_ceil_usize(log_batch_sizes.len());
	let first_fold_arity = params.log_batch_size();
	let outer = &verifier_challenges[..log_n_oracles];
	let later = &verifier_challenges[log_n_oracles..log_n_oracles + max_later];
	let tail = &verifier_challenges[first_fold_arity..];
	let outer_tensor = eq_ind_partial_eval_scalars::<F>(outer);

	let mut expected = F::ZERO;
	for (i, (msg, &log_batch_size)) in iter::zip(&messages, &log_batch_sizes).enumerate() {
		let later_i = &later[max_later - log_batch_size..];
		let mut eval_point = later_i.iter().chain(tail).copied().collect::<Vec<_>>();
		eval_point.reverse();
		expected += outer_tensor[i] * evaluate(msg, &eval_point);
	}
	assert_eq!(final_value, expected);
}

/// Full FRI round trip that batches a ZK oracle (early window) with a non-ZK oracle (later
/// window), exercising the heterogeneous `[early ++ outer ++ later]` challenge layout.
///
/// Oracle 0 is ZK with `log_batch_size = 1`, so it is purely *early* (`log_early_batch_size = 1`)
/// and folds with the single early challenge — the shared masking challenge γ a ZK BaseFold oracle
/// reserves. Oracle 1 is non-ZK with `log_batch_size = 2`, so it is purely *later*
/// (`log_later_batch_size = 2`) and folds with the two later challenges, which are sampled after
/// the `log_n_oracles = 1` outer (oracle-combine) challenge. The first fold therefore draws
/// `max_early + log_n_oracles + max_later = 1 + 1 + 2 = 4` challenges, with the two oracles folding
/// disjoint within-oracle windows on opposite sides of the outer challenge.
#[test]
fn test_commit_prove_verify_batched_mixed_skip() {
	type F = B128;
	type P = PackedBinaryGhash1x128b;

	let mut rng = StdRng::seed_from_u64(0);

	let log_dim = 8;
	let log_inv_rate = 2;
	// (log_batch_size, is_zk) per oracle. The ZK oracle (batch 1) folds with the early window
	// (before the outer challenge); the non-ZK oracle (batch 2) folds with the later window (after
	// the outer challenge).
	let oracle_params_spec = [(1usize, true), (2usize, false)];
	let n_test_queries = 3;

	// Every oracle reduces to the shared dimension `log_dim`, so no lifting is exercised here — the
	// focus is the inner-challenge windowing.
	let subspace = BinarySubspace::with_dim(log_dim + log_inv_rate);
	let domain_context = GenericOnTheFly::generate_from_subspace(&subspace);
	let ntt = NeighborsLastSingleThread::new(domain_context);

	// Every oracle sits at the reduced dimension `log_dim` (no lifting). The ZK oracle (batch 1)
	// is purely early; the non-ZK oracle (batch 2) is purely later. With `max_early = 1`,
	// `max_later = 2`, and `log_n_oracles = 1`, the first fold draws `1 + 1 + 2 = 4` challenges.
	let oracle_specs = oracle_params_spec
		.iter()
		.map(|&(log_batch_size, is_zk)| CodewordSpec {
			log_lift: 0,
			log_early_batch_size: if is_zk { log_batch_size } else { 0 },
			log_later_batch_size: if is_zk { 0 } else { log_batch_size },
		})
		.collect::<Vec<_>>();
	let rs_code =
		ReedSolomonCode::with_domain_context_subspace(ntt.domain_context(), log_dim, log_inv_rate);
	let params = FRIParams::<F>::new_batch(rs_code, oracle_specs, vec![], n_test_queries);
	assert_eq!(params.rs_code().log_dim(), log_dim);

	// Encode each input oracle's interleaved codeword separately and commit it over the channel.
	let mut prover_challenger = ProverTranscript::new(StdChallenger::default());
	let mut prover_channel =
		ProverMerkleTranscriptChannel::<_, StdChallenger, _, StdHashSuite>::new(
			&mut prover_challenger,
		);
	let mut messages = Vec::new();
	let mut committed_codewords = Vec::new();
	for &(log_batch_size, _) in &oracle_params_spec {
		let oracle_params =
			FRIParams::new(params.rs_code().clone(), log_batch_size, vec![], n_test_queries);
		let msg = random_field_buffer::<P>(&mut rng, log_dim + log_batch_size);
		let codeword = encode_interleaved(&oracle_params, 0, &ntt, msg.to_ref());
		let commitment =
			prover_channel.send_merkle_commitment(codeword.to_ref(), 1 << log_batch_size);
		messages.push(msg);
		committed_codewords.push((codeword, commitment));
	}

	// Run the prover to fold the committed codewords.
	let mut round_prover = FRIFoldProver::new_batch(&params, &ntt, committed_codewords);

	round_prover.execute_fold_round(&mut prover_channel);
	for _ in 0..params.n_fold_rounds() {
		let challenge: F = IPProverChannel::sample(&mut prover_channel);
		round_prover.receive_challenge(challenge);
		round_prover.execute_fold_round(&mut prover_channel);
	}
	round_prover.finish_proof(&mut prover_channel);
	drop(prover_channel);

	// Run the verifier, receiving commitments and openings over a Merkle channel.
	let mut verifier_challenger = prover_challenger.into_verifier();
	let mut channel = VerifierMerkleTranscriptChannel::<_, StdChallenger, _, StdHashSuite>::new(
		&mut verifier_challenger,
	);
	// Each input oracle's Merkle tree has one interleaved coset per leaf and depth equal to its
	// own (possibly lifted) codeword dimension plus the inverse rate.
	let read_commitments = params
		.input_oracles()
		.iter()
		.map(|spec| {
			let depth = (params.rs_code().log_dim() - spec.log_lift) + log_inv_rate;
			channel
				.recv_merkle_commitment(1 << spec.log_batch_size(), depth)
				.unwrap()
		})
		.collect::<Vec<_>>();

	let mut verifier_challenges = Vec::with_capacity(params.n_fold_rounds());
	let mut fri_fold_verifier = FRIFoldVerifier::new(&params);
	fri_fold_verifier.process_round(&mut channel).unwrap();
	for _ in 0..params.n_fold_rounds() {
		verifier_challenges.push(IPVerifierChannel::<F>::sample(&mut channel));
		fri_fold_verifier.process_round(&mut channel).unwrap();
	}
	let round_commitments = fri_fold_verifier.finalize();

	let verifier = FRIQueryVerifier::new_batch(
		&params,
		&read_commitments,
		&round_commitments,
		&verifier_challenges,
	);
	let final_value = verifier.verify(&mut channel).unwrap();

	// The first-fold challenge slice is `[early ++ outer ++ later]`. Oracle `i` folds with
	// `early_window ++ later_window`, the suffixes of the early and later groups of lengths
	// `log_early_batch_size_i` and `log_later_batch_size_i`. The remaining (tail) challenges fold
	// the shared reduced codeword. So the final value is
	//   sum_i outer_tensor[i] * evaluate(msg_i, reversed(early_i ++ later_i ++ tail)).
	let max_early = params
		.input_oracles()
		.iter()
		.map(|spec| spec.log_early_batch_size)
		.max()
		.unwrap();
	let max_later = params
		.input_oracles()
		.iter()
		.map(|spec| spec.log_later_batch_size)
		.max()
		.unwrap();
	let log_n_oracles = log2_ceil_usize(oracle_params_spec.len());
	let first_fold_arity = params.log_batch_size();
	let early = &verifier_challenges[..max_early];
	let outer = &verifier_challenges[max_early..max_early + log_n_oracles];
	let later = &verifier_challenges[max_early + log_n_oracles..first_fold_arity];
	let tail = &verifier_challenges[first_fold_arity..];
	let outer_tensor = eq_ind_partial_eval_scalars::<F>(outer);

	let mut expected = F::ZERO;
	for (i, (msg, &(_log_batch_size, _is_zk))) in
		iter::zip(&messages, &oracle_params_spec).enumerate()
	{
		let spec = &params.input_oracles()[i];
		let early_i = &early[max_early - spec.log_early_batch_size..];
		let later_i = &later[max_later - spec.log_later_batch_size..];
		let mut eval_point = early_i
			.iter()
			.chain(later_i)
			.chain(tail)
			.copied()
			.collect::<Vec<_>>();
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

	// A single shared domain context covers the reduced code; every per-oracle (smaller) code is
	// derived from it so their subspaces are nested prefixes of the reduced subspace.
	let subspace = BinarySubspace::with_dim(reduced_log_dim + log_inv_rate);
	let domain_context = GenericOnTheFly::generate_from_subspace(&subspace);
	let ntt = NeighborsLastSingleThread::new(domain_context);

	// Each oracle is lifted from its own RS dimension up to `reduced_log_dim` (`log_lift` is the
	// gap). These are all non-ZK oracles, so their batch folds are "later" challenges; oracle `i`
	// folds with the `log_batch_size_i`-length suffix of the later group.
	let oracle_specs = iter::zip(oracle_log_dims, log_batch_sizes)
		.map(|(oracle_log_dim, log_batch_size)| CodewordSpec {
			log_lift: reduced_log_dim - oracle_log_dim,
			log_early_batch_size: 0,
			log_later_batch_size: log_batch_size,
		})
		.collect::<Vec<_>>();
	let rs_code = ReedSolomonCode::with_domain_context_subspace(
		ntt.domain_context(),
		reduced_log_dim,
		log_inv_rate,
	);
	let params = FRIParams::<F>::new_batch(rs_code, oracle_specs, vec![], n_test_queries);
	assert_eq!(params.rs_code().log_dim(), reduced_log_dim);

	// Encode each input oracle's interleaved codeword separately, using its own smaller code built
	// from the shared domain context, and commit it over the channel.
	let mut prover_challenger = ProverTranscript::new(StdChallenger::default());
	let mut prover_channel =
		ProverMerkleTranscriptChannel::<_, StdChallenger, _, StdHashSuite>::new(
			&mut prover_challenger,
		);
	let mut messages = Vec::new();
	let mut committed_codewords = Vec::new();
	for (&log_dim, &log_batch_size) in iter::zip(&oracle_log_dims, &log_batch_sizes) {
		let rs_code = ReedSolomonCode::with_domain_context_subspace(
			ntt.domain_context(),
			log_dim,
			log_inv_rate,
		);
		let oracle_params = FRIParams::new(rs_code, log_batch_size, vec![], n_test_queries);
		let msg = random_field_buffer::<P>(&mut rng, log_dim + log_batch_size);
		let codeword = encode_interleaved(&oracle_params, 0, &ntt, msg.to_ref());
		let commitment =
			prover_channel.send_merkle_commitment(codeword.to_ref(), 1 << log_batch_size);
		messages.push(msg);
		committed_codewords.push((codeword, commitment));
	}

	// Run the prover to fold the committed codewords.
	let mut round_prover = FRIFoldProver::new_batch(&params, &ntt, committed_codewords);

	round_prover.execute_fold_round(&mut prover_channel);
	for _ in 0..params.n_fold_rounds() {
		let challenge: F = IPProverChannel::sample(&mut prover_channel);
		round_prover.receive_challenge(challenge);
		round_prover.execute_fold_round(&mut prover_channel);
	}
	round_prover.finish_proof(&mut prover_channel);
	drop(prover_channel);

	// Run the verifier, receiving commitments and openings over a Merkle channel.
	let mut verifier_challenger = prover_challenger.into_verifier();
	let mut channel = VerifierMerkleTranscriptChannel::<_, StdChallenger, _, StdHashSuite>::new(
		&mut verifier_challenger,
	);
	// Each input oracle's Merkle tree has one interleaved coset per leaf and depth equal to its
	// own (possibly lifted) codeword dimension plus the inverse rate.
	let read_commitments = params
		.input_oracles()
		.iter()
		.map(|spec| {
			let depth = (params.rs_code().log_dim() - spec.log_lift) + log_inv_rate;
			channel
				.recv_merkle_commitment(1 << spec.log_batch_size(), depth)
				.unwrap()
		})
		.collect::<Vec<_>>();

	let mut verifier_challenges = Vec::with_capacity(params.n_fold_rounds());
	let mut fri_fold_verifier = FRIFoldVerifier::new(&params);
	fri_fold_verifier.process_round(&mut channel).unwrap();
	for _ in 0..params.n_fold_rounds() {
		verifier_challenges.push(IPVerifierChannel::<F>::sample(&mut channel));
		fri_fold_verifier.process_round(&mut channel).unwrap();
	}
	let round_commitments = fri_fold_verifier.finalize();

	let verifier = FRIQueryVerifier::new_batch(
		&params,
		&read_commitments,
		&round_commitments,
		&verifier_challenges,
	);
	let final_value = verifier.verify(&mut channel).unwrap();

	// The first-fold challenge slice is `[early ++ outer ++ later]`. Every oracle is non-ZK here,
	// so `max_early = 0` and the slice is `[outer ++ later]`. Oracle `i` folds with the
	// `log_batch_size_i`-length suffix of the later group and combines with the outer-challenge
	// tensor. The remaining `reduced_log_dim` tail challenges fold the combined codeword.
	//
	// Lifting oracle `i` embeds its codeword as `Enc_M(ZeroPadMSB_eta(m_i'))`, whose multilinear
	// extension factors as `m_i'(low vars) * prod_j (1 - Y_j)` over the `eta_i = reduced_log_dim -
	// log_dim_i` padded high variables. Folding `Enc_M(X)` evaluates `X` at the reversed tail
	// challenges, so the padded high variables are bound by the first `eta_i` tail challenges
	// (contributing the factor `prod (1 - tail_k)`), and the surviving `log_dim_i` tail challenges
	// bind the real message. Hence the final value is
	//   sum_i outer_tensor[i] * prod_{k<eta_i}(1 - tail_k)
	//                          * evaluate(msg_i, reversed(later_i ++ tail[eta_i..])).
	let max_later = log_batch_sizes.iter().copied().max().unwrap();
	let log_n_oracles = log2_ceil_usize(log_batch_sizes.len());
	let first_fold_arity = params.log_batch_size();
	let outer = &verifier_challenges[..log_n_oracles];
	let later = &verifier_challenges[log_n_oracles..log_n_oracles + max_later];
	let tail = &verifier_challenges[first_fold_arity..];
	let outer_tensor = eq_ind_partial_eval_scalars::<F>(outer);

	let mut expected = F::ZERO;
	for (i, ((msg, &log_batch_size), &log_dim)) in
		iter::zip(iter::zip(&messages, &log_batch_sizes), &oracle_log_dims).enumerate()
	{
		let eta = reduced_log_dim - log_dim;
		// The lift (oracle padding) factor from the zero-padded high message variables.
		let pad_factor = tail[..eta].iter().map(|&t| F::ONE - t).product::<F>();
		let later_i = &later[max_later - log_batch_size..];
		let mut eval_point = later_i
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

	let committed_rs_code = ReedSolomonCode::<F>::new(log_dimension, log_inv_rate);

	let n_test_queries = 3;
	let params =
		FRIParams::new(committed_rs_code, log_batch_size, arities.to_vec(), n_test_queries);

	let subspace = BinarySubspace::with_dim(params.rs_code().log_len());
	let domain_context = GenericOnTheFly::generate_from_subspace(&subspace);
	let ntt = NeighborsLastSingleThread::new(domain_context);

	let msg = random_field_buffer::<P>(&mut rng, params.log_msg_len());

	let codeword = encode_interleaved(&params, 0, &ntt, msg.to_ref());

	let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
	let mut prover_channel =
		ProverMerkleTranscriptChannel::<_, StdChallenger, _, StdHashSuite>::new(
			&mut prover_transcript,
		);
	let codeword_commitment =
		prover_channel.send_merkle_commitment(codeword.to_ref(), 1 << log_batch_size);

	let mut round_prover = FRIFoldProver::new(&params, &ntt, codeword, codeword_commitment);

	round_prover.execute_fold_round(&mut prover_channel);

	for _ in 0..params.n_fold_rounds() {
		let challenge: F = IPProverChannel::sample(&mut prover_channel);
		round_prover.receive_challenge(challenge);
		round_prover.execute_fold_round(&mut prover_channel);
	}

	round_prover.finish_proof(&mut prover_channel);
	drop(prover_channel);

	let scheme = binius_iop::merkle_tree::BinaryMerkleTreeScheme::new();
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
