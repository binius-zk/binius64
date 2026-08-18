// Copyright 2026 The Binius Developers

//! Compares real BaseFold/FRI proof size with and without same-round oracle merging.
//!
//! Both scenarios commit the exact same witness data over the exact same round shape.
//! The only difference is whether each round's oracles get one Merkle commitment each, or
//! one combined commitment per round.
//! Both proofs are also verified, so the size numbers below are for proofs that actually
//! check out.
//!
//! Run with: cargo run --release --example merge_channel_proof_size -p binius-iop-prover

use binius_compute::GlobalAllocator;
use binius_field::{BinaryField128bGhash, PackedBinaryGhash1x128b};
use binius_hash::{StdDigest, StdHashSuite};
use binius_iop::{
	basefold::compiler::BaseFoldVerifierCompiler,
	channel::{IOPVerifierChannel, OracleSpec, merge::MergeVerifierChannel},
	fri::MinProofSizeStrategy,
	merkle_tree::BinaryMerkleTreeScheme,
};
use binius_iop_prover::{
	basefold::compiler::BaseFoldProverCompiler,
	channel::{IOPProverChannel, merge::MergeProverChannel},
};
use binius_ip::channel::IPVerifierChannel;
use binius_ip_prover::channel::IPProverChannel;
use binius_math::{
	FieldBuffer,
	inner_product::inner_product_buffers,
	multilinear::eq::eq_ind_partial_eval,
	ntt::{NeighborsLastSingleThread, domain_context::GaoMateerOnTheFly},
	test_utils::{random_field_buffer, random_scalars},
};
use binius_transcript::{ProverTranscript, VerifierTranscript, fiat_shamir::HasherChallenger};
use binius_utils::checked_arithmetics::log2_ceil_usize;
use rand::{SeedableRng, rngs::StdRng};

type F = BinaryField128bGhash;
type P = PackedBinaryGhash1x128b;
type StdChallenger = HasherChallenger<StdDigest>;

// A rate-1/2 code with 32 test queries, the same parameters the crate's own BaseFold tests
// use.
const LOG_INV_RATE: usize = 1;
const SECURITY_BITS: usize = 32;

const fn n_test_queries() -> usize {
	SECURITY_BITS.div_ceil(LOG_INV_RATE)
}

fn make_ntt(log_domain_size: usize) -> NeighborsLastSingleThread<GaoMateerOnTheFly<F>> {
	NeighborsLastSingleThread::new(GaoMateerOnTheFly::generate(log_domain_size))
}

fn make_merkle_scheme() -> BinaryMerkleTreeScheme<F, StdHashSuite> {
	BinaryMerkleTreeScheme::new()
}

/// A random witness for one oracle, together with a transparent polynomial and the claim
/// their inner product produces.
fn generate_oracle_data(rng: &mut StdRng, n_vars: usize) -> (FieldBuffer<P>, FieldBuffer<P>, F) {
	let buffer = random_field_buffer::<P>(&mut *rng, n_vars);
	let point = random_scalars::<F>(&mut *rng, n_vars);
	let transparent = eq_ind_partial_eval::<P>(&point);
	let claim = inner_product_buffers(&buffer, &transparent);
	(buffer, transparent, claim)
}

/// Commits, proves, and verifies every oracle in `rounds` as its own commitment.
///
/// Returns the finalized proof size in bytes.
fn run_unmerged(rounds: &[&[usize]]) -> usize {
	let mut rng = StdRng::seed_from_u64(0);
	let sizes: Vec<usize> = rounds
		.iter()
		.flat_map(|round| round.iter().copied())
		.collect();
	let data: Vec<(FieldBuffer<P>, FieldBuffer<P>, F)> = sizes
		.iter()
		.map(|&n| generate_oracle_data(&mut rng, n))
		.collect();
	let oracle_specs: Vec<OracleSpec> = sizes.iter().map(|&n| OracleSpec::new_zk(n)).collect();

	let verifier_compiler = BaseFoldVerifierCompiler::new(
		&make_merkle_scheme(),
		oracle_specs,
		LOG_INV_RATE,
		n_test_queries(),
		&MinProofSizeStrategy,
	);
	let ntt = make_ntt(verifier_compiler.max_log_domain_size());
	let prover_compiler =
		BaseFoldProverCompiler::<P, _>::from_verifier_compiler(&verifier_compiler, ntt);

	// Prover side: one commitment per oracle, in arrival order.
	let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
	let mut prover_channel = prover_compiler
		.create_channel_from_transcript::<StdHashSuite, StdChallenger, _, _>(
			&mut prover_transcript,
			StdRng::seed_from_u64(1),
			GlobalAllocator,
		);

	let oracles: Vec<_> = data
		.iter()
		.map(|(buffer, _, _)| prover_channel.send_oracle(buffer.to_ref()))
		.collect();
	for (&oracle, (_, transparent, claim)) in oracles.iter().zip(&data) {
		prover_channel.prove_oracle_relation(oracle, transparent.clone(), *claim);
	}
	for (oracle, (buffer, _, _)) in oracles.iter().copied().zip(data.iter().cloned()) {
		prover_channel.finalize_oracle(oracle, buffer);
	}
	prover_channel.finish();
	let proof = prover_transcript.finalize();
	let proof_size = proof.len();

	// Verifier side: replay the exact same commitments and relations.
	let mut verifier_transcript = VerifierTranscript::new(StdChallenger::default(), proof);
	let mut verifier_channel = verifier_compiler
		.create_channel_from_transcript::<StdHashSuite, StdChallenger, _>(&mut verifier_transcript);

	let v_oracles: Vec<_> = sizes
		.iter()
		.map(|&n| verifier_channel.recv_oracle(n, true).unwrap())
		.collect();
	for (oracle, (_, transparent, claim)) in v_oracles.into_iter().zip(data) {
		verifier_channel
			.verify_oracle_relation(
				oracle,
				Box::new(move |point: &[F]| {
					let eq = eq_ind_partial_eval::<P>(point);
					inner_product_buffers(&transparent, &eq)
				}),
				claim,
			)
			.unwrap();
	}
	verifier_channel
		.finish()
		.expect("unmerged proof must verify");

	proof_size
}

/// Commits, proves, and verifies every round in `rounds` as one combined commitment.
///
/// Returns the finalized proof size in bytes.
fn run_merged(rounds: &[&[usize]]) -> usize {
	let mut rng = StdRng::seed_from_u64(0);
	let fine_sizes: Vec<usize> = rounds
		.iter()
		.flat_map(|round| round.iter().copied())
		.collect();
	let fine_specs: Vec<OracleSpec> = fine_sizes.iter().map(|&n| OracleSpec::new_zk(n)).collect();
	let data: Vec<(FieldBuffer<P>, FieldBuffer<P>, F)> = fine_sizes
		.iter()
		.map(|&n| generate_oracle_data(&mut rng, n))
		.collect();

	// One coarse spec per round, sized to fit that round's total.
	let coarse_specs: Vec<OracleSpec> = rounds
		.iter()
		.map(|sizes| {
			let total: usize = sizes.iter().map(|&n| 1usize << n).sum();
			OracleSpec::new_zk(log2_ceil_usize(total))
		})
		.collect();

	let verifier_compiler = BaseFoldVerifierCompiler::new(
		&make_merkle_scheme(),
		coarse_specs,
		LOG_INV_RATE,
		n_test_queries(),
		&MinProofSizeStrategy,
	);
	let ntt = make_ntt(verifier_compiler.max_log_domain_size());
	let prover_compiler =
		BaseFoldProverCompiler::<P, _>::from_verifier_compiler(&verifier_compiler, ntt);

	// Prover side: every round is sent, then sampled once, so the merging decorator commits
	// each round as a single oracle underneath.
	let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
	let base_channel = prover_compiler
		.create_channel_from_transcript::<StdHashSuite, StdChallenger, _, _>(
			&mut prover_transcript,
			StdRng::seed_from_u64(1),
			GlobalAllocator,
		);
	let mut merge_channel = MergeProverChannel::new(base_channel, &fine_specs, GlobalAllocator);

	let mut oracles = Vec::new();
	let mut index = 0;
	for sizes in rounds {
		for _ in *sizes {
			let (buffer, _, _) = &data[index];
			oracles.push(merge_channel.send_oracle(buffer.to_ref()));
			index += 1;
		}
		merge_channel.sample();
	}
	for (&oracle, (_, transparent, claim)) in oracles.iter().zip(&data) {
		merge_channel.prove_oracle_relation(oracle, transparent.clone(), *claim);
	}
	for (oracle, (buffer, _, _)) in oracles.iter().copied().zip(data.iter().cloned()) {
		merge_channel.finalize_oracle(oracle, buffer);
	}
	merge_channel.into_inner().finish();
	let proof = prover_transcript.finalize();
	let proof_size = proof.len();

	// Verifier side: the same round boundaries, wrapped the same way.
	let mut verifier_transcript = VerifierTranscript::new(StdChallenger::default(), proof);
	let base_verifier_channel = verifier_compiler
		.create_channel_from_transcript::<StdHashSuite, StdChallenger, _>(&mut verifier_transcript);
	let mut merge_verifier_channel = MergeVerifierChannel::new(base_verifier_channel, &fine_specs);

	let mut v_oracles = Vec::new();
	for sizes in rounds {
		for &n in *sizes {
			v_oracles.push(merge_verifier_channel.recv_oracle(n, true).unwrap());
		}
		merge_verifier_channel.sample();
	}
	for (oracle, (_, transparent, claim)) in v_oracles.into_iter().zip(data) {
		merge_verifier_channel
			.verify_oracle_relation(
				oracle,
				Box::new(move |point: &[F]| {
					let eq = eq_ind_partial_eval::<P>(point);
					inner_product_buffers(&transparent, &eq)
				}),
				claim,
			)
			.unwrap();
	}
	merge_verifier_channel
		.into_inner()
		.expect("merged proof must verify")
		.finish()
		.expect("merged proof must verify");

	proof_size
}

fn main() {
	// Two rounds, three oracles each: sizes are log2 variable counts.
	let rounds: &[&[usize]] = &[&[8, 7, 7], &[9, 6, 6]];
	let n_oracles: usize = rounds.iter().map(|round| round.len()).sum();
	let n_rounds = rounds.len();

	let unmerged_bytes = run_unmerged(rounds);
	let merged_bytes = run_merged(rounds);

	let reduction = 100.0 * (1.0 - merged_bytes as f64 / unmerged_bytes as f64);

	println!("Same-round oracle merging: proof-size comparison");
	println!("Rounds (log2 oracle sizes): {rounds:?}");
	println!();
	println!("Without merging: {n_oracles} commitments -> {unmerged_bytes} bytes");
	println!("With merging:    {n_rounds} commitments -> {merged_bytes} bytes");
	println!("Reduction: {reduction:.1}% (both proofs verified)");
}
