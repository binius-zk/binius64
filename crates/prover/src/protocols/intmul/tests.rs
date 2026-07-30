// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use binius_compute::GlobalAllocator;
use binius_core::word::Word;
use binius_field::{BinaryField128bGhash, PackedBinaryGhash2x128b, Random};
use binius_iop::channel::{OracleSpec, naive::NaiveVerifierChannel};
use binius_iop_prover::channel::naive::NaiveProverChannel;
use binius_math::{inner_product::inner_product_buffers, multilinear::eq::eq_ind_partial_eval};
use binius_transcript::{ProverTranscript, VerifierTranscript};
use binius_utils::checked_arithmetics::log2_ceil_usize;
use binius_verifier::{
	config::StdChallenger,
	protocols::intmul::{
		common::{IntMulOutput, LIMB_BITS},
		verify,
	},
};
use itertools::izip;
use rand::prelude::*;

use super::{
	prove::{IntMulProver, prove},
	witness::Witness,
};
use crate::fold_word::fold_words;

type F = BinaryField128bGhash;
type P = PackedBinaryGhash2x128b;

pub fn evaluate_witness(words: &[Word], eval_point: &[F]) -> F {
	let (prefix, suffix) = eval_point.split_at(Word::LOG_BITS);
	let prefix_tensor = eq_ind_partial_eval::<F>(prefix);
	let suffix_tensor = eq_ind_partial_eval::<F>(suffix);

	let partially_folded_witness =
		fold_words::<_, F, _>(&GlobalAllocator, words, prefix_tensor.as_ref());

	inner_product_buffers(&partially_folded_witness, &suffix_tensor)
}

/// The four operand columns `(a, b, c_lo, c_hi)` of an IntMul witness.
type WordColumns = (Vec<Word>, Vec<Word>, Vec<Word>, Vec<Word>);

/// Build a valid IntMul witness of `n` random products `a * b = c_hi || c_lo`.
fn random_products(rng: &mut impl Rng, n: usize) -> WordColumns {
	let mut a = Vec::with_capacity(n);
	let mut b = Vec::with_capacity(n);
	let mut c_lo = Vec::with_capacity(n);
	let mut c_hi = Vec::with_capacity(n);

	for _ in 0..n {
		let a_i = rng.random_range(1..u64::MAX);
		let b_i = rng.random_range(1..u64::MAX);

		let full_result = (a_i as u128) * (b_i as u128);

		a.push(Word::from_u64(a_i));
		b.push(Word::from_u64(b_i));
		c_lo.push(Word::from_u64(full_result as u64));
		c_hi.push(Word::from_u64((full_result >> 64) as u64));
	}

	(a, b, c_lo, c_hi)
}

#[test]
fn prove_and_verify() {
	let mut rng = StdRng::seed_from_u64(0);

	const LOG_EXPONENTS: usize = 5;
	let (a, b, c_lo, c_hi) = random_products(&mut rng, 1 << LOG_EXPONENTS);

	let alloc = GlobalAllocator;
	let witness = Witness::<_, P>::new(&alloc, &a, &b, &c_lo, &c_hi).unwrap();

	// The one oracle in the protocol is the logup* pushforward, over the table variables.
	let oracle_specs = [OracleSpec::new(LIMB_BITS)];

	// Run prover
	let mut prover_transcript = ProverTranscript::<StdChallenger>::default();
	let mut prover_channel =
		NaiveProverChannel::<F, _>::new(&mut prover_transcript, oracle_specs.to_vec());
	let mut prover = IntMulProver::new(0, &mut prover_channel, &alloc);
	let prove_output = prover.prove(witness);
	prover_channel.finish();

	let IntMulOutput {
		eval_point,
		a_evals,
		b_evals,
		c_lo_evals,
		c_hi_evals,
	} = prove_output.clone();

	// Instead of evaluating each exponent bit column
	// separately, we batch them together with a `z_challenge`
	// and check consistency by evaluating at a single point `consistency_check_eval_point`.
	let z_challenge: Vec<F> = (0..Word::LOG_BITS).map(|_| F::random(&mut rng)).collect();
	let z_tensor = eq_ind_partial_eval::<F>(&z_challenge);
	let consistency_check_eval_point = [z_challenge, eval_point].concat();
	let get_consistency_check_eval =
		|evals| izip!(evals, z_tensor.as_ref()).map(|(x, y)| x * y).sum();

	let test_cases = [
		(a, a_evals),
		(b, b_evals),
		(c_lo, c_lo_evals),
		(c_hi, c_hi_evals),
	];

	for (exponents, evals) in test_cases {
		let expected_eval = evaluate_witness(&exponents, &consistency_check_eval_point);
		let given_eval = get_consistency_check_eval(evals);
		assert_eq!(expected_eval, given_eval);
	}
	// Run verifier
	let mut verifier_transcript = prover_transcript.into_verifier();
	let mut verifier_channel = NaiveVerifierChannel::new(&mut verifier_transcript, &oracle_specs);
	let verify_output = verify(LOG_EXPONENTS, &mut verifier_channel).unwrap();
	verifier_channel.finish();

	// Check verifier output is consistent with prover output
	assert_eq!(prove_output, verify_output);
}

// A witness whose constraint count is not a power of two proves exactly as the same witness
// zero-extended up to the constraint axis: byte-identical transcript, identical output claims, and
// the proof still verifies against the axis's variable count.
//
// This is the guarantee that lets the M4 prover stop materializing the padding rows (BINIUS-388)
// without moving anything on the wire. Byte-identity is the whole story here, and it is exactly
// what makes the word-level zero-extension the safe design: the padded side is the pre-existing,
// tested path, so the two sides agreeing byte for byte leaves no room for the reduction's phases to
// have been given the wrong padding value.
#[test]
fn unpadded_columns_prove_identically_to_padded_ones() {
	let mut rng = StdRng::seed_from_u64(1);

	// Constraint counts short of, just past, and well past a power of two.
	for n in [3, 5, 33] {
		let (a, b, c_lo, c_hi) = random_products(&mut rng, n);
		let n_vars = log2_ceil_usize(n);

		// The same witness with explicit zero rows up to the axis. `0 * 0 = 0 || 0`, so the padded
		// witness is equally satisfying.
		let pad = |words: &[Word]| {
			let mut padded = words.to_vec();
			padded.resize(1 << n_vars, Word::ZERO);
			padded
		};
		let (a_pad, b_pad, c_lo_pad, c_hi_pad) = (pad(&a), pad(&b), pad(&c_lo), pad(&c_hi));

		// One proof per shape, each on a fresh transcript over the same challenger seed.
		let run = |columns: [&[Word]; 4]| {
			let oracle_specs = [OracleSpec::new(LIMB_BITS)];
			let mut transcript = ProverTranscript::<StdChallenger>::default();
			let mut channel =
				NaiveProverChannel::<F, _>::new(&mut transcript, oracle_specs.to_vec());
			let output = prove::<_, F, P, _>(columns, &mut channel, &GlobalAllocator)
				.expect("the four columns have equal length");
			channel.finish();
			(output, transcript.finalize())
		};
		let (unpadded_output, unpadded_proof) = run([&a, &b, &c_lo, &c_hi]);
		let (padded_output, padded_proof) = run([&a_pad, &b_pad, &c_lo_pad, &c_hi_pad]);

		assert_eq!(unpadded_output, padded_output, "claims differ at n = {n}");
		assert_eq!(unpadded_proof, padded_proof, "transcript differs at n = {n}");

		let oracle_specs = [OracleSpec::new(LIMB_BITS)];
		let mut verifier_transcript =
			VerifierTranscript::new(StdChallenger::default(), unpadded_proof);
		let mut verifier_channel =
			NaiveVerifierChannel::new(&mut verifier_transcript, &oracle_specs);
		let verify_output = verify(n_vars, &mut verifier_channel).unwrap();
		verifier_channel.finish();

		assert_eq!(unpadded_output, verify_output, "verifier disagrees at n = {n}");
	}
}
