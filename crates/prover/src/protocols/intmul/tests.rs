// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use binius_core::word::Word;
use binius_field::{BinaryField128bGhash, PackedBinaryGhash2x128b, Random};
use binius_iop::{channel::OracleSpec, naive_channel::NaiveVerifierChannel};
use binius_iop_prover::naive_channel::NaiveProverChannel;
use binius_math::{inner_product::inner_product_buffers, multilinear::eq::eq_ind_partial_eval};
use binius_transcript::ProverTranscript;
use binius_verifier::{
	config::{LOG_WORD_SIZE_BITS, StdChallenger},
	protocols::intmul::{
		common::{IntMulOutput, LIMB_BITS},
		verify,
	},
};
use itertools::izip;
use rand::prelude::*;

use super::{prove::IntMulProver, witness::Witness};
use crate::fold_word::fold_words;

type F = BinaryField128bGhash;
type P = PackedBinaryGhash2x128b;

pub fn evaluate_witness(words: &[Word], eval_point: &[F]) -> F {
	let (prefix, suffix) = eval_point.split_at(LOG_WORD_SIZE_BITS);
	let prefix_tensor = eq_ind_partial_eval::<F>(prefix);
	let suffix_tensor = eq_ind_partial_eval::<F>(suffix);

	let partially_folded_witness = fold_words::<_, F>(words, prefix_tensor.as_ref());

	inner_product_buffers(&partially_folded_witness, &suffix_tensor)
}

#[test]
fn prove_and_verify() {
	let mut rng = StdRng::seed_from_u64(0);

	const LOG_EXPONENTS: usize = 5;
	const NUM_EXPONENTS: usize = 1 << LOG_EXPONENTS;
	let mut a = Vec::with_capacity(NUM_EXPONENTS);
	let mut b = Vec::with_capacity(NUM_EXPONENTS);
	let mut c_lo = Vec::with_capacity(NUM_EXPONENTS);
	let mut c_hi = Vec::with_capacity(NUM_EXPONENTS);

	for _ in 0..NUM_EXPONENTS {
		let a_i = rng.random_range(1..u64::MAX);
		let b_i = rng.random_range(1..u64::MAX);

		let full_result = (a_i as u128) * (b_i as u128);

		let c_lo_i = full_result as u64;
		let c_hi_i = (full_result >> 64) as u64;

		a.push(Word::from_u64(a_i));
		b.push(Word::from_u64(b_i));
		c_lo.push(Word::from_u64(c_lo_i));
		c_hi.push(Word::from_u64(c_hi_i));
	}

	let witness = Witness::<P>::new(&a, &b, &c_lo, &c_hi).unwrap();

	// The one oracle in the protocol is the logup* pushforward, over the table variables.
	let oracle_specs = [OracleSpec::new(LIMB_BITS)];

	// Run prover
	let mut prover_transcript = ProverTranscript::<StdChallenger>::default();
	let mut prover_channel =
		NaiveProverChannel::<F, _>::new(&mut prover_transcript, oracle_specs.to_vec());
	let mut prover = IntMulProver::new(0, &mut prover_channel);
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
	let z_challenge: Vec<F> = (0..LOG_WORD_SIZE_BITS)
		.map(|_| F::random(&mut rng))
		.collect();
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
