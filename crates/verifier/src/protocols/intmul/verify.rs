// Copyright 2025 Irreducible Inc.

use binius_field::{BinaryField, Field};
use binius_math::{multilinear::eq::eq_ind, univariate::evaluate_univariate};
use binius_transcript::{
	VerifierTranscript,
	fiat_shamir::{CanSample, Challenger},
};
use itertools::{Itertools, izip};

use super::{
	common::{
		IntMulOutput, Phase1Output, Phase2Output, Phase3Output, Phase4Output, Phase5Output,
		frobenius_twist, make_phase_3_output, normalize_a_c_exponent_evals,
	},
	error::Error,
};
use crate::protocols::sumcheck::{BatchSumcheckOutput, batch_verify};

fn read_scalar_slice<F: Field, C: Challenger>(
	transcript: &mut VerifierTranscript<C>,
	len: usize,
) -> Result<Vec<F>, Error> {
	Ok(transcript.message().read_scalar_slice::<F>(len)?)
}

struct BivariateProductMleLayerOutput<F: Field> {
	challenges: Vec<F>,
	multilinear_evals: Vec<F>,
}

fn verify_multi_bivariate_product_mle_layer<F: Field, C: Challenger>(
	eval_point: &[F],
	evals: &[F],
	transcript: &mut VerifierTranscript<C>,
) -> Result<BivariateProductMleLayerOutput<F>, Error> {
	let n_vars = eval_point.len();

	let BatchSumcheckOutput {
		batch_coeff,
		mut challenges,
		eval,
	} = batch_verify(n_vars, 3, evals, transcript)?;

	challenges.reverse();

	let multilinear_evals = read_scalar_slice(transcript, 2 * evals.len())?;

	let eq_ind_eval = eq_ind(eval_point, &challenges);
	let expected_unbatched_terms = multilinear_evals
		.iter()
		.tuples()
		.map(|(&left, &right)| eq_ind_eval * left * right)
		.collect::<Vec<_>>();

	let expected_eval = evaluate_univariate(&expected_unbatched_terms, batch_coeff);
	if expected_eval != eval {
		return Err(Error::CompositionClaimMismatch);
	}

	Ok(BivariateProductMleLayerOutput {
		challenges,
		multilinear_evals,
	})
}

fn verify_phase_1<F: Field, C: Challenger>(
	log_bits: usize,
	initial_eval_point: &[F],
	initial_b_eval: F,
	transcript: &mut VerifierTranscript<C>,
) -> Result<Phase1Output<F>, Error> {
	let mut eval_point = initial_eval_point.to_vec();
	let mut evals = vec![initial_b_eval];

	for depth in 0..log_bits {
		assert_eq!(evals.len(), 1 << depth);

		let BivariateProductMleLayerOutput {
			challenges,
			multilinear_evals,
		} = verify_multi_bivariate_product_mle_layer::<F, C>(&eval_point, &evals, transcript)?;

		eval_point = challenges;
		evals = multilinear_evals;
	}

	assert_eq!(evals.len(), 1 << log_bits);

	Ok(Phase1Output {
		eval_point,
		b_leaves_evals: evals,
	})
}

// PHASE 2: frobenius

// PHASE THREE: selector sumcheck

fn verify_phase_3<F: Field, C: Challenger>(
	log_bits: usize,
	// selector sumcheck stuff
	phase_2_output: Phase2Output<F>,
	// c sumcheck stuff
	c_eval_point: &[F],
	c_eval: F,
	transcript: &mut VerifierTranscript<C>,
) -> Result<Phase3Output<F>, Error> {
	let n_vars = c_eval_point.len();

	let Phase2Output { twisted_claims } = phase_2_output;
	assert_eq!(twisted_claims.len(), 1 << log_bits);

	let (twisted_eval_points, twisted_evals) =
		twisted_claims.into_iter().unzip::<_, _, Vec<_>, Vec<_>>();

	for twisted_eval_point in &twisted_eval_points {
		assert_eq!(twisted_eval_point.len(), c_eval_point.len());
	}

	let mut evals = Vec::with_capacity(twisted_evals.len());
	evals.extend(twisted_evals);
	evals.push(c_eval);

	let BatchSumcheckOutput {
		batch_coeff,
		mut challenges,
		eval,
	} = batch_verify(n_vars, 3, &evals, transcript)?;
	challenges.reverse();

	let selector_prover_evals = read_scalar_slice::<F, _>(transcript, (1 << log_bits) + 1)?;
	let c_root_prover_evals = read_scalar_slice::<F, _>(transcript, 2)?;

	let output =
		make_phase_3_output(log_bits, &challenges, &selector_prover_evals, &c_root_prover_evals);
	let Phase3Output {
		eval_point,
		b_exponent_evals,
		selector_eval,
		c_lo_root_eval,
		c_hi_root_eval,
	} = &output;

	let mut expected_unbatched_terms = Vec::with_capacity((1 << log_bits) + 1);

	for (twisted_eval_point, b_exponent_eval) in izip!(twisted_eval_points, b_exponent_evals) {
		let twisted_eq_eval = eq_ind(&twisted_eval_point, eval_point);
		let expected = twisted_eq_eval * (*b_exponent_eval * (*selector_eval - F::ONE) + F::ONE);
		expected_unbatched_terms.push(expected);
	}

	let c_eq_eval = eq_ind(c_eval_point, eval_point);
	expected_unbatched_terms.extend([c_eq_eval * c_lo_root_eval * c_hi_root_eval]);

	let expected_batched_eval = evaluate_univariate(&expected_unbatched_terms, batch_coeff);

	if expected_batched_eval != eval {
		return Err(Error::CompositionClaimMismatch);
	}

	Ok(output)
}

// PHASE 4: all but last layer of a_layers and c_layers

fn verify_phase_4<F: Field, C: Challenger>(
	log_bits: usize,
	eval_point: &[F],
	a_root_eval: F,
	c_lo_root_eval: F,
	c_hi_root_eval: F,
	transcript: &mut VerifierTranscript<C>,
) -> Result<Phase4Output<F>, Error> {
	assert!(log_bits >= 1);

	let mut eval_point = eval_point.to_vec();
	let mut evals = vec![a_root_eval, c_lo_root_eval, c_hi_root_eval];

	for depth in 0..log_bits - 1 {
		assert_eq!(evals.len(), 3 << depth);

		let BivariateProductMleLayerOutput {
			challenges,
			multilinear_evals,
		} = verify_multi_bivariate_product_mle_layer(&eval_point, &evals, transcript)?;

		eval_point = challenges;
		evals = multilinear_evals;
	}

	assert_eq!(evals.len(), 3 << (log_bits - 1));
	let c_hi_evals = evals.split_off(2 << (log_bits - 1));
	let c_lo_evals = evals.split_off(1 << (log_bits - 1));
	let a_evals = evals;

	Ok(Phase4Output {
		eval_point,
		a_evals,
		c_lo_evals,
		c_hi_evals,
	})
}

// PHASE 5: final layer

#[allow(clippy::too_many_arguments)]
fn verify_phase_5<F: Field, C: Challenger>(
	log_bits: usize,
	// a and c stuff
	a_c_eval_point: &[F],
	a_evals: &[F],
	c_lo_evals: &[F],
	c_hi_evals: &[F],
	// b stuff
	b_eval_point: &[F],
	b_exponent_evals: &[F],
	transcript: &mut VerifierTranscript<C>,
) -> Result<Phase5Output<F>, Error> {
	assert!(log_bits >= 1);
	assert_eq!(2 * a_evals.len(), 1 << log_bits);
	assert_eq!(2 * c_lo_evals.len(), 1 << log_bits);
	assert_eq!(2 * c_hi_evals.len(), 1 << log_bits);

	let n_vars = a_c_eval_point.len();
	assert_eq!(b_eval_point.len(), n_vars);

	// This is the eval of `a_0 * b_0` and `c_lo_0`.
	let overflow_zerocheck_eval = transcript.message().read_scalar::<F>()?;

	let evals = [
		a_evals,
		c_lo_evals,
		c_hi_evals,
		// For `a_0 * b_0 `bivariate product.
		&[overflow_zerocheck_eval],
		// For `c_lo_0` rerand sumcheck.
		&[overflow_zerocheck_eval],
		b_exponent_evals,
	]
	.concat();

	let BatchSumcheckOutput {
		batch_coeff,
		mut challenges,
		eval,
	} = batch_verify(n_vars, 3, &evals, transcript)?;
	challenges.reverse();

	// Read the evals of all multilinears in the bivariate prouct sumcheck: 64 for `a`, 128 for `c`,
	// 2 for `a_0` and `b_0`
	let mut bivariate_evals = read_scalar_slice(transcript, 64 + 128 + 2)?;
	// Read the single eval of the `c_lo_0` rerand sumcheck
	let c_lo_0_eval = transcript.message().read_scalar::<F>()?;
	// Read the 64 evals of the `b` rerand sumcheck
	let b_exponent_evals = read_scalar_slice(transcript, 64)?;

	// Compose the expected evaluation of the batched composition via
	// the prover's claimed multilinear evals extracted above.
	// For every pair (p,q) of multilinears, the verifier can be sure that
	// the MLE of p*q at `a_c_eq_eval` equals the corresponding eval in `evals`.
	// The last of these pairs implies the MLE of `a_0 * b_0` at `a_c_eq_eval` equals
	// `overflow_zerocheck_eval`.
	let a_c_eq_eval = eq_ind(a_c_eval_point, &challenges);
	let expected_bivariate_unbatched_evals = bivariate_evals
		.iter()
		.tuples()
		.map(|(left, right)| a_c_eq_eval * left * right)
		.collect::<Vec<F>>();

	// Likewise, the verifier can be sure that the MLE of `c_lo_0` at `a_c_eq_eval`
	// equals `overflow_zerocheck_eval`. Combined with the MLE of `a_0 * b_0` at `a_c_eq_eval`
	// being `overflow_zerocheck_eval`, the verifier can conclude the
	// MLE of `a_0 * b_0 - c_lo_0` at `a_c_eq_eval` equals zero. By the Schwartz-Zippel lemma,
	// the verifier concludes `a_0_i * b_0_i - c_lo_0_i = 0` for all rows `i`.
	let expected_c_lo_0_rerand_unbatched_eval = a_c_eq_eval * c_lo_0_eval;

	let b_eq_eval = eq_ind(b_eval_point, &challenges);
	let expected_b_rerand_unbatched_evals = b_exponent_evals
		.iter()
		.map(|&b_exponent_eval| b_eq_eval * b_exponent_eval)
		.collect::<Vec<F>>();

	let expected_unbatched_evals = [
		expected_bivariate_unbatched_evals,
		vec![expected_c_lo_0_rerand_unbatched_eval],
		expected_b_rerand_unbatched_evals,
	]
	.concat();
	let expected_batched_eval = evaluate_univariate(&expected_unbatched_evals, batch_coeff);

	// Compare expected evaluation against given evaluation `eval`.
	if expected_batched_eval != eval {
		return Err(Error::CompositionClaimMismatch);
	}

	// Evals `b_0_eval`, `a_0_eval`, and `c_lo_0_eval` will be verified following phase 5.
	let b_0_eval = bivariate_evals
		.pop()
		.expect("non-empty scaled a_c exponent evals");
	let a_0_eval = bivariate_evals
		.pop()
		.expect("non-empty scaled a_c exponent evals");

	Ok(Phase5Output {
		eval_point: challenges,
		scaled_a_c_exponent_evals: bivariate_evals,
		b_exponent_evals,
		a_0_eval,
		b_0_eval,
		c_lo_0_eval,
	})
}

/// This method verifies an integer multiplication reduction to obtain evaluation claims on 1-bit
/// multilinears. Verification consists of five phases:
///  - Phase 1: GKR tree roots for B & C are evaluated at a sampled point, after which reductions
///    are performed to obtain evaluation claims on $(b * (G^{a_i} - 1) + 1)^{2^i}$
///  - Phase 2: Frobenius twist is applied to obtain claims on $b * (G^{a_i} - 1) + 1$
///  - Phase 3: Two batched sumchecks:
///    - Selector mlecheck to reduce claims on $b * (G^{a_i} - 1) + 1$ to claims on $G^{a_i}$ and
///      $b$
///    - First layer of GPA reduction for the `c_lo || c_hi` combined `c` tree
///  - Phase 4: Batching all but last layers and `a`, `c_lo` and `c_hi`
///  - Phase 5: Verifying the last (widest) layers of `a`, `c_lo` and `c_hi` batched with
///    rerandomization degree-1 mlecheck on `b` evaluations from phase 3. We must also verify that
///    `a_0 * b_0 = c_lo_0` across all rows (where these are the least significant bits of the
///    respective values). This prevents an attack when `a*b = 0`: a malicious prover could set `c =
///    2^128 - 1`, which satisfies `a*b ≡ c (mod 2^128-1)` since `0 ≡ 2^128-1 (mod 2^128-1)`, but we
///    need `a*b = c (mod 2^128)`. The check catches this because if `c = 2^128-1` then `c_lo_0 = 1`
///    (odd), but `a_0 * b_0 = 0` when `a=0` or `b=0`.
pub fn verify<F: BinaryField, C: Challenger>(
	log_bits: usize,
	n_vars: usize,
	transcript: &mut VerifierTranscript<C>,
) -> Result<IntMulOutput<F>, Error> {
	assert!(log_bits >= 1);
	let initial_eval_point: Vec<F> = transcript.sample_vec(n_vars);

	let mut reader = transcript.message();
	let initial_b_eval: F = reader.read_scalar::<F>()?;
	let initial_c_eval: F = reader.read_scalar::<F>()?;

	// Phase 1
	let Phase1Output {
		eval_point: phase_1_eval_point,
		b_leaves_evals,
	} = verify_phase_1(log_bits, &initial_eval_point, initial_b_eval, transcript)?;

	assert_eq!(phase_1_eval_point.len(), n_vars);
	assert_eq!(b_leaves_evals.len(), 1 << log_bits);

	// Phase 2
	let phase2_output = frobenius_twist(log_bits, &phase_1_eval_point, &b_leaves_evals);

	// Phase 3
	let Phase3Output {
		eval_point: phase_3_eval_point,
		b_exponent_evals,
		selector_eval,
		c_lo_root_eval,
		c_hi_root_eval,
	} = verify_phase_3(log_bits, phase2_output, &initial_eval_point, initial_c_eval, transcript)?;

	// Phase 4
	let Phase4Output {
		eval_point: phase_4_eval_point,
		a_evals,
		c_lo_evals,
		c_hi_evals,
	} = verify_phase_4(
		log_bits,
		&phase_3_eval_point,
		selector_eval,
		c_lo_root_eval,
		c_hi_root_eval,
		transcript,
	)?;

	// Phase 5
	let Phase5Output {
		eval_point: phase_5_eval_point,
		scaled_a_c_exponent_evals,
		b_exponent_evals,
		a_0_eval,
		b_0_eval,
		c_lo_0_eval,
	} = verify_phase_5(
		log_bits,
		&phase_4_eval_point,
		&a_evals,
		&c_lo_evals,
		&c_hi_evals,
		&phase_3_eval_point,
		&b_exponent_evals,
		transcript,
	)?;

	let [a_exponent_evals, c_lo_exponent_evals, c_hi_exponent_evals] =
		normalize_a_c_exponent_evals(log_bits, scaled_a_c_exponent_evals);

	assert_eq!(a_exponent_evals[0], a_0_eval);
	assert_eq!(b_exponent_evals[0], b_0_eval);
	assert_eq!(c_lo_exponent_evals[0], c_lo_0_eval);

	Ok(IntMulOutput {
		eval_point: phase_5_eval_point,
		a_evals: a_exponent_evals,
		b_evals: b_exponent_evals,
		c_lo_evals: c_lo_exponent_evals,
		c_hi_evals: c_hi_exponent_evals,
	})
}
