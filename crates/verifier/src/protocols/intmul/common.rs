// Copyright 2025 Irreducible Inc.

use binius_field::{BinaryField, field::FieldOps};
use itertools::{iterate, izip};

#[derive(Debug, Clone, PartialEq)]
pub struct IntMulOutput<F> {
	pub eval_point: Vec<F>,
	pub a_evals: Vec<F>,
	pub b_evals: Vec<F>,
	pub c_lo_evals: Vec<F>,
	pub c_hi_evals: Vec<F>,
}

pub struct Phase1Output<F> {
	pub eval_point: Vec<F>,
	pub b_leaves_evals: Vec<F>,
}

pub struct Phase2Output<F> {
	pub twisted_claims: Vec<(Vec<F>, F)>,
}

#[derive(Debug, Clone)]
pub struct Phase3Output<F> {
	pub eval_point: Vec<F>,
	pub b_exponent_evals: Vec<F>,
	pub selector_eval: F,
	pub c_lo_root_eval: F,
	pub c_hi_root_eval: F,
}

pub fn make_phase_3_output<F: FieldOps>(
	log_bits: usize,
	eval_point: &[F],
	selector_prover_evals: &[F],
	c_root_prover_evals: Vec<F>,
) -> Phase3Output<F> {
	assert_eq!(selector_prover_evals.len(), 1 + (1 << log_bits));
	let (selector_eval, b_exponent_evals) = selector_prover_evals
		.split_last()
		.expect("non-empty selector sumcheck output");

	let Ok([c_lo_root_eval, c_hi_root_eval]) = TryInto::<[F; 2]>::try_into(c_root_prover_evals) else {
		unreachable!("expect two multilinears in the c_root prover in phase 3")
	};

	Phase3Output {
		eval_point: eval_point.to_vec(),
		b_exponent_evals: b_exponent_evals.to_vec(),
		selector_eval: selector_eval.clone(),
		c_lo_root_eval,
		c_hi_root_eval,
	}
}

pub struct Phase4Output<F> {
	pub eval_point: Vec<F>,
	pub a_evals: Vec<F>,
	pub c_lo_evals: Vec<F>,
	pub c_hi_evals: Vec<F>,
}

pub struct Phase5Output<F> {
	pub eval_point: Vec<F>,
	pub scaled_a_c_exponent_evals: Vec<F>,
	pub b_exponent_evals: Vec<F>,
	pub a_0_eval: F,
	pub b_0_eval: F,
	pub c_lo_0_eval: F,
}

/// Applying the inverse of $\phi$ to the selector columns.
pub fn frobenius_twist<F: FieldOps>(
	log_bits: usize,
	degree: usize,
	eval_point: &[F],
	evals: &[F],
) -> Phase2Output<F> {
	let inv_phi = |arg: F, i: usize| {
		iterate(arg, |g| g.clone().square())
			.nth(degree - i)
			.expect("infinite iterator")
	};

	assert_eq!(evals.len(), 1 << log_bits);
	let twisted_claims = evals
		.iter()
		.enumerate()
		.map(|(i, eval)| {
			let twisted_eval = inv_phi(eval.clone(), i);
			let twisted_eval_point = eval_point
				.iter()
				.map(|coord| inv_phi(coord.clone(), i))
				.collect();
			(twisted_eval_point, twisted_eval)
		})
		.collect();

	Phase2Output { twisted_claims }
}

pub fn normalize_a_c_exponent_evals<F, E>(log_bits: usize, evals: Vec<E>) -> [Vec<E>; 3]
where
	F: BinaryField,
	E: FieldOps<Scalar = F> + From<F>,
{
	assert_eq!(evals.len(), 3 << log_bits);

	// for i in 0..1 << log_bits: evals[i] = (1-EvalMLE_i)*1 + EvalMLE_i*g^{2^i} =
	// EvalMLE_i*(g^{2^i}-1) + 1 where EvalMLE_i is the evaluation of the multilinear extension of
	// bit i of the exponents of `a` (the point of evaluation is irrelevant in this function)
	// we can then compute desired evaluation EvalMLE_i as (evals[i] - 1) / (g^{2^i}-1)
	// similarly for `c` for evals[1 << log_bits..3 << log_bits] and i in 0..2 << log_bits

	let mut a_scaled_evals = evals;
	let mut c_lo_scaled_evals = a_scaled_evals.split_off(1 << log_bits);
	let mut c_hi_scaled_evals = c_lo_scaled_evals.split_off(1 << log_bits);

	// Compute the normalization factors (conjugate - 1)^{-1} in F, then convert to E.
	let inv_factors: Vec<E> = iterate(F::MULTIPLICATIVE_GENERATOR, |g| g.square())
		.take(2 << log_bits)
		.map(|conjugate| E::from((conjugate - F::ONE).invert().expect("non-zero")))
		.collect();

	let (lo_inv_factors, hi_inv_factors) = inv_factors.split_at(1 << log_bits);

	fn normalize<E: FieldOps>(eval: &mut E, inv_factor: &E) {
		*eval -= E::one();
		*eval *= inv_factor.clone();
	}

	for (inv_factor, a_eval, c_lo_eval) in
		izip!(lo_inv_factors, &mut a_scaled_evals, &mut c_lo_scaled_evals)
	{
		normalize(a_eval, inv_factor);
		normalize(c_lo_eval, inv_factor);
	}

	for (inv_factor, c_hi_eval) in izip!(hi_inv_factors, &mut c_hi_scaled_evals) {
		normalize(c_hi_eval, inv_factor);
	}

	[a_scaled_evals, c_lo_scaled_evals, c_hi_scaled_evals]
}
