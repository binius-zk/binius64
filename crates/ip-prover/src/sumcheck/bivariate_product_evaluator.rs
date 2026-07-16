// Copyright 2026 The Binius Developers

use binius_field::{Field, PackedField, WideMul};
use binius_ip::sumcheck::RoundCoeffs;
use binius_math::FieldBuffer;
use itertools::izip;

use super::{
	mle_store::{ColId, EvaluationChunk, MleStore},
	round_evals::WideRoundEvals2,
	round_evaluator::{RoundEvaluator, SharedSumcheckProver},
	round_state::RoundState,
};

/// Sumcheck round evaluator for a composite defined as the product of two store columns.
///
/// This is the store-backed counterpart of the bivariate product sumcheck prover: it proves the
/// plain (non-eq-weighted) sum claim of the product over the hypercube, emitting regular
/// sumcheck round polynomials.
pub struct BivariateProductEvaluator<P: PackedField> {
	cols: [ColId; 2],
	// State machine storage: last round's sum (interpolate input) or coeffs (fold input).
	last_coeffs_or_sum: RoundState<RoundCoeffs<P::Scalar>, P::Scalar>,
}

impl<F: Field, P: PackedField<Scalar = F>> BivariateProductEvaluator<P> {
	/// Creates an evaluator for the claimed sum of the product of two store columns.
	pub const fn new(cols: [ColId; 2], sum: F) -> Self {
		Self {
			cols,
			last_coeffs_or_sum: RoundState::Claim(sum),
		}
	}
}

/// Builds a [`SumcheckProver`](crate::sumcheck::common::SumcheckProver) for the plain hypercube sum
/// of the product of two multilinears.
///
/// This is the store-backed replacement for the former bespoke bivariate-product prover: it loads
/// the two columns into a fresh [`MleStore`] and drives a single [`BivariateProductEvaluator`]. The
/// multilinears must have the same number of variables; the returned prover's `finish` emits their
/// two evaluations in the given order.
pub fn bivariate_product_prover<F: Field, P: PackedField<Scalar = F>>(
	multilinears: [FieldBuffer<P>; 2],
	sum: F,
) -> SharedSumcheckProver<'static, P, BivariateProductEvaluator<P>> {
	assert_eq!(
		multilinears[0].log_len(),
		multilinears[1].log_len(),
		"multilinears must have equal number of variables"
	);

	let mut store = MleStore::new(multilinears[0].log_len());
	let cols = multilinears.map(|col| store.push_owned(col));
	SharedSumcheckProver::new(store, vec![BivariateProductEvaluator::new(cols, sum)])
}

impl<F: Field, P: PackedField<Scalar = F>> RoundEvaluator<F, P> for BivariateProductEvaluator<P> {
	fn degree(&self) -> usize {
		// Product of two multilinears: two sampled evaluations, `y_1` and `y_inf`.
		2
	}

	fn round_claim(&self, _store: &MleStore<'_, P>) -> F {
		// A plain product claim carries no eq factor, so the round claim needs no point
		// coordinates.
		match &self.last_coeffs_or_sum {
			RoundState::Claim(sum) => *sum,
			RoundState::Coeffs(coeffs) => coeffs.sum_over_endpoints(),
		}
	}

	fn accumulate(&self, chunk: &EvaluationChunk<'_, P>, accum: &mut [<P as WideMul>::Output]) {
		let a = chunk.col(self.cols[0]);
		let b = chunk.col(self.cols[1]);

		// Accumulate F(1) and F(∞) where F = ∑_{v ∈ B} A(v || X) B(v || X). The low half is the
		// v-prefix at x=0, the high half at x=1.
		//
		// The per-point products are accumulated in unreduced (wide) form and reduced a single
		// time in interpolate, amortizing the GF(2^128) reduction over the whole sum.
		let mut evals = WideRoundEvals2::<<P as WideMul>::Output>::default();
		for (&a_0_i, &a_1_i, &b_0_i, &b_1_i) in
			izip!(a.lo.as_ref(), a.hi.as_ref(), b.lo.as_ref(), b.hi.as_ref())
		{
			// Evaluate M(∞) = M(0) + M(1)
			let a_inf_i = a_0_i + a_1_i;
			let b_inf_i = b_0_i + b_1_i;

			evals += WideRoundEvals2 {
				y_1: P::wide_mul(a_1_i, b_1_i),
				y_inf: P::wide_mul(a_inf_i, b_inf_i),
			};
		}

		// The evaluator's single-claim run holds `y_1` in slot 0 and `y_inf` in slot 1.
		accum[0] += evals.y_1;
		accum[1] += evals.y_inf;
	}

	fn interpolate(
		&mut self,
		store: &MleStore<'_, P>,
		accum: &[<P as WideMul>::Output],
	) -> RoundCoeffs<F> {
		let last_sum = *self.last_coeffs_or_sum.claim();

		// The store has not yet folded this round, so its remaining-variable count is this round's.
		let n_vars_remaining = store.n_vars();
		assert!(n_vars_remaining > 0);

		let evals = WideRoundEvals2 {
			y_1: accum[0].clone(),
			y_inf: accum[1].clone(),
		};
		let round_coeffs = evals
			.reduce::<P>()
			.sum_scalars(n_vars_remaining)
			.interpolate(last_sum);

		self.last_coeffs_or_sum = RoundState::Coeffs(round_coeffs.clone());
		round_coeffs
	}

	fn fold(&mut self, challenge: F) {
		// The store folds the columns (advancing its remaining count); only the sum claim advances
		// here.
		let round_sum = self.last_coeffs_or_sum.coeffs().evaluate(challenge);
		self.last_coeffs_or_sum = RoundState::Claim(round_sum);
	}
}

#[cfg(test)]
mod tests {
	use binius_field::arch::{OptimalB128, OptimalPackedB128};
	use binius_ip::sumcheck::verify;
	use binius_math::{
		inner_product::inner_product_par, multilinear::evaluate::evaluate,
		test_utils::random_field_buffer,
	};
	use binius_transcript::{ProverTranscript, fiat_shamir::HasherChallenger};
	use rand::prelude::*;

	use super::*;
	use crate::sumcheck::prove::prove_single;

	type StdChallenger = HasherChallenger<sha2::Sha256>;

	// Proving the product sum of two multilinears via the shared store, then verifying, recovers
	// the two multilinear evaluations at the challenge point and their product as the reduced
	// eval.
	#[test]
	fn test_bivariate_product_sumcheck() {
		type F = OptimalB128;
		type P = OptimalPackedB128;

		let n_vars = 8;
		let mut rng = StdRng::seed_from_u64(0);

		let multilinear_a = random_field_buffer::<P>(&mut rng, n_vars);
		let multilinear_b = random_field_buffer::<P>(&mut rng, n_vars);
		let expected_sum = inner_product_par(&multilinear_a, &multilinear_b);

		let prover =
			bivariate_product_prover([multilinear_a.clone(), multilinear_b.clone()], expected_sum);

		let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
		let output = prove_single(prover, &mut prover_transcript);
		prover_transcript
			.message()
			.write_slice(&output.multilinear_evals);

		let mut verifier_transcript = prover_transcript.into_verifier();
		let sumcheck_output = verify(n_vars, 2, expected_sum, &mut verifier_transcript).unwrap();
		let multilinear_evals: Vec<F> = verifier_transcript.message().read_vec(2).unwrap();

		assert_eq!(
			multilinear_evals[0] * multilinear_evals[1],
			sumcheck_output.eval,
			"product of the multilinear evaluations should equal the reduced evaluation"
		);

		// The prover binds variables high-to-low; `evaluate` expects them low-to-high.
		let mut eval_point = sumcheck_output.challenges.clone();
		eval_point.reverse();
		assert_eq!(evaluate(&multilinear_a, &eval_point), multilinear_evals[0]);
		assert_eq!(evaluate(&multilinear_b, &eval_point), multilinear_evals[1]);
		assert_eq!(output.challenges, sumcheck_output.challenges);
	}
}
