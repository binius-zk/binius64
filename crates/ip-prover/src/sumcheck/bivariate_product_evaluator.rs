// Copyright 2026 The Binius Developers

use binius_field::{Field, PackedField, WideMul};
use binius_ip::sumcheck::RoundCoeffs;
use binius_math::FieldSlice;
use itertools::izip;

use super::{
	mle_store::{ColId, MleStore},
	round_evals::WideRoundEvals2,
	round_evaluator::{RoundContext, RoundEvaluator},
};

/// Sumcheck round evaluator for a composite defined as the product of two store columns.
///
/// This is the store-backed counterpart of the bivariate product sumcheck prover: it proves the
/// plain (non-eq-weighted) sum claim of the product over the hypercube, emitting regular
/// sumcheck round polynomials.
pub struct BivariateProductEvaluator<P: PackedField> {
	cols: [ColId; 2],
	// State machine storage: last round's sum (interpolate input) or coeffs (fold input).
	last_coeffs_or_sum: RoundCoeffsOrSum<P::Scalar>,
}

impl<F: Field, P: PackedField<Scalar = F>> BivariateProductEvaluator<P> {
	/// Creates an evaluator for the claimed sum of the product of two store columns.
	pub const fn new(cols: [ColId; 2], sum: F) -> Self {
		Self {
			cols,
			last_coeffs_or_sum: RoundCoeffsOrSum::Sum(sum),
		}
	}
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
			RoundCoeffsOrSum::Sum(sum) => *sum,
			RoundCoeffsOrSum::Coeffs(coeffs) => coeffs.sum_over_endpoints(),
		}
	}

	fn accumulate(
		&self,
		ctx: &RoundContext<'_, '_, P>,
		chunk_index: usize,
		accum: &mut [<P as WideMul>::Output],
	) {
		let chunk_vars = ctx.chunk_vars();

		let [a, b]: [FieldSlice<'_, P>; 2] = self.cols.map(|id| ctx.col(id));
		let (a_0, a_1) = a.split_half_ref();
		let (b_0, b_1) = b.split_half_ref();
		let a_0 = a_0.chunk(chunk_vars, chunk_index);
		let a_1 = a_1.chunk(chunk_vars, chunk_index);
		let b_0 = b_0.chunk(chunk_vars, chunk_index);
		let b_1 = b_1.chunk(chunk_vars, chunk_index);

		// Accumulate F(1) and F(∞) where F = ∑_{v ∈ B} A(v || X) B(v || X).
		//
		// The per-point products are accumulated in unreduced (wide) form and reduced a single
		// time in interpolate, amortizing the GF(2^128) reduction over the whole sum.
		let mut evals = WideRoundEvals2::<<P as WideMul>::Output>::default();
		for (&a_0_i, &a_1_i, &b_0_i, &b_1_i) in
			izip!(a_0.as_ref(), a_1.as_ref(), b_0.as_ref(), b_1.as_ref())
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
		let RoundCoeffsOrSum::Sum(last_sum) = self.last_coeffs_or_sum else {
			panic!("interpolate called out of order; expected fold");
		};

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

		self.last_coeffs_or_sum = RoundCoeffsOrSum::Coeffs(round_coeffs.clone());
		round_coeffs
	}

	fn fold(&mut self, challenge: F) {
		let RoundCoeffsOrSum::Coeffs(coeffs) = &self.last_coeffs_or_sum else {
			panic!("fold called out of order; expected interpolate");
		};

		// The store folds the columns (advancing its remaining count); only the sum claim advances
		// here.
		let round_sum = coeffs.evaluate(challenge);
		self.last_coeffs_or_sum = RoundCoeffsOrSum::Sum(round_sum);
	}
}

#[derive(Debug, Clone)]
enum RoundCoeffsOrSum<F: Field> {
	Coeffs(RoundCoeffs<F>),
	Sum(F),
}
