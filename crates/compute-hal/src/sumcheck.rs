// Copyright 2026 The Binius Developers

//! The mini-prover: a sumcheck driver written once, generic over [`Compute`].
//!
//! Rounds bind the highest variable first, matching `binius_ip::sumcheck::verify`.

use std::{iter, ops::Mul};

use binius_field::{BinaryField, BinaryField1b, ExtensionField, FieldOps};
use binius_ip::sumcheck::RoundCoeffs;
use binius_transcript::{ProverTranscript, fiat_shamir::Challenger};

use crate::{Composition, Compute, PairKernel};

/// Evaluation points of the round polynomial a [`RoundKernel`] reports: `1, b_1, ..., b_{d-1}`.
///
/// The leading coefficient at infinity is not used because a composition need not be homogeneous.
fn eval_points<F: BinaryField>(degree: usize) -> Vec<F> {
	iter::once(F::ONE)
		.chain((1..degree).map(<F as ExtensionField<BinaryField1b>>::basis))
		.collect()
}

/// Turns a [`Composition`] into a [`PairKernel`] that evaluates it on the line through `lo` and
/// `hi` at the points of [`eval_points`].
pub struct RoundKernel<'c, C>(pub &'c C);

impl<F: BinaryField, C: Composition<F>> PairKernel<F> for RoundKernel<'_, C> {
	const N_IN: usize = C::N_IN;
	const N_OUT: usize = C::DEGREE;

	fn map<T: FieldOps<Scalar = F> + Mul<F, Output = T>>(&self, lo: &[T], hi: &[T], out: &mut [T]) {
		let diff: Vec<T> = iter::zip(lo, hi).map(|(l, h)| l.clone() + h).collect();
		for (o, x) in iter::zip(out, eval_points::<F>(C::DEGREE)) {
			let line: Vec<T> = iter::zip(lo, &diff)
				.map(|(l, d)| l.clone() + d.clone() * x)
				.collect();
			*o = self.0.eval(&line);
		}
	}
}

/// Lagrange interpolation: the coefficients of the polynomial through `(xs[i], ys[i])`.
fn interpolate<F: BinaryField>(xs: &[F], ys: &[F]) -> Vec<F> {
	let mut coeffs = vec![F::ZERO; xs.len()];
	for (i, (&xi, &yi)) in iter::zip(xs, ys).enumerate() {
		// numerator = prod_{j != i} (X - x_j), denominator = prod_{j != i} (x_i - x_j)
		let (num, den) = xs.iter().enumerate().filter(|&(j, _)| j != i).fold(
			(vec![F::ONE], F::ONE),
			|(num, den), (_, &xj)| {
				let next = (0..=num.len())
					.map(|k| {
						let shifted = if k > 0 { num[k - 1] } else { F::ZERO };
						shifted + num.get(k).map_or(F::ZERO, |&c| c * xj)
					})
					.collect();
				(next, den * (xi - xj))
			},
		);
		let scale = yi * den.invert_or_zero();
		iter::zip(&mut coeffs, num).for_each(|(c, n)| *c += n * scale);
	}
	coeffs
}

/// Proves `sum_x comp(cols(x)) = claim` over `n_vars` variables.
///
/// Each round is ONE `exec`: it folds the columns with the previous challenge and evaluates the
/// next round polynomial in the same pipeline. A final `exec` folds with the last challenge and
/// downloads the column evaluations.
///
/// Returns the challenges (highest variable first) and the final evaluation of every column.
pub fn prove<F, C, Comp, Ch>(
	compute: &C,
	comp: &Comp,
	n_vars: usize,
	mut cols: Vec<C::FieldVec>,
	claim: F,
	transcript: &mut ProverTranscript<Ch>,
) -> (Vec<F>, Vec<F>)
where
	F: BinaryField,
	C: Compute<F>,
	Comp: Composition<F>,
	Ch: Challenger,
{
	assert!(n_vars > 0);
	let kernel = RoundKernel(comp);
	let xs: Vec<F> = iter::once(F::ZERO)
		.chain(eval_points::<F>(Comp::DEGREE))
		.collect();
	let mut sum = claim;
	let mut challenges = Vec::with_capacity(n_vars);

	for _ in 0..n_vars {
		let round = match challenges.last() {
			None => compute.pair_map_reduce(cols, &kernel),
			Some(&r) => compute.then(compute.fold_highest_var(cols, r), |cols| {
				compute.pair_map_reduce(cols, &kernel)
			}),
		};
		let (next_cols, evals) = compute.exec(round);
		cols = next_cols;

		// The claim fixes R(0) = sum - R(1).
		let ys: Vec<F> = iter::once(sum + evals[0]).chain(evals).collect();
		let coeffs = RoundCoeffs(interpolate(&xs, &ys));
		transcript
			.message()
			.write_scalar_slice(coeffs.clone().truncate().coeffs());
		let r = binius_transcript::fiat_shamir::CanSample::sample(transcript);
		sum = coeffs.evaluate(&r);
		challenges.push(r);
	}

	let last = *challenges.last().expect("n_vars > 0");
	let finals = compute
		.exec(compute.then(compute.fold_highest_var(cols, last), |cols| compute.to_host(cols)));
	(challenges, finals.into_iter().map(|col| col[0]).collect())
}
