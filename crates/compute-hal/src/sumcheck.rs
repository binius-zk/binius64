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
pub(crate) fn eval_points<F: BinaryField>(degree: usize) -> impl Iterator<Item = F> {
	iter::once(F::ONE).chain((1..degree).map(<F as ExtensionField<BinaryField1b>>::basis))
}

/// Turns a [`Composition`] into a [`PairKernel`] that evaluates it on the line through `lo` and
/// `hi` at the points of [`eval_points`].
pub struct RoundKernel<'c, C>(pub &'c C);

impl<F: BinaryField, C: Composition<F>> PairKernel<F> for RoundKernel<'_, C> {
	const N_IN: usize = C::N_IN;
	const N_OUT: usize = C::DEGREE;
	const SCRATCH: usize = C::N_IN;

	fn map<T: FieldOps<Scalar = F> + Mul<F, Output = T>>(
		&self,
		lo: &[T],
		hi: &[T],
		scratch: &mut [T],
		out: &mut [T],
	) {
		for (o, x) in iter::zip(out, eval_points::<F>(C::DEGREE)) {
			for (s, (l, h)) in iter::zip(&mut *scratch, iter::zip(lo, hi)) {
				*s = l.clone() + (l.clone() + h) * x;
			}
			*o = self.0.eval(scratch);
		}
	}
}

/// Lagrange interpolation: the coefficients of the polynomial through `(xs[i], ys[i])`.
pub(crate) fn interpolate<F: BinaryField>(xs: &[F], ys: &[F]) -> Vec<F> {
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
		let round = compute.fold_pair_map_reduce(&mut cols, challenges.last().copied(), &kernel);
		let evals = compute.exec(round);

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

	let finals = read_folded(compute, &mut cols, *challenges.last().expect("n_vars > 0"));
	(challenges, finals)
}

/// Folds `cols` by `r` and reads their (constant) values, in one `exec`.
fn read_folded<F: BinaryField, C: Compute<F>>(
	compute: &C,
	cols: &mut [C::FieldVec],
	r: F,
) -> Vec<F> {
	compute
		.exec(compute.then(compute.fold_highest_var(cols, r), |cols| {
			let cols = &*cols;
			let reads = cols
				.iter()
				.map(|col| compute.to_host(compute.as_slice(col)))
				.collect();
			compute.join_all(reads)
		}))
		.into_iter()
		.map(|col| col[0])
		.collect()
}

/// Proves `sum_x eq(point, x) * comp(cols(x)) = claim` as an MLE-check with Gruen's eq factoring.
///
/// `comp` takes the eq column first, then the `cols`; its `DEGREE` is the degree along a line, so
/// the (broadcast) eq column adds none. Each round computes the prime polynomial
/// `R'(X) = sum_x' eq(point_rest, x') comp(line(X, x'))` and the verifier supplies the factor
/// `eq(point_k, X)` itself. The eq tensor over the remaining variables is re-expanded from the
/// prefix of `point` each round, in the same pipeline as the fold, and passed to
/// `pair_map_reduce` as a column one variable shorter than the rest.
///
/// Returns the challenges (highest variable first) and the final evaluation of every column.
pub fn prove_mlecheck<F, C, Comp, Ch>(
	compute: &C,
	comp: &Comp,
	point: &[F],
	cols: Vec<C::FieldVec>,
	claim: F,
	transcript: &mut ProverTranscript<Ch>,
) -> (Vec<F>, Vec<F>)
where
	F: BinaryField,
	C: Compute<F>,
	Comp: Composition<F>,
	Ch: Challenger,
{
	let n_vars = point.len();
	assert!(n_vars > 0);
	let kernel = RoundKernel(comp);
	let xs: Vec<F> = iter::once(F::ZERO)
		.chain(eval_points::<F>(Comp::DEGREE))
		.collect();
	let mut sum = claim;
	let mut challenges = Vec::with_capacity(n_vars);

	// Slot 0 holds the eq tensor over the variables left after the round.
	let eq = compute.exec(compute.tensor_expand(&point[..n_vars - 1]));
	let mut all: Vec<C::FieldVec> = iter::once(eq).chain(cols).collect();

	for (round, &alpha) in point.iter().rev().enumerate() {
		let evals = match challenges.last().copied() {
			None => compute.exec(compute.fold_pair_map_reduce(&mut all, None, &kernel)),
			Some(r) => {
				let next_eq = compute.tensor_expand(&point[..n_vars - 1 - round]);
				let (all, kernel) = (&mut all, &kernel);
				compute.exec(compute.then(next_eq, move |eq| {
					all[0] = eq;
					compute.fold_pair_map_reduce(all, Some(r), kernel)
				}))
			}
		};

		// The claim is (1 - alpha) R'(0) + alpha R'(1).
		let y0 = (sum - alpha * evals[0]) * (F::ONE - alpha).invert_or_zero();
		let coeffs =
			RoundCoeffs(interpolate(&xs, &iter::once(y0).chain(evals).collect::<Vec<_>>()));
		transcript
			.message()
			.write_scalar_slice(binius_ip::mlecheck::RoundProof::truncate(coeffs.clone()).coeffs());
		let r = binius_transcript::fiat_shamir::CanSample::sample(transcript);
		sum = coeffs.evaluate(&r);
		challenges.push(r);
	}

	let finals = read_folded(compute, &mut all[1..], *challenges.last().expect("n_vars > 0"));
	(challenges, finals)
}
