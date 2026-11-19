// Copyright 2026 The Binius Developers

//! The mini-prover: a sumcheck driver written once, generic over [`Compute`].
//!
//! Rounds bind the highest variable first, matching `binius_ip::sumcheck::verify`.

use std::iter;

use binius_field::{BinaryField, BinaryField1b, ExtensionField};
use binius_ip::sumcheck::RoundCoeffs;
use binius_transcript::{ProverTranscript, fiat_shamir::Challenger};

use crate::{Column, Composition, Compute, Elem, PairKernel};

/// The finite evaluation points of the round polynomial a [`RoundKernel`] reports, after `X = 1`:
/// `b_1, ..., b_{d-2}`. The round polynomial is also sampled at `X = infinity`.
pub(crate) fn extra_points<F: BinaryField>(degree: usize) -> impl Iterator<Item = F> {
	(1..degree - 1).map(<F as ExtensionField<BinaryField1b>>::basis)
}

/// Turns a [`Composition`] into a [`PairKernel`] whose outputs are the round polynomial at
/// `1, infinity, b_1, ..., b_{d-2}`, the order of `binius_ip_prover::sumcheck::round_evals`.
///
/// The line through `lo` and `hi` is `lo + X (lo + hi)`. At `X = 1` that is `hi`. The leading
/// coefficient (`X = infinity`) is the top-degree part at the direction `lo + hi`.
pub struct RoundKernel<'c, C>(pub &'c C);

impl<F: BinaryField, C: Composition<F>> PairKernel<F> for RoundKernel<'_, C> {
	fn n_in(&self) -> usize {
		C::N_IN
	}
	fn n_out(&self) -> usize {
		C::DEGREE
	}
	fn scratch(&self) -> usize {
		C::N_IN
	}

	fn map<T: Elem<F>>(&self, lo: &[T], hi: &[T], scratch: &mut [T], out: &mut [T]) {
		assert!(C::DEGREE >= 2);
		out[0] = self.0.eval(hi);
		for (s, (l, h)) in iter::zip(&mut *scratch, iter::zip(lo, hi)) {
			*s = l.clone() + h.clone();
		}
		out[1] = self.0.eval_infinity(scratch);
		for (o, x) in iter::zip(&mut out[2..], extra_points::<F>(C::DEGREE)) {
			for (s, (l, h)) in iter::zip(&mut *scratch, iter::zip(lo, hi)) {
				*s = l.clone() + (l.clone() + h.clone()) * x;
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

/// The round polynomial from `y0 = R(0)` and the kernel's `evals` (`R(1), R(inf), R(b_i)...`).
///
/// Subtracting `R(inf) X^d` leaves a degree `d - 1` polynomial, which the `d` finite points fix.
/// (`binius_ip_prover`'s `RoundEvals` does the same for `d <= 2` only.)
pub(crate) fn round_coeffs<F: BinaryField>(y0: F, evals: &[F]) -> RoundCoeffs<F> {
	let d = evals.len();
	let y_inf = evals[1];
	let xs: Vec<F> = [F::ZERO, F::ONE]
		.into_iter()
		.chain(extra_points(d))
		.collect();
	let ys: Vec<F> = iter::once(y0)
		.chain(iter::once(evals[0]))
		.chain(evals[2..].iter().copied())
		.zip(&xs)
		.map(|(y, &x)| y - y_inf * iter::repeat_n(x, d).product::<F>())
		.collect();
	let mut coeffs = interpolate(&xs, &ys);
	coeffs.push(y_inf);
	RoundCoeffs(coeffs)
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
	assert!(n_vars > 0);
	let mut cols: Vec<_> = cols.into_iter().map(Column::Owned).collect();
	let kernel = RoundKernel(comp);
	let mut sum = claim;
	let mut challenges = Vec::with_capacity(n_vars);

	for _ in 0..n_vars {
		let round =
			compute.fold_pair_map_reduce(&mut cols, challenges.last().copied(), None, &kernel);
		let evals = compute.exec(round);

		// The claim fixes R(0) = sum - R(1).
		let coeffs = round_coeffs(sum + evals[0], &evals);
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
pub(crate) fn read_folded<'s, F: BinaryField, C: Compute<F>>(
	compute: &C,
	cols: &mut [Column<C::FieldVec, C::FieldSlice<'s>>],
	r: F,
) -> Vec<F> {
	compute
		.exec(compute.then(compute.fold_highest_var(cols, r), |cols| {
			let cols = &*cols;
			let reads = cols
				.iter()
				.map(|col| compute.to_host(compute.as_slice(col.owned())))
				.collect();
			compute.join_all(reads)
		}))
		.into_iter()
		.map(|col| col[0])
		.collect()
}
