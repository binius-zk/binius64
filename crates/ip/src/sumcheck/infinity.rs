// Copyright 2026 The Binius Developers

//! Sumcheck over the infinity hypercube $\{0, \infty\}^n$.
//!
//! This is the ordinary sumcheck with the round check $R(0) + R(\infty) = s$, where $R(\infty)$
//! is the leading coefficient of $R$.
//!
//! Evaluating a multilinear on $\{0, \infty\}^n$ reads off its monomial coefficients. So for the
//! multilinears $u^\infty$, $w^\infty$ whose monomial coefficients are the tables $u$, $w$,
//!
//! $$
//! \sum_{x \in \{0, \infty\}^n} u^\infty(x) \cdot w^\infty(x) = \langle u, w \rangle
//! $$
//!
//! The sumcheck reduces such a claim to the evaluations $u^\infty(r)$, $w^\infty(r)$ at the
//! challenge point $r$, which has one coordinate per round.
//!
//! A claim on fewer variables than the batch needs no padding factor: summing out a coordinate the
//! summand does not depend on leaves the claim unchanged.

use binius_field::{Field, field::FieldOps};
use binius_math::univariate::evaluate_univariate;

use crate::{
	channel::IPVerifierChannel,
	sumcheck::{BatchSumcheckOutput, Error, RoundCoeffs, RoundProof, SumcheckOutput},
};

/// Verify a sumcheck over $\{0, \infty\}^n$.
///
/// This is [`super::verify()`] with each round checked as $R(0) + R(\infty) = s$.
///
/// ## Arguments
///
/// * `n_vars` - The number of variables in the multivariate polynomial
/// * `degree` - The degree of the univariate polynomial in each round
/// * `sum` - The claimed sum of the multivariate polynomial over $\{0, \infty\}^n$
/// * `channel` - The channel for receiving prover messages and sampling challenges
pub fn verify<F, C>(
	n_vars: usize,
	degree: usize,
	mut sum: C::Elem,
	channel: &mut C,
) -> Result<SumcheckOutput<C::Elem>, Error>
where
	F: Field,
	C: IPVerifierChannel<F>,
{
	let mut challenges = Vec::with_capacity(n_vars);
	for _round in 0..n_vars {
		let round_proof = RoundProof(RoundCoeffs(channel.recv_many(degree)?));
		let challenge = channel.sample();

		sum = round_proof.recover_infinity(sum).evaluate(&challenge);
		challenges.push(challenge);
	}

	Ok(SumcheckOutput {
		eval: sum,
		challenges,
	})
}

/// Verify a batched sumcheck over $\{0, \infty\}^n$.
///
/// The claims are combined with powers of a sampled batching coefficient, as in
/// [`super::batch_verify`]. Claims may have fewer than `n_vars` variables: a claim on `k`
/// variables closes at the low `k` coordinates of the evaluation point, which are the last `k`
/// challenges.
pub fn batch_verify<F, C>(
	n_vars: usize,
	degree: usize,
	sums: &[C::Elem],
	channel: &mut C,
) -> Result<BatchSumcheckOutput<C::Elem>, Error>
where
	F: Field,
	C: IPVerifierChannel<F>,
{
	let batch_coeff = channel.sample();
	let sum = evaluate_univariate(sums, &batch_coeff);

	let SumcheckOutput { eval, challenges } = verify::<F, C>(n_vars, degree, sum, channel)?;

	Ok(BatchSumcheckOutput {
		batch_coeff,
		eval,
		challenges,
	})
}

/// Translates a point $\rho$ of a sumcheck over $\{0, \infty\}^n$ to the Boolean hypercube.
///
/// In characteristic 2, whenever no $\rho_j = 1$, the multilinear $u^\infty$ with monomial
/// coefficients $u$ and the multilinear extension $\tilde{u}$ of $u$ agree up to a scale:
///
/// $$
/// u^\infty(\rho_{<k}) = \prod_{j < k} (1 + \rho_j) \cdot \tilde{u}(\sigma_{<k}), \qquad
/// \sigma_j = \frac{\rho_j}{1 + \rho_j}
/// $$
///
/// Returns $\sigma$ and the scales $\prod_{j < k} (1 + \rho_j)$ for every $k = 0, \ldots, n$.
///
/// A coordinate $\rho_j = 1$ has no translation. It maps to $\sigma_j = 0$ and zeroes every scale
/// past it, and occurs with probability $n / |F|$ over a uniform $\rho$.
pub fn to_mle_point<F: FieldOps>(point: &[F]) -> (Vec<F>, Vec<F>) {
	let mut scales = Vec::with_capacity(point.len() + 1);
	scales.push(F::one());
	let sigma = point
		.iter()
		.map(|rho_j| {
			let one_plus_rho = F::one() + rho_j;
			let scale = scales.last().expect("scales starts non-empty").clone() * &one_plus_rho;
			scales.push(scale);
			rho_j.clone() * one_plus_rho.invert_or_zero()
		})
		.collect();
	(sigma, scales)
}

#[cfg(test)]
mod tests {
	use binius_field::arch::OptimalB128 as B128;
	use binius_math::{
		multilinear::{
			eq::eq_ind_partial_eval_scalars,
			hypercube::{InfCube, eq_ind_partial_eval_scalars as monomials},
		},
		test_utils::random_scalars,
	};
	use rand::prelude::*;

	use super::*;

	#[test]
	fn recover_infinity_round_trips() {
		let mut rng = StdRng::seed_from_u64(0);
		for degree in 1..=4 {
			let coeffs = RoundCoeffs(random_scalars::<B128>(&mut rng, degree + 1));
			// R(0) + R(∞) is the constant plus the leading coefficient.
			let sum = coeffs[0] + coeffs[degree];
			assert_eq!(coeffs.clone().truncate().recover_infinity(sum), coeffs);
		}
	}

	#[test]
	fn to_mle_point_matches_the_monomial_evaluation() {
		let mut rng = StdRng::seed_from_u64(0);
		let n_vars = 4;
		let point = random_scalars::<B128>(&mut rng, n_vars);
		let (sigma, scales) = to_mle_point(&point);
		for k in 0..=n_vars {
			let u = random_scalars::<B128>(&mut rng, 1 << k);
			let dot = |basis: Vec<B128>| u.iter().zip(basis).map(|(u, b)| *u * b).sum::<B128>();
			let monomial_eval = dot(monomials::<InfCube, B128>(&point[..k]));
			let mle_eval = dot(eq_ind_partial_eval_scalars(&sigma[..k]));
			assert_eq!(monomial_eval, scales[k] * mle_eval, "k={k}");
		}
	}
}
