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

use binius_field::Field;
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

#[cfg(test)]
mod tests {
	use binius_field::arch::OptimalB128 as B128;
	use binius_math::test_utils::random_scalars;
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
}
