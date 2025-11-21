// Copyright 2025 Irreducible Inc.
use binius_field::Field;
use binius_transcript::{
	VerifierTranscript,
	fiat_shamir::{CanSample, Challenger},
};

use crate::protocols::sumcheck::{self, RoundCoeffs, SumcheckOutput};

/// An MLE-check protocol is an interactive protocol similar to sumcheck, but with modifications
/// introduced in [Gruen24], Section 3.
///
/// The prover in an MLE-check argues a that for some $n$-variate polynomial
/// $F(X_0, \ldots, X_{n-1})$ (which is not necessarily multilinear), for a given point
/// $(z_0, \ldots, z_{n-1})$ and claimed value $s$, that
///
/// $$
/// s = \sum_{v \in B_n} F(v) \cdot eq(v, z)
/// $$
///
/// Unless $F$ is indeed multilinear, $s \ne F(z)$ necessarily. While the prover and verifier could
/// engage in a standard sumcheck protocol to reduce this claim, it is concretely more efficient to
/// use the optimized protocol from [Gruen24], which we call an "MLE-check".
///
/// [Gruen24]: <https://eprint.iacr.org/2024/108>
///
/// ## Arguments
///
/// * `n_vars` - The number of variables in the multivariate polynomial
/// * `degree` - The degree of the univariate polynomial in each round
/// * `eval` - The claimed multilinear-extension evaluation of the multivariate polynomial
/// * `transcript` - The transcript containing the prover's messages and randomness for challenges
///
/// ## Returns
///
/// Returns a `Result` containing the `SumcheckOutput` with the reduced evaluation and challenge
/// point, or an error if verification fails.
pub fn verify<F: Field, Challenger_: Challenger>(
	point: &[F],
	degree: usize,
	mut eval: F,
	transcript: &mut VerifierTranscript<Challenger_>,
) -> Result<SumcheckOutput<F>, sumcheck::Error> {
	let n_vars = point.len();

	let mut challenges = Vec::with_capacity(n_vars);
	for &z_i in point.iter().rev() {
		let round_proof = RoundProof(RoundCoeffs(transcript.message().read_vec(degree)?));
		let challenge = transcript.sample();

		let round_coeffs = round_proof.recover(eval, z_i);
		eval = round_coeffs.evaluate(challenge);
		challenges.push(challenge);
	}

	Ok(SumcheckOutput { eval, challenges })
}

/// An MLE-check round proof is a univariate polynomial in monomial basis with the coefficient of
/// the lowest-degree term truncated off.
///
/// Since the verifier knows the claimed linear extrapolation of the polynomial values at the
/// points 0 and 1, the low-degree term coefficient can be easily recovered. Truncating the
/// coefficient off saves a small amount of proof data.
///
/// This is an analogous struct to [`sumcheck::RoundProof`], except that we truncate the low-degree
/// coefficient instead of the high-degree coefficient.
///
/// In a sumcheck protocol, the verifier has a claimed sum $s$ and the round polynomial $R(X)$ must
/// satisfy $R(0) + R(1) = s$. In an MLE-check protocol, the verifier has a claimed coordinate
/// $\alpha$ and extrapolated value $s$ and the round polynomial must satisfy
/// $(1 - \alpha) R(0) + \alpha R(1) = s$. This difference changes the recovery procedure and which
/// polynomial coefficient is most convenient to truncate.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct RoundProof<F: Field>(pub RoundCoeffs<F>);

impl<F: Field> RoundProof<F> {
	/// Truncates the polynomial coefficients to a round proof.
	///
	/// Removes the first coefficient. See the struct documentation for more info.
	///
	/// ## Pre-conditions
	///
	/// * `coeffs` must not be empty
	pub fn truncate(mut coeffs: RoundCoeffs<F>) -> Self {
		coeffs.0.remove(0);
		Self(coeffs)
	}

	/// Recovers all univariate polynomial coefficients from the compressed round proof.
	///
	/// The prover has sent coefficients for the purported $i$'th round polynomial
	/// $R(X) = \sum_{j=0}^d a_j * X^j$.
	///
	/// However, the prover has not sent the lowest degree coefficient $a_0$. The verifier will
	/// need to recover this missing coefficient.
	///
	/// Let $s$ denote the current round's claimed sum and $\alpha_i$ be the $i$'th coordinate of
	/// the evaluation point.
	///
	/// The verifier expects the round polynomial $R_i$ to satisfy the identity
	/// $s = (1 - \alpha) R(0) + \alpha R(1)$, or equivalently, $s = R(0) + (R(1) - R(0)) \alpha$.
	///
	/// Using
	///     $R(0) = a_0$
	///     $R(1) = \sum_{j=0}^d a_j$
	/// There is a unique $a_0$ that allows $R$ to satisfy the above identity. Specifically,
	/// $a_0 = s - \alpha \sum_{j=1}^d a_j$.
	pub fn recover(self, eval: F, alpha: F) -> RoundCoeffs<F> {
		let Self(RoundCoeffs(mut coeffs)) = self;
		let first_coeff = eval - alpha * coeffs.iter().sum::<F>();
		coeffs.insert(0, first_coeff);
		RoundCoeffs(coeffs)
	}

	/// The truncated polynomial coefficients.
	pub fn coeffs(&self) -> &[F] {
		&self.0.0
	}
}

#[cfg(test)]
mod tests {
	use binius_field::Random;
	use binius_math::{line::extrapolate_line_packed, test_utils::random_scalars};
	use rand::prelude::*;

	use super::*;
	use crate::config::B128;

	fn test_recover_with_degree<F: Field>(mut rng: impl Rng, alpha: F, degree: usize) {
		let coeffs = RoundCoeffs(random_scalars(&mut rng, degree + 1));

		let v0 = coeffs.evaluate(F::ZERO);
		let v1 = coeffs.evaluate(F::ONE);
		let eval = extrapolate_line_packed(v0, v1, alpha);

		let proof = RoundProof::truncate(coeffs.clone());
		assert_eq!(proof.recover(eval, alpha), coeffs);
	}

	#[test]
	fn test_recover() {
		let mut rng = StdRng::seed_from_u64(0);
		let alpha = B128::random(&mut rng);

		for degree in 0..4 {
			// Test with random coordinate
			test_recover_with_degree(&mut rng, alpha, degree);

			// Test edge case coordinate values
			test_recover_with_degree(&mut rng, B128::ZERO, degree);
			test_recover_with_degree(&mut rng, B128::ONE, degree);
		}
	}
}
