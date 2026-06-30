// Copyright 2026 Irreducible Inc.
use binius_field::Field;
use binius_ip::sumcheck::RoundCoeffs;
use binius_math::multilinear::eq::eq_one_var;

use crate::sumcheck::{Error, common::SumcheckProver};

/// Decorator that pads the number of variables of an inner [`SumcheckProver`].
///
/// Given an inner prover for the hypercube sum of an `n`-variate multilinear
/// $f(X_0, \ldots, X_{n-1}) = s$, this exposes a prover over `n + n_extra_vars` variables for the
/// equivalent claim
///
/// $$
/// f(X_0, \ldots, X_{n-1}) \cdot \text{eq}(0^{n_\text{extra}}, X_n, \ldots, X_{n+n_\text{extra}-1})
/// = s $$
///
/// Since $\text{eq}(0, \cdot)$ sums to 1 over the hypercube, this padded claim holds iff the
/// original does. This is useful for batching sumchecks with unequal numbers of variables: a
/// shorter sumcheck can be padded up to match a longer one.
///
/// The sumcheck protocol binds variables in high-index to low-index order, so the `n_extra_vars`
/// padding variables (the highest-indexed ones) are bound first. Concretely:
///
/// - In a padding round `i < n_extra_vars`, the round polynomial is $R_i(X) = s \cdot \text{eq}(0,
///   X) \cdot \prod_{k<i} \text{eq}(0, r_k)$, where $r_k$ is the $k$-th challenge. This is a
///   genuine degree-1 polynomial built without touching the inner prover.
/// - In an inner round `i \ge n_extra_vars`, the round polynomial is $R^\text{inner}_{i -
///   n_\text{extra}}(X) \cdot \prod_{k<n_\text{extra}} \text{eq}(0, r_k)$.
///
/// The accumulated equality factor $\prod_{k} \text{eq}(0, r_k)$ over the padding challenges is
/// tracked in `eq_prefix`; it stops changing once the padding rounds are done.
#[derive(Debug, Clone)]
pub struct PaddedSumcheckDecorator<F: Field, Inner> {
	inner: Inner,
	n_extra_vars: usize,
	/// Number of folds performed so far.
	round: usize,
	/// $\prod_{k < \min(\text{round}, n_\text{extra})} \text{eq}(0, r_k)$.
	eq_prefix: F,
}

impl<F: Field, Inner: SumcheckProver<F>> PaddedSumcheckDecorator<F, Inner> {
	/// Wraps `inner`, padding its claim with `n_extra_vars` extra (highest-indexed) variables.
	pub const fn new(inner: Inner, n_extra_vars: usize) -> Self {
		Self {
			inner,
			n_extra_vars,
			round: 0,
			eq_prefix: F::ONE,
		}
	}

	/// Whether the current round binds one of the padding variables.
	const fn in_padding_phase(&self) -> bool {
		self.round < self.n_extra_vars
	}
}

impl<F: Field, Inner: SumcheckProver<F>> SumcheckProver<F> for PaddedSumcheckDecorator<F, Inner> {
	fn n_vars(&self) -> usize {
		self.inner.n_vars() + self.n_extra_vars.saturating_sub(self.round)
	}

	fn n_claims(&self) -> usize {
		self.inner.n_claims()
	}

	fn round_claim(&self) -> Vec<F> {
		// In the padding phase the inner prover is untouched, so its round claim is the original
		// sum `s`; in the inner phase `eq_prefix` is the full padding product. Either way the
		// padded round claim is the inner round claim scaled by `eq_prefix`.
		self.inner
			.round_claim()
			.into_iter()
			.map(|claim| claim * self.eq_prefix)
			.collect()
	}

	fn execute(&mut self) -> Result<Vec<RoundCoeffs<F>>, Error> {
		if self.in_padding_phase() {
			// R_i(X) = (s * eq_prefix) * eq(0, X), with eq(0, X) = 1 - X. The inner prover is not
			// touched during padding rounds.
			let round_coeffs = self
				.round_claim()
				.into_iter()
				.map(|claim| RoundCoeffs(vec![claim, -claim]))
				.collect();
			Ok(round_coeffs)
		} else {
			let round_coeffs = self
				.inner
				.execute()?
				.into_iter()
				.map(|coeffs| coeffs * self.eq_prefix)
				.collect();
			Ok(round_coeffs)
		}
	}

	fn fold(&mut self, challenge: F) -> Result<(), Error> {
		if self.in_padding_phase() {
			// eq(0, challenge) = 1 - challenge.
			self.eq_prefix *= eq_one_var(F::ZERO, challenge);
		} else {
			self.inner.fold(challenge)?;
		}
		self.round += 1;
		Ok(())
	}

	fn finish(self) -> Result<Vec<F>, Error> {
		// The final multilinear evaluations are those of the inner prover; padding does not add
		// multilinears.
		self.inner.finish()
	}
}

#[cfg(test)]
mod tests {
	use binius_field::{
		Random,
		arch::{OptimalB128, OptimalPackedB128},
	};
	use binius_ip::{
		channel::IPVerifierChannel,
		sumcheck::{RoundCoeffs, RoundProof},
	};
	use binius_math::{
		inner_product::inner_product_par,
		multilinear::{eq::eq_one_var, evaluate::evaluate},
		test_utils::random_field_buffer,
	};
	use binius_transcript::{ProverTranscript, fiat_shamir::HasherChallenger};
	use rand::prelude::*;

	use super::*;
	use crate::sumcheck::{bivariate_product::BivariateProductSumcheckProver, prove::prove_single};

	type F = OptimalB128;
	type P = OptimalPackedB128;
	type StdChallenger = HasherChallenger<sha2::Sha256>;

	fn make_inner(rng: &mut impl Rng, n_vars: usize) -> (BivariateProductSumcheckProver<P>, F) {
		let a = random_field_buffer::<P>(&mut *rng, n_vars);
		let b = random_field_buffer::<P>(&mut *rng, n_vars);
		let sum = inner_product_par(&a, &b);
		let prover = BivariateProductSumcheckProver::new([a, b], sum).unwrap();
		(prover, sum)
	}

	/// The padding rounds emit `s * eq(0, X) * prod eq(0, r_k)`, and the inner rounds emit the
	/// inner round polynomial scaled by the same padding product.
	#[test]
	fn test_round_polynomials_closed_form() {
		let mut rng = StdRng::seed_from_u64(0);
		let n_vars = 6;
		let n_extra_vars = 3;

		let (inner, sum) = make_inner(&mut rng, n_vars);
		// A parallel bare inner prover, driven only on the inner-phase challenges, to compare round
		// polynomials against.
		let (mut bare_inner, _) = make_inner(&mut StdRng::seed_from_u64(0), n_vars);
		let mut padded = PaddedSumcheckDecorator::new(inner, n_extra_vars);

		let challenges = (0..n_vars + n_extra_vars)
			.map(|_| F::random(&mut rng))
			.collect::<Vec<_>>();

		let mut eq_prefix = F::ONE;
		for (i, &challenge) in challenges.iter().enumerate() {
			assert_eq!(padded.n_vars(), n_vars + n_extra_vars - i);

			let round_coeffs = padded.execute().unwrap();
			assert_eq!(round_coeffs.len(), 1);

			if i < n_extra_vars {
				// Degree-1 polynomial v * (1 - X) with v = s * eq_prefix.
				let v = sum * eq_prefix;
				assert_eq!(round_coeffs[0], RoundCoeffs(vec![v, -v]));
			} else {
				// Inner round polynomial scaled by the (now complete) padding product.
				let inner_coeffs = bare_inner.execute().unwrap();
				let expected = inner_coeffs[0].clone() * eq_prefix;
				assert_eq!(round_coeffs[0], expected);
				bare_inner.fold(challenge).unwrap();
			}

			padded.fold(challenge).unwrap();
			if i < n_extra_vars {
				eq_prefix *= eq_one_var(F::ZERO, challenge);
			}
		}

		assert_eq!(padded.n_vars(), 0);
	}

	/// `round_claim()` called before `execute()` must equal `R(0) + R(1)` of each returned round
	/// polynomial.
	#[test]
	fn test_round_claim_invariant() {
		let mut rng = StdRng::seed_from_u64(1);
		let n_vars = 5;
		let n_extra_vars = 2;

		let (inner, _) = make_inner(&mut rng, n_vars);
		let mut padded = PaddedSumcheckDecorator::new(inner, n_extra_vars);

		for _ in 0..n_vars + n_extra_vars {
			let claims = padded.round_claim();
			let round_coeffs = padded.execute().unwrap();
			assert_eq!(claims.len(), round_coeffs.len());
			for (claim, coeffs) in claims.iter().zip(&round_coeffs) {
				assert_eq!(*claim, coeffs.sum_over_endpoints());
			}
			padded.fold(F::random(&mut rng)).unwrap();
		}
	}

	/// Full prove/verify roundtrip through a transcript, with a degree-aware verifier loop (padding
	/// rounds are degree 1, inner rounds degree 2).
	#[test]
	fn test_prove_verify_roundtrip() {
		let mut rng = StdRng::seed_from_u64(2);
		let n_vars = 7;
		let n_extra_vars = 4;
		let total_vars = n_vars + n_extra_vars;

		let a = random_field_buffer::<P>(&mut rng, n_vars);
		let b = random_field_buffer::<P>(&mut rng, n_vars);
		let sum = inner_product_par(&a, &b);
		let inner = BivariateProductSumcheckProver::new([a.clone(), b.clone()], sum).unwrap();
		let padded = PaddedSumcheckDecorator::new(inner, n_extra_vars);

		let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
		let output = prove_single(padded, &mut prover_transcript).unwrap();
		prover_transcript
			.message()
			.write_slice(&output.multilinear_evals);

		// Degree-aware verification mirroring `binius_ip::sumcheck::verify`.
		let mut verifier_transcript = prover_transcript.into_verifier();
		let mut running_sum = sum;
		let mut challenges = Vec::with_capacity(total_vars);
		for round in 0..total_vars {
			let degree = if round < n_extra_vars { 1 } else { 2 };
			let round_proof =
				RoundProof(RoundCoeffs(verifier_transcript.recv_many(degree).unwrap()));
			let challenge = verifier_transcript.sample();
			let round_coeffs = round_proof.recover(running_sum);
			running_sum = round_coeffs.evaluate(challenge);
			challenges.push(challenge);
		}
		let reduced_eval = running_sum;

		let multilinear_evals: Vec<F> = verifier_transcript.message().read_vec(2).unwrap();
		assert_eq!(output.multilinear_evals, multilinear_evals);
		assert_eq!(output.challenges, challenges);

		// The padding challenges (bound first) define the eq factor.
		let eq_pad: F = challenges[..n_extra_vars]
			.iter()
			.map(|&r| eq_one_var(F::ZERO, r))
			.product();

		// Reduced eval = A(r_inner) * B(r_inner) * prod eq(0, r_pad).
		assert_eq!(multilinear_evals[0] * multilinear_evals[1] * eq_pad, reduced_eval);

		// The inner multilinears evaluate to the claimed values at the (reversed) inner challenges.
		let mut inner_point = challenges[n_extra_vars..].to_vec();
		inner_point.reverse();
		assert_eq!(evaluate(&a, &inner_point), multilinear_evals[0]);
		assert_eq!(evaluate(&b, &inner_point), multilinear_evals[1]);
	}

	/// With no extra variables the decorator is a transparent passthrough.
	#[test]
	fn test_no_padding_passthrough() {
		let mut rng = StdRng::seed_from_u64(3);
		let n_vars = 6;

		let (inner, sum) = make_inner(&mut rng, n_vars);
		let padded = PaddedSumcheckDecorator::new(inner, 0);
		assert_eq!(padded.n_vars(), n_vars);

		let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
		let output = prove_single(padded, &mut prover_transcript).unwrap();
		prover_transcript
			.message()
			.write_slice(&output.multilinear_evals);

		let mut verifier_transcript = prover_transcript.into_verifier();
		let sumcheck_output = binius_ip::sumcheck::verify(n_vars, 2, sum, &mut verifier_transcript)
			.expect("verification should succeed");
		let multilinear_evals: Vec<F> = verifier_transcript.message().read_vec(2).unwrap();

		assert_eq!(multilinear_evals[0] * multilinear_evals[1], sumcheck_output.eval);
		assert_eq!(output.challenges, sumcheck_output.challenges);
	}
}
