// Copyright 2025 Irreducible Inc.

use binius_field::Field;
use binius_math::univariate::evaluate_univariate;
use binius_transcript::{
	VerifierTranscript,
	fiat_shamir::{CanSample, Challenger},
};

use crate::protocols::sumcheck::{self, Error, SumcheckOutput};

/// The reduced output of a sumcheck verification.
///
/// The [`batch_verify`] function reduces a set of claims on multivariate polynomials over the
/// boolean hypercube to their evaluation at a challenge point. See the function docstring for
/// details.
pub struct BatchSumcheckOutput<F: Field> {
	/// The challenge value of the batching variable.
	pub batch_coeff: F,
	/// The evaluation of the sumcheck multivariate at the challenge point.
	pub eval: F,
	/// Verifier challenges for each round of the sumcheck protocol.
	///
	/// One challenge is generated per variable in the multivariate polynomial,
	/// with challenges\[i\] corresponding to the i-th round of the protocol.
	///
	/// Note: reverse when folding high-to-low to obtain evaluation claim.
	pub challenges: Vec<F>,
}

/// Verify a batched sumcheck protocol interaction.
///
/// The batched sumcheck verifier reduces a set of claims about the sums of multivariate polynomials
/// over the boolean hypercube to their evaluation at a (shared) challenge point. This is achieved
/// by constructing an `n_vars + 1`-variate polynomial whose coefficients in the "new variable" are
/// the individual sum claims and evaluating it at a random point.
pub fn batch_verify<F: Field, Challenger_: Challenger>(
	n_vars: usize,
	degree: usize,
	sums: &[F],
	transcript: &mut VerifierTranscript<Challenger_>,
) -> Result<BatchSumcheckOutput<F>, Error> {
	let batch_coeff = transcript.sample();
	let sum = evaluate_univariate(sums, batch_coeff);

	let SumcheckOutput { eval, challenges } = sumcheck::verify(n_vars, degree, sum, transcript)?;

	Ok(BatchSumcheckOutput {
		batch_coeff,
		challenges,
		eval,
	})
}
