// Copyright 2025 Irreducible Inc.

use binius_field::Field;
use binius_transcript::{
	VerifierTranscript,
	fiat_shamir::{CanSample, Challenger},
};

use super::error::Error;
use crate::protocols::sumcheck::{RoundCoeffs, RoundProof};

/// The reduced output of a sumcheck verification.
///
/// The [`verify`] function reduces a claim about the sum of a multivariate polynomial over the
/// boolean hypercube to its evaluation at a challenge point.
#[derive(Debug, Clone, PartialEq)]
pub struct SumcheckOutput<F: Field> {
	/// The evaluation of the sumcheck multivariate at the challenge point.
	pub eval: F,
	/// The sequence of sumcheck challenges defining the evaluation point.
	pub challenges: Vec<F>,
}

/// Verify a sumcheck protocol interaction.
///
/// The sumcheck verifier reduces a claim about the sum of a multivariate polynomial over the
/// boolean hypercube to its evaluation at a challenge point.
///
/// ## Arguments
///
/// * `n_vars` - The number of variables in the multivariate polynomial
/// * `degree` - The degree of the univariate polynomial in each round
/// * `sum` - The claimed sum of the multivariate polynomial over the boolean hypercube
/// * `transcript` - The transcript containing the prover's messages and randomness for challenges
///
/// ## Returns
///
/// Returns a `Result` containing the `SumcheckOutput` with the reduced evaluation and challenge
/// point, or an error if verification fails.
pub fn verify<F: Field, Challenger_: Challenger>(
	n_vars: usize,
	degree: usize,
	mut sum: F,
	transcript: &mut VerifierTranscript<Challenger_>,
) -> Result<SumcheckOutput<F>, Error> {
	let mut challenges = Vec::with_capacity(n_vars);
	for _round in 0..n_vars {
		let round_proof = RoundProof(RoundCoeffs(transcript.message().read_vec(degree)?));
		let challenge = transcript.sample();

		let round_coeffs = round_proof.recover(sum);
		sum = round_coeffs.evaluate(challenge);
		challenges.push(challenge);
	}

	Ok(SumcheckOutput {
		eval: sum,
		challenges,
	})
}
