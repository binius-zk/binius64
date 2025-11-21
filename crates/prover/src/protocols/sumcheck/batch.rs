// Copyright 2024-2025 Irreducible Inc.

use binius_field::Field;
use binius_transcript::{
	ProverTranscript,
	fiat_shamir::{CanSample, Challenger},
};
use binius_verifier::protocols::sumcheck::common::RoundCoeffs;

use crate::protocols::sumcheck::{common::SumcheckProver, error::Error};

/// Prover view of the execution result of a batched sumcheck.
#[derive(Debug, PartialEq, Eq)]
pub struct BatchSumcheckOutput<F: Field> {
	/// Verifier challenges for each round of the sumcheck protocol.
	///
	/// One challenge is generated per variable in the multivariate polynomial,
	/// with challenges\[i\] corresponding to the i-th round of the protocol.
	///
	/// Note: reverse when folding high-to-low to obtain evaluation claim.
	pub challenges: Vec<F>,
	/// Evaluation claims on non-transparent multilinears, per prover.
	///
	/// Each inner vector contains the evaluation values for one prover's
	/// multilinear polynomials at the challenge point.
	pub multilinear_evals: Vec<Vec<F>>,
}

/// Prove a batched sumcheck protocol execution, where all provers have the same number of rounds.
///
/// The batched sumcheck reduces a set of claims about the sums of multivariate polynomials over
/// the boolean hypercube to their evaluation at a (shared) challenge point. This is achieved by
/// constructing an `n_vars + 1`-variate polynomial whose coefficients in the "new variable" are the
/// individual sum claims and evaluating it at a random point. Due to linearity of sums each claim
/// can be proven separately with an individual [`SumcheckProver`] followed by weighted summation of
/// the round polynomials.
///
/// This function performs the sumcheck protocol and returns the challenges and evaluation claims,
/// but does not write the evaluation claims to the transcript. Use [`batch_prove_and_write_evals`]
/// if you need to write the evaluations to the transcript.
pub fn batch_prove<F, Prover, Challenger_>(
	mut provers: Vec<Prover>,
	transcript: &mut ProverTranscript<Challenger_>,
) -> Result<BatchSumcheckOutput<F>, Error>
where
	F: Field,
	Prover: SumcheckProver<F>,
	Challenger_: Challenger,
{
	let Some(first_prover) = provers.first() else {
		return Ok(BatchSumcheckOutput {
			challenges: Vec::new(),
			multilinear_evals: Vec::new(),
		});
	};

	let n_vars = first_prover.n_vars();

	if provers.iter().any(|prover| prover.n_vars() != n_vars) {
		return Err(Error::ProverRoundCountMismatch);
	}

	let batch_coeff = transcript.sample();

	let mut challenges = Vec::with_capacity(n_vars);
	for _ in 0..n_vars {
		let mut all_round_coeffs = Vec::new();

		for prover in &mut provers {
			all_round_coeffs.extend(prover.execute()?);
		}

		let batched_round_coeffs = all_round_coeffs
			.into_iter()
			.rfold(RoundCoeffs::default(), |acc, coeffs| acc * batch_coeff + &coeffs);

		let round_proof = batched_round_coeffs.truncate();

		transcript
			.message()
			.write_scalar_slice(round_proof.coeffs());

		let challenge = transcript.sample();
		challenges.push(challenge);

		for prover in &mut provers {
			prover.fold(challenge)?;
		}
	}

	// TODO: this differs from prove_single, which doesn't reverse
	challenges.reverse();

	let multilinear_evals = provers
		.into_iter()
		.map(|prover| prover.finish())
		.collect::<Result<Vec<_>, _>>()?;

	Ok(BatchSumcheckOutput {
		challenges,
		multilinear_evals,
	})
}

/// Prove a batched sumcheck protocol and write evaluation claims to the transcript.
///
/// This function combines [`batch_prove`] with writing the evaluation claims to the transcript.
/// It performs the batched sumcheck protocol execution and then writes all the multilinear
/// evaluation values to the transcript in order.
///
/// # Arguments
///
/// * `provers` - Vector of sumcheck provers, each handling one claim in the batch
/// * `transcript` - The prover's transcript for the Fiat-Shamir protocol
///
/// # Returns
///
/// Returns [`BatchSumcheckOutput`] containing the challenges and evaluation claims that were
/// written to the transcript.
pub fn batch_prove_and_write_evals<F, Prover, Challenger_>(
	provers: Vec<Prover>,
	transcript: &mut ProverTranscript<Challenger_>,
) -> Result<BatchSumcheckOutput<F>, Error>
where
	F: Field,
	Prover: SumcheckProver<F>,
	Challenger_: Challenger,
{
	let output = batch_prove(provers, transcript)?;

	let mut writer = transcript.message();
	for evals in &output.multilinear_evals {
		writer.write_scalar_slice(evals);
	}
	Ok(output)
}
