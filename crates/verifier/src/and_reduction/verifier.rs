// Copyright 2025 Irreducible Inc.

use binius_field::BinaryField;
use binius_math::BinarySubspace;
use binius_transcript::{
	VerifierTranscript,
	fiat_shamir::{CanSample, Challenger},
};

use crate::{
	Error,
	and_reduction::{
		univariate::univariate_poly::{GenericPo2UnivariatePoly, UnivariatePolyIsomorphic},
		utils::constants::ROWS_PER_HYPERCUBE_VERTEX,
	},
	error::VerificationError,
	protocols::{mlecheck::verify, sumcheck::SumcheckOutput},
};

/// Output from the AND constraint reduction protocol verification.
#[derive(Debug, PartialEq)]
pub struct AndCheckOutput<F> {
	pub a_eval: F,
	pub b_eval: F,
	pub c_eval: F,
	pub z_challenge: F,
	pub eval_point: Vec<F>,
}

/// Verifies the AND constraint reduction protocol via univariate zerocheck.
///
/// Note: Following section 4.4 of the Binius64 writeup, Z is the bit index within a word, and X is
/// the word index
///
/// Let our oblong polynomials be A(Z, X₀, ...), B(Z, X₀, ...), and C(Z, X₀, ...)
///
/// Let our zerocheck challenges be (r₀, ...)
///
/// This protocol reduces the verification of AND constraints (A(Z,X₀,...,Xₙ₋₁)·B(Z,X₀,...,Xₙ₋₁) -
/// C(Z,X₀,...,Xₙ₋₁) = 0) over a multivariate domain to a single multilinear polynomial evaluation.
/// The key insight is that A·B-C = 0 if and only if for all Z, the multilinear extension of A·B-C
/// evaluates to zero at a random point (Z,r₀,...,rₙ₋₁), (up to some negligible error probability).
///
/// Note: This is equivalent to proving |D| multilinear zerochecks at once, all using the same
/// random zerocheck challenges
///
/// ## Phase 1: Univariate Polynomial Verification
///
/// The prover sends a univariate polynomial R₀(Z) that encodes the sum:
///
/// R₀(Z) = ∑_{X₀,...,Xₙ₋₁ ∈ {0,1}} (A(Z,X₀,...,Xₙ₋₁)·B(Z,X₀,...,Xₙ₋₁) -
/// C(Z,X₀,...,Xₙ₋₁))·eq(X₀,...,Xₙ₋₁; r₀,...,rₙ₋₁)
///
/// where:
/// - A(Z,X₀,...,Xₙ₋₁), B(Z,X₀,...,Xₙ₋₁), C(Z,X₀,...,Xₙ₋₁) are oblong multilinear polynomials
///   representing the AND constraint operands
/// - eq(X₀,...,Xₙ₋₁; r₀,...,rₙ₋₁) is the multilinear equality indicator partially evaluated at a
///   series of random and compile-time pre-known challenges r₀,...,rₙ₋₁ (note: Z is not included in
///   the equality check)
/// - Z ranges over a univariate domain of size 2^(SKIPPED_VARS + 1)
///
/// The equality indicator eq(X₀,...,Xₙ₋₁; r₀,...,rₙ₋₁) = ∏ᵢ₌₀ⁿ⁻¹(Xᵢ·rᵢ + (1-Xᵢ)·(1-rᵢ)) ensures
/// we're checking that the multilinear extension of A·B-C evaluates to zero at the random point
/// (Z, r₀,...,rₙ₋₁) for each Z in the domain.
///
/// The polynomial R₀(Z) has degree at most 2*(|D| - 1) where |D| is the domain size. The prover
/// only sends evaluations on an extension domain (the upper half) since R₀(Z) = 0 on the base
/// domain when all AND constraints are satisfied.
///
/// ## Phase 2: Multilinear Sumcheck Reduction
///
/// After the verifier samples a random challenge z for Z, the protocol continues with a standard
/// sumcheck protocol on the remaining variables X₀,...,Xₙ₋₁ to verify that:
///
/// R₀(z) = ∑_{X₀,...,Xₙ₋₁ ∈ {0,1}} (A(z,X₀,...,Xₙ₋₁)·B(z,X₀,...,Xₙ₋₁) -
/// C(z,X₀,...,Xₙ₋₁))·eq(X₀,...,Xₙ₋₁; r₀,...,rₙ₋₁)
///
///
/// This reduces to a single evaluation of the folded polynomial at the sumcheck challenge point.
///
/// ## Arguments
///
/// * `n_vars` - The number of variables in the sumcheck protocol (excluding the univariate variable
///   Z)
/// * `transcript` - The verifier's transcript for reading prover messages and sampling challenges
/// * `round_message_univariate_domain` - The univariate domain D for polynomial evaluations
///
/// ## Returns
///
/// Returns `AndCheckOutput` containing:
/// - `z_challenge`: The univariate challenge z sampled for the bit-index variable
/// - `eval_point`: The multilinear evaluation point. Prepened with the `z_challenge` this makes the
///   oblong evaluation point
/// - `a_eval`, `b_eval`, `c_eval`: The claimed evaluations of the A, B, and C at the oblong
///   evaluation point
pub fn verify_with_transcript<F, TranscriptChallenger>(
	all_zerocheck_challenges: &[F],
	transcript: &mut VerifierTranscript<TranscriptChallenger>,
	round_message_univariate_domain: &BinarySubspace<F>,
) -> Result<AndCheckOutput<F>, Error>
where
	F: BinaryField,
	TranscriptChallenger: Challenger,
{
	let univariate_message_coeffs_ext_domain: Vec<F> = transcript
		.message()
		.read_scalar_slice(ROWS_PER_HYPERCUBE_VERTEX)?;

	let mut univariate_message_coeffs = vec![F::ZERO; 2 * ROWS_PER_HYPERCUBE_VERTEX];

	univariate_message_coeffs[ROWS_PER_HYPERCUBE_VERTEX..]
		.copy_from_slice(&univariate_message_coeffs_ext_domain);

	let univariate_message = GenericPo2UnivariatePoly::new(
		univariate_message_coeffs,
		round_message_univariate_domain.clone(),
	);
	let univariate_sumcheck_challenge: F = transcript.sample();

	let sumcheck_claim = univariate_message.evaluate_at_challenge(univariate_sumcheck_challenge);

	let SumcheckOutput {
		eval,
		challenges: mut eval_point,
	} = verify(all_zerocheck_challenges, 2, sumcheck_claim, transcript)?;

	let mut reader = transcript.message();
	let a_eval = reader.read()?;
	let b_eval = reader.read()?;
	let c_eval = reader.read()?;

	if eval != a_eval * b_eval - c_eval {
		return Err(VerificationError::AndReductionMLECheckFailed.into());
	}

	eval_point.reverse();

	Ok(AndCheckOutput {
		a_eval,
		b_eval,
		c_eval,
		z_challenge: univariate_sumcheck_challenge,
		eval_point,
	})
}
