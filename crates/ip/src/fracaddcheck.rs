// Copyright 2025-2026 The Binius Developers

//! Reduction from fractional-addition layers to a multilinear evaluation claim.
//!
//! Each layer represents combining siblings with the fractional-addition rule:
//! (a0 / b0) + (a1 / b1) = (a0 * b1 + a1 * b0) / (b0 * b1).

use binius_field::{Field, field::FieldOps};
use binius_math::{line::extrapolate_line, multilinear::eq::eq_one_var};
use binius_transcript::Error as TranscriptError;

use crate::{
	channel::IPVerifierChannel,
	sumcheck::{self, BatchSumcheckOutput},
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FracAddEvalClaim<F> {
	/// The evaluation of the numerator and denominator multilinears.
	pub num_eval: F,
	pub den_eval: F,
	/// The evaluation point.
	pub point: Vec<F>,
}

pub fn verify<F, C>(
	k: usize,
	claim: FracAddEvalClaim<C::Elem>,
	channel: &mut C,
) -> Result<FracAddEvalClaim<C::Elem>, Error>
where
	F: Field,
	C: IPVerifierChannel<F>,
{
	if k == 0 {
		return Ok(claim);
	}

	let FracAddEvalClaim {
		num_eval,
		den_eval,
		point,
	} = claim;

	let evals = [num_eval, den_eval];

	// Reduce numerator and denominator sum claims to evaluations at a challenge point.
	let BatchSumcheckOutput {
		batch_coeff,
		eval,
		mut challenges,
	} = sumcheck::batch_verify_mle(&point, 2, &evals, channel)?;

	// Read evaluations of numerator/denominator halves at the reduced point.
	let [num_0, num_1, den_0, den_1] = channel.recv_array()?;

	// Sumcheck binds variables high-to-low; reverse to low-to-high for point evaluation.
	challenges.reverse();
	let reduced_eval_point = challenges;

	let numerator_eval = num_0.clone() * den_1.clone() + num_1.clone() * den_0.clone();
	let denominator_eval = den_0.clone() * den_1.clone();
	let batched_eval = numerator_eval + denominator_eval * batch_coeff;

	channel.assert_zero(batched_eval - eval)?;

	// Reduce evaluations of the two halves to a single evaluation at the next point.
	let r = channel.sample();
	let next_num = extrapolate_line(num_0, num_1, r.clone());
	let next_den = extrapolate_line(den_0, den_1, r.clone());

	let mut next_point = reduced_eval_point;
	next_point.push(r);

	verify(
		k - 1,
		FracAddEvalClaim {
			num_eval: next_num,
			den_eval: next_den,
			point: next_point,
		},
		channel,
	)
}

/// Pads a leaf fraction — the forward map that [`unpad_leaf_claim`] inverts.
///
/// Padding a tree scales its numerator by the padding coordinates' equality weight $q$ and sends
/// its denominator through the one-padding selector $\textsf{sel}(q, v) = 1 + (v - 1) q$. A
/// verifier that rebuilds a padded batch's leaf claim from transparent parts applies this to each
/// tree, so the two directions live together.
///
/// The weight is a parameter rather than the padding coordinates, so a caller padding several
/// trees computes each distinct one once.
///
/// # Arguments
///
/// * `fraction` - The unpadded leaf's numerator and denominator.
/// * `pad_eq` - The padding coordinates' equality weight $\text{eq}(0^\nu; X_\text{pad})$, which is
///   [`eq_ind_zero`] over the lowest coordinates of the leaf point.
///
/// [`eq_ind_zero`]: binius_math::multilinear::eq::eq_ind_zero
pub fn pad_leaf_fraction<E: FieldOps>(fraction: (E, E), pad_eq: E) -> (E, E) {
	let (num, den) = fraction;
	(num * pad_eq.clone(), E::one() + (den - E::one()) * pad_eq)
}

/// Reduces a leaf claim on a zero-fraction-padded witness to the claim on the witness itself.
///
/// A batched fractional-addition check over trees of unequal depths lifts each shallow tree to the
/// batch's depth by filling `n_pad_vars` extra leaf positions with the zero fraction $0/1$, which
/// leaves its fractional sum unchanged. [`verify`] is oblivious to that, so the claims it outputs
/// for such a tree are claims on the padded witness
///
/// $$
/// N'(X_\text{pad}, X_\text{real}) = N(X_\text{real}) \cdot \text{eq}(0^\nu; X_\text{pad}),
/// \qquad
/// D'(X_\text{pad}, X_\text{real}) = 1 + \bigl( D(X_\text{real}) - 1 \bigr) \cdot
/// \text{eq}(0^\nu; X_\text{pad}),
/// $$
///
/// whose padding variables are the lowest ones. This divides out their equality weight and drops
/// them from the point, leaving the claims on $N$ and $D$.
///
/// # Arguments
///
/// * `fraction` - The claimed numerator and denominator evaluations of the padded witness.
/// * `point` - The reduced evaluation point, with the batch's selector coordinates already
///   stripped.
/// * `n_pad_vars` - How much depth this tree was padded by: the batch's layer count less the tree's
///   own.
///
/// # Preconditions
/// * `point.len() >= n_pad_vars`
///
/// # Panics
///
/// Panics if the padding coordinates' equality weight is zero, which requires one of them to equal
/// one. They are the verifier's own challenges, so no prover can induce this; it happens with
/// probability at most $\nu / |K|$.
pub fn unpad_leaf_claim<F: Field>(
	fraction: (F, F),
	point: &[F],
	n_pad_vars: usize,
) -> FracAddEvalClaim<F> {
	assert!(point.len() >= n_pad_vars); // precondition

	let pad_eq = point[..n_pad_vars]
		.iter()
		.map(|&coord| eq_one_var(F::ZERO, coord))
		.product::<F>();
	assert!(pad_eq != F::ZERO, "a padding coordinate equals one");
	let pad_eq_inv = pad_eq.invert_or_zero();

	let (num_eval, den_eval) = fraction;
	FracAddEvalClaim {
		num_eval: num_eval * pad_eq_inv,
		den_eval: F::ONE + (den_eval - F::ONE) * pad_eq_inv,
		point: point[n_pad_vars..].to_vec(),
	}
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
	#[error("sumcheck error: {0}")]
	Sumcheck(#[source] sumcheck::Error),
	#[error("transcript error: {0}")]
	Transcript(#[source] TranscriptError),
	#[error("verification error: {0}")]
	Verification(#[from] VerificationError),
}

impl From<sumcheck::Error> for Error {
	fn from(err: sumcheck::Error) -> Self {
		match err {
			sumcheck::Error::Verification(err) => VerificationError::Sumcheck(err).into(),
			_ => Error::Sumcheck(err),
		}
	}
}

impl From<TranscriptError> for Error {
	fn from(err: TranscriptError) -> Self {
		match err {
			TranscriptError::NotEnoughBytes => VerificationError::TranscriptIsEmpty.into(),
			_ => Error::Transcript(err),
		}
	}
}

impl From<crate::channel::Error> for Error {
	fn from(err: crate::channel::Error) -> Self {
		match err {
			crate::channel::Error::ProofEmpty => VerificationError::TranscriptIsEmpty.into(),
			crate::channel::Error::InvalidAssert => VerificationError::InvalidAssert.into(),
		}
	}
}

#[derive(Debug, thiserror::Error)]
pub enum VerificationError {
	#[error("sumcheck: {0}")]
	Sumcheck(#[from] sumcheck::VerificationError),
	#[error("incorrect layer fraction sum evaluation: {round}")]
	IncorrectLayerFractionSumEvaluation { round: usize },
	#[error("incorrect round evaluation: {round}")]
	IncorrectRoundEvaluation { round: usize },
	#[error("transcript is empty")]
	TranscriptIsEmpty,
	#[error("invalid assertion: value is not zero")]
	InvalidAssert,
}
