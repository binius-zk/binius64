// Copyright 2025-2026 The Binius Developers

//! Reduction from the products over the sumcubes of a multilinear to a multilinear evaluation.
//!
//! The reduction input is a multilinear $f(Z_0, \ldots, Z_{k-1}, X_0, \ldots, X_{n-1})$. The
//! product polynomial is the multilinear
//!
//! $$
//! p(X_0, \ldots, X_{n-1}) = \sum_{x \in B_n} \text{eq}(x ; X) \prod_{z \in B_k} f(z, x)
//! $$
//!
//! This protocol is a GKR-based protocol with $k$ sumcheck invocations. We define a sequence of
//! multilinears $p_0, \ldots, p_k$, where $p_k = f$ and for all $i < k$:
//!
//! $$
//! p_i(Z_0, \ldots, Z_{i-1}, X_0, \ldots, X_{n-1}) = \sum_{x \in B_n} \sum_{z \in B_i} \text{eq}(x
//! ; X) \text{eq}(z ; Z) p_{i+1}(z, 0, x) p_{i+1}(z, 1, x) $$

use binius_field::Field;
use binius_math::{line::extrapolate_line, multilinear::eq::eq_one_var};
use binius_transcript::Error as TranscriptError;

// Re-export MultilinearEvalClaim from crate root for backward compatibility
pub use crate::MultilinearEvalClaim;
use crate::{
	channel::IPVerifierChannel,
	mlecheck,
	sumcheck::{self, SumcheckOutput},
};

pub fn verify<F, C>(
	k: usize,
	claim: MultilinearEvalClaim<C::Elem>,
	channel: &mut C,
) -> Result<MultilinearEvalClaim<C::Elem>, Error>
where
	F: Field,
	C: IPVerifierChannel<F>,
{
	if k == 0 {
		return Ok(claim);
	}

	let MultilinearEvalClaim { eval, point } = claim;

	// Reduce p_i evaluation to two evaluations of p_{i+1}.
	let SumcheckOutput { eval, challenges } = mlecheck::verify(&point, 2, eval, channel)?;

	// Read evaluations of p_{i+1)(0, \ldots) and p_{i+1}(1, \ldots).
	let [eval_0, eval_1] = channel.recv_array()?;

	channel.assert_zero(eval_0.clone() * eval_1.clone() - eval)?;

	// Reduce evaluations of p_{i+1}(0, \ldots) and p_{i+1}(1, \ldots) to single eval at
	// p_{i+1}(r, \ldots).
	let r = channel.sample();

	let next_eval = extrapolate_line(eval_0, eval_1, r.clone());

	let mut next_point = challenges;
	next_point.reverse();
	next_point.push(r);

	verify(
		k - 1,
		MultilinearEvalClaim {
			eval: next_eval,
			point: next_point,
		},
		channel,
	)
}

/// Reduces a leaf claim on a one-padded witness to the claim on the witness itself.
///
/// A batched product check over trees of unequal depths lifts each shallow tree to the batch's
/// depth by filling `n_pad_vars` extra leaf positions with ones, which leaves its product
/// unchanged. [`verify`] is oblivious to that, so the claim it outputs for such a tree is a claim
/// on the padded witness
///
/// $$
/// M'(X_\text{pad}, X_\text{real}) = 1 + \bigl( M(X_\text{real}) - 1 \bigr) \cdot
/// \text{eq}(0^\nu; X_\text{pad}),
/// $$
///
/// whose padding variables are the lowest ones. This divides out their equality weight and drops
/// them from the point, leaving the claim on $M$.
///
/// # Arguments
///
/// * `eval` - The claimed evaluation of the padded witness.
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
	eval: F,
	point: &[F],
	n_pad_vars: usize,
) -> MultilinearEvalClaim<F> {
	assert!(point.len() >= n_pad_vars); // precondition

	let pad_eq = point[..n_pad_vars]
		.iter()
		.map(|&coord| eq_one_var(F::ZERO, coord))
		.product::<F>();
	assert!(pad_eq != F::ZERO, "a padding coordinate equals one");

	MultilinearEvalClaim {
		eval: F::ONE + (eval - F::ONE) * pad_eq.invert_or_zero(),
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
	#[error("incorrect round evaluation: {round}")]
	IncorrectRoundEvaluation { round: usize },
	#[error("transcript is empty")]
	TranscriptIsEmpty,
	#[error("invalid assertion: value is not zero")]
	InvalidAssert,
}
