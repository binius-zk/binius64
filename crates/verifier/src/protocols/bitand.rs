// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use std::iter::{self};

use binius_field::{BinaryField, field::FieldOps};
use binius_ip::channel::IPVerifierChannel;
use binius_math::{BinarySubspace, univariate::EvaluationDomain};

use super::rerand::RerandOutput;
use crate::Error;

/// log2 size of the univariate domain
pub const SKIPPED_VARS: usize = binius_core::Word::LOG_BITS;

/// Size of the univariate domain
pub const ROWS_PER_HYPERCUBE_VERTEX: usize = 1 << SKIPPED_VARS;

/// Output of the univariate skip.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnivariateSkipOutput<F> {
	/// The univariate challenge `z` sampled for the bit-index variable.
	pub z_challenge: F,
	/// The univariate polynomial at `z`, `g(z)`: the claim the multilinear rounds prove.
	pub claim: F,
}

/// Output of the whole BitAnd reduction: the univariate skip, then the batched sumcheck.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AndCheckOutput<F> {
	/// The univariate challenge `z` sampled for the bit-index variable.
	pub z_challenge: F,
	/// The batched sumcheck's evaluation claims. See [`super::rerand`].
	pub rerand: RerandOutput<F>,
}

/// Verifies the univariate skip of the AND constraint reduction.
///
/// Note: Following section 4.4 of the Binius64 writeup, Z is the bit index within a word, and X is
/// the word index
///
/// Let our oblong polynomials be A(Z, X₀, ...), B(Z, X₀, ...), and C(Z, X₀, ...)
///
/// Let our zerocheck challenges be (r₀, ...)
///
/// The reduction checks A·B-C = 0 on every row. That holds if and only if, for all Z, the
/// multilinear extension of A·B-C evaluates to zero at a random point (Z,r₀,...,rₙ₋₁), up to some
/// negligible error probability.
///
/// The prover sends a univariate polynomial R₀(Z) that encodes the sum:
///
/// R₀(Z) = ∑_{X₀,...,Xₙ₋₁ ∈ {0,1}} (A(Z,X₀,...,Xₙ₋₁)·B(Z,X₀,...,Xₙ₋₁) -
/// C(Z,X₀,...,Xₙ₋₁))·eq(X₀,...,Xₙ₋₁; r₀,...,rₙ₋₁)
///
/// Z ranges over a univariate domain of size 2^(SKIPPED_VARS + 1). The polynomial R₀(Z) has degree
/// at most 2*(|D| - 1). The prover only sends evaluations on the upper half of the domain, since
/// R₀(Z) = 0 on the base domain when all AND constraints are satisfied.
///
/// The verifier samples a challenge z for Z. The claim R₀(z) is then proven by the batched
/// sumcheck in [`super::rerand`], over the remaining variables X₀,...,Xₙ₋₁.
pub fn verify_univariate_skip<F, C>(
	channel: &mut C,
	domain: &BinarySubspace<F>,
) -> Result<UnivariateSkipOutput<C::Elem>, Error>
where
	F: BinaryField,
	C: IPVerifierChannel<F>,
	// This bound is necessary to make Barycentric evaluation constants symbolic
	C::Elem: From<F>,
{
	let univariate_message_coeffs_ext_domain = channel.recv_many(ROWS_PER_HYPERCUBE_VERTEX)?;

	let univariate_message_coeffs = iter::chain(
		iter::repeat_n(C::Elem::zero(), ROWS_PER_HYPERCUBE_VERTEX),
		univariate_message_coeffs_ext_domain,
	)
	.collect::<Vec<_>>();

	let z_challenge = channel.sample();
	let claim = domain.extrapolate(&univariate_message_coeffs, &z_challenge);

	Ok(UnivariateSkipOutput { z_challenge, claim })
}
