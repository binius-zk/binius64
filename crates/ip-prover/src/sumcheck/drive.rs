// Copyright 2026 The Binius Developers

//! The round-by-round driver shared by every sumcheck and MLE-check entry point.
//!
//! Both protocols run the identical round loop.
//! They differ only in which coefficient of the round polynomial is left off the wire.
//!
//! The two prover traits are unrelated, so a wrapper each adapts them to the shared loop:
//!
//! ```text
//!     a sumcheck prover  -> its wrapper  -.
//!                                          >-- one round loop
//!     an MLE-check prover -> its wrapper -'
//! ```
//!
//! A wrapper also carries the format its prover's polynomials go out in, so the two cannot be
//! crossed.

use binius_field::Field;
use binius_ip::{mlecheck, sumcheck::RoundCoeffs};

use super::{
	batch::BatchSumcheckOutput,
	common::{MleCheckProver, SumcheckProver},
	prove::ProveSingleOutput,
};
use crate::channel::IPProverChannel;

/// Which round-proof format the verifier expects.
///
/// - A round proof omits one coefficient of the round polynomial to save a field element per round.
/// - The verifier reconstructs the omitted coefficient from the round claim it already holds.
/// - The two protocols omit opposite ends, so sending the wrong one yields a rejected proof.
#[derive(Debug, Clone, Copy)]
pub enum RoundProofKind {
	/// Omits the highest-degree coefficient, recovered from `claim = R(0) + R(1)`.
	Sumcheck,
	/// Omits the constant term, recovered from `claim = (1 - alpha) * R(0) + alpha * R(1)`.
	MleCheck,
}

impl RoundProofKind {
	/// Compresses one round polynomial to this format and sends it to the verifier.
	fn send<F: Field>(self, coeffs: RoundCoeffs<F>, channel: &mut impl IPProverChannel<F>) {
		// Each arm drops the one coefficient its verifier can reconstruct.
		match self {
			Self::Sumcheck => channel.send_many(coeffs.truncate().coeffs()),
			Self::MleCheck => channel.send_many(mlecheck::RoundProof::truncate(coeffs).coeffs()),
		}
	}
}

/// A prover the round loop can drive, tagged with the format its round polynomials go out in.
///
/// The format is a constant on the type rather than an argument to the loop.
/// A caller could pass an argument naming a format its prover does not emit; it cannot pick a
/// wrong constant, because wrapping is the only way in and each wrapper sets its own.
pub trait RoundProver<F: Field> {
	/// The round-proof format this prover's round polynomials must be sent in.
	const ROUND_PROOF_KIND: RoundProofKind;

	/// The number of variables remaining, one round each.
	fn n_vars(&self) -> usize;

	/// Computes this round's polynomials, one per claim.
	fn execute(&mut self) -> Vec<RoundCoeffs<F>>;

	/// Binds the round's variable to the verifier challenge.
	fn fold(&mut self, challenge: F);

	/// Returns the multilinear evaluations at the challenge point.
	fn finish(self) -> Vec<F>;
}

/// Drives a [`SumcheckProver`] as a plain sumcheck.
///
/// The layout matches the wrapped prover, so mapping a vector reuses its allocation.
#[repr(transparent)]
pub struct SumcheckRounds<Prover>(pub Prover);

impl<F: Field, Prover: SumcheckProver<F>> RoundProver<F> for SumcheckRounds<Prover> {
	const ROUND_PROOF_KIND: RoundProofKind = RoundProofKind::Sumcheck;

	fn n_vars(&self) -> usize {
		self.0.n_vars()
	}

	fn execute(&mut self) -> Vec<RoundCoeffs<F>> {
		self.0.execute()
	}

	fn fold(&mut self, challenge: F) {
		self.0.fold(challenge)
	}

	fn finish(self) -> Vec<F> {
		self.0.finish()
	}
}

/// Drives an [`MleCheckProver`] as an MLE-check.
///
/// The layout serves the same purpose as it does for the plain sumcheck wrapper.
#[repr(transparent)]
pub struct MleCheckRounds<Prover>(pub Prover);

impl<F: Field, Prover: MleCheckProver<F>> RoundProver<F> for MleCheckRounds<Prover> {
	const ROUND_PROOF_KIND: RoundProofKind = RoundProofKind::MleCheck;

	fn n_vars(&self) -> usize {
		self.0.n_vars()
	}

	fn execute(&mut self) -> Vec<RoundCoeffs<F>> {
		self.0.execute()
	}

	fn fold(&mut self, challenge: F) {
		self.0.fold(challenge)
	}

	fn finish(self) -> Vec<F> {
		self.0.finish()
	}
}

/// Drives one prover of a single composition through all of its rounds.
///
/// # Panics
///
/// Panics if the prover returns more than one composition from a round.
pub fn single<F, Prover>(
	mut prover: Prover,
	channel: &mut impl IPProverChannel<F>,
) -> ProveSingleOutput<F>
where
	F: Field,
	Prover: RoundProver<F>,
{
	let n_vars = prover.n_vars();
	let mut challenges = Vec::with_capacity(n_vars);

	for _ in 0..n_vars {
		// This driver proves one composition, so the prover owes exactly one round polynomial.
		let mut round_coeffs_vec = prover.execute();
		assert_eq!(
			round_coeffs_vec.len(),
			1,
			"function expects prover to evaluate one composition, but it returned {} from execute()",
			round_coeffs_vec.len()
		);
		let round_coeffs = round_coeffs_vec.pop().expect("round_coeffs_vec.len() == 1");

		// Commit to the round polynomial, then sample the challenge that binds this variable.
		Prover::ROUND_PROOF_KIND.send(round_coeffs, channel);
		let challenge = channel.sample();
		challenges.push(challenge);
		prover.fold(challenge);
	}

	let multilinear_evals = prover.finish();
	ProveSingleOutput {
		multilinear_evals,
		challenges,
	}
}

/// Drives a group of provers that share a round count, drawing the batching coefficient.
///
/// An empty group returns without touching the channel, so it draws no coefficient.
///
/// # Panics
///
/// Panics if the provers do not all have the same number of rounds.
pub fn batch<F, Prover>(
	provers: impl IntoIterator<Item = Prover>,
	channel: &mut impl IPProverChannel<F>,
) -> BatchSumcheckOutput<F>
where
	F: Field,
	Prover: RoundProver<F>,
{
	let provers = provers.into_iter().collect::<Vec<_>>();

	let Some(first_prover) = provers.first() else {
		return BatchSumcheckOutput {
			challenges: Vec::new(),
			multilinear_evals: Vec::new(),
		};
	};

	let n_vars = first_prover.n_vars();

	assert!(
		provers.iter().all(|prover| prover.n_vars() == n_vars),
		"batched provers must have the same number of rounds"
	);

	// Random linear-combination coefficient for batching multiple claims.
	let batch_coeff = channel.sample();

	batch_with_coeff(provers, batch_coeff, channel)
}

/// Drives a group of provers with a batching coefficient the caller already drew.
///
/// The group's round polynomials are combined into one, so it costs a single round proof per
/// variable. An empty group runs zero rounds.
///
/// The coefficient must come from the same channel immediately before this call, so that the
/// transcript matches the one [`batch`] would have produced.
///
/// The group arrives materialized because every round walks all of it.
pub fn batch_with_coeff<F, Prover>(
	mut provers: Vec<Prover>,
	batch_coeff: F,
	channel: &mut impl IPProverChannel<F>,
) -> BatchSumcheckOutput<F>
where
	F: Field,
	Prover: RoundProver<F>,
{
	let n_vars = provers.first().map_or(0, |prover| prover.n_vars());

	let mut challenges = Vec::with_capacity(n_vars);
	for _ in 0..n_vars {
		let mut all_round_coeffs = Vec::new();

		for prover in &mut provers {
			// Each prover emits its round polynomial; we batch across provers.
			all_round_coeffs.extend(prover.execute());
		}

		// Horner-fold round polynomials into a single batched polynomial.
		let batched_round_coeffs = all_round_coeffs
			.into_iter()
			.rfold(RoundCoeffs::default(), |acc, coeffs| acc * batch_coeff + &coeffs);

		// Commit to the batched round polynomial, then sample the next challenge.
		Prover::ROUND_PROOF_KIND.send(batched_round_coeffs, channel);

		let challenge = channel.sample();
		challenges.push(challenge);

		// Fold all provers on the shared challenge to advance the state machine.
		for prover in &mut provers {
			prover.fold(challenge);
		}
	}

	let multilinear_evals = provers
		.into_iter()
		.map(|prover| prover.finish())
		.collect::<Vec<_>>();

	BatchSumcheckOutput {
		challenges,
		multilinear_evals,
	}
}
