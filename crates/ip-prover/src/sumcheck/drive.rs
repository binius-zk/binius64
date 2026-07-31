// Copyright 2026 The Binius Developers

//! The round-by-round driver shared by every sumcheck and MLE-check entry point.
//!
//! Both protocols run the identical round loop.
//! They differ only in which coefficient of the round polynomial is left off the wire.

use binius_field::Field;
use binius_ip::{mlecheck, sumcheck::RoundCoeffs};

use super::{batch::BatchSumcheckOutput, common::SumcheckProver, prove::ProveSingleOutput};
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

/// Drives one prover of a single composition through all of its rounds.
///
/// # Panics
///
/// Panics if the prover returns more than one composition from a round.
pub fn single<F: Field>(
	kind: RoundProofKind,
	mut prover: impl SumcheckProver<F>,
	channel: &mut impl IPProverChannel<F>,
) -> ProveSingleOutput<F> {
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
		kind.send(round_coeffs, channel);
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

/// Drives a group of provers that share a round count through all of their rounds.
///
/// The group's round polynomials are combined into one, so it costs a single round proof per
/// variable. An empty group is a no-op.
///
/// # Panics
///
/// Panics if the provers do not all have the same number of rounds.
pub fn batch<F, Prover>(
	kind: RoundProofKind,
	mut provers: Vec<Prover>,
	channel: &mut impl IPProverChannel<F>,
) -> BatchSumcheckOutput<F>
where
	F: Field,
	Prover: SumcheckProver<F>,
{
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
		kind.send(batched_round_coeffs, channel);

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

/// Sends each prover's evaluation claims to the verifier.
pub fn send_evals<F: Field>(
	output: &BatchSumcheckOutput<F>,
	channel: &mut impl IPProverChannel<F>,
) {
	for evals in &output.multilinear_evals {
		// Preserve per-prover ordering when emitting evaluation claims.
		channel.send_many(evals);
	}
}
