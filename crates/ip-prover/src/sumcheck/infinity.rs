// Copyright 2026 The Binius Developers

//! Prover for the sumcheck over the infinity hypercube $\{0, \infty\}^n$.
//!
//! See [`binius_ip::sumcheck::infinity`] for the protocol and the verifier.

use binius_compute::BufferData;
use binius_field::{Field, PackedField};
use binius_ip::sumcheck::RoundCoeffs;
use binius_math::{
	FieldBuffer,
	inner_product::inner_product_par,
	multilinear::hypercube::{InfCube, fold_highest_var_inplace},
};

use super::batch::BatchSumcheckOutput;
use crate::channel::IPProverChannel;

/// Proves a batch of claims $\sum_{x \in \{0, \infty\}^n} u^\infty(x) \cdot w^\infty(x)$, one per
/// pair of monomial-coefficient tables $(u, w)$.
///
/// The claimed sum of each pair is $\langle u, w \rangle$. The pairs may have different variable
/// counts; the batch runs on the largest, binding the top variable each round. A pair on `k`
/// variables sits out the first rounds, contributing its unchanged claim as a constant round
/// polynomial, and closes at the low `k` coordinates of the evaluation point.
///
/// Verify with [`binius_ip::sumcheck::infinity::batch_verify`] at degree 2.
///
/// Each prover's evaluations are $[u^\infty(r), w^\infty(r)]$, the multilinears with monomial
/// coefficients $u$, $w$ at the claim's challenge point $r$. The evaluations are not written to the
/// channel.
///
/// ## Preconditions
///
/// * the two tables of a pair must have the same number of variables
pub fn batch_prove<F, P, Data>(
	pairs: Vec<[FieldBuffer<P, Data>; 2]>,
	channel: &mut impl IPProverChannel<F>,
) -> BatchSumcheckOutput<F>
where
	F: Field,
	P: PackedField<Scalar = F>,
	Data: BufferData<P>,
{
	let mut provers = pairs
		.into_iter()
		.map(ProductProver::new)
		.collect::<Vec<_>>();
	let n_vars = provers.iter().map(|p| p.n_vars()).max().unwrap_or(0);

	let batch_coeff = channel.sample();

	let mut challenges = Vec::with_capacity(n_vars);
	for round in 0..n_vars {
		let n_vars_remaining = n_vars - round;
		let round_coeffs = provers
			.iter()
			.map(|prover| {
				if prover.n_vars() == n_vars_remaining {
					prover.round_coeffs()
				} else {
					// The summand does not depend on this variable, so R(X) is the constant claim.
					RoundCoeffs(vec![prover.claim])
				}
			})
			.collect::<Vec<_>>();

		// The largest prover is always active, so the batch has all degree + 1 coefficients.
		let batched = RoundCoeffs::batch(round_coeffs.clone(), &batch_coeff);
		channel.send_many(batched.truncate().coeffs());

		let challenge = channel.sample();
		challenges.push(challenge);
		for (prover, coeffs) in provers.iter_mut().zip(round_coeffs) {
			if prover.n_vars() == n_vars_remaining {
				prover.fold(&coeffs, challenge);
			}
		}
	}

	BatchSumcheckOutput {
		challenges,
		multilinear_evals: provers.into_iter().map(ProductProver::finish).collect(),
	}
}

/// The state of one claim $\sum_x u^\infty(x) \cdot w^\infty(x) = s$.
///
/// [`batch_prove`] drives these with random batching weights. A caller that needs another
/// combination of the round polynomials drives them itself.
pub struct ProductProver<P: PackedField, Data: BufferData<P>> {
	tables: [FieldBuffer<P, Data>; 2],
	claim: P::Scalar,
}

impl<F, P, Data> ProductProver<P, Data>
where
	F: Field,
	P: PackedField<Scalar = F>,
	Data: BufferData<P>,
{
	/// Creates the prover of $\langle u, w \rangle$ for the tables `[u, w]`.
	///
	/// ## Preconditions
	///
	/// * the two tables must have the same number of variables
	pub fn new(tables: [FieldBuffer<P, Data>; 2]) -> Self {
		let [u, w] = &tables;
		assert_eq!(
			u.log_len(),
			w.log_len(),
			"precondition: tables must have equal variable counts"
		);
		let claim = inner_product_par(u, w);
		Self { tables, claim }
	}

	/// The number of variables not yet bound.
	pub const fn n_vars(&self) -> usize {
		self.tables[0].log_len()
	}

	/// The current claim $s$, the sum over the unbound variables.
	pub const fn claim(&self) -> F {
		self.claim
	}

	/// The round polynomial in the top variable $X$.
	///
	/// With $u = u_0 + X u_1$ split on the top variable, and likewise $w$,
	///
	/// $$
	/// R(X) = \langle u_0, w_0 \rangle
	///     + X (\langle u_0, w_1 \rangle + \langle u_1, w_0 \rangle)
	///     + X^2 \langle u_1, w_1 \rangle
	/// $$
	///
	/// and $R(0) = s - R(\infty)$ comes from the claim.
	pub fn round_coeffs(&self) -> RoundCoeffs<F> {
		let [u, w] = &self.tables;
		let (u_0, u_1) = u.split_half();
		let (w_0, w_1) = w.split_half();

		let r_inf = inner_product_par(&u_1, &w_1);
		let r_mid = inner_product_par(&u_0, &w_1) + inner_product_par(&u_1, &w_0);
		RoundCoeffs(vec![self.claim - r_inf, r_mid, r_inf])
	}

	/// Binds the top variable to `challenge` $r$: $u'(v) = u(v, 0) + r \cdot u(v, 1)$.
	pub fn fold(&mut self, round_coeffs: &RoundCoeffs<F>, challenge: F) {
		self.claim = round_coeffs.evaluate(&challenge);

		for table in &mut self.tables {
			fold_highest_var_inplace::<InfCube, _, _>(table, challenge);
		}
	}

	/// The evaluations $[u^\infty(r), w^\infty(r)]$ once every variable is bound.
	pub fn finish(self) -> Vec<F> {
		self.tables.iter().map(|table| table.get(0)).collect()
	}
}

#[cfg(test)]
mod tests {
	use binius_field::arch::{OptimalB128, OptimalPackedB128};
	use binius_ip::sumcheck::infinity::batch_verify;
	use binius_math::{
		multilinear::hypercube::eq_ind_partial_eval, test_utils::random_field_buffer,
		univariate::evaluate_univariate,
	};
	use binius_transcript::{ProverTranscript, VerifierTranscript, fiat_shamir::HasherChallenger};
	use rand::prelude::*;

	use super::*;

	type F = OptimalB128;
	type P = OptimalPackedB128;
	type StdChallenger = HasherChallenger<sha2::Sha256>;

	const N_VARS: [usize; 5] = [5, 0, 3, 5, 1];

	/// The multilinear with monomial coefficients `u`, evaluated at `point`.
	fn monomial_eval(u: &FieldBuffer<P>, point: &[F]) -> F {
		inner_product_par(u, &eq_ind_partial_eval::<InfCube, P>(point))
	}

	fn random_pairs() -> Vec<[FieldBuffer<P>; 2]> {
		let mut rng = StdRng::seed_from_u64(0);
		N_VARS
			.iter()
			.map(|&n| {
				[
					random_field_buffer(&mut rng, n),
					random_field_buffer(&mut rng, n),
				]
			})
			.collect()
	}

	/// Proves the batch and returns the sums and the proof bytes.
	fn prove(pairs: &[[FieldBuffer<P>; 2]]) -> (Vec<F>, Vec<u8>) {
		let sums = pairs.iter().map(|[u, w]| inner_product_par(u, w)).collect();
		let mut transcript = ProverTranscript::new(StdChallenger::default());
		batch_prove(pairs.to_vec(), &mut transcript);
		(sums, transcript.finalize())
	}

	#[test]
	fn batch_round_trip_over_mixed_sizes() {
		let pairs = random_pairs();
		let n_vars = *N_VARS.iter().max().unwrap();
		let sums = pairs
			.iter()
			.map(|[u, w]| inner_product_par(u, w))
			.collect::<Vec<_>>();

		let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
		let output = batch_prove(pairs.clone(), &mut prover_transcript);

		let mut verifier_transcript = prover_transcript.into_verifier();
		let verified = batch_verify(n_vars, 2, &sums, &mut verifier_transcript).unwrap();
		assert_eq!(output.challenges, verified.challenges);

		// Challenges bind the top variable first, so the point reads them in reverse.
		let mut point = verified.challenges.clone();
		point.reverse();

		let mut composed = Vec::new();
		for ([u, w], evals) in pairs.iter().zip(&output.multilinear_evals) {
			// A claim on k variables closes at the low k coordinates.
			let point = &point[..u.log_len()];
			assert_eq!(evals[0], monomial_eval(u, point));
			assert_eq!(evals[1], monomial_eval(w, point));
			composed.push(evals[0] * evals[1]);
		}
		assert_eq!(evaluate_univariate(&composed, &verified.batch_coeff), verified.eval);
	}

	#[test]
	fn tampered_round_polynomial_is_rejected() {
		let pairs = random_pairs();
		let n_vars = *N_VARS.iter().max().unwrap();
		let (sums, mut proof) = prove(&pairs);

		// Flip a bit of the first round's constant coefficient.
		proof[0] ^= 1;

		let mut transcript = VerifierTranscript::new(StdChallenger::default(), proof);
		let verified = batch_verify(n_vars, 2, &sums, &mut transcript).unwrap();

		let mut point = verified.challenges.clone();
		point.reverse();
		let composed = pairs
			.iter()
			.map(|[u, w]| {
				let point = &point[..u.log_len()];
				monomial_eval(u, point) * monomial_eval(w, point)
			})
			.collect::<Vec<_>>();
		assert_ne!(evaluate_univariate(&composed, &verified.batch_coeff), verified.eval);
	}
}
