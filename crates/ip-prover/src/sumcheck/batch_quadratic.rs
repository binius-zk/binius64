// Copyright 2025-2026 The Binius Developers

use std::cmp::max;

use binius_field::{Field, PackedField};
use binius_ip::sumcheck::RoundCoeffs;
use binius_math::{
	AsSlicesMut, FieldBuffer, FieldSliceMut, multilinear::fold::fold_highest_var_inplace,
};
use binius_utils::rayon::prelude::*;
use itertools::{Itertools, izip};

use crate::sumcheck::{Error, common::SumcheckProver, round_evals::RoundEvals2};

/// Batch sumcheck prover for M quadratic compositions over N multilinears.
///
/// This prover runs a single sumcheck instance that amortizes the work of M independent
/// quadratic sumchecks by evaluating all compositions in one pass per round. It uses the
/// Karatsuba-style degree-2 interpolation trick (via evaluation at 1 and infinity) but keeps a
/// vector of sum claims and folds all multilinears in lockstep.
pub struct BatchQuadraticSumcheckProver<
	P: PackedField,
	Composition,
	InfinityComposition,
	const N: usize,
	const M: usize,
> {
	// Packed evaluations of the input multilinears; mutated in-place during folding.
	multilinears: Box<dyn AsSlicesMut<P, N> + Send>,
	// Full quadratic composition evaluated on the "x = 1" branch for each multilinear.
	composition: Composition,
	// Composition restricted to highest-degree terms for the "x = infinity" evaluation (Karatsuba).
	infinity_composition: InfinityComposition,
	// State machine storage: last round's sum claims (execute input) or current coeffs (fold input).
	last_coeffs_or_sum: RoundCoeffsOrSums<P::Scalar, M>,
	// Tracks the number of variables remaining in the sumcheck.
	n_vars_remaining: usize,
}

impl<F, P, Composition, InfinityComposition, const N: usize, const M: usize>
	BatchQuadraticSumcheckProver<P, Composition, InfinityComposition, N, M>
where
	F: Field,
	P: PackedField<Scalar = F>,
	Composition: Fn([P; N], &mut [P; M]) + Sync,
	InfinityComposition: Fn([P; N], &mut [P; M]) + Sync,
{
	pub fn new(
		mut multilinears: impl AsSlicesMut<P, N> + Send + 'static,
		composition: Composition,
		infinity_composition: InfinityComposition,
		sum_claims: [F; M],
	) -> Result<Self, Error> {
		assert!(N > 0 && M > 0);

		let mut slices = multilinears.as_slices_mut();
		let n_vars = slices[0].log_len();
		for multilinear in &mut slices {
			if multilinear.log_len() != n_vars {
				return Err(Error::MultilinearSizeMismatch);
			}
		}

		Ok(Self {
			multilinears: Box::new(multilinears),
			composition,
			infinity_composition,
			last_coeffs_or_sum: RoundCoeffsOrSums::Sums(sum_claims),
			n_vars_remaining: n_vars,
		})
	}

	/// Gets mutable slices of the multilinears, truncated to the current number of variables.
	fn multilinears_mut(&mut self) -> [FieldSliceMut<'_, P>; N] {
		let n_vars = self.n_vars_remaining;
		let mut slices = self.multilinears.as_slices_mut();
		for slice in &mut slices {
			slice.truncate(n_vars);
		}
		slices
	}
}

impl<F, P, Composition, InfinityComposition, const N: usize, const M: usize> SumcheckProver<F>
	for BatchQuadraticSumcheckProver<P, Composition, InfinityComposition, N, M>
where
	F: Field,
	P: PackedField<Scalar = F>,
	Composition: Fn([P; N], &mut [P; M]) + Sync,
	InfinityComposition: Fn([P; N], &mut [P; M]) + Sync,
{
	fn n_vars(&self) -> usize {
		self.n_vars_remaining
	}

	fn n_claims(&self) -> usize {
		M
	}

	fn execute(&mut self) -> Result<Vec<RoundCoeffs<F>>, Error> {
		let last_sums = match &self.last_coeffs_or_sum {
			RoundCoeffsOrSums::Sums(sums) => *sums,
			RoundCoeffsOrSums::Coeffs(_) => return Err(Error::ExpectedFold),
		};

		let n_vars_remaining = self.n_vars_remaining;
		assert!(n_vars_remaining > 0);

		let comp = &self.composition;
		let inf_comp = &self.infinity_composition;

		let mut multilinears = self.multilinears.as_slices_mut();
		for slice in &mut multilinears {
			slice.truncate(n_vars_remaining);
		}

		let (splits_0, splits_1) = multilinears
			.iter()
			.map(FieldBuffer::split_half_ref)
			.collect::<(Vec<_>, Vec<_>)>();

		const MAX_CHUNK_VARS: usize = 8;
		let chunk_vars = max(MAX_CHUNK_VARS, P::LOG_WIDTH).min(n_vars_remaining - 1);
		let chunk_count = 1 << (n_vars_remaining - 1 - chunk_vars);

		let packed_prime_evals = (0..chunk_count)
			.into_par_iter()
			.try_fold(
				|| [[P::default(); M]; 2],
				|mut packed_prime_evals, chunk_index| -> Result<_, Error> {
					let [mut y_1_scratch, mut y_inf_scratch] = [[P::default(); M]; 2];

					let splits_0_chunk = splits_0
						.iter()
						.map(|slice| slice.chunk(chunk_vars, chunk_index))
						.collect::<Vec<_>>();
					let splits_1_chunk = splits_1
						.iter()
						.map(|slice| slice.chunk(chunk_vars, chunk_index))
						.collect::<Vec<_>>();

					let [y_1, y_inf] = &mut packed_prime_evals;
					for idx in 0..splits_0_chunk[0].as_ref().len() {
						let mut evals_1 = [P::default(); N];
						let mut evals_inf = [P::default(); N];

						for i in 0..N {
							let lo_i = splits_0_chunk[i].as_ref()[idx];
							let hi_i = splits_1_chunk[i].as_ref()[idx];

							evals_1[i] = hi_i;
							evals_inf[i] = lo_i + hi_i;
						}

						comp(evals_1, &mut y_1_scratch);
						inf_comp(evals_inf, &mut y_inf_scratch);

						for i in 0..M {
							y_1[i] += y_1_scratch[i];
							y_inf[i] += y_inf_scratch[i];
						}
					}

					Ok(packed_prime_evals)
				},
			)
			.try_reduce(
				|| [[P::default(); M]; 2],
				|lhs, rhs| {
					let mut out = [[P::default(); M]; 2];
					for claim_idx in 0..M {
						out[0][claim_idx] = lhs[0][claim_idx] + rhs[0][claim_idx];
						out[1][claim_idx] = lhs[1][claim_idx] + rhs[1][claim_idx];
					}
					Ok(out)
				},
			)?;

		let round_coeffs = izip!(
			last_sums.iter().copied(),
			packed_prime_evals[0].iter().copied(),
			packed_prime_evals[1].iter().copied()
		)
		.map(|(sum, y_1, y_inf)| {
			let round_evals = RoundEvals2 { y_1, y_inf }.sum_scalars(n_vars_remaining);
			round_evals.interpolate(sum)
		})
		.collect::<Vec<_>>();

		self.last_coeffs_or_sum = RoundCoeffsOrSums::Coeffs(
			round_coeffs
				.clone()
				.try_into()
				.expect("Will have M elements."),
		);
		Ok(round_coeffs)
	}

	fn fold(&mut self, challenge: F) -> Result<(), Error> {
		let RoundCoeffsOrSums::Coeffs(prime_coeffs) = &self.last_coeffs_or_sum else {
			return Err(Error::ExpectedExecute);
		};

		assert!(
			self.n_vars() > 0,
			"n_vars is decremented in fold; \
			fold changes last_coeffs_or_sum to Sum variant; \
			fold only executes with Coeffs variant; \
			thus, n_vars should be > 0"
		);

		let sums = prime_coeffs
			.iter()
			.map(|coeffs| coeffs.evaluate(challenge))
			.collect_array()
			.expect("Will have size M");

		for multilinear in &mut self.multilinears_mut() {
			fold_highest_var_inplace(multilinear, challenge);
		}

		self.n_vars_remaining -= 1;
		self.last_coeffs_or_sum = RoundCoeffsOrSums::Sums(sums);
		Ok(())
	}

	fn finish(mut self) -> Result<Vec<F>, Error> {
		if self.n_vars() > 0 {
			let error = match self.last_coeffs_or_sum {
				RoundCoeffsOrSums::Coeffs(_) => Error::ExpectedFold,
				RoundCoeffsOrSums::Sums(_) => Error::ExpectedExecute,
			};
			return Err(error);
		}

		let multilinear_evals = self
			.multilinears_mut()
			.into_iter()
			.map(|multilinear| multilinear.get(0))
			.collect();
		Ok(multilinear_evals)
	}
}

#[derive(Debug, Clone)]
enum RoundCoeffsOrSums<F: Field, const M: usize> {
	Coeffs([RoundCoeffs<F>; M]),
	Sums([F; M]),
}

#[cfg(test)]
mod tests {
	use std::array;

	use binius_ip::sumcheck::batch_verify;
	use binius_math::{
		FieldBuffer,
		multilinear::evaluate::evaluate,
		test_utils::{Packed128b, random_field_buffer},
		univariate::evaluate_univariate,
	};
	use binius_transcript::{ProverTranscript, fiat_shamir::HasherChallenger};
	type StdChallenger = HasherChallenger<sha2::Sha256>;
	use itertools::Itertools;
	use rand::{SeedableRng, rngs::StdRng};

	use super::*;
	use crate::sumcheck::batch::batch_prove;

	const N: usize = 3;
	const M: usize = 2;

	fn comp_0<P: PackedField>([a, b, c]: [P; N]) -> P {
		a * b - c
	}

	fn comp_1<P: PackedField>([a, b, c]: [P; N]) -> P {
		(a + b) * c
	}

	fn comp_0_scalar<F: Field>([a, b, c]: [F; N]) -> F {
		a * b - c
	}

	fn comp_1_scalar<F: Field>([a, b, c]: [F; N]) -> F {
		(a + b) * c
	}

	fn batch_comp<P: PackedField>(evals: [P; N], out: &mut [P; M]) {
		let [a, b, c] = evals;
		out[0] = a * b - c;
		out[1] = (a + b) * c;
	}

	fn batch_inf_comp<P: PackedField>(evals: [P; N], out: &mut [P; M]) {
		let [a, b, c] = evals;
		out[0] = a * b;
		out[1] = (a + b) * c;
	}

	fn sum_claims<F, P>(multilinears: &[FieldBuffer<P>; N]) -> [F; M]
	where
		F: Field,
		P: PackedField<Scalar = F>,
	{
		let n_vars = multilinears[0].log_len();
		array::from_fn(|claim_idx| {
			let composite_vals = (0..1 << n_vars.saturating_sub(P::LOG_WIDTH))
				.map(|i| {
					let evals = array::from_fn(|j| multilinears[j].as_ref()[i]);
					match claim_idx {
						0 => comp_0(evals),
						1 => comp_1(evals),
						_ => unreachable!("M is fixed to 2"),
					}
				})
				.collect_vec();
			let composite_buffer = FieldBuffer::new(n_vars, composite_vals);
			composite_buffer.iter_scalars().sum()
		})
	}

	#[test]
	fn test_batch_quadratic_sumcheck_prove_verify() {
		type P = Packed128b;
		type F = <P as PackedField>::Scalar;

		let n_vars = 8;
		let mut rng = StdRng::seed_from_u64(0);

		let multilinears: [FieldBuffer<P>; N] =
			array::from_fn(|_| random_field_buffer::<P>(&mut rng, n_vars));
		let sum_claims = sum_claims::<F, P>(&multilinears);

		let prover = BatchQuadraticSumcheckProver::new(
			multilinears.clone(),
			batch_comp::<P>,
			batch_inf_comp::<P>,
			sum_claims,
		)
		.unwrap();

		let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
		let output = batch_prove(vec![prover], &mut prover_transcript).unwrap();

		assert_eq!(output.multilinear_evals.len(), 1);
		let prover_evals = output.multilinear_evals[0].clone();

		prover_transcript
			.message()
			.write_scalar_slice(&prover_evals);

		let mut verifier_transcript = prover_transcript.into_verifier();
		let sumcheck_output =
			batch_verify::<F, _>(n_vars, 2, &sum_claims, &mut verifier_transcript).unwrap();

		let mut eval_point = sumcheck_output.challenges.clone();
		eval_point.reverse();

		let expected_evals: [F; N] = array::from_fn(|i| evaluate(&multilinears[i], &eval_point));
		assert_eq!(expected_evals.as_slice(), prover_evals.as_slice());

		let composed_evals = [
			comp_0_scalar::<F>(expected_evals),
			comp_1_scalar::<F>(expected_evals),
		];
		let combined_eval = evaluate_univariate(&composed_evals, sumcheck_output.batch_coeff);
		assert_eq!(combined_eval, sumcheck_output.eval);
	}
}
