// Copyright 2025-2026 The Binius Developers

use binius_field::{Field, PackedField};
use binius_math::{
	AsSlicesMut, FieldBuffer, FieldSliceMut, field_buffer::FieldBufferSplitMut,
	multilinear::fold::fold_highest_var_inplace,
};
use binius_utils::rayon::prelude::*;
use binius_verifier::protocols::sumcheck::RoundCoeffs;
use itertools::izip;

use super::error::Error;
use crate::protocols::sumcheck::{
	common::{MleCheckProver, SumcheckProver},
	gruen32::Gruen32,
	round_evals::RoundEvals2,
};

pub type FractionalBuffer<P> = (FieldBuffer<P>, FieldBuffer<P>);
#[derive(Debug, Clone)]
enum RoundCoeffsOrEvals<F: Field> {
	Coeffs([RoundCoeffs<F>; 2]),
	Evals([F; 2]),
}

// Prover for the fractional additional claims required in LogUp*. We keep numerators and
// denominators to be added in a single buffer respectively, with the assumption that the 2
// collections to be added are in either half.
pub struct FracAddMleCheckProver<P: PackedField> {
	// Parallel arrays: index 0 = numerator MLE evals, index 1 = denominator MLE evals.
	fraction_pairs: [FieldBuffer<P>; 2],
	// Alternates between the last round's polynomial coefficients and the folded evaluation
	// values.
	last_coeffs_or_evals: RoundCoeffsOrEvals<P::Scalar>,
	gruen32: Gruen32<P>,
}

impl<F: Field, P: PackedField<Scalar = F>> FracAddMleCheckProver<P> {
	/// Constructs a prover, given the multilinear polynomial evaluations (in pairs) and
	/// evaluation claims on the shared evaluation point.
	pub fn new(
		fraction: (FieldBuffer<P>, FieldBuffer<P>),
		eval_point: &[F],
		eval_claims: [F; 2],
	) -> Result<Self, Error> {
		let n_vars = eval_point.len();

		let (num, den) = fraction;
		// One extra variable for the numerator/denominator selector bit.
		if num.log_len() != n_vars + 1 || den.log_len() != n_vars + 1 {
			return Err(Error::MultilinearSizeMismatch);
		}

		let last_coeffs_or_evals = RoundCoeffsOrEvals::Evals(eval_claims);

		let gruen32 = Gruen32::new(eval_point);

		let fraction_pairs = [num, den];
		Ok(Self {
			fraction_pairs,
			last_coeffs_or_evals,
			gruen32,
		})
	}
}

impl<F, P> MleCheckProver<F> for FracAddMleCheckProver<P>
where
	F: Field,
	P: PackedField<Scalar = F>,
{
	// Expose the evaluation point so wrappers can lift this MLE-check prover into sumcheck.
	fn eval_point(&self) -> &[F] {
		self.gruen32.eval_point()
	}
}

impl<F, P> SumcheckProver<F> for FracAddMleCheckProver<P>
where
	F: Field,
	P: PackedField<Scalar = F>,
{
	fn n_vars(&self) -> usize {
		self.gruen32.n_vars_remaining()
	}

	fn n_claims(&self) -> usize {
		2
	}

	fn execute(&mut self) -> Result<Vec<RoundCoeffs<F>>, Error> {
		let RoundCoeffsOrEvals::Evals(sums) = &self.last_coeffs_or_evals else {
			return Err(Error::ExpectedFold);
		};

		// We need at least one variable to produce a round polynomial.
		assert!(self.n_vars() > 0);
		let n_vars = self.n_vars();
		let [num, den] = &mut self.fraction_pairs;

		let mut num_split = num.split_half_mut()?;
		let mut den_split = den.split_half_mut()?;

		// Fixed ordering expected by accumulate_chunk: num(0), num(1), den(0), den(1).
		let slices = split_and_truncate(&mut num_split, &mut den_split, n_vars);

		// Perform chunked summation for benefits detailed in bivariate_product_multi_mle.
		const MAX_CHUNK_VARS: usize = 8;
		// Keep enough vars per chunk to amortize eq-eval overhead, but never exceed n_vars - 1
		// because the highest variable is folded by the round polynomial.
		let chunk_vars = std::cmp::max(MAX_CHUNK_VARS, P::LOG_WIDTH).min(n_vars - 1);

		let packed_prime_evals: [RoundEvals2<P>; 2] = (0..1 << (n_vars - 1 - chunk_vars))
			.into_par_iter()
			.try_fold(
				|| [RoundEvals2::default(); 2],
				|mut packed_prime_evals: [RoundEvals2<P>; 2], chunk_index| -> Result<_, Error> {
					accumulate_chunk(
						&self.gruen32,
						&slices,
						chunk_vars,
						&mut packed_prime_evals,
						chunk_index,
					)?;
					Ok(packed_prime_evals)
				},
			)
			.try_reduce(
				|| [RoundEvals2::default(); 2],
				|lhs, rhs| Ok([lhs[0] + &rhs[0], lhs[1] + &rhs[1]]),
			)?;

		// These are MLE-check "prime" round polynomials; sumcheck wrappers apply the eq factor.
		let alpha = self.gruen32.next_coordinate();
		let round_coeffs = izip!(sums, packed_prime_evals)
			.map(|(&sum, packed_evals)| {
				let round_evals = packed_evals.sum_scalars(n_vars);
				round_evals.interpolate_eq(sum, alpha)
			})
			.collect::<Vec<_>>();

		self.last_coeffs_or_evals = RoundCoeffsOrEvals::Coeffs(
			round_coeffs.clone().try_into().expect("Will have length 2"),
		);
		Ok(round_coeffs)
	}

	fn fold(&mut self, challenge: F) -> Result<(), Error> {
		let RoundCoeffsOrEvals::Coeffs(prime_coeffs) = &self.last_coeffs_or_evals else {
			return Err(Error::ExpectedExecute);
		};

		// Folding substitutes the newest challenge into the highest variable.
		assert!(self.n_vars() > 0);

		let evals = [
			prime_coeffs[0].evaluate(challenge),
			prime_coeffs[1].evaluate(challenge),
		];

		let n_vars = self.n_vars();
		let [num, den] = &mut self.fraction_pairs;

		let mut num_split = num.split_half_mut()?;
		let mut den_split = den.split_half_mut()?;
		let mut multilinears = split_and_truncate(&mut num_split, &mut den_split, n_vars);

		for multilinear in &mut multilinears {
			fold_highest_var_inplace(multilinear, challenge)?
		}

		self.gruen32.fold(challenge)?;
		// After folding, we keep only the new evaluations for the next round.
		self.last_coeffs_or_evals = RoundCoeffsOrEvals::Evals(evals);
		Ok(())
	}

	fn finish(self) -> Result<Vec<F>, Error> {
		if self.n_vars() > 0 {
			let error = match self.last_coeffs_or_evals {
				RoundCoeffsOrEvals::Coeffs(_) => Error::ExpectedFold,
				RoundCoeffsOrEvals::Evals(_) => Error::ExpectedExecute,
			};

			return Err(error);
		}

		let multilinear_evals = self
			.fraction_pairs
			.into_iter()
			.flat_map(|multilinear| {
				let (lo, hi) = multilinear.split_half_ref().expect("Should have 2 values");
				[lo.get(0), hi.get(0)]
			})
			.collect();

		Ok(multilinear_evals)
	}
}

fn accumulate_chunk<P: PackedField>(
	gruen32: &Gruen32<P>,
	fraction_pairs: &[FieldSliceMut<P>; 4],
	chunk_vars: usize,
	packed_prime_evals: &mut [RoundEvals2<P>; 2],
	chunk_index: usize,
) -> Result<(), Error> {
	let eq_chunk = gruen32.eq_expansion().chunk(chunk_vars, chunk_index)?;

	let splits = fraction_pairs
		.iter()
		.map(|slice| slice.split_half_ref())
		.collect::<Result<Vec<_>, _>>()?;

	let chunks = splits
		.iter()
		.flat_map(|(lo, hi)| {
			[
				lo.chunk(chunk_vars, chunk_index),
				hi.chunk(chunk_vars, chunk_index),
			]
		})
		.collect::<Result<Vec<_>, _>>()?;
	// Ordering: [num_a, den_a, num_b, den_b] × {low, high} chunks.

	let [
		evals_num_a_0_chunk,
		evals_num_a_1_chunk,
		evals_num_b_0_chunk,
		evals_num_b_1_chunk,
		evals_den_a_0_chunk,
		evals_den_a_1_chunk,
		evals_den_b_0_chunk,
		evals_den_b_1_chunk,
	]: [FieldBuffer<P, _>; 8] = chunks
		.try_into()
		.expect(
			"The destructuring contains the high and low chunk slices for each of the 4 MLES, resulting in 8 slices total"
		);

	for (
		&eq_i,
		&evals_num_a_0_i,
		&evals_num_a_1_i,
		&evals_den_a_0_i,
		&evals_den_a_1_i,
		&evals_num_b_0_i,
		&evals_num_b_1_i,
		&evals_den_b_0_i,
		&evals_den_b_1_i,
	) in izip!(
		eq_chunk.as_ref(),
		evals_num_a_0_chunk.as_ref(),
		evals_num_a_1_chunk.as_ref(),
		evals_den_a_0_chunk.as_ref(),
		evals_den_a_1_chunk.as_ref(),
		evals_num_b_0_chunk.as_ref(),
		evals_num_b_1_chunk.as_ref(),
		evals_den_b_0_chunk.as_ref(),
		evals_den_b_1_chunk.as_ref()
	) {
		// Infinity evals are computed by M(∞) = M(0) + M(1) for each multilinear.
		let evals_num_a_inf_i = evals_num_a_0_i + evals_num_a_1_i;
		let evals_den_a_inf_i = evals_den_a_0_i + evals_den_a_1_i;
		let evals_num_b_inf_i = evals_num_b_0_i + evals_num_b_1_i;
		let evals_den_b_inf_i = evals_den_b_0_i + evals_den_b_1_i;

		// Numerator composition: a0/b0 + a1/b1 => a0*b1 + a1*b0.
		let num_1_i = evals_num_a_1_i * evals_den_b_1_i + evals_num_b_1_i * evals_den_a_1_i;
		let num_inf_i =
			evals_num_a_inf_i * evals_den_b_inf_i + evals_num_b_inf_i * evals_den_a_inf_i;

		// Denominator composition: b0*b1.
		let den_1_i = evals_den_a_1_i * evals_den_b_1_i;
		let den_inf_i = evals_den_a_inf_i * evals_den_b_inf_i;

		// Accumulate eq-weighted round evals for numerator (0) and denominator (1).
		packed_prime_evals[0].y_1 += eq_i * num_1_i;
		packed_prime_evals[0].y_inf += eq_i * num_inf_i;
		packed_prime_evals[1].y_1 += eq_i * den_1_i;
		packed_prime_evals[1].y_inf += eq_i * den_inf_i;
	}

	Ok(())
}

fn split_and_truncate<'a, P: PackedField>(
	num_split: &'a mut FieldBufferSplitMut<P, &mut [P]>,
	den_split: &'a mut FieldBufferSplitMut<P, &mut [P]>,
	n_vars: usize,
) -> [FieldSliceMut<'a, P>; 4] {
	let [mut num_a, mut num_b] = num_split.as_slices_mut();
	let [mut den_a, mut den_b] = den_split.as_slices_mut();

	num_a.truncate(n_vars);
	num_b.truncate(n_vars);
	den_a.truncate(n_vars);
	den_b.truncate(n_vars);
	// Fixed ordering expected by accumulate_chunk: num(0), num(1), den(0), den(1).
	[num_a, num_b, den_a, den_b]
}

#[cfg(test)]
mod tests {
	use binius_field::arch::{OptimalB128, OptimalPackedB128};
	use binius_math::{
		FieldBuffer,
		multilinear::{eq::eq_ind, evaluate::evaluate},
		test_utils::{random_field_buffer, random_scalars},
	};
	use binius_transcript::ProverTranscript;
	use binius_verifier::{config::StdChallenger, protocols::sumcheck::batch_verify};
	use itertools::{Itertools, izip};
	use rand::{SeedableRng, prelude::StdRng};

	use super::*;
	use crate::protocols::sumcheck::{MleToSumCheckDecorator, batch::batch_prove};

	fn test_frac_add_sumcheck_prove_verify<F, P>(
		prover: MleToSumCheckDecorator<F, FracAddMleCheckProver<P>>,
		eval_claims: [F; 2],
		eval_point: &[F],
		num: FieldBuffer<P>,
		den: FieldBuffer<P>,
	) where
		F: Field,
		P: PackedField<Scalar = F>,
	{
		let n_vars = prover.n_vars();
		let (num_a, num_b) = num.split_half_ref().unwrap();
		let (den_a, den_b) = den.split_half_ref().unwrap();
		// Run the proving protocol
		let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
		let output = batch_prove(vec![prover], &mut prover_transcript).unwrap();

		assert_eq!(output.multilinear_evals.len(), 1);
		let prover_evals = output.multilinear_evals[0].clone();

		// Write the multilinear evaluations to the transcript
		prover_transcript
			.message()
			.write_scalar_slice(&prover_evals);

		// Convert to verifier transcript and run verification
		let mut verifier_transcript = prover_transcript.into_verifier();
		let sumcheck_output =
		// Degree 3 because quadratic prime polynomials are multiplied by a linear eq term.
		batch_verify(n_vars, 3, &eval_claims, &mut verifier_transcript).unwrap();

		// The prover binds variables from high to low, but evaluate expects them from low to high
		let mut reduced_eval_point = sumcheck_output.challenges.clone();
		reduced_eval_point.reverse();

		// Read the multilinear evaluations from the transcript
		let multilinear_evals: Vec<F> = verifier_transcript.message().read_vec(4).unwrap();

		// Evaluate the equality indicator
		let eq_ind_eval = eq_ind(eval_point, &reduced_eval_point);

		// Check that the original multilinears evaluate to the claimed values at the challenge
		// point
		let eval_num_a = evaluate(&num_a, &reduced_eval_point).unwrap();
		let eval_den_a = evaluate(&den_a, &reduced_eval_point).unwrap();
		let eval_num_b = evaluate(&num_b, &reduced_eval_point).unwrap();
		let eval_den_b = evaluate(&den_b, &reduced_eval_point).unwrap();

		assert_eq!(
			eval_num_a, multilinear_evals[0],
			"Numerator A should evaluate to the first claimed evaluation"
		);

		assert_eq!(
			eval_num_b, multilinear_evals[1],
			"Numerator B should evaluate to the second claimed evaluation"
		);
		assert_eq!(
			eval_den_a, multilinear_evals[2],
			"Denominator A should evaluate to the third claimed evaluation"
		);

		assert_eq!(
			eval_den_b, multilinear_evals[3],
			"Denominator B should evaluate to the fourth claimed evaluation"
		);

		// Check that the batched evaluation matches the sumcheck output
		// Sumcheck wraps the prime polynomial with an eq factor, so include eq_ind_eval here.
		let numerator_eval = (eval_num_a * eval_den_b + eval_num_b * eval_den_a) * eq_ind_eval;
		let denominator_eval = (eval_den_a * eval_den_b) * eq_ind_eval;
		let batched_eval = numerator_eval + denominator_eval * sumcheck_output.batch_coeff;

		assert_eq!(
			batched_eval, sumcheck_output.eval,
			"Batched evaluation should equal the reduced evaluation"
		);

		// Also verify the challenges match what the prover saw
		let mut prover_challenges = output.challenges.clone();
		prover_challenges.reverse();
		assert_eq!(
			prover_challenges, sumcheck_output.challenges,
			"Prover and verifier challenges should match"
		);
	}

	#[test]
	fn test_frac_add_sumcheck() {
		type F = OptimalB128;
		type P = OptimalPackedB128;

		let n_vars = 8;
		let mut rng = StdRng::seed_from_u64(0);

		let num = random_field_buffer::<P>(&mut rng, n_vars + 1);
		let den = random_field_buffer::<P>(&mut rng, n_vars + 1);
		let (num_a, num_b) = num.split_half_ref().unwrap();
		let (den_a, den_b) = den.split_half_ref().unwrap();

		let numerator_values =
			izip!(num_a.as_ref(), den_a.as_ref(), num_b.as_ref(), den_b.as_ref())
				.map(|(&num_a, &den_a, &num_b, &den_b)| num_a * den_b + num_b * den_a)
				.collect_vec();

		let denominator_values = izip!(den_a.as_ref(), den_b.as_ref())
			.map(|(&den_a, &den_b)| den_a * den_b)
			.collect_vec();

		let numerator_buffer = FieldBuffer::new(n_vars, numerator_values).unwrap();
		let denominator_buffer = FieldBuffer::new(n_vars, denominator_values).unwrap();

		let eval_point = random_scalars::<F>(&mut rng, n_vars);
		// Claims are at the original eval_point; verifier handles challenge ordering separately.
		let eval_claims = [
			evaluate(&numerator_buffer, &eval_point).unwrap(),
			evaluate(&denominator_buffer, &eval_point).unwrap(),
		];

		let frac_prover =
			FracAddMleCheckProver::new((num.clone(), den.clone()), &eval_point, eval_claims)
				.unwrap();

		// Wrap the MLE-check prover so it emits sumcheck-compatible round polynomials.
		let prover = MleToSumCheckDecorator::new(frac_prover);

		test_frac_add_sumcheck_prove_verify(prover, eval_claims, &eval_point, num, den);
	}
}
