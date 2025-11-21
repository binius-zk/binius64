// Copyright 2023-2025 Irreducible Inc.

use binius_field::{Field, PackedField};
use binius_math::{FieldBuffer, multilinear::fold::fold_highest_var_inplace};
use binius_utils::rayon::prelude::*;
use binius_verifier::protocols::sumcheck::RoundCoeffs;

use super::{common::SumcheckProver, error::Error, gruen32::Gruen32, round_evals::RoundEvals2};
use crate::protocols::sumcheck::common::MleCheckProver;

/// A [`SumcheckProver`] implementation that reduces an evaluation claim on a multilinear extension
/// of the product of two multilinears to evaluation claims on said multilinears. We call such
/// reductions Mlechecks.
///
/// ## Mathematical Definition
/// * $n \in N$ - number of variables in multilinear polynomials
/// * $A, B \in F\[x\], x = \(x_1, \ldots, x_n\)$ - multilinears being multiplied
/// * $(\widetilde{AB})\[x\] = y$ - evaluation claim on the product MLE
///
/// The claim is equivalent to $P(x) = \sum_{v \in B} \widetilde{eq}(v, x) A(v) B(v) = y$, and the
/// reduction can be achieved by sumchecking the latter degree-3 composition. The paper [Gruen24],
/// however, describes a way to partition the $\widetilde{eq}(v, x)$ into three parts in round $j
/// \in 1, \ldots, n$ during specialization of variable $v_{n-j+1}$, with $j-1$ challenges
/// $\alpha_i$ already sampled:
///
/// $$ \widetilde{eq}(x_{n-j+2}, \ldots, x_n; \alpha_{j-1}, \ldots, \alpha_{1}) \tag{1} $$
/// $$ \widetilde{eq}(x_{n-j+1}; v_{n-j+1}) \tag{2} $$
/// $$ \widetilde{eq}(x_1, \ldots, x_{n-j}; v_1, \ldots, v_{n-j}) \tag{3} $$
///
/// The following holds:
/// * (1) is a constant that can be incrementally updated in O(1) time,
/// * (2) is a linear polynomial that is easy to compute in monomial form specialized to either
///   variable
/// * (3) is a an equality indicator over the claim point suffix
///
/// These observations allow us to instead sumcheck:
/// $$
/// P'(x) = \sum_{v \in B} \widetilde{eq}(x_1, \ldots, x_{n-j}; v_1, \ldots, v_{n-j}) A(v) B(v)
/// $$
///
/// Which is simpler because:
/// * $P'(x)$ is degree-2 in $j$-th variable, requiring one less evaluation point
/// * Equality indicator expansion does not depend on $j$-th variable and thus doesn't need to be
///   interpolated
///
/// After computing the round polynomial for $P'(x)$ in monomial form, one can simply multiply by
/// (2) and (1) in polynomial form. For more details, see [`Gruen32`]('gruen32::Gruen32') struct and
/// [Gruen24] Section 3.2.
///
/// Note 1: as evident from the definition, this prover binds variables in high-to-low index order.
///
/// Note 2: evaluation points are 0 (implicit), 1 and Karatsuba infinity.
///
/// # Invariants
///
/// - The length of both multilinears is always equal.
///
/// [Gruen24]: <https://eprint.iacr.org/2024/108>
#[derive(Debug, Clone)]
pub struct BivariateProductMlecheckProver<P: PackedField> {
	multilinears: [FieldBuffer<P>; 2],
	last_coeffs_or_eval: RoundCoeffsOrEval<P::Scalar>,
	gruen32: Gruen32<P>,
}

impl<F: Field, P: PackedField<Scalar = F>> BivariateProductMlecheckProver<P> {
	/// Constructs a prover, given the multilinear polynomial evaluations and the evaluation
	/// claim on the multilinear extension of their product.
	pub fn new(
		multilinears: [FieldBuffer<P>; 2],
		eval_point: &[F],
		eval_claim: F,
	) -> Result<Self, Error> {
		if multilinears[0].log_len() != multilinears[1].log_len() {
			return Err(Error::MultilinearSizeMismatch);
		}

		if multilinears[0].log_len() != eval_point.len() {
			return Err(Error::EvalPointLengthMismatch);
		}

		let last_coeffs_or_sum = RoundCoeffsOrEval::Eval(eval_claim);

		let gruen32 = Gruen32::new(eval_point);

		Ok(Self {
			multilinears,
			last_coeffs_or_eval: last_coeffs_or_sum,
			gruen32,
		})
	}
}

impl<F, P> SumcheckProver<F> for BivariateProductMlecheckProver<P>
where
	F: Field,
	P: PackedField<Scalar = F>,
{
	fn n_vars(&self) -> usize {
		self.gruen32.n_vars_remaining()
	}

	fn n_claims(&self) -> usize {
		1
	}

	fn execute(&mut self) -> Result<Vec<RoundCoeffs<F>>, Error> {
		let RoundCoeffsOrEval::Eval(last_eval) = &self.last_coeffs_or_eval else {
			return Err(Error::ExpectedFold);
		};

		// Multilinear inputs are the same length by invariant
		debug_assert_eq!(self.multilinears[0].len(), self.multilinears[1].len());

		let n_vars_remaining = self.n_vars();
		assert!(n_vars_remaining > 0);

		// For P' the eq expansion does not depend on the currently specialized variable and
		// thus doesn't need to be interpolated.
		let eq_expansion = self.gruen32.eq_expansion();

		let (evals_a_0, evals_a_1) = self.multilinears[0].split_half()?;
		let (evals_b_0, evals_b_1) = self.multilinears[1].split_half()?;

		// Compute F(1) and F(∞) where F = ∑_{v ∈ B} A(v || X) B(v || X) eq(v, z)
		let round_evals = (
			eq_expansion.as_ref(),
			evals_a_0.as_ref(),
			evals_a_1.as_ref(),
			evals_b_0.as_ref(),
			evals_b_1.as_ref(),
		)
			.into_par_iter()
			.map(|(&eq_i, &evals_a_0_i, &evals_a_1_i, &evals_b_0_i, &evals_b_1_i)| {
				// Evaluate M(∞) = M(0) + M(1)
				let evals_a_inf_i = evals_a_0_i + evals_a_1_i;
				let evals_b_inf_i = evals_b_0_i + evals_b_1_i;

				let prod_1_i = eq_i * evals_a_1_i * evals_b_1_i;
				let prod_inf_i = eq_i * evals_a_inf_i * evals_b_inf_i;

				RoundEvals2 {
					y_1: prod_1_i,
					y_inf: prod_inf_i,
				}
			})
			.reduce(RoundEvals2::default, |lhs, rhs| lhs + &rhs)
			.sum_scalars(n_vars_remaining);

		let alpha = self.gruen32.next_coordinate();
		let round_coeffs = round_evals.interpolate_eq(*last_eval, alpha);

		self.last_coeffs_or_eval = RoundCoeffsOrEval::Coeffs(round_coeffs.clone());
		Ok(vec![round_coeffs])
	}

	fn fold(&mut self, challenge: F) -> Result<(), Error> {
		let RoundCoeffsOrEval::Coeffs(prime_coeffs) = &self.last_coeffs_or_eval else {
			return Err(Error::ExpectedExecute);
		};

		assert!(self.n_vars() > 0);

		let sum = prime_coeffs.evaluate(challenge);

		self.multilinears
			.par_iter_mut()
			.try_for_each(|multilinear| fold_highest_var_inplace(multilinear, challenge))?;

		self.gruen32.fold(challenge)?;
		self.last_coeffs_or_eval = RoundCoeffsOrEval::Eval(sum);
		Ok(())
	}

	fn finish(self) -> Result<Vec<F>, Error> {
		if self.n_vars() > 0 {
			let error = match self.last_coeffs_or_eval {
				RoundCoeffsOrEval::Coeffs(_) => Error::ExpectedFold,
				RoundCoeffsOrEval::Eval(_) => Error::ExpectedExecute,
			};

			return Err(error);
		}

		let multilinear_evals = self
			.multilinears
			.into_iter()
			.map(|multilinear| multilinear.get_checked(0).expect("multilinear.len() == 1"))
			.collect();

		Ok(multilinear_evals)
	}
}

impl<F, P> MleCheckProver<F> for BivariateProductMlecheckProver<P>
where
	F: Field,
	P: PackedField<Scalar = F>,
{
	fn eval_point(&self) -> &[F] {
		&self.gruen32.eval_point()[..self.n_vars()]
	}
}

#[derive(Debug, Clone)]
enum RoundCoeffsOrEval<F: Field> {
	Coeffs(RoundCoeffs<F>),
	Eval(F),
}

#[cfg(test)]
mod tests {
	use binius_field::arch::{OptimalB128, OptimalPackedB128};
	use binius_math::{
		multilinear::{eq::eq_ind, evaluate::evaluate},
		test_utils::{random_field_buffer, random_scalars},
	};
	use binius_transcript::ProverTranscript;
	use binius_verifier::{
		config::StdChallenger,
		protocols::{mlecheck, sumcheck::verify},
	};
	use itertools::{self, Itertools};
	use rand::{SeedableRng, prelude::StdRng};

	use super::*;
	use crate::protocols::sumcheck::{
		MleToSumCheckDecorator, prove::prove_single, prove_single_mlecheck,
	};

	fn test_mlecheck_prove_verify<F, P>(
		prover: impl MleCheckProver<F>,
		eval_claim: F,
		eval_point: &[F],
		multilinear_a: FieldBuffer<P>,
		multilinear_b: FieldBuffer<P>,
	) where
		F: Field,
		P: PackedField<Scalar = F>,
	{
		// Run the proving protocol
		let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
		let output = prove_single_mlecheck(prover, &mut prover_transcript).unwrap();

		// Write the multilinear evaluations to the transcript
		prover_transcript
			.message()
			.write_slice(&output.multilinear_evals);

		// Convert to verifier transcript and run verification
		let mut verifier_transcript = prover_transcript.into_verifier();
		let sumcheck_output = mlecheck::verify::<F, _>(
			eval_point,
			2, // degree 2 for bivariate product
			eval_claim,
			&mut verifier_transcript,
		)
		.unwrap();

		let mut reduced_eval_point = sumcheck_output.challenges.clone();
		reduced_eval_point.reverse();

		// Read the multilinear evaluations from the transcript
		let multilinear_evals: Vec<F> = verifier_transcript.message().read_vec(2).unwrap();

		// Check that the product of the evaluations equals the reduced evaluation
		assert_eq!(
			multilinear_evals[0] * multilinear_evals[1],
			sumcheck_output.eval,
			"Product of multilinear evaluations should equal the reduced evaluation"
		);

		// Check that the original multilinears evaluate to the claimed values at the challenge
		// point The prover binds variables from high to low, but evaluate expects them from low
		// to high
		let eval_a = evaluate(&multilinear_a, &reduced_eval_point).unwrap();
		let eval_b = evaluate(&multilinear_b, &reduced_eval_point).unwrap();

		assert_eq!(
			eval_a, multilinear_evals[0],
			"Multilinear A should evaluate to the first claimed evaluation"
		);
		assert_eq!(
			eval_b, multilinear_evals[1],
			"Multilinear B should evaluate to the second claimed evaluation"
		);

		// Also verify the challenges match what the prover saw
		assert_eq!(
			output.challenges, sumcheck_output.challenges,
			"Prover and verifier challenges should match"
		);
	}

	fn test_wrapped_sumcheck_prove_verify<F, P>(
		mlecheck_prover: impl MleCheckProver<F>,
		eval_claim: F,
		eval_point: &[F],
		multilinear_a: FieldBuffer<P>,
		multilinear_b: FieldBuffer<P>,
	) where
		F: Field,
		P: PackedField<Scalar = F>,
	{
		let n_vars = mlecheck_prover.n_vars();
		let prover = MleToSumCheckDecorator::new(mlecheck_prover);

		// Run the proving protocol
		let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
		let output = prove_single(prover, &mut prover_transcript).unwrap();

		// Write the multilinear evaluations to the transcript
		prover_transcript
			.message()
			.write_slice(&output.multilinear_evals);

		// Convert to verifier transcript and run verification
		let mut verifier_transcript = prover_transcript.into_verifier();
		let sumcheck_output = verify::<F, _>(
			n_vars,
			3, // degree 3 for trivariate product (bivariate by equality indicator)
			eval_claim,
			&mut verifier_transcript,
		)
		.unwrap();

		// The prover binds variables from high to low, but evaluate expects them from low
		// to high
		let mut reduced_eval_point = sumcheck_output.challenges.clone();
		reduced_eval_point.reverse();

		// Read the multilinear evaluations from the transcript
		let multilinear_evals: Vec<F> = verifier_transcript.message().read_vec(2).unwrap();

		// Evaluate the equality indicator
		let eq_ind_eval = eq_ind(eval_point, &reduced_eval_point);

		// Check that the product of the evaluations equals the reduced evaluation
		assert_eq!(
			multilinear_evals[0] * multilinear_evals[1] * eq_ind_eval,
			sumcheck_output.eval,
			"Product of multilinear evaluations should equal the reduced evaluation"
		);

		// Check that the original multilinears evaluate to the claimed values at the challenge
		// point
		let eval_a = evaluate(&multilinear_a, &reduced_eval_point).unwrap();
		let eval_b = evaluate(&multilinear_b, &reduced_eval_point).unwrap();

		assert_eq!(
			eval_a, multilinear_evals[0],
			"Multilinear A should evaluate to the first claimed evaluation"
		);
		assert_eq!(
			eval_b, multilinear_evals[1],
			"Multilinear B should evaluate to the second claimed evaluation"
		);

		// Also verify the challenges match what the prover saw
		assert_eq!(
			output.challenges, sumcheck_output.challenges,
			"Prover and verifier challenges should match"
		);
	}

	#[test]
	fn test_bivariate_product_mlecheck() {
		type F = OptimalB128;
		type P = OptimalPackedB128;

		let n_vars = 8;
		let mut rng = StdRng::seed_from_u64(0);

		// Generate two random multilinear polynomials
		let multilinear_a = random_field_buffer::<P>(&mut rng, n_vars);
		let multilinear_b = random_field_buffer::<P>(&mut rng, n_vars);

		// Compute product multilinear
		let product = itertools::zip_eq(multilinear_a.as_ref(), multilinear_b.as_ref())
			.map(|(&l, &r)| l * r)
			.collect_vec();
		let product_buffer = FieldBuffer::new(n_vars, product).unwrap();

		let eval_point = random_scalars::<F>(&mut rng, n_vars);
		let eval_claim = evaluate(&product_buffer, &eval_point).unwrap();

		// Create the prover
		let mlecheck_prover = BivariateProductMlecheckProver::new(
			[multilinear_a.clone(), multilinear_b.clone()],
			&eval_point,
			eval_claim,
		)
		.unwrap();

		test_mlecheck_prove_verify(
			mlecheck_prover.clone(),
			eval_claim,
			&eval_point,
			multilinear_a.clone(),
			multilinear_b.clone(),
		);

		test_wrapped_sumcheck_prove_verify(
			mlecheck_prover.clone(),
			eval_claim,
			&eval_point,
			multilinear_a.clone(),
			multilinear_b.clone(),
		);
	}
}
