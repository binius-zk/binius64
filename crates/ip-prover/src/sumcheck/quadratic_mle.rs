// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use binius_field::{Field, PackedField};
use binius_ip::sumcheck::RoundCoeffs;
use binius_math::{AsSlicesMut, FieldSliceMut, multilinear::fold::fold_highest_var_inplace};
use binius_utils::rayon::prelude::*;

use super::{common::SumcheckProver, error::Error, gruen32::Gruen32, round_evals::WideRoundEvals2};
use crate::sumcheck::common::MleCheckProver;

/// MLE-check prover for polynomials defined as quadratic compositions of N multilinear polynomials.
///
/// This prover implements the MLE (multilinear extension) check protocol. Given N multilinear
/// polynomials M₁, M₂, ..., Mₙ and a quadratic composition function C, it proves claims about
/// the multilinear extension of the composite polynomial C(M₁, M₂, ..., Mₙ).
///
/// The prover uses the sumcheck protocol to reduce claims about this multilinear extension
/// to claims about the individual multilinear evaluations, employing the Karatsuba optimization
/// for efficient degree-2 polynomial interpolation.
pub struct QuadraticMleCheckProver<P: PackedField, Composition, InfinityComposition, const N: usize>
{
	multilinears: Box<dyn AsSlicesMut<P, N> + Send>,
	composition: Composition,
	infinity_composition: InfinityComposition,
	last_coeffs_or_eval: RoundCoeffsOrEval<P::Scalar>,
	gruen32: Gruen32<P>,
}

impl<F, P, Composition, InfinityComposition, const N: usize>
	QuadraticMleCheckProver<P, Composition, InfinityComposition, N>
where
	F: Field,
	P: PackedField<Scalar = F>,
	Composition: Fn([P; N]) -> P + Sync,
	InfinityComposition: Fn([P; N]) -> P + Sync,
{
	/// Creates a new prover for verifying quadratic composite polynomial evaluations.
	///
	/// # Arguments
	///
	/// * `multilinears` - Array of N multilinear polynomials that serve as inputs to the
	///   composition. Each multilinear must have the same number of variables.
	///
	/// * `composition` - Function for evaluating the N-variate quadratic composition over packed
	///   field elements. Takes an array of N packed field elements (one from each multilinear) and
	///   returns their composition value. For example, for a product of two multilinears, this
	///   would compute `M₁(X) * M₂(X)`.
	///
	/// * `infinity_composition` - Polynomial evaluator that computes only the highest-degree terms
	///   of the composition. This is used for evaluation at the "infinity point" in the Karatsuba
	///   optimization, where we take the limit of P(X)/X^d as X approaches infinity. This limit
	///   equals the coefficient of the highest-degree term, effectively ignoring lower-degree
	///   terms. For example, if composition is `a*b - c`, the infinity_composition would be just
	///   `a*b`.
	///
	/// * `eval_point` - The point at which the multilinear extension is being evaluated. Must have
	///   length equal to the number of variables in the multilinears.
	///
	/// * `eval_claim` - The claimed value of the multilinear extension of the composite polynomial
	///   at the evaluation point. This is the multilinear extension of the function that maps v ∈
	///   {0,1}ⁿ to C(M₁(v), M₂(v), ..., Mₙ(v)), evaluated at eval_point.
	///
	/// # Returns
	///
	/// A configured prover instance ready to execute the sumcheck protocol.
	///
	/// # Errors
	///
	/// Returns `Error::MultilinearSizeMismatch` if any multilinear has a different number of
	/// variables than the length of `eval_point`.
	pub fn new(
		mut multilinears: impl AsSlicesMut<P, N> + Send + 'static,
		composition: Composition,
		infinity_composition: InfinityComposition,
		eval_point: Vec<F>,
		eval_claim: F,
	) -> Result<Self, Error> {
		let n_vars = eval_point.len();

		for multilinear in &multilinears.as_slices_mut() {
			if multilinear.log_len() != n_vars {
				return Err(Error::MultilinearSizeMismatch);
			}
		}

		let last_coeffs_or_eval = RoundCoeffsOrEval::Eval(eval_claim);
		let gruen32 = Gruen32::new(&eval_point);

		Ok(Self {
			multilinears: Box::new(multilinears),
			composition,
			infinity_composition,
			last_coeffs_or_eval,
			gruen32,
		})
	}

	/// Gets mutable slices of the multilinears, truncated to the current number of variables.
	fn multilinears_mut(&mut self) -> [FieldSliceMut<'_, P>; N] {
		let n_vars = self.gruen32.n_vars_remaining();
		let mut slices = self.multilinears.as_slices_mut();
		for slice in &mut slices {
			slice.truncate(n_vars);
		}
		slices
	}
}

impl<F, P, Composition, InfinityComposition, const N: usize> SumcheckProver<F>
	for QuadraticMleCheckProver<P, Composition, InfinityComposition, N>
where
	F: Field,
	P: PackedField<Scalar = F>,
	Composition: Fn([P; N]) -> P + Sync,
	InfinityComposition: Fn([P; N]) -> P + Sync,
{
	fn n_vars(&self) -> usize {
		self.gruen32.n_vars_remaining()
	}

	fn n_claims(&self) -> usize {
		1
	}

	fn round_claim(&self) -> Vec<F> {
		let claim = match &self.last_coeffs_or_eval {
			RoundCoeffsOrEval::Eval(eval) => *eval,
			RoundCoeffsOrEval::Coeffs(coeffs) => {
				coeffs.lerp_over_endpoints(self.gruen32.next_coordinate())
			}
		};
		vec![claim]
	}

	fn execute(&mut self) -> Result<Vec<RoundCoeffs<F>>, Error> {
		let last_eval = match &self.last_coeffs_or_eval {
			RoundCoeffsOrEval::Eval(eval) => *eval,
			RoundCoeffsOrEval::Coeffs(_) => return Err(Error::ExpectedFold),
		};

		let n_vars_remaining = self.gruen32.n_vars_remaining();
		assert!(n_vars_remaining > 0);

		let eq_expansion = self.gruen32.eq_expansion();
		assert_eq!(eq_expansion.log_len(), n_vars_remaining - 1);

		// Get references to compositions - these don't conflict with multilinears borrow
		let composition = &self.composition;
		let infinity_composition = &self.infinity_composition;

		// Get multilinear slices and truncate to current n_vars
		let mut multilinears = self.multilinears.as_slices_mut();
		for slice in &mut multilinears {
			slice.truncate(n_vars_remaining);
		}

		// Split each multilinear in half
		let (splits_0, splits_1) = multilinears
			.iter()
			.map(|multilinear| multilinear.split_half_ref())
			.unzip::<_, _, Vec<_>, Vec<_>>();

		// Compute F(1) and F(∞) where F = ∑_{v ∈ B} C(M_1(v || X), ..., M_N(v || X)) eq(v, z).
		// We need to iterate over all positions in parallel.
		//
		// The per-position products `C(..) * eq_i` are accumulated in unreduced (wide) form and
		// reduced a single time at the end, which amortizes the GF(2^128) reduction across all
		// hypercube points.
		let round_evals = eq_expansion
			.as_ref()
			.into_par_iter()
			.enumerate()
			.map(|(i, &eq_i)| {
				// Collect evaluations at 1 and ∞ for each multilinear
				let mut evals_1 = [P::default(); N];
				let mut evals_inf = [P::default(); N];
				for j in 0..N {
					// Monomial basis: the two halves are coeffs `[c0, c1]`, so `M(1) = c0 + c1`
					// and `M(∞) = c1` (the high half / leading coefficient).
					evals_1[j] = splits_0[j].as_ref()[i] + splits_1[j].as_ref()[i];
					evals_inf[j] = splits_1[j].as_ref()[i];
				}

				WideRoundEvals2 {
					// Evaluate composition at X=1
					y_1: P::wide_mul(composition(evals_1), eq_i),
					// Evaluate composition at X=∞ (where M(∞) = M(0) + M(1))
					y_inf: P::wide_mul(infinity_composition(evals_inf), eq_i),
				}
			})
			.reduce(WideRoundEvals2::default, |lhs, rhs| lhs + rhs)
			.reduce::<P>()
			.sum_scalars(n_vars_remaining - 1);

		let alpha = self.gruen32.next_coordinate();
		let round_coeffs = round_evals.interpolate_eq(last_eval, alpha);

		self.last_coeffs_or_eval = RoundCoeffsOrEval::Coeffs(round_coeffs.clone());
		Ok(vec![round_coeffs])
	}

	fn fold(&mut self, challenge: F) -> Result<(), Error> {
		let RoundCoeffsOrEval::Coeffs(coeffs) = &self.last_coeffs_or_eval else {
			return Err(Error::ExpectedExecute);
		};

		assert!(
			self.n_vars() > 0,
			"n_vars is decremented in fold; \
			fold changes last_coeffs_or_eval to Eval variant; \
			fold only executes with Coeffs variant; \
			thus, n_vars should be > 0"
		);

		let eval = coeffs.evaluate(challenge);

		for multilinear in &mut self.multilinears_mut() {
			fold_highest_var_inplace(multilinear, challenge);
		}

		self.gruen32.fold(challenge);
		self.last_coeffs_or_eval = RoundCoeffsOrEval::Eval(eval);
		Ok(())
	}

	fn finish(mut self) -> Result<Vec<F>, Error> {
		if self.n_vars() > 0 {
			let error = match self.last_coeffs_or_eval {
				RoundCoeffsOrEval::Coeffs(_) => Error::ExpectedFold,
				RoundCoeffsOrEval::Eval(_) => Error::ExpectedExecute,
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

impl<F, P, Composition, InfinityComposition, const N: usize> MleCheckProver<F>
	for QuadraticMleCheckProver<P, Composition, InfinityComposition, N>
where
	F: Field,
	P: PackedField<Scalar = F>,
	Composition: Fn([P; N]) -> P + Sync,
	InfinityComposition: Fn([P; N]) -> P + Sync,
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
	use std::{array, iter};

	use binius_field::{arch::OptimalPackedB128, field::FieldOps};
	use binius_ip::mlecheck;
	use binius_math::{
		FieldBuffer,
		multilinear::evaluate::evaluate,
		test_utils::{random_field_buffer, random_scalars},
	};
	use binius_transcript::{ProverTranscript, fiat_shamir::HasherChallenger};

	type StdChallenger = HasherChallenger<sha2::Sha256>;
	use itertools::{self, Itertools};
	use rand::prelude::*;

	use super::*;
	use crate::{
		channel::IPProverChannel,
		sumcheck::{multilinear_eval::MultilinearEvalProver, prove_single_mlecheck},
	};

	/// Computes the MLE-check claimed value in the monomial basis: the composite
	/// `F = C(M_1, ..., M_N)` summed over the infinity hypercube and eq-weighted at `point`.
	///
	/// Equivalently `evaluate(F_proj, point)`, where `F_proj[v]` is `F` evaluated at the
	/// infinity-hypercube vertex `v` (the coordinate is `∞` where `v`'s bit is set and `0`
	/// otherwise). The recursion projects the top variable: its `0`-branch keeps the full
	/// `composition`; its `∞`-branch keeps only the leading-degree part `infinity_composition`,
	/// which is its own leading part on the remaining variables.
	///
	/// This is the basis-correct replacement for `evaluate(composition(coefficients), point)`,
	/// which only agrees for homogeneous compositions.
	fn composite_infinity_eval<F, P, const N: usize>(
		multilinears: &[FieldBuffer<P>; N],
		composition: impl Fn([P; N]) -> P,
		infinity_composition: impl Fn([P; N]) -> P,
		point: &[F],
	) -> F
	where
		F: Field,
		P: PackedField<Scalar = F>,
	{
		fn scalar<F: Field, P: PackedField<Scalar = F>, const N: usize>(
			c: &impl Fn([P; N]) -> P,
			vals: [F; N],
		) -> F {
			c(vals.map(P::broadcast)).iter().next().unwrap()
		}
		fn proj<F: Field, const N: usize>(
			mls: [Vec<F>; N],
			n: usize,
			comp: &dyn Fn([F; N]) -> F,
			inf: &dyn Fn([F; N]) -> F,
		) -> Vec<F> {
			if n == 0 {
				return vec![comp(array::from_fn(|j| mls[j][0]))];
			}
			let half = 1usize << (n - 1);
			let los: [Vec<F>; N] = array::from_fn(|j| mls[j][..half].to_vec());
			let his: [Vec<F>; N] = array::from_fn(|j| mls[j][half..].to_vec());
			let mut table = proj(los, n - 1, comp, inf);
			// The ∞-branch keeps only the leading part, which is homogeneous, so it is its own
			// leading part on deeper variables.
			table.extend(proj(his, n - 1, inf, inf));
			table
		}

		let n = point.len();
		let coeffs: [Vec<F>; N] = array::from_fn(|j| multilinears[j].iter_scalars().collect());
		let comp = |vals: [F; N]| scalar(&composition, vals);
		let inf = |vals: [F; N]| scalar(&infinity_composition, vals);
		let table = proj(coeffs, n, &comp, &inf);
		evaluate(&FieldBuffer::<P>::from_values(&table), point)
	}

	fn test_mlecheck_prove_verify<F, P, Composition, InfinityComposition, const N: usize>(
		prover: QuadraticMleCheckProver<P, Composition, InfinityComposition, N>,
		composition: Composition,
		eval_claim: F,
		eval_point: &[F],
		multilinears: Vec<FieldBuffer<P>>,
	) where
		F: Field,
		P: PackedField<Scalar = F>,
		Composition: Fn([P; N]) -> P + Sync,
		InfinityComposition: Fn([P; N]) -> P + Sync,
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
		let sumcheck_output = mlecheck::verify(
			eval_point,
			2, // degree 2 for composite polynomials
			eval_claim,
			&mut verifier_transcript,
		)
		.unwrap();

		let mut reduced_eval_point = sumcheck_output.challenges.clone();
		reduced_eval_point.reverse();

		// Read the multilinear evaluations from the transcript
		let multilinear_evals: Vec<F> = verifier_transcript.message().read_vec(N).unwrap();

		// Check that the composition of the evaluations equals the reduced evaluation
		let evals_packed: [P; N] = array::from_fn(|i| P::broadcast(multilinear_evals[i]));
		let composition_result = composition(evals_packed);
		assert_eq!(
			composition_result.iter().next().unwrap(),
			sumcheck_output.eval,
			"Composition of multilinear evaluations should equal the reduced evaluation"
		);

		// Check that the original multilinears evaluate to the claimed values at the challenge
		// point
		for (multilinear, claimed_eval) in iter::zip(&multilinears, multilinear_evals) {
			let actual_eval = evaluate(multilinear, &reduced_eval_point);
			assert_eq!(actual_eval, claimed_eval);
		}

		// Also verify the challenges match what the prover saw
		assert_eq!(
			output.challenges, sumcheck_output.challenges,
			"Prover and verifier challenges should match"
		);
	}

	fn test_quadratic_mlecheck_prove_verify<F, P, const N: usize>(
		composition: impl Fn([P; N]) -> P + Clone + Sync,
		infinity_composition: impl Fn([P; N]) -> P + Clone + Sync,
	) where
		F: Field,
		P: PackedField<Scalar = F>,
	{
		let n_vars = 8;
		let mut rng = StdRng::seed_from_u64(0);

		// Generate random multilinear polynomials
		let multilinears: [_; N] = array::from_fn(|_| random_field_buffer::<P>(&mut rng, n_vars));

		let eval_point = random_scalars::<F>(&mut rng, n_vars);
		// Monomial basis: the claim is the composite summed over the infinity hypercube, not the
		// multilinear extension of `composition(coefficients)`.
		let eval_claim = composite_infinity_eval(
			&multilinears,
			&composition,
			&infinity_composition,
			&eval_point,
		);

		// Create the prover
		let mlecheck_prover = QuadraticMleCheckProver::new(
			multilinears.clone(),
			composition.clone(),
			infinity_composition,
			eval_point.clone(),
			eval_claim,
		)
		.unwrap();

		test_mlecheck_prove_verify(
			mlecheck_prover,
			composition,
			eval_claim,
			&eval_point,
			multilinears.to_vec(),
		);
	}

	// Test that quadratic MLE-check handles multilinears. It's not the most efficient strategy
	// for a multilinear MLE-check, but it's a good edge case.
	#[test]
	fn test_linear_mlecheck() {
		test_quadratic_mlecheck_prove_verify::<_, OptimalPackedB128, 2>(
			|[a, b]| a + b,
			|[_a, _b]| OptimalPackedB128::zero(), // coefficient on the quadratic term is 0
		);
	}

	#[test]
	fn test_bivariate_product_mlecheck() {
		test_quadratic_mlecheck_prove_verify::<_, OptimalPackedB128, 2>(
			|[a, b]| a * b,
			|[a, b]| a * b,
		);
	}

	#[test]
	fn test_mul_gate_mlecheck() {
		test_quadratic_mlecheck_prove_verify::<_, OptimalPackedB128, 3>(
			|[a, b, c]| a * b - c,
			|[a, b, _c]| a * b,
		);
	}

	#[test]
	fn test_4_variate_composition_mlecheck() {
		test_quadratic_mlecheck_prove_verify::<_, OptimalPackedB128, 4>(
			|[a, b, c, d]| (a + b) * (c + d),
			|[a, b, c, d]| (a + b) * (c + d),
		);
	}

	// `round_claim` must return the same value before and after `execute()`: the MLE-check claim
	// recovered from the round coefficients via `lerp_over_endpoints` must equal the stored claim.
	#[test]
	fn test_round_claim_lerp_recovery() {
		use binius_field::{Random, arch::OptimalB128};
		type P = OptimalPackedB128;
		type F = OptimalB128;

		let n_vars = 8;
		let mut rng = StdRng::seed_from_u64(0);

		let multilinears: [_; 2] = array::from_fn(|_| random_field_buffer::<P>(&mut rng, n_vars));
		let composition = |[a, b]: [P; 2]| a * b;
		let composite_vals = (0..1 << n_vars.saturating_sub(P::LOG_WIDTH))
			.map(|i| composition(array::from_fn(|j| multilinears[j].as_ref()[i])))
			.collect_vec();
		let composite_vals = FieldBuffer::new(n_vars, composite_vals);
		let eval_point = random_scalars::<F>(&mut rng, n_vars);
		let eval_claim = evaluate(&composite_vals, &eval_point);

		let mut prover = QuadraticMleCheckProver::new(
			multilinears,
			composition,
			composition,
			eval_point,
			eval_claim,
		)
		.unwrap();

		let mut expected = vec![eval_claim];
		for _ in 0..n_vars {
			assert_eq!(prover.round_claim(), expected, "claim before execute");
			let round = prover.execute().unwrap();
			assert_eq!(prover.round_claim(), expected, "claim recovered from coeffs");
			let challenge = F::random(&mut rng);
			expected = round.iter().map(|r| r.evaluate(challenge)).collect();
			prover.fold(challenge).unwrap();
		}
	}

	// EXPERIMENT (BINIUS-114): can the non-homogeneous mul-gate composite `A*B - C` be proven by
	// SPLITTING the prover into a quadratic prover for the homogeneous product `A*B` and a separate
	// prover for the multilinear `C`, batched with hardcoded weights `+1` / `-1`, while leaving the
	// verifier as the unchanged single degree-2 `mlecheck::verify`?
	//
	// `make_c_prover` builds the `C` sub-prover (its framing varies across experiments) and
	// `claim_c` chooses how `C`'s claim is computed. Returns `Ok(())` on a full verify, or `Err`
	// describing the first inconsistency, so the driver can tabulate outcomes instead of
	// panicking.
	fn run_split_mul_gate_experiment<F, P, CProver>(
		n_vars: usize,
		claim_c: ClaimC,
		make_c_prover: impl FnOnce(FieldBuffer<P>, Vec<F>, F) -> CProver,
	) -> Result<(), String>
	where
		F: Field,
		P: PackedField<Scalar = F>,
		CProver: SumcheckProver<F>,
	{
		let mut rng = StdRng::seed_from_u64(0);

		let multilinears: [FieldBuffer<P>; 3] =
			array::from_fn(|_| random_field_buffer::<P>(&mut rng, n_vars));
		let eval_point = random_scalars::<F>(&mut rng, n_vars);

		let ab_multilinears = [multilinears[0].clone(), multilinears[1].clone()];
		let claim_ab = composite_infinity_eval(
			&ab_multilinears,
			|[a, b]: [P; 2]| a * b,
			|[a, b]: [P; 2]| a * b,
			&eval_point,
		);
		// Two candidate claims for the `C` term:
		// - `Eval`: the standalone monomial-basis multilinear evaluation `evaluate(C, point)`,
		//   which is what `MultilinearEvalProver` proves. It includes `C`'s mass at every infinity
		//   vertex.
		// - `ConstCoeff`: `C`'s contribution *inside the degree-2 composite*, i.e. its
		//   leading-degree-2 part summed over the infinity hypercube — `composite_infinity_eval(C,
		//   inf = 0)`, which collapses to the constant coefficient `c[0]`. This is the value the
		//   single quadratic prover (and `composite_infinity_eval(a*b - c, a*b)`) implicitly uses
		//   for `-C`.
		let claim_c_val = match claim_c {
			ClaimC::Eval => evaluate(&multilinears[2], &eval_point),
			ClaimC::ConstCoeff => composite_infinity_eval(
				&[multilinears[2].clone()],
				|[c]: [P; 1]| c,
				|[_c]: [P; 1]| P::zero(),
				&eval_point,
			),
		};

		// Hardcoded batch weights: `A*B - C` (the two coincide in characteristic 2, written out
		// anyway).
		let weight_ab = F::ONE;
		let weight_c = -F::ONE;
		let verifier_claim = weight_ab * claim_ab + weight_c * claim_c_val;

		let mut ab_prover = QuadraticMleCheckProver::new(
			ab_multilinears,
			|[a, b]: [P; 2]| a * b,
			|[a, b]: [P; 2]| a * b,
			eval_point.clone(),
			claim_ab,
		)
		.unwrap();
		let mut c_prover = make_c_prover(multilinears[2].clone(), eval_point.clone(), claim_c_val);

		let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
		for _ in 0..n_vars {
			let ab_round = ab_prover.execute().unwrap();
			let c_round = c_prover.execute().unwrap();
			// batched(X) = 1 * R_{A*B}(X) + (-1) * R_C(X). `RoundCoeffs` addition zero-pads the
			// shorter polynomial, so a lower-degree `R_C` contributes only to the low
			// coefficients and the degree-2 leading coefficient comes entirely from `R_{A*B}`.
			let batched = ab_round[0].clone() * weight_ab + &(c_round[0].clone() * weight_c);
			prover_transcript.send_many(mlecheck::RoundProof::truncate(batched).coeffs());
			let challenge = prover_transcript.sample();
			ab_prover.fold(challenge).unwrap();
			c_prover.fold(challenge).unwrap();
		}

		let ab_evals = ab_prover.finish().unwrap();
		let c_evals = c_prover.finish().unwrap();
		let multilinear_evals = vec![ab_evals[0], ab_evals[1], c_evals[0]];
		prover_transcript.message().write_slice(&multilinear_evals);

		let mut verifier_transcript = prover_transcript.into_verifier();
		let sumcheck_output =
			mlecheck::verify(&eval_point, 2, verifier_claim, &mut verifier_transcript).unwrap();
		let verifier_evals: Vec<F> = verifier_transcript.message().read_vec(3).unwrap();

		let [a, b, c] = [verifier_evals[0], verifier_evals[1], verifier_evals[2]];
		if a * b - c != sumcheck_output.eval {
			return Err(format!("composite A*B - C != reduced eval (n_vars={n_vars})"));
		}
		let mut reduced_eval_point = sumcheck_output.challenges.clone();
		reduced_eval_point.reverse();
		for (multilinear, &claimed) in iter::zip(&multilinears, &verifier_evals) {
			if evaluate(multilinear, &reduced_eval_point) != claimed {
				return Err(format!("multilinear eval mismatch (n_vars={n_vars})"));
			}
		}
		Ok(())
	}

	#[derive(Clone, Copy)]
	enum ClaimC {
		Eval,
		ConstCoeff,
	}

	// Tabulates the split experiment across `C`-prover framings, claim choices, and sizes. Each row
	// prints PASS/FAIL so the protocol-level conclusion is visible in the test output. The asserts
	// at the end encode the empirically-observed conclusion.
	#[test]
	fn test_split_mul_gate_experiment_matrix() {
		type P = OptimalPackedB128;
		let sizes = [1usize, 2, 3, 8];

		let eval_deg1 = |n, claim| {
			run_split_mul_gate_experiment::<_, P, _>(n, claim, |witness, point, claim_val| {
				MultilinearEvalProver::new(witness, &point, claim_val).unwrap()
			})
		};
		let eval_deg2 = |n, claim| {
			run_split_mul_gate_experiment::<_, P, _>(n, claim, |witness, point, claim_val| {
				QuadraticMleCheckProver::new(
					[witness],
					|[c]: [P; 1]| c,
					|[_c]: [P; 1]| P::zero(),
					point,
					claim_val,
				)
				.unwrap()
			})
		};

		let print_row =
			|name: &str, claim: ClaimC, run: &dyn Fn(usize, ClaimC) -> Result<(), String>| {
				let claim_name = match claim {
					ClaimC::Eval => "claim_C=evaluate(C)",
					ClaimC::ConstCoeff => "claim_C=c[0]      ",
				};
				let results: Vec<String> = sizes
					.iter()
					.map(|&n| match run(n, claim) {
						Ok(()) => format!("n={n}:PASS"),
						Err(_) => format!("n={n}:FAIL"),
					})
					.collect();
				println!("{name:32} | {claim_name} | {}", results.join("  "));
			};

		for claim in [ClaimC::Eval, ClaimC::ConstCoeff] {
			print_row("MultilinearEvalProver (deg1)", claim, &eval_deg1);
			print_row("QuadraticMleCheckProver  (deg2)", claim, &eval_deg2);
		}

		// Empirical conclusion (see printed matrix): under the UNCHANGED degree-2 verifier, no
		// (framing, claim) combination proves `A*B - C` across multiple rounds. The only passing
		// cells are single-round (`n_vars=1`), where there is no infinity-mass to carry between
		// rounds.
		//
		// - deg2 framing + `claim_C=c[0]` reproduces the single quadratic prover exactly: it
		//   carries the right claim but breaks for `n>=2` (the multi-round `-C` handling, the
		//   original bug).
		// - deg1 framing (`MultilinearEvalProver`) + `claim_C=evaluate(C)` proves the correct
		//   standalone `C` claim, but that claim over-counts `C`'s infinity-vertex mass relative to
		//   the degree-2 composite, and the degree-2 verifier's single `∞` slot cannot carry it —
		//   fails even at n=1.
		let passes_multiround = |run: &dyn Fn(usize, ClaimC) -> Result<(), String>, claim| {
			[2usize, 3, 8].iter().all(|&n| run(n, claim).is_ok())
		};
		assert!(
			!passes_multiround(&eval_deg1, ClaimC::Eval),
			"deg1 + evaluate(C) unexpectedly verified across rounds"
		);
		assert!(
			!passes_multiround(&eval_deg2, ClaimC::ConstCoeff),
			"deg2 + c[0] unexpectedly verified across rounds"
		);
	}
}
