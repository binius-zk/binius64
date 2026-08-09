// Copyright 2025-2026 The Binius Developers

use binius_field::{Field, PackedField, WideMul};
use binius_ip::sumcheck::RoundCoeffs;
use binius_math::FieldSlice;

use crate::sumcheck::{
	mle_store::{ColId, ColumnChunk, EvaluationChunk, RoundContext},
	quadratic_mle_evaluator::QuadraticMleEvaluator,
	round_evals::RoundEvals,
	round_evaluator::MleCheckRoundEvaluator,
};

/// Creates the round evaluators for the fractional-addition claims required in logUp*.
///
/// The columns are `[num_a, num_b, den_a, den_b]`: the numerators and denominators of the two
/// fraction collections being added, split as either half of one layer buffer. The two claims —
/// one evaluator each — are the fractional-addition numerator `num_a * den_b + num_b * den_a` over
/// all four columns and the denominator `den_a * den_b` over `[den_a, den_b]`, both weighted by the
/// equality indicator at the shared evaluation point. The driving [`SharedMleCheckProver`] owns
/// that point's eq tracker and holds the two claimed evaluations, so the evaluators carry neither.
///
/// [`SharedMleCheckProver`]: crate::sumcheck::round_evaluator::SharedMleCheckProver
pub fn evaluators<F, P>(
	cols: [ColId; 4],
) -> (impl MleCheckRoundEvaluator<F, P> + 'static, impl MleCheckRoundEvaluator<F, P> + 'static)
where
	F: Field,
	P: PackedField<Scalar = F>,
{
	let [num_a, num_b, den_a, den_b] = cols;
	// The fractional addition formulas are purely quadratic, so each infinity composition matches
	// its regular composition.
	let numerator = QuadraticMleEvaluator::new(
		[num_a, num_b, den_a, den_b],
		|[num_a, num_b, den_a, den_b]: [P; 4]| num_a * den_b + num_b * den_a,
		|[num_a, num_b, den_a, den_b]: [P; 4]| num_a * den_b + num_b * den_a,
	);
	let denominator = QuadraticMleEvaluator::new(
		[den_a, den_b],
		|[den_a, den_b]: [P; 2]| den_a * den_b,
		|[den_a, den_b]: [P; 2]| den_a * den_b,
	);
	(numerator, denominator)
}

/// Round evaluator that reduces both fractional-addition claims in a single pass.
///
/// [`evaluators`] returns one evaluator per claim, so each claim walks the columns itself:
///
/// ```text
///     numerator pass    num_a num_b den_a den_b   eq      8 column loads + 1 eq load
///     denominator pass              den_a den_b   eq      4 column loads + 1 eq load
/// ```
///
/// This evaluator reads the four columns once and fills both claims' accumulator slots.
/// The sums `den_a.lo + den_a.hi` and `den_b.lo + den_b.hi` are then formed once, not twice.
///
/// It emits one round polynomial, already batched, rather than one per claim.
/// That is exact, because [`RoundEvals::interpolate_eq`] is affine in its claim and evaluations:
///
/// ```text
///     interp(c_num + z*c_den, y_num + z*y_den) = interp(c_num, y_num) + z*interp(c_den, y_den)
/// ```
///
/// So the driving prover carries the single batched claim `c_num + z*c_den`.
/// The polynomial this emits is the one the verifier's batched MLE-check reconstructs.
pub struct FracAddFusedEvaluator<F> {
	/// Layer columns in the order `[num_a, num_b, den_a, den_b]`.
	cols: [ColId; 4],
	/// The coefficient that batches the denominator claim onto the numerator claim.
	batch_coeff: F,
}

impl<F: Field> FracAddFusedEvaluator<F> {
	/// Creates the fused evaluator over the four layer columns.
	///
	/// # Arguments
	///
	/// * `cols` - The columns `[num_a, num_b, den_a, den_b]`.
	/// * `batch_coeff` - The coefficient the driving protocol batches the two claims with.
	///
	/// The prover's claim for this evaluator must be `num_claim + batch_coeff * den_claim`.
	pub const fn new(cols: [ColId; 4], batch_coeff: F) -> Self {
		Self { cols, batch_coeff }
	}
}

impl<F, P> MleCheckRoundEvaluator<F, P> for FracAddFusedEvaluator<F>
where
	F: Field,
	P: PackedField<Scalar = F>,
{
	fn degree(&self) -> usize {
		// Four slots: the numerator's `y_1` and `y_inf`, then the denominator's.
		// Each quadratic claim contributes the two sampled evaluations of its prime polynomial.
		// The emitted polynomial is still degree 2.
		4
	}

	fn accumulate(
		&self,
		chunk: &EvaluationChunk<'_, P>,
		eq_ind: FieldSlice<'_, P>,
		accum: &mut [<P as WideMul>::Output],
	) {
		// Each column arrives split on the round's top variable: `lo` fixes it to 0, `hi` to 1.
		let [num_a, num_b, den_a, den_b]: [&ColumnChunk<'_, P>; 4] =
			self.cols.map(|id| chunk.col(id));

		let mut num_y_1 = <P as WideMul>::Output::default();
		let mut num_y_inf = <P as WideMul>::Output::default();
		let mut den_y_1 = <P as WideMul>::Output::default();
		let mut den_y_inf = <P as WideMul>::Output::default();

		for (idx, &eq_i) in eq_ind.as_ref().iter().enumerate() {
			// One load per column half, shared by both claims.
			let na_lo = num_a.lo.as_ref()[idx];
			let na_hi = num_a.hi.as_ref()[idx];
			let nb_lo = num_b.lo.as_ref()[idx];
			let nb_hi = num_b.hi.as_ref()[idx];
			let da_lo = den_a.lo.as_ref()[idx];
			let da_hi = den_a.hi.as_ref()[idx];
			let db_lo = den_b.lo.as_ref()[idx];
			let db_hi = den_b.hi.as_ref()[idx];

			// The `x = 1` branch reads the high halves directly.
			num_y_1 += P::wide_mul(na_hi * db_hi + nb_hi * da_hi, eq_i);
			den_y_1 += P::wide_mul(da_hi * db_hi, eq_i);

			// The `x = infinity` branch reads `lo + hi`, each multilinear's leading coefficient.
			// The two denominator sums serve both claims.
			let na = na_lo + na_hi;
			let nb = nb_lo + nb_hi;
			let da = da_lo + da_hi;
			let db = db_lo + db_hi;
			num_y_inf += P::wide_mul(na * db + nb * da, eq_i);
			den_y_inf += P::wide_mul(da * db, eq_i);
		}

		// The run holds the two claims back to back: numerator first, then denominator.
		let (numerator, denominator) = accum.split_at_mut(2);
		RoundEvals([num_y_1, num_y_inf]).add_to(numerator);
		RoundEvals([den_y_1, den_y_inf]).add_to(denominator);
	}

	fn interpolate(
		&self,
		ctx: &RoundContext<'_, P>,
		accum: &[P],
		claim: F,
		alpha: F,
	) -> RoundCoeffs<F> {
		// The store has not yet folded this round, so its count is this round's.
		let n_vars_remaining = ctx.n_vars();
		assert!(n_vars_remaining > 0);

		// Sum each claim's packed lanes into scalars.
		let (num_slots, den_slots) = accum.split_at(2);
		let numerator = RoundEvals::<P, 2>::from_slots(num_slots).sum_scalars(n_vars_remaining);
		let denominator = RoundEvals::<P, 2>::from_slots(den_slots).sum_scalars(n_vars_remaining);

		// Batch the sampled evaluations, then interpolate once against the batched claim.
		// That order is equivalent to interpolating each claim and batching the polynomials.
		(numerator + &(denominator * self.batch_coeff)).interpolate_eq(claim, alpha)
	}
}

#[cfg(test)]
mod tests {
	use binius_field::arch::{OptimalB128, OptimalPackedB128};
	use binius_ip::sumcheck::batch_verify;
	use binius_math::{
		FieldBuffer,
		multilinear::{eq::eq_ind, evaluate::evaluate},
		test_utils::{random_field_buffer, random_scalars},
	};
	use binius_transcript::{ProverTranscript, fiat_shamir::HasherChallenger};

	type StdChallenger = HasherChallenger<sha2::Sha256>;
	use binius_compute::GlobalAllocator;
	use itertools::{Itertools, izip};
	use rand::prelude::*;

	use super::*;
	use crate::sumcheck::{
		MleToSumCheckEvaluator,
		batch::batch_prove,
		common::{MleCheckProver, SumcheckProver},
		mle_store::MleStore,
		round_evaluator::{SharedMleCheckProver, SharedSumcheckProver, SumcheckRoundEvaluator},
	};

	// One layer's two honest claims: the numerator and denominator compositions.
	// Each is its composite multilinear evaluated at the shared point.
	fn layer_claims<F, P>(
		num_parent: &FieldBuffer<P>,
		den_parent: &FieldBuffer<P>,
		eval_point: &[F],
	) -> (F, F)
	where
		F: Field,
		P: PackedField<Scalar = F>,
	{
		let n_vars = eval_point.len();
		let (num_a, num_b) = num_parent.split_half_ref();
		let (den_a, den_b) = den_parent.split_half_ref();
		let packed_len = 1 << n_vars.saturating_sub(P::LOG_WIDTH);

		// The composite values over the halved hypercube, word by word.
		let numerator: Vec<P> = (0..packed_len)
			.map(|i| num_a.as_ref()[i] * den_b.as_ref()[i] + num_b.as_ref()[i] * den_a.as_ref()[i])
			.collect();
		let denominator: Vec<P> = (0..packed_len)
			.map(|i| den_a.as_ref()[i] * den_b.as_ref()[i])
			.collect();

		(
			evaluate(&FieldBuffer::new(n_vars, numerator), eval_point),
			evaluate(&FieldBuffer::new(n_vars, denominator), eval_point),
		)
	}

	#[test]
	fn fused_evaluator_matches_batched_split_evaluators() {
		type F = OptimalB128;
		type P = OptimalPackedB128;

		let mut rng = StdRng::seed_from_u64(0);
		let alloc = GlobalAllocator;

		// 1 and 3 are single-chunk rounds throughout.
		// 8 and 14 cross the chunk cap, so 14 also exercises eq suffix folding in the driver.
		for n_vars in [1, 3, 8, 14] {
			// The parents carry the variable that splits them into the four layer columns.
			let num_parent = random_field_buffer::<P>(&mut rng, n_vars + 1);
			let den_parent = random_field_buffer::<P>(&mut rng, n_vars + 1);
			let eval_point = random_scalars::<F>(&mut rng, n_vars);
			let batch_coeff = random_scalars::<F>(&mut rng, 1)[0];
			let (num_claim, den_claim) = layer_claims(&num_parent, &den_parent, &eval_point);

			// Reference: one evaluator per claim, the two round polynomials batched by the driver.
			let mut split_store = MleStore::new(n_vars, &alloc);
			let [s_num_a, s_num_b] = split_store.push_split_half(num_parent.clone());
			let [s_den_a, s_den_b] = split_store.push_split_half(den_parent.clone());
			let (num_evaluator, den_evaluator) =
				evaluators::<F, P>([s_num_a, s_num_b, s_den_a, s_den_b]);
			let split_claims: [(F, Box<dyn MleCheckRoundEvaluator<F, P>>); 2] = [
				(num_claim, Box::new(num_evaluator)),
				(den_claim, Box::new(den_evaluator)),
			];
			let mut split_prover =
				SharedMleCheckProver::new(split_store, split_claims, eval_point.clone());

			// Fused: one evaluator, one claim, already batched.
			let mut fused_store = MleStore::new(n_vars, &alloc);
			let [f_num_a, f_num_b] = fused_store.push_split_half(num_parent.clone());
			let [f_den_a, f_den_b] = fused_store.push_split_half(den_parent.clone());
			let fused_evaluator =
				FracAddFusedEvaluator::new([f_num_a, f_num_b, f_den_a, f_den_b], batch_coeff);
			let mut fused_prover = SharedMleCheckProver::new(
				fused_store,
				[(num_claim + batch_coeff * den_claim, fused_evaluator)],
				eval_point.clone(),
			);

			// Both provers see the same challenges, so every round must agree.
			for round in 0..n_vars {
				// The driver Horner-folds the two claims into `num + batch_coeff * den`.
				let batched_reference = split_prover
					.execute()
					.into_iter()
					.rfold(RoundCoeffs::default(), |acc, coeffs| acc * batch_coeff + &coeffs);

				let fused_round = fused_prover.execute();
				assert_eq!(fused_round.len(), 1, "the fused evaluator emits one polynomial");
				assert_eq!(
					fused_round[0].0, batched_reference.0,
					"round {round} polynomial differs at n_vars={n_vars}"
				);

				let challenge = random_scalars::<F>(&mut rng, 1)[0];
				split_prover.fold(challenge);
				fused_prover.fold(challenge);
			}

			// Both stores hold the same four columns, so they must reduce to the same evaluations.
			assert_eq!(
				split_prover.finish(),
				fused_prover.finish(),
				"column evaluations differ at n_vars={n_vars}"
			);
		}
	}

	fn test_frac_add_sumcheck_prove_verify<F, P>(
		prover: impl SumcheckProver<F>,
		eval_claims: [F; 2],
		eval_point: &[F],
		num_a: &FieldBuffer<P>,
		num_b: &FieldBuffer<P>,
		den_a: &FieldBuffer<P>,
		den_b: &FieldBuffer<P>,
	) where
		F: Field,
		P: PackedField<Scalar = F>,
	{
		let n_vars = prover.n_vars();
		// Run the proving protocol
		let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
		let output = batch_prove(vec![prover], &mut prover_transcript);

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
		let eval_num_a = evaluate(num_a, &reduced_eval_point);
		let eval_den_a = evaluate(den_a, &reduced_eval_point);
		let eval_num_b = evaluate(num_b, &reduced_eval_point);
		let eval_den_b = evaluate(den_b, &reduced_eval_point);

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

		assert_eq!(
			output.challenges, sumcheck_output.challenges,
			"Prover and verifier challenges should match"
		);
	}

	#[test]
	fn test_frac_add_sumcheck() {
		type F = OptimalB128;
		type P = OptimalPackedB128;

		let n_vars = 8;
		let mut rng = StdRng::seed_from_u64(0);
		let alloc = GlobalAllocator;

		let num_a = random_field_buffer::<P>(&mut rng, n_vars);
		let num_b = random_field_buffer::<P>(&mut rng, n_vars);
		let den_a = random_field_buffer::<P>(&mut rng, n_vars);
		let den_b = random_field_buffer::<P>(&mut rng, n_vars);

		let numerator_values =
			izip!(num_a.as_ref(), den_a.as_ref(), num_b.as_ref(), den_b.as_ref())
				.map(|(&num_a, &den_a, &num_b, &den_b)| num_a * den_b + num_b * den_a)
				.collect_vec();

		let denominator_values = izip!(den_a.as_ref(), den_b.as_ref())
			.map(|(&den_a, &den_b)| den_a * den_b)
			.collect_vec();

		let numerator_buffer = FieldBuffer::new(n_vars, numerator_values);
		let denominator_buffer = FieldBuffer::new(n_vars, denominator_values);

		let eval_point = random_scalars::<F>(&mut rng, n_vars);
		// Claims are at the original eval_point; verifier handles challenge ordering separately.
		let eval_claims = [
			evaluate(&numerator_buffer, &eval_point),
			evaluate(&denominator_buffer, &eval_point),
		];

		let mut store = MleStore::new(n_vars, &alloc);
		let cols = [num_a.clone(), num_b.clone(), den_a.clone(), den_b.clone()]
			.map(|col| store.push_owned(col));
		let (num_evaluator, den_evaluator) = evaluators(cols);

		// Register the shared point's eq tracker for the sumcheck wrappers (a plain sumcheck prover
		// knows nothing of eq indicators, so the wrappers hold the tracker themselves).
		let eq_tracker = store.register_eq_tracker(&eval_point);
		// Wrap each MLE-check evaluator so it emits sumcheck-compatible round polynomials.
		let claims_with_evaluators: [(F, Box<dyn SumcheckRoundEvaluator<F, P>>); 2] = [
			(eval_claims[0], Box::new(MleToSumCheckEvaluator::new(num_evaluator, eq_tracker))),
			(eval_claims[1], Box::new(MleToSumCheckEvaluator::new(den_evaluator, eq_tracker))),
		];
		let prover = SharedSumcheckProver::new(store, claims_with_evaluators);

		test_frac_add_sumcheck_prove_verify(
			prover,
			eval_claims,
			&eval_point,
			&num_a,
			&num_b,
			&den_a,
			&den_b,
		);
	}
}
