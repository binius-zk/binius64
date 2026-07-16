// Copyright 2026 The Binius Developers

use binius_field::{Field, PackedField, WideMul};
use binius_ip::sumcheck::RoundCoeffs;
use binius_math::FieldBuffer;

use super::{
	mle_store::{ColId, ColumnChunk, EqId, EvaluationChunk, MleStore},
	round_evals::RoundEvals2,
	round_evaluator::{RoundEvaluator, SharedMleCheckProver},
	round_state::RoundState,
};

/// MLE-check round evaluator for one quadratic composition over N store columns.
///
/// This is the store-backed successor of the quadratic MLE-check prover: it evaluates the
/// composition in one pass per round, using the Gruen32-style degree-2 interpolation trick. Batch
/// several quadratic MLE checks by registering one evaluator per claim on a shared store; they read
/// the shared columns from the same round pass.
///
/// The evaluator emits the prime (eq-factored) round polynomial of the MLE-check protocol. Wrap it
/// in [`MleToSumCheckEvaluator`](super::MleToSumCheckEvaluator) to emit a regular sumcheck round
/// polynomial.
pub struct QuadraticMleEvaluator<P: PackedField, Composition, InfinityComposition, const N: usize> {
	// Store columns holding the packed evaluations of the input multilinears.
	cols: [ColId; N],
	// The store's (possibly shared) eq-indicator tracker for `eval_point`.
	eq_tracker: EqId,
	// Full quadratic composition evaluated on the "x = 1" branch for each multilinear.
	composition: Composition,
	// Composition restricted to highest-degree terms for the "x = ∞" evaluation (Karatsuba).
	infinity_composition: InfinityComposition,
	// State machine storage: last round's eval (interpolate input) or coeffs (fold input).
	last_coeffs_or_eval: RoundState<RoundCoeffs<P::Scalar>, P::Scalar>,
}

impl<F, P, Composition, InfinityComposition, const N: usize>
	QuadraticMleEvaluator<P, Composition, InfinityComposition, N>
where
	F: Field,
	P: PackedField<Scalar = F>,
	Composition: Fn([P; N]) -> P + Send + Sync,
	InfinityComposition: Fn([P; N]) -> P + Send + Sync,
{
	/// Creates an evaluator over `cols` reading the eq tracker `eq_tracker` from `store`.
	///
	/// The caller registers the evaluation point on the store (via
	/// [`MleStore::register_eq_tracker`]) and passes the resulting [`EqId`]; several evaluators
	/// sharing a point pass the same id, so the store folds that tracker once. The evaluator holds
	/// only the tracker id and queries the store for the round's alpha and remaining-variable count
	/// as they change — it keeps no copy of the point.
	///
	/// # Arguments
	///
	/// * `cols` - The N store columns the composition reads.
	/// * `eq_tracker` - The registered eq tracker for the claim's evaluation point.
	/// * `composition` - Evaluates the quadratic composition of the N column values.
	/// * `infinity_composition` - The composition restricted to its highest-degree terms, for the
	///   Karatsuba evaluation at infinity.
	/// * `eval_claim` - The claimed evaluation of the composition's MLE at the point.
	pub fn new(
		cols: [ColId; N],
		eq_tracker: EqId,
		composition: Composition,
		infinity_composition: InfinityComposition,
		eval_claim: F,
	) -> Self {
		// precondition
		assert!(N > 0);

		Self {
			cols,
			eq_tracker,
			composition,
			infinity_composition,
			last_coeffs_or_eval: RoundState::Claim(eval_claim),
		}
	}
}

/// Builds an MLE-check prover for one quadratic composition of N owned multilinears.
///
/// The reduced claim is
///
/// ```text
///     sum_{v in B} C(M_0(v), ..., M_{N-1}(v)) * eq(v, eval_point) = eval_claim
/// ```
///
/// for composition `C` over the boolean hypercube `B`.
/// It reduces to one evaluation claim per input multilinear at the challenge point.
///
/// This is the single-claim path.
/// Several quadratic claims that share columns are instead proved together by registering one
/// evaluator per claim on a shared store, which folds each shared column only once.
///
/// # Arguments
///
/// * `multilinears` - The N input multilinears, each over `eval_point.len()` variables.
/// * `composition` - Evaluates the quadratic composition, e.g. `|[a, b, c]| a * b - c`.
/// * `infinity_composition` - The composition's highest-degree terms, for the Karatsuba evaluation
///   at infinity, e.g. `|[a, b, _c]| a * b`.
/// * `eval_point` - The point at which the composite MLE is claimed.
/// * `eval_claim` - The claimed evaluation of the composite MLE at `eval_point`.
///
/// # Returns
///
/// A prover whose reduction emits the N column evaluations in the order given.
///
/// # Panics
///
/// Panics if any multilinear's variable count differs from `eval_point.len()`, or if `N == 0`.
pub fn quadratic_mlecheck_prover<F, P, Composition, InfinityComposition, const N: usize>(
	multilinears: [FieldBuffer<P>; N],
	composition: Composition,
	infinity_composition: InfinityComposition,
	eval_point: Vec<F>,
	eval_claim: F,
) -> SharedMleCheckProver<
	'static,
	F,
	P,
	QuadraticMleEvaluator<P, Composition, InfinityComposition, N>,
>
where
	F: Field,
	P: PackedField<Scalar = F>,
	Composition: Fn([P; N]) -> P + Send + Sync,
	InfinityComposition: Fn([P; N]) -> P + Send + Sync,
{
	let mut store = MleStore::new(eval_point.len());
	// Hand each column to the store, which checks its variable count against the point length.
	let cols = multilinears.map(|col| store.push_owned(col));
	// Register the evaluation point's equality-indicator tracker once for the sole evaluator.
	let eq_tracker = store.register_eq_tracker(&eval_point);
	let evaluator =
		QuadraticMleEvaluator::new(cols, eq_tracker, composition, infinity_composition, eval_claim);
	SharedMleCheckProver::new(store, vec![evaluator], eval_point)
}

impl<F, P, Composition, InfinityComposition, const N: usize> RoundEvaluator<F, P>
	for QuadraticMleEvaluator<P, Composition, InfinityComposition, N>
where
	F: Field,
	P: PackedField<Scalar = F>,
	Composition: Fn([P; N]) -> P + Send + Sync,
	InfinityComposition: Fn([P; N]) -> P + Send + Sync,
{
	fn degree(&self) -> usize {
		// Quadratic composition: two sampled evaluations, `y_1` and `y_inf`.
		2
	}

	fn round_claim(&self, store: &MleStore<'_, P>) -> F {
		match &self.last_coeffs_or_eval {
			RoundState::Claim(eval) => *eval,
			RoundState::Coeffs(coeffs) => {
				let alpha = store.eq_alpha(self.eq_tracker);
				coeffs.lerp_over_endpoints(alpha)
			}
		}
	}

	fn accumulate(&self, chunk: &EvaluationChunk<'_, P>, accum: &mut [<P as WideMul>::Output]) {
		let eq_chunk = chunk.eq(self.eq_tracker);

		// Each column arrives split into low/high halves for the top variable: the low half
		// corresponds to x=0, the high half to x=1.
		let cols: [&ColumnChunk<'_, P>; N] = self.cols.map(|id| chunk.col(id));

		// The evaluator's run holds `y_1` in slot 0 and `y_inf` in slot 1.
		let mut y_1 = <P as WideMul>::Output::default();
		let mut y_inf = <P as WideMul>::Output::default();
		for (idx, &eq_i) in eq_chunk.as_ref().iter().enumerate() {
			// Gather the idx-th evaluations of every multilinear at both halves.
			let mut evals_1 = [P::default(); N];
			let mut evals_inf = [P::default(); N];

			for i in 0..N {
				let lo_i = cols[i].lo.as_ref()[idx];
				let hi_i = cols[i].hi.as_ref()[idx];

				// Compose once with the high half and once with the lo+hi combination.
				// The lo+hi branch corresponds to evaluation at infinity for multilinears.
				evals_1[i] = hi_i;
				evals_inf[i] = lo_i + hi_i;
			}

			// Weight the composition by the eq indicator to keep the sumcheck claim aligned to
			// eval_point. Only this final multiply is widened; the composition products are already
			// reduced.
			y_1 += P::wide_mul((self.composition)(evals_1), eq_i);
			y_inf += P::wide_mul((self.infinity_composition)(evals_inf), eq_i);
		}

		accum[0] += y_1;
		accum[1] += y_inf;
	}

	fn interpolate(
		&mut self,
		store: &MleStore<'_, P>,
		accum: &[<P as WideMul>::Output],
	) -> RoundCoeffs<F> {
		// State machine: interpolate consumes the eval from the previous round and produces coeffs.
		let last_eval = *self.last_coeffs_or_eval.claim();

		// The store has not yet folded this round, so its remaining-variable count is this round's.
		let n_vars_remaining = store.n_vars();
		assert!(n_vars_remaining > 0);

		// Reduce the wide accumulators, sum packed lanes into scalars, then interpolate. The
		// round's coordinate ties this round's sum to the original evaluation point.
		let alpha = store.eq_alpha(self.eq_tracker);
		let round_coeffs = RoundEvals2 {
			y_1: P::reduce(accum[0].clone()),
			y_inf: P::reduce(accum[1].clone()),
		}
		.sum_scalars(n_vars_remaining)
		.interpolate_eq(last_eval, alpha);

		// State transition: interpolate produces coeffs for fold to consume.
		self.last_coeffs_or_eval = RoundState::Coeffs(round_coeffs.clone());
		round_coeffs
	}

	fn fold(&mut self, challenge: F) {
		// State machine: fold consumes coeffs and produces the eval at the verifier challenge.
		// Evaluate the round polynomial at the verifier's challenge to form the next claim. The
		// store folds the columns and the eq tracker (advancing its remaining count and alpha) with
		// the same challenge, so this only advances the claim state.
		let eval = self.last_coeffs_or_eval.coeffs().evaluate(challenge);

		self.last_coeffs_or_eval = RoundState::Claim(eval);
	}
}

#[cfg(test)]
mod tests {
	use std::{array, iter};

	use binius_field::{Random, arch::OptimalPackedB128, field::FieldOps};
	use binius_ip::mlecheck;
	use binius_math::{
		multilinear::evaluate::evaluate,
		test_utils::{random_field_buffer, random_scalars},
	};
	use binius_transcript::{ProverTranscript, fiat_shamir::HasherChallenger};
	use itertools::Itertools;
	use rand::prelude::*;

	use super::*;
	use crate::sumcheck::{common::SumcheckProver, prove_single_mlecheck};

	type StdChallenger = HasherChallenger<sha2::Sha256>;

	// Prove one quadratic MLE-check via the shared store, then verify it through the verifier.
	// The verifier's reduced evaluation must equal the composition of the recovered column evals.
	// Each column must evaluate to its claimed value at the challenge point.
	// Prover and verifier challenges must agree.
	fn prove_verify<F, P, const N: usize>(
		composition: impl Fn([P; N]) -> P + Clone + Send + Sync,
		infinity_composition: impl Fn([P; N]) -> P + Send + Sync,
	) where
		F: Field,
		P: PackedField<Scalar = F>,
	{
		let n_vars = 8;
		let mut rng = StdRng::seed_from_u64(0);

		let multilinears: [_; N] = array::from_fn(|_| random_field_buffer::<P>(&mut rng, n_vars));

		// The honest claim is the composite MLE evaluated at the point.
		let composite_vals = (0..1 << n_vars.saturating_sub(P::LOG_WIDTH))
			.map(|i| composition(array::from_fn(|j| multilinears[j].as_ref()[i])))
			.collect_vec();
		let composite_vals = FieldBuffer::new(n_vars, composite_vals);
		let eval_point = random_scalars::<F>(&mut rng, n_vars);
		let eval_claim = evaluate(&composite_vals, &eval_point);

		let prover = quadratic_mlecheck_prover(
			multilinears.clone(),
			composition.clone(),
			infinity_composition,
			eval_point.clone(),
			eval_claim,
		);

		let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
		let output = prove_single_mlecheck(prover, &mut prover_transcript);
		prover_transcript
			.message()
			.write_slice(&output.multilinear_evals);

		let mut verifier_transcript = prover_transcript.into_verifier();
		let sumcheck_output = mlecheck::verify(
			&eval_point,
			2, // quadratic compositions have degree-2 round polynomials
			eval_claim,
			&mut verifier_transcript,
		)
		.unwrap();

		// The prover binds variables high-to-low.
		// `evaluate` expects them low-to-high, so reverse the challenges.
		let mut reduced_eval_point = sumcheck_output.challenges.clone();
		reduced_eval_point.reverse();

		let multilinear_evals: Vec<F> = verifier_transcript.message().read_vec(N).unwrap();

		// The reduced evaluation is the composition of the column evaluations.
		let evals_packed: [P; N] = array::from_fn(|i| P::broadcast(multilinear_evals[i]));
		assert_eq!(
			composition(evals_packed).iter().next().unwrap(),
			sumcheck_output.eval,
			"composition of the column evaluations should equal the reduced evaluation"
		);

		// Each column evaluates to its claimed value at the challenge point.
		for (multilinear, claimed_eval) in iter::zip(&multilinears, multilinear_evals) {
			assert_eq!(evaluate(multilinear, &reduced_eval_point), claimed_eval);
		}

		assert_eq!(
			output.challenges, sumcheck_output.challenges,
			"prover and verifier challenges should match"
		);
	}

	#[test]
	fn test_linear_mlecheck() {
		prove_verify::<_, OptimalPackedB128, 2>(
			|[a, b]| a + b,
			|[_a, _b]| OptimalPackedB128::zero(),
		);
	}

	#[test]
	fn test_bivariate_product_mlecheck() {
		prove_verify::<_, OptimalPackedB128, 2>(|[a, b]| a * b, |[a, b]| a * b);
	}

	#[test]
	fn test_mul_gate_mlecheck() {
		prove_verify::<_, OptimalPackedB128, 3>(|[a, b, c]| a * b - c, |[a, b, _c]| a * b);
	}

	#[test]
	fn test_4_variate_composition_mlecheck() {
		prove_verify::<_, OptimalPackedB128, 4>(
			|[a, b, c, d]| (a + b) * (c + d),
			|[a, b, c, d]| (a + b) * (c + d),
		);
	}

	#[test]
	fn test_round_claim_lerp_recovery() {
		type P = OptimalPackedB128;
		type F = <P as FieldOps>::Scalar;

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

		let mut prover = quadratic_mlecheck_prover(
			multilinears,
			composition,
			composition,
			eval_point,
			eval_claim,
		);

		let mut expected = vec![eval_claim];
		for _ in 0..n_vars {
			assert_eq!(prover.round_claim(), expected, "claim before execute");
			let round = prover.execute();
			assert_eq!(prover.round_claim(), expected, "claim recovered from coeffs");
			let challenge = F::random(&mut rng);
			expected = round.iter().map(|r| r.evaluate(challenge)).collect();
			prover.fold(challenge);
		}
	}
}
