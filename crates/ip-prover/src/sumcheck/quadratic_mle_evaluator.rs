// Copyright 2026 The Binius Developers

use std::array;

use binius_field::{Field, PackedField, WideMul};
use binius_ip::sumcheck::RoundCoeffs;
use binius_math::FieldSlice;

use super::{
	mle_store::{ColId, EqId, MleStore},
	round_evals::RoundEvals2,
	round_evaluator::{RoundContext, RoundEvaluator},
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
	last_coeffs_or_eval: RoundCoeffsOrEval<P::Scalar>,
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
			last_coeffs_or_eval: RoundCoeffsOrEval::Eval(eval_claim),
		}
	}
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
			RoundCoeffsOrEval::Eval(eval) => *eval,
			RoundCoeffsOrEval::Coeffs(coeffs) => {
				let alpha = store.eq_alpha(self.eq_tracker);
				coeffs.lerp_over_endpoints(alpha)
			}
		}
	}

	fn accumulate(
		&self,
		ctx: &RoundContext<'_, '_, P>,
		chunk_index: usize,
		accum: &mut [<P as WideMul>::Output],
	) {
		let chunk_vars = ctx.chunk_vars();

		let eq_chunk = ctx
			.eq_expansion(self.eq_tracker)
			.chunk(chunk_vars, chunk_index);

		// Split each column into low/high halves for the top variable and take this pass's chunk:
		// the low half corresponds to x=0, the high half to x=1.
		let cols: [FieldSlice<'_, P>; N] = self.cols.map(|id| ctx.col(id));
		let halves: [_; N] = array::from_fn(|i| cols[i].split_half_ref());
		let lo_chunks: [_; N] = array::from_fn(|i| halves[i].0.chunk(chunk_vars, chunk_index));
		let hi_chunks: [_; N] = array::from_fn(|i| halves[i].1.chunk(chunk_vars, chunk_index));

		// The evaluator's run holds `y_1` in slot 0 and `y_inf` in slot 1.
		let mut y_1 = <P as WideMul>::Output::default();
		let mut y_inf = <P as WideMul>::Output::default();
		for (idx, &eq_i) in eq_chunk.as_ref().iter().enumerate() {
			// Gather the idx-th evaluations of every multilinear at both halves.
			let mut evals_1 = [P::default(); N];
			let mut evals_inf = [P::default(); N];

			for i in 0..N {
				let lo_i = lo_chunks[i].as_ref()[idx];
				let hi_i = hi_chunks[i].as_ref()[idx];

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
		let last_eval = match &self.last_coeffs_or_eval {
			RoundCoeffsOrEval::Eval(eval) => *eval,
			RoundCoeffsOrEval::Coeffs(_) => {
				panic!("interpolate called out of order; expected fold")
			}
		};

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
		self.last_coeffs_or_eval = RoundCoeffsOrEval::Coeffs(round_coeffs.clone());
		round_coeffs
	}

	fn fold(&mut self, challenge: F) {
		// State machine: fold consumes coeffs and produces the eval at the verifier challenge.
		let RoundCoeffsOrEval::Coeffs(coeffs) = &self.last_coeffs_or_eval else {
			panic!("fold called out of order; expected interpolate");
		};

		// Evaluate the round polynomial at the verifier's challenge to form the next claim. The
		// store folds the columns and the eq tracker (advancing its remaining count and alpha) with
		// the same challenge, so this only advances the claim state.
		let eval = coeffs.evaluate(challenge);

		self.last_coeffs_or_eval = RoundCoeffsOrEval::Eval(eval);
	}
}

#[derive(Debug, Clone)]
enum RoundCoeffsOrEval<F: Field> {
	Coeffs(RoundCoeffs<F>),
	Eval(F),
}
