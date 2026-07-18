// Copyright 2026 The Binius Developers

//! Round evaluators over a shared [`MleStore`] and the provers that drive them.
//!
//! A [`RoundEvaluator`] holds the per-round-polynomial logic for one composite claim over store
//! columns. Evaluators hold [`ColId`]s and receive column data by argument; they hold no mutable
//! per-round state — they neither fold nor track the round claim. The driving prover owns the
//! `RoundState` machine (the claim ↔ coeffs alternation) for each evaluator, and the store folds
//! the columns (see the [`mle_store`](super::mle_store) module documentation).
//!
//! [`SharedSumcheckProver`] adapts a store plus a list of evaluators — one per claim — to the
//! existing [`SumcheckProver`] interface, and [`SharedMleCheckProver`] to the [`MleCheckProver`]
//! interface, so they batch alongside standalone provers unchanged.

use std::iter;

use auto_impl::auto_impl;
use binius_field::{Field, PackedField, WideMul};
use binius_ip::sumcheck::RoundCoeffs;
use binius_math::FieldBuffer;

use super::{
	MleToSumCheckEvaluator,
	common::{MleCheckProver, SumcheckProver},
	mle_store::{ColId, EvaluationChunk, MleStore},
	round_state::RoundState,
};

/// Per-round-polynomial logic for one composite claim over store columns.
///
/// The driving prover makes one parallel pass over column chunks per round; the hot loops stay
/// monomorphized inside each evaluator and only the per-chunk [`Self::accumulate`] entry is
/// virtual. Within a round the calls are: [`Self::accumulate`] from parallel workers into
/// per-worker accumulator slices, then [`Self::interpolate`] once on the slot-wise summed slice.
/// The driving prover, not the evaluator, holds the round claim and reduces the round polynomial
/// against the verifier challenge; see [`SharedSumcheckProver`].
///
/// The accumulator is a flat slice of [`Self::degree`] wide (unreduced)
/// [`WideMul::Output`](WideMul::Output) slots. The driving prover owns the buffer, sizes it from
/// [`Self::degree`], and sums the workers' slices slot-wise, so evaluators implement neither
/// allocation nor merging — only the write pass and the interpolation. The slot layout within an
/// evaluator's run is private to that evaluator.
///
/// Evaluators are stateless across rounds: they hold no round claim and never fold. The prover
/// passes the round claim into [`Self::interpolate`] and recovers it back out of an emitted round
/// polynomial via [`Self::claim_from_coeffs`], so the whole claim ↔ coeffs state machine lives in
/// the prover's `RoundState`.
///
/// The `auto_impl(Box)` derive forwards the trait through `Box`, so a heterogeneous group of
/// evaluators can drive a shared prover as `Vec<Box<dyn RoundEvaluator<F, P>>>` while a homogeneous
/// group avoids boxing.
#[auto_impl(Box)]
pub trait RoundEvaluator<F: Field, P: PackedField<Scalar = F>>: Send + Sync {
	/// The number of accumulator slots this evaluator's claim uses.
	///
	/// This is the count of sampled round-polynomial evaluations the accumulation pass collects;
	/// the remaining evaluation is recovered from the round's sum claim in [`Self::interpolate`].
	/// The driving prover reserves this many slots for the evaluator. It is the degree of the
	/// accumulated (prime/composite) polynomial, which for an eq-factored MLE-check evaluator is
	/// the prime degree, not the emitted round-polynomial degree.
	fn degree(&self) -> usize;

	/// Recovers this round's claim from an emitted round polynomial.
	///
	/// The prover holds the round claim directly before [`Self::interpolate`] runs, but once it
	/// has replaced that claim with the round coefficients it recovers the (unchanged) claim from
	/// them for [`SumcheckProver::round_claim`]. A regular sumcheck evaluator recovers it as
	/// $R(0) + R(1)$ ([`RoundCoeffs::sum_over_endpoints`]); an MLE-check evaluator emitting prime
	/// polynomials recovers it as $(1 - \alpha) R(0) + \alpha R(1)$
	/// ([`RoundCoeffs::lerp_over_endpoints`]) with $\alpha$ read from the shared `store`.
	fn claim_from_coeffs(&self, store: &MleStore<'_, P>, coeffs: &RoundCoeffs<F>) -> F;

	/// Accumulates one chunk of the halved hypercube into `accum`.
	///
	/// The driving prover prepares `chunk` — the split, per-chunk column halves and eq-indicator
	/// expansions — so the evaluator only reads its columns by [`ColId`] and eq trackers by
	/// [`EqId`](super::mle_store::EqId). `accum` is this evaluator's run of [`Self::degree`] wide
	/// slots, zero-initialized on the first chunk and carried across the worker's chunks.
	fn accumulate(&self, chunk: &EvaluationChunk<'_, P>, accum: &mut [<P as WideMul>::Output]);

	/// Interpolates this round's polynomial from the slot-wise summed accumulator slice and the
	/// round claim.
	///
	/// `accum` is this evaluator's run of [`Self::degree`] slots, summed across all workers, and
	/// `claim` is the prover's round claim for this evaluator. Reads the shared `store` for the
	/// round's point coordinates; the store has not yet folded this round, so `store.n_vars()` and
	/// the eq trackers are at the current round's state.
	fn interpolate(
		&self,
		store: &MleStore<'_, P>,
		accum: &[<P as WideMul>::Output],
		claim: F,
	) -> RoundCoeffs<F>;
}

/// Maximum log2 chunk size of the parallel round pass.
///
/// Chunked accumulation keeps the equality-indicator chunk resident in L1 cache while all
/// evaluators read it, mirroring the chunking of the pre-store quadratic prover.
const MAX_CHUNK_VARS: usize = 12;

/// A [`SumcheckProver`] over a shared [`MleStore`] and a list of [`RoundEvaluator`]s, one per
/// claim.
///
/// Each round makes one parallel pass over the store's column chunks, feeding every evaluator,
/// and lists the round polynomials in evaluator registration order. [`Self::fold`] folds each
/// shared column and eq tracker once. [`Self::finish`] emits each store column's evaluation once,
/// computed a single time by the store no matter how many claims read the column.
pub struct SharedSumcheckProver<'a, P: PackedField, Evaluator> {
	store: MleStore<'a, P>,
	evaluators: Vec<Evaluator>,
	/// The claim ↔ round-coeffs state machine, one entry per evaluator (parallel to `evaluators`).
	///
	/// Holds each evaluator's current round claim before its round polynomial is produced, and the
	/// produced round coefficients afterwards. The prover — not the evaluator — advances this
	/// state: [`SumcheckProver::execute`] hands the claim to [`RoundEvaluator::interpolate`] and
	/// stores the resulting coefficients; [`SumcheckProver::fold`] reduces them against the
	/// challenge back to a claim.
	round_states: Vec<RoundState<RoundCoeffs<P::Scalar>, P::Scalar>>,
	/// A fold challenge whose store fold has been deferred so the next [`SumcheckProver::execute`]
	/// can fuse it into that round's read pass (see [`MleStore::map_reduce_with_fold`]). Only the
	/// store's column and eq fold waits here; the round claims advance eagerly in
	/// [`SumcheckProver::fold`].
	buffered_challenge: Option<P::Scalar>,
}

impl<'a, F, P, Evaluator> SharedSumcheckProver<'a, P, Evaluator>
where
	F: Field,
	P: PackedField<Scalar = F>,
	Evaluator: RoundEvaluator<F, P>,
{
	/// Creates a prover from a store and the evaluators reading its columns, each paired with its
	/// initial claim — one `(claim, evaluator)` per claim.
	///
	/// Taking an [`IntoIterator`] lets callers pass an array of pairs directly, rather than two
	/// parallel `Vec`s; the pairs are unzipped into the evaluators and their initial round states.
	pub fn new(
		store: MleStore<'a, P>,
		claims_with_evaluators: impl IntoIterator<Item = (F, Evaluator)>,
	) -> Self {
		let (round_states, evaluators) = claims_with_evaluators
			.into_iter()
			.map(|(claim, evaluator)| (RoundState::Claim(claim), evaluator))
			.unzip();
		Self {
			store,
			evaluators,
			round_states,
			buffered_challenge: None,
		}
	}

	/// Returns a shared reference to the underlying column store.
	pub const fn store(&self) -> &MleStore<'a, P> {
		&self.store
	}

	/// Pushes an owned column onto the store, returning its id.
	///
	/// Lets a caller extend the shared store with a fresh column that a later-added evaluator
	/// reads: the logUp* final layer pushes the table halves this way before adding its product
	/// evaluators. See [`MleStore::push_owned`].
	pub fn push_owned_column(&mut self, column: FieldBuffer<P>) -> ColId {
		self.store.push_owned(column)
	}

	/// Adds one more evaluator — a claim reading the shared store, with its initial claim — to the
	/// group.
	///
	/// Its round polynomial is appended after the existing evaluators' in [`Self::execute`].
	pub fn add_evaluator(&mut self, claim: F, evaluator: Evaluator) {
		self.evaluators.push(evaluator);
		self.round_states.push(RoundState::Claim(claim));
	}

	/// Prefix sums of each evaluator's [`RoundEvaluator::degree`] slot count.
	///
	/// The returned Vec has one more entry than there are evaluators; `offsets[i]..offsets[i + 1]`
	/// is evaluator `i`'s run in the flat accumulator buffer, and the last entry is its length.
	fn accum_offsets(&self) -> Vec<usize> {
		iter::once(0)
			.chain(self.evaluators.iter().scan(0, |acc, evaluator| {
				*acc += evaluator.degree();
				Some(*acc)
			}))
			.collect()
	}
}

impl<F, P, Evaluator> SumcheckProver<F> for SharedSumcheckProver<'_, P, Evaluator>
where
	F: Field,
	P: PackedField<Scalar = F>,
	Evaluator: RoundEvaluator<F, P>,
{
	fn n_vars(&self) -> usize {
		// A buffered challenge is a fold that has not yet reached the store, so the logical
		// remaining-variable count is one below the store's until the next execute applies it.
		self.store.n_vars() - self.buffered_challenge.is_some() as usize
	}

	fn n_claims(&self) -> usize {
		self.evaluators.len()
	}

	fn round_claim(&self) -> Vec<F> {
		// The claim is held directly before this round's polynomial is produced, and recovered from
		// that polynomial afterwards (each evaluator knows how to invert its own emission).
		iter::zip(&self.evaluators, &self.round_states)
			.map(|(evaluator, state)| match state {
				RoundState::Claim(claim) => *claim,
				RoundState::Coeffs(coeffs) => evaluator.claim_from_coeffs(&self.store, coeffs),
			})
			.collect()
	}

	fn execute(&mut self) -> Vec<RoundCoeffs<F>> {
		let n_vars_remaining = self.n_vars();
		assert!(n_vars_remaining > 0);

		// One parallel pass over the halved hypercube feeds every evaluator, so shared columns
		// and eq-indicator chunks are read once per round while they are cache-resident.
		//
		// TODO: dynamically choose chunk size based on the number of columns and P byte-size,
		// based on estimated L1 cache size.
		let chunk_vars = (n_vars_remaining - 1).min(MAX_CHUNK_VARS.max(P::LOG_WIDTH));

		// Each evaluator owns a contiguous run of `degree` wide slots in one flat per-worker
		// buffer; `offsets` holds the run boundaries as a prefix sum, so `offsets[i]..offsets[i +
		// 1]` is evaluator `i`'s slice.
		let offsets = self.accum_offsets();
		let total_slots = *offsets
			.last()
			.expect("offsets has one entry per evaluator, plus one");

		// The store prepares one `EvaluationChunk` per chunk of the halved hypercube — the split
		// column halves and eq-indicator expansions each evaluator reads.
		let buffered_challenge = self.buffered_challenge.take();
		let evaluators = &self.evaluators;
		let store = &mut self.store;
		let map = |chunk: EvaluationChunk<'_, P>| {
			let mut accum = vec![Default::default(); total_slots];
			for (evaluator, window) in iter::zip(evaluators, offsets.windows(2)) {
				evaluator.accumulate(&chunk, &mut accum[window[0]..window[1]]);
			}
			accum
		};
		let reduce = |mut lhs: Vec<<P as WideMul>::Output>, rhs: Vec<<P as WideMul>::Output>| {
			// The only merge: sum the workers' slices slot-wise, generic over every evaluator.
			for (dst, src) in iter::zip(&mut lhs, rhs) {
				*dst += src;
			}
			lhs
		};
		let accum = match buffered_challenge {
			// The previous fold deferred its store fold; apply it and this round's read in one
			// pass.
			Some(challenge) => store.map_reduce_with_fold(chunk_vars, challenge, map, reduce),
			None => store.map_reduce(chunk_vars, map, reduce),
		};

		// The store is read (immutably) for the round's point coordinates. Each evaluator takes its
		// current round claim (held in `round_states`) and produces its round polynomial; the
		// prover then records those coefficients as the round state for the coming fold.
		let store = &self.store;
		let round_coeffs: Vec<RoundCoeffs<F>> =
			iter::zip(iter::zip(&self.evaluators, &self.round_states), offsets.windows(2))
				.map(|((evaluator, state), window)| {
					let claim = *state.claim();
					evaluator.interpolate(store, &accum[window[0]..window[1]], claim)
				})
				.collect();
		for (state, coeffs) in iter::zip(&mut self.round_states, &round_coeffs) {
			*state = RoundState::Coeffs(coeffs.clone());
		}
		round_coeffs
	}

	fn fold(&mut self, challenge: F) {
		// Reduce each evaluator's round polynomial against the challenge to form its next claim.
		// The store's column and eq fold is deferred: the challenge is buffered so the next
		// execute can fuse it into that round's read pass.
		for state in &mut self.round_states {
			let claim = state.coeffs().evaluate(challenge);
			*state = RoundState::Claim(claim);
		}
		debug_assert!(
			self.buffered_challenge.is_none(),
			"fold called twice without an intervening execute"
		);
		self.buffered_challenge = Some(challenge);
	}

	fn finish(mut self) -> Vec<F> {
		// The last round's fold is still buffered; apply it to the store before reading
		// evaluations.
		if let Some(challenge) = self.buffered_challenge.take() {
			self.store.fold(challenge);
		}
		// The store owns each column once and computes its evaluation a single time, no matter how
		// many claims read it; emit every column's evaluation in store order.
		self.store.final_evals()
	}
}

/// A [`MleCheckProver`] over a shared [`MleStore`] and a group of [`RoundEvaluator`]s.
///
/// This is the MLE-check flavor of [`SharedSumcheckProver`]: every evaluator's claim shares the
/// prover's evaluation point, and the round polynomials are the eq-factored prime polynomials of
/// the MLE-check protocol.
pub struct SharedMleCheckProver<'a, F: Field, P: PackedField<Scalar = F>, Evaluator> {
	inner: SharedSumcheckProver<'a, P, Evaluator>,
	eval_point: Vec<F>,
}

impl<'a, F, P, Evaluator> SharedMleCheckProver<'a, F, P, Evaluator>
where
	F: Field,
	P: PackedField<Scalar = F>,
	Evaluator: RoundEvaluator<F, P>,
{
	/// Creates a prover from a store, the evaluators reading its columns each paired with its
	/// initial claim — one `(claim, evaluator)` per claim — and the evaluation point shared by all
	/// of the evaluators' claims.
	pub fn new(
		store: MleStore<'a, P>,
		claims_with_evaluators: impl IntoIterator<Item = (F, Evaluator)>,
		eval_point: Vec<F>,
	) -> Self {
		// precondition
		assert_eq!(
			eval_point.len(),
			store.n_vars(),
			"evaluation point length must equal the store's number of variables"
		);
		Self {
			inner: SharedSumcheckProver::new(store, claims_with_evaluators),
			eval_point,
		}
	}

	/// Converts this MLE-check prover into a plain [`SharedSumcheckProver`] by folding each claim's
	/// equality factor into its emitted round polynomials — the [Gruen24] technique of
	/// [`MleToSumCheckEvaluator`].
	///
	/// This lets the eq-weighted fractional claims batch in one evaluator group alongside plain
	/// sumcheck claims over the same store: the logUp* final layer converts the popped table
	/// layer's prover, then adds its product evaluators to it. The store and its columns carry over
	/// untouched; only the evaluators are wrapped.
	///
	/// [Gruen24]: <https://eprint.iacr.org/2024/108>
	pub fn into_shared_sumcheck(self) -> SharedSumcheckProver<'a, P, Box<dyn RoundEvaluator<F, P>>>
	where
		Evaluator: 'static,
	{
		let Self { inner, eval_point } = self;
		// Conversion happens before proving starts, so no fold challenge is buffered yet and every
		// round state is still an unreduced initial claim.
		let SharedSumcheckProver {
			mut store,
			evaluators,
			round_states,
			buffered_challenge: _,
		} = inner;
		// Every claim of an MLE-check prover shares the prover's evaluation point, so its eq
		// tracker is already registered on the store (by the evaluators reading it). Recover that
		// shared id — `register_eq_tracker` deduplicates — and hand it to each wrapper, which
		// reads the round's alpha and equality prefix from the tracker the store folds.
		let eq_tracker = store.register_eq_tracker(&eval_point);
		let evaluators = evaluators
			.into_iter()
			.map(|evaluator| {
				Box::new(MleToSumCheckEvaluator::new(evaluator, eq_tracker))
					as Box<dyn RoundEvaluator<F, P>>
			})
			.collect();
		// The wrappers emit the same claims: at the point of conversion no variable has been
		// folded, so the equality prefix is one and the sumcheck claim equals the inner MLE-check
		// claim.
		SharedSumcheckProver {
			store,
			evaluators,
			round_states,
			buffered_challenge: None,
		}
	}
}

impl<F, P, Evaluator> SumcheckProver<F> for SharedMleCheckProver<'_, F, P, Evaluator>
where
	F: Field,
	P: PackedField<Scalar = F>,
	Evaluator: RoundEvaluator<F, P>,
{
	fn n_vars(&self) -> usize {
		self.inner.n_vars()
	}

	fn n_claims(&self) -> usize {
		self.inner.n_claims()
	}

	fn round_claim(&self) -> Vec<F> {
		self.inner.round_claim()
	}

	fn execute(&mut self) -> Vec<RoundCoeffs<F>> {
		self.inner.execute()
	}

	fn fold(&mut self, challenge: F) {
		self.inner.fold(challenge)
	}

	fn finish(self) -> Vec<F> {
		self.inner.finish()
	}
}

impl<F, P, Evaluator> MleCheckProver<F> for SharedMleCheckProver<'_, F, P, Evaluator>
where
	F: Field,
	P: PackedField<Scalar = F>,
	Evaluator: RoundEvaluator<F, P>,
{
	fn eval_point(&self) -> &[F] {
		&self.eval_point[..self.inner.n_vars()]
	}
}

// Prove-and-verify coverage for the shared store provers: a batched fractional-addition MLE-check,
// and the logUp* final-layer shape (eq-weighted fractional addition batched with plain product
// claims over shared columns).
#[cfg(test)]
mod tests {
	use binius_field::FieldOps;
	use binius_ip::sumcheck::{batch_verify, batch_verify_mle};
	use binius_math::{
		FieldBuffer,
		inner_product::inner_product_par,
		multilinear::{eq::eq_ind, evaluate::evaluate},
		test_utils::{Packed128b, random_field_buffer, random_scalars},
		univariate::evaluate_univariate,
	};
	use binius_transcript::{ProverTranscript, fiat_shamir::HasherChallenger};
	use rand::prelude::*;

	use super::*;
	use crate::sumcheck::{
		MleToSumCheckEvaluator,
		batch::{batch_prove, batch_prove_mle},
		bivariate_product_evaluator::BivariateProductEvaluator,
		frac_add_mle,
	};

	type P = Packed128b;
	type F = <P as FieldOps>::Scalar;
	type StdChallenger = HasherChallenger<sha2::Sha256>;
	type CompFn = fn([P; 4]) -> P;

	// The fractional-addition numerator composition, as a single-claim function.
	fn comp_num([num_a, num_b, den_a, den_b]: [P; 4]) -> P {
		num_a * den_b + num_b * den_a
	}

	// The fractional-addition denominator composition, as a single-claim function.
	fn comp_den([_num_a, _num_b, den_a, den_b]: [P; 4]) -> P {
		den_a * den_b
	}

	// Split a multilinear on its highest variable into owned low and high halves.
	fn owned_halves(buffer: &FieldBuffer<P>) -> [FieldBuffer<P>; 2] {
		let (lo, hi) = buffer.split_half_ref();
		[
			FieldBuffer::new(lo.log_len(), lo.as_ref().into()),
			FieldBuffer::new(hi.log_len(), hi.as_ref().into()),
		]
	}

	// Random fractional-addition instance: four columns, evaluation point, and honest claims.
	fn frac_instance(rng: &mut StdRng, n_vars: usize) -> ([FieldBuffer<P>; 4], Vec<F>, [F; 2]) {
		let cols: [FieldBuffer<P>; 4] =
			std::array::from_fn(|_| random_field_buffer::<P>(&mut *rng, n_vars));
		let eval_point = random_scalars::<F>(&mut *rng, n_vars);

		// The honest claim is each composition's MLE evaluated at the point.
		let claims = [comp_num as CompFn, comp_den as CompFn].map(|comp| {
			let vals = (0..1usize << n_vars)
				.map(|i| {
					let scalars = [
						cols[0].get(i),
						cols[1].get(i),
						cols[2].get(i),
						cols[3].get(i),
					];
					comp(scalars.map(P::broadcast))
						.iter()
						.next()
						.expect("packed field has at least one lane")
				})
				.collect::<Vec<_>>();
			evaluate(&FieldBuffer::<P>::from_values(&vals), &eval_point)
		});

		(cols, eval_point, claims)
	}

	// The store + evaluator MLE-check prover for the two fractional-addition claims, borrowing the
	// four shared columns.
	fn new_frac_prover<'a>(
		cols: &'a [FieldBuffer<P>; 4],
		eval_point: &[F],
		claims: [F; 2],
	) -> SharedMleCheckProver<'a, F, P, Box<dyn RoundEvaluator<F, P>>> {
		let mut store = MleStore::new(eval_point.len());
		let col_ids = cols.each_ref().map(|col| store.push(col.to_ref()));
		let (num_ev, den_ev) = frac_add_mle::evaluators(&mut store, col_ids, eval_point.to_vec());
		let claims_with_evaluators: [(F, Box<dyn RoundEvaluator<F, P>>); 2] =
			[(claims[0], Box::new(num_ev)), (claims[1], Box::new(den_ev))];
		SharedMleCheckProver::new(store, claims_with_evaluators, eval_point.to_vec())
	}

	// Prove the two fractional-addition claims through the MLE-check batch driver, then verify.
	// The two claims share one store, so the four columns are folded and evaluated once.
	#[test]
	fn test_shared_frac_add_prove_verify() {
		for n_vars in [1, 2, 3, 8] {
			let mut rng = StdRng::seed_from_u64(0);
			let (cols, eval_point, claims) = frac_instance(&mut rng, n_vars);

			// Prove: one shared prover carries both claims over the four columns.
			let mut transcript = ProverTranscript::new(StdChallenger::default());
			let output =
				batch_prove_mle(vec![new_frac_prover(&cols, &eval_point, claims)], &mut transcript);

			// The shared prover emits the four column evaluations once, in push order.
			assert_eq!(output.multilinear_evals.len(), 1);
			let evals = output.multilinear_evals[0].clone();
			assert_eq!(evals.len(), 4);
			transcript.message().write_scalar_slice(&evals);

			// Verify: quadratic prime polynomials give degree-2 MLE-check rounds.
			let mut verifier = transcript.into_verifier();
			let sumcheck_output = batch_verify_mle(&eval_point, 2, &claims, &mut verifier).unwrap();
			let verified_evals: Vec<F> = verifier.message().read_vec(4).unwrap();
			assert_eq!(evals, verified_evals, "prover and verifier column evaluations must match");

			// The prover binds variables high-to-low; `evaluate` expects them low-to-high.
			let mut point = sumcheck_output.challenges.clone();
			point.reverse();

			// Each recovered column evaluation is the column's evaluation at the challenge point.
			for (col, &eval) in cols.iter().zip(&verified_evals) {
				assert_eq!(evaluate(col, &point), eval);
			}

			// The reduced evaluation is the batch combination of the two compositions at the evals.
			let packed = std::array::from_fn(|i| P::broadcast(verified_evals[i]));
			let composed = [comp_num, comp_den]
				.map(|comp| comp(packed).iter().next().expect("packed field has a lane"));
			let expected = evaluate_univariate(&composed, sumcheck_output.batch_coeff);
			assert_eq!(expected, sumcheck_output.eval, "reduced evaluation must match the batch");

			// Prover challenges, reversed, match the verifier's.
			let mut prover_challenges = output.challenges.clone();
			prover_challenges.reverse();
			assert_eq!(prover_challenges, sumcheck_output.challenges);
		}
	}

	// The logUp* final-layer shape: an eq-weighted fractional addition (two claims) batched with
	// two plain product claims, all sharing the pushforward halves in one store. Prove through the
	// sumcheck batch driver, then verify the reduced evaluation against the batched compositions.
	#[test]
	fn test_shared_final_layer_prove_verify() {
		for m in [1, 2, 3, 6] {
			let mut rng = StdRng::seed_from_u64(0);

			// Three parent buffers of m variables; each splits into two m-1 variable halves.
			let pushforward = random_field_buffer::<P>(&mut rng, m);
			let denominator = random_field_buffer::<P>(&mut rng, m);
			let table = random_field_buffer::<P>(&mut rng, m);
			let [y_0, y_1] = owned_halves(&pushforward);
			let [d_0, d_1] = owned_halves(&denominator);
			let [t_0, t_1] = owned_halves(&table);

			// Fractional claims at z; product claims are the inner products of the pushforward and
			// table halves.
			let z = random_scalars::<F>(&mut rng, m - 1);
			let frac_claims: [F; 2] = [comp_num as CompFn, comp_den as CompFn].map(|comp| {
				let vals = (0..1usize << (m - 1))
					.map(|i| {
						let scalars = [y_0.get(i), y_1.get(i), d_0.get(i), d_1.get(i)];
						comp(scalars.map(P::broadcast))
							.iter()
							.next()
							.expect("packed field has at least one lane")
					})
					.collect::<Vec<_>>();
				evaluate(&FieldBuffer::<P>::from_values(&vals), &z)
			});
			let e_0 = inner_product_par(&y_0, &t_0);
			let e_1 = inner_product_par(&y_1, &t_1);

			// One store with six borrowed columns, in push order [Y_0, Y_1, D_0, D_1, T_0, T_1].
			let mut store = MleStore::new(m - 1);
			let [y_0_col, y_1_col, d_0_col, d_1_col, t_0_col, t_1_col] =
				[&y_0, &y_1, &d_0, &d_1, &t_0, &t_1].map(|col| store.push(col.to_ref()));

			// The eq-weighted fractional evaluators, wrapped so they emit sumcheck round
			// polynomials.
			let (num_evaluator, den_evaluator) = frac_add_mle::evaluators(
				&mut store,
				[y_0_col, y_1_col, d_0_col, d_1_col],
				z.to_vec(),
			);
			let eq_tracker = store.register_eq_tracker(&z);
			let num_evaluator = MleToSumCheckEvaluator::new(num_evaluator, eq_tracker);
			let den_evaluator = MleToSumCheckEvaluator::new(den_evaluator, eq_tracker);

			// The two plain product claims over the pushforward and table halves.
			let product_0 = BivariateProductEvaluator::new([y_0_col, t_0_col]);
			let product_1 = BivariateProductEvaluator::new([y_1_col, t_1_col]);

			// Claims paired with evaluators in order: the two fractional claims, then the two
			// product sums.
			let claims_with_evaluators: [(F, Box<dyn RoundEvaluator<F, P>>); 4] = [
				(frac_claims[0], Box::new(num_evaluator)),
				(frac_claims[1], Box::new(den_evaluator)),
				(e_0, Box::new(product_0)),
				(e_1, Box::new(product_1)),
			];
			let shared = SharedSumcheckProver::new(store, claims_with_evaluators);

			// Prove and record the four claim sums in evaluator order.
			let mut transcript = ProverTranscript::new(StdChallenger::default());
			let output = batch_prove(vec![shared], &mut transcript);

			// The shared prover emits each store column's evaluation once, in push order.
			assert_eq!(output.multilinear_evals.len(), 1);
			let evals = output.multilinear_evals[0].clone();
			assert_eq!(evals.len(), 6);
			transcript.message().write_scalar_slice(&evals);

			// Verify: the eq-wrapped fractional rounds have degree 3, the product rounds degree 2,
			// so the batched round polynomial has degree 3.
			let claims = [frac_claims[0], frac_claims[1], e_0, e_1];
			let mut verifier = transcript.into_verifier();
			let sumcheck_output = batch_verify(m - 1, 3, &claims, &mut verifier).unwrap();
			let verified_evals: Vec<F> = verifier.message().read_vec(6).unwrap();
			assert_eq!(evals, verified_evals, "prover and verifier column evaluations must match");

			// The prover binds variables high-to-low; `evaluate` expects them low-to-high.
			let mut point = sumcheck_output.challenges.clone();
			point.reverse();

			// Each recovered column evaluation is the column's evaluation at the challenge point.
			for (col, &eval) in [&y_0, &y_1, &d_0, &d_1, &t_0, &t_1]
				.iter()
				.zip(&verified_evals)
			{
				assert_eq!(evaluate(col, &point), eval);
			}

			// The reduced evaluation batches the four claims' reduced compositions in evaluator
			// order. The fractional compositions carry the equality factor at z; the products do
			// not.
			let [y0, y1, d0, d1, t0, t1] =
				<[F; 6]>::try_from(verified_evals).expect("six column evaluations");
			let eq = eq_ind(&z, &point);
			let reduced = [(y0 * d1 + y1 * d0) * eq, (d0 * d1) * eq, y0 * t0, y1 * t1];
			let expected = evaluate_univariate(&reduced, sumcheck_output.batch_coeff);
			assert_eq!(expected, sumcheck_output.eval, "reduced evaluation must match the batch");

			// Prover challenges, reversed, match the verifier's.
			let mut prover_challenges = output.challenges.clone();
			prover_challenges.reverse();
			assert_eq!(prover_challenges, sumcheck_output.challenges);
		}
	}
}
