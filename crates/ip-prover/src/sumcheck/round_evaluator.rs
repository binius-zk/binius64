// Copyright 2026 The Binius Developers

//! Round evaluators over a shared [`MleStore`] and the provers that drive them.
//!
//! A [`RoundEvaluator`] holds the per-round-polynomial logic for one composite claim over store
//! columns. Evaluators hold [`ColId`]s and receive column data by argument; they never fold and
//! hold no column references — the store folds (see the [`mle_store`](super::mle_store) module
//! documentation).
//!
//! [`SharedSumcheckProver`] adapts a store plus a list of evaluators — one per claim — to the
//! existing [`SumcheckProver`] interface, and [`SharedMleCheckProver`] to the [`MleCheckProver`]
//! interface, so they batch alongside standalone provers unchanged.

use std::iter;

use auto_impl::auto_impl;
use binius_field::{Field, PackedField, WideMul};
use binius_ip::sumcheck::RoundCoeffs;
use binius_math::{FieldBuffer, multilinear::eq::eq_one_var};

use super::{
	MleToSumCheckEvaluator,
	common::{MleCheckProver, SumcheckProver},
	mle_store::{ColId, EvaluationChunk, MleStore},
};

/// Per-round-polynomial logic for one composite claim over store columns.
///
/// The driving prover makes one parallel pass over column chunks per round; the hot loops stay
/// monomorphized inside each evaluator and only the per-chunk [`Self::accumulate`] entry is
/// virtual. Within a round the calls are: [`Self::accumulate`] from parallel workers into
/// per-worker accumulator slices, then [`Self::interpolate`] once on the slot-wise summed slice,
/// then [`Self::fold`] once.
///
/// The accumulator is a flat slice of [`Self::degree`] wide (unreduced)
/// [`WideMul::Output`](WideMul::Output) slots. The driving prover owns the buffer, sizes it from
/// [`Self::degree`], and sums the workers' slices slot-wise, so evaluators implement neither
/// allocation nor merging — only the write pass and the interpolation. The slot layout within an
/// evaluator's run is private to that evaluator.
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

	/// The current round claim.
	///
	/// Reads the shared `store` for the point coordinates it needs (the store has not yet folded
	/// this round, so `store.n_vars()` and the eq trackers are at the current round's state). See
	/// [`SumcheckProver::round_claim`] for the contract.
	fn round_claim(&self, store: &MleStore<'_, P>) -> F;

	/// Accumulates one chunk of the halved hypercube into `accum`.
	///
	/// The driving prover prepares `chunk` — the split, per-chunk column halves and eq-indicator
	/// expansions — so the evaluator only reads its columns by [`ColId`] and eq trackers by
	/// [`EqId`](super::mle_store::EqId). `accum` is this evaluator's run of [`Self::degree`] wide
	/// slots, zero-initialized on the first chunk and carried across the worker's chunks.
	fn accumulate(&self, chunk: &EvaluationChunk<'_, P>, accum: &mut [<P as WideMul>::Output]);

	/// Interpolates this round's polynomial from the slot-wise summed accumulator slice.
	///
	/// `accum` is this evaluator's run of [`Self::degree`] slots, summed across all workers. Reads
	/// the shared `store` for the round's point coordinates; the store has not yet folded this
	/// round, so `store.n_vars()` and the eq trackers are at the current round's state.
	fn interpolate(
		&mut self,
		store: &MleStore<'_, P>,
		accum: &[<P as WideMul>::Output],
	) -> RoundCoeffs<F>;

	/// Advances evaluator-local bookkeeping past a fold challenge.
	///
	/// The store folds the shared columns and eq trackers; this only updates claim state and
	/// equality prefix products.
	fn fold(&mut self, challenge: F);

	/// The number of padding variables this claim's columns still carry.
	///
	/// A full-length claim returns 0.
	/// A shorter claim returns a positive count that the driving prover spends over its leading
	/// rounds before the claim becomes active.
	/// All of a claim's columns share one padding, so this reports the padding of any one of them,
	/// read from the store.
	fn n_padding(&self, store: &MleStore<'_, P>) -> usize;
}

/// Maximum log2 chunk size of the parallel round pass.
///
/// Chunked accumulation keeps the equality-indicator chunk resident in L1 cache while all
/// evaluators read it, mirroring the chunking of the pre-store quadratic prover.
const MAX_CHUNK_VARS: usize = 12;

/// A [`SumcheckProver`] over a shared [`MleStore`] and a list of [`RoundEvaluator`]s, one per
/// claim.
///
/// Each round makes one parallel pass over the store's active column chunks, feeding every active
/// evaluator, and lists the round polynomials in evaluator order. Folding folds each shared active
/// column and eq tracker once. Finishing emits each store column's evaluation once, computed a
/// single time by the store.
///
/// # Padding
///
/// A claim whose columns are shorter than the store is padded up to it, so one prover can batch
/// sumchecks of unequal length.
/// The store folds only the active (zero-padding) columns each round.
/// This prover handles the padded claims.
///
/// The padding variables are the highest-indexed ones, so they are bound first.
/// While a claim is still binding one, it is in a padding round, where this prover:
/// - emits the degree-1 polynomial `v * (1 - X)`, with `v` the padding-scaled round claim,
/// - leaves that claim's evaluator and columns untouched, and
/// - multiplies that claim's equality-to-zero prefix by `eq(0, r)`.
///
/// Once a claim's padding rounds are done, its prefix is fixed.
/// Every later round polynomial and round claim is then scaled by it.
///
/// The prefix is `prod_k eq(0, r_k)` over the claim's padding challenges `r_k`.
/// Since `eq(0, .)` sums to 1 over the hypercube, scaling by it keeps the claim's sum unchanged.
/// So the batch stays sound.
///
/// The fused fold-and-read fast path is used whenever no column is padded — every current caller,
/// and every round of a padded batch once the short claims have caught up. While padding is live
/// the store is folded in a separate padding-aware pass.
pub struct SharedSumcheckProver<'a, P: PackedField, Evaluator> {
	store: MleStore<'a, P>,
	evaluators: Vec<Evaluator>,
	/// A fold challenge whose store fold has been deferred so the next [`SumcheckProver::execute`]
	/// can fuse it into that round's read pass (see [`MleStore::map_reduce_with_fold`]). The
	/// evaluators fold eagerly; only the store's column and eq fold waits here.
	buffered_challenge: Option<P::Scalar>,
	/// Accumulated equality-to-zero prefix of each claim, indexed like the evaluators.
	///
	/// This is `prod_k eq(0, r_k)` over the padding challenges bound so far.
	/// It stays `1` for a full-length claim.
	eq_prefixes: Vec<P::Scalar>,
}

impl<'a, F, P, Evaluator> SharedSumcheckProver<'a, P, Evaluator>
where
	F: Field,
	P: PackedField<Scalar = F>,
	Evaluator: RoundEvaluator<F, P>,
{
	/// Creates a prover from a store and the evaluators reading its columns, one per claim.
	pub fn new(store: MleStore<'a, P>, evaluators: Vec<Evaluator>) -> Self {
		// Every claim starts with the identity prefix; a padded claim grows its own over its
		// padding rounds.
		let eq_prefixes = vec![F::ONE; evaluators.len()];
		Self {
			store,
			evaluators,
			buffered_challenge: None,
			eq_prefixes,
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

	/// Adds one more evaluator — a claim reading the shared store — to the group.
	///
	/// Its round polynomial is appended after the existing evaluators' in [`Self::execute`].
	pub fn add_evaluator(&mut self, evaluator: Evaluator) {
		self.evaluators.push(evaluator);
		self.eq_prefixes.push(F::ONE);
	}

	/// Prefix sums of the accumulator-slot counts of the active (non-padded) evaluators.
	///
	/// The `k`-th active evaluator owns the window from entry `k` to entry `k + 1` of the flat
	/// accumulator buffer.
	/// The returned Vec has one more entry than there are active evaluators, the last being the
	/// buffer's total length.
	/// A padded evaluator holds no slots this round.
	fn active_accum_offsets(&self, active: &[bool]) -> Vec<usize> {
		iter::once(0)
			.chain(
				iter::zip(&self.evaluators, active)
					.filter(|(_, is_active)| **is_active)
					.scan(0, |acc, (evaluator, _)| {
						*acc += evaluator.degree();
						Some(*acc)
					}),
			)
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
		// Each claim's round claim is its evaluator's claim scaled by the equality-to-zero prefix.
		// A padded claim's evaluator is untouched, so its claim is the original sum.
		// A full-length claim has prefix 1, so it passes through.
		iter::zip(&self.evaluators, &self.eq_prefixes)
			.map(|(evaluator, &eq_prefix)| evaluator.round_claim(&self.store) * eq_prefix)
			.collect()
	}

	fn execute(&mut self) -> Vec<RoundCoeffs<F>> {
		let n_vars_remaining = self.n_vars();
		assert!(n_vars_remaining > 0);

		// TODO: dynamically choose chunk size based on the number of columns and P byte-size,
		// based on estimated L1 cache size.
		let chunk_vars = (n_vars_remaining - 1).min(MAX_CHUNK_VARS.max(P::LOG_WIDTH));

		// The buffered fold can be fused into this round's read only when it folds every column.
		// With padded columns present, apply it eagerly (padding-aware) first, then read.
		let buffered_challenge = self.buffered_challenge.take();
		let fused = buffered_challenge.is_some() && !self.store.has_padded_columns();
		if let Some(challenge) = buffered_challenge
			&& !fused
		{
			self.store.fold(challenge);
		}

		// A claim binding a padding variable this round sits out the shared column pass and emits a
		// synthetic polynomial; the rest are active. On the fused path no column is padded, so all
		// are active.
		let active: Vec<bool> = self
			.evaluators
			.iter()
			.map(|evaluator| evaluator.n_padding(&self.store) == 0)
			.collect();

		// Each active evaluator owns a contiguous slot run in one flat per-worker buffer.
		let offsets = self.active_accum_offsets(&active);
		let total_slots = *offsets.last().expect("offsets has the leading zero");

		// One parallel pass over the halved hypercube feeds every active evaluator, so shared
		// active columns and eq-indicator chunks are read once per round, while resident in
		// cache.
		let evaluators = &self.evaluators;
		let active_ref = &active;
		let offsets_ref = &offsets;
		let store = &mut self.store;
		let map = |chunk: EvaluationChunk<'_, P>| {
			let mut accum = vec![Default::default(); total_slots];
			// Hand each active evaluator its slot run; a padded evaluator holds no slots.
			let mut slot = 0;
			for (evaluator, &is_active) in iter::zip(evaluators, active_ref) {
				if is_active {
					evaluator
						.accumulate(&chunk, &mut accum[offsets_ref[slot]..offsets_ref[slot + 1]]);
					slot += 1;
				}
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
		let accum = if fused {
			// The buffered fold folds every column and this round's read in one pass.
			let challenge = buffered_challenge.expect("a fused round has a buffered challenge");
			store.map_reduce_with_fold(chunk_vars, challenge, map, reduce)
		} else {
			// The store already reflects this round; just read it.
			store.map_reduce(chunk_vars, map, reduce)
		};

		// Assemble the round polynomials in evaluator order.
		// The store and prefixes are read while the evaluators are interpolated mutably; they are
		// disjoint fields, so borrow them separately.
		let store = &self.store;
		let eq_prefixes = &self.eq_prefixes;
		let mut slot = 0;
		iter::zip(&mut self.evaluators, &active)
			.enumerate()
			.map(|(i, (evaluator, &is_active))| {
				if is_active {
					// Active claim: interpolate from the pass, then scale by the frozen prefix.
					let coeffs =
						evaluator.interpolate(store, &accum[offsets[slot]..offsets[slot + 1]]);
					slot += 1;
					coeffs * eq_prefixes[i]
				} else {
					// Padding round: emit `v * (1 - X)`, with `v` the scaled round claim.
					// With `eq(0, X) = 1 - X`, its coefficients are `[v, -v]`.
					// The evaluator and store are left untouched.
					let v = evaluator.round_claim(store) * eq_prefixes[i];
					RoundCoeffs(vec![v, -v])
				}
			})
			.collect()
	}

	fn fold(&mut self, challenge: F) {
		// A claim binding a padding variable this round only grows its equality-to-zero prefix; its
		// evaluator and columns are untouched. The store reflects this round, so a positive padding
		// marks a padding round for that claim.
		let padded: Vec<bool> = self
			.evaluators
			.iter()
			.map(|evaluator| evaluator.n_padding(&self.store) > 0)
			.collect();

		// `eq(0, challenge) = 1 - challenge` is the factor a padding round contributes.
		let eq_zero = eq_one_var(F::ZERO, challenge);
		for (i, evaluator) in self.evaluators.iter_mut().enumerate() {
			if padded[i] {
				self.eq_prefixes[i] *= eq_zero;
			} else {
				evaluator.fold(challenge);
			}
		}

		// The store's column and eq fold is deferred: the challenge is buffered so the next execute
		// can fuse it (or, with padding live, apply it in a padding-aware pass).
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
	/// Creates a prover from a store, the evaluators reading its columns — one per claim — and the
	/// evaluation point shared by all of the evaluators' claims.
	pub fn new(store: MleStore<'a, P>, evaluators: Vec<Evaluator>, eval_point: Vec<F>) -> Self {
		// precondition
		assert_eq!(
			eval_point.len(),
			store.n_vars(),
			"evaluation point length must equal the store's number of variables"
		);
		Self {
			inner: SharedSumcheckProver::new(store, evaluators),
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
		// Conversion happens before proving starts, so no fold challenge is buffered and every
		// equality-to-zero prefix is still the identity; the wrapped prover rebuilds them.
		let SharedSumcheckProver {
			mut store,
			evaluators,
			buffered_challenge: _,
			eq_prefixes: _,
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
		SharedSumcheckProver::new(store, evaluators)
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
	use binius_field::{FieldOps, Random};
	use binius_ip::sumcheck::{RoundCoeffs, batch_verify, batch_verify_mle};
	use binius_math::{
		FieldBuffer,
		inner_product::inner_product_par,
		multilinear::{
			eq::{eq_ind, eq_ind_zero},
			evaluate::evaluate,
		},
		test_utils::{Packed128b, random_field_buffer, random_scalars},
		univariate::evaluate_univariate,
	};
	use binius_transcript::{ProverTranscript, fiat_shamir::HasherChallenger};
	use proptest::prelude::*;
	use rand::prelude::*;

	use super::*;
	use crate::sumcheck::{
		MleToSumCheckEvaluator, PaddedSumcheckDecorator,
		batch::{batch_prove, batch_prove_mle},
		bivariate_product_evaluator::{BivariateProductEvaluator, bivariate_product_prover},
		common::SumcheckProver,
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
		let (num_ev, den_ev) =
			frac_add_mle::evaluators(&mut store, col_ids, eval_point.to_vec(), claims);
		let evaluators: Vec<Box<dyn RoundEvaluator<F, P>>> =
			vec![Box::new(num_ev), Box::new(den_ev)];
		SharedMleCheckProver::new(store, evaluators, eval_point.to_vec())
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
				frac_claims,
			);
			let eq_tracker = store.register_eq_tracker(&z);
			let num_evaluator = MleToSumCheckEvaluator::new(num_evaluator, eq_tracker);
			let den_evaluator = MleToSumCheckEvaluator::new(den_evaluator, eq_tracker);

			// The two plain product claims over the pushforward and table halves.
			let product_0 = BivariateProductEvaluator::new([y_0_col, t_0_col], e_0);
			let product_1 = BivariateProductEvaluator::new([y_1_col, t_1_col], e_1);

			let evaluators: Vec<Box<dyn RoundEvaluator<F, P>>> = vec![
				Box::new(num_evaluator),
				Box::new(den_evaluator),
				Box::new(product_0),
				Box::new(product_1),
			];
			let shared = SharedSumcheckProver::new(store, evaluators);

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

	// One product claim `<a_i, b_i> = s_i` per size, with random columns and honest sums.
	fn product_instances(
		sizes: &[usize],
		rng: &mut StdRng,
	) -> Vec<(FieldBuffer<P>, FieldBuffer<P>, F)> {
		sizes
			.iter()
			.map(|&n| {
				let a = random_field_buffer::<P>(&mut *rng, n);
				let b = random_field_buffer::<P>(&mut *rng, n);
				let sum = inner_product_par(&a, &b);
				(a, b, sum)
			})
			.collect()
	}

	// One shared prover over a `max_n` store, holding every claim's columns padded up to `max_n`.
	fn padded_product_shared(
		instances: &[(FieldBuffer<P>, FieldBuffer<P>, F)],
		max_n: usize,
	) -> SharedSumcheckProver<'_, P, BivariateProductEvaluator<P>> {
		let mut store = MleStore::new(max_n);
		let mut evaluators = Vec::with_capacity(instances.len());
		for (a, b, sum) in instances {
			// Columns shorter than the store are padded up to it automatically.
			let a_col = store.push(a.to_ref());
			let b_col = store.push(b.to_ref());
			evaluators.push(BivariateProductEvaluator::new([a_col, b_col], *sum));
		}
		SharedSumcheckProver::new(store, evaluators)
	}

	// The reference batch: one `PaddedSumcheckDecorator` per claim, each padding its own bivariate
	// product prover up to `max_n`.
	fn padded_product_reference(
		instances: &[(FieldBuffer<P>, FieldBuffer<P>, F)],
		max_n: usize,
	) -> Vec<
		PaddedSumcheckDecorator<F, SharedSumcheckProver<'static, P, BivariateProductEvaluator<P>>>,
	> {
		instances
			.iter()
			.map(|(a, b, sum)| {
				let inner = bivariate_product_prover([a.clone(), b.clone()], *sum);
				PaddedSumcheckDecorator::new(inner, max_n - a.log_len())
			})
			.collect()
	}

	// The shared padded prover reproduces the `PaddedSumcheckDecorator` batch exactly: identical
	// round claims, round polynomials, and final evaluations on the same challenge sequence.
	fn assert_padded_matches_reference(sizes: &[usize], seed: u64) {
		let mut rng = StdRng::seed_from_u64(seed);
		let max_n = *sizes.iter().max().expect("at least one claim");
		let instances = product_instances(sizes, &mut rng);

		let mut reference = padded_product_reference(&instances, max_n);
		let mut shared = padded_product_shared(&instances, max_n);

		assert_eq!(shared.n_vars(), max_n);
		assert_eq!(shared.n_claims(), sizes.len());

		let mut challenge_rng = StdRng::seed_from_u64(seed.wrapping_add(1));
		for _ in 0..max_n {
			assert_eq!(shared.n_vars(), reference[0].n_vars());

			// Round claims agree, whether read before or (implicitly) after execute.
			let reference_claims: Vec<F> = reference
				.iter()
				.flat_map(|prover| prover.round_claim())
				.collect();
			assert_eq!(
				shared.round_claim(),
				reference_claims,
				"round claims must match the reference"
			);

			// Round polynomials agree, batched in the same claim order.
			let reference_coeffs: Vec<RoundCoeffs<F>> = reference
				.iter_mut()
				.flat_map(|prover| prover.execute())
				.collect();
			assert_eq!(
				shared.execute(),
				reference_coeffs,
				"round polynomials must match the reference"
			);

			let challenge = F::random(&mut challenge_rng);
			for prover in &mut reference {
				prover.fold(challenge);
			}
			shared.fold(challenge);
		}

		let reference_evals: Vec<F> = reference
			.into_iter()
			.flat_map(|prover| prover.finish())
			.collect();
		assert_eq!(shared.finish(), reference_evals, "final evaluations must match the reference");
	}

	// A full prove/verify roundtrip of the shared padded batch through a real transcript and the
	// standard batched sumcheck verifier.
	fn assert_padded_prove_verify(sizes: &[usize], seed: u64) {
		let mut rng = StdRng::seed_from_u64(seed);
		let max_n = *sizes.iter().max().expect("at least one claim");
		let instances = product_instances(sizes, &mut rng);
		let claims: Vec<F> = instances.iter().map(|(_, _, sum)| *sum).collect();

		let shared = padded_product_shared(&instances, max_n);
		let mut transcript = ProverTranscript::new(StdChallenger::default());
		let output = batch_prove(vec![shared], &mut transcript);

		// One shared prover emits every column evaluation once, in push order a_0, b_0, a_1, ...
		assert_eq!(output.multilinear_evals.len(), 1);
		let evals = output.multilinear_evals[0].clone();
		assert_eq!(evals.len(), 2 * sizes.len());
		transcript.message().write_scalar_slice(&evals);

		// The largest claim is never padded, so every batched round polynomial has degree 2.
		let mut verifier = transcript.into_verifier();
		let sumcheck_output = batch_verify(max_n, 2, &claims, &mut verifier).unwrap();
		let verified_evals: Vec<F> = verifier.message().read_vec(2 * sizes.len()).unwrap();
		assert_eq!(evals, verified_evals, "prover and verifier column evaluations must match");

		// Binding is high-to-low, so reverse for `evaluate`, which wants low-to-high.
		let mut point = sumcheck_output.challenges.clone();
		point.reverse();

		// Each claim's reduced evaluation is its product at the point times its equality-to-zero
		// padding factor over the high (padding) coordinates.
		let mut composed = Vec::with_capacity(sizes.len());
		for (i, (a, b, _)) in instances.iter().enumerate() {
			let n_i = a.log_len();
			let a_eval = verified_evals[2 * i];
			let b_eval = verified_evals[2 * i + 1];
			// A padded column binds only its active rounds: the low `n_i` coordinates of the point.
			assert_eq!(evaluate(a, &point[..n_i]), a_eval);
			assert_eq!(evaluate(b, &point[..n_i]), b_eval);
			composed.push(a_eval * b_eval * eq_ind_zero(&point[n_i..]));
		}
		let expected = evaluate_univariate(&composed, sumcheck_output.batch_coeff);
		assert_eq!(expected, sumcheck_output.eval, "reduced evaluation must match the batch");

		let mut prover_challenges = output.challenges;
		prover_challenges.reverse();
		assert_eq!(prover_challenges, sumcheck_output.challenges);
	}

	#[test]
	fn test_padded_batch_matches_decorator_reference() {
		// A single full-length claim: no padding, a plain passthrough.
		assert_padded_matches_reference(&[8], 0);
		// Equal sizes: still no padding, the ordinary shared-prover path.
		assert_padded_matches_reference(&[6, 6, 6], 1);
		// One shorter claim padded up to the longer, in both orders.
		assert_padded_matches_reference(&[5, 8], 2);
		assert_padded_matches_reference(&[8, 5], 3);
		// A mix of sizes with the largest active from the first round.
		assert_padded_matches_reference(&[3, 6, 8], 4);
		// A claim padded by all but one variable.
		assert_padded_matches_reference(&[1, 8], 5);
		// A zero-variable claim, padded every round and never active.
		assert_padded_matches_reference(&[0, 4, 8], 6);
		// Several claims at several padding depths at once.
		assert_padded_matches_reference(&[8, 8, 3, 1], 7);
	}

	#[test]
	fn test_padded_batch_prove_verify() {
		// The same size mixes as the reference test, driven end-to-end through a real transcript.
		assert_padded_prove_verify(&[8], 10);
		assert_padded_prove_verify(&[7, 7], 11);
		assert_padded_prove_verify(&[5, 8], 12);
		assert_padded_prove_verify(&[3, 6, 8], 13);
		assert_padded_prove_verify(&[0, 4, 8], 14);
		assert_padded_prove_verify(&[8, 2, 6, 1], 15);
	}

	proptest! {
		#![proptest_config(ProptestConfig::with_cases(24))]

		#[test]
		fn test_padded_batch_matches_reference_proptest(
			// Any mix of 1 to 5 claims, each 0 to 7 variables, sharing a common maximum width.
			sizes in prop::collection::vec(0usize..=7, 1..=5),
			seed: u64,
		) {
			// For every such mix, the shared padded prover must match the decorator batch exactly.
			assert_padded_matches_reference(&sizes, seed);
		}
	}
}
