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

use std::{cmp::max, iter};

use auto_impl::auto_impl;
use binius_field::{Field, PackedField, WideMul};
use binius_ip::sumcheck::RoundCoeffs;
use binius_math::{FieldBuffer, FieldSlice};
use binius_utils::rayon::prelude::*;

use super::{
	MleToSumCheckEvaluator,
	common::{MleCheckProver, SumcheckProver},
	mle_store::{ColId, EqId, MleStore},
};

/// One store column's low and high halves at a single chunk of the halved hypercube.
///
/// The column is split on the round's highest variable: `lo` fixes that variable to 0, `hi` to 1.
/// Both range over the chunk's `2^chunk_vars` scalars.
pub struct ColumnChunk<'c, P: PackedField> {
	pub lo: FieldSlice<'c, P>,
	pub hi: FieldSlice<'c, P>,
}

/// One chunk of the halved hypercube, prepared by the driving prover for the round evaluators.
///
/// The pass runs over the halved hypercube: with `n` variables remaining, each column splits on
/// the highest variable into two halves of `n - 1` variables, and both halves divide into chunks
/// of `2^chunk_vars` scalars.
///
/// The driver splits every store column on the round's highest variable and slices this chunk out
/// of each half, and slices the same chunk out of each eq-indicator expansion — once per chunk, so
/// a column read by several evaluators is split and chunked a single time. Evaluators read their
/// columns by [`ColId`] and their eq trackers by [`EqId`].
pub struct EvaluationChunk<'c, P: PackedField> {
	cols: Vec<ColumnChunk<'c, P>>,
	eqs: Vec<FieldSlice<'c, P>>,
}

impl<'c, P: PackedField> EvaluationChunk<'c, P> {
	/// Returns the low and high halves of a column at this chunk.
	pub fn col(&self, id: ColId) -> &ColumnChunk<'c, P> {
		&self.cols[id.index()]
	}

	/// Returns the equality-indicator expansion of a registered tracker at this chunk.
	///
	/// The expansion ranges over the halved hypercube, so it is chunked with the same chunk index
	/// as the column halves.
	pub fn eq(&self, id: EqId) -> &FieldSlice<'c, P> {
		&self.eqs[id.index()]
	}
}

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
	/// [`EqId`]. `accum` is this evaluator's run of [`Self::degree`] wide slots, zero-initialized
	/// on the first chunk and carried across the worker's chunks.
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
}

impl<'a, F, P, Evaluator> SharedSumcheckProver<'a, P, Evaluator>
where
	F: Field,
	P: PackedField<Scalar = F>,
	Evaluator: RoundEvaluator<F, P>,
{
	/// Creates a prover from a store and the evaluators reading its columns, one per claim.
	pub const fn new(store: MleStore<'a, P>, evaluators: Vec<Evaluator>) -> Self {
		Self { store, evaluators }
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
		self.store.n_vars()
	}

	fn n_claims(&self) -> usize {
		self.evaluators.len()
	}

	fn round_claim(&self) -> Vec<F> {
		self.evaluators
			.iter()
			.map(|evaluator| evaluator.round_claim(&self.store))
			.collect()
	}

	fn execute(&mut self) -> Vec<RoundCoeffs<F>> {
		let n_vars_remaining = self.store.n_vars();
		assert!(n_vars_remaining > 0);

		// One parallel pass over the halved hypercube feeds every evaluator, so shared columns
		// and eq-indicator chunks are read once per round while they are cache-resident.
		let chunk_vars = max(MAX_CHUNK_VARS, P::LOG_WIDTH).min(n_vars_remaining - 1);
		let chunk_count = 1usize << (n_vars_remaining - 1 - chunk_vars);

		// Each evaluator owns a contiguous run of `degree` wide slots in one flat per-worker
		// buffer; `offsets` holds the run boundaries as a prefix sum, so `offsets[i]..offsets[i +
		// 1]` is evaluator `i`'s slice.
		let offsets = self.accum_offsets();
		let total_slots = *offsets
			.last()
			.expect("offsets has one entry per evaluator, plus one");

		// Split every column on the round's highest variable once; the halves are re-sliced per
		// chunk below into the `EvaluationChunk` each evaluator reads. The eq-indicator expansions
		// already range over the halved hypercube, so they are chunked directly.
		let col_slices = self.store.column_slices();
		let col_halves = col_slices
			.iter()
			.map(|col| col.split_half_ref())
			.collect::<Vec<_>>();
		let eq_expansions = self.store.eq_expansions();

		let evaluators = &self.evaluators;
		let new_accum = || -> Vec<<P as WideMul>::Output> { vec![Default::default(); total_slots] };
		let accum = (0..chunk_count)
			.into_par_iter()
			.fold(new_accum, |mut accum, chunk_index| {
				// Prepare this chunk's column halves and eq expansions once for every evaluator.
				let chunk = EvaluationChunk {
					cols: col_halves
						.iter()
						.map(|(lo, hi)| ColumnChunk {
							lo: lo.chunk(chunk_vars, chunk_index),
							hi: hi.chunk(chunk_vars, chunk_index),
						})
						.collect(),
					eqs: eq_expansions
						.iter()
						.map(|eq| eq.chunk(chunk_vars, chunk_index))
						.collect(),
				};
				for (evaluator, window) in iter::zip(evaluators, offsets.windows(2)) {
					evaluator.accumulate(&chunk, &mut accum[window[0]..window[1]]);
				}
				accum
			})
			.reduce(new_accum, |mut lhs, rhs| {
				// The only merge: sum the workers' slices slot-wise, generic over every evaluator.
				for (dst, src) in iter::zip(&mut lhs, rhs) {
					*dst += src;
				}
				lhs
			});

		// The store is read (immutably) for the round's point coordinates while the evaluators are
		// interpolated (mutably); they are disjoint fields, so borrow the store separately.
		let store = &self.store;
		iter::zip(&mut self.evaluators, offsets.windows(2))
			.map(|(evaluator, window)| evaluator.interpolate(store, &accum[window[0]..window[1]]))
			.collect()
	}

	fn fold(&mut self, challenge: F) {
		// The store folds — columns and eq trackers both; evaluators only advance local
		// bookkeeping such as claim state and equality prefix products.
		self.store.fold(challenge);
		for evaluator in &mut self.evaluators {
			evaluator.fold(challenge);
		}
	}

	fn finish(self) -> Vec<F> {
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
		let SharedSumcheckProver {
			mut store,
			evaluators,
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

// Differential equivalence tests: identical claims run through the pre-store provers and the
// store + evaluator path, asserting byte-identical round polynomials, challenges, and emitted
// evaluations.
#[cfg(test)]
mod tests {
	use binius_field::{FieldOps, Random};
	use binius_math::{
		FieldBuffer,
		inner_product::inner_product_par,
		multilinear::evaluate::evaluate,
		test_utils::{Packed128b, random_field_buffer, random_scalars},
	};
	use binius_transcript::{ProverTranscript, fiat_shamir::HasherChallenger};
	use either::Either;
	use rand::prelude::*;

	use super::*;
	use crate::sumcheck::{
		MleToSumCheckDecorator, MleToSumCheckEvaluator,
		batch::{batch_prove, batch_prove_mle},
		bivariate_product::BivariateProductSumcheckProver,
		bivariate_product_evaluator::BivariateProductEvaluator,
		frac_add_mle,
		quadratic_mle::QuadraticMleCheckProver,
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

	// The pre-store single-claim provers for the two fractional-addition compositions.
	fn old_frac_provers(
		cols: &[FieldBuffer<P>; 4],
		eval_point: &[F],
		claims: [F; 2],
	) -> [QuadraticMleCheckProver<P, CompFn, CompFn, 4>; 2] {
		[
			(comp_num as CompFn, claims[0]),
			(comp_den as CompFn, claims[1]),
		]
		.map(|(comp, claim)| {
			QuadraticMleCheckProver::new(cols.clone(), comp, comp, eval_point.to_vec(), claim)
		})
	}

	// The store + evaluator path for the same claims, with borrowed columns.
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

	// Lockstep comparison against the surviving single-claim provers: round claims, round
	// polynomials, evaluation points, and final evaluations must all match exactly.
	#[test]
	fn test_shared_frac_add_matches_single_provers_lockstep() {
		for n_vars in [1, 2, 3, 8] {
			let mut rng = StdRng::seed_from_u64(0);
			let (cols, eval_point, claims) = frac_instance(&mut rng, n_vars);

			let [mut old_num, mut old_den] = old_frac_provers(&cols, &eval_point, claims);
			let mut shared = new_frac_prover(&cols, &eval_point, claims);

			for _round in 0..n_vars {
				assert_eq!(
					shared.round_claim(),
					[old_num.round_claim(), old_den.round_claim()].concat(),
					"round claims must match before execute (n_vars={n_vars})"
				);
				assert_eq!(shared.eval_point(), old_num.eval_point());

				let new_coeffs = shared.execute();
				let old_coeffs = [old_num.execute(), old_den.execute()].concat();
				assert_eq!(
					new_coeffs, old_coeffs,
					"round polynomials must match (n_vars={n_vars})"
				);

				assert_eq!(
					shared.round_claim(),
					[old_num.round_claim(), old_den.round_claim()].concat(),
					"round claims must match after execute (n_vars={n_vars})"
				);

				let challenge = F::random(&mut rng);
				shared.fold(challenge);
				old_num.fold(challenge);
				old_den.fold(challenge);
			}

			let old_evals = old_num.finish();
			assert_eq!(old_den.finish(), old_evals);
			assert_eq!(
				shared.finish(),
				old_evals,
				"final evaluations must match (n_vars={n_vars})"
			);
		}
	}

	// Full-transcript comparison through the MLE-check batch driver: the store path must produce
	// byte-identical transcripts to the pre-store provers.
	#[test]
	fn test_shared_frac_add_transcript_matches_single_provers() {
		for n_vars in [1, 3, 8] {
			let mut rng = StdRng::seed_from_u64(0);
			let (cols, eval_point, claims) = frac_instance(&mut rng, n_vars);

			let mut old_transcript = ProverTranscript::new(StdChallenger::default());
			let old_output = batch_prove_mle(
				Vec::from(old_frac_provers(&cols, &eval_point, claims)),
				&mut old_transcript,
			);

			let mut new_transcript = ProverTranscript::new(StdChallenger::default());
			let new_output = batch_prove_mle(
				vec![new_frac_prover(&cols, &eval_point, claims)],
				&mut new_transcript,
			);

			assert_eq!(old_output.challenges, new_output.challenges);
			// Both single provers evaluate all four shared columns; the store path emits the
			// per-claim-group evaluations of its one evaluator.
			assert_eq!(old_output.multilinear_evals[1], old_output.multilinear_evals[0]);
			assert_eq!(new_output.multilinear_evals, vec![old_output.multilinear_evals[0].clone()]);
			assert_eq!(
				old_transcript.finalize(),
				new_transcript.finalize(),
				"transcripts must be byte-identical (n_vars={n_vars})"
			);
		}
	}

	// Full-transcript comparison of the logUp* final-layer shape: an eq-weighted fractional
	// addition batched with two plain product claims, sharing the pushforward columns.
	#[test]
	fn test_shared_final_layer_transcript_matches_old_provers() {
		for m in [1, 2, 3, 6] {
			let mut rng = StdRng::seed_from_u64(0);

			let pushforward = random_field_buffer::<P>(&mut rng, m);
			let denominator = random_field_buffer::<P>(&mut rng, m);
			let table = random_field_buffer::<P>(&mut rng, m);

			let [y_0, y_1] = owned_halves(&pushforward);
			let [d_0, d_1] = owned_halves(&denominator);
			let [t_0, t_1] = owned_halves(&table);

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

			// Old path: a decorated MLE-check prover per fractional claim and a standalone
			// product prover per product claim, each on its own copies of the shared columns.
			let frac_cols = [y_0.clone(), y_1.clone(), d_0.clone(), d_1.clone()];
			let [old_num, old_den] = old_frac_provers(&frac_cols, &z, frac_claims);
			let old_provers = vec![
				Either::Left(MleToSumCheckDecorator::new(old_num)),
				Either::Left(MleToSumCheckDecorator::new(old_den)),
				Either::Right(BivariateProductSumcheckProver::new([y_0.clone(), t_0.clone()], e_0)),
				Either::Right(BivariateProductSumcheckProver::new([y_1.clone(), t_1.clone()], e_1)),
			];
			let mut old_transcript = ProverTranscript::new(StdChallenger::default());
			let old_output = batch_prove(old_provers, &mut old_transcript);

			// New path: one store with six borrowed columns, folded once each.
			let mut store = MleStore::new(m - 1);
			let [y_0_col, y_1_col, d_0_col, d_1_col, t_0_col, t_1_col] =
				[&y_0, &y_1, &d_0, &d_1, &t_0, &t_1].map(|col| store.push(col.to_ref()));
			let (num_evaluator, den_evaluator) = frac_add_mle::evaluators(
				&mut store,
				[y_0_col, y_1_col, d_0_col, d_1_col],
				z.to_vec(),
				frac_claims,
			);
			// The fractional evaluators share z's eq tracker; recover its id for the sumcheck
			// wrappers, which read the round's alpha and equality prefix from the store.
			let eq_tracker = store.register_eq_tracker(&z);
			let num_evaluator = MleToSumCheckEvaluator::new(num_evaluator, eq_tracker);
			let den_evaluator = MleToSumCheckEvaluator::new(den_evaluator, eq_tracker);
			let product_0 = BivariateProductEvaluator::new([y_0_col, t_0_col], e_0);
			let product_1 = BivariateProductEvaluator::new([y_1_col, t_1_col], e_1);
			let evaluators: Vec<Box<dyn RoundEvaluator<F, P>>> = vec![
				Box::new(num_evaluator),
				Box::new(den_evaluator),
				Box::new(product_0),
				Box::new(product_1),
			];
			let shared = SharedSumcheckProver::new(store, evaluators);
			let mut new_transcript = ProverTranscript::new(StdChallenger::default());
			let new_output = batch_prove(vec![shared], &mut new_transcript);

			assert_eq!(old_output.challenges, new_output.challenges);

			// The shared prover emits each store column's evaluation once, in push order
			// [Y_0, Y_1, D_0, D_1, T_0, T_1] — no per-claim duplication of the shared columns.
			let new_evals = &new_output.multilinear_evals[0];
			assert_eq!(new_evals.len(), 6);
			// Y_0, Y_1, D_0, D_1 match the fractional provers' shared columns.
			assert_eq!(new_evals[..4], old_output.multilinear_evals[0][..]);
			assert_eq!(old_output.multilinear_evals[1], old_output.multilinear_evals[0]);
			// The product provers evaluate [Y_half, T_half]; T_0, T_1 are their second evaluations.
			assert_eq!(new_evals[0], old_output.multilinear_evals[2][0]);
			assert_eq!(new_evals[4], old_output.multilinear_evals[2][1]);
			assert_eq!(new_evals[1], old_output.multilinear_evals[3][0]);
			assert_eq!(new_evals[5], old_output.multilinear_evals[3][1]);

			assert_eq!(
				old_transcript.finalize(),
				new_transcript.finalize(),
				"transcripts must be byte-identical (m={m})"
			);
		}
	}
}
