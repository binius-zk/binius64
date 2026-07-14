// Copyright 2026 The Binius Developers

//! Shared multilinear column store for sumcheck round evaluators.
//!
//! An [`MleStore`] owns the equal-length multilinear columns that a group of
//! [`RoundEvaluator`](super::round_evaluator::RoundEvaluator)s reads, along with the deduplicated
//! [`Gruen32`] equality-indicator trackers for MLE-check evaluation points. Columns enter the
//! store either borrowed ([`MleStore::push`]) or owned ([`MleStore::push_owned`]) and are
//! addressed by the returned [`ColId`], so several evaluators can read — and the store can fold —
//! one shared column exactly once per challenge.
//!
//! # Invariant
//!
//! The store folds — columns and eq trackers both; evaluators only read. Every column and every
//! registered tracker advances exactly once per [`MleStore::fold`] call, no matter how many
//! evaluators reference it.
//!
//! Folding of the columns is *deferred*: [`MleStore::fold`] records the challenge — advancing the
//! logical variable count and the (small) eq trackers — but leaves the column data untouched, so
//! it is O(1) in the columns. The next round's pass then folds the columns while it reads them:
//! [`MleStore::fused_execute`] applies the pending fold to each chunk in place and feeds the folded
//! halves straight to the evaluators, fusing fold and read into one memory pass (~1.5x the column
//! length of memory traffic instead of a plain read plus a separate fold's ~2.5x). The eager
//! [`MleStore::apply_pending_fold`] serves the first round, the last fold before
//! [`MleStore::final_evals`], and rounds too small to chunk. Fusing only reorders the associative
//! field additions, so all paths produce identical round polynomials and evaluations.

use binius_field::{Field, PackedField};
use binius_math::{
	FieldBuffer, FieldSlice, line::extrapolate_line_packed,
	multilinear::fold::fold_highest_var_inplace,
};
use binius_utils::rayon::prelude::*;
use itertools::izip;

use super::gruen32::Gruen32;

/// One logical column's four packed segments at a single chunk, for the fused deferred fold.
///
/// The pre-fold column at this chunk is `[q0 | q1 | q2 | q3]` (the four quarters at the chunk
/// offset). Folding on the highest variable pairs the front half `[q0 | q1]` with the back half
/// `[q2 | q3]`: the folded low half is `extrapolate(q0, q2, r)` written back into `q0`, and the
/// folded high half is `extrapolate(q1, q3, r)` written back into `q1`. After the fold `q0`/`q1`
/// hold the folded column's low/high halves, fed straight to the evaluators while cache-hot.
struct FoldCol<'c, P: PackedField> {
	q0: &'c mut [P],
	q1: &'c mut [P],
	q2: &'c [P],
	q3: &'c [P],
}

/// One chunk's fold-and-accumulate work: every logical column's four segments and every eq
/// expansion's slice at this chunk. Built by transposing the columns' split segments so each is a
/// distinct rayon task owning disjoint mutable sub-slices.
struct ChunkTask<'c, P: PackedField> {
	cols: Vec<FoldCol<'c, P>>,
	eqs: Vec<FieldSlice<'c, P>>,
}

/// Identifier of a column held by an [`MleStore`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ColId(usize);

impl ColId {
	/// Returns the position of the column in the store, which indexes the
	/// [`MleStore::final_evals`] output.
	pub const fn index(self) -> usize {
		self.0
	}
}

/// Identifier of an equality-indicator tracker held by an [`MleStore`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EqId(usize);

impl EqId {
	/// Returns the registration position of the tracker in the store.
	pub const fn index(self) -> usize {
		self.0
	}
}

/// One physical entry in the store, holding one or two logical columns.
///
/// A `Borrowed` or `Owned` entry is a single column. A `SplitHalf` entry holds two adjacent
/// columns — the low and high halves of one parent buffer — in a single allocation, so no copy is
/// made to separate them.
enum Column<'a, P: PackedField> {
	Borrowed(FieldSlice<'a, P>),
	Owned(FieldBuffer<P>),
	/// A parent buffer whose low and high halves are two adjacent columns.
	///
	/// Pushed by [`MleStore::push_split_half`]. The buffer keeps its original length for the life
	/// of the store; each [`MleStore::fold`] advances both halves in place within it, and the two
	/// columns are read as the front `2^n_vars` scalars of the low and high halves. This shares one
	/// allocation between the sibling columns with no copy at any point.
	SplitHalf(FieldBuffer<P>),
}

/// A store of equal-length multilinear columns shared by a group of round evaluators.
///
/// See the [module documentation](self) for the folding invariant.
pub struct MleStore<'a, P: PackedField> {
	/// The *physical* number of variables the columns are stored at.
	///
	/// This is the size of the buffers on disk. The logical remaining-variable count seen by
	/// callers ([`Self::n_vars`]) is this minus one when a [`Self::fold`] is pending, since a
	/// pending fold has already advanced the claim state but not yet touched the column data.
	physical_n_vars: usize,
	columns: Vec<Column<'a, P>>,
	/// Number of logical columns, counting each [`Column::SplitHalf`] entry as two. This is the
	/// number of assigned [`ColId`]s and the length of the [`Self::final_evals`] output.
	n_cols: usize,
	eq_trackers: Vec<Gruen32<P>>,
	/// The challenge of a deferred fold not yet applied to the columns.
	///
	/// [`Self::fold`] records the challenge here (advancing the logical variable count and the eq
	/// trackers) without touching the column data; the next [`Self::execute_context`] pass — or
	/// [`Self::apply_pending_fold`] — folds the columns while it reads them. At most one fold can
	/// be pending, since every fold is followed by a round pass that consumes it.
	pending_fold: Option<P::Scalar>,
}

impl<'a, F: Field, P: PackedField<Scalar = F>> MleStore<'a, P> {
	/// Creates an empty store over columns with `n_vars` variables.
	pub const fn new(n_vars: usize) -> Self {
		Self {
			physical_n_vars: n_vars,
			columns: Vec::new(),
			n_cols: 0,
			eq_trackers: Vec::new(),
			pending_fold: None,
		}
	}

	/// Returns the number of variables remaining in the columns.
	///
	/// Decrements with each [`Self::fold`] call. A pending (deferred) fold counts as already
	/// applied: it has advanced the claim state, so the logical count drops even before the column
	/// data is folded.
	pub const fn n_vars(&self) -> usize {
		self.physical_n_vars - self.pending_fold.is_some() as usize
	}

	/// Pushes a borrowed column and returns its identifier.
	///
	/// The column is not copied; the first [`Self::fold`] writes into a fresh half-size buffer.
	pub fn push(&mut self, column: FieldSlice<'a, P>) -> ColId {
		// precondition
		assert_eq!(
			column.log_len(),
			self.n_vars(),
			"column must have number of variables equal to the store"
		);
		self.columns.push(Column::Borrowed(column));
		self.next_col_id()
	}

	/// Pushes an owned column and returns its identifier.
	pub fn push_owned(&mut self, column: FieldBuffer<P>) -> ColId {
		// precondition
		assert_eq!(
			column.log_len(),
			self.n_vars(),
			"column must have number of variables equal to the store"
		);
		self.columns.push(Column::Owned(column));
		self.next_col_id()
	}

	/// Allocates the identifier for one newly pushed logical column.
	const fn next_col_id(&mut self) -> ColId {
		let id = ColId(self.n_cols);
		self.n_cols += 1;
		id
	}

	/// Pushes the low and high halves of `buffer` as two columns, returning their ids `[low,
	/// high]`.
	///
	/// The halves are not copied: the store takes ownership of `buffer` and holds both columns in
	/// it as a single split-half entry, so no up-front copy of the full buffer is made.
	/// Each [`Self::fold`] advances both halves in place within the buffer. `buffer` splits on its
	/// highest variable, so its low half fixes that variable to 0 and its high half to 1 —
	/// matching the store's high-to-low fold order.
	pub fn push_split_half(&mut self, buffer: FieldBuffer<P>) -> [ColId; 2] {
		// precondition
		assert_eq!(
			buffer.log_len(),
			self.n_vars() + 1,
			"buffer must have one more variable than the store so each half matches it"
		);
		self.columns.push(Column::SplitHalf(buffer));
		let low = ColId(self.n_cols);
		let high = ColId(self.n_cols + 1);
		self.n_cols += 2;
		[low, high]
	}

	/// Registers an equality-indicator tracker for an MLE-check evaluation point.
	///
	/// Trackers are deduplicated: evaluators registering the same evaluation point share one
	/// tracker, which the store folds once per challenge.
	pub fn register_eq_tracker(&mut self, eval_point: &[F]) -> EqId {
		// precondition
		assert_eq!(
			eval_point.len(),
			self.n_vars(),
			"evaluation point length must equal the store's number of variables"
		);
		// Trackers fold in lockstep with the store, so the remaining coordinates of an existing
		// tracker are the prefix of its original evaluation point.
		let existing = self
			.eq_trackers
			.iter()
			.position(|tracker| &tracker.eval_point()[..self.n_vars()] == eval_point);
		let index = existing.unwrap_or_else(|| {
			self.eq_trackers.push(Gruen32::new(eval_point));
			self.eq_trackers.len() - 1
		});
		EqId(index)
	}

	/// Returns the equality-indicator expansion of a registered tracker.
	///
	/// The expansion has `n_vars() - 1` variables: the tracker keeps the indicator folded on the
	/// variable currently being bound.
	pub fn eq_expansion(&self, id: EqId) -> &FieldBuffer<P> {
		self.eq_trackers[id.0].eq_expansion()
	}

	/// Returns the equality-indicator expansion of every registered tracker, in [`EqId`] order.
	///
	/// The driving prover slices each expansion per chunk once per round; the returned order
	/// matches [`EqId::index`], so an evaluator's tracker id indexes the resulting per-chunk
	/// slices.
	pub fn eq_expansions(&self) -> Vec<&FieldBuffer<P>> {
		self.eq_trackers
			.iter()
			.map(|tracker| tracker.eq_expansion())
			.collect()
	}

	/// Returns the full evaluation point of a registered eq tracker.
	///
	/// The point spans all of the store's original variables — it is not truncated as the store
	/// folds, so the remaining (unbound) coordinates are the prefix `eq_point(id)[..n_vars()]`. An
	/// evaluator registers its point once (via [`Self::register_eq_tracker`]) and reads it back
	/// here from the returned [`EqId`], rather than owning a second copy. Most evaluators only need
	/// the current round's coordinate ([`Self::eq_alpha`]) and equality prefix
	/// ([`Self::eq_prefix`]) and can avoid handling the point directly.
	pub fn eq_point(&self, id: EqId) -> &[F] {
		self.eq_trackers[id.0].eval_point()
	}

	/// Returns the highest remaining coordinate of a registered eq tracker.
	///
	/// This is the coordinate of the variable bound in the current round — the round's `alpha`. The
	/// store pops one coordinate off each tracker as [`Self::fold`] advances, so this is always the
	/// coordinate for the round about to run, and an evaluator reads it here instead of tracking
	/// the point and remaining-variable count itself.
	pub fn eq_alpha(&self, id: EqId) -> F {
		self.eq_trackers[id.0].next_coordinate()
	}

	/// Returns the equality prefix of a registered eq tracker.
	///
	/// This is the product of the equality terms of all previously bound coordinates, which the
	/// [Gruen24] technique multiplies into each round polynomial. The store maintains it on the
	/// tracker across [`Self::fold`] calls, so an eq-weighted evaluator reads it here rather than
	/// accumulating its own copy.
	///
	/// [Gruen24]: <https://eprint.iacr.org/2024/108>
	pub fn eq_prefix(&self, id: EqId) -> F {
		self.eq_trackers[id.0].eq_prefix_eval()
	}

	/// Records a deferred fold of every column and folds every eq tracker with a verifier
	/// challenge.
	///
	/// The column fold is *deferred*: it is recorded (advancing the logical variable count and the
	/// eq trackers) but not applied to the column data, so this is O(1) in the columns. The next
	/// round pass folds the columns while it reads them — [`Self::execute_context`] applies the
	/// pending fold as its first act — fusing fold and read into one memory pass. The eq trackers,
	/// small relative to the columns and read by evaluators before the columns are folded, advance
	/// eagerly here.
	///
	/// Columns fold on the highest variable, matching the high-to-low binding order of the
	/// sumcheck provers this store backs.
	pub fn fold(&mut self, challenge: F) {
		// precondition
		assert!(self.n_vars() > 0, "fold requires at least one remaining variable");
		debug_assert!(
			self.pending_fold.is_none(),
			"a pending fold must be consumed by a round pass before folding again"
		);

		for tracker in &mut self.eq_trackers {
			tracker.fold(challenge);
		}
		self.pending_fold = Some(challenge);
	}

	/// Returns whether a deferred fold has been recorded but not yet applied to the columns.
	pub const fn has_pending_fold(&self) -> bool {
		self.pending_fold.is_some()
	}

	/// Applies any pending deferred fold to the column data, eagerly.
	///
	/// After this the columns are physically at [`Self::n_vars`] variables and no fold is pending.
	/// The fused round pass ([`Self::execute_context`]) applies the pending fold as it reads
	/// instead; this eager path is used by [`Self::final_evals`] after the last round and by rounds
	/// too small to fuse.
	pub fn apply_pending_fold(&mut self) {
		if let Some(challenge) = self.pending_fold.take() {
			self.fold_columns(challenge);
			self.physical_n_vars -= 1;
		}
	}

	/// Folds every column in place with `challenge`, on the highest of its `physical_n_vars`
	/// variables. Does not update `physical_n_vars` or `pending_fold`.
	fn fold_columns(&mut self, challenge: F) {
		// The number of live variables in each column before this fold; a split-half buffer keeps
		// its full length, so its halves must be truncated to this before folding.
		let physical_n_vars = self.physical_n_vars;
		for column in &mut self.columns {
			match column {
				Column::Owned(buffer) => fold_highest_var_inplace(buffer, challenge),
				Column::Borrowed(slice) => {
					// The first fold of a borrowed column writes into a fresh half-size owned
					// buffer, avoiding an up-front copy of the full column.
					*column = Column::Owned(fold_highest_var(slice, challenge));
				}
				Column::SplitHalf(buffer) => {
					// Fold each half on its own highest variable in place. The two halves are the
					// two columns, so folding the whole buffer's highest variable would instead
					// combine them; splitting first binds each column's variable independently. The
					// buffer keeps its length — the folded columns are the (now shorter) fronts of
					// its halves — so no copy is made.
					let mut split = buffer.split_half_mut();
					let (mut low, mut high) = split.halves();
					low.truncate(physical_n_vars);
					high.truncate(physical_n_vars);
					fold_highest_var_inplace(&mut low, challenge);
					fold_highest_var_inplace(&mut high, challenge);
				}
			}
		}
	}

	/// Expands the store into one borrowed slice per logical column, in [`ColId`] order.
	///
	/// A split-half entry expands into the front `2^n_vars` scalars of its low and high
	/// halves, so the returned length is the logical column count — larger than the physical entry
	/// count whenever a split-half column is present.
	///
	/// Private: with a fold pending it would misreport each column's live region (notably the
	/// split-half fronts), so it is only ever called after the pending fold is resolved.
	fn column_slices(&self) -> Vec<FieldSlice<'_, P>> {
		debug_assert!(
			self.pending_fold.is_none(),
			"column data must not be read with a fold pending; apply it first"
		);
		let mut slices = Vec::with_capacity(self.n_cols);
		for column in &self.columns {
			match column {
				Column::Borrowed(slice) => slices.push(slice.to_ref()),
				Column::Owned(buffer) => slices.push(buffer.to_ref()),
				Column::SplitHalf(buffer) => {
					// The buffer holds the two columns as its low and high halves; each column is
					// the front `2^n_vars` scalars of one half, so read it as that half's
					// chunk 0.
					let high_start = 1 << (buffer.log_len() - 1 - self.physical_n_vars);
					slices.push(buffer.chunk(self.physical_n_vars, 0));
					slices.push(buffer.chunk(self.physical_n_vars, high_start));
				}
			}
		}
		slices
	}

	/// Returns the evaluation of every column at the challenge point, indexed by [`ColId`].
	///
	/// Each column's evaluation is computed once, no matter how many claims read the column.
	pub fn final_evals(&self) -> Vec<F> {
		// precondition
		assert_eq!(self.n_vars(), 0, "final_evals requires all variables to be folded");
		assert!(
			self.pending_fold.is_none(),
			"final_evals requires the last fold to be applied; call apply_pending_fold first"
		);

		self.column_slices()
			.iter()
			.map(|slice| slice.get(0))
			.collect()
	}

	/// Prepares one round's accumulation pass over the columns and eq trackers.
	///
	/// The returned [`ExecuteContext`] borrows the store's expanded column slices and eq-indicator
	/// expansions and hands each parallel chunk to the round evaluators (see
	/// [`ExecuteContext::par_chunks`]).
	pub fn execute_context(&self) -> ExecuteContext<'_, P> {
		debug_assert!(
			self.pending_fold.is_none(),
			"apply the pending fold before building an execute context"
		);
		ExecuteContext {
			n_vars: self.physical_n_vars,
			cols: self.column_slices(),
			eqs: self.eq_expansions(),
		}
	}

	/// Returns whether any column is still borrowed (never yet folded).
	///
	/// A borrowed column's first fold must go out of place (it cannot be written), so the fused
	/// deferred pass ([`Self::fused_execute`]) does not accept it; the caller folds eagerly
	/// instead. Borrowed columns arise only from [`Self::push`], which the production provers do
	/// not use.
	pub fn has_borrowed_column(&self) -> bool {
		self.columns
			.iter()
			.any(|column| matches!(column, Column::Borrowed(_)))
	}

	/// Runs the round's accumulation pass while applying the pending deferred fold in a single
	/// memory pass, feeding each folded chunk to `fold_op` (a rayon fold over per-worker `A`).
	///
	/// For each chunk of the post-fold halved hypercube this folds every column's four segments in
	/// place — writing the folded low/high halves back — and immediately hands those
	/// still-cache-hot halves, as an [`EvaluationChunk`], to `fold_op`. Versus the eager path
	/// ([`Self::apply_pending_fold`] then [`Self::execute_context`]) it reads each column once and
	/// writes half — ~1.5x the column length rather than ~2.5x. Results are identical: fusing only
	/// reorders the GF(2^k) additions, which are associative.
	///
	/// After the pass the columns are physically at [`Self::n_vars`] variables and no fold is
	/// pending.
	///
	/// ## Preconditions
	///
	/// * a fold is pending,
	/// * `P::LOG_WIDTH <= chunk_vars` and `chunk_vars <= n_vars() - 1`,
	/// * no column is borrowed (see [`Self::has_borrowed_column`]).
	pub fn fused_execute<A: Send>(
		&mut self,
		chunk_vars: usize,
		identity: impl Fn() -> A + Sync + Send,
		fold_op: impl Fn(A, &EvaluationChunk<'_, P>) -> A + Sync + Send,
		reduce_op: impl Fn(A, A) -> A + Sync + Send,
	) -> A {
		let challenge = self
			.pending_fold
			.expect("fused_execute requires a pending fold");
		debug_assert!(!self.has_borrowed_column(), "fused_execute needs no borrowed columns");
		let physical_n_vars = self.physical_n_vars;
		let log_width = P::LOG_WIDTH;
		// The folded column has `physical_n_vars - 1` variables and its halved hypercube (over
		// which the round accumulates) has `physical_n_vars - 2`, chunked at `chunk_vars`.
		debug_assert!(log_width <= chunk_vars && chunk_vars + 1 < physical_n_vars);

		let challenge_bcast = P::broadcast(challenge);
		// Packed lengths. A pre-fold column is `2^physical_n_vars` scalars = four quarters of
		// `m_packed` packed elements; each chunk covers `chunk_packed` packed elements.
		let m_packed = 1usize << (physical_n_vars - 2 - log_width);
		let chunk_packed = 1usize << (chunk_vars - log_width);
		let chunk_count = m_packed / chunk_packed;
		let region_packed = 4 * m_packed;

		// Borrow the eq expansions (read-only) from a field disjoint from `columns`.
		let eq_bufs: Vec<&FieldBuffer<P>> = self
			.eq_trackers
			.iter()
			.map(|tracker| tracker.eq_expansion())
			.collect();

		// Transpose the per-column split segments into one task per chunk, so rayon owns disjoint
		// mutable sub-slices. This is O(chunk_count * n_cols), negligible against the pass itself.
		let mut tasks: Vec<ChunkTask<'_, P>> = (0..chunk_count)
			.map(|_| ChunkTask {
				cols: Vec::with_capacity(self.n_cols),
				eqs: Vec::with_capacity(eq_bufs.len()),
			})
			.collect();
		for column in &mut self.columns {
			match column {
				Column::Owned(buffer) => {
					distribute_region(buffer.as_mut(), m_packed, chunk_packed, &mut tasks)
				}
				Column::SplitHalf(buffer) => {
					// The buffer's low and high halves are the two logical columns; each folds
					// independently as the front `region_packed` of its half.
					let full = buffer.as_mut();
					let half = full.len() / 2;
					let (low, high) = full.split_at_mut(half);
					distribute_region(
						&mut low[..region_packed],
						m_packed,
						chunk_packed,
						&mut tasks,
					);
					distribute_region(
						&mut high[..region_packed],
						m_packed,
						chunk_packed,
						&mut tasks,
					);
				}
				Column::Borrowed(_) => unreachable!("fused_execute needs no borrowed columns"),
			}
		}
		for eq in &eq_bufs {
			for (task, chunk) in tasks.iter_mut().zip(eq.chunks(chunk_vars)) {
				task.eqs.push(chunk);
			}
		}

		let result = tasks
			.into_par_iter()
			.fold(&identity, |acc, mut task| {
				// Fold each column's chunk in place: q0/q1 become the folded low/high halves.
				for fc in &mut task.cols {
					for (a0, a1, &b2, &b3) in
						izip!(fc.q0.iter_mut(), fc.q1.iter_mut(), fc.q2.iter(), fc.q3.iter())
					{
						*a0 = extrapolate_line_packed(*a0, b2, challenge_bcast);
						*a1 = extrapolate_line_packed(*a1, b3, challenge_bcast);
					}
				}
				// Build the chunk from the just-folded, cache-hot halves and accumulate.
				let cols = task
					.cols
					.iter()
					.map(|fc| ColumnChunk {
						lo: FieldSlice::from_slice(chunk_vars, &fc.q0[..]),
						hi: FieldSlice::from_slice(chunk_vars, &fc.q1[..]),
					})
					.collect();
				let chunk = EvaluationChunk {
					cols,
					eqs: task.eqs,
				};
				fold_op(acc, &chunk)
			})
			.reduce(&identity, reduce_op);

		// The columns are now folded in their front halves. Owned buffers shrink to the folded
		// size; split-half buffers keep their length (their columns are read as the front of each
		// half).
		for column in &mut self.columns {
			if let Column::Owned(buffer) = column {
				buffer.truncate(physical_n_vars - 1);
			}
		}
		self.physical_n_vars -= 1;
		self.pending_fold = None;
		result
	}
}

/// One store column's low and high halves at a single chunk of the halved hypercube.
///
/// The column is split on the round's highest variable: `lo` fixes that variable to 0, `hi` to 1.
/// Both range over the chunk's `2^chunk_vars` scalars.
pub struct ColumnChunk<'c, P: PackedField> {
	pub lo: FieldSlice<'c, P>,
	pub hi: FieldSlice<'c, P>,
}

/// One chunk of the halved hypercube, prepared for the round evaluators.
///
/// With `n` variables remaining, each column splits on the highest variable into two halves of
/// `n - 1` variables, and both halves divide into chunks of `2^chunk_vars` scalars. This holds one
/// such chunk: the split halves of every logical column and the same chunk of every eq-indicator
/// expansion. A column read by several evaluators is chunked a single time. Evaluators read their
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

/// A round's expanded columns and eq-indicator expansions, borrowed from an [`MleStore`].
///
/// Produced by [`MleStore::execute_context`]. It expands the store's columns once — a split-half
/// column becomes its two halves — and drives the parallel round pass through
/// [`Self::par_chunks`], which slices each column and eq expansion per chunk into an
/// [`EvaluationChunk`].
pub struct ExecuteContext<'b, P: PackedField> {
	// The store's remaining variable count; the halved hypercube has `n_vars - 1` variables.
	n_vars: usize,
	// One slice per logical column, over all `n_vars` remaining variables, in `ColId` order.
	cols: Vec<FieldSlice<'b, P>>,
	// One eq-indicator expansion per registered tracker, over `n_vars - 1` variables, in `EqId`
	// order.
	eqs: Vec<&'b FieldBuffer<P>>,
}

impl<'b, P: PackedField> ExecuteContext<'b, P> {
	/// Returns a parallel iterator over the chunks of the halved hypercube.
	///
	/// Each item is one [`EvaluationChunk`]: the split low/high halves of every column and the
	/// matching chunk of every eq-indicator expansion, at `2^chunk_vars` scalars per chunk. A
	/// column's low half is the front chunk `chunk_index` of the full column; its high half is
	/// chunk `chunk_count + chunk_index`, the corresponding chunk of the back half — so the column
	/// is sliced without materializing its halves separately.
	///
	/// ## Preconditions
	///
	/// * `chunk_vars` must be at most `n_vars - 1`.
	pub fn par_chunks(
		&self,
		chunk_vars: usize,
	) -> impl IndexedParallelIterator<Item = EvaluationChunk<'_, P>> {
		// precondition
		assert!(
			chunk_vars < self.n_vars,
			"chunk_vars must be at most the halved hypercube's variable count"
		);

		let chunk_count = 1usize << (self.n_vars - 1 - chunk_vars);
		(0..chunk_count).into_par_iter().map(move |chunk_index| {
			// The full column at `chunk_vars` holds the low half in its first `chunk_count` chunks
			// and the high half in the next `chunk_count`, so the two halves of this chunk are the
			// full column's chunks `chunk_index` and `chunk_count + chunk_index`.
			let cols = self
				.cols
				.iter()
				.map(|col| ColumnChunk {
					lo: col.chunk(chunk_vars, chunk_index),
					hi: col.chunk(chunk_vars, chunk_count + chunk_index),
				})
				.collect();
			let eqs = self
				.eqs
				.iter()
				.map(|eq| eq.chunk(chunk_vars, chunk_index))
				.collect();
			EvaluationChunk { cols, eqs }
		})
	}
}

/// Splits one logical column's pre-fold `region` into its four quarters and appends one
/// [`FoldCol`] per chunk to `tasks`.
///
/// `region` is the `4 * m_packed` packed elements of a column with the fold pending. Its front half
/// `[q0 | q1]` is taken mutably (the folded low/high halves are written back there) and its back
/// half `[q2 | q3]` read-only; each quarter is sliced into `chunk_packed`-sized chunks aligned with
/// `tasks`.
fn distribute_region<'c, P: PackedField>(
	region: &'c mut [P],
	m_packed: usize,
	chunk_packed: usize,
	tasks: &mut [ChunkTask<'c, P>],
) {
	let (front, back) = region.split_at_mut(2 * m_packed);
	let (q0, q1) = front.split_at_mut(m_packed);
	let (q2, q3) = back.split_at(m_packed);
	for (task, (((c0, c1), c2), c3)) in tasks.iter_mut().zip(
		q0.chunks_mut(chunk_packed)
			.zip(q1.chunks_mut(chunk_packed))
			.zip(q2.chunks(chunk_packed))
			.zip(q3.chunks(chunk_packed)),
	) {
		task.cols.push(FoldCol {
			q0: c0,
			q1: c1,
			q2: c2,
			q3: c3,
		});
	}
}

/// Computes the partial evaluation of a multilinear on its highest variable, out of place.
///
/// This is the out-of-place counterpart of [`fold_highest_var_inplace`], used for the first fold
/// of a borrowed column.
fn fold_highest_var<P: PackedField>(
	values: &FieldSlice<P>,
	challenge: P::Scalar,
) -> FieldBuffer<P> {
	assert!(values.log_len() > 0);

	let challenge_broadcast = P::broadcast(challenge);
	let (lo, hi) = values.split_half_ref();
	let out_vals = (lo.as_ref(), hi.as_ref())
		.into_par_iter()
		.map(|(&lo_i, &hi_i)| extrapolate_line_packed(lo_i, hi_i, challenge_broadcast))
		.collect();
	FieldBuffer::new(values.log_len() - 1, out_vals)
}

#[cfg(test)]
mod tests {
	use std::ops::Deref;

	use binius_field::{FieldOps, Random};
	use binius_math::{
		multilinear::fold::fold_highest_var_inplace, test_utils::random_field_buffer,
	};
	use rand::prelude::*;

	use super::*;

	type P = binius_math::test_utils::Packed128b;
	type F = <P as FieldOps>::Scalar;

	// Flatten a field buffer or borrowed slice into its live scalars, for column comparison.
	fn scalars<Data: Deref<Target = [P]>>(buffer: &FieldBuffer<P, Data>) -> Vec<F> {
		(0..1usize << buffer.log_len())
			.map(|i| buffer.get(i))
			.collect()
	}

	// The fused deferred fold must produce column data identical to an independent eager reference,
	// across all three column variants and both the fused and eager-fallback paths.
	//
	// The reference folds owned copies of every logical column with the `binius-math` fold
	// primitive; the store folds lazily, resolving each pending fold through the fused pass when it
	// is legal (no borrowed column, packed-aligned chunks) and through the eager path otherwise.
	// `Packed128b` has `LOG_WIDTH == 2`, so with `n_vars = 10` the rounds walk the whole matrix: an
	// eager first round (the borrowed column's out-of-place first fold), then fused rounds — sized
	// to split into multiple parallel chunks while large enough — then an eager tail, ending with
	// the last fold applied with no trailing pass (the `finish` edge) before `final_evals`.
	#[test]
	fn test_fused_fold_matches_eager_reference() {
		let n_vars = 10;
		let mut rng = StdRng::seed_from_u64(0);

		// One column of each variant. A split-half buffer carries two logical columns as its low
		// and high halves, so it holds one extra variable.
		let borrowed_src = random_field_buffer::<P>(&mut rng, n_vars);
		let owned_src = random_field_buffer::<P>(&mut rng, n_vars);
		let split_src = random_field_buffer::<P>(&mut rng, n_vars + 1);

		// Independent reference: owned copies in `ColId` order (borrowed, owned, split low, split
		// high), each folded eagerly with the math primitive.
		let (split_lo, split_hi) = split_src.split_half_ref();
		let mut refs: Vec<FieldBuffer<P>> = vec![
			borrowed_src.clone(),
			owned_src.clone(),
			FieldBuffer::new(split_lo.log_len(), split_lo.as_ref().into()),
			FieldBuffer::new(split_hi.log_len(), split_hi.as_ref().into()),
		];

		let mut store = MleStore::<P>::new(n_vars);
		store.push(borrowed_src.to_ref());
		store.push_owned(owned_src);
		store.push_split_half(split_src);

		for round in 0..n_vars {
			let challenge = F::random(&mut rng);
			store.fold(challenge);
			for reference in &mut refs {
				fold_highest_var_inplace(reference, challenge);
			}

			// A fold is now pending, so the physical column size is one above the logical count.
			// Fuse when no borrowed column remains and the halved hypercube is packed-aligned;
			// otherwise fall back to the eager apply. The last round (physical 1) always falls
			// back — the `finish` edge: the last fold applied with no trailing pass.
			let physical_n_vars = store.n_vars() + 1;
			if !store.has_borrowed_column() && physical_n_vars >= P::LOG_WIDTH + 2 {
				// `chunk_vars` in `[LOG_WIDTH, physical - 2]`, biased small to split into several
				// parallel chunks whenever the round is large enough to.
				let chunk_vars = physical_n_vars
					.saturating_sub(3)
					.clamp(P::LOG_WIDTH, physical_n_vars - 2);
				store.fused_execute(chunk_vars, || (), |acc, _chunk| acc, |acc, ()| acc);
			} else {
				store.apply_pending_fold();
			}

			let store_cols = store.column_slices();
			assert_eq!(store_cols.len(), refs.len());
			for (id, (store_col, reference)) in store_cols.iter().zip(&refs).enumerate() {
				assert_eq!(
					scalars(store_col),
					scalars(reference),
					"column {id} data must match the eager reference after round {round}"
				);
			}
		}

		assert_eq!(
			store.final_evals(),
			refs.iter()
				.map(|reference| reference.get(0))
				.collect::<Vec<_>>(),
			"final evaluations must match the eager reference"
		);
	}
}
