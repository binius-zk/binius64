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
//! Folding is eager: [`MleStore::fold`] advances every column immediately, and the round pass
//! over the columns is a plain read. A deferred-fold variant that fuses the fold into the next
//! round's read pass can replace the internals without changing this interface.

use binius_field::{Field, PackedField};
use binius_math::{
	FieldBuffer, FieldSlice,
	multilinear::fold::{fold_highest_var, fold_highest_var_inplace},
};
use binius_utils::rayon::prelude::*;

use super::gruen32::Gruen32;

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
	n_vars: usize,
	columns: Vec<Column<'a, P>>,
	/// Number of logical columns, counting each [`Column::SplitHalf`] entry as two. This is the
	/// number of assigned [`ColId`]s and the length of the [`Self::final_evals`] output.
	n_cols: usize,
	eq_trackers: Vec<Gruen32<P>>,
}

impl<'a, F: Field, P: PackedField<Scalar = F>> MleStore<'a, P> {
	/// Creates an empty store over columns with `n_vars` variables.
	pub const fn new(n_vars: usize) -> Self {
		Self {
			n_vars,
			columns: Vec::new(),
			n_cols: 0,
			eq_trackers: Vec::new(),
		}
	}

	/// Returns the number of variables remaining in the columns.
	///
	/// Decrements with each [`Self::fold`] call.
	pub const fn n_vars(&self) -> usize {
		self.n_vars
	}

	/// Pushes a borrowed column and returns its identifier.
	///
	/// The column is not copied; the first [`Self::fold`] writes into a fresh half-size buffer.
	pub fn push(&mut self, column: FieldSlice<'a, P>) -> ColId {
		// precondition
		assert_eq!(
			column.log_len(),
			self.n_vars,
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
			self.n_vars,
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
			self.n_vars + 1,
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
			self.n_vars,
			"evaluation point length must equal the store's number of variables"
		);
		// Trackers fold in lockstep with the store, so the remaining coordinates of an existing
		// tracker are the prefix of its original evaluation point.
		let existing = self
			.eq_trackers
			.iter()
			.position(|tracker| &tracker.eval_point()[..self.n_vars] == eval_point);
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

	/// Folds every column and every eq tracker with a verifier challenge.
	///
	/// Columns fold on the highest variable, matching the high-to-low binding order of the
	/// sumcheck provers this store backs.
	pub fn fold(&mut self, challenge: F) {
		// precondition
		assert!(self.n_vars > 0, "fold requires at least one remaining variable");

		// The number of live variables in each column before this fold; a split-half buffer keeps
		// its full length, so its halves must be truncated to this before folding.
		let n_vars = self.n_vars;
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
					low.truncate(n_vars);
					high.truncate(n_vars);
					fold_highest_var_inplace(&mut low, challenge);
					fold_highest_var_inplace(&mut high, challenge);
				}
			}
		}
		for tracker in &mut self.eq_trackers {
			tracker.fold(challenge);
		}
		self.n_vars -= 1;
	}

	/// Expands the store into one borrowed slice per logical column, in [`ColId`] order.
	///
	/// A split-half entry expands into the front `2^n_vars` scalars of its low and high
	/// halves, so the returned length is the logical column count — larger than the physical entry
	/// count whenever a split-half column is present.
	pub fn column_slices(&self) -> Vec<FieldSlice<'_, P>> {
		let mut slices = Vec::with_capacity(self.n_cols);
		for column in &self.columns {
			match column {
				Column::Borrowed(slice) => slices.push(slice.to_ref()),
				Column::Owned(buffer) => slices.push(buffer.to_ref()),
				Column::SplitHalf(buffer) => {
					// The buffer holds the two columns as its low and high halves; each column is
					// the front `2^n_vars` scalars of one half, so read it as that half's
					// chunk 0.
					let high_start = 1 << (buffer.log_len() - 1 - self.n_vars);
					slices.push(buffer.chunk(self.n_vars, 0));
					slices.push(buffer.chunk(self.n_vars, high_start));
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
		assert_eq!(self.n_vars, 0, "final_evals requires all variables to be folded");

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
		ExecuteContext {
			n_vars: self.n_vars,
			cols: self.column_slices(),
			eqs: self.eq_expansions(),
		}
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
