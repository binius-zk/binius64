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
use binius_utils::rayon;

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

	/// Maps every chunk of the halved hypercube through `map` and combines the results with
	/// `reduce`, driven by a recursive [`rayon::join`] tree.
	///
	/// The store's columns are expanded once — a split-half column becomes its two halves — and
	/// each column is split on the round's highest variable into its low and high halves. The
	/// recursion peels the remaining variables off both halves, and off every eq-indicator
	/// expansion, together, so each leaf hands `map` one [`EvaluationChunk`]: the paired column
	/// halves and the matching eq chunk, at `2^chunk_vars` scalars per half.
	///
	/// `chunk_vars` is capped at `n_vars() - 1`, so leaves never exceed the halved hypercube.
	///
	/// ## Preconditions
	///
	/// * `n_vars()` must be greater than 0.
	pub fn map_reduce<T: Send>(
		&self,
		chunk_vars: usize,
		map: impl (for<'c> Fn(EvaluationChunk<'c, P>) -> T) + Sync,
		reduce: impl (Fn(T, T) -> T) + Sync,
	) -> T {
		assert!(self.n_vars > 0);
		let chunk_vars = chunk_vars.min(self.n_vars - 1);

		let col_slices = self.column_slices();
		let cols = col_slices
			.iter()
			.map(|col| {
				let (lo, hi) = col.split_half_ref();
				ColumnChunk { lo, hi }
			})
			.collect();
		let eqs = self.eq_expansions().iter().map(|eq| eq.to_ref()).collect();
		let chunk = EvaluationChunk {
			n_vars: self.n_vars - 1,
			cols,
			eqs,
		};
		map_reduce_helper(chunk, chunk_vars, &map, &reduce)
	}
}

/// One column's low and high halves within an [`EvaluationChunk`].
///
/// The column is split on the round's highest variable: `lo` fixes that variable to 0, `hi` to 1.
/// Both range over the chunk's scalars.
pub struct ColumnChunk<'c, P: PackedField> {
	pub lo: FieldSlice<'c, P>,
	pub hi: FieldSlice<'c, P>,
}

/// A range of the halved hypercube, prepared for the round evaluators.
///
/// Holds, over `n_vars` variables, the paired low/high halves of every logical column and the
/// eq-indicator expansion of every tracker. Each column was split on the round's highest variable,
/// so a [`ColumnChunk`]'s `lo` and `hi` differ only in that variable. The range is bisected — the
/// highest remaining variable peeled off both halves of every column and off every eq expansion —
/// down to the leaves that [`MleStore::map_reduce`] hands to its `map` callback. A
/// column read by several evaluators is split a single time. Evaluators read their columns by
/// [`ColId`] and their eq trackers by [`EqId`].
pub struct EvaluationChunk<'c, P: PackedField> {
	n_vars: usize,
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

	/// Bisects the range into its two halves on the highest remaining variable, splitting both
	/// halves of every column and every eq expansion. Each returned chunk has one fewer variable.
	fn split_half(&self) -> [EvaluationChunk<'_, P>; 2] {
		let Self { n_vars, cols, eqs } = self;
		let (cols_0, cols_1) = cols
			.iter()
			.map(|ColumnChunk { lo, hi }| {
				let (lo_0, lo_1) = lo.split_half_ref();
				let (hi_0, hi_1) = hi.split_half_ref();
				(ColumnChunk { lo: lo_0, hi: hi_0 }, ColumnChunk { lo: lo_1, hi: hi_1 })
			})
			.unzip();
		let (eqs_0, eqs_1) = eqs.iter().map(|col| col.split_half_ref()).unzip();
		[
			EvaluationChunk {
				n_vars: n_vars - 1,
				cols: cols_0,
				eqs: eqs_0,
			},
			EvaluationChunk {
				n_vars: n_vars - 1,
				cols: cols_1,
				eqs: eqs_1,
			},
		]
	}
}

/// Recursively maps and reduces an [`EvaluationChunk`] for [`MleStore::map_reduce`].
///
/// Once the chunk has been narrowed to `sub_vars` variables it is handed to `map`; otherwise it is
/// bisected with [`EvaluationChunk::split_half`] and the two halves are mapped in parallel and
/// combined with `reduce`.
fn map_reduce_helper<P: PackedField, T: Send>(
	chunk: EvaluationChunk<'_, P>,
	sub_vars: usize,
	map: &(impl (for<'a> Fn(EvaluationChunk<'a, P>) -> T) + Sync),
	reduce: &(impl (Fn(T, T) -> T) + Sync),
) -> T {
	if sub_vars == chunk.n_vars {
		return map(chunk);
	}

	let [chunk_0, chunk_1] = chunk.split_half();
	let (ret_0, ret_1) = rayon::join(
		move || map_reduce_helper(chunk_0, sub_vars, map, reduce),
		move || map_reduce_helper(chunk_1, sub_vars, map, reduce),
	);
	reduce(ret_0, ret_1)
}

#[cfg(test)]
mod tests {
	use binius_field::{Field, FieldOps, PackedField};
	use binius_math::test_utils::{Packed128b, random_field_buffer, random_scalars};
	use rand::{SeedableRng, rngs::StdRng};

	use super::*;

	// A per-chunk aggregate that is sensitive to both the low/high pairing within a column and the
	// alignment of each eq expansion with its column, so a wrong recursion pairing changes the sum.
	fn chunk_aggregate<P: PackedField>(
		chunk: &EvaluationChunk<'_, P>,
		col_ids: &[ColId],
		eq_ids: &[EqId],
	) -> P::Scalar {
		let mut acc = P::Scalar::ZERO;
		for (i, &col_id) in col_ids.iter().enumerate() {
			let col = chunk.col(col_id);
			let eq = chunk.eq(eq_ids[i % eq_ids.len()]);
			for j in 0..col.lo.len() {
				acc += eq.get(j) * col.lo.get(j) * col.hi.get(j);
			}
		}
		acc
	}

	#[test]
	fn map_reduce_pairs_on_highest_variable() {
		type P = Packed128b;
		type F = <P as FieldOps>::Scalar;

		let n_vars = 7;
		let mut rng = StdRng::seed_from_u64(0);

		// A mix of column kinds so `chunk` exercises borrowed, owned, and split-half entries.
		let borrowed = [
			random_field_buffer::<P>(&mut rng, n_vars),
			random_field_buffer::<P>(&mut rng, n_vars),
		];
		let mut store = MleStore::<P>::new(n_vars);
		let mut col_ids = borrowed
			.iter()
			.map(|col| store.push(col.to_ref()))
			.collect::<Vec<_>>();
		col_ids.push(store.push_owned(random_field_buffer::<P>(&mut rng, n_vars)));
		col_ids.extend(store.push_split_half(random_field_buffer::<P>(&mut rng, n_vars + 1)));

		let eq_ids = (0..2)
			.map(|_| store.register_eq_tracker(&random_scalars::<F>(&mut rng, n_vars)))
			.collect::<Vec<_>>();

		// Independent reference: the aggregate over the whole halved hypercube, pairing each
		// logical column's front half (highest variable = 0) with its back half (= 1) at the same
		// index. This is the pairing `map_reduce` must reproduce, whatever the chunking.
		let cols = store.column_slices();
		let eqs = store.eq_expansions();
		let half = 1usize << (n_vars - 1);
		let mut expected = F::ZERO;
		for (i, col) in cols.iter().enumerate() {
			let eq = eqs[i % eqs.len()];
			for j in 0..half {
				expected += eq.get(j) * col.get(j) * col.get(half + j);
			}
		}

		for chunk_vars in 0..n_vars {
			let got = store.map_reduce(
				chunk_vars,
				|chunk| chunk_aggregate(&chunk, &col_ids, &eq_ids),
				|lhs, rhs| lhs + rhs,
			);
			assert_eq!(got, expected, "mismatch at chunk_vars = {chunk_vars}");
		}
	}
}
