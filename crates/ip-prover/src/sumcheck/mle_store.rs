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

use std::sync::Arc;

use binius_field::{Field, PackedField};
use binius_math::{FieldBuffer, FieldSlice, multilinear::fold::fold_highest_var_inplace};
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

/// State of one multilinear column.
///
/// A column starts out `Borrowed` when pushed by reference, or `SplitHalf` when pushed as one half
/// of a shared parent buffer. In both cases the first fold writes the folded values into a fresh
/// half-size owned buffer, so a column is never copied at full size up front.
enum Column<'a, P: PackedField> {
	Borrowed(FieldSlice<'a, P>),
	Owned(FieldBuffer<P>),
	/// One half of a parent buffer shared with its sibling column, selected by `high`.
	///
	/// Pushed by [`MleStore::push_split_half`], which splits a buffer into its low and high halves
	/// without copying: both halves share the parent via the [`Arc`], and the parent is freed once
	/// both siblings have folded (each fold replaces its `SplitHalf` with an [`Owned`](Self::Owned)
	/// half-size buffer, dropping that side's reference).
	SplitHalf {
		parent: Arc<FieldBuffer<P>>,
		high: bool,
	},
}

impl<P: PackedField> Column<'_, P> {
	fn as_slice(&self) -> FieldSlice<'_, P> {
		match self {
			Column::Borrowed(slice) => slice.to_ref(),
			Column::Owned(buffer) => buffer.to_ref(),
			Column::SplitHalf { parent, high } => {
				let (low, hi) = parent.split_half_ref();
				if *high { hi } else { low }
			}
		}
	}
}

/// A store of equal-length multilinear columns shared by a group of round evaluators.
///
/// See the [module documentation](self) for the folding invariant.
pub struct MleStore<'a, P: PackedField> {
	n_vars: usize,
	columns: Vec<Column<'a, P>>,
	eq_trackers: Vec<Gruen32<P>>,
}

impl<'a, F: Field, P: PackedField<Scalar = F>> MleStore<'a, P> {
	/// Creates an empty store over columns with `n_vars` variables.
	pub const fn new(n_vars: usize) -> Self {
		Self {
			n_vars,
			columns: Vec::new(),
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
		ColId(self.columns.len() - 1)
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
		ColId(self.columns.len() - 1)
	}

	/// Pushes the low and high halves of `buffer` as two columns, returning their ids `[low,
	/// high]`.
	///
	/// The halves are not copied: the store takes ownership of `buffer` and both columns share it,
	/// so no up-front copy of the full buffer is made. The first [`Self::fold`] of each half writes
	/// into a fresh quarter-size buffer, and the shared parent is freed once both halves have
	/// folded. `buffer` splits on its highest variable, so its low half fixes that variable to 0
	/// and its high half to 1 — matching the store's high-to-low fold order.
	pub fn push_split_half(&mut self, buffer: FieldBuffer<P>) -> [ColId; 2] {
		// precondition
		assert_eq!(
			buffer.log_len(),
			self.n_vars + 1,
			"buffer must have one more variable than the store so each half matches it"
		);
		let parent = Arc::new(buffer);
		let low = {
			self.columns.push(Column::SplitHalf {
				parent: Arc::clone(&parent),
				high: false,
			});
			ColId(self.columns.len() - 1)
		};
		let high = {
			self.columns.push(Column::SplitHalf { parent, high: true });
			ColId(self.columns.len() - 1)
		};
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

	/// Returns a borrowed view of a column.
	pub fn col(&self, id: ColId) -> FieldSlice<'_, P> {
		self.columns[id.0].as_slice()
	}

	/// Returns the equality-indicator expansion of a registered tracker.
	///
	/// The expansion has `n_vars() - 1` variables: the tracker keeps the indicator folded on the
	/// variable currently being bound.
	pub fn eq_expansion(&self, id: EqId) -> &FieldBuffer<P> {
		self.eq_trackers[id.0].eq_expansion()
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

		for column in &mut self.columns {
			match column {
				Column::Owned(buffer) => fold_highest_var_inplace(buffer, challenge),
				Column::Borrowed(slice) => {
					// The first fold of a borrowed column writes into a fresh half-size owned
					// buffer, avoiding an up-front copy of the full column.
					*column = Column::Owned(fold_highest_var(slice, challenge));
				}
				Column::SplitHalf { parent, high } => {
					// The first fold of a shared half writes into a fresh owned buffer, then drops
					// this side's reference to the parent; no full-size copy is ever made.
					let (low, hi) = parent.split_half_ref();
					let half = if *high { hi } else { low };
					*column = Column::Owned(fold_highest_var(&half, challenge));
				}
			}
		}
		for tracker in &mut self.eq_trackers {
			tracker.fold(challenge);
		}
		self.n_vars -= 1;
	}

	/// Returns the evaluation of every column at the challenge point, indexed by [`ColId`].
	///
	/// Each column's evaluation is computed once, no matter how many claims read the column.
	pub fn final_evals(&self) -> Vec<F> {
		// precondition
		assert_eq!(self.n_vars, 0, "final_evals requires all variables to be folded");

		self.columns
			.iter()
			.map(|column| column.as_slice().get(0))
			.collect()
	}
}

/// Computes the partial evaluation of a multilinear on its highest variable, out of place.
///
/// This is the out-of-place counterpart of [`fold_highest_var_inplace`], used for the first fold
/// of a borrowed column.
fn fold_highest_var<P: PackedField>(values: &FieldSlice<P>, scalar: P::Scalar) -> FieldBuffer<P> {
	let broadcast_scalar = P::broadcast(scalar);
	let (lo, hi) = values.split_half_ref();
	let mut out = FieldBuffer::zeros(values.log_len() - 1);
	(out.as_mut(), lo.as_ref(), hi.as_ref())
		.into_par_iter()
		.for_each(|(out, &lo, &hi)| {
			*out = lo + broadcast_scalar * (hi - lo);
		});
	out
}
