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
/// A column starts out `Borrowed` when pushed by reference. The first fold writes the folded
/// values into a fresh half-size owned buffer, so borrowed columns are never copied at full size.
enum Column<'a, P: PackedField> {
	Borrowed(FieldSlice<'a, P>),
	Owned(FieldBuffer<P>),
}

impl<P: PackedField> Column<'_, P> {
	fn as_slice(&self) -> FieldSlice<'_, P> {
		match self {
			Column::Borrowed(slice) => slice.to_ref(),
			Column::Owned(buffer) => buffer.to_ref(),
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
