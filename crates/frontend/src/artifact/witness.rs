// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

//! Assigning a circuit's input wires, and what population reports when an assertion fails.

use std::{
	fmt,
	ops::{Index, IndexMut},
};

use binius_core::{ValueVec, Word};
use binius_utils::strided_array::StridedArray2DViewMut;

use crate::{Circuit, Wire};

/// A single assertion that did not hold while populating the witness.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AssertionFailure {
	/// The circuit path the assertion was declared under, such as `.sha256.round[3]`.
	///
	/// Empty for an assertion at the circuit root.
	pub path: String,
	/// What the assertion required, against the words it saw instead.
	///
	/// A diagnostic for a human to read.
	/// Its wording is not part of the API, so do not match on it.
	pub detail: String,
}

impl fmt::Display for AssertionFailure {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		if self.path.is_empty() {
			f.write_str(&self.detail)
		} else {
			write!(f, "{}: {}", self.path, self.detail)
		}
	}
}

/// Witness population failed because the circuit is not satisfied.
///
/// Evaluation runs to completion rather than stopping at the first bad assertion.
/// So a caller sees every violation at once.
///
/// The retained list is capped at [`MAX_ASSERTION_FAILURES`](crate::MAX_ASSERTION_FAILURES).
/// [`Self::total`] counts every violation, capped or not.
/// The two disagree exactly when the cap was reached.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub struct PopulateError {
	/// The failures that were retained, in the order evaluation found them.
	pub failures: Vec<AssertionFailure>,
	/// How many assertions failed in total, which may exceed `failures.len()`.
	pub total: usize,
}

impl fmt::Display for PopulateError {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		// No trailing newline: the caller owns how this is framed.
		write!(f, "circuit not satisfied: {} assertion(s) failed", self.total)?;
		for failure in &self.failures {
			write!(f, "\n  {failure}")?;
		}
		let omitted = self.total.saturating_sub(self.failures.len());
		if omitted > 0 {
			write!(f, "\n  ... and {omitted} more, omitted")?;
		}
		Ok(())
	}
}

/// A helper struct for filling witness values in a circuit.
pub struct WitnessFiller<'a> {
	pub(crate) circuit: &'a Circuit,
	pub(crate) value_vec: ValueVec,
}

impl WitnessFiller<'_> {
	/// Destruct the witness filler and extracts the underlying value vector.
	pub fn into_value_vec(self) -> ValueVec {
		self.value_vec
	}

	/// Returns a reference to the underlying value vector.
	pub const fn value_vec(&self) -> &ValueVec {
		&self.value_vec
	}

	/// Returns a mutable reference to the underlying value vector.
	///
	/// Raw access is what plants a witness the evaluator would never produce.
	pub const fn value_vec_mut(&mut self) -> &mut ValueVec {
		&mut self.value_vec
	}

	/// Populates the given wires from bytes as little-endian packed 64-bit words.
	///
	/// If `bytes` is not a multiple of 8, the last word is zero-padded.
	/// Any wires past those needed to hold `bytes` are filled with `Word::ZERO`.
	///
	/// # Panics
	/// Panics if `bytes.len()` exceeds `wires.len() * 8`.
	pub fn pack_bytes_le(&mut self, wires: &[Wire], bytes: &[u8]) {
		let max_value_size = wires.len() * 8;
		assert!(
			bytes.len() <= max_value_size,
			"bytes length {} exceeds maximum {}",
			bytes.len(),
			max_value_size
		);

		// Pack each 8-byte chunk into one little-endian word.
		for (&wire, chunk) in std::iter::zip(wires, bytes.chunks(8)) {
			let mut chunk_arr = [0u8; 8];
			chunk_arr[..chunk.len()].copy_from_slice(chunk);
			self[wire] = Word(u64::from_le_bytes(chunk_arr));
		}

		// Zero any wires the bytes did not reach.
		for &wire in &wires[bytes.len().div_ceil(8)..] {
			self[wire] = Word::ZERO;
		}
	}
}

impl Index<Wire> for WitnessFiller<'_> {
	type Output = Word;

	/// # Panics
	///
	/// Panics if the wire's storage is a pooled scratch slot shared with another value.
	fn index(&self, wire: Wire) -> &Self::Output {
		let index = self.circuit.witness_index(wire);
		self.circuit.assert_not_pooled(wire, index);
		&self.value_vec[index]
	}
}

impl IndexMut<Wire> for WitnessFiller<'_> {
	/// # Panics
	///
	/// Panics if the wire's storage is a pooled scratch slot shared with another value.
	fn index_mut(&mut self, wire: Wire) -> &mut Self::Output {
		let index = self.circuit.witness_index(wire);
		self.circuit.assert_not_pooled(wire, index);
		&mut self.value_vec[index]
	}
}

/// Assigns witness input wires of one instance into a [`ValueTable`] working buffer.
///
/// Indexing by [`Wire`] targets that wire's row in the instance's column, mirroring the
/// single-instance [`WitnessFiller`].
///
/// [`ValueTable`]: binius_core::ValueTable
pub struct BatchWitnessFiller<'a, 'v> {
	circuit: &'a Circuit,
	values: &'a mut StridedArray2DViewMut<'v, Word>,
	instance: usize,
}

impl<'a, 'v> BatchWitnessFiller<'a, 'v> {
	/// A filler targeting one instance's column of a batch working buffer.
	pub(crate) const fn new(
		circuit: &'a Circuit,
		values: &'a mut StridedArray2DViewMut<'v, Word>,
		instance: usize,
	) -> Self {
		Self {
			circuit,
			values,
			instance,
		}
	}
}

impl Index<Wire> for BatchWitnessFiller<'_, '_> {
	type Output = Word;

	fn index(&self, wire: Wire) -> &Self::Output {
		&self.values[(self.circuit.witness_row(wire), self.instance)]
	}
}

impl IndexMut<Wire> for BatchWitnessFiller<'_, '_> {
	fn index_mut(&mut self, wire: Wire) -> &mut Self::Output {
		let row = self.circuit.witness_row(wire);
		&mut self.values[(row, self.instance)]
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	fn failure(path: &str, detail: &str) -> AssertionFailure {
		AssertionFailure {
			path: path.to_string(),
			detail: detail.to_string(),
		}
	}

	#[test]
	fn a_failure_at_the_root_renders_without_a_separator() {
		// A root assertion has no path, so there is nothing to prefix and no stray colon.
		assert_eq!(failure("", "Word(0x1) != Word(0x2)").to_string(), "Word(0x1) != Word(0x2)");
	}

	#[test]
	fn a_nested_failure_renders_path_then_detail() {
		// The path and the detail are stored apart; rendering is what joins them.
		assert_eq!(
			failure(".sha256.round", "Word(0x1) != 0").to_string(),
			".sha256.round: Word(0x1) != 0"
		);
	}

	#[test]
	fn the_error_lists_every_retained_failure_and_never_ends_with_a_newline() {
		// Invariant: a caller frames the message, so it must not arrive with its own line break.
		let err = PopulateError {
			failures: vec![failure(".a", "one"), failure(".b", "two")],
			total: 2,
		};
		let rendered = err.to_string();
		assert_eq!(rendered, "circuit not satisfied: 2 assertion(s) failed\n  .a: one\n  .b: two");
		assert!(!rendered.ends_with('\n'));
	}

	#[test]
	fn a_capped_error_reports_how_many_it_dropped() {
		// `total` counts past the cap, so the difference is what the list does not show.
		let err = PopulateError {
			failures: vec![failure(".a", "one")],
			total: 7,
		};
		assert_eq!(
			err.to_string(),
			"circuit not satisfied: 7 assertion(s) failed\n  .a: one\n  ... and 6 more, omitted"
		);
	}

	#[test]
	fn an_uncapped_error_reports_no_omissions() {
		// Equal counts mean the cap was never reached, so no trailing note is added.
		let err = PopulateError {
			failures: vec![failure(".a", "one")],
			total: 1,
		};
		assert_eq!(err.to_string(), "circuit not satisfied: 1 assertion(s) failed\n  .a: one");
	}

	#[test]
	fn the_error_is_a_std_error() {
		// The whole point of the type: it can cross an API boundary as a `dyn Error`.
		let err = PopulateError {
			failures: vec![failure(".a", "one")],
			total: 1,
		};
		let boxed: Box<dyn std::error::Error> = Box::new(err);
		assert!(boxed.to_string().starts_with("circuit not satisfied"));
		assert!(boxed.source().is_none());
	}
}
