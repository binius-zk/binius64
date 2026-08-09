// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers
use std::{
	fmt,
	ops::{Index, IndexMut},
};

use binius_core::{
	constraint_system::{ConstraintSystem, ValueIndex, ValueVec, ValueVecLayout},
	word::Word,
};
use binius_utils::strided_array::StridedArray2DViewMut;
use cranelift_entity::SecondaryMap;

use crate::compiler::{
	dump::dump_composition,
	eval_form::{BatchPopulateError, EvalForm},
	gate_graph::{GateGraph, Wire},
};

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

	fn index(&self, wire: Wire) -> &Self::Output {
		&self.value_vec[self.circuit.witness_index(wire)]
	}
}

impl IndexMut<Wire> for WitnessFiller<'_> {
	fn index_mut(&mut self, wire: Wire) -> &mut Self::Output {
		&mut self.value_vec[self.circuit.witness_index(wire)]
	}
}

/// An artifact that represents a built circuit.
///
/// The difference from [`ConstraintSystem`] is that a circuit retains enough information to
/// perform circuit evaluation to generate internal witness values.
pub struct Circuit {
	gate_graph: GateGraph,
	constraint_system: ConstraintSystem,
	value_vec_layout: ValueVecLayout,
	wire_mapping: SecondaryMap<Wire, ValueIndex>,
	eval_form: EvalForm,
	scratch_peak_live: usize,
}

impl Circuit {
	/// Creates a new circuit with the given shared data and wire mapping. Only used during building
	/// by the circuit builder.
	pub(super) const fn new(
		gate_graph: GateGraph,
		constraint_system: ConstraintSystem,
		value_vec_layout: ValueVecLayout,
		wire_mapping: SecondaryMap<Wire, ValueIndex>,
		eval_form: EvalForm,
		scratch_peak_live: usize,
	) -> Self {
		Self {
			gate_graph,
			constraint_system,
			value_vec_layout,
			wire_mapping,
			eval_form,
			scratch_peak_live,
		}
	}

	/// Returns the smallest scratch segment this circuit could run with.
	///
	/// This is the largest number of uncommitted temporaries alive at the same time.
	/// It is what the segment shrinks to once slots are shared.
	/// It is reported whether or not sharing is on, so the unused headroom stays visible.
	pub const fn scratch_peak_live(&self) -> usize {
		self.scratch_peak_live
	}

	/// For the given wire, returns its index in the witness vector.
	#[inline(always)]
	pub fn witness_index(&self, wire: Wire) -> ValueIndex {
		self.wire_mapping[wire]
	}

	/// For the given wire, returns the row it occupies in a transposed value array.
	///
	/// This is the wire's flat position in the value vector, counting the scratch tail, which is
	/// how [`Self::populate_wire_witness_batched`] numbers the rows it fills.
	#[inline(always)]
	pub fn witness_row(&self, wire: Wire) -> usize {
		self.value_vec_layout.word_offset(self.witness_index(wire))
	}

	/// Creates a new witness filler for this circuit.
	pub fn new_witness_filler(&self) -> WitnessFiller<'_> {
		WitnessFiller {
			circuit: self,
			value_vec: ValueVec::new(&self.value_vec_layout),
		}
	}

	/// Populates non-input values (wires) in the witness.
	///
	/// Specifically, this will evaluate the circuit gate-by-gate and save the results in the
	/// witness vector.
	///
	/// This function expects that the input wires are already filled. The input wires are
	///
	/// - [`CircuitBuilder::add_inout`],
	/// - [`CircuitBuilder::add_witness`] that were not created by the gates,
	///
	/// The wires created by [`CircuitBuilder::add_constant`] (and its convenience methods)
	/// are automatically populated by this function as well.
	///
	/// # Errors
	///
	/// Returns [`PopulateError`] when any assertion fails.
	/// Each failure names the circuit path the assertion was declared under.
	/// Evaluation runs to completion first, so every violation is reported at once.
	///
	/// [`CircuitBuilder::add_constant`]: super::CircuitBuilder::add_constant
	/// [`CircuitBuilder::add_inout`]: super::CircuitBuilder::add_inout
	/// [`CircuitBuilder::add_witness`]: super::CircuitBuilder::add_witness
	pub fn populate_wire_witness(&self, w: &mut WitnessFiller) -> Result<(), PopulateError> {
		// Fill the constant part from the witness.
		for (index, constant) in self.constraint_system.constants.iter().enumerate() {
			w.value_vec[ValueIndex::constant(index as u32)] = *constant;
		}

		// Execute the evaluation form - it modifies the ValueVec in place
		// Pass the PathSpecTree for assertion error symbolication
		self.eval_form
			.evaluate(&mut w.value_vec, Some(&self.gate_graph.path_spec_tree))?;

		Ok(())
	}

	/// Populates non-input values for a batch of instances at once.
	///
	/// This is the structure-of-arrays counterpart to [`Self::populate_wire_witness`]. `values` is
	/// the transposed value array: rows are value-vector indices (in the same order a single
	/// instance's [`ValueVec`] uses) and columns are instances. Its height must be the full
	/// value-vector length (including scratch) and its width is the instance count.
	///
	/// The caller must fill each instance's input rows first — the witness wires and any inout
	/// wires. This function fills the constant rows (broadcasting each constant across every
	/// instance) and then evaluates the circuit gate-by-gate for all instances.
	///
	/// # Errors
	///
	/// If any instance is not satisfiable, returns an error naming the lowest-indexed failing
	/// instance and its assertion failures.
	pub fn populate_wire_witness_batched(
		&self,
		values: &mut StridedArray2DViewMut<'_, Word>,
	) -> Result<(), BatchPopulateError> {
		// Broadcast each constant into its row across every instance. The constants are the same
		// for all instances, so this fills the constant rows uniformly.
		let n_instances = values.width();
		for (index, &constant) in self.constraint_system.constants.iter().enumerate() {
			for instance in 0..n_instances {
				values[(index, instance)] = constant;
			}
		}

		// Evaluate the bytecode across all instances, symbolicating assertion failures.
		self.eval_form
			.evaluate_batched(values, Some(&self.gate_graph.path_spec_tree))
	}

	/// Populates non-input values for a batch of instances split into vertical stripes.
	///
	/// This is the parallel counterpart to [`Self::populate_wire_witness_batched`]. Constants are
	/// broadcast once over the full value array, then the bytecode interpreter runs independently
	/// on disjoint instance-column stripes of at most `stripe_width` columns.
	///
	/// # Errors
	///
	/// If any instance is not satisfiable, returns an error naming a failing instance and its
	/// assertion failures. The reported instance is not guaranteed to be the lowest failing
	/// instance across all stripes.
	pub fn populate_wire_witness_batched_parallel(
		&self,
		mut values: StridedArray2DViewMut<'_, Word>,
		stripe_width: usize,
	) -> Result<(), BatchPopulateError> {
		assert!(stripe_width > 0, "stripe width must be positive");

		// Broadcast each constant into its row across every instance. The constants are the same
		// for all instances, so this fills the constant rows uniformly before the stripes split.
		let n_instances = values.width();
		for (index, &constant) in self.constraint_system.constants.iter().enumerate() {
			for instance in 0..n_instances {
				values[(index, instance)] = constant;
			}
		}

		// Evaluate independent instance stripes in parallel, symbolicating assertion failures.
		self.eval_form.evaluate_batched_parallel(
			values,
			stripe_width,
			Some(&self.gate_graph.path_spec_tree),
		)
	}

	/// Returns the constraint system for this circuit.
	pub const fn constraint_system(&self) -> &ConstraintSystem {
		&self.constraint_system
	}

	/// Returns the layout of the value vector this circuit fills.
	pub const fn value_vec_layout(&self) -> &ValueVecLayout {
		&self.value_vec_layout
	}

	/// Returns the number of gates in this circuit.
	///
	/// Depending on what type of gates this circuit uses, the number of constraints might be
	/// significantly larger.
	pub fn n_gates(&self) -> usize {
		self.gate_graph.gates.len()
	}

	/// Returns the number of evaluation instructions in this circuit.
	pub const fn n_eval_insn(&self) -> usize {
		self.eval_form.n_eval_insn()
	}

	/// Returns a string with a JSON dump that is useful to profile the circuit.
	pub fn simple_json_dump(&self) -> String {
		dump_composition(&self.gate_graph)
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
