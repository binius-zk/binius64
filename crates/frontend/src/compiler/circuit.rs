// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers
use std::{error, fmt};

use binius_core::{
	constraint_system::{ConstraintSystem, ValueIndex, ValueVec},
	word::Word,
};
use binius_utils::strided_array::StridedArray2DViewMut;
use cranelift_entity::SecondaryMap;

use crate::compiler::{
	eval_form::{BatchPopulateError, EvalForm},
	gate_graph::{GateGraph, Wire},
};

/// Error returned when populating wire witness fails due to assertion failures.
#[derive(Debug)]
pub struct PopulateError {
	/// List of assertion failure messages (limited to MAX_ASSERTION_MESSAGES).
	pub messages: Vec<String>,
	/// Total count of assertion failures (may exceed messages.len()).
	pub total_count: usize,
}

impl fmt::Display for PopulateError {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		writeln!(f, "assertions failed:")?;
		for message in &self.messages {
			writeln!(f, "{message}")?;
		}
		if self.total_count > self.messages.len() {
			writeln!(f, "(Some assertions are omitted. Total: {})", self.total_count)?;
		}
		Ok(())
	}
}

impl error::Error for PopulateError {}

/// A helper struct for filling witness values in a circuit.
pub struct WitnessFiller<'a> {
	pub(crate) circuit: &'a Circuit,
	pub(crate) value_vec: ValueVec,
}

impl<'a> WitnessFiller<'a> {
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
}

impl<'a> std::ops::Index<Wire> for WitnessFiller<'a> {
	type Output = Word;

	fn index(&self, wire: Wire) -> &Self::Output {
		&self.value_vec[self.circuit.witness_index(wire)]
	}
}

impl<'a> std::ops::IndexMut<Wire> for WitnessFiller<'a> {
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
	wire_mapping: SecondaryMap<Wire, ValueIndex>,
	eval_form: EvalForm,
}

impl Circuit {
	/// Creates a new circuit with the given shared data and wire mapping. Only used during building
	/// by the circuit builder.
	pub(super) fn new(
		gate_graph: GateGraph,
		constraint_system: ConstraintSystem,
		wire_mapping: SecondaryMap<Wire, ValueIndex>,
		eval_form: EvalForm,
	) -> Self {
		assert!(constraint_system.value_vec_layout.validate().is_ok());
		Self {
			gate_graph,
			constraint_system,
			wire_mapping,
			eval_form,
		}
	}

	/// For the given wire, returns its index in the witness vector.
	#[inline(always)]
	pub fn witness_index(&self, wire: Wire) -> ValueIndex {
		self.wire_mapping[wire]
	}

	/// Creates a new witness filler for this circuit.
	pub fn new_witness_filler(&self) -> WitnessFiller<'_> {
		WitnessFiller {
			circuit: self,
			value_vec: ValueVec::new(self.constraint_system.value_vec_layout.clone()),
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
	/// In case the circuit is not satisfiable (any assertion fails), this function will return
	/// an error with a list of assertion failure messages.
	///
	///
	/// [`CircuitBuilder::add_constant`]: super::CircuitBuilder::add_constant
	/// [`CircuitBuilder::add_inout`]: super::CircuitBuilder::add_inout
	/// [`CircuitBuilder::add_witness`]: super::CircuitBuilder::add_witness
	pub fn populate_wire_witness(&self, w: &mut WitnessFiller) -> Result<(), PopulateError> {
		// Fill the constant part from the witness.
		for (index, constant) in self.constraint_system.constants.iter().enumerate() {
			w.value_vec[ValueIndex(index as u32)] = *constant;
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

	/// Returns the constraint system for this circuit.
	pub const fn constraint_system(&self) -> &ConstraintSystem {
		&self.constraint_system
	}

	/// Returns the evaluation form (witness-filling bytecode) for this circuit.
	pub const fn eval_form(&self) -> &EvalForm {
		&self.eval_form
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
		crate::compiler::dump::dump_composition(&self.gate_graph)
	}
}
