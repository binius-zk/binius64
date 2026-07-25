// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers
//! Circuit representation in the evaluation form.
//!
//! The main purpose of the evaluation form is to evaluate and assign the intermediate witness
//! values. Those are also referred as internal wires.

mod assertion;
mod batch;
mod builder;
mod const_eval;
mod exec;
mod scalar;
#[cfg(test)]
mod tests;

use batch::BatchExecutionContext;
pub use batch::BatchPopulateError;
use binius_core::{ValueIndex, ValueVec, Word};
use binius_utils::{rayon::prelude::*, strided_array::StridedArray2DViewMut};
pub use builder::BytecodeBuilder;
pub use const_eval::evaluate_gate_constants;
use cranelift_entity::SecondaryMap;
use exec::Executor;
use scalar::ExecutionContext;

use crate::compiler::{
	circuit::PopulateError,
	gate,
	gate_graph::{GateGraph, Wire},
	hints::HintRegistry,
	pathspec::PathSpecTree,
};

/// Compiled evaluation form for circuit witness computation
pub struct EvalForm {
	/// Compiled bytecode instructions
	bytecode: Vec<u8>,
	/// Number of evaluation instructions
	n_eval_insn: usize,
	/// Registered hint handlers
	hint_registry: HintRegistry,
}

impl EvalForm {
	/// Build the evaluation form from the gate graph.
	///
	/// `hint_registry` already holds every hint the caller registered via
	/// [`CircuitBuilder::call_hint`](crate::compiler::CircuitBuilder::call_hint); bytecode
	/// emission only reads from it to resolve `Opcode::Hint` gates.
	pub(crate) fn build(
		gate_graph: &GateGraph,
		wire_mapping: &SecondaryMap<Wire, ValueIndex>,
		hint_registry: HintRegistry,
	) -> Self {
		let mut builder = BytecodeBuilder::new();

		// Combined wire to register mapping
		let wire_to_reg = |wire: Wire| -> u32 {
			if let Some(&ValueIndex(idx)) = wire_mapping.get(wire) {
				idx // ValueVec index
			} else {
				panic!("Wire {wire:?} not mapped");
			}
		};

		// Build bytecode for each gate
		for (gate_id, data) in gate_graph.gates.iter() {
			gate::emit_gate_bytecode(
				gate_id,
				data,
				gate_graph,
				&mut builder,
				wire_to_reg,
				&hint_registry,
			);
		}

		let (bytecode, n_eval_insn) = builder.finalize();
		EvalForm {
			bytecode,
			n_eval_insn,
			hint_registry,
		}
	}

	/// Execute the evaluation form to populate witness values
	pub fn evaluate(
		&self,
		value_vec: &mut ValueVec,
		path_spec_tree: Option<&PathSpecTree>,
	) -> Result<(), PopulateError> {
		let mut ctx = ExecutionContext::new(value_vec);
		self.executor().run(&mut ctx);
		ctx.check_assertions(path_spec_tree)
	}

	/// Execute the evaluation form over a batch of instances at once.
	///
	/// `values` is the transposed value array: rows are value-vector indices and columns are
	/// instances. The constant and input rows must already be populated for every instance. This
	/// is the structure-of-arrays counterpart to [`Self::evaluate`].
	pub fn evaluate_batched(
		&self,
		values: &mut StridedArray2DViewMut<'_, Word>,
		path_spec_tree: Option<&PathSpecTree>,
	) -> Result<(), BatchPopulateError> {
		self.evaluate_stripe(values, 0, path_spec_tree)
	}

	/// Execute the evaluation form over disjoint vertical instance stripes in parallel.
	///
	/// Returns an error from one failing stripe if any instance is unsatisfiable. Unlike
	/// [`Self::evaluate_batched`], this does not guarantee that the reported instance is the
	/// lowest-indexed failing instance across the full batch.
	pub fn evaluate_batched_parallel(
		&self,
		values: StridedArray2DViewMut<'_, Word>,
		stripe_width: usize,
		path_spec_tree: Option<&PathSpecTree>,
	) -> Result<(), BatchPopulateError> {
		assert!(stripe_width > 0, "stripe width must be positive");

		values
			.into_par_strides(stripe_width)
			.enumerate()
			.map(|(stripe_index, mut stripe)| {
				self.evaluate_stripe(&mut stripe, stripe_index * stripe_width, path_spec_tree)
			})
			.collect::<Result<Vec<_>, _>>()?;

		Ok(())
	}

	/// Evaluate the bytecode over a view whose local column 0 is the global `instance_offset`.
	fn evaluate_stripe(
		&self,
		values: &mut StridedArray2DViewMut<'_, Word>,
		instance_offset: usize,
		path_spec_tree: Option<&PathSpecTree>,
	) -> Result<(), BatchPopulateError> {
		let mut ctx = BatchExecutionContext::new(values, instance_offset);
		self.executor().run(&mut ctx);
		ctx.check_assertions(path_spec_tree)
	}

	/// A fresh executor over this form's bytecode, with its cursor at the first instruction.
	fn executor(&self) -> Executor<'_> {
		Executor::new(&self.bytecode, &self.hint_registry)
	}

	/// Get the number of evaluation instructions
	pub const fn n_eval_insn(&self) -> usize {
		self.n_eval_insn
	}

	/// Returns the compiled evaluation bytecode.
	pub fn bytecode(&self) -> &[u8] {
		&self.bytecode
	}
}
