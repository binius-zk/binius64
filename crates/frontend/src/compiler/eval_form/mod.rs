// Copyright 2025 Irreducible Inc.
//! Circuit representation in the evaluation form.
//!
//! The main purpose of the evaluation form is to evaluate and assign the intermediate witness
//! values. Those are also referred as internal wires.

mod builder;
mod const_eval;
mod interpreter;
#[cfg(test)]
mod tests;

use std::{collections::BTreeMap, ops::Range};

use binius_core::{ValueIndex, ValueVec};
use binius_utils::rayon::prelude::*;
pub use builder::BytecodeBuilder;
pub use const_eval::evaluate_gate_constants;
use cranelift_entity::{EntityRef, SecondaryMap};

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
	/// Bytecode ranges for independent connected components.
	parallel_components: Vec<EvalComponent>,
}

#[derive(Debug, Clone)]
struct EvalComponent {
	byte_ranges: Vec<Range<usize>>,
}

struct UnionFind {
	parent: Vec<usize>,
	rank: Vec<u8>,
}

impl UnionFind {
	fn new(len: usize) -> Self {
		Self {
			parent: (0..len).collect(),
			rank: vec![0; len],
		}
	}

	fn find(&mut self, index: usize) -> usize {
		let parent = self.parent[index];
		if parent != index {
			let root = self.find(parent);
			self.parent[index] = root;
		}
		self.parent[index]
	}

	fn union(&mut self, a: usize, b: usize) {
		let mut root_a = self.find(a);
		let mut root_b = self.find(b);
		if root_a == root_b {
			return;
		}
		if self.rank[root_a] < self.rank[root_b] {
			std::mem::swap(&mut root_a, &mut root_b);
		}
		self.parent[root_b] = root_a;
		if self.rank[root_a] == self.rank[root_b] {
			self.rank[root_a] += 1;
		}
	}
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
		let capture_parallel_components = cfg!(feature = "rayon");
		let gate_components = if capture_parallel_components {
			gate_component_roots(gate_graph, &hint_registry)
		} else {
			Vec::new()
		};
		let mut component_ranges = BTreeMap::<usize, Vec<Range<usize>>>::new();
		let mut component_writes = BTreeMap::<usize, Vec<usize>>::new();

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
			let component = capture_parallel_components.then(|| gate_components[gate_id.index()]);
			let start = component.map(|_| builder.byte_len());
			gate::emit_gate_bytecode(
				gate_id,
				data,
				gate_graph,
				&mut builder,
				wire_to_reg,
				&hint_registry,
			);
			if let (Some(component), Some(start)) = (component, start) {
				let end = builder.byte_len();
				if start != end {
					let ranges = component_ranges.entry(component).or_default();
					if let Some(last) = ranges.last_mut() {
						if last.end == start {
							last.end = end;
						} else {
							ranges.push(start..end);
						}
					} else {
						ranges.push(start..end);
					}
				}

				let gate_param = data.gate_param_with_registry(&hint_registry);
				let writes = component_writes.entry(component).or_default();
				for &wire in gate_param
					.outputs
					.iter()
					.chain(gate_param.aux)
					.chain(gate_param.scratch)
				{
					if let Some(&ValueIndex(index)) = wire_mapping.get(wire) {
						writes.push(index as usize);
					}
				}
			}
		}

		let (bytecode, n_eval_insn) = builder.finalize();
		let parallel_components =
			if capture_parallel_components && component_writes_are_disjoint(&component_writes) {
				component_ranges
					.into_values()
					.map(|byte_ranges| EvalComponent { byte_ranges })
					.collect()
			} else {
				Vec::new()
			};
		EvalForm {
			bytecode,
			n_eval_insn,
			hint_registry,
			parallel_components,
		}
	}

	/// Execute the evaluation form to populate witness values
	pub fn evaluate(
		&self,
		value_vec: &mut ValueVec,
		path_spec_tree: Option<&PathSpecTree>,
	) -> Result<(), PopulateError> {
		let mut interpreter = interpreter::Interpreter::new(&self.bytecode, &self.hint_registry);
		interpreter.run_with_value_vec(value_vec, path_spec_tree)?;
		Ok(())
	}

	/// Execute independent evaluation components in parallel.
	pub(crate) fn evaluate_parallel(
		&self,
		value_vec: &mut ValueVec,
		path_spec_tree: Option<&PathSpecTree>,
	) -> Result<(), PopulateError> {
		if self.parallel_components.len() <= 1 {
			return self.evaluate(value_vec, path_spec_tree);
		}

		let store = interpreter::ValueStore::new(value_vec);
		let component_results = self
			.parallel_components
			.par_iter()
			.map(|component| {
				let mut failures = Vec::new();
				let mut total_count = 0;
				for range in &component.byte_ranges {
					let mut interpreter = interpreter::Interpreter::new(
						&self.bytecode[range.clone()],
						&self.hint_registry,
					);
					let (mut range_failures, range_count) = interpreter.run_with_store(store)?;
					failures.append(&mut range_failures);
					total_count += range_count;
				}
				Ok::<_, PopulateError>((failures, total_count))
			})
			.collect::<Vec<_>>();

		let mut failures = Vec::new();
		let mut total_count = 0;
		for result in component_results {
			let (mut component_failures, component_count) = result?;
			failures.append(&mut component_failures);
			total_count += component_count;
		}

		interpreter::report_failures(failures, total_count, path_spec_tree)
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

fn gate_component_roots(gate_graph: &GateGraph, hint_registry: &HintRegistry) -> Vec<usize> {
	let gate_count = gate_graph.gates.len();
	let mut union_find = UnionFind::new(gate_count);
	let mut wire_defs = SecondaryMap::<Wire, Option<_>>::new();

	// Build local use-def data instead of relying on `GateGraph::wire_def`. The graph-level map is
	// only rebuilt by optional optimization passes, while eval-form component splitting must stay
	// correct even when those passes are disabled.
	for (gate, gate_data) in gate_graph.gates.iter() {
		let gate_param = gate_data.gate_param_with_registry(hint_registry);
		// Gate bytecode reads cross-gate values only through constants/inputs. Outputs, aux, and
		// scratch wires are the values written by this gate; tracking all of them here keeps the
		// partition conservative if a future opcode exposes scratch as an input.
		for &wire in gate_param
			.outputs
			.iter()
			.chain(gate_param.aux)
			.chain(gate_param.scratch)
		{
			if let Some(previous) = wire_defs[wire].replace(gate) {
				debug_assert_eq!(previous, gate, "wire {wire:?} is defined by multiple gates");
			}
		}
	}

	for (gate, gate_data) in gate_graph.gates.iter() {
		let gate_param = gate_data.gate_param_with_registry(hint_registry);
		for &wire in gate_param.constants.iter().chain(gate_param.inputs) {
			if let Some(def_gate) = wire_defs[wire] {
				union_find.union(gate.index(), def_gate.index());
			}
		}
	}

	let mut roots = vec![0; gate_count];
	for gate in gate_graph.gates.keys() {
		roots[gate.index()] = union_find.find(gate.index());
	}
	roots
}

fn component_writes_are_disjoint(component_writes: &BTreeMap<usize, Vec<usize>>) -> bool {
	let mut owners = BTreeMap::new();
	for (&component, writes) in component_writes {
		for &index in writes {
			if let Some(owner) = owners.insert(index, component)
				&& owner != component
			{
				return false;
			}
		}
	}
	true
}
