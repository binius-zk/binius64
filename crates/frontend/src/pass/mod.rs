// Copyright 2026 The Binius Developers

//! Transformations over the gate graph, each rewriting it in place or reporting what it found.

use crate::ir::Gate;

pub mod const_prop;
pub mod cse;
pub mod dce;
pub mod fusion;
pub mod layout;
pub mod zero_fold;

use crate::ir::{
	GateBody, GateGraph,
	path::{PathSpec, PathSpecTree},
};

/// What a built circuit keeps once the gate graph has done its job.
///
/// Building walks the full graph to compile the constraint system and the evaluation form.
/// That walk touches every wire and every gate's inputs, outputs and immediates.
/// None of it is needed again once building finishes.
///
/// A circuit only reads back the path names for assertions and dumps afterwards.
/// It also reads one path-and-kind pair per gate, for the composition breakdown.
/// This holds exactly that, so the rest of the graph can be dropped.
pub struct BuiltGates {
	/// The tree of circuit paths, named at build time and read back for assertions and dumps.
	pub path_spec_tree: PathSpecTree,
	/// One entry per gate, in construction order.
	///
	/// Dead and duplicate gates are included, exactly as the source graph counted them.
	pub gate_records: Vec<(PathSpec, GateBody)>,
}

impl BuiltGates {
	/// Extracts what a built circuit keeps, dropping the rest of the graph.
	pub fn from_graph(graph: GateGraph) -> Self {
		let gate_records = graph
			.gates
			.iter()
			.map(|(gate, data)| (graph.gate_origin[gate], data.body))
			.collect();
		Self {
			path_spec_tree: graph.path_spec_tree,
			gate_records,
		}
	}
}

/// A gate whose constant inputs can never satisfy it.
///
/// Constant propagation evaluates a gate once every one of its inputs is a constant.
/// If that evaluation fails, no assignment elsewhere in the circuit can make the gate hold.
/// So the circuit is unsatisfiable regardless of the witness.
#[derive(Debug, thiserror::Error)]
#[error("gate {gate:?} always fails when evaluated with constant inputs: {reason}")]
#[non_exhaustive]
pub struct AlwaysFailingGateError {
	/// The gate whose constant evaluation failed.
	pub gate: Gate,
	/// The message the failing evaluation reported.
	pub reason: String,
}
