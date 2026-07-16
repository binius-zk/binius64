// Copyright 2026 The Binius Developers
//! Dead-code elimination for the gate graph.
//!
//! A gate is *live* when it can affect the constraint system:
//! - it has no output wires — an assertion, which is itself a constraint;
//! - one of its outputs is observable — a public input/output, or a wire pinned as committed;
//! - one of its outputs feeds a live gate.
//!
//! A dead gate only constrains wires that no assertion or public output transitively reads.
//! Skipping it at constraint-emission time drops those AND/IMUL constraints.
//! Its output wire is then referenced by nothing.
//! An unreferenced internal wire is allocated as scratch, never committed.
//!
//! Dropping a dead constraint is a bit-exact rewrite.
//! The pinned wire is unread by any output or assertion.
//! Removing a constraint on an unread wire cannot change soundness.

use cranelift_entity::EntitySet;

use super::{
	gate_graph::{Gate, GateGraph, Wire, WireKind},
	hints::HintRegistry,
};

/// Returns the gates that can affect the constraint system.
///
/// A gate qualifies when it is an assertion, or when it transitively feeds an observable wire.
/// Observable means a public input/output, or a wire pinned as committed.
///
/// Rebuilds the use-def chains first.
/// The result stays correct after passes that rewire gate inputs, such as constant propagation.
pub fn live_gates(
	graph: &mut GateGraph,
	force_committed: &EntitySet<Wire>,
	hint_registry: &HintRegistry,
) -> EntitySet<Gate> {
	// Earlier passes rewire gate inputs, so refresh the def and use links before walking them.
	graph.rebuild_use_def_chains(hint_registry);

	let mut live = EntitySet::new();
	let mut work: Vec<Gate> = Vec::new();

	// Seed 1: assertions.
	// A gate with no output wire produces only a constraint, so it is a root and never dead.
	for (gate, data) in graph.gates.iter() {
		if data
			.gate_param_with_registry(hint_registry)
			.outputs
			.is_empty()
			&& live.insert(gate)
		{
			work.push(gate);
		}
	}

	// Seed 2: gates that define an observable wire.
	// Observable means a public input/output, or a wire the circuit author pinned as committed.
	for (wire, wire_data) in graph.wires.iter() {
		let observable =
			matches!(wire_data.kind, WireKind::Inout) || force_committed.contains(wire);
		if observable
			&& let Some(def) = graph.wire_def[wire]
			&& live.insert(def)
		{
			work.push(def);
		}
	}

	// Propagate liveness backward along def→use edges.
	// A live gate needs its inputs, so each input wire and the gate defining it are also live.
	while let Some(gate) = work.pop() {
		// Copy the inputs out first, ending the graph borrow before the set is mutated.
		let inputs = graph
			.gate_data(gate)
			.gate_param_with_registry(hint_registry)
			.inputs
			.to_vec();
		for input in inputs {
			if let Some(def) = graph.wire_def[input]
				&& live.insert(def)
			{
				work.push(def);
			}
		}
	}

	live
}

#[cfg(test)]
mod tests {
	use binius_core::word::Word;

	use super::*;
	use crate::compiler::gate::opcode::Opcode;

	#[test]
	fn dead_gate_excluded_live_gate_kept() {
		// Invariant: a gate is live only if it transitively reaches an assertion or public output.
		//
		// Fixture: two identical AND gates over the same public inputs x, y.
		//
		//   live_out = x & y  →  fed into an AssertZero (a sink)  →  LIVE
		//   dead_out = x & y  →  read by nothing                  →  DEAD
		//
		// Sharing inputs with the live cone must not, on its own, keep the parallel gate alive.
		let mut graph = GateGraph::new();
		let root = graph.path_spec_tree.root();
		let registry = HintRegistry::new();

		// Public inputs are roots on their own, but they have no defining gate to keep alive.
		let x = graph.add_inout();
		let y = graph.add_inout();

		// Live cone: the AND output flows into an assertion, which is the sink that anchors it.
		let live_out = graph.add_internal();
		let live_gate = graph.emit_gate(root, Opcode::Band, vec![x, y], vec![live_out]);
		graph.emit_gate(root, Opcode::AssertZero, vec![live_out], vec![]);

		// Dead cone: an identical AND whose output no gate consumes.
		let dead_out = graph.add_internal();
		let dead_gate = graph.emit_gate(root, Opcode::Band, vec![x, y], vec![dead_out]);

		let live = live_gates(&mut graph, &EntitySet::new(), &registry);

		// The assertion pulls its feeding gate into the live set.
		assert!(live.contains(live_gate), "gate feeding an assertion must be live");
		// Nothing reads the parallel gate, so it stays out of the live set.
		assert!(!live.contains(dead_gate), "gate whose output is unread must be dead");
	}

	#[test]
	fn force_committed_output_is_live() {
		// Invariant: pinning a wire as committed makes it observable, so its producer is live.
		//
		// Fixture: a single AND whose output has no reader other than the commitment pin.
		//
		//   out = a & all_ones  →  pinned committed  →  LIVE (must appear in the witness)
		let mut graph = GateGraph::new();
		let root = graph.path_spec_tree.root();
		let registry = HintRegistry::new();

		let a = graph.add_inout();
		let b = graph.add_constant(Word::ALL_ONE);
		let out = graph.add_internal();
		let gate = graph.emit_gate(root, Opcode::Band, vec![a, b], vec![out]);

		// Mutation: pin the otherwise-unread output as committed.
		let mut force_committed = EntitySet::new();
		force_committed.insert(out);

		let live = live_gates(&mut graph, &force_committed, &registry);
		// The pin is the sole reason the gate survives.
		assert!(live.contains(gate), "gate defining a committed wire must be live");
	}
}
