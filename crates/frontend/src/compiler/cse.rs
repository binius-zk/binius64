// Copyright 2026 The Binius Developers
//! Common-subexpression elimination on the gate graph.
//!
//! Two gates are structurally identical when they match on all of:
//! - opcode;
//! - constant and input wires;
//! - immediates;
//! - dimensions.
//!
//! Identical gates compute identical outputs.
//! Every occurrence after the first is therefore redundant.
//!
//! The pass runs one forward sweep in emission (topological) order:
//! - each gate's input wires are first rewritten to their canonical form;
//! - the first gate seen with a given structure is kept as canonical;
//! - a later duplicate is collapsed onto the canonical gate and reported as dead.
//!
//! A reader is always visited after the gate that feeds it.
//! Rewriting each gate's own inputs therefore redirects every reader of a collapsed duplicate.
//! The pass needs no use-def chains.
//! It allocates almost nothing.
//!
//! Before this pass only constants were interned.
//! An identical non-constant subexpression built twice now costs one AND/MUL constraint, not two.

use std::hash::{Hash, Hasher};

use cranelift_entity::EntitySet;
use rustc_hash::{FxHashMap, FxHasher};

use super::{
	gate::opcode::Opcode,
	gate_graph::{Gate, GateGraph, Wire, WireKind},
	hints::HintRegistry,
};

/// Deduplicates structurally-identical gates, returning the gates left dead.
///
/// - The first gate seen with a given structure is the canonical one.
/// - A later gate with that structure is redirected onto the canonical gate and marked dead.
///
/// The caller skips dead gates at emission.
/// Each collapsed duplicate then costs one fewer AND/MUL constraint and one fewer committed wire.
pub fn dedup_gates(
	graph: &mut GateGraph,
	force_committed: &EntitySet<Wire>,
	hint_registry: &HintRegistry,
) -> EntitySet<Gate> {
	// Maps a collapsed duplicate's output wire to the canonical wire that replaces it.
	let mut remap: FxHashMap<Wire, Wire> = FxHashMap::default();
	// Maps a structural hash to the first (canonical) gate seen with that hash.
	let mut canonical: FxHashMap<u64, Gate> = FxHashMap::default();
	let mut dead = EntitySet::new();

	// Gate ids are collected up front so the graph can be mutated inside the loop.
	let gate_ids: Vec<Gate> = graph.gates.keys().collect();
	for gate in gate_ids {
		// Hint gates carry their shape in the registry and are rarely duplicated, so skip them.
		if matches!(graph.gate_data(gate).opcode, Opcode::Hint) {
			continue;
		}

		// Rewrite this gate's wires to their canonical form.
		// An input that fed an earlier duplicate is remapped.
		// Output, aux, and scratch wires are never keys in the map, so they pass through unchanged.
		for wire in graph.gates[gate].wires.iter_mut() {
			if let Some(&canon) = remap.get(wire) {
				*wire = canon;
			}
		}

		let hash = structural_hash(graph, gate, hint_registry);

		// A public output or force-committed wire must stay exactly where it is.
		// A gate producing one is never collapsed, though it can still be a canonical target.
		let produces_observable = graph
			.gate_data(gate)
			.gate_param_with_registry(hint_registry)
			.outputs
			.iter()
			.any(|&wire| {
				matches!(graph.wire_data(wire).kind, WireKind::Inout)
					|| force_committed.contains(wire)
			});

		// A structural match is a duplicate only if it is truly identical, not a hash collision.
		let duplicate_of = (!produces_observable)
			.then(|| canonical.get(&hash).copied())
			.flatten()
			.filter(|&canon| structurally_equal(graph, gate, canon, hint_registry));

		match duplicate_of {
			// Map each output of the duplicate onto the canonical gate's matching output.
			Some(canon) => {
				let param = graph
					.gate_data(gate)
					.gate_param_with_registry(hint_registry);
				let dup_outputs = param.outputs.to_vec();
				let canon_param = graph
					.gate_data(canon)
					.gate_param_with_registry(hint_registry);
				let canon_outputs = canon_param.outputs.to_vec();
				for (dup_out, canon_out) in dup_outputs.into_iter().zip(canon_outputs) {
					remap.insert(dup_out, canon_out);
				}
				dead.insert(gate);
			}
			// Reached for a first-of-its-hash gate, an observable gate, or a rare collision.
			// Record it as canonical, keeping any earlier entry for this hash.
			None => {
				canonical.entry(hash).or_insert(gate);
			}
		}
	}

	dead
}

/// Hashes a gate's structural identity: opcode, constant and input wires, immediates, dimensions.
///
/// Output, auxiliary, and scratch wires are excluded.
/// They are freshly allocated per gate, so keeping them would make identical gates hash apart.
fn structural_hash(graph: &GateGraph, gate: Gate, hint_registry: &HintRegistry) -> u64 {
	let data = graph.gate_data(gate);
	let param = data.gate_param_with_registry(hint_registry);
	let mut hasher = FxHasher::default();
	data.opcode.hash(&mut hasher);
	param.constants.hash(&mut hasher);
	param.inputs.hash(&mut hasher);
	data.immediates.hash(&mut hasher);
	data.dimensions.hash(&mut hasher);
	hasher.finish()
}

/// Reports whether two gates have the same structural identity, comparing the fields directly.
///
/// This is the exact check behind [`structural_hash`], used to reject hash collisions.
fn structurally_equal(graph: &GateGraph, a: Gate, b: Gate, hint_registry: &HintRegistry) -> bool {
	let (da, db) = (graph.gate_data(a), graph.gate_data(b));
	if da.opcode != db.opcode || da.immediates != db.immediates || da.dimensions != db.dimensions {
		return false;
	}
	let (pa, pb) =
		(da.gate_param_with_registry(hint_registry), db.gate_param_with_registry(hint_registry));
	pa.constants == pb.constants && pa.inputs == pb.inputs
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::compiler::gate_graph::GateGraph;

	#[test]
	fn duplicate_and_gate_is_collapsed() {
		// Invariant: two ANDs over the same inputs compute the same value.
		// The second is redundant and should collapse into the first.
		//
		// Fixture: g1 = x & y and g2 = x & y, with an assertion reading the duplicate's output.
		//
		//   g1: x & y -> o1  (canonical)
		//   g2: x & y -> o2  (duplicate -> dead, readers of o2 point at o1)
		let mut graph = GateGraph::new();
		let root = graph.path_spec_tree.root();
		let registry = HintRegistry::new();

		let x = graph.add_inout();
		let y = graph.add_inout();

		let o1 = graph.add_internal();
		let g1 = graph.emit_gate(root, Opcode::Band, vec![x, y], vec![o1]);
		let o2 = graph.add_internal();
		let g2 = graph.emit_gate(root, Opcode::Band, vec![x, y], vec![o2]);

		// A reader of the duplicate output, to check the redirect lands on the canonical wire.
		let reader = graph.emit_gate(root, Opcode::AssertZero, vec![o2], vec![]);

		let dead = dedup_gates(&mut graph, &EntitySet::new(), &registry);

		// The canonical gate survives and the duplicate collapses.
		assert!(!dead.contains(g1), "canonical gate must survive");
		assert!(dead.contains(g2), "duplicate gate must be dead");

		// The reader now consumes the canonical output, not the dead duplicate's.
		let reader_inputs = graph.gate_data(reader).gate_param().inputs.to_vec();
		assert_eq!(reader_inputs, vec![o1], "reader must be redirected to the canonical wire");
	}

	#[test]
	fn distinct_inputs_are_not_collapsed() {
		// Different input wires mean different values, so neither gate is a duplicate.
		let mut graph = GateGraph::new();
		let root = graph.path_spec_tree.root();
		let registry = HintRegistry::new();

		let x = graph.add_inout();
		let y = graph.add_inout();
		let z = graph.add_inout();

		let o1 = graph.add_internal();
		let g1 = graph.emit_gate(root, Opcode::Band, vec![x, y], vec![o1]);
		let o2 = graph.add_internal();
		let g2 = graph.emit_gate(root, Opcode::Band, vec![x, z], vec![o2]);

		let dead = dedup_gates(&mut graph, &EntitySet::new(), &registry);
		assert!(!dead.contains(g1));
		assert!(!dead.contains(g2), "gates with different inputs must both survive");
	}

	#[test]
	fn transitive_duplicate_is_collapsed() {
		// A duplicate feeding another duplicate collapses in a single sweep.
		// Rewriting the first pair's output makes the second pair structurally identical.
		//
		//   a1 = x & y      a2 = x & y            (a2 -> a1)
		//   b1 = a1 & x     b2 = a2 & x  == a1 & x (b2 -> b1, after a2's input is canonicalized)
		let mut graph = GateGraph::new();
		let root = graph.path_spec_tree.root();
		let registry = HintRegistry::new();

		let x = graph.add_inout();
		let y = graph.add_inout();

		let a1 = graph.add_internal();
		graph.emit_gate(root, Opcode::Band, vec![x, y], vec![a1]);
		let a2 = graph.add_internal();
		graph.emit_gate(root, Opcode::Band, vec![x, y], vec![a2]);

		let b1 = graph.add_internal();
		let g_b1 = graph.emit_gate(root, Opcode::Band, vec![a1, x], vec![b1]);
		let b2 = graph.add_internal();
		let g_b2 = graph.emit_gate(root, Opcode::Band, vec![a2, x], vec![b2]);

		graph.emit_gate(root, Opcode::AssertZero, vec![b2], vec![]);

		let dead = dedup_gates(&mut graph, &EntitySet::new(), &registry);

		// The second AND cone collapses onto the first once its input a2 is remapped to a1.
		assert!(!dead.contains(g_b1), "first b gate is canonical");
		assert!(dead.contains(g_b2), "second b gate must collapse transitively");
	}

	#[test]
	fn observable_duplicate_is_kept() {
		// A duplicate whose output is force-committed must not be collapsed.
		// Its wire is observable and must stay in place.
		let mut graph = GateGraph::new();
		let root = graph.path_spec_tree.root();
		let registry = HintRegistry::new();

		let x = graph.add_inout();
		let y = graph.add_inout();

		let o1 = graph.add_internal();
		graph.emit_gate(root, Opcode::Band, vec![x, y], vec![o1]);
		let o2 = graph.add_internal();
		let g2 = graph.emit_gate(root, Opcode::Band, vec![x, y], vec![o2]);

		// Pin the duplicate's output, making it observable.
		let mut force_committed = EntitySet::new();
		force_committed.insert(o2);

		let dead = dedup_gates(&mut graph, &force_committed, &registry);
		assert!(!dead.contains(g2), "gate producing a committed wire must be kept");
	}
}
