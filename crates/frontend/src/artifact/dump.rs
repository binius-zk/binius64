// Copyright 2025 Irreducible Inc.
use std::collections::{BTreeMap, BTreeSet};

use serde::ser::SerializeStruct;

use crate::{
	gates::opcode::Opcode,
	ir::{Gate, GateGraph, path::PathSpec},
};

struct PathSpecData {
	name: String,
	gates: Vec<Gate>,
	parent: Option<PathSpec>,
	children: Vec<PathSpec>,
	breakdown: Option<GateBreakdown>,
	cum_breakdown: Option<GateBreakdown>,
}

impl PathSpecData {
	const fn new() -> Self {
		PathSpecData {
			name: String::new(),
			gates: Vec::new(),
			parent: None,
			children: Vec::new(),
			breakdown: None,
			cum_breakdown: None,
		}
	}
}

#[derive(Clone)]
struct GateBreakdown {
	/// Shows how many opcodes of every type there is.
	by_opcode: BTreeMap<Opcode, usize>,
}

impl GateBreakdown {
	fn count(gg: &GateGraph, gates: &[Gate]) -> GateBreakdown {
		let mut breakdown = GateBreakdown {
			by_opcode: BTreeMap::new(),
		};
		for gate in gates {
			*breakdown
				.by_opcode
				.entry(gg.gates[*gate].opcode)
				.or_insert(0) += 1;
		}
		breakdown
	}

	fn merge(mut self, other: &GateBreakdown) -> GateBreakdown {
		for (&opcode, count) in &other.by_opcode {
			*self.by_opcode.entry(opcode).or_insert(0) += count;
		}
		self
	}
}

impl serde::Serialize for GateBreakdown {
	/// Serializes the counts under opcode names.
	///
	/// Names are rendered here rather than while counting.
	/// So a circuit with a million gates names each opcode once, not once per gate.
	fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
		let by_name: BTreeMap<String, usize> = self
			.by_opcode
			.iter()
			.map(|(opcode, count)| (format!("{opcode:?}"), *count))
			.collect();

		let mut breakdown = serializer.serialize_struct("GateBreakdown", 1)?;
		breakdown.serialize_field("by_opcode", &by_name)?;
		breakdown.end()
	}
}

struct Cx {
	data: BTreeMap<PathSpec, PathSpecData>,
	post_order: Vec<PathSpec>,
}

impl Cx {
	const fn new() -> Self {
		Self {
			data: BTreeMap::new(),
			post_order: Vec::new(),
		}
	}

	fn bucket_gates(&mut self, gg: &GateGraph) {
		self.data
			.insert(gg.path_spec_tree.root(), PathSpecData::new());

		// First, collect all PathSpecs that have gates
		let mut path_specs_with_gates = BTreeSet::new();
		for gate in gg.gates.keys() {
			path_specs_with_gates.insert(gg.gate_origin[gate]);
		}

		// Add all ancestors of PathSpecs with gates to ensure complete hierarchy
		let mut all_needed_paths = BTreeSet::new();
		for &path_spec in &path_specs_with_gates {
			let mut current = path_spec;
			loop {
				all_needed_paths.insert(current);
				if let Some(parent) = gg.path_spec_tree.parent(current) {
					current = parent;
				} else {
					break;
				}
			}
		}

		// Ensure all needed paths exist in data map
		for path_spec in all_needed_paths {
			self.data.entry(path_spec).or_insert_with(PathSpecData::new);
		}

		// Now add gates to their respective PathSpecs
		for gate in gg.gates.keys() {
			self.data
				.get_mut(&gg.gate_origin[gate])
				.unwrap()
				.gates
				.push(gate);
		}
	}

	fn recover_hierarchy(&mut self, gg: &GateGraph) {
		let paths = self.data.keys().cloned().collect::<Vec<_>>();
		for current in paths {
			if let Some(parent) = gg.path_spec_tree.parent(current) {
				self.data.get_mut(&current).unwrap().parent = Some(parent);
				self.data.get_mut(&parent).unwrap().children.push(current);
			}
		}
	}

	fn symbolicate_paths(&mut self, gg: &GateGraph) {
		for (path, data) in &mut self.data {
			gg.path_spec_tree.stringify(*path, &mut data.name);
		}
	}

	fn compute_breakdowns(&mut self, gg: &GateGraph) {
		for data in self.data.values_mut() {
			data.breakdown = Some(GateBreakdown::count(gg, &data.gates));
		}
	}

	/// Computes the post-order order of traversal. That's where we visit the children first and
	/// then the parent.
	///
	/// Requires to be called after recovering the hierarchy.
	fn compute_postorder(&mut self, gg: &GateGraph) {
		fn collect_postorder(
			data: &BTreeMap<PathSpec, PathSpecData>,
			visited: &mut BTreeSet<PathSpec>,
			postorder: &mut Vec<PathSpec>,
			current: PathSpec,
		) {
			if visited.contains(&current) {
				return;
			}
			visited.insert(current);
			if let Some(node_data) = data.get(&current) {
				for &child in &node_data.children {
					collect_postorder(data, visited, postorder, child);
				}
			}

			// Then visit current node (post-order)
			postorder.push(current);
		}

		let mut visited = BTreeSet::new();

		// Start from root to ensure proper traversal
		let root = gg.path_spec_tree.root();
		collect_postorder(&self.data, &mut visited, &mut self.post_order, root);
	}

	/// Traverses the paths in the post order and computes the cumulative gate breakdowns for
	/// each path.
	fn compute_cum_breakdowns(&mut self) {
		// Children are visited before their parent, so each child total is ready when read.
		for path_spec in self.post_order.clone() {
			// The child list is taken out rather than copied, so the map can be read while
			// accumulating and the list handed straight back.
			let children = std::mem::take(&mut self.data.get_mut(&path_spec).unwrap().children);

			let mut cum_breakdown = self.data[&path_spec].breakdown.as_ref().unwrap().clone();
			for child in &children {
				if let Some(child_cum) = self.data[child].cum_breakdown.as_ref() {
					cum_breakdown = cum_breakdown.merge(child_cum);
				}
			}

			let data = self.data.get_mut(&path_spec).unwrap();
			data.children = children;
			data.cum_breakdown = Some(cum_breakdown);
		}
	}

	/// Builds the hierarchical SubcircuitInfo structure starting from root
	fn build_subcircuit_info(&self, gg: &GateGraph) -> SubcircuitInfo {
		let root = gg.path_spec_tree.root();
		self.build_subcircuit_info_recursive(root)
	}

	/// Recursively builds SubcircuitInfo for a given PathSpec and its children
	fn build_subcircuit_info_recursive(&self, path_spec: PathSpec) -> SubcircuitInfo {
		let data = &self.data[&path_spec];

		// Build children first (pre-order traversal for construction)
		let mut children = Vec::new();
		for &child_path in &data.children {
			children.push(self.build_subcircuit_info_recursive(child_path));
		}

		// Calculate total gates from cumulative breakdown
		let n_gates = data
			.cum_breakdown
			.as_ref()
			.unwrap()
			.by_opcode
			.values()
			.sum();

		SubcircuitInfo {
			name: data.name.clone(),
			n_gates,
			children,
			breakdown: data.cum_breakdown.as_ref().unwrap().clone(),
		}
	}
}

#[derive(serde::Serialize)]
struct SubcircuitInfo {
	name: String,
	n_gates: usize,
	children: Vec<SubcircuitInfo>,
	breakdown: GateBreakdown,
}

/// Dumps a hierarchical JSON representation of the given circuit.
pub(crate) fn dump_composition(gg: &GateGraph) -> String {
	let mut cx = Cx::new();
	cx.bucket_gates(gg);
	cx.recover_hierarchy(gg);
	cx.compute_postorder(gg);
	cx.compute_breakdowns(gg);
	cx.compute_cum_breakdowns();
	cx.symbolicate_paths(gg);

	let subcircuit_info = cx.build_subcircuit_info(gg);
	serde_json::to_string_pretty(&subcircuit_info).unwrap()
}
