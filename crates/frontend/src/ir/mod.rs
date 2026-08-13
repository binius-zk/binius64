// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

//! The gate graph a circuit is built into, and the wires, gates and paths it is made of.

use std::collections::hash_map::Entry;

use binius_core::word::Word;
use cranelift_entity::{EntityRef, PrimaryMap, SecondaryMap, entity_impl};
use rustc_hash::FxHashMap;
use smallvec::{SmallVec, smallvec};

use crate::{
	gates::opcode::{Opcode, OpcodeShape},
	ir::{
		hints::{HintId, HintRegistry},
		path::{PathSpec, PathSpecTree},
	},
};

pub mod hints;
pub mod path;

/// A wire through which a value flows in and out of gates.
///
/// The difference from `ValueIndex` is that a wire is abstract. Some wires could be moved during
/// compilation and some wires might be pruned altogether.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct Wire(u32);
entity_impl!(Wire);

#[derive(Copy, Clone, Debug)]
pub enum WireKind {
	Constant(Word),
	Inout,
	Witness,
	/// An internal wire is a wire created inside a gate.
	Internal,
	/// A scratch wire is a temporary wire used only during evaluation.
	Scratch,
}
impl WireKind {
	/// Returns `true` if this is a constant wire.
	pub const fn is_const(&self) -> bool {
		matches!(self, WireKind::Constant(_))
	}

	/// The word this wire holds, when it is a constant.
	pub const fn const_value(&self) -> Option<Word> {
		match self {
			WireKind::Constant(word) => Some(*word),
			_ => None,
		}
	}
}

/// Gate ID - identifies a gate in the graph
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Gate(u32);

entity_impl!(Gate);

/// A handy struct that allows a more type safe destructure.
pub struct GateParam<'a> {
	pub constants: &'a [Wire],
	pub inputs: &'a [Wire],
	pub outputs: &'a [Wire],
	pub aux: &'a [Wire],
	pub scratch: &'a [Wire],
	pub imm: &'a [u32],
}

/// Describes a particular gate in the gate graph, it's type, input and output wires and
/// immediate parameters.
pub struct GateData {
	/// The code of operation of this gate.
	pub opcode: Opcode,

	/// The input and output wires of this gate.
	///
	/// They are laid out in the following order:
	///
	/// - Constants
	/// - Inputs
	/// - Outputs
	/// - Aux
	/// - Scratch
	///
	/// The number of input and output wires is specified by the opcode's shape.
	///
	/// Five slots are held inline, which is every fixed-shape opcode but three.
	/// Five is chosen over four because `SmallVec` rounds both to the same 32 bytes.
	/// A wider gate spills to the heap exactly as a vector would.
	pub wires: SmallVec<[Wire; 5]>,

	/// The immediate parameters of this gate.
	///
	/// The immediates contain compile-time parameters of the circuits, such as shift amounts,
	/// byte indices, etc.
	///
	/// The length of the immediates is specified by the opcode's shape.
	///
	/// Two inline slots cover every opcode: `Shift` declares two, `Hint` one, the rest none.
	/// At that width a `SmallVec` is the same 24 bytes as a vector, so this is free.
	pub immediates: SmallVec<[u32; 2]>,

	/// The dimensions of this gate.
	///
	/// This is empty for gates of constant shape. When the shape is variable, the number of
	/// input, output and internal wires is a function of non-empty `dimensions`. This function is
	/// typically linear.
	pub dimensions: Vec<usize>,
}

impl GateData {
	/// Slice this gate's wire vector into its semantic portions.
	///
	/// Panics for [`Opcode::Hint`] gates — those carry their shape in the [`HintRegistry`]
	/// and must use [`gate_param_with_registry`](Self::gate_param_with_registry) instead.
	/// The ~25 per-gate-module callers never see `Opcode::Hint`, so they use this method.
	pub fn gate_param(&self) -> GateParam<'_> {
		self.gate_param_for_shape(self.opcode.shape(&self.dimensions))
	}

	/// Like [`gate_param`](Self::gate_param) but works for [`Opcode::Hint`] gates by looking
	/// up the shape in the provided registry.
	pub fn gate_param_with_registry(&self, registry: &HintRegistry) -> GateParam<'_> {
		self.gate_param_for_shape(self.shape(registry))
	}

	fn gate_param_for_shape(&self, shape: OpcodeShape) -> GateParam<'_> {
		let start_const = 0;
		let end_const = shape.const_in.len();
		let start_input = end_const;
		let end_input = start_input + shape.n_in;
		let start_output = end_input;
		let end_output = start_output + shape.n_out;
		let start_aux = end_output;
		let end_aux = start_aux + shape.n_aux;
		let start_scratch = end_aux;
		let end_scratch = start_scratch + shape.n_scratch;
		GateParam {
			constants: &self.wires[start_const..end_const],
			inputs: &self.wires[start_input..end_input],
			outputs: &self.wires[start_output..end_output],
			aux: &self.wires[start_aux..end_aux],
			scratch: &self.wires[start_scratch..end_scratch],
			imm: &self.immediates,
		}
	}

	/// The gate shape (takes dimensions into account).
	///
	/// For [`Opcode::Hint`] the shape is looked up via `registry`; the hint id lives in
	/// `immediates[0]` and the user dimensions are `&self.dimensions`.
	pub fn shape(&self, registry: &HintRegistry) -> OpcodeShape {
		match self.opcode {
			Opcode::Hint => {
				let hint_id = self.immediates[0];
				let (n_in, n_out) = registry.shape(hint_id, &self.dimensions);
				OpcodeShape {
					const_in: &[],
					n_in,
					n_out,
					n_aux: 0,
					n_scratch: 0,
					n_imm: 1,
				}
			}
			_ => self.opcode.shape(&self.dimensions),
		}
	}

	/// Ensures the gate has the right shape.
	pub fn validate_shape(&self, registry: &HintRegistry) {
		let shape = self.shape(registry);
		let expected_wires =
			shape.const_in.len() + shape.n_in + shape.n_out + shape.n_aux + shape.n_scratch;
		assert_eq!(self.wires.len(), expected_wires);
		assert_eq!(self.immediates.len(), shape.n_imm);
	}
}

/// Gate graph replaces the current Shared struct
pub struct GateGraph {
	// Primary maps
	pub gates: PrimaryMap<Gate, GateData>,
	pub wires: PrimaryMap<Wire, WireKind>,

	pub path_spec_tree: PathSpecTree,
	pub gate_origin: SecondaryMap<Gate, PathSpec>,
	pub assertion_names: SecondaryMap<Gate, PathSpec>,

	/// Interning table mapping each constant value to the wire that holds it.
	///
	/// Keyed by a 64-bit word, so a fast integer hasher beats the default SipHash here.
	pub const_pool: FxHashMap<Word, Wire>,

	/// The all-one constant wire, seeded as the first constant at construction.
	///
	/// - The linear-to-AND lowering and gate fusion both need an all-one word.
	/// - Seeding it before any other wire makes it Wire 0 and the first constant.
	/// - So it lands at constant index 0 of the value vector, with no later fix-up.
	pub all_one: Wire,

	// Use-def analysis
	/// Maps each wire to the gate that defines it (if any)
	pub wire_def: SecondaryMap<Wire, Option<Gate>>,
	/// The gates that read each wire, as one flat edge list.
	///
	/// A wire's readers occupy the run `use_edges[use_offsets[w]..use_offsets[w + 1]]`.
	/// One allocation holds every edge, rather than one container per wire.
	use_edges: Vec<Gate>,
	/// Start of each wire's run in [`Self::use_edges`], with a final entry for the total.
	///
	/// Empty until the index is built, which reads as no wire having any reader.
	use_offsets: Vec<u32>,
}

impl GateGraph {
	pub fn new() -> Self {
		let path_spec_tree = PathSpecTree::new();
		let root = path_spec_tree.root();
		let mut graph = Self {
			gates: PrimaryMap::new(),
			wires: PrimaryMap::new(),
			path_spec_tree,
			gate_origin: SecondaryMap::with_default(root),
			assertion_names: SecondaryMap::with_default(root),
			const_pool: FxHashMap::default(),
			// Placeholder; overwritten by the seeding call below before any other wire exists.
			all_one: Wire::from_u32(0),
			wire_def: SecondaryMap::new(),
			use_edges: Vec::new(),
			use_offsets: Vec::new(),
		};
		// Seed the all-one constant before anything else is added.
		// This makes it Wire 0 and the first constant in the value vector's constants segment.
		graph.all_one = graph.add_constant(Word::ALL_ONE);
		graph
	}

	/// Runs a validation pass ensuring all the invariants hold.
	pub fn validate(&self, hint_registry: &HintRegistry) {
		// Every gate holds shape.
		for gate in self.gates.values() {
			gate.validate_shape(hint_registry);
		}
	}

	pub fn add_inout(&mut self) -> Wire {
		self.wires.push(WireKind::Inout)
	}

	pub fn add_witness(&mut self) -> Wire {
		self.wires.push(WireKind::Witness)
	}

	pub fn add_internal(&mut self) -> Wire {
		self.wires.push(WireKind::Internal)
	}

	pub fn add_scratch(&mut self) -> Wire {
		self.wires.push(WireKind::Scratch)
	}

	/// Returns the wire holding the given constant, creating it on first use.
	pub fn add_constant(&mut self, word: Word) -> Wire {
		// One hash whether or not a wire already holds the word.
		match self.const_pool.entry(word) {
			Entry::Occupied(entry) => *entry.get(),
			Entry::Vacant(entry) => *entry.insert(self.wires.push(WireKind::Constant(word))),
		}
	}

	/// Emits a gate with the given opcode, inputs and outputs.
	pub fn emit_gate(
		&mut self,
		gate_origin: PathSpec,
		opcode: Opcode,
		inputs: impl IntoIterator<Item = Wire>,
		outputs: impl IntoIterator<Item = Wire>,
	) -> Gate {
		self.emit_gate_generic(gate_origin, opcode, inputs, outputs, &[], &[])
	}

	/// Creates a gate inline with the given opcode's shape parametrized with the inputs, outputs
	/// and immediates.
	///
	/// Panics if the resulting opcode shape is not valid.
	pub fn emit_gate_generic(
		&mut self,
		gate_origin: PathSpec,
		opcode: Opcode,
		inputs: impl IntoIterator<Item = Wire>,
		outputs: impl IntoIterator<Item = Wire>,
		dimensions: &[usize],
		immediates: &[u32],
	) -> Gate {
		// Hint gates go through `emit_hint_gate`, which knows the hint's shape from the
		// `Hint` impl directly without needing the registry here.
		assert!(
			opcode != Opcode::Hint,
			"emit_gate_generic does not handle Opcode::Hint; use emit_hint_gate"
		);
		let shape = opcode.shape(dimensions);
		let mut wires: SmallVec<[Wire; 5]> = SmallVec::with_capacity(
			shape.const_in.len() + shape.n_in + shape.n_out + shape.n_aux + shape.n_scratch,
		);
		for c in shape.const_in {
			wires.push(self.add_constant(*c));
		}
		wires.extend(inputs);
		wires.extend(outputs);
		for _ in 0..shape.n_aux {
			// We create internal wires as auxiliary.
			wires.push(self.add_internal());
		}
		for _ in 0..shape.n_scratch {
			wires.push(self.add_scratch());
		}
		let data = GateData {
			opcode,
			wires,
			dimensions: dimensions.to_vec(),
			immediates: SmallVec::from_slice(immediates),
		};
		// Inline validate_shape: non-hint shape doesn't need a registry.
		let expected_wires =
			shape.const_in.len() + shape.n_in + shape.n_out + shape.n_aux + shape.n_scratch;
		assert_eq!(data.wires.len(), expected_wires);
		assert_eq!(data.immediates.len(), shape.n_imm);

		let gate = self.gates.push(data);

		self.gate_origin[gate] = gate_origin;

		gate
	}

	/// Emit a generic [`Opcode::Hint`] gate. Caller has already validated input arity
	/// against the hint's [`Hint::shape`](crate::ir::hints::Hint::shape) and allocated
	/// `n_out` output wires.
	pub fn emit_hint_gate(
		&mut self,
		gate_origin: PathSpec,
		hint_id: HintId,
		dimensions: &[usize],
		inputs: impl IntoIterator<Item = Wire>,
		outputs: impl IntoIterator<Item = Wire>,
	) -> Gate {
		let mut wires: SmallVec<[Wire; 5]> = SmallVec::new();
		wires.extend(inputs);
		wires.extend(outputs);
		let data = GateData {
			opcode: Opcode::Hint,
			wires,
			dimensions: dimensions.to_vec(),
			immediates: smallvec![hint_id],
		};
		let gate = self.gates.push(data);
		self.gate_origin[gate] = gate_origin;
		gate
	}

	/// Rebuilds the record of which gate defines each wire.
	///
	/// A gate defines its outputs and its auxiliary wires.
	///
	/// `hint_registry` must contain the hint of every [`Opcode::Hint`] gate.
	pub fn rebuild_wire_defs(&mut self, hint_registry: &HintRegistry) {
		self.wire_def.clear();
		for (gate, data) in self.gates.iter() {
			let param = data.gate_param_with_registry(hint_registry);
			for &wire in param.outputs.iter().chain(param.aux) {
				self.wire_def[wire] = Some(gate);
			}
		}
	}

	/// Rebuilds the record of which gates read each wire.
	///
	/// A gate reads its constant and input wires.
	///
	/// # Algorithm
	///
	/// A counting sort over wires, so the whole index is three linear passes and two allocations:
	///
	/// ```text
	///   pass 1: count readers per wire      -> [0, 2, 1, 0, 3, ...]
	///   pass 2: prefix-sum into run starts  -> [0, 0, 2, 3, 3, 6, ...]
	///   pass 3: place each reader in its run
	/// ```
	///
	/// Gates are visited in order in pass 3, so every run comes out sorted by gate.
	/// A gate that reads one wire more than once is recorded against it only once.
	///
	/// The result describes the graph as it stands. Rewiring a gate afterwards leaves it stale,
	/// so a pass that mutates and then re-reads has to rebuild.
	pub fn rebuild_wire_uses(&mut self, hint_registry: &HintRegistry) {
		let n_wires = self.wires.len();

		// Marks the last gate counted against a wire, so a repeated read is counted once.
		let mut last_seen: SecondaryMap<Wire, Option<Gate>> = SecondaryMap::new();

		// Pass 1: count each wire's readers, one slot to the right so the counts can be
		// prefix-summed in place into run starts.
		let mut offsets = vec![0u32; n_wires + 1];
		for (gate, data) in self.gates.iter() {
			let param = data.gate_param_with_registry(hint_registry);
			for &wire in param.constants.iter().chain(param.inputs) {
				if last_seen[wire] != Some(gate) {
					last_seen[wire] = Some(gate);
					offsets[wire.index() + 1] += 1;
				}
			}
		}

		// Pass 2: accumulate the counts into the start offset of each wire's run.
		for i in 0..n_wires {
			offsets[i + 1] += offsets[i];
		}

		// Pass 3: fill the runs, advancing a cursor per wire.
		let mut edges = vec![Gate::from_u32(0); offsets[n_wires] as usize];
		let mut cursor = offsets.clone();
		last_seen.clear();
		for (gate, data) in self.gates.iter() {
			let param = data.gate_param_with_registry(hint_registry);
			for &wire in param.constants.iter().chain(param.inputs) {
				if last_seen[wire] != Some(gate) {
					last_seen[wire] = Some(gate);
					edges[cursor[wire.index()] as usize] = gate;
					cursor[wire.index()] += 1;
				}
			}
		}

		self.use_edges = edges;
		self.use_offsets = offsets;
	}

	/// Rebuilds both halves of the use-def analysis.
	///
	/// `hint_registry` must contain the hint of every [`Opcode::Hint`] gate.
	pub fn rebuild_use_def_chains(&mut self, hint_registry: &HintRegistry) {
		self.rebuild_wire_defs(hint_registry);
		self.rebuild_wire_uses(hint_registry);
	}

	/// Returns the gates that read the given wire, in increasing gate order.
	///
	/// Yields nothing for a wire created after the index was last built.
	pub fn get_wire_uses(&self, wire: Wire) -> impl Iterator<Item = Gate> + '_ {
		// A missing entry means the index predates this wire, or was never built.
		let run = match self.use_offsets.get(wire.index() + 1) {
			Some(&end) => self.use_offsets[wire.index()] as usize..end as usize,
			None => 0..0,
		};
		self.use_edges[run].iter().copied()
	}

	/// Returns an iterator over all constant wires and their kind
	pub fn iter_const_wires(&self) -> impl Iterator<Item = (Wire, &WireKind)> {
		self.wires.iter().filter(|(_, kind)| kind.is_const())
	}

	/// Gets the kind of the given wire
	pub fn wire_kind(&self, wire: Wire) -> WireKind {
		self.wires[wire]
	}

	/// Gets gate data by reference
	pub fn gate_data(&self, gate: Gate) -> &GateData {
		&self.gates[gate]
	}

	/// Replaces all occurrences of a wire in a gate with another wire
	/// Returns the number of wire slots rewritten.
	pub fn replace_gate_wire(&mut self, gate: Gate, old_wire: Wire, new_wire: Wire) -> usize {
		let gate_data = &mut self.gates[gate];
		let mut rewritten = 0;
		for wire in &mut gate_data.wires {
			if *wire == old_wire {
				*wire = new_wire;
				rewritten += 1;
			}
		}
		rewritten
	}

	/// Replaces every use of `old_wire` with a constant wire holding `value`.
	pub fn replace_wire_with_constant(&mut self, old_wire: Wire, value: Word) -> WireReplacement {
		let const_wire = self.add_constant(value);
		self.replace_wire_with_wire(old_wire, const_wire)
	}

	/// Replaces every use of `old_wire` with `new_wire`.
	///
	/// The defining gate keeps `old_wire` as its output, so it is left unread and the dead-code
	/// pass is what actually removes it.
	///
	/// The use-def index is not updated, and forwarding to an ordinary wire grows that wire's set
	/// of readers. A caller that forwards again has to rebuild the index in between, or it will
	/// miss the readers this call moved.
	pub fn replace_wire_with_wire(&mut self, old_wire: Wire, new_wire: Wire) -> WireReplacement {
		if new_wire == old_wire {
			return WireReplacement {
				n_slots_rewritten: 0,
				affected_gates: Vec::new(),
			};
		}

		// Collected up front: the index borrows the graph, which is about to be rewritten.
		let affected_gates: Vec<Gate> = self.get_wire_uses(old_wire).collect();

		// The count comes from the rewrite itself, so it covers every slot the rewrite touched.
		// Tallying inputs and outputs separately would miss a wire held only in an aux slot.
		let n_slots_rewritten = affected_gates
			.iter()
			.map(|&gate| self.replace_gate_wire(gate, old_wire, new_wire))
			.sum();

		WireReplacement {
			n_slots_rewritten,
			affected_gates,
		}
	}

	/// Returns every gate in the graph, in construction order.
	pub fn gates(&self) -> impl Iterator<Item = Gate> + '_ {
		self.gates.iter().map(|(gate, _)| gate)
	}
}

/// What replacing one wire with another changed.
pub struct WireReplacement {
	/// How many wire slots were rewritten, across every affected gate.
	pub n_slots_rewritten: usize,
	/// The gates whose wires were rewritten.
	pub affected_gates: Vec<Gate>,
}

impl Default for GateGraph {
	fn default() -> Self {
		Self::new()
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::gates::opcode::Opcode;

	// Test helper functions
	fn get_wire_def(graph: &GateGraph, wire: Wire) -> Option<Gate> {
		graph.wire_def[wire]
	}

	fn wire_use_count(graph: &GateGraph, wire: Wire) -> usize {
		graph.get_wire_uses(wire).count()
	}

	fn is_wire_single_use(graph: &GateGraph, wire: Wire) -> bool {
		wire_use_count(graph, wire) == 1
	}

	fn get_wire_single_use(graph: &GateGraph, wire: Wire) -> Option<Gate> {
		let mut uses = graph.get_wire_uses(wire);
		match (uses.next(), uses.next()) {
			(Some(only), None) => Some(only),
			_ => None,
		}
	}

	fn get_gate_inputs(graph: &GateGraph, gate: Gate) -> Vec<Wire> {
		let gate_data = &graph.gates[gate];
		let gate_param = gate_data.gate_param();

		let mut inputs = Vec::new();
		inputs.extend_from_slice(gate_param.constants);
		inputs.extend_from_slice(gate_param.inputs);
		inputs
	}

	fn get_gate_outputs(graph: &GateGraph, gate: Gate) -> Vec<Wire> {
		let gate_data = &graph.gates[gate];
		let gate_param = gate_data.gate_param();

		let mut outputs = Vec::new();
		outputs.extend_from_slice(gate_param.outputs);
		outputs
	}

	#[test]
	fn test_use_def_analysis() {
		let mut graph = GateGraph::new();
		let root = graph.path_spec_tree.root();

		// Create some wires
		let in1 = graph.add_inout();
		let in2 = graph.add_inout();
		let out1 = graph.add_witness();
		let out2 = graph.add_witness();

		// Create a gate that uses in1 and in2, produces out1
		let gate1 = graph.emit_gate(root, Opcode::Bxor, vec![in1, in2], vec![out1]);

		// Create another gate that uses out1 and in1, produces out2
		let gate2 = graph.emit_gate(root, Opcode::Band, vec![out1, in1], vec![out2]);

		// Build use-def chains
		graph.rebuild_use_def_chains(&HintRegistry::new());

		// Check that gate1 defines out1
		assert_eq!(get_wire_def(&graph, out1), Some(gate1));

		// Check that gate2 defines out2
		assert_eq!(get_wire_def(&graph, out2), Some(gate2));

		// Check that in1 and in2 are used by gate1
		assert!(graph.get_wire_uses(in1).any(|g| g == gate1));
		assert!(graph.get_wire_uses(in2).any(|g| g == gate1));

		// Check that out1 is used by gate2
		assert!(graph.get_wire_uses(out1).any(|g| g == gate2));

		// Check wire use counts
		assert_eq!(wire_use_count(&graph, in1), 2); // Used by gate1 and gate2
		assert_eq!(wire_use_count(&graph, in2), 1);
		assert_eq!(wire_use_count(&graph, out1), 1);
		assert_eq!(wire_use_count(&graph, out2), 0);

		// Check single use queries
		assert!(!is_wire_single_use(&graph, in1)); // Used twice
		assert!(is_wire_single_use(&graph, in2));
		assert!(is_wire_single_use(&graph, out1));
		assert!(!is_wire_single_use(&graph, out2)); // No uses

		// Check get_wire_single_use
		assert_eq!(get_wire_single_use(&graph, in1), None); // Used twice
		assert_eq!(get_wire_single_use(&graph, out1), Some(gate2));
		assert_eq!(get_wire_single_use(&graph, out2), None); // No uses
	}

	#[test]
	fn wire_uses_are_ordered_and_deduplicated() {
		// Invariant: readers come back in increasing gate order, and a gate that reads one wire
		// twice is reported against it once. Constant propagation seeds its worklist from this
		// order, so it is what keeps that pass deterministic.
		//
		// Fixture: one wire read by three gates, the middle one reading it twice.
		//
		//   g0: x & y      -> reads x once
		//   g1: x ^ x      -> reads x twice, counts once
		//   g2: x & y      -> reads x once
		let mut graph = GateGraph::new();
		let root = graph.path_spec_tree.root();

		let x = graph.add_inout();
		let y = graph.add_inout();

		let o0 = graph.add_internal();
		let g0 = graph.emit_gate(root, Opcode::Band, vec![x, y], vec![o0]);
		let o1 = graph.add_internal();
		let g1 = graph.emit_gate(root, Opcode::Bxor, vec![x, x], vec![o1]);
		let o2 = graph.add_internal();
		let g2 = graph.emit_gate(root, Opcode::Band, vec![x, y], vec![o2]);

		graph.rebuild_use_def_chains(&HintRegistry::new());

		// Sorted, and the doubled read contributes a single entry.
		let readers: Vec<Gate> = graph.get_wire_uses(x).collect();
		assert_eq!(readers, vec![g0, g1, g2]);

		// A wire only one gate reads.
		assert_eq!(graph.get_wire_uses(o0).collect::<Vec<_>>(), Vec::new());
		assert_eq!(graph.get_wire_uses(y).collect::<Vec<_>>(), vec![g0, g2]);
	}

	#[test]
	fn wire_uses_are_empty_for_a_wire_added_after_the_rebuild() {
		// Invariant: the index describes the graph as it stood when built. A wire created later
		// has no run in it, and reads as having no readers rather than panicking.
		let mut graph = GateGraph::new();
		graph.rebuild_use_def_chains(&HintRegistry::new());

		let late = graph.add_inout();
		assert_eq!(graph.get_wire_uses(late).count(), 0);
	}

	#[test]
	fn test_constant_use_def() {
		let mut graph = GateGraph::new();
		let root = graph.path_spec_tree.root();

		// Create a constant wire
		let const_wire = graph.add_constant(Word(42u64));
		let in_wire = graph.add_inout();
		let out = graph.add_witness();

		// Create a gate that uses the constant and input wire
		let gate = graph.emit_gate(root, Opcode::Bxor, vec![const_wire, in_wire], vec![out]);

		// Build use-def chains
		graph.rebuild_use_def_chains(&HintRegistry::new());

		// Constants are not defined by gates
		assert_eq!(get_wire_def(&graph, const_wire), None);

		// But they should be tracked as used
		assert!(graph.get_wire_uses(const_wire).any(|g| g == gate));
		assert_eq!(wire_use_count(&graph, const_wire), 1);
	}

	#[test]
	fn test_rebuild_use_def_chains() {
		let mut graph = GateGraph::new();
		let root = graph.path_spec_tree.root();

		// Create wires and gates
		let in1 = graph.add_inout();
		let in2 = graph.add_inout();
		let out = graph.add_witness();

		graph.emit_gate(root, Opcode::Bxor, vec![in1, in2], vec![out]);

		// Clear the analysis by hand, standing in for a pass that rewired the graph.
		graph.wire_def.clear();
		graph.use_edges.clear();
		graph.use_offsets.clear();

		// Verify it's cleared
		assert_eq!(get_wire_def(&graph, out), None);
		assert!(graph.get_wire_uses(in1).next().is_none());

		// Rebuild
		graph.rebuild_use_def_chains(&HintRegistry::new());

		// Verify it's restored
		assert!(get_wire_def(&graph, out).is_some());
		assert!(!graph.get_wire_uses(in1).next().is_none());
		assert!(!graph.get_wire_uses(in2).next().is_none());
	}

	#[test]
	fn test_gate_inputs_outputs() {
		let mut graph = GateGraph::new();
		let root = graph.path_spec_tree.root();

		let a = graph.add_inout();
		let b = graph.add_inout();
		let bin = graph.add_inout();
		let diff = graph.add_witness();
		let bout = graph.add_witness();

		// IsubBinBout declares one constant input (ALL_ONE) alongside its three regular inputs.
		let gate = graph.emit_gate(root, Opcode::IsubBinBout, vec![a, b, bin], vec![diff, bout]);

		// No need to rebuild use-def chains for this test
		// as we're just checking the gate structure

		let inputs = get_gate_inputs(&graph, gate);
		// 1 constant input (ALL_ONE) + 3 regular inputs.
		assert_eq!(inputs.len(), 4);
		assert!(inputs.contains(&a));
		assert!(inputs.contains(&b));
		assert!(inputs.contains(&bin));
		// The constant wire is surfaced first.
		let const_wire = inputs[0];
		match graph.wires[const_wire] {
			WireKind::Constant(word) => assert_eq!(word, Word::ALL_ONE),
			_ => panic!("Expected constant wire"),
		}

		let outputs = get_gate_outputs(&graph, gate);
		assert_eq!(outputs.len(), 2);
		assert!(outputs.contains(&diff));
		assert!(outputs.contains(&bout));
	}
}
