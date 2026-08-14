// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers
//! linear expression graph.

use binius_core::constraint_system::Shift;
use cranelift_entity::{EntityRef, EntitySet, PrimaryMap, SecondaryMap, entity_impl};
use petgraph::graph::{DiGraph, NodeIndex};

use crate::{
	ir::Wire,
	lower::{ConstraintBuilder, WireOperand},
};

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ConstraintRef {
	And { index: usize },
	Imul { index: usize },
	Bmul { index: usize },
	Zero { index: usize },
	Linear { index: usize },
}

/// Identifies a linear-definition node: a linear constraint that assigns a wire.
///
/// Its position matches the constraint's own position in the constraint builder's linear list.
/// So an id's index doubles as that list's index.
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LinDefId(u32);
entity_impl!(LinDefId);

/// Identifies a root node: a use site in a constraint that defines no wire.
///
/// A root is a sink of the graph.
/// A linear definition flows into it as an AND, IMUL, BMUL, or Zero constraint.
/// A root has no consumer of its own.
/// Inlining always terminates here.
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RootId(u32);
entity_impl!(RootId);

/// Identifies an opaque node: a wire that is not defined by a linear constraint.
///
/// Inputs, constants, and the outputs of non-linear operations are opaque.
/// Inlining cannot reach past one.
/// So an opaque node is a source of the graph.
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct OpaqueId(u32);
entity_impl!(OpaqueId);

/// Which of the three node kinds a graph node holds.
///
/// A traversal walks the graph through petgraph's own untyped node index.
/// This is the one place all three kinds still meet.
/// Everywhere else a node is named by its own specific kind of id.
/// So the compiler rejects handing one kind's id to code that expects another.
#[derive(Debug, Copy, Clone)]
pub(super) enum NodeKind {
	/// A linear definition, named by its id.
	LinDef(LinDefId),
	/// A root use.
	///
	/// Its constraint lives in a separate collection, addressed by discovery order.
	/// That is not by this node.
	/// So the variant itself carries no id.
	Root,
	/// A wire that is not defined by a linear constraint.
	Opaque,
}

/// Data associated with edges in the Linear Expression Graph.
///
/// Each edge represents a use of a wire (producer) by another constraint (consumer),
/// annotated with the shift operation applied to the producer value.
///
/// # Example
/// ```text
/// y = x << 5      // Edge from x to y has shift = Sll(5)
/// z = y ^ a       // Edge from y to z has shift = None
/// w = z >> 3      // Edge from z to w has shift = Srl(3)
/// ```
#[derive(Debug)]
pub struct EdgeData {
	/// The shift operation applied when the producer is used by the consumer.
	pub shift: Shift,
}

/// Linear Expression Graph (LeGraph) - the core data structure for gate fusion optimization.
///
/// This graph represents the data flow relationships between linear constraints (XOR and shift
/// operations) and their uses in non-linear constraints (AND/IMUL operations). The graph is used
/// to determine which linear constraints can be inlined into their consumers to reduce the total
/// number of AND constraints in the final circuit.
///
/// # Graph Structure
///
/// The graph consists of three types of nodes:
///
/// 1. **Linear Definition Nodes** (`LinDef`): Represent linear constraints that define a wire as an
///    XOR combination of shifted values. These are candidates for inlining.
///
/// 2. **Root Nodes**: Represent uses of linear definitions in non-linear constraints (AND/IMUL).
///    These are the sinks of the graph where inlining decisions terminate.
///
/// 3. **Opaque Nodes**: Represent wires that are not defined by linear constraints (e.g., inputs or
///    outputs of non-linear operations). These cannot be inlined.
///
/// Edges in the graph flow from producers to consumers, with each edge annotated with a shift
/// operation that describes how the producer value is transformed when used by the consumer.
///
/// # Example
///
/// ```text
/// // Circuit:
/// y = a ^ b        // Linear definition
/// z = y >> 5       // Linear definition using y
/// w = z & c        // Non-linear use of z
///
/// // Graph representation:
/// [a] ──┐
///       ├─> [y = a ^ b] ──srl(5)──> [z = y >> 5] ──none──> [AND root]
/// [b] ──┘
/// ```
///
/// In this example, both `y` and `z` can potentially be inlined into the AND constraint,
/// resulting in `w = ((a ^ b) >> 5) & c` without intermediate wire commitments.
pub struct LeGraph {
	pub pg: DiGraph<NodeKind, EdgeData>,
	/// Node holding each wire, for the wires that have one.
	///
	/// Wire identifiers are dense, so this indexes directly instead of hashing.
	pub wire_to_node: SecondaryMap<Wire, Option<NodeIndex>>,
	pub lin_def: EntitySet<Wire>,
	pub lin_committed: EntitySet<Wire>,
	/// The linear-definition id that assigns each wire, for wires that have one.
	wire_to_lin_def: SecondaryMap<Wire, Option<LinDefId>>,
	/// The wire each linear-definition id assigns.
	lin_defs: PrimaryMap<LinDefId, Wire>,
	/// The constraint each root id names, in the order the roots were discovered.
	pub roots: PrimaryMap<RootId, ConstraintRef>,
	/// The graph node each opaque id names, in the order the wires were discovered.
	pub opaque: PrimaryMap<OpaqueId, NodeIndex>,
}

impl LeGraph {
	/// Constructs a new Linear Expression Graph from the constraint builder.
	///
	/// This method analyzes all constraints in the builder and constructs a graph that captures
	/// the use-def relationships between linear and non-linear constraints.
	///
	/// # Process
	///
	/// 1. Identifies all linear constraint definitions
	/// 2. Tracks uses of linear definitions in other linear constraints
	/// 3. Identifies "root" uses where linear definitions flow into non-linear constraints
	/// 4. Builds edges with appropriate shift annotations
	pub fn new(cb: &ConstraintBuilder) -> Self {
		let mut leg = Self {
			pg: DiGraph::new(),
			wire_to_node: SecondaryMap::new(),
			lin_committed: EntitySet::new(),
			lin_def: EntitySet::new(),
			wire_to_lin_def: SecondaryMap::new(),
			lin_defs: PrimaryMap::new(),
			roots: PrimaryMap::new(),
			opaque: PrimaryMap::new(),
		};
		build_use_def(cb, &mut leg);
		leg
	}

	/// Classifies a graph node as a linear definition, a root, or an opaque wire.
	pub(super) fn node_kind(&self, node: NodeIndex) -> NodeKind {
		self.pg[node]
	}

	/// Returns the set of wires that must be committed (converted to AND constraints).
	///
	/// This is populated only after running the commit_set decision pass.
	pub const fn commit_set(&self) -> &EntitySet<Wire> {
		&self.lin_committed
	}

	/// The operand of the linear definition that assigns `wire`.
	///
	/// The operand lives in `cb`; this only resolves which constraint to read.
	///
	/// # Panics
	///
	/// Panics if `wire` is not assigned by a linear definition.
	pub fn lin_def_operand<'a>(&self, cb: &'a ConstraintBuilder, wire: Wire) -> &'a WireOperand {
		&cb.linear_constraints[self.lin_def_id(wire).index()].rhs
	}

	/// The id of the linear definition that assigns `wire`.
	///
	/// # Panics
	///
	/// Panics if `wire` is not assigned by a linear definition.
	fn lin_def_id(&self, wire: Wire) -> LinDefId {
		self.wire_to_lin_def[wire]
			.unwrap_or_else(|| panic!("{wire:?} is not assigned by a linear definition"))
	}

	/// Returns the node holding the given wire.
	///
	/// # Panics
	///
	/// Panics if the wire has no node.
	fn node_of(&self, wire: Wire) -> NodeIndex {
		self.wire_to_node[wire].unwrap_or_else(|| panic!("{wire:?} has no node in the graph"))
	}

	/// The wire a linear-definition id assigns.
	pub fn lin_dst(&self, id: LinDefId) -> Wire {
		self.lin_defs[id]
	}

	pub fn lin_def_constraint_ref(&self, wire: Wire) -> ConstraintRef {
		ConstraintRef::Linear {
			index: self.lin_def_id(wire).index(),
		}
	}

	/// Checks if a wire is defined by a linear constraint.
	///
	/// Returns `true` if the wire is the output of a linear constraint (XOR combination
	/// of shifted values), `false` if it's an opaque wire or not in the graph at all.
	pub fn is_lin_def(&self, wire: Wire) -> bool {
		self.lin_def.contains(wire)
	}

	/// Adds a linear-definition node to the graph for the constraint that assigns `dst`.
	///
	/// Returns the id the new node is addressed by.
	fn add_lin_def(&mut self, dst: Wire) -> LinDefId {
		let lin_def_id = self.lin_defs.push(dst);
		let lin_node = self.pg.add_node(NodeKind::LinDef(lin_def_id));
		let prev = self.wire_to_node[dst].replace(lin_node);
		assert!(prev.is_none(), "wire already has a node");
		self.wire_to_lin_def[dst] = Some(lin_def_id);
		self.lin_def.insert(dst);
		lin_def_id
	}

	/// Notes a use of a wire by a linear user.
	///
	/// `shift` is how much the producer is shifted by the consumer expression.
	///
	/// Note:
	///
	/// 1. directionality matters, the value flows from the producer into the consumer.
	/// 2. a single consumer possibly can refer the same producer multiple times. In that case there
	///    are going to be multiple edges.
	fn note_lin_use(&mut self, producer: Wire, shift: Shift, consumer: Wire) {
		let node_c = self.node_of(consumer);
		if self.is_lin_def(producer) {
			let node_p = self.node_of(producer);
			self.pg.add_edge(node_p, node_c, EdgeData { shift });
		} else {
			// This is a use of a wire that is not defined by a linear. That means it's opaque!
			let opaque_node = match self.wire_to_node[producer] {
				Some(node) => node,
				None => {
					let node = self.pg.add_node(NodeKind::Opaque);
					self.opaque.push(node);
					self.wire_to_node[producer] = Some(node);
					node
				}
			};
			self.pg.add_edge(opaque_node, node_c, EdgeData { shift });
		}
	}

	/// Notes a use of a wire of a linear producer by a user that defines no wire.
	fn note_root_use(&mut self, producer: Wire, shift: Shift, constraint: ConstraintRef) {
		let node_p = self.node_of(producer);
		self.roots.push(constraint);
		let root_node = self.pg.add_node(NodeKind::Root);
		self.pg.add_edge(node_p, root_node, EdgeData { shift });
	}
}

fn build_use_def(cb: &ConstraintBuilder, leg: &mut LeGraph) {
	// Collect defs from linear constraints.
	//
	// Linear constraints are simple definitions. We assert that this is the case here.
	// In future we should actually define `linear_constraints`.
	for lin in &cb.linear_constraints {
		leg.add_lin_def(lin.dst);
	}

	for lin in &cb.linear_constraints {
		let consumer = lin.dst;
		for term in &lin.rhs {
			leg.note_lin_use(term.wire, term.sole_shift(), consumer);
		}
	}

	for (index, and) in cb.and_constraints.iter().enumerate() {
		harvest_root_uses(&and.a, leg, ConstraintRef::And { index });
		harvest_root_uses(&and.b, leg, ConstraintRef::And { index });
		harvest_root_uses(&and.c, leg, ConstraintRef::And { index });
	}

	for (index, mul) in cb.imul_constraints.iter().enumerate() {
		harvest_root_uses(&mul.a, leg, ConstraintRef::Imul { index });
		harvest_root_uses(&mul.b, leg, ConstraintRef::Imul { index });
		harvest_root_uses(&mul.hi, leg, ConstraintRef::Imul { index });
		harvest_root_uses(&mul.lo, leg, ConstraintRef::Imul { index });
	}

	for (index, mul) in cb.bmul_constraints.iter().enumerate() {
		harvest_root_uses(&mul.a_lo, leg, ConstraintRef::Bmul { index });
		harvest_root_uses(&mul.a_hi, leg, ConstraintRef::Bmul { index });
		harvest_root_uses(&mul.b_lo, leg, ConstraintRef::Bmul { index });
		harvest_root_uses(&mul.b_hi, leg, ConstraintRef::Bmul { index });
		harvest_root_uses(&mul.c_lo, leg, ConstraintRef::Bmul { index });
		harvest_root_uses(&mul.c_hi, leg, ConstraintRef::Bmul { index });
	}

	for (index, zero) in cb.zero_constraints.iter().enumerate() {
		harvest_root_uses(&zero.val, leg, ConstraintRef::Zero { index });
	}
}

fn harvest_root_uses(operand: &WireOperand, leg: &mut LeGraph, constraint: ConstraintRef) {
	for term in operand {
		if leg.is_lin_def(term.wire) {
			leg.note_root_use(term.wire, term.sole_shift(), constraint);
		}
	}
}
