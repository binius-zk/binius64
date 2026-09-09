// Copyright 2026 The Binius Developers
//! Zero propagation over the gate graph.

use std::collections::VecDeque;

use binius_core::word::Word;
use cranelift_entity::EntitySet;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::{
	gates::Opcode,
	ir::{Gate, GateBody, GateGraph, Wire, hints::HintRegistry},
};

/// Rewrites every gate that a zero operand turns into the identity.
///
/// Such a gate's result is already carried by another wire, so its readers are pointed at that
/// wire instead. The gate is then unread, and the dead-code pass drops it along with the
/// constraint it would have emitted.
///
/// Zeros arise from padding rather than from anything a circuit author writes by hand. A message
/// block zero-filled to its block size is the common case, and a hash that mixes every word of
/// its block pays a carry chain per zero word per round without this.
///
/// Returns the number of wire slots rewritten.
///
/// # Algorithm
///
/// A gate matches a rule only while it reads the zero wire.
/// So the worklist starts at that wire's readers, and grows by the gates a rewrite hands a zero.
///
/// The reader index is a snapshot, so the readers a rewrite moves are tracked alongside it.
pub fn zero_propagation(
	graph: &mut GateGraph,
	force_committed: &EntitySet<Wire>,
	hint_registry: &HintRegistry,
) -> usize {
	// Nothing to propagate unless the circuit already holds a zero constant. Looking it up
	// rather than interning it keeps a circuit that has none completely untouched.
	let Some(zero) = find_zero_constant(graph) else {
		return 0;
	};

	graph.rebuild_wire_uses(hint_registry);

	let mut queue: VecDeque<Gate> = graph.get_wire_uses(zero).collect();
	let mut queued: FxHashSet<Gate> = queue.iter().copied().collect();
	let mut readers = Readers::default();
	let mut total_rewritten = 0;

	while let Some(gate) = queue.pop_front() {
		queued.remove(&gate);
		let Some(forwardings) = zero_identity(graph, gate, zero, force_committed, hint_registry)
		else {
			continue;
		};

		for (from, to) in forwardings {
			if from == to {
				continue;
			}

			let moving = readers.take(graph, from);
			let mut landed = Vec::with_capacity(moving.len());
			for reader in moving {
				// Asked before the rewrite, after which every reader holds the target.
				let had_target = reads(graph, reader, to, hint_registry);
				total_rewritten += graph.replace_gate_wire(reader, from, to);
				if !had_target {
					landed.push(reader);
				}
				// A rewrite makes a gate newly match only by handing it a zero operand.
				if to == zero && queued.insert(reader) {
					queue.push_back(reader);
				}
			}
			readers.record(to, landed);
		}
	}

	total_rewritten
}

/// The gates reading each wire, past the point the snapshot index still describes them.
#[derive(Default)]
struct Readers {
	/// Wires whose snapshot run has already been handed out.
	drained: FxHashSet<Wire>,
	/// Readers a rewrite moved onto a wire.
	moved: FxHashMap<Wire, Vec<Gate>>,
}

impl Readers {
	/// Removes and returns every gate reading the wire.
	fn take(&mut self, graph: &GateGraph, wire: Wire) -> Vec<Gate> {
		let mut readers: Vec<Gate> = if self.drained.insert(wire) {
			graph.get_wire_uses(wire).collect()
		} else {
			Vec::new()
		};
		readers.extend(self.moved.remove(&wire).into_iter().flatten());
		readers
	}

	/// Adds gates that now read the wire.
	fn record(&mut self, wire: Wire, gates: Vec<Gate>) {
		if !gates.is_empty() {
			self.moved.entry(wire).or_default().extend(gates);
		}
	}
}

/// Whether the gate takes the wire as a constant or input operand.
fn reads(graph: &GateGraph, gate: Gate, wire: Wire, hint_registry: &HintRegistry) -> bool {
	let param = graph.gate_data(gate).gate_param(hint_registry);
	param
		.constants
		.iter()
		.chain(param.inputs)
		.any(|w| *w == wire)
}

/// The zero constant wire, if the circuit already has one.
fn find_zero_constant(graph: &GateGraph) -> Option<Wire> {
	graph.const_pool.get(&Word::ZERO).copied()
}

/// Where each of `gate`'s outputs moves to, when a zero operand makes the gate an identity.
///
/// Returns nothing when no rule applies, so the gate keeps its constraint.
///
/// The rules, for `z` the zero word:
///
/// ```text
///     shift(z, n)   -> z          every variant fills with zeros or rotates them
///     x ^ z         -> x          the exclusive-or leaves every bit alone
///     x + z         -> x          and carries nothing, so the carry word is z
/// ```
///
/// A gate with one output pads the pair with a wire forwarded to itself, which rewrites nothing.
fn zero_identity(
	graph: &GateGraph,
	gate: Gate,
	zero: Wire,
	force_committed: &EntitySet<Wire>,
	hint_registry: &HintRegistry,
) -> Option<[(Wire, Wire); 2]> {
	let data = graph.gate_data(gate);
	let param = data.gate_param(hint_registry);

	// No rule below applies to a hint.
	let GateBody::Op(opcode) = data.body else {
		return None;
	};

	// A force-committed output has to stay a committed wire backed by its own constraint, so
	// leave the gate alone rather than strand it.
	if param
		.outputs
		.iter()
		.any(|wire| force_committed.contains(*wire))
	{
		return None;
	}

	// The padding pair, which the forwarding loop skips.
	let nothing = (zero, zero);

	match (opcode, param.inputs, param.outputs) {
		// A shift of zero is zero, for every variant and amount.
		(Opcode::Shift, [x], [z]) if *x == zero => Some([(*z, zero), nothing]),

		// Exclusive-or with zero is the other operand.
		(Opcode::Bxor, [a, b], [z]) if *b == zero => Some([(*z, *a), nothing]),
		(Opcode::Bxor, [a, b], [z]) if *a == zero => Some([(*z, *b), nothing]),

		// Adding zero leaves each half alone and produces no carry in either half.
		(Opcode::Iadd32, [a, b], [sum, cout]) if *b == zero => Some([(*sum, *a), (*cout, zero)]),
		(Opcode::Iadd32, [a, b], [sum, cout]) if *a == zero => Some([(*sum, *b), (*cout, zero)]),

		_ => None,
	}
}

#[cfg(test)]
mod tests {
	use binius_core::constraint_system::ShiftVariant;
	use proptest::prelude::*;

	use super::*;
	use crate::{Options, builder::CircuitBuilder};

	/// Builds with only zero propagation and dead-code elimination active, so the AND count
	/// reflects the pass under test rather than the rest of the pipeline.
	fn opts_with_zero_fold(enable: bool) -> Options {
		Options {
			enable_zero_propagation: enable,
			..Options::default()
		}
	}

	#[test]
	fn zero_addend_costs_no_carry_chain() {
		// Invariant: a zero addend leaves the sum alone, so the carry chain is dead weight.
		// The zero reaches the addition only after a shift and an exclusive-or, which is the
		// shape lane packing produces, so all three rules have to fire in turn.
		let count_and = |enable| {
			let builder = CircuitBuilder::with_opts(opts_with_zero_fold(enable));
			let x = builder.add_inout();
			let zero = builder.add_constant(Word::ZERO);

			// zero --shift--> zero --exclusive-or--> zero, the packed form of a padding word.
			let packed = builder.bxor(builder.shl(zero, 32), zero);
			let sum = builder.iadd_32(x, packed);

			let out = builder.add_inout();
			builder.assert_eq("sum", sum, out);
			let circuit = builder.build();
			crate::CircuitStat::collect(&circuit).n_and_constraints
		};

		// Folded, only the equality assertion remains, and that is a ZERO constraint; unfolded, the
		// carry chain is paid in AND constraints.
		assert!(count_and(true) < count_and(false));
		assert_eq!(count_and(true), 0);
	}

	#[test]
	fn folded_circuit_still_computes_the_sum() {
		// Invariant: the pass rewrites which wire carries a value, never the value itself.
		let builder = CircuitBuilder::new();
		let x = builder.add_inout();
		let y = builder.add_inout();
		let zero = builder.add_constant(Word::ZERO);

		// Two additions, one foldable and one not, so the folded path has to agree with a
		// carry chain that really runs.
		let folded = builder.iadd_32(x, zero);
		let real = builder.iadd_32(folded, y);
		let out = builder.add_inout();
		builder.assert_eq("sum", real, out);

		let circuit = builder.build();
		let mut w = circuit.new_witness_filler();
		// Low halves overflow and high halves do not, so a carry crossing bit 32 would show.
		w[x] = Word(0x0000_0002_FFFF_FFFF);
		w[y] = Word(0x0000_0003_0000_0001);
		w[out] = Word(0x0000_0005_0000_0000);
		circuit.populate_wire_witness(&mut w).unwrap();
	}

	#[test]
	fn a_circuit_without_zero_is_untouched() {
		// Invariant: the pass interns nothing, so a circuit holding no zero constant keeps
		// exactly the constants it declared.
		let builder = CircuitBuilder::new();
		let x = builder.add_inout();
		let out = builder.add_inout();
		builder.assert_eq("shifted", builder.shl(x, 8), out);
		let circuit = builder.build();

		// Only the all-one word the gate graph seeds, plus the one the assertion uses.
		let constants = &circuit.constraint_system().constants;
		assert!(!constants.contains(&Word::ZERO));
	}

	fn zero_propagation_by_sweeps(
		graph: &mut GateGraph,
		force_committed: &EntitySet<Wire>,
		hint_registry: &HintRegistry,
	) -> usize {
		let Some(zero) = find_zero_constant(graph) else {
			return 0;
		};

		let mut total_rewritten = 0;
		loop {
			graph.rebuild_wire_uses(hint_registry);

			let mut forwardings = Vec::new();
			for gate in graph.gates() {
				if let Some(pairs) =
					zero_identity(graph, gate, zero, force_committed, hint_registry)
				{
					forwardings.extend(pairs);
				}
			}

			let mut rewritten = 0;
			for (from, to) in forwardings {
				rewritten += graph.replace_wire_with_wire(from, to).n_slots_rewritten;
			}

			if rewritten == 0 {
				return total_rewritten;
			}
			total_rewritten += rewritten;
		}
	}

	/// One gate of a generated fixture, naming its operands by position in the wire pool.
	#[derive(Clone, Copy, Debug)]
	enum Op {
		Shift(usize, u32),
		Bxor(usize, usize),
		Iadd32(usize, usize),
		Band(usize, usize),
	}

	/// Builds the fixture, returning the graph and the wire pool the operand positions index.
	fn build_fixture(ops: &[Op]) -> (GateGraph, Vec<Wire>) {
		let mut graph = GateGraph::new();
		let root = graph.path_spec_tree.root();

		// Position 0 is the zero constant, position 1 an ordinary input.
		let mut pool = vec![graph.add_constant(Word::ZERO), graph.add_inout()];

		for &op in ops {
			let n = pool.len();
			match op {
				Op::Shift(a, amount) => {
					let out = graph.add_internal();
					graph.emit_gate_generic(
						root,
						Opcode::Shift,
						vec![pool[a % n]],
						vec![out],
						&[],
						&[ShiftVariant::Sll as u32, amount % 64],
					);
					pool.push(out);
				}
				Op::Bxor(a, b) => {
					let out = graph.add_internal();
					graph.emit_gate(root, Opcode::Bxor, vec![pool[a % n], pool[b % n]], vec![out]);
					pool.push(out);
				}
				Op::Iadd32(a, b) => {
					let sum = graph.add_internal();
					let cout = graph.add_internal();
					graph.emit_gate(
						root,
						Opcode::Iadd32,
						vec![pool[a % n], pool[b % n]],
						vec![sum, cout],
					);
					pool.push(sum);
					pool.push(cout);
				}
				Op::Band(a, b) => {
					let out = graph.add_internal();
					graph.emit_gate(root, Opcode::Band, vec![pool[a % n], pool[b % n]], vec![out]);
					pool.push(out);
				}
			}
		}

		// An assertion per wire, so every forwarding has a reader to move.
		for &wire in &pool {
			graph.emit_gate(root, Opcode::AssertZero, vec![wire], vec![]);
		}

		(graph, pool)
	}

	/// The pinned set the fixture's operand positions name.
	fn pin_wires(pool: &[Wire], pins: &[usize]) -> EntitySet<Wire> {
		let mut pinned = EntitySet::new();
		for &pin in pins {
			pinned.insert(pool[pin % pool.len()]);
		}
		pinned
	}

	/// Every gate's wire slots, which is the whole of what the pass rewrites.
	fn wiring(graph: &GateGraph) -> Vec<Vec<Wire>> {
		graph
			.gates
			.values()
			.map(|data| data.wires.to_vec())
			.collect()
	}

	/// A cone whose every level turns zero only once the level below it has.
	fn deep_zero_cone(depth: usize) -> Vec<Op> {
		let mut ops = Vec::new();
		let mut zero_at = 0;
		let mut next = 2;
		for _ in 0..depth {
			// The shift and the exclusive-or each hand their level's zero one step further on.
			ops.push(Op::Shift(zero_at, 3));
			let shifted = next;
			ops.push(Op::Bxor(shifted, 1));
			let folded = next + 1;
			// The carry-out is the next level's zero, and it is zero only once both above fired.
			ops.push(Op::Iadd32(folded, shifted));
			zero_at = next + 3;
			next += 4;
		}
		ops
	}

	#[test]
	fn a_deep_zero_cone_reaches_the_sweep_fixpoint() {
		let registry = HintRegistry::new();
		let ops = deep_zero_cone(16);

		let (mut swept, _) = build_fixture(&ops);
		zero_propagation_by_sweeps(&mut swept, &EntitySet::new(), &registry);

		let (mut worked, _) = build_fixture(&ops);
		zero_propagation(&mut worked, &EntitySet::new(), &registry);

		assert_eq!(wiring(&swept), wiring(&worked));

		// The cone really does collapse, so the comparison above is not between two untouched
		// graphs.
		let (untouched, _) = build_fixture(&ops);
		assert_ne!(wiring(&untouched), wiring(&worked));
	}

	fn any_op() -> impl Strategy<Value = Op> {
		prop_oneof![
			(0usize..64, 0u32..64).prop_map(|(a, n)| Op::Shift(a, n)),
			(0usize..64, 0usize..64).prop_map(|(a, b)| Op::Bxor(a, b)),
			(0usize..64, 0usize..64).prop_map(|(a, b)| Op::Iadd32(a, b)),
			(0usize..64, 0usize..64).prop_map(|(a, b)| Op::Band(a, b)),
		]
	}

	proptest! {
		#[test]
		fn the_worklist_reaches_the_sweep_fixpoint(
			ops in prop::collection::vec(any_op(), 1..48),
			pins in prop::collection::vec(0usize..64, 0..4),
		) {
			let registry = HintRegistry::new();

			let (mut swept, pool) = build_fixture(&ops);
			let pinned = pin_wires(&pool, &pins);
			zero_propagation_by_sweeps(&mut swept, &pinned, &registry);

			let (mut worked, _) = build_fixture(&ops);
			zero_propagation(&mut worked, &pinned, &registry);

			// The two schedules land on the same graph, slot for slot.
			prop_assert_eq!(wiring(&swept), wiring(&worked));

			// And the worklist result is a sweep fixpoint in its own right.
			prop_assert_eq!(zero_propagation_by_sweeps(&mut worked, &pinned, &registry), 0);
		}
	}
}
