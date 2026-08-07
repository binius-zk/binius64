// Copyright 2026 The Binius Developers
//! Zero propagation over the gate graph.

use binius_core::word::Word;
use cranelift_entity::EntitySet;

use crate::compiler::{
	gate::Opcode,
	gate_graph::{Gate, GateGraph, Wire},
	hints::HintRegistry,
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
/// Zeros travel: clearing a zero leaves zero, and packing zero into a lane leaves zero. So one
/// sweep is not enough, and the sweeps repeat until one changes nothing.
///
/// Each sweep rebuilds the reader index first. Forwarding to an ordinary wire grows that wire's
/// set of readers, which a stale index would not list, and a later sweep forwarding that same
/// wire would then miss those readers.
///
/// Only the reader half is rebuilt. This pass follows readers, never definitions.
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

	let mut total_rewritten = 0;
	loop {
		graph.rebuild_wire_uses(hint_registry);

		// Gathered before any rewriting, since reading the graph and mutating it cannot overlap.
		let mut forwardings = Vec::new();
		for gate in graph.gates() {
			if let Some(pairs) = zero_identity(graph, gate, zero, force_committed, hint_registry) {
				forwardings.extend(pairs);
			}
		}

		let mut rewritten = 0;
		for (from, to) in forwardings {
			rewritten += graph.replace_wire_with_wire(from, to).n_slots_rewritten;
		}

		// A sweep that moved no reader has reached the fixpoint.
		if rewritten == 0 {
			return total_rewritten;
		}
		total_rewritten += rewritten;
	}
}

/// The zero constant wire, if the circuit already has one.
fn find_zero_constant(graph: &GateGraph) -> Option<Wire> {
	graph
		.iter_const_wires()
		.find(|(_, kind)| kind.const_value() == Some(Word::ZERO))
		.map(|(wire, _)| wire)
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
	let param = data.gate_param_with_registry(hint_registry);
	let opcode = data.opcode;

	// A force-committed output has to stay a committed wire backed by its own constraint, so
	// leave the gate alone rather than strand it.
	if param
		.outputs
		.iter()
		.any(|wire| force_committed.contains(*wire))
	{
		return None;
	}

	// The padding pair, which `replace_wire_with_wire` recognises as a no-op.
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
	use super::*;
	use crate::compiler::{CircuitBuilder, Options};

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

		// Folded, only the equality assertion remains; unfolded, the carry chain is paid too.
		assert!(count_and(true) < count_and(false));
		assert_eq!(count_and(true), 1);
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
}
