// Copyright 2025 Irreducible Inc.
//! 64-bit unsigned integer addition without carry-in.
//!
//! # Wires
//!
//! - `a`, `b`: Input wires for the summands
//! - `sum`: Output wire containing the resulting sum = a + b
//! - `cout` (carry-out): Output wire containing a carry word where each bit position indicates
//!   whether a carry occurred at that position during the addition.
//!
//! The carry-out is computed as: `cout = (a & b) | ((a ^ b) & ¬sum)`.
//!
//! # Constraints
//!
//! The gate generates 1 AND constraint and 1 linear constraint:
//!
//! 1. Carry propagation: `(a ⊕ (cout << 1)) ∧ (b ⊕ (cout << 1)) = cout ⊕ (cout << 1)`
//! 2. Sum: `sum = a ⊕ b ⊕ (cout << 1)`

use crate::compiler::{
	constraint_builder::{ConstraintBuilder, sll, xor2, xor3},
	gate::opcode::OpcodeShape,
	gate_graph::{Gate, GateData, GateParam, Wire},
};

pub fn shape() -> OpcodeShape {
	OpcodeShape {
		const_in: &[],
		n_in: 2,
		n_out: 2,
		n_aux: 0,
		n_scratch: 0,
		n_imm: 0,
	}
}

pub fn constrain(_gate: Gate, data: &GateData, builder: &mut ConstraintBuilder) {
	let GateParam {
		inputs, outputs, ..
	} = data.gate_param();
	let [a, b] = inputs else { unreachable!() };
	let [sum, cout] = outputs else { unreachable!() };

	let cout_sll_1 = sll(*cout, 1);

	// Constraint 1: Carry propagation
	//
	// (a ⊕ (cout << 1)) ∧ (b ⊕ (cout << 1)) = cout ⊕ (cout << 1)
	builder
		.and()
		.a(xor2(*a, cout_sll_1))
		.b(xor2(*b, cout_sll_1))
		.c(xor2(*cout, cout_sll_1))
		.build();

	// Constraint 2: Sum equality (linear)
	//
	// (a ⊕ b ⊕ (cout << 1)) = sum
	builder
		.linear()
		.rhs(xor3(*a, *b, cout_sll_1))
		.dst(*sum)
		.build();
}

pub fn emit_eval_bytecode(
	_gate: Gate,
	data: &GateData,
	builder: &mut crate::compiler::eval_form::BytecodeBuilder,
	wire_to_reg: impl Fn(Wire) -> u32,
) {
	let GateParam {
		inputs, outputs, ..
	} = data.gate_param();
	let [a, b] = inputs else { unreachable!() };
	let [sum, cout] = outputs else { unreachable!() };
	builder.emit_iadd_cout(
		wire_to_reg(*sum),
		wire_to_reg(*cout),
		wire_to_reg(*a),
		wire_to_reg(*b),
	);
}
