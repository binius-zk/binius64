// Copyright 2025 Irreducible Inc.
//! Parallel 32-bit unsigned integer addition without carry-in.
//!
//! Performs simultaneous independent 32-bit additions on the upper and lower 32-bit halves of
//! the 64-bit word (like [`sll32`](super::sll32) operates on independent halves).
//!
//! # Wires
//!
//! - `x`, `y`: Input wires for the summands
//! - `z`: Output wire containing the resulting sum
//! - `cout` (carry-out): Output wire containing a carry word where each bit position indicates
//!   whether a carry occurred at that position during the addition. In particular, bit 31 and bit
//!   63 indicate the carry-out of the lower and upper 32-bit halves respectively.
//!
//! # Constraints
//!
//! The gate generates 1 AND constraint and 1 linear constraint:
//! 1. Carry propagation: `(x ⊕ (cout <<₃₂ 1)) ∧ (y ⊕ (cout <<₃₂ 1)) = cout ⊕ (cout <<₃₂ 1)`
//! 2. Result: `z = x ⊕ y ⊕ (cout <<₃₂ 1)`
//!
//! `<<₃₂` denotes a shift that operates independently on each 32-bit half.

use crate::compiler::{
	constraint_builder::{ConstraintBuilder, sll32, xor2, xor3},
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
	let [x, y] = inputs else { unreachable!() };
	let [z, cout] = outputs else { unreachable!() };

	let cout_shifted = sll32(*cout, 1);

	// Constraint 1: Carry propagation
	//
	// (x ⊕ (cout <<₃₂ 1)) ∧ (y ⊕ (cout <<₃₂ 1)) = cout ⊕ (cout <<₃₂ 1)
	builder
		.and()
		.a(xor2(*x, cout_shifted))
		.b(xor2(*y, cout_shifted))
		.c(xor2(*cout, cout_shifted))
		.build();

	// Constraint 2: Result
	//
	// z = x ⊕ y ⊕ (cout <<₃₂ 1)
	builder
		.linear()
		.dst(*z)
		.rhs(xor3(*x, *y, cout_shifted))
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
	builder.emit_iadd32_cout(
		wire_to_reg(*sum),
		wire_to_reg(*cout),
		wire_to_reg(*a),
		wire_to_reg(*b),
	);
}
