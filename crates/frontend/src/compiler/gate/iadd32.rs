// Copyright 2025 Irreducible Inc.
//! Parallel 32-bit unsigned integer addition with carry-in and carry-out.
//!
//! Performs simultaneous independent 32-bit additions on the upper and lower 32-bit halves of
//! the 64-bit word (like [`sll32`](super::sll32) operates on independent halves).
//!
//! # Wires
//!
//! - `x`, `y`: Input wires for the summands
//! - `cin` (carry-in): Input wire for the previous carry word. The MSB of each 32-bit half
//!   is used as the carry-in bit for that half (bit 31 for the lower half, bit 63 for the upper).
//! - `z`: Output wire containing the resulting sum
//! - `cout` (carry-out): Output wire containing a carry word where each bit position indicates
//!   whether a carry occurred at that position during the addition. In particular, bit 31 and
//!   bit 63 indicate the carry-out of the lower and upper 32-bit halves respectively.
//!
//! # Constraints
//!
//! The gate generates 1 AND constraint and 1 linear constraint:
//! 1. Carry propagation: `(x ⊕ ci) ∧ (y ⊕ ci) = cout ⊕ ci`
//!    where `ci = (cout <<₃₂ 1) ⊕ (cin >>₃₂ 31)`
//! 2. Result: `z = x ⊕ y ⊕ ci`
//!
//! `<<₃₂` and `>>₃₂` denote shifts that operate independently on each 32-bit half.

use crate::compiler::{
	constraint_builder::{ConstraintBuilder, sll32, srl32, xor3, xor4},
	gate::opcode::OpcodeShape,
	gate_graph::{Gate, GateData, GateParam, Wire},
};

pub fn shape() -> OpcodeShape {
	OpcodeShape {
		const_in: &[],
		n_in: 3,
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
	let [x, y, cin] = inputs else { unreachable!() };
	let [z, cout] = outputs else { unreachable!() };

	let cout_shifted = sll32(*cout, 1);
	let cin_bit = srl32(*cin, 31);

	// Constraint 1: Carry propagation
	//
	// (x ⊕ ci) ∧ (y ⊕ ci) = cout ⊕ ci
	// where ci = (cout <<₃₂ 1) ⊕ (cin >>₃₂ 31)
	builder
		.and()
		.a(xor3(*x, cout_shifted, cin_bit))
		.b(xor3(*y, cout_shifted, cin_bit))
		.c(xor3(*cout, cout_shifted, cin_bit))
		.build();

	// Constraint 2: Result
	//
	// z = x ⊕ y ⊕ ci
	builder
		.linear()
		.dst(*z)
		.rhs(xor4(*x, *y, cout_shifted, cin_bit))
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
	let [a, b, cin] = inputs else { unreachable!() };
	let [sum, cout] = outputs else { unreachable!() };
	builder.emit_iadd32_cin_cout(
		wire_to_reg(*sum),
		wire_to_reg(*cout),
		wire_to_reg(*a),
		wire_to_reg(*b),
		wire_to_reg(*cin),
	);
}
