// Copyright 2025 Irreducible Inc.
//! Parallel 32-bit unsigned integer addition with carry propagation.
//!
//! Performs simultaneous independent 32-bit additions on the upper and lower 32-bit halves of
//! the 64-bit word (like [`sll32`](super::sll32) operates on independent halves). Returns
//! `z = x + y` (with carry not crossing the 32-bit boundary) and `cout` containing carry bits.
//!
//! # Constraints
//!
//! The gate generates 1 AND constraint and 1 linear constraint:
//! 1. Carry propagation: `(x ⊕ (cout <<₃₂ 1)) ∧ (y ⊕ (cout <<₃₂ 1)) = cout ⊕ (cout <<₃₂ 1)`
//! 2. Result: `z = x ⊕ y ⊕ (cout <<₃₂ 1)`
//!
//! where `<<₃₂` denotes [`sll32`](super::sll32), a shift that operates independently on each
//! 32-bit half.

use crate::compiler::{
	constraint_builder::{ConstraintBuilder, sll32, xor2, xor3},
	gate::opcode::OpcodeShape,
	gate_graph::{Gate, GateData, GateParam, Wire},
};

pub fn shape() -> OpcodeShape {
	OpcodeShape {
		const_in: &[],
		n_in: 2,
		n_out: 1,
		n_aux: 1,
		n_scratch: 0,
		n_imm: 0,
	}
}

pub fn constrain(_gate: Gate, data: &GateData, builder: &mut ConstraintBuilder) {
	let GateParam {
		inputs,
		outputs,
		aux,
		..
	} = data.gate_param();
	let [x, y] = inputs else { unreachable!() };
	let [z] = outputs else { unreachable!() };
	let [cout] = aux else { unreachable!() };

	let cin = sll32(*cout, 1);

	// Constraint 1: Carry propagation
	//
	// (x ⊕ (cout << 1)) ∧ (y ⊕ (cout << 1)) = cout ⊕ (cout << 1)
	builder
		.and()
		.a(xor2(*x, cin))
		.b(xor2(*y, cin))
		.c(xor2(*cout, cin))
		.build();

	// Constraint 2: Result
	//
	// z = x ⊕ y ⊕ (cout <<₃₂ 1)
	builder.linear().dst(*z).rhs(xor3(*x, *y, cin)).build();
}

pub fn emit_eval_bytecode(
	_gate: Gate,
	data: &GateData,
	builder: &mut crate::compiler::eval_form::BytecodeBuilder,
	wire_to_reg: impl Fn(Wire) -> u32,
) {
	let GateParam {
		inputs,
		outputs,
		aux,
		..
	} = data.gate_param();
	let [a, b] = inputs else { unreachable!() };
	let [sum] = outputs else { unreachable!() };
	let [cout] = aux else { unreachable!() };
	builder.emit_iadd_cout32(
		wire_to_reg(*sum),
		wire_to_reg(*cout),
		wire_to_reg(*a),
		wire_to_reg(*b),
	);
}
