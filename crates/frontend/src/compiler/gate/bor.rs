// Copyright 2025 Irreducible Inc.
//! Bitwise OR operation.
//!
//! Returns `z = x | y`.
//!
//! # Algorithm
//!
//! Computes the bitwise OR from the identity `x | y = (x ∧ y) ⊕ x ⊕ y`.
//! Rearranged so the AND stands alone, that identity is the constraint below.
//!
//! # Constraints
//!
//! The gate generates 1 AND constraint:
//! - `x ∧ y = x ⊕ y ⊕ z`

use crate::compiler::{
	constraint_builder::{ConstraintBuilder, expr},
	eval_form::BytecodeBuilder,
	gate::opcode::OpcodeShape,
	gate_graph::{GateData, GateParam, Wire},
};

pub const fn shape() -> OpcodeShape {
	OpcodeShape {
		const_in: &[],
		n_in: 2,
		n_out: 1,
		n_aux: 0,
		n_scratch: 0,
		n_imm: 0,
	}
}

pub fn constrain(data: &GateData, builder: &mut ConstraintBuilder) {
	let GateParam {
		inputs, outputs, ..
	} = data.gate_param();
	let [x, y] = inputs else { unreachable!() };
	let [z] = outputs else { unreachable!() };

	// Constraint: Bitwise OR
	//
	// x ∧ y = x ⊕ y ⊕ z
	builder.and(*x, *y, expr::xor3(*x, *y, *z));
}

pub fn emit_eval_bytecode(
	data: &GateData,
	builder: &mut BytecodeBuilder,
	wire_to_reg: impl Fn(Wire) -> u32,
) {
	let GateParam {
		inputs, outputs, ..
	} = data.gate_param();
	let [x, y] = inputs else { unreachable!() };
	let [z] = outputs else { unreachable!() };

	builder.emit_bor(wire_to_reg(*z), wire_to_reg(*x), wire_to_reg(*y));
}
