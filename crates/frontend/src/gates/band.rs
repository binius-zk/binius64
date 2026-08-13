// Copyright 2025 Irreducible Inc.
//! Bitwise AND operation.
//!
//! Returns `z = x & y`.
//!
//! # Algorithm
//!
//! Computes the bitwise AND of two 64-bit words using a single AND constraint.
//!
//! # Constraints
//!
//! The gate generates 1 AND constraint:
//! - `x ∧ y = z`

use crate::{
	eval_form::BytecodeBuilder,
	gates::opcode::OpcodeShape,
	ir::{GateData, GateParam, Wire},
	lower::ConstraintBuilder,
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

	// Constraint: Bitwise AND
	//
	// x ∧ y = z
	builder.and(*x, *y, *z);
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

	builder.emit_band(wire_to_reg(*z), wire_to_reg(*x), wire_to_reg(*y));
}
