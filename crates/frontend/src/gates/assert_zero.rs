// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers
//! Assert that a wire equals zero.
//!
//! Enforces `x = 0` using a ZERO constraint.
//!
//! # Constraints
//!
//! The gate generates 1 ZERO constraint:
//! - `x = 0`

use crate::{
	eval_form::BytecodeBuilder,
	gates::opcode::OpcodeShape,
	ir::{GateData, GateParam, Wire, path::PathSpec},
	lower::ConstraintBuilder,
};

pub const fn shape() -> OpcodeShape {
	OpcodeShape {
		const_in: &[],
		n_in: 1,
		n_out: 0,
		n_aux: 0,
		n_scratch: 0,
		n_imm: 0,
	}
}

pub fn constrain(data: &GateData, builder: &mut ConstraintBuilder) {
	let GateParam { inputs, .. } = data.gate_param();
	let [x] = inputs else { unreachable!() };

	// Constraint: x = 0
	builder.zero(*x);
}

pub fn emit_eval_bytecode(
	data: &GateData,
	assertion_path: PathSpec,
	builder: &mut BytecodeBuilder,
	wire_to_reg: impl Fn(Wire) -> u32,
) {
	let GateParam { inputs, .. } = data.gate_param();
	let [x] = inputs else { unreachable!() };
	builder.emit_assert_zero(wire_to_reg(*x), assertion_path.as_u32());
}
