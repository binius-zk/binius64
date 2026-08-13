// Copyright 2025 Irreducible Inc.
//! Conditional equality assertion.
//!
//! Enforces `x = y` when the MSB-bool value of `cond` is true, and no constraint otherwise.
//!
//! # Algorithm
//!
//! Uses a mask to conditionally enforce equality: `(x ^ y) & (cond ~>> 63) = 0`.
//! When `cond` is MSB-bool-true, this enforces `x = y`. otherwise, the constraint is satisfied
//! trivially.
//!
//! # Constraints
//!
//! The gate generates 1 AND constraint:
//! - `(x ⊕ y) ∧ (cond ~>> 63) = 0`

use crate::{
	eval_form::BytecodeBuilder,
	gates::opcode::OpcodeShape,
	ir::{GateData, GateParam, Wire, path::PathSpec},
	lower::{ConstraintBuilder, expr},
};

pub const fn shape() -> OpcodeShape {
	OpcodeShape {
		const_in: &[],
		n_in: 3,
		n_out: 0,
		n_aux: 0,
		n_scratch: 0,
		n_imm: 0,
	}
}

pub fn constrain(data: &GateData, builder: &mut ConstraintBuilder) {
	let GateParam { inputs, .. } = data.gate_param();
	let [x, y, cond] = inputs else { unreachable!() };

	// Constraint: (x ⊕ y) ∧ (cond ~>> 63) = 0
	let mask = expr::sar(*cond, 63);
	builder.and(expr::xor2(*x, *y), mask, expr::empty());
}

pub fn emit_eval_bytecode(
	data: &GateData,
	assertion_path: PathSpec,
	builder: &mut BytecodeBuilder,
	wire_to_reg: impl Fn(Wire) -> u32,
) {
	let GateParam { inputs, .. } = data.gate_param();
	let [x, y, cond] = inputs else { unreachable!() };

	// The condition is read as an MSB-bool, and broadcasting the sign bit preserves it.
	// So the condition is passed as it stands, with no mask to compute or hold.
	builder.emit_assert_eq_cond(
		wire_to_reg(*cond),
		wire_to_reg(*x),
		wire_to_reg(*y),
		assertion_path.as_u32(),
	);
}
