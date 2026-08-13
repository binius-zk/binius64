// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers
//! Assert that a wire, interpreted as a MSB-bool, is true.
//! i.e., we are checking whether its most-significant bit is 1. all lower bits get ignored.
//!
//! Enforces `MSB(x) = 1` using a ZERO constraint.
//!
//! # Algorithm
//!
//! `sar(x, 63)` broadcasts the most-significant bit across the whole word, so it is all-1 when the
//! bit is set and 0 when it is clear. Equating it with all-1 therefore says the bit is set.
//!
//! # Constraints
//!
//! The gate generates 1 ZERO constraint:
//! - `sar(x, 63) ⊕ all-1 = 0`

use binius_core::word::Word;

use crate::{
	eval_form::BytecodeBuilder,
	gates::opcode::OpcodeShape,
	ir::{GateData, GateParam, Wire, path::PathSpec},
	lower::{ConstraintBuilder, expr},
};

pub const fn shape() -> OpcodeShape {
	OpcodeShape {
		const_in: &[Word::ALL_ONE],
		n_in: 1,
		n_out: 0,
		n_aux: 0,
		n_scratch: 0,
		n_imm: 0,
	}
}

pub fn constrain(data: &GateData, builder: &mut ConstraintBuilder) {
	let GateParam {
		constants, inputs, ..
	} = data.gate_param();
	let [all_one] = constants else { unreachable!() };
	let [x] = inputs else { unreachable!() };

	// Constraint: sar(x, 63) ⊕ all-1 = 0
	builder.zero(expr::xor2(expr::sar(*x, 63), *all_one));
}

pub fn emit_eval_bytecode(
	data: &GateData,
	assertion_path: PathSpec,
	builder: &mut BytecodeBuilder,
	wire_to_reg: impl Fn(Wire) -> u32,
) {
	let GateParam { inputs, .. } = data.gate_param();
	let [x] = inputs else { unreachable!() };
	builder.emit_assert_true(wire_to_reg(*x), assertion_path.as_u32());
}
