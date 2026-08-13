// Copyright 2026 The Binius Developers
//! Bmul gate implements multiplication in the GHASH field GF(2^128).
//!
//! Each field element is carried by a `(lo, hi)` pair of 64-bit words. Uses the BmulConstraint:
//! `(A_LO, A_HI) * (B_LO, B_HI) = (C_LO, C_HI)`.

use crate::{
	eval_form::BytecodeBuilder,
	gates::opcode::OpcodeShape,
	ir::{GateData, GateParam, Wire},
	lower::ConstraintBuilder,
};

pub const fn shape() -> OpcodeShape {
	OpcodeShape {
		const_in: &[],
		n_in: 4,
		n_out: 2,
		n_aux: 0,
		n_scratch: 0,
		n_imm: 0,
	}
}

pub fn constrain(data: &GateData, builder: &mut ConstraintBuilder) {
	let GateParam {
		inputs, outputs, ..
	} = data.gate_param();
	let [a_lo, a_hi, b_lo, b_hi] = inputs else {
		unreachable!()
	};
	let [c_lo, c_hi] = outputs else {
		unreachable!()
	};

	// Create BmulConstraint: (A_LO, A_HI) * (B_LO, B_HI) = (C_LO, C_HI) in GF(2^128).
	builder.bmul(*a_lo, *a_hi, *b_lo, *b_hi, *c_lo, *c_hi);
}

pub fn emit_eval_bytecode(
	data: &GateData,
	builder: &mut BytecodeBuilder,
	wire_to_reg: impl Fn(Wire) -> u32,
) {
	let GateParam {
		inputs, outputs, ..
	} = data.gate_param();
	let [a_lo, a_hi, b_lo, b_hi] = inputs else {
		unreachable!()
	};
	let [c_lo, c_hi] = outputs else {
		unreachable!()
	};
	builder.emit_bmul(
		wire_to_reg(*c_lo),
		wire_to_reg(*c_hi),
		wire_to_reg(*a_lo),
		wire_to_reg(*a_hi),
		wire_to_reg(*b_lo),
		wire_to_reg(*b_hi),
	);
}
