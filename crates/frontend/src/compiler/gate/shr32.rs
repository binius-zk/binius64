// Copyright 2025 Irreducible Inc.
//! 32-bit logical right shift.
//!
//! Returns `z = (x >> n) & MASK_32`.
//!
//! # Algorithm
//!
//! Shifts the input right by `n` bits and masks to 32 bits.
//!
//! # Constraints
//!
//! The gate generates 1 AND constraint:
//! - `(x >> n) ∧ MASK_32 = z`

use binius_core::word::Word;

use crate::compiler::{
	constraint_builder::{ConstraintBuilder, srl},
	gate::opcode::OpcodeShape,
	gate_graph::{Gate, GateData, GateParam, Wire},
};

pub fn shape() -> OpcodeShape {
	OpcodeShape {
		const_in: &[Word::MASK_32],
		n_in: 1,
		n_out: 1,
		n_aux: 0,
		n_scratch: 0,
		n_imm: 1,
	}
}

pub fn constrain(_gate: Gate, data: &GateData, builder: &mut ConstraintBuilder) {
	let GateParam {
		constants,
		inputs,
		outputs,
		imm,
		..
	} = data.gate_param();
	let [mask32] = constants else { unreachable!() };
	let [x] = inputs else { unreachable!() };
	let [z] = outputs else { unreachable!() };
	let [n] = imm else { unreachable!() };

	// Constraint: Shift right with masking
	// (x >> n) ∧ MASK_32 = z
	builder.and().a(srl(*x, *n)).b(*mask32).c(*z).build();
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
		imm,
		..
	} = data.gate_param();
	let [x] = inputs else { unreachable!() };
	let [z] = outputs else { unreachable!() };
	let [n] = imm else { unreachable!() };
	builder.emit_shr32(wire_to_reg(*z), wire_to_reg(*x), *n as u8);
}
