// Copyright 2025 Irreducible Inc.
//! 32-bit rotate right.
//!
//! Returns `z = ((x >> n) | (x << (32-n))) & MASK_32`.
//!
//! # Algorithm
//!
//! Rotates a 32-bit value right by `n` positions:
//! 1. Shift right by n: `t1 = x >> n` (bits n-31 move to positions 0-(31-n))
//! 2. Shift left by 32-n: `t2 = x << (32-n)` (bits 0-(n-1) move to positions (32-n)-31)
//! 3. Combine with XOR: Since the shifted ranges don't overlap, `t1 | t2 = t1 ^ t2`
//! 4. Mask to 32 bits: `z = (t1 ^ t2) & MASK_32`
//!
//! The non-overlapping property is crucial: right-shifted bits occupy positions 0-(31-n),
//! while left-shifted bits occupy positions (32-n)-31, with no overlap.
//!
//! # Constraints
//!
//! The gate generates 1 AND constraint:
//! - `((x >> n) ⊕ (x << (32-n))) ∧ MASK_32 = z`

use binius_core::word::Word;

use crate::compiler::{
	constraint_builder::{ConstraintBuilder, sll, srl, xor2},
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

	// Constraint: Rotate right
	// ((x >> n) ⊕ (x << (32-n))) ∧ MASK_32 = z
	builder
		.and()
		.a(xor2(srl(*x, *n), sll(*x, 32 - *n)))
		.b(*mask32)
		.c(*z)
		.build();
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
	builder.emit_rotr32(wire_to_reg(*z), wire_to_reg(*x), *n as u8);
}
