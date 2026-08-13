// Copyright 2025 Irreducible Inc.
//! Unsigned less-than test returning a mask.
//!
//! Returns a wire whose value as an MSB-bool is true if `x < y`, and false otherwise.
//! It is undefined what the NON-most-significant bits of the output wire will be.
//!
//! # Algorithm
//!
//! The gate computes `x < y` by checking if there's a borrow when computing `x - y`.
//! This is done by computing `¬x + y` and checking if it carries out (≥ 2^64).
//!
//! 1. Compute carry bits `bout` from `¬x + y` using the constraint: `(¬x ⊕ bin) ∧ (y ⊕ bin) = bin ⊕
//!    bout` where `bin = bout << 1`
//! 2. The MSB of `bout` indicates the comparison result:
//!    - MSB = 1: carry out occurred, meaning `x < y`
//!    - MSB = 0: no carry out, meaning `x ≥ y`
//!
//! # Constraints
//!
//! The gate generates 1 AND constraint:
//! 1. Borrow propagation: `(¬x ⊕ bin) ∧ (y ⊕ bin) = bin ⊕ bout`

use binius_core::word::Word;

use crate::{
	eval_form::BytecodeBuilder,
	gates::opcode::OpcodeShape,
	ir::{GateData, GateParam, Wire},
	lower::{ConstraintBuilder, expr},
};

pub const fn shape() -> OpcodeShape {
	OpcodeShape {
		const_in: &[Word::ALL_ONE],
		n_in: 2,
		n_out: 1,
		n_aux: 0,
		n_scratch: 1, // Holds the negated left operand
		n_imm: 0,
	}
}

pub fn constrain(data: &GateData, builder: &mut ConstraintBuilder) {
	let GateParam {
		inputs,
		outputs,
		constants,
		..
	} = data.gate_param();
	let [all_one] = constants else { unreachable!() };
	let [x, y] = inputs else { unreachable!() };
	let [bout] = outputs else { unreachable!() };

	// Constraint 1: Carry propagation for comparison
	// ((x ⊕ all-1) ⊕ (bout << 1)) ∧ (y ⊕ (bout << 1)) = bout ⊕ (bout << 1)
	builder.and(
		expr::xor3(*x, *all_one, expr::sll(*bout, 1)),
		expr::xor2(*y, expr::sll(*bout, 1)),
		expr::xor2(*bout, expr::sll(*bout, 1)),
	);
}

pub fn emit_eval_bytecode(
	data: &GateData,
	builder: &mut BytecodeBuilder,
	wire_to_reg: impl Fn(Wire) -> u32,
) {
	let GateParam {
		constants,
		inputs,
		outputs,
		scratch,
		..
	} = data.gate_param();
	let [all_one] = constants else { unreachable!() };
	let [x, y] = inputs else { unreachable!() };
	let [bout] = outputs else { unreachable!() };
	let [scratch_nx] = scratch else {
		unreachable!()
	};

	// Compute ¬x (x XOR all_one)
	builder.emit_bxor(wire_to_reg(*scratch_nx), wire_to_reg(*x), wire_to_reg(*all_one));

	// Carry bits of ¬x + y.
	// Only the carries matter, so the sum is not stored.
	builder.emit_iadd_carry(wire_to_reg(*bout), wire_to_reg(*scratch_nx), wire_to_reg(*y));
}
