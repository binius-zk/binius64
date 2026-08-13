// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers
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
	gates::{EmitCtx, GateKind, OpcodeShape},
	ir::GateParam,
	lower::{ConstraintBuilder, expr},
};

/// Whether one word is less than another, as an MSB-bool.
pub struct IcmpUlt;

impl GateKind for IcmpUlt {
	// The scratch wire holds the negated left operand during evaluation.
	const SHAPE: OpcodeShape = OpcodeShape::new(2, 1)
		.with_consts(&[Word::ALL_ONE])
		.with_scratch(1);

	fn constrain(gate: GateParam<'_>, cb: &mut ConstraintBuilder) {
		let [all_one] = gate.const_wires();
		let [x, y] = gate.in_wires();
		let [bout] = gate.out_wires();

		// Carry propagation for the comparison:
		// ((x ⊕ all-1) ⊕ (bout << 1)) ∧ (y ⊕ (bout << 1)) = bout ⊕ (bout << 1)
		cb.and(
			expr::xor3(x, all_one, expr::sll(bout, 1)),
			expr::xor2(y, expr::sll(bout, 1)),
			expr::xor2(bout, expr::sll(bout, 1)),
		);
	}

	fn emit(gate: GateParam<'_>, ctx: EmitCtx<'_>, bc: &mut BytecodeBuilder) {
		let [all_one] = gate.const_wires();
		let [x, y] = gate.in_wires();
		let [bout] = gate.out_wires();
		let [negated_x] = gate.scratch_wires();

		// ¬x, as x exclusive-ored with all-1.
		bc.emit_bxor(ctx.reg(negated_x), ctx.reg(x), ctx.reg(all_one));

		// Carry bits of ¬x + y. Only the carries matter, so the sum is not stored.
		bc.emit_iadd_carry(ctx.reg(bout), ctx.reg(negated_x), ctx.reg(y));
	}
}
