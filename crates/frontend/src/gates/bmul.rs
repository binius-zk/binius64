// Copyright 2026 The Binius Developers
//! Bmul gate implements multiplication in the GHASH field GF(2^128).
//!
//! Each field element is carried by a `(lo, hi)` pair of 64-bit words. Uses the BmulConstraint:
//! `(A_LO, A_HI) * (B_LO, B_HI) = (C_LO, C_HI)`.

use crate::{
	eval_form::BytecodeBuilder,
	gates::{EmitCtx, GateKind, OpcodeShape},
	ir::GateParam,
	lower::ConstraintBuilder,
};

/// The product of two GHASH-field elements, each carried by a pair of words.
pub struct Bmul;

impl GateKind for Bmul {
	const SHAPE: OpcodeShape = OpcodeShape::new(4, 2);

	fn constrain(gate: GateParam<'_>, cb: &mut ConstraintBuilder) {
		let [a_lo, a_hi, b_lo, b_hi] = gate.in_wires();
		let [c_lo, c_hi] = gate.out_wires();

		// (a_lo, a_hi) * (b_lo, b_hi) = (c_lo, c_hi) in GF(2^128).
		cb.bmul(a_lo, a_hi, b_lo, b_hi, c_lo, c_hi);
	}

	fn emit(gate: GateParam<'_>, ctx: EmitCtx<'_>, bc: &mut BytecodeBuilder) {
		let [a_lo, a_hi, b_lo, b_hi] = gate.in_wires();
		let [c_lo, c_hi] = gate.out_wires();

		bc.emit_bmul(
			ctx.reg(c_lo),
			ctx.reg(c_hi),
			ctx.reg(a_lo),
			ctx.reg(a_hi),
			ctx.reg(b_lo),
			ctx.reg(b_hi),
		);
	}
}
