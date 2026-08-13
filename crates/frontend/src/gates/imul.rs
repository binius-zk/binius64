// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers
//! Imul gate implements 64-bit × 64-bit → 128-bit unsigned multiplication.
//! Uses the ImulConstraint: X * Y = (HI << 64) | LO

use crate::{
	eval_form::BytecodeBuilder,
	gates::{EmitCtx, GateKind, OpcodeShape},
	ir::GateParam,
	lower::ConstraintBuilder,
};

/// The 128-bit unsigned product of two words.
pub struct Imul;

impl GateKind for Imul {
	const SHAPE: OpcodeShape = OpcodeShape::new(2, 2);

	fn constrain(gate: GateParam<'_>, cb: &mut ConstraintBuilder) {
		let [x, y] = gate.in_wires();
		let [hi, lo] = gate.out_wires();

		// x * y = (hi << 64) | lo
		cb.imul(x, y, hi, lo);
	}

	fn emit(gate: GateParam<'_>, ctx: EmitCtx<'_>, bc: &mut BytecodeBuilder) {
		let [x, y] = gate.in_wires();
		let [hi, lo] = gate.out_wires();

		bc.emit_imul(ctx.reg(hi), ctx.reg(lo), ctx.reg(x), ctx.reg(y));
	}
}
