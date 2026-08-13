// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers
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
	gates::{EmitCtx, GateKind, OpcodeShape},
	ir::GateParam,
	lower::{ConstraintBuilder, expr},
};

/// That two words are equal, where a third is true as an MSB-bool.
pub struct AssertEqCond;

impl GateKind for AssertEqCond {
	const SHAPE: OpcodeShape = OpcodeShape::new(3, 0);

	fn constrain(gate: GateParam<'_>, cb: &mut ConstraintBuilder) {
		let [x, y, cond] = gate.in_wires();

		// (x ⊕ y) ∧ (cond ~>> 63) = 0
		cb.and(expr::xor2(x, y), expr::sar(cond, 63), expr::empty());
	}

	fn emit(gate: GateParam<'_>, ctx: EmitCtx<'_>, bc: &mut BytecodeBuilder) {
		let [x, y, cond] = gate.in_wires();

		// The condition is read as an MSB-bool, and broadcasting the sign bit preserves it.
		// So the condition is passed as it stands, with no mask to compute or hold.
		bc.emit_assert_eq_cond(ctx.reg(cond), ctx.reg(x), ctx.reg(y), ctx.path().as_u32());
	}
}
