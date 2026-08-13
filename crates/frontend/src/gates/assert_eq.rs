// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers
//! Equality assertion.
//!
//! Enforces `x = y` using a ZERO constraint.
//!
//! # Algorithm
//!
//! Uses the property that `x = y` iff `x ^ y = 0`.
//!
//! # Constraints
//!
//! The gate generates 1 ZERO constraint:
//! - `x ⊕ y = 0`

use crate::{
	eval_form::BytecodeBuilder,
	gates::{EmitCtx, GateKind, OpcodeShape},
	ir::GateParam,
	lower::{ConstraintBuilder, expr},
};

/// That two words are equal.
pub struct AssertEq;

impl GateKind for AssertEq {
	const SHAPE: OpcodeShape = OpcodeShape::new(2, 0);

	fn constrain(gate: GateParam<'_>, cb: &mut ConstraintBuilder) {
		let [x, y] = gate.in_wires();

		// x ⊕ y = 0
		cb.zero(expr::xor2(x, y));
	}

	fn emit(gate: GateParam<'_>, ctx: EmitCtx<'_>, bc: &mut BytecodeBuilder) {
		let [x, y] = gate.in_wires();

		bc.emit_assert_eq(ctx.reg(x), ctx.reg(y), ctx.path().as_u32());
	}
}
