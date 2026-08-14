// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers
//! Assert that a wire equals zero.
//!
//! Enforces `x = 0` using a ZERO constraint.
//!
//! # Constraints
//!
//! The gate generates 1 ZERO constraint:
//! - `x = 0`

use crate::{
	eval_form::BytecodeBuilder,
	gates::{EmitCtx, GateKind, OpcodeShape},
	ir::GateParam,
	lower::ConstraintBuilder,
};

/// That a word is zero.
pub struct AssertZero;

impl GateKind for AssertZero {
	const SHAPE: OpcodeShape = OpcodeShape::new(1, 0);

	fn constrain(gate: GateParam<'_>, cb: &mut ConstraintBuilder) {
		let [x] = gate.in_wires();

		// x = 0
		cb.zero(x);
	}

	fn emit(gate: GateParam<'_>, ctx: EmitCtx<'_>, bc: &mut BytecodeBuilder) {
		let [x] = gate.in_wires();

		bc.emit_assert_zero(ctx.reg(x), ctx.path());
	}
}
