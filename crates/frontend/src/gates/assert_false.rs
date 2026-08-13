// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers
//! Assert that a wire, interpreted as a MSB-bool, is false.
//! i.e., we are checking whether its most-significant bit is 0. all lower bits get ignored.
//!
//! Enforces `MSB(x) = 0` using a ZERO constraint.
//!
//! # Algorithm
//!
//! `sar(x, 63)` broadcasts the most-significant bit across the whole word, so it vanishes exactly
//! when the bit is clear.
//!
//! # Constraints
//!
//! The gate generates 1 ZERO constraint:
//! - `sar(x, 63) = 0`

use crate::{
	eval_form::BytecodeBuilder,
	gates::{EmitCtx, GateKind, OpcodeShape},
	ir::GateParam,
	lower::{ConstraintBuilder, expr},
};

/// That a word's most significant bit is clear.
pub struct AssertFalse;

impl GateKind for AssertFalse {
	const SHAPE: OpcodeShape = OpcodeShape::new(1, 0);

	fn constrain(gate: GateParam<'_>, cb: &mut ConstraintBuilder) {
		let [x] = gate.in_wires();

		// sar(x, 63) = 0
		cb.zero(expr::sar(x, 63));
	}

	fn emit(gate: GateParam<'_>, ctx: EmitCtx<'_>, bc: &mut BytecodeBuilder) {
		let [x] = gate.in_wires();

		bc.emit_assert_false(ctx.reg(x), ctx.path().as_u32());
	}
}
