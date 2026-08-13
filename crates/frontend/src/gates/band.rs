// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers
//! Bitwise AND operation.
//!
//! Returns `z = x & y`.
//!
//! # Algorithm
//!
//! Computes the bitwise AND of two 64-bit words using a single AND constraint.
//!
//! # Constraints
//!
//! The gate generates 1 AND constraint:
//! - `x ∧ y = z`

use crate::{
	eval_form::BytecodeBuilder,
	gates::{EmitCtx, GateKind, OpcodeShape},
	ir::GateParam,
	lower::ConstraintBuilder,
};

/// Bitwise AND of two words.
pub struct Band;

impl GateKind for Band {
	const SHAPE: OpcodeShape = OpcodeShape::new(2, 1);

	fn constrain(gate: GateParam<'_>, cb: &mut ConstraintBuilder) {
		let [x, y] = gate.in_wires();
		let [z] = gate.out_wires();

		// x ∧ y = z
		cb.and(x, y, z);
	}

	fn emit(gate: GateParam<'_>, ctx: EmitCtx<'_>, bc: &mut BytecodeBuilder) {
		let [x, y] = gate.in_wires();
		let [z] = gate.out_wires();

		bc.emit_band(ctx.reg(z), ctx.reg(x), ctx.reg(y));
	}
}
