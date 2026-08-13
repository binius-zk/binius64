// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers
//! Bitwise OR operation.
//!
//! Returns `z = x | y`.
//!
//! # Algorithm
//!
//! Computes the bitwise OR from the identity `x | y = (x ∧ y) ⊕ x ⊕ y`.
//! Rearranged so the AND stands alone, that identity is the constraint below.
//!
//! # Constraints
//!
//! The gate generates 1 AND constraint:
//! - `x ∧ y = x ⊕ y ⊕ z`

use crate::{
	eval_form::BytecodeBuilder,
	gates::{EmitCtx, GateKind, OpcodeShape},
	ir::GateParam,
	lower::{ConstraintBuilder, expr},
};

/// Bitwise or of two words.
pub struct Bor;

impl GateKind for Bor {
	const SHAPE: OpcodeShape = OpcodeShape::new(2, 1);

	fn constrain(gate: GateParam<'_>, cb: &mut ConstraintBuilder) {
		let [x, y] = gate.in_wires();
		let [z] = gate.out_wires();

		// x ∧ y = x ⊕ y ⊕ z
		cb.and(x, y, expr::xor3(x, y, z));
	}

	fn emit(gate: GateParam<'_>, ctx: EmitCtx<'_>, bc: &mut BytecodeBuilder) {
		let [x, y] = gate.in_wires();
		let [z] = gate.out_wires();

		bc.emit_bor(ctx.reg(z), ctx.reg(x), ctx.reg(y));
	}
}
