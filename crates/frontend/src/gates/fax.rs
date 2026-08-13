// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers
//! Fused AND-XOR operation.
//!
//! Returns `z = (x & y) ^ w`.
//!
//! # Algorithm
//!
//! Computes the bitwise AND of two words followed by XOR with a third word.
//! This common pattern is fused into a single gate for efficiency.
//!
//! # Constraints
//!
//! The gate generates 1 AND constraint:
//! - `x & y = t` where `t ^ w = z`

use crate::{
	eval_form::BytecodeBuilder,
	gates::{EmitCtx, GateKind, OpcodeShape},
	ir::GateParam,
	lower::{ConstraintBuilder, expr},
};

/// Bitwise and of two words, exclusive-ored with a third.
pub struct Fax;

impl GateKind for Fax {
	const SHAPE: OpcodeShape = OpcodeShape::new(3, 1);

	fn constrain(gate: GateParam<'_>, cb: &mut ConstraintBuilder) {
		let [x, y, w] = gate.in_wires();
		let [z] = gate.out_wires();

		// x ∧ y = z ⊕ w, which is z = (x ∧ y) ⊕ w rearranged.
		cb.and(x, y, expr::xor2(z, w));
	}

	fn emit(gate: GateParam<'_>, ctx: EmitCtx<'_>, bc: &mut BytecodeBuilder) {
		let [x, y, w] = gate.in_wires();
		let [z] = gate.out_wires();

		bc.emit_fax(ctx.reg(z), ctx.reg(x), ctx.reg(y), ctx.reg(w));
	}
}
