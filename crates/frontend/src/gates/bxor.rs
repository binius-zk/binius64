// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers
//! Bitwise XOR operation.
//!
//! Returns `z = x ^ y`.
//!
//! # Algorithm
//!
//! Computes the bitwise XOR using a linear constraint.
//!
//! # Constraints
//!
//! The gate generates 1 linear constraint:
//! - `x ⊕ y = z`

use crate::{
	eval_form::BytecodeBuilder,
	gates::{EmitCtx, GateKind, OpcodeShape},
	ir::GateParam,
	lower::{ConstraintBuilder, expr},
};

/// Bitwise exclusive-or of two words.
pub struct Bxor;

impl GateKind for Bxor {
	const SHAPE: OpcodeShape = OpcodeShape::new(2, 1);

	fn constrain(gate: GateParam<'_>, cb: &mut ConstraintBuilder) {
		let [x, y] = gate.in_wires();
		let [z] = gate.out_wires();

		// (x ⊕ y) = z
		cb.linear(expr::xor2(x, y), z);
	}

	fn emit(gate: GateParam<'_>, ctx: EmitCtx<'_>, bc: &mut BytecodeBuilder) {
		let [x, y] = gate.in_wires();
		let [z] = gate.out_wires();

		bc.emit_bxor(ctx.reg(z), ctx.reg(x), ctx.reg(y));
	}
}
