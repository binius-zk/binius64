// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers
//! Assert that a wire, interpreted as a MSB-bool, is true.
//! i.e., we are checking whether its most-significant bit is 1. all lower bits get ignored.
//!
//! Enforces `MSB(x) = 1` using a ZERO constraint.
//!
//! # Algorithm
//!
//! `sar(x, 63)` broadcasts the most-significant bit across the whole word, so it is all-1 when the
//! bit is set and 0 when it is clear. Equating it with all-1 therefore says the bit is set.
//!
//! # Constraints
//!
//! The gate generates 1 ZERO constraint:
//! - `sar(x, 63) ⊕ all-1 = 0`

use binius_core::word::Word;

use crate::{
	eval_form::BytecodeBuilder,
	gates::{EmitCtx, GateKind, OpcodeShape},
	ir::GateParam,
	lower::{ConstraintBuilder, expr},
};

/// That a word's most significant bit is set.
pub struct AssertTrue;

impl GateKind for AssertTrue {
	const SHAPE: OpcodeShape = OpcodeShape::new(1, 0).with_consts(&[Word::ALL_ONE]);

	fn constrain(gate: GateParam<'_>, cb: &mut ConstraintBuilder) {
		let [all_one] = gate.const_wires();
		let [x] = gate.in_wires();

		// sar(x, 63) ⊕ all-1 = 0
		cb.zero(expr::xor2(expr::sar(x, 63), all_one));
	}

	fn emit(gate: GateParam<'_>, ctx: EmitCtx<'_>, bc: &mut BytecodeBuilder) {
		let [x] = gate.in_wires();

		bc.emit_assert_true(ctx.reg(x), ctx.path().as_u32());
	}
}
