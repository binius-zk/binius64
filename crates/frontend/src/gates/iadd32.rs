// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers
//! Parallel 32-bit unsigned integer addition without carry-in.
//!
//! Performs simultaneous independent 32-bit additions on the upper and lower 32-bit halves of
//! the 64-bit word. Carries do not cross the 32-bit lane boundary.
//!
//! # Wires
//!
//! - `x`, `y`: Input wires for the summands
//! - `z`: Output wire containing the resulting sum
//! - `cout` (carry-out): Output wire containing a carry word where each bit position indicates
//!   whether a carry occurred at that position during the addition. In particular, bit 31 and bit
//!   63 indicate the carry-out of the lower and upper 32-bit halves respectively.
//!
//! # Constraints
//!
//! The gate generates 1 AND constraint and 1 linear constraint:
//! 1. Carry propagation: `(x ⊕ (cout <<₃₂ 1)) ∧ (y ⊕ (cout <<₃₂ 1)) = cout ⊕ (cout <<₃₂ 1)`
//! 2. Result: `z = x ⊕ y ⊕ (cout <<₃₂ 1)`
//!
//! `<<₃₂` denotes a shift that operates independently on each 32-bit half.

use crate::{
	eval_form::BytecodeBuilder,
	gates::{EmitCtx, GateKind, OpcodeShape},
	ir::GateParam,
	lower::{ConstraintBuilder, expr},
};

/// Two independent 32-bit sums, one per half of the word.
pub struct Iadd32;

impl GateKind for Iadd32 {
	const SHAPE: OpcodeShape = OpcodeShape::new(2, 2);

	fn constrain(gate: GateParam<'_>, cb: &mut ConstraintBuilder) {
		let [x, y] = gate.in_wires();
		let [z, cout] = gate.out_wires();

		let cout_shifted = expr::sll32(cout, 1);

		// Carry propagation:
		// (x ⊕ (cout <<₃₂ 1)) ∧ (y ⊕ (cout <<₃₂ 1)) = cout ⊕ (cout <<₃₂ 1)
		cb.and(
			expr::xor2(x, cout_shifted),
			expr::xor2(y, cout_shifted),
			expr::xor2(cout, cout_shifted),
		);

		// Result: z = x ⊕ y ⊕ (cout <<₃₂ 1)
		cb.linear(expr::xor3(x, y, cout_shifted), z);
	}

	fn emit(gate: GateParam<'_>, ctx: EmitCtx<'_>, bc: &mut BytecodeBuilder) {
		let [a, b] = gate.in_wires();
		let [sum, cout] = gate.out_wires();

		bc.emit_iadd32_cout(ctx.reg(sum), ctx.reg(cout), ctx.reg(a), ctx.reg(b));
	}
}
