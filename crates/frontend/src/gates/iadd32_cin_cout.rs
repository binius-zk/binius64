// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers
//! Parallel 32-bit unsigned integer addition with carry-in and carry-out.
//!
//! Performs simultaneous independent 32-bit additions on the upper and lower 32-bit halves of
//! the 64-bit word. Carries do not cross the 32-bit lane boundary.
//!
//! # Wires
//!
//! - `x`, `y`: Input wires for the summands
//! - `cin` (carry-in): Input wire for the previous carry word. The MSB of each 32-bit half is used
//!   as the carry-in bit for that half (bit 31 for the lower half, bit 63 for the upper).
//! - `z`: Output wire containing the resulting sum
//! - `cout` (carry-out): Output wire containing a carry word where each bit position indicates
//!   whether a carry occurred at that position during the addition. In particular, bit 31 and bit
//!   63 indicate the carry-out of the lower and upper 32-bit halves respectively.
//!
//! # Constraints
//!
//! The gate generates 1 AND constraint and 1 linear constraint:
//! 1. Carry propagation: `(x ⊕ ci) ∧ (y ⊕ ci) = cout ⊕ ci` where `ci = (cout <<₃₂ 1) ⊕ (cin >>₃₂
//!    31)`
//! 2. Result: `z = x ⊕ y ⊕ ci`
//!
//! `<<₃₂` and `>>₃₂` denote shifts that operate independently on each 32-bit half.

use crate::{
	eval_form::BytecodeBuilder,
	gates::{EmitCtx, GateKind, OpcodeShape},
	ir::GateParam,
	lower::{ConstraintBuilder, expr},
};

/// Two independent 32-bit sums with per-half carry-in, one per half of the word.
pub struct Iadd32CinCout;

impl GateKind for Iadd32CinCout {
	const SHAPE: OpcodeShape = OpcodeShape::new(3, 2);

	fn constrain(gate: GateParam<'_>, cb: &mut ConstraintBuilder) {
		let [x, y, cin] = gate.in_wires();
		let [z, cout] = gate.out_wires();

		let cout_shifted = expr::sll32(cout, 1);
		let cin_bit = expr::srl32(cin, 31);

		// Carry propagation, for ci = (cout <<₃₂ 1) ⊕ (cin >>₃₂ 31):
		// (x ⊕ ci) ∧ (y ⊕ ci) = cout ⊕ ci
		cb.and(
			expr::xor3(x, cout_shifted, cin_bit),
			expr::xor3(y, cout_shifted, cin_bit),
			expr::xor3(cout, cout_shifted, cin_bit),
		);

		// Result: z = x ⊕ y ⊕ ci
		cb.linear(expr::xor4(x, y, cout_shifted, cin_bit), z);
	}

	fn emit(gate: GateParam<'_>, ctx: EmitCtx<'_>, bc: &mut BytecodeBuilder) {
		let [a, b, cin] = gate.in_wires();
		let [sum, cout] = gate.out_wires();

		bc.emit_iadd32_cin_cout(ctx.reg(sum), ctx.reg(cout), ctx.reg(a), ctx.reg(b), ctx.reg(cin));
	}
}
