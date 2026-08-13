// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers
//! 64-bit unsigned integer addition with carry propagation.
//!
//! # Wires
//!
//! - `a`, `b`: Input wires for the summands
//! - `cin` (carry-in): Input wire for the previous carry word. Only the MSB is used as the actual
//!   carry bit
//! - `sum`: Output wire containing the resulting sum = a + b + carry_bit
//! - `cout` (carry-out): Output wire containing a carry word where each bit position indicates
//!   whether a carry occurred at that position during the addition.
//!
//! ## Carry-out Computation
//!
//! The carry-out is computed as: `cout = (a & b) | ((a ^ b) & ¬sum)`
//!
//! For example:
//! - `0x0000000000000003 + 0x0000000000000001 = 0x0000000000000004` with `cout =
//!   0x0000000000000003` (carries at bits 0 and 1)
//! - `0xFFFFFFFFFFFFFFFF + 0x0000000000000001 = 0x0000000000000000` with `cout =
//!   0xFFFFFFFFFFFFFFFF` (carries at all bit positions)
//!
//! # Constraints
//!
//! The gate generates two AND constraints:
//!
//! 1. **Carry generation constraint**: Ensures correct carry propagation
//! 2. **Sum constraint**: Ensures the sum equals `a ^ b ^ (cout << 1) ^ cin_msb`

use crate::{
	eval_form::BytecodeBuilder,
	gates::{EmitCtx, GateKind, OpcodeShape},
	ir::GateParam,
	lower::{ConstraintBuilder, expr},
};

/// The 64-bit sum of two words and a carry-in, with its carry word.
pub struct IaddCinCout;

impl GateKind for IaddCinCout {
	const SHAPE: OpcodeShape = OpcodeShape::new(3, 2);

	fn constrain(gate: GateParam<'_>, cb: &mut ConstraintBuilder) {
		let [a, b, cin] = gate.in_wires();
		let [sum, cout] = gate.out_wires();

		let cout_sll_1 = expr::sll(cout, 1);
		let cin_msb = expr::srl(cin, 63);

		// Carry propagation:
		// (a ⊕ (cout << 1) ⊕ cin_msb) ∧ (b ⊕ (cout << 1) ⊕ cin_msb) = cout ⊕ (cout << 1) ⊕ cin_msb
		cb.and(
			expr::xor3(a, cout_sll_1, cin_msb),
			expr::xor3(b, cout_sll_1, cin_msb),
			expr::xor3(cout, cout_sll_1, cin_msb),
		);

		// Sum equality (linear): (a ⊕ b ⊕ (cout << 1) ⊕ cin_msb) = sum
		cb.linear(expr::xor4(a, b, cout_sll_1, cin_msb), sum);
	}

	fn emit(gate: GateParam<'_>, ctx: EmitCtx<'_>, bc: &mut BytecodeBuilder) {
		let [a, b, cin] = gate.in_wires();
		let [sum, cout] = gate.out_wires();

		bc.emit_iadd_cin_cout(ctx.reg(sum), ctx.reg(cout), ctx.reg(a), ctx.reg(b), ctx.reg(cin));
	}
}
