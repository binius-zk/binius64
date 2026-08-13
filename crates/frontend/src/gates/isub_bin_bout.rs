// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers
//! 64-bit unsigned subtraction with borrow propagation.
//!
//! # Constraints
//!
//! The gate generates 1 AND constraint and 1 linear constraint, for `bi = (bout << 1) ⊕ bin_msb`:
//! 1. Borrow propagation: `(¬a ⊕ bi) ∧ (b ⊕ bi) = bout ⊕ bi`
//! 2. Result: `diff = a ⊕ b ⊕ bi`

use binius_core::word::Word;

use crate::{
	eval_form::BytecodeBuilder,
	gates::{EmitCtx, GateKind, OpcodeShape},
	ir::GateParam,
	lower::{ConstraintBuilder, expr},
};

/// The 64-bit difference of two words and a borrow-in, with its borrow word.
pub struct IsubBinBout;

impl GateKind for IsubBinBout {
	const SHAPE: OpcodeShape = OpcodeShape::new(3, 2).with_consts(&[Word::ALL_ONE]);

	fn constrain(gate: GateParam<'_>, cb: &mut ConstraintBuilder) {
		let [all_one] = gate.const_wires();
		let [a, b, bin] = gate.in_wires();
		let [diff, bout] = gate.out_wires();

		let bout_sll_1 = expr::sll(bout, 1);
		let bin_msb = expr::srl(bin, 63);

		// Borrow propagation:
		// (¬a ⊕ (bout << 1) ⊕ bin_msb) ∧ (b ⊕ (bout << 1) ⊕ bin_msb) = bout ⊕ (bout << 1) ⊕ bin_msb
		cb.and(
			expr::xor4(all_one, a, bout_sll_1, bin_msb),
			expr::xor3(b, bout_sll_1, bin_msb),
			expr::xor3(bout, bout_sll_1, bin_msb),
		);

		// Difference (linear): (a ⊕ b ⊕ (bout << 1) ⊕ bin_msb) = diff
		cb.linear(expr::xor4(a, b, bout_sll_1, bin_msb), diff);
	}

	fn emit(gate: GateParam<'_>, ctx: EmitCtx<'_>, bc: &mut BytecodeBuilder) {
		let [a, b, bin] = gate.in_wires();
		let [diff, bout] = gate.out_wires();

		bc.emit_isub_bin_bout(ctx.reg(diff), ctx.reg(bout), ctx.reg(a), ctx.reg(b), ctx.reg(bin));
	}
}
