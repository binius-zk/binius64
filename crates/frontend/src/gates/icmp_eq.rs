// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers
//! 64-bit equality test that returns an MSB-bool indicating equality.
//!
//! Returns a wire whose value, as an MSB-bool, is true if `x == y` and false otherwise.
//! It is undefined what the NON-most-significant bits of the output wire will be.
//!
//! # Algorithm
//!
//! The gate exploits the property that when adding `all-1` to a value:
//! - If the value is 0: `0 + all-1 = all-1` with no carry out (MSB of cout = 0)
//! - If the value is non-zero: `value + all-1` wraps around with carry out (MSB of cout = 1)
//!
//! 1. Compute `diff = x ⊕ y` (which is 0 iff x == y)
//! 2. Compute carry bits `cout` from `diff + all-1` using the constraint: `(x ⊕ y ⊕ cin) ∧ (all-1 ⊕
//!    cin) = cin ⊕ cout` where `cin = cout << 1`
//! 3. The MSB of `cout` indicates the comparison result:
//!    - MSB = 0: no carry out, meaning `diff = 0`, so `x == y`
//!    - MSB = 1: carry out occurred, meaning `diff ≠ 0`, so `x ≠ y`
//! 4. Negate the MSB: `out_wire ≔ cout ⊕ 0x8000000000000000`.
//!
//! we do this in a slightly more roundabout way, in order to make things work with the builder.
//! since `out_wire` and `cout` will differ only in their MSB, it's valid to take
//! `cin ≔ out_wire << 1`, instead of `cin ≔ cout << 1`. thus, we never materialize cout.
//! in the one place we need it, we inline the expression `out_wire ⊕ 0x8000000000000000` instead.
//!
//! of course, in theory, we could have instead done out_wire ≔ cout ⊕ 0xFFFFFFFFFFFFFFFF.
//! this still would have been correct, as far as the MSB of `out_wire` was concerned.
//! the problem with doing that is that we then would have had to put `cin ≔ ¬out-wire << 1`.
//! doubling up on the negation and the shift caused a problem, so this works out quite nicely.
//!
//! # Constraints
//!
//! The gate generates one AND constraint:
//! 1. Carry propagation: `(x ⊕ y ⊕ cin) ∧ (all-1 ⊕ cin) = cin ⊕ cout`

use binius_core::word::Word;

use crate::{
	eval_form::BytecodeBuilder,
	gates::{EmitCtx, GateKind, OpcodeShape},
	ir::GateParam,
	lower::{ConstraintBuilder, expr},
};

/// Whether two words are equal, as an MSB-bool.
pub struct IcmpEq;

impl GateKind for IcmpEq {
	// The auxiliary wire is the carry-out; the scratch wire holds the operands' difference.
	const SHAPE: OpcodeShape = OpcodeShape::new(2, 1)
		.with_consts(&[Word::ALL_ONE, Word::MSB_ONE])
		.with_aux(1)
		.with_scratch(1);

	fn constrain(gate: GateParam<'_>, cb: &mut ConstraintBuilder) {
		let [all_one, msb_one] = gate.const_wires();
		let [x, y] = gate.in_wires();
		let [out] = gate.out_wires();

		let cin = expr::sll(out, 1);

		// Carry-out: (x ⊕ y ⊕ cin) ∧ (all-1 ⊕ cin) = cin ⊕ cout
		cb.and(expr::xor3(x, y, cin), expr::xor2(all_one, cin), expr::xor3(cin, out, msb_one));
	}

	fn emit(gate: GateParam<'_>, ctx: EmitCtx<'_>, bc: &mut BytecodeBuilder) {
		let [all_one, msb_one] = gate.const_wires();
		let [x, y] = gate.in_wires();
		let [out] = gate.out_wires();
		let [cout] = gate.aux_wires();
		let [diff] = gate.scratch_wires();

		// diff = x ⊕ y
		bc.emit_bxor(ctx.reg(diff), ctx.reg(x), ctx.reg(y));

		// Carry bits of all-1 + diff. Only the carries matter, so the sum is not stored.
		bc.emit_iadd_carry(ctx.reg(cout), ctx.reg(all_one), ctx.reg(diff));

		// Invert the most significant bit, turning "differ" into "equal".
		bc.emit_bxor(ctx.reg(out), ctx.reg(cout), ctx.reg(msb_one));
	}
}
