// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers
//! Assert that a wire isn't zero.
//!
//! Enforces `x ≠ 0`.
//!
//! # Algorithm
//!
//! The idea is similar to `icmp_eq`, but actually simpler.
//! First off, we only have one operand, not two;
//! secondly, we don't need to negate the MSB of the result.
//!
//! The gate exploits the property that when adding `all-1` to a value:
//! - If the value is 0: `0 + all-1 = all-1` with no carry out (MSB of cout = 0)
//! - If the value is non-zero: `value + all-1` wraps around with carry out (MSB of cout = 1)
//!
//! The algorithm is as follows:
//! 1. Compute carry bits `cout` from `x + all-1` using the constraint: `(x ⊕ cin) ∧ (all-1 ⊕ cin) =
//!    cin ⊕ cout` where `cin = cout << 1`
//! 2. The MSB of `cout` tells us whether x ≠ 0; i.e.,
//!    - MSB = 0: no carry out, meaning `x = 0`
//!    - MSB = 1: carry out occurred, meaning `x ≠ 0`
//!
//! # Constraints
//!
//! The gate generates 2 constraints:
//! - AND: `(x ⊕ cin) ∧ (all-1 ⊕ cin) = cin ⊕ cout`
//! - ZERO: `sar(cout, 63) ⊕ all-1 = 0` (forces `MSB(cout) = 1`, i.e. `x ≠ 0`)
//!
//! No gadget in the workspace calls this gate.
//! It is kept as a primitive, and because the two-constraint split is easy to get wrong.

use binius_core::word::Word;

use crate::{
	eval_form::BytecodeBuilder,
	gates::{EmitCtx, GateKind, OpcodeShape},
	ir::GateParam,
	lower::{ConstraintBuilder, expr},
};

/// That a word is not zero.
pub struct AssertNonZero;

impl GateKind for AssertNonZero {
	// The constant is the addend for the carry; the auxiliary wire holds the carry-out.
	const SHAPE: OpcodeShape = OpcodeShape::new(1, 0)
		.with_consts(&[Word::ALL_ONE])
		.with_aux(1);

	fn constrain(gate: GateParam<'_>, cb: &mut ConstraintBuilder) {
		let [all_one] = gate.const_wires();
		let [x] = gate.in_wires();
		let [cout] = gate.aux_wires();

		let cin = expr::sll(cout, 1);

		// Carry-out: (x ⊕ cin) ∧ (all-1 ⊕ cin) = cin ⊕ cout
		cb.and(expr::xor2(x, cin), expr::xor2(all_one, cin), expr::xor2(cin, cout));

		// MSB(cout) = 1, as sar(cout, 63) ⊕ all-1 = 0.
		// Fusion cannot inline an assertion, so the constant stays out of the carry constraint.
		cb.zero(expr::xor2(expr::sar(cout, 63), all_one));
	}

	fn emit(gate: GateParam<'_>, ctx: EmitCtx<'_>, bc: &mut BytecodeBuilder) {
		let [all_one] = gate.const_wires();
		let [x] = gate.in_wires();
		let [cout] = gate.aux_wires();

		// Carry bits of all-1 + x. Only the carries matter, so the sum is not stored.
		bc.emit_iadd_carry(ctx.reg(cout), ctx.reg(all_one), ctx.reg(x));

		bc.emit_assert_non_zero(ctx.reg(cout), ctx.path());
	}
}

#[cfg(test)]
mod tests {
	use binius_core::word::Word;
	use rand::prelude::*;

	use crate::builder::CircuitBuilder;

	#[test]
	fn test_assert_non_zero_basic() {
		// Build a circuit with assert_non_zero gate
		let builder = CircuitBuilder::new();
		let x = builder.add_inout();
		builder.assert_non_zero("non_zero", x);
		let circuit = builder.build();

		// Test specific non-zero cases
		let test_cases = [
			1_u64,
			0xFFFFFFFFFFFFFFFF_u64,
			0x1234567890ABCDEF_u64,
			0x8000000000000000_u64,
			0x0000000000000001_u64,
		];

		for x_val in test_cases {
			let mut w = circuit.new_witness_filler();
			w[x] = Word(x_val);
			w.circuit.populate_wire_witness(&mut w).unwrap();

			// Verify constraints pass for non-zero values
			let cs = circuit.constraint_system();
			cs.verify(&w.into_value_vec()).unwrap();
		}
	}

	#[test]
	fn test_assert_non_zero_random() {
		// Build a circuit with assert_non_zero gate
		let builder = CircuitBuilder::new();
		let x = builder.add_inout();
		builder.assert_non_zero("non_zero", x);
		let circuit = builder.build();

		// Test with random non-zero values
		let mut rng = StdRng::seed_from_u64(42);
		for _ in 0..1000 {
			let mut x_val = rng.next_u64();
			// Ensure we don't test with zero
			if x_val == 0 {
				x_val = 1;
			}

			let mut w = circuit.new_witness_filler();
			w[x] = Word(x_val);
			w.circuit.populate_wire_witness(&mut w).unwrap();

			// Verify constraints pass
			let cs = circuit.constraint_system();
			cs.verify(&w.into_value_vec()).unwrap();
		}
	}

	#[test]
	fn test_assert_non_zero_forge_zero_rejected() {
		use binius_core::constraint_system::ValueIndex;

		// Soundness regression: a malicious prover claims `x ≠ 0` while actually planting
		// `x = 0` and the aux carry-out `cout = 0`. Before the `MSB(cout) = 1` constraint
		// (`sar(cout, 63) ∧ all_one = all_one`) was added, only the carry-defining AND was
		// emitted, and `x = 0, cout = 0` satisfies it, so constraint verification wrongly accepted
		// this forged witness.
		//
		// The existing `test_assert_non_zero_fails_on_zero` only exercises the prover-side
		// `populate_wire_witness` panic; it does not touch the verifier-side hole. This test
		// bypasses the prover and injects the malicious witness directly.
		let builder = CircuitBuilder::new();
		let x = builder.add_inout();
		builder.assert_non_zero("non_zero", x);
		let circuit = builder.build();

		// Build the forged witness by hand. A fresh value vec is all zeros, so the input `x`
		// and the aux carry-out `cout` are already 0. We cannot call `populate_wire_witness`
		// (it panics on `x = 0`), so we only fill the constants section directly, exactly as
		// `populate_wire_witness` would, so the verifier's constant check passes.
		let mut w = circuit.new_witness_filler();
		let cs = circuit.constraint_system();
		for (i, c) in cs.constants.iter().enumerate() {
			w.value_vec_mut()[ValueIndex::constant(i as u32)] = *c;
		}

		// The carry-out constraint is satisfied by the all-zero witness, but the AND
		// `sar(cout, 63) ∧ all_one = all_one` constraint (`MSB(cout) = 1`) must reject it.
		let result = cs.verify(w.value_vec());
		assert!(
			result.is_err(),
			"constraint verification must reject the forged x = 0 witness, got: {result:?}"
		);
	}

	#[test]
	#[should_panic(expected = "Word(0x0000000000000000) == 0")]
	fn test_assert_non_zero_fails_on_zero() {
		// Build a circuit with assert_non_zero gate
		let builder = CircuitBuilder::new();
		let x = builder.add_inout();
		builder.assert_non_zero("non_zero", x);
		let circuit = builder.build();

		// Test with zero value (should panic)
		let mut w = circuit.new_witness_filler();
		w[x] = Word(0);
		// This should panic when trying to assert non-zero on zero
		w.circuit.populate_wire_witness(&mut w).unwrap();
	}
}
