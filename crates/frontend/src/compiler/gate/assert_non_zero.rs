// Copyright 2025 Irreducible Inc.
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
//! - AND: `sar(cout, 63) ∧ all-1 = all-1` (forces `MSB(cout) = 1`, i.e. `x ≠ 0`)

use binius_core::word::Word;

use crate::compiler::{
	constraint_builder::{ConstraintBuilder, expr},
	gate::opcode::OpcodeShape,
	gate_graph::{Gate, GateData, GateParam, Wire},
	pathspec::PathSpec,
};

pub const fn shape() -> OpcodeShape {
	OpcodeShape {
		// ALL_ONE is the addend for the carry.
		const_in: &[Word::ALL_ONE],
		n_in: 1,
		n_out: 0,
		n_aux: 1,
		n_scratch: 1,
		n_imm: 0,
	}
}

pub fn constrain(_gate: Gate, data: &GateData, builder: &mut ConstraintBuilder) {
	let GateParam {
		constants,
		inputs,
		aux,
		..
	} = data.gate_param();
	let [all_one] = constants else { unreachable!() };
	let [x] = inputs else { unreachable!() };
	let [cout] = aux else { unreachable!() };

	let cin = expr::sll(*cout, 1);

	// Constraint 1: Constrain carry-out
	// (x ⊕ cin) ∧ (all-1 ⊕ cin) = cin ⊕ cout
	builder
		.and()
		.a(expr::xor2(*x, cin))
		.b(expr::xor2(*all_one, cin))
		.c(expr::xor2(cin, *cout))
		.build();

	// Constraint 2 (AND): sar(cout, 63) ∧ all_one = all_one, i.e. MSB(cout) = 1 (x ≠ 0).
	// sar(cout, 63) sign-extends the MSB across all 64 bits, so it equals all_one iff
	// MSB(cout) = 1; the AND with all_one then equals all_one iff MSB(cout) = 1. This is
	// emitted as an AND (which defines no wire) so gate fusion cannot inline it and
	// substitute the constant back into Constraint 1, reopening the soundness hole.
	builder
		.and()
		.a(expr::sar(*cout, 63))
		.b(*all_one)
		.c(*all_one)
		.build();
}

pub fn emit_eval_bytecode(
	_gate: Gate,
	data: &GateData,
	assertion_path: PathSpec,
	builder: &mut crate::compiler::eval_form::BytecodeBuilder,
	wire_to_reg: impl Fn(Wire) -> u32,
) {
	let GateParam {
		constants,
		inputs,
		aux,
		scratch,
		..
	} = data.gate_param();
	let [all_one] = constants else { unreachable!() };
	let [x] = inputs else { unreachable!() };
	let [cout] = aux else { unreachable!() };
	let [scratch_sum_unused] = scratch else {
		unreachable!()
	};

	// Compute carry bits from all_one + x (cin = 0 implicit)
	builder.emit_iadd_cout(
		wire_to_reg(*scratch_sum_unused), // sum (unused)
		wire_to_reg(*cout),               // cout
		wire_to_reg(*all_one),            // all_one
		wire_to_reg(*x),                  // x
	);

	builder.emit_assert_non_zero(wire_to_reg(*cout), assertion_path.as_u32());
}

#[cfg(test)]
mod tests {
	use binius_core::{verify::verify_constraints, word::Word};
	use rand::prelude::*;

	use crate::compiler::CircuitBuilder;

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
			verify_constraints(cs, &w.into_value_vec()).unwrap();
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
			verify_constraints(cs, &w.into_value_vec()).unwrap();
		}
	}

	#[test]
	fn test_assert_non_zero_forge_zero_rejected() {
		use binius_core::constraint_system::ValueIndex;

		// Soundness regression: a malicious prover claims `x ≠ 0` while actually planting
		// `x = 0` and the aux carry-out `cout = 0`. Before the `MSB(cout) = 1` constraint
		// (`sar(cout, 63) ∧ all_one = all_one`) was added, only the carry-defining AND was
		// emitted, and `x = 0, cout = 0` satisfies it, so `verify_constraints` wrongly accepted
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
			w.value_vec_mut()[ValueIndex(i as u32)] = *c;
		}

		// The carry-out constraint is satisfied by the all-zero witness, but the AND
		// `sar(cout, 63) ∧ all_one = all_one` constraint (`MSB(cout) = 1`) must reject it.
		let result = verify_constraints(cs, w.value_vec());
		assert!(
			result.is_err(),
			"verify_constraints must reject the forged x = 0 witness, got: {result:?}"
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
