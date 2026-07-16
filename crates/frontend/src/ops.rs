// Copyright 2025-2026 The Binius Developers
// Copyright 2025 Irreducible Inc.
//! Circuit gadgets: reusable sub-circuits built from primitive gates.
//!
//! A gadget emits a fixed pattern of existing gates.
//! It adds no new opcode and no bespoke constraint logic.

use crate::{CircuitBuilder, Wire};

/// 64-bit × 64-bit → 128-bit signed multiplication.
///
/// Returns `(hi, lo)` where the two's complement product equals `(hi << 64) | lo`.
///
/// # Algorithm
///
/// Unsigned multiplication with a high-word correction.
///
/// Source: Hennessy & Patterson, "Computer Architecture: A Quantitative Approach", 6th ed. (2019),
/// App. J.2, pp. J-11 to J-13.
///
/// Read as unsigned bit patterns, the signed operands expand as shown below.
///
/// ```text
/// a_signed = a - 2^64 * a_sign
/// b_signed = b - 2^64 * b_sign
/// ```
///
/// Here `a_sign` and `b_sign` are the sign bits (bit 63).
///
/// Multiplying and reducing modulo 2^128 leaves two correction terms.
///
/// ```text
/// a_signed * b_signed = a * b
///                     - 2^64 * (a_sign * b)
///                     - 2^64 * (b_sign * a)
/// ```
///
/// The `2^128 * a_sign * b_sign` term is zero modulo 2^128.
///
/// - The low word matches the unsigned product unchanged.
/// - The high word subtracts `b` when `a` is negative.
/// - The high word subtracts `a` when `b` is negative.
///
/// # Cost
///
/// - 1 IMUL constraint for the unsigned product.
/// - 1 AND constraint for the unsigned product security check.
/// - 2 AND constraints for the sign masks.
/// - 2 AND constraints for the corrections.
/// - 2 AND constraints for the two high-word subtractions.
pub fn smul64(builder: &CircuitBuilder, a: Wire, b: Wire) -> (Wire, Wire) {
	// Unsigned 128-bit product of the raw bit patterns.
	// Only the high word needs a signed correction; the low word is already final.
	let (hi_u, lo) = builder.imul(a, b);

	// Sign masks via arithmetic shift by 63.
	// All-ones when the operand is negative, all-zeros otherwise.
	let a_sign = builder.sar(a, 63);
	let b_sign = builder.sar(b, 63);

	// Corrections to subtract from the high word:
	//   correction_a = if a < 0 { b } else { 0 }
	//   correction_b = if b < 0 { a } else { 0 }
	let correction_a = builder.band(a_sign, b);
	let correction_b = builder.band(b_sign, a);

	// High word: subtract both corrections modulo 2^64.
	// Borrows past bit 63 fall outside the 128-bit product, so they are discarded.
	let zero = builder.add_constant_64(0);
	let (hi, _) = builder.isub_bin_bout(hi_u, correction_a, zero);
	let (hi, _) = builder.isub_bin_bout(hi, correction_b, zero);

	(hi, lo)
}

impl CircuitBuilder {
	/// 64-bit unsigned integer addition, returning the sum and carry-out.
	///
	/// Addition with a carry-in is the general primitive.
	/// Plain addition is the special case where the carry-in is zero.
	///
	/// Returns `(sum, cout)` where:
	///
	/// - `sum` is the 64-bit result `a + b`.
	/// - `cout` has a set bit at every position where a carry occurred.
	///
	/// # Cost
	///
	/// - 1 AND constraint,
	/// - 1 linear constraint.
	pub fn iadd(&self, a: Wire, b: Wire) -> (Wire, Wire) {
		// Zero carry-in: the MSB of `cin` is the carry bit, and zero carries nothing.
		let cin = self.add_constant_64(0);
		self.iadd_cin_cout(a, b, cin)
	}

	/// 64-bit × 64-bit → 128-bit signed multiplication.
	///
	/// Handles two's complement operands, including overflow cases.
	///
	/// Returns `(hi, lo)` where the signed product equals `(hi << 64) | lo`.
	///
	/// The high word is the sign extension of the product.
	///
	/// Thin wrapper over [`smul64`].
	pub fn smul(&self, a: Wire, b: Wire) -> (Wire, Wire) {
		smul64(self, a, b)
	}
}

#[cfg(test)]
mod tests {
	use binius_core::{verify::verify_constraints, word::Word};
	use proptest::prelude::*;

	use crate::CircuitBuilder;

	proptest! {
		#[test]
		fn test_smul_correctness(x_val: i64, y_val: i64) {
			// Invariant: the gadget's 128-bit output equals native i128 signed multiplication.
			let builder = CircuitBuilder::new();
			// Two signed 64-bit operands as public inputs.
			let x = builder.add_inout();
			let y = builder.add_inout();
			// Gadget under test.
			let (hi, lo) = builder.smul(x, y);
			// Expected 128-bit result, pinned as two public words.
			let expected_hi = builder.add_inout();
			let expected_lo = builder.add_inout();
			builder.assert_eq("smul_hi", hi, expected_hi);
			builder.assert_eq("smul_lo", lo, expected_lo);
			let circuit = builder.build();

			// Assign the random operands.
			let mut w = circuit.new_witness_filler();
			w[x] = Word(x_val as u64);
			w[y] = Word(y_val as u64);

			// Reference: native 128-bit signed product, split into high and low words.
			let result = (x_val as i128) * (y_val as i128);
			w[expected_hi] = Word((result >> 64) as u64);
			w[expected_lo] = Word(result as u64);
			// Evaluate the circuit to fill every internal wire.
			w.circuit.populate_wire_witness(&mut w).unwrap();

			// All AND/IMUL constraints must hold for the correct witness.
			let cs = circuit.constraint_system();
			verify_constraints(cs, &w.into_value_vec()).unwrap();
		}
	}

	proptest! {
		#[test]
		fn test_smul_commutative(x_val: i64, y_val: i64) {
			// Invariant: signed multiplication is commutative, so x*y and y*x agree word-for-word.
			let builder = CircuitBuilder::new();
			let x = builder.add_inout();
			let y = builder.add_inout();

			// Compute both operand orders.
			let (hi1, lo1) = builder.smul(x, y);
			let (hi2, lo2) = builder.smul(y, x);

			// The two products must match in both words.
			builder.assert_eq("hi_equal", hi1, hi2);
			builder.assert_eq("lo_equal", lo1, lo2);
			let circuit = builder.build();

			let mut w = circuit.new_witness_filler();
			w[x] = Word(x_val as u64);
			w[y] = Word(y_val as u64);
			w.circuit.populate_wire_witness(&mut w).unwrap();

			let cs = circuit.constraint_system();
			verify_constraints(cs, &w.into_value_vec()).unwrap();
		}
	}

	proptest! {
		#[test]
		fn test_smul_zero_identity(x_val: i64) {
			// Invariant: multiplying by zero yields zero in both words.
			let builder = CircuitBuilder::new();
			let x = builder.add_inout();
			let zero = builder.add_constant_64(0);
			let (hi, lo) = builder.smul(x, zero);

			// Both output words must be zero.
			builder.assert_zero("hi_is_zero", hi);
			builder.assert_zero("lo_is_zero", lo);
			let circuit = builder.build();

			let mut w = circuit.new_witness_filler();
			w[x] = Word(x_val as u64);
			w.circuit.populate_wire_witness(&mut w).unwrap();

			let cs = circuit.constraint_system();
			verify_constraints(cs, &w.into_value_vec()).unwrap();
		}
	}

	proptest! {
		#[test]
		fn test_smul_one_identity(x_val: i64) {
			// Invariant: multiplying by one returns x, sign-extended into the high word.
			let builder = CircuitBuilder::new();
			let x = builder.add_inout();
			let one = builder.add_constant_64(1);
			let (hi, lo) = builder.smul(x, one);

			// Low word is x unchanged.
			builder.assert_eq("lo_equals_x", lo, x);
			// High word is the sign extension: all-ones for negative x, all-zeros otherwise.
			let expected_hi = if x_val < 0 {
				builder.add_constant(Word::ALL_ONE)
			} else {
				builder.add_constant_64(0)
			};
			builder.assert_eq("hi_sign_extended", hi, expected_hi);
			let circuit = builder.build();

			let mut w = circuit.new_witness_filler();
			w[x] = Word(x_val as u64);
			w.circuit.populate_wire_witness(&mut w).unwrap();

			let cs = circuit.constraint_system();
			verify_constraints(cs, &w.into_value_vec()).unwrap();
		}
	}

	proptest! {
		#[test]
		fn test_smul_neg_one(x_val: i64) {
			// Invariant: multiplying by -1 negates x across the full 128-bit result.
			let builder = CircuitBuilder::new();
			let x = builder.add_inout();
			// -1 is all-ones in two's complement.
			let neg_one = builder.add_constant(Word::ALL_ONE);
			let (hi, lo) = builder.smul(x, neg_one);
			let expected_hi = builder.add_inout();
			let expected_lo = builder.add_inout();
			builder.assert_eq("smul_hi", hi, expected_hi);
			builder.assert_eq("smul_lo", lo, expected_lo);
			let circuit = builder.build();

			let mut w = circuit.new_witness_filler();
			w[x] = Word(x_val as u64);

			// Reference: -x as a 128-bit value, split into high and low words.
			let result = -(x_val as i128);
			w[expected_hi] = Word((result >> 64) as u64);
			w[expected_lo] = Word(result as u64);
			w.circuit.populate_wire_witness(&mut w).unwrap();

			let cs = circuit.constraint_system();
			verify_constraints(cs, &w.into_value_vec()).unwrap();
		}
	}

	#[test]
	fn test_simul_constraint_verification() {
		// Invariant: negative × negative gives a positive product.
		let builder = CircuitBuilder::new();
		let x = builder.add_inout();
		let y = builder.add_inout();
		let (hi, lo) = builder.smul(x, y);
		let expected_hi = builder.add_inout();
		let expected_lo = builder.add_inout();
		builder.assert_eq("smul_hi", hi, expected_hi);
		builder.assert_eq("smul_lo", lo, expected_lo);
		let circuit = builder.build();

		// Fixture: -5 * -7 = 35, which fits entirely in the low word.
		let mut w = circuit.new_witness_filler();
		w[x] = Word(-5i64 as u64);
		w[y] = Word(-7i64 as u64);
		w[expected_hi] = Word(0);
		w[expected_lo] = Word(35);
		w.circuit.populate_wire_witness(&mut w).unwrap();

		let cs = circuit.constraint_system();
		verify_constraints(cs, &w.into_value_vec()).unwrap();
	}

	#[test]
	fn test_smul_edge_cases() {
		// Invariant: sign correction is exact at the extreme operands.
		let builder = CircuitBuilder::new();
		let x = builder.add_inout();
		let y = builder.add_inout();
		let (hi, lo) = builder.smul(x, y);
		let expected_hi = builder.add_inout();
		let expected_lo = builder.add_inout();
		builder.assert_eq("smul_hi", hi, expected_hi);
		builder.assert_eq("smul_lo", lo, expected_lo);
		let circuit = builder.build();

		// Boundary operands where the two's complement correction matters most.
		let test_cases = [
			(i64::MIN, i64::MIN),        // both at the negative extreme
			(i64::MIN, i64::MAX),        // opposite extremes
			(i64::MAX, i64::MAX),        // both at the positive extreme
			(i64::MIN, -1),              // negation overflow of the minimum
			(1i64 << 31, 1i64 << 31),    // exact power-of-two product
			(-(1i64 << 31), 1i64 << 31), // one negative operand
			(1i64 << 32, 1i64 << 31),    // product straddling bit 63
			(-(1i64 << 32), 1i64 << 31), // negative straddling bit 63
		];

		for (x_val, y_val) in test_cases {
			let mut w = circuit.new_witness_filler();
			w[x] = Word(x_val as u64);
			w[y] = Word(y_val as u64);

			// Reference: native 128-bit signed product, split into high and low words.
			let result = (x_val as i128) * (y_val as i128);
			w[expected_hi] = Word((result >> 64) as u64);
			w[expected_lo] = Word(result as u64);
			w.circuit.populate_wire_witness(&mut w).unwrap();

			let cs = circuit.constraint_system();
			verify_constraints(cs, &w.into_value_vec()).unwrap();
		}
	}
}
