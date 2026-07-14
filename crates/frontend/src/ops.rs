// Copyright 2025-2026 The Binius Developers
// Copyright 2025 Irreducible Inc.
//! Circuit gadgets: reusable sub-circuits built from primitive gates.
//!
//! A gadget emits a fixed pattern of existing gates.
//! It adds no new opcode and no bespoke constraint logic.

use binius_core::word::Word;

use crate::{CircuitBuilder, Wire};

/// Bitwise OR of two 64-bit words.
///
/// Returns `z = a | b`.
///
/// # Algorithm
///
/// OR decomposes into an AND and two XORs, bit for bit:
///
/// ```text
/// a | b = (a & b) ^ a ^ b
/// ```
///
/// A fused AND-XOR gate evaluates `(a & b) ^ w` inside a single AND constraint.
/// Setting `w = a ^ b` yields the OR with no additional constraint.
///
/// # Cost
///
/// 1 AND constraint.
/// The XOR operand folds into that constraint for free.
pub fn bor64(builder: &CircuitBuilder, a: Wire, b: Wire) -> Wire {
	// The XOR half of the identity a | b = (a & b) ^ (a ^ b).
	let a_xor_b = builder.bxor(a, b);
	// Fuse it into the AND: (a & b) ^ (a ^ b) = a | b in one constraint.
	builder.fax(a, b, a_xor_b)
}

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
/// - 1 MUL constraint for the unsigned product.
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
	/// Bitwise OR.
	///
	/// Returns `z = a | b`.
	///
	/// # Cost
	///
	/// - 1 AND constraint in the general case.
	/// - None when either operand is a constant.
	/// - None when both operands are the same wire.
	pub fn bor(&self, a: Wire, b: Wire) -> Wire {
		// a | a = a holds bit for bit.
		// Return the operand and emit no gate.
		if self.algebraic_folding() && a == b {
			return a;
		}
		// Constant-operand identities that hold bit for bit, so they need no AND constraint:
		//   c | d    -> fold to the constant c | d
		//   0 | b    -> b
		//   all-1 | b -> all-1
		match (self.const_of(a), self.const_of(b)) {
			(Some(x), Some(y)) => return self.add_constant(Word(x.0 | y.0)),
			(Some(x), _) if x == Word::ZERO => return b,
			(Some(x), _) if x == Word::ALL_ONE => return a,
			(_, Some(y)) if y == Word::ZERO => return a,
			(_, Some(y)) if y == Word::ALL_ONE => return b,
			_ => {}
		}
		// General case: neither operand is constant and they differ.
		bor64(self, a, b)
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
		fn test_bor_correctness(a_val: u64, b_val: u64) {
			// Invariant: the gadget output equals the native bitwise OR.
			let builder = CircuitBuilder::new();
			// Operands are public inputs, so no build-time folding fires.
			// This forces the general gadget path (one AND constraint) to run.
			let a = builder.add_inout();
			let b = builder.add_inout();
			// Gadget under test.
			let z = builder.bor(a, b);
			// Expected OR pinned as a public word.
			let expected = builder.add_inout();
			builder.assert_eq("bor", z, expected);
			let circuit = builder.build();

			// Assign the random operands and their OR.
			let mut w = circuit.new_witness_filler();
			w[a] = Word(a_val);
			w[b] = Word(b_val);
			w[expected] = Word(a_val | b_val);
			w.circuit.populate_wire_witness(&mut w).unwrap();

			// The single AND constraint must hold for the correct witness.
			let cs = circuit.constraint_system();
			verify_constraints(cs, &w.into_value_vec()).unwrap();
		}
	}

	#[test]
	fn test_bor_boundary_values() {
		// Invariant: the OR is exact at the bitwise extremes.
		let builder = CircuitBuilder::new();
		let a = builder.add_inout();
		let b = builder.add_inout();
		let z = builder.bor(a, b);
		let expected = builder.add_inout();
		builder.assert_eq("bor", z, expected);
		let circuit = builder.build();

		// Operand pairs that stress each bit interaction.
		let cases = [
			(0u64, 0u64),                                   // no bits set
			(0, u64::MAX),                                  // one operand full
			(u64::MAX, u64::MAX),                           // both full
			(0xAAAA_AAAA_AAAA_AAAA, 0x5555_5555_5555_5555), // disjoint alternating bits
			(0xFF00_FF00_FF00_FF00, 0x00FF_00FF_00FF_00FF), // disjoint byte lanes
		];

		for (a_val, b_val) in cases {
			let mut w = circuit.new_witness_filler();
			w[a] = Word(a_val);
			w[b] = Word(b_val);
			w[expected] = Word(a_val | b_val);
			w.circuit.populate_wire_witness(&mut w).unwrap();

			let cs = circuit.constraint_system();
			verify_constraints(cs, &w.into_value_vec()).unwrap();
		}
	}

	#[test]
	fn test_bor_folds_constants() {
		// Invariant: OR of two constants folds to a constant wire at build time.
		// No AND constraint is emitted for the fold.
		let builder = CircuitBuilder::new();
		// 0xF0F0... | 0x0F0F... covers every bit, so the fold is all-ones.
		let a = builder.add_constant_64(0xF0F0_F0F0_F0F0_F0F0);
		let b = builder.add_constant_64(0x0F0F_0F0F_0F0F_0F0F);
		let z = builder.bor(a, b);
		// The folded wire is itself a constant carrying the OR value.
		assert_eq!(builder.const_of(z), Some(Word(0xFFFF_FFFF_FFFF_FFFF)));
	}

	#[test]
	fn test_bor_identity_and_idempotence() {
		// Invariant: the zero, all-ones, and idempotent identities emit no gate.
		// Each returns one of the input wires unchanged.
		let builder = CircuitBuilder::new();
		let x = builder.add_inout();
		let zero = builder.add_constant_64(0);
		let all_one = builder.add_constant(Word::ALL_ONE);

		// 0 | x = x, in either operand order.
		assert_eq!(builder.bor(zero, x), x);
		assert_eq!(builder.bor(x, zero), x);
		// all-1 | x = all-1, in either operand order.
		assert_eq!(builder.bor(all_one, x), all_one);
		assert_eq!(builder.bor(x, all_one), all_one);
		// x | x = x.
		assert_eq!(builder.bor(x, x), x);
	}

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

			// All AND/MUL constraints must hold for the correct witness.
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
	fn test_smul_constraint_verification() {
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
