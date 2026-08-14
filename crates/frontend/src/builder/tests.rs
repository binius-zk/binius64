// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers
use std::collections::HashSet;

use binius_core::{
	constraint_system::{Operand, ShiftedValueIndex, ValueIndex, ValueSegment},
	word::Word,
};
use binius_utils::strided_array::StridedArray2DViewMut;
use proptest::prelude::*;
use rand::{RngExt, SeedableRng, rngs::StdRng};

use super::*;

#[test]
fn test_icmp_ult() {
	// Build a circuit with only two inputs and check c = a < b.
	let builder = CircuitBuilder::new();
	let a = builder.add_inout();
	let b = builder.add_inout();
	let actual = builder.icmp_ult(a, b);
	let expected = builder.add_inout();
	builder.assert_false("lt", builder.bxor(actual, expected));
	let circuit = builder.build();

	// check that it actually works.
	let mut rng = StdRng::seed_from_u64(42);
	for _ in 0..10000 {
		let mut w = circuit.new_witness_filler();
		w[a] = Word(rng.random());
		w[b] = Word(rng.random());
		w[expected] = Word(if w[a].0 < w[b].0 { u64::MAX } else { 0 });
		w.circuit.populate_wire_witness(&mut w).unwrap();
	}
}

#[test]
fn test_icmp_eq() {
	// Build a circuit with only two inputs and check c = a == b.
	let builder = CircuitBuilder::new();
	let a = builder.add_inout();
	let b = builder.add_inout();
	let actual = builder.icmp_eq(a, b);
	let expected = builder.add_inout();
	builder.assert_false("eq", builder.bxor(actual, expected));
	let circuit = builder.build();

	// check that it actually works.
	let mut rng = StdRng::seed_from_u64(42);
	for _ in 0..10000 {
		let mut w = circuit.new_witness_filler();
		w[a] = Word(rng.random());
		w[b] = Word(rng.random());
		w[expected] = Word(if w[a].0 == w[b].0 { u64::MAX } else { 0 });
		w.circuit.populate_wire_witness(&mut w).unwrap();
	}
}

#[test]
fn test_algebraic_folds_return_operand_directly() {
	// Idempotent and self-inverse identities on equal wires fold at build time.
	let builder = CircuitBuilder::new();
	let x = builder.add_witness();
	let cond = builder.add_witness();

	// x & x = x, x | x = x, and select(_, t, t) = t all return the operand wire itself.
	assert_eq!(builder.band(x, x), x);
	assert_eq!(builder.bor(x, x), x);
	assert_eq!(builder.select(cond, x, x), x);

	// x ^ x = 0 returns the interned zero constant.
	assert_eq!(builder.bxor(x, x), builder.add_constant(Word::ZERO));
}

#[test]
fn test_algebraic_fold_bxor_self_is_zero_in_witness() {
	// The folded x ^ x wire must carry 0 for any x, and the circuit must still verify.
	let builder = CircuitBuilder::new();
	let x = builder.add_inout();
	let zero = builder.bxor(x, x);
	let circuit = builder.build();

	let mut w = circuit.new_witness_filler();
	w[x] = Word(0x1234_5678_9abc_def0);
	circuit.populate_wire_witness(&mut w).unwrap();

	assert_eq!(w[zero], Word::ZERO);
	circuit.constraint_system().verify(&w.value_vec).unwrap();
}

/// Builds `assert_eq(x ^ y, z)`.
///
/// Gate fusion is off so that the `bxor` gate keeps its own linear constraint instead of being
/// inlined into the assertion's operand.
fn build_xor_circuit() -> (Circuit, Wire, Wire, Wire) {
	let builder = CircuitBuilder::with_opts(Options {
		enable_gate_fusion: false,
		..Options::default()
	});
	let x = builder.add_inout();
	let y = builder.add_inout();
	let z = builder.add_inout();
	builder.assert_eq("xor", builder.bxor(x, y), z);
	(builder.build(), x, y, z)
}

#[test]
fn test_linear_constraints_lower_to_zero_constraints() {
	let (zero_circuit, x, y, z) = build_xor_circuit();
	let cs = zero_circuit.constraint_system();

	// Both the `bxor` linear constraint and the assertion land in the ZERO set, so the AND set is
	// empty.
	assert_eq!(cs.n_zero_constraints(), 2);
	assert_eq!(cs.n_and_constraints(), 0);

	// The Zero constraint XORs the linear constraint's terms with its destination, and names no
	// all-ones constant.
	let all_one = ValueIndex::constant(0);
	let val = cs.zero_constraints[0].val();
	assert_eq!(val.len(), 3);
	assert!(val.iter().all(|svi| svi.value_index != all_one));

	let mut filler = zero_circuit.new_witness_filler();
	filler[x] = Word(0x1234_5678_9abc_def0);
	filler[y] = Word(0x0fed_cba9_8765_4321);
	filler[z] = Word(0x1234_5678_9abc_def0 ^ 0x0fed_cba9_8765_4321);
	zero_circuit.populate_wire_witness(&mut filler).unwrap();
	cs.verify(&filler.value_vec).unwrap();
}

/// Builds `sar(rotr(x << 3, 5), 7) & y == z` with gate fusion left on.
///
/// A term carries two shifts, and none of these three collapse into another, so the chain does not
/// fit however it is grouped. Fusion therefore cannot inline the intermediate into the `band` and
/// has to commit it. That committed definition lowers to a ZERO constraint like any other linear
/// constraint.
///
/// Two shifts would not do it: `(x << 32) >> 32` is one term now, which is the point of carrying a
/// sequence.
fn build_committed_lin_def_circuit() -> (Circuit, Wire, Wire, Wire) {
	let builder = CircuitBuilder::new();
	let x = builder.add_inout();
	let y = builder.add_inout();
	let z = builder.add_inout();
	let chained = builder.sar(builder.rotr(builder.shl(x, 3), 5), 7);
	builder.assert_eq("and", builder.band(chained, y), z);
	(builder.build(), x, y, z)
}

#[test]
fn test_zero_constraints_reach_a_fused_committed_lin_def() {
	let (zero_circuit, x, y, z) = build_committed_lin_def_circuit();
	let zero_cs = zero_circuit.constraint_system();

	// The committed shift pair is a ZERO constraint, and it names no all-ones constant — the AND
	// set holds only the `band`, since the assertion is a ZERO constraint too.
	assert_eq!(zero_cs.n_zero_constraints(), 2);
	assert_eq!(zero_cs.n_and_constraints(), 1);
	assert!(
		zero_cs.zero_constraints[0]
			.val()
			.iter()
			.all(|svi| svi.value_index != ValueIndex::constant(0))
	);

	let mut filler = zero_circuit.new_witness_filler();
	let (x_val, y_val) = (0x1234_5678_9abc_def0u64, 0x0fed_cba9_8765_4321u64);
	filler[x] = Word(x_val);
	filler[y] = Word(y_val);
	filler[z] = Word(((x_val << 3).rotate_right(5) as i64 >> 7) as u64 & y_val);
	zero_circuit.populate_wire_witness(&mut filler).unwrap();
	zero_cs.verify(&filler.value_vec).unwrap();
}

#[test]
fn test_iadd_cin_cout_max_values() {
	let builder = CircuitBuilder::new();

	let a = builder.add_constant_64(0xFFFFFFFFFFFFFFFF);
	let b = builder.add_constant_64(0xFFFFFFFFFFFFFFFF);
	let cin_wire = builder.add_constant(Word::ZERO);
	let (sum_wire, cout_wire) = builder.iadd_cin_cout(a, b, cin_wire);

	let circuit = builder.build();
	let mut w = circuit.new_witness_filler();
	circuit.populate_wire_witness(&mut w).unwrap();

	assert_eq!(w[sum_wire], Word(0xFFFFFFFFFFFFFFFE));
	assert_eq!(w[cout_wire], Word(0xFFFFFFFFFFFFFFFF));
}

#[test]
fn test_iadd_cin_cout_zero() {
	let builder = CircuitBuilder::new();

	let a = builder.add_constant_64(0);
	let b = builder.add_constant_64(0);
	let cin_wire = builder.add_constant(Word::ZERO);
	let (sum_wire, cout_wire) = builder.iadd_cin_cout(a, b, cin_wire);

	let circuit = builder.build();
	let mut w = circuit.new_witness_filler();
	circuit.populate_wire_witness(&mut w).unwrap();

	assert_eq!(w[sum_wire], Word(0));
	assert_eq!(w[cout_wire], Word(0));
}

#[test]
fn test_isub_bin_bout_from_zero() {
	let builder = CircuitBuilder::new();

	let a = builder.add_constant_64(0);
	let b = builder.add_constant_64(u64::MAX);
	let bin_wire = builder.add_constant(Word::ONE << 63);
	let (diff_wire, bout_wire) = builder.isub_bin_bout(a, b, bin_wire);

	let circuit = builder.build();
	let mut w = circuit.new_witness_filler();
	circuit.populate_wire_witness(&mut w).unwrap();

	assert_eq!(w[diff_wire], Word(0));
	assert_eq!(w[bout_wire], Word(u64::MAX));
}

#[test]
fn test_all_one_is_first_constant() {
	// The gate graph seeds the all-one word as its first constant at construction.
	// So every built circuit exposes it at constant index 0, ahead of any user constant.
	let builder = CircuitBuilder::new();
	// A user constant added first still does not displace the seeded all-one word.
	builder.add_constant_64(0x1234);
	let circuit = builder.build();

	let constants = &circuit.constraint_system().constants;
	assert_eq!(constants[0], Word::ALL_ONE);
}

#[test]
fn test_call_hint_user_registered() {
	use crate::ir::hints::Hint;

	/// User-defined hint that XORs all of its inputs into a single output word.
	struct XorAllHint;

	impl Hint for XorAllHint {
		const NAME: &'static str = "test::xor_all";

		fn shape(&self, dimensions: &[usize]) -> (usize, usize) {
			let [n_in] = dimensions else {
				panic!("XorAllHint requires 1 dimension");
			};
			(*n_in, 1)
		}

		fn execute(&self, _dimensions: &[usize], inputs: &[Word], outputs: &mut [Word]) {
			let acc = inputs.iter().fold(0u64, |a, w| a ^ w.0);
			outputs[0] = Word(acc);
		}
	}

	let builder = CircuitBuilder::new();
	let inputs = [
		builder.add_constant_64(0xdead_beef_0000_0000),
		builder.add_constant_64(0x0000_0000_cafe_babe),
		builder.add_constant_64(0xffff_ffff_ffff_ffff),
	];

	// Calling twice with the same hint type should reuse the same registry entry.
	let out1 = builder.call_hint(XorAllHint, &[inputs.len()], &inputs);
	let out2 = builder.call_hint(XorAllHint, &[inputs.len()], &inputs);
	assert_eq!(out1.len(), 1);
	assert_eq!(out2.len(), 1);

	let circuit = builder.build();
	let mut w = circuit.new_witness_filler();
	circuit.populate_wire_witness(&mut w).unwrap();

	let expected = Word(0xdead_beef_0000_0000 ^ 0x0000_0000_cafe_babe ^ 0xffff_ffff_ffff_ffff);
	assert_eq!(w[out1[0]], expected);
	assert_eq!(w[out2[0]], expected);
}

fn prop_check_icmp_ult(a: u64, b: u64, expected_result: Word) {
	let builder = CircuitBuilder::new();
	let a_wire = builder.add_constant_64(a);
	let b_wire = builder.add_constant_64(b);
	let result_wire = builder.icmp_ult(a_wire, b_wire);

	let circuit = builder.build();
	let mut w = circuit.new_witness_filler();
	circuit.populate_wire_witness(&mut w).unwrap();

	assert_eq!(w[result_wire] >> 63, expected_result >> 63);

	let cs = circuit.constraint_system();
	cs.verify(&w.value_vec).unwrap();
}

fn prop_check_icmp_eq(a: u64, b: u64, expected_result: Word) {
	let builder = CircuitBuilder::new();
	let a_wire = builder.add_constant_64(a);
	let b_wire = builder.add_constant_64(b);
	let result_wire = builder.icmp_eq(a_wire, b_wire);

	let circuit = builder.build();
	let mut w = circuit.new_witness_filler();
	circuit.populate_wire_witness(&mut w).unwrap();

	assert_eq!(w[result_wire] >> 63, expected_result >> 63);

	let cs = circuit.constraint_system();
	cs.verify(&w.value_vec).unwrap();
}

proptest! {
	#[test]
	fn prop_iadd_cin_cout_carry_chain(a1 in any::<u64>(), b1 in any::<u64>(), a2 in any::<u64>(), b2 in any::<u64>()) {
		let builder = CircuitBuilder::new();

		// First addition
		let a1_wire = builder.add_constant_64(a1);
		let b1_wire = builder.add_constant_64(b1);
		let cin_wire = builder.add_constant(Word::ZERO);
		let (sum1_wire, cout1_wire) = builder.iadd_cin_cout(a1_wire, b1_wire, cin_wire);

		// Second addition with carry from first
		let a2_wire = builder.add_constant_64(a2);
		let b2_wire = builder.add_constant_64(b2);
		let (sum2_wire, cout2_wire) = builder.iadd_cin_cout(a2_wire, b2_wire, cout1_wire);

		let circuit = builder.build();
		let mut w = circuit.new_witness_filler();
		circuit.populate_wire_witness(&mut w).unwrap();

		// Check first addition
		let expected_sum1 = a1.wrapping_add(b1);
		let expected_cout1 = (a1 & b1) | ((a1 ^ b1) & !expected_sum1);
		assert_eq!(w[sum1_wire], Word(expected_sum1));
		assert_eq!(w[cout1_wire], Word(expected_cout1));

		// Check second addition with carry
		// Extract MSB of cout1 as the carry-in bit
		let cin2 = expected_cout1 >> 63;
		let expected_sum2 = a2.wrapping_add(b2).wrapping_add(cin2);
		let expected_cout2 = (a2 & b2) | ((a2 ^ b2) & !expected_sum2);
		assert_eq!(w[sum2_wire], Word(expected_sum2));
		assert_eq!(w[cout2_wire], Word(expected_cout2));

		let cs = circuit.constraint_system();
		cs.verify(&w.value_vec).unwrap();
	}

	#[test]
	fn prop_icmp_ult_gte(a in any::<u64>(), b in any::<u64>()) {
		prop_assume!(a >= b);
		prop_check_icmp_ult(a, b, Word::ZERO);
	}

	#[test]
	fn prop_icmp_ult_lt(a in any::<u64>(), b in any::<u64>()) {
		prop_assume!(a < b);
		prop_check_icmp_ult(a, b, Word::ALL_ONE);
	}

	#[test]
	fn prop_check_assert_eq(x in any::<u64>(), y in any::<u64>()) {
		let builder = CircuitBuilder::new();
		let is_equal = x == y;
		let x_wire = builder.add_inout();
		let y_wire = builder.add_inout();
		builder.assert_eq("eq", x_wire, y_wire);

		let circuit = builder.build();
		let mut w = circuit.new_witness_filler();

		w[x_wire] = Word(x);
		w[y_wire] = Word(y);
		let result = circuit.populate_wire_witness(&mut w);

		if is_equal {
			// When values are equal, witness population should succeed
			assert!(result.is_ok());
			// And constraints should verify
			let cs = circuit.constraint_system();
			cs.verify(&w.value_vec).unwrap();
		} else {
			// When values are not equal, witness population should fail
			assert!(result.is_err());
		}
	}

	#[test]
	fn prop_icmp_eq_equal(a in any::<u64>()) {
		prop_check_icmp_eq(a, a, Word::ALL_ONE);
	}

	#[test]
	fn prop_icmp_eq_not_equal(a in any::<u64>(), b in any::<u64>()) {
		prop_assume!(a != b);
		prop_check_icmp_eq(a, b, Word::ZERO);
	}
}

#[test]
fn test_bxor_linear_constraint() {
	// Test that bxor operation internally uses linear constraints
	// which are then expanded to AND constraints with all_one
	let builder = CircuitBuilder::new();

	let a = builder.add_inout();
	let b = builder.add_inout();

	// bxor internally creates a linear constraint
	let c = builder.bxor(a, b);

	let circuit = builder.build();

	// Verify the circuit builds successfully and bxor works correctly
	let mut w = circuit.new_witness_filler();
	w[a] = Word(0x123456789abcdef0);
	w[b] = Word(0xfedcba9876543210);

	circuit.populate_wire_witness(&mut w).unwrap();

	// Verify result is correct
	assert_eq!(w[c], Word(0x123456789abcdef0 ^ 0xfedcba9876543210));

	// Verify constraints are satisfied
	let cs = circuit.constraint_system();
	cs.verify(&w.value_vec).unwrap();
}

#[test]
fn test_shift_operations_with_linear_constraints() {
	// Test that shift operations (shl, shr, sar) work correctly
	// These operations internally use linear constraints
	let builder = CircuitBuilder::new();

	let a = builder.add_inout();
	let b = builder.add_inout();

	// Test shift left
	let shl_result = builder.shl(a, 8);
	// Test shift right
	let shr_result = builder.shr(b, 16);
	// Combine with XOR
	let combined = builder.bxor(shl_result, shr_result);

	let circuit = builder.build();

	// Test with specific values
	let mut w = circuit.new_witness_filler();
	w[a] = Word(0xff00ff00ff00ff00);
	w[b] = Word(0x0000abcd0000ef12);

	circuit.populate_wire_witness(&mut w).unwrap();

	// Verify results
	assert_eq!(w[shl_result], Word(0xff00ff00ff00ff00 << 8));
	assert_eq!(w[shr_result], Word(0x0000abcd0000ef12 >> 16));
	assert_eq!(w[combined], Word((0xff00ff00ff00ff00 << 8) ^ (0x0000abcd0000ef12 >> 16)));

	// Verify constraints are satisfied
	let cs = circuit.constraint_system();
	cs.verify(&w.value_vec).unwrap();
}

#[test]
fn test_32bit_half_shift_operations() {
	let builder = CircuitBuilder::new();

	let a = builder.add_inout();
	let sll32_result = builder.sll32(a, 4);
	let srl32_result = builder.srl32(a, 4);
	let sra32_result = builder.sra32(a, 4);
	let rotr32_result = builder.rotr32(a, 4);

	let circuit = builder.build();

	let input = 0x12345678_89abcdef_u64;
	let mut w = circuit.new_witness_filler();
	w[a] = Word(input);

	circuit.populate_wire_witness(&mut w).unwrap();

	let expected_sll32 = Word(input).sll32(4);
	let expected_srl32 = Word(input).srl32(4);
	let expected_sra32 = Word(input).sra32(4);
	let expected_rotr32 = Word(input).rotr32(4);

	assert_eq!(w[sll32_result], expected_sll32);
	assert_eq!(w[srl32_result], expected_srl32);
	assert_eq!(w[sra32_result], expected_sra32);
	assert_eq!(w[rotr32_result], expected_rotr32);

	// These are lane-local operations, so they should differ from the plain 64-bit shifts
	// for inputs where bits would otherwise cross the 32-bit boundary.
	assert_ne!(w[sll32_result], Word(input << 4));
	assert_ne!(w[srl32_result], Word(input >> 4));
	assert_ne!(w[sra32_result], Word(((input as i64) >> 4) as u64));
	assert_ne!(w[rotr32_result], Word(input.rotate_right(4)));

	let cs = circuit.constraint_system();
	cs.verify(&w.value_vec).unwrap();
}

#[test]
fn test_rotr_operation_expansion() {
	// Test that rotr operation correctly expands to (srl XOR sll)
	// This tests the expansion logic in constraint_builder.rs
	let builder = CircuitBuilder::new();

	let a = builder.add_inout();
	let b = builder.add_inout();

	// rotr internally expands to: (a >> 12) XOR (a << 52)
	let rotr_result = builder.rotr(a, 12);
	let combined = builder.bxor(rotr_result, b);

	let circuit = builder.build();

	// Test with specific values
	let mut w = circuit.new_witness_filler();
	w[a] = Word(0xabcdef1234567890);
	w[b] = Word(0x1111111111111111);

	circuit.populate_wire_witness(&mut w).unwrap();

	// Verify rotr works correctly: rotr(a, 12)
	let expected_rotr = 0xabcdef1234567890u64.rotate_right(12);
	assert_eq!(w[rotr_result], Word(expected_rotr));
	assert_eq!(w[combined], Word(expected_rotr ^ 0x1111111111111111));

	// Verify constraints are satisfied
	let cs = circuit.constraint_system();
	cs.verify(&w.value_vec).unwrap();
}

#[test]
fn test_multiple_xor_operations() {
	// Test multiple XOR operations that internally use linear constraints
	let builder = CircuitBuilder::new();

	let a = builder.add_inout();
	let b = builder.add_inout();
	let c = builder.add_inout();
	let d = builder.add_inout();

	// Multiple XOR operations, each creating linear constraints
	let result1 = builder.bxor(a, b);
	let result2 = builder.bxor(c, d);
	// Chain XOR operations
	let final_result = builder.bxor(result1, result2);

	let circuit = builder.build();

	// Test with specific values
	let mut w = circuit.new_witness_filler();
	w[a] = Word(0xaaaaaaaaaaaaaaaa);
	w[b] = Word(0x5555555555555555);
	w[c] = Word(0x0f0f0f0f0f0f0f0f);
	w[d] = Word(0xf0f0f0f0f0f0f0f0);

	circuit.populate_wire_witness(&mut w).unwrap();

	// Verify intermediate results
	assert_eq!(w[result1], Word(0xaaaaaaaaaaaaaaaa ^ 0x5555555555555555));
	assert_eq!(w[result2], Word(0x0f0f0f0f0f0f0f0f ^ 0xf0f0f0f0f0f0f0f0));
	assert_eq!(w[final_result], Word(w[result1].0 ^ w[result2].0));

	// Verify constraints are satisfied
	let cs = circuit.constraint_system();
	cs.verify(&w.value_vec).unwrap();
}

#[test]
fn test_linear_constraint_conversion_to_zero() {
	// This test verifies that linear constraints (created by XOR/shift operations)
	// are properly converted to AND constraints during circuit building.
	// The conversion happens in constraint_builder.rs build() method.

	let builder = CircuitBuilder::new();

	// Create a circuit with various operations that generate linear constraints
	let a = builder.add_inout();
	let b = builder.add_inout();

	// These operations create linear constraints internally:
	let xor_result = builder.bxor(a, b);
	let shift_left = builder.shl(a, 5);
	let shift_right = builder.shr(b, 10);
	let sar_result = builder.sar(a, 3);
	let rotr_result = builder.rotr(b, 7);

	// Combine some results
	let combined1 = builder.bxor(shift_left, shift_right);
	let combined2 = builder.bxor(sar_result, rotr_result);
	let final_result = builder.bxor(combined1, combined2);

	// Pin the result as committed so its linear cone survives dead-code elimination.
	// A computation read by nothing is otherwise dropped, leaving no constraint to check.
	builder.force_commit(final_result);

	let circuit = builder.build();

	// Get the constraint system which should have ZERO constraints
	// (linear constraints were converted to ZERO constraints)
	let cs = circuit.constraint_system();

	// The circuit should have ZERO constraints but no separate linear constraints
	// (they were all converted during build)
	assert!(
		!cs.zero_constraints.is_empty(),
		"Should have ZERO constraints from converted linear constraints"
	);

	// Test with values to ensure correctness
	let mut w = circuit.new_witness_filler();
	w[a] = Word(0xdeadbeefcafe1234);
	w[b] = Word(0x1234567890abcdef);

	circuit.populate_wire_witness(&mut w).unwrap();

	// Verify all operations computed correctly
	assert_eq!(w[xor_result], Word(0xdeadbeefcafe1234 ^ 0x1234567890abcdef));
	assert_eq!(w[shift_left], Word(0xdeadbeefcafe1234 << 5));
	assert_eq!(w[shift_right], Word(0x1234567890abcdef >> 10));
	assert_eq!(w[sar_result], Word(((0xdeadbeefcafe1234u64 as i64) >> 3) as u64));
	assert_eq!(w[rotr_result], Word(0x1234567890abcdef_u64.rotate_right(7)));

	// Verify final results
	assert_eq!(w[combined1], Word(w[shift_left].0 ^ w[shift_right].0));
	assert_eq!(w[combined2], Word(w[sar_result].0 ^ w[rotr_result].0));
	assert_eq!(w[final_result], Word(w[combined1].0 ^ w[combined2].0));

	// Verify all constraints are satisfied
	cs.verify(&w.value_vec).unwrap();
}

proptest! {
	#[test]
	fn prop_xor_operations_with_shifts(a: u64, b: u64, shift1: u32, shift2: u32) {
		// Limit shifts to 0-63
		let shift1 = shift1 % 64;
		let shift2 = shift2 % 64;

		// Test that XOR operations with shifts work correctly
		let builder = CircuitBuilder::new();

		let wire_a = builder.add_constant_64(a);
		let wire_b = builder.add_constant_64(b);

		// Create shifted values
		let shifted_a = builder.shl(wire_a, shift1);
		let shifted_b = builder.shr(wire_b, shift2);

		// XOR the shifted values
		let result = builder.bxor(shifted_a, shifted_b);

		let circuit = builder.build();
		let mut w = circuit.new_witness_filler();
		circuit.populate_wire_witness(&mut w).unwrap();

		// Verify the result is computed correctly
		let expected = (a << shift1) ^ (b >> shift2);
		assert_eq!(w[result], Word(expected));

		// Verify constraints are satisfied
		let cs = circuit.constraint_system();
		cs.verify(&w.value_vec).unwrap();
	}

	#[test]
	fn prop_rotr_operation(value: u64, shift: u32) {
		// Limit shift to 0-63
		let shift = shift % 64;

		// Test that rotr operation works correctly
		let builder = CircuitBuilder::new();

		let wire_value = builder.add_constant_64(value);
		let rotr_result = builder.rotr(wire_value, shift);

		let circuit = builder.build();
		let mut w = circuit.new_witness_filler();
		circuit.populate_wire_witness(&mut w).unwrap();

		// Verify rotr is computed correctly
		let expected = value.rotate_right(shift);
		assert_eq!(w[rotr_result], Word(expected));

		// Verify constraints are satisfied
		let cs = circuit.constraint_system();
		cs.verify(&w.value_vec).unwrap();
	}
}

/// Rotate distance used by one round of the fixture chain below.
///
/// Staying in the range 1 to 63 keeps every round a real rotation rather than the identity.
const fn chain_rot(i: u32) -> u32 {
	i % 63 + 1
}

/// Left-shift distance used by one round of the fixture chain below.
///
/// Staying in the range 1 to 31 keeps every round a real shift and never discards the whole word.
const fn chain_shl(i: u32) -> u32 {
	i % 31 + 1
}

/// Rounds in the fixture chain.
///
/// Long enough that many temporaries are written and die before the end.
/// That is what makes slot sharing observable.
const CHAIN_ROUNDS: u32 = 48;

/// The default pass set with scratch-slot sharing turned on.
fn pooled_opts() -> Options {
	Options {
		enable_scratch_pooling: true,
		..Options::default()
	}
}

/// Builds a chain of rotates, shifts and exclusive-ors, pinned by an equality assertion.
///
/// Every intermediate result is linear, so gate fusion inlines it and leaves it uncommitted.
/// Each one dies a gate or two after it is written, which is the shape slot sharing exploits.
///
/// # Returns
///
/// The input value, and the value holding the expected result.
fn build_chain(builder: &CircuitBuilder) -> (Wire, Wire) {
	// Both ends are public, so they stay committed under either layout policy.
	let x = builder.add_inout();
	let expected = builder.add_inout();

	// Each round reads the running value twice and produces a new one.
	// The two operands are alive together while the previous value is already dead.
	let mut acc = x;
	for i in 0..CHAIN_ROUNDS {
		let r = builder.rotr(acc, chain_rot(i));
		let s = builder.shl(acc, chain_shl(i));
		acc = builder.bxor(r, s);
	}

	// The assertion anchors the chain, without which dead-code elimination would drop all of it.
	builder.assert_eq("chain", acc, expected);
	(x, expected)
}

/// Evaluates natively what the fixture chain computes, as an independent reference.
fn chain_reference(x: u64) -> u64 {
	let mut acc = x;
	// Mirror the circuit round for round, using the same two distance schedules.
	for i in 0..CHAIN_ROUNDS {
		acc = acc.rotate_right(chain_rot(i)) ^ (acc << chain_shl(i));
	}
	acc
}

#[test]
fn test_scratch_pooling_preserves_the_committed_witness() {
	// Invariant: sharing slots changes where uncommitted values are stored, nothing else.
	// The constraint system and every committed word must come out identical.
	//
	// Fixture state: the same 48-round chain compiled twice, once per layout policy.
	//
	//   unpooled:  one slot per uncommitted value
	//   pooled:    slots reused once a value's last reader has run
	let unpooled = CircuitBuilder::new();
	let (x_unpooled, expected_unpooled) = build_chain(&unpooled);
	let unpooled = unpooled.build();

	let pooled = CircuitBuilder::with_opts(pooled_opts());
	let (x_pooled, expected_pooled) = build_chain(&pooled);
	let pooled = pooled.build();

	// The fixture has to produce values that can actually share, or everything below is vacuous.
	let unpooled_layout = unpooled.value_vec_layout();
	let pooled_layout = pooled.value_vec_layout();
	assert!(
		pooled_layout.n_scratch < unpooled_layout.n_scratch,
		"pooling should shrink the scratch segment, got {} vs {}",
		pooled_layout.n_scratch,
		unpooled_layout.n_scratch
	);
	// Under sharing the segment is exactly the peak, since that is what the layout targets.
	assert_eq!(pooled_layout.n_scratch, pooled.scratch_peak_live());
	// The peak describes the graph, not the policy, so both builds must report the same figure.
	assert_eq!(unpooled.scratch_peak_live(), pooled.scratch_peak_live());

	// Every other part of the layout has to be untouched.
	// An uncommitted value appears in no constraint operand, so the proof cannot see it.
	assert_eq!(unpooled_layout.n_const, pooled_layout.n_const);
	assert_eq!(unpooled_layout.n_inout, pooled_layout.n_inout);
	assert_eq!(unpooled_layout.n_witness, pooled_layout.n_witness);
	assert_eq!(unpooled_layout.n_internal, pooled_layout.n_internal);
	assert_eq!(unpooled.constraint_system().constants, pooled.constraint_system().constants);

	// Flatten every operand of every constraint into one ordered list.
	// Comparing the lists checks the contents, the ordering and the counts in a single assertion.
	let operands = |cs: &ConstraintSystem| -> Vec<Vec<ShiftedValueIndex>> {
		chain!(
			cs.and_constraints.iter().flat_map(|c| c.0.iter()),
			cs.imul_constraints.iter().flat_map(|c| c.0.iter()),
			cs.bmul_constraints.iter().flat_map(|c| c.0.iter()),
		)
		.cloned()
		.collect()
	};
	assert_eq!(operands(pooled.constraint_system()), operands(unpooled.constraint_system()));

	// Boundary inputs: all zeros, the lowest bit, all ones, a mixed pattern, the sign bit.
	// Together they exercise the rotate and shift schedules across every bit position.
	for x_val in [
		0u64,
		1,
		u64::MAX,
		0x0123_4567_89ab_cdef,
		0x8000_0000_0000_0000,
	] {
		// Fill both builds with the same input and the same independently computed expectation.
		let mut w_unpooled = unpooled.new_witness_filler();
		w_unpooled[x_unpooled] = Word(x_val);
		w_unpooled[expected_unpooled] = Word(chain_reference(x_val));
		unpooled.populate_wire_witness(&mut w_unpooled).unwrap();

		let mut w_pooled = pooled.new_witness_filler();
		w_pooled[x_pooled] = Word(x_val);
		w_pooled[expected_pooled] = Word(chain_reference(x_val));
		pooled.populate_wire_witness(&mut w_pooled).unwrap();

		// The committed prefix is what the proof is built from, so it must agree word for word.
		assert_eq!(
			w_pooled.value_vec().combined_witness(),
			w_unpooled.value_vec().combined_witness(),
			"committed witness differs for x = {x_val:#018x}"
		);

		// Both assignments must still satisfy every constraint, not merely match each other.
		unpooled
			.constraint_system()
			.verify(&w_unpooled.value_vec)
			.unwrap();
		pooled
			.constraint_system()
			.verify(&w_pooled.value_vec)
			.unwrap();
	}
}

#[test]
fn test_scratch_pooling_rejects_a_bad_assignment() {
	// Invariant: reusing storage must not weaken what the circuit enforces.
	// A wrong assignment is rejected exactly as it would be under one slot per value.
	//
	// Fixture state: the 48-round chain, compiled with slots shared.
	let builder = CircuitBuilder::with_opts(pooled_opts());
	let (x, expected) = build_chain(&builder);
	let circuit = builder.build();

	// Mutation: flip the lowest bit of the correct result, the smallest possible perturbation.
	//
	//   input:    0x1234_5678_9abc_def0
	//   expected: correct result ^ 1     -> the equality assertion cannot hold
	let mut w = circuit.new_witness_filler();
	w[x] = Word(0x1234_5678_9abc_def0);
	w[expected] = Word(chain_reference(0x1234_5678_9abc_def0) ^ 1);

	let err = circuit
		.populate_wire_witness(&mut w)
		.expect_err("a perturbed expected value must fail the chain assertion");
	// Exactly one assertion exists in the fixture, so exactly one failure is reported.
	assert_eq!(err.total, 1);
	assert_eq!(err.failures.len(), 1);
	// The failure names the path of the assertion that failed, apart from the detail.
	assert_eq!(err.failures[0].path, ".chain");
	assert!(!err.failures[0].detail.is_empty());
}

/// Compiles one fixture under one option, returning the resulting statistics.
///
/// `opts` names the flag under test; every other flag stays at its default.
fn stat_with(opts: Options, build: impl FnOnce(&CircuitBuilder)) -> crate::CircuitStat {
	let builder = CircuitBuilder::with_opts(opts);
	build(&builder);
	let circuit = builder.build();
	crate::CircuitStat::collect(&circuit)
}

#[test]
fn constant_propagation_flag_is_honoured() {
	// Invariant: the flag decides whether an all-constant gate folds away or stays a gate.
	//
	// A comparison never folds at build time the way `band`/`bxor` do, so the flag is the
	// only thing standing between two constants and a real AND constraint.
	let and_count = |enable| {
		stat_with(
			Options {
				enable_constant_propagation: enable,
				..Options::default()
			},
			|b| {
				let a = b.add_constant(Word(3));
				let c = b.add_constant(Word(5));
				let lt = b.icmp_ult(a, c);
				let out = b.add_inout();
				b.assert_eq("lt", lt, out);
			},
		)
		.n_and_constraints
	};
	assert_eq!(and_count(true), 0, "folded away, so no AND constraint remains");
	assert_eq!(and_count(false), 1, "left as a gate, so its AND constraint stands");
}

#[test]
fn algebraic_folding_flag_is_honoured() {
	// Invariant: `x & x = x` bit for bit, so the flag decides whether that identity is applied
	// at build time instead of spending an AND constraint on it.
	let and_count = |enable| {
		stat_with(
			Options {
				enable_algebraic_folding: enable,
				..Options::default()
			},
			|b| {
				let x = b.add_inout();
				let z = b.band(x, x);
				let out = b.add_inout();
				b.assert_eq("self_and", z, out);
			},
		)
		.n_and_constraints
	};
	assert_eq!(and_count(true), 0, "the identity fires, so no gate is emitted");
	assert_eq!(and_count(false), 1, "the identity is skipped, so a real AND gate remains");
}

#[test]
fn gate_fusion_flag_is_honoured() {
	// Invariant: fusion inlines a linear definition into the AND constraint that consumes it,
	// dropping the linear constraint that would otherwise commit it on its own.
	//
	//     x --xor-- y --> lin --and-- z --> out
	//
	// With fusion, `lin`'s definition is folded into the AND's operand, leaving one constraint.
	// Without it, `lin` is committed by its own linear constraint, and the AND references it.
	let zero_count = |enable| {
		stat_with(
			Options {
				enable_gate_fusion: enable,
				..Options::default()
			},
			|b| {
				let x = b.add_inout();
				let y = b.add_inout();
				let z = b.add_inout();
				let lin = b.bxor(x, y);
				let out = b.band(lin, z);
				b.mark_inout(out);
			},
		)
		.n_zero_constraints
	};
	assert_eq!(zero_count(true), 0, "the linear step is inlined into the AND operand");
	assert_eq!(zero_count(false), 1, "the linear step is committed on its own");
}

#[test]
fn common_subexpression_elimination_flag_is_honoured() {
	// Invariant: two gates over the same operation and operands compute the same value, so one
	// of them is redundant. The flag decides whether the pass notices.
	//
	// Dead-code elimination is turned off so its own removal cannot be mistaken for this one.
	let and_count = |enable| {
		stat_with(
			Options {
				enable_common_subexpression_elimination: enable,
				enable_dead_code_elimination: false,
				..Options::default()
			},
			|b| {
				let x = b.add_inout();
				let y = b.add_inout();
				let z1 = b.band(x, y);
				let z2 = b.band(x, y);
				let out1 = b.add_inout();
				let out2 = b.add_inout();
				b.assert_eq("z1", z1, out1);
				b.assert_eq("z2", z2, out2);
			},
		)
		.n_and_constraints
	};
	assert_eq!(and_count(true), 1, "the duplicate collapses onto the first gate");
	assert_eq!(and_count(false), 2, "both gates keep their own AND constraint");
}

#[test]
fn dead_code_elimination_flag_is_honoured() {
	// Invariant: a gate nothing reads contributes no constraint, but only once the pass says so.
	let and_count = |enable| {
		stat_with(
			Options {
				enable_dead_code_elimination: enable,
				..Options::default()
			},
			|b| {
				let x = b.add_inout();
				let y = b.add_inout();
				let _unread = b.band(x, y);
			},
		)
		.n_and_constraints
	};
	assert_eq!(and_count(true), 0, "the unread gate is dropped before constraining");
	assert_eq!(and_count(false), 1, "the unread gate still constrains, unread or not");
}

#[test]
fn test_scratch_pooling_matches_scalar_per_instance_batched() {
	// Invariant: the batched fill and the one-at-a-time fill must agree for every instance.
	// This is where shared slots are most at risk.
	// One buffer holds many instances, so a reused slot is written far more often.
	//
	// Fixture state: eight instances of the 48-round chain, laid out one column each.
	//
	//   row = value index, column = instance
	//
	//     value 0  [ inst0 | inst1 | ... | inst7 ]
	//     value 1  [ inst0 | inst1 | ... | inst7 ]
	let builder = CircuitBuilder::with_opts(pooled_opts());
	let (x, expected) = build_chain(&builder);
	let circuit = builder.build();

	// The buffer spans the committed prefix plus the shared tail, one row per value index.
	let layout = circuit.value_vec_layout().clone();
	let combined = layout.combined_len();
	let full_len = combined + layout.n_scratch;
	let n = 8usize;

	// Distinct inputs per instance, so a slot leaking across columns would change a result.
	let inputs: Vec<u64> = (0..n as u64)
		.map(|i| i.wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ 0xdead_beef)
		.collect();

	// Reference: fill each instance on its own, which is the already-trusted path.
	let scalar: Vec<Vec<Word>> = inputs
		.iter()
		.map(|&x_val| {
			let mut w = circuit.new_witness_filler();
			w[x] = Word(x_val);
			w[expected] = Word(chain_reference(x_val));
			circuit.populate_wire_witness(&mut w).unwrap();
			w.value_vec().combined_witness().to_vec()
		})
		.collect();

	// Locate the two public rows, then seed every instance's column before evaluating.
	let x_row = circuit.witness_row(x);
	let expected_row = circuit.witness_row(expected);
	let mut data = vec![Word::ZERO; full_len * n];
	let mut view = StridedArray2DViewMut::without_stride(&mut data, full_len, n).unwrap();
	for (instance, &x_val) in inputs.iter().enumerate() {
		view[(x_row, instance)] = Word(x_val);
		view[(expected_row, instance)] = Word(chain_reference(x_val));
	}
	// One pass fills every instance's remaining values.
	circuit.populate_wire_witness_batched(&mut view).unwrap();

	// Compare only the committed prefix.
	// The tail beyond it is shared storage with no defined contents after evaluation.
	for instance in 0..n {
		for row in 0..combined {
			assert_eq!(
				view[(row, instance)],
				scalar[instance][row],
				"mismatch at row {row}, instance {instance}"
			);
		}
	}
}

#[test]
fn test_zero_constant_not_in_binius64_operands() {
	// Build a circuit where a zero constant is used as a gate input; after compilation
	// the zero constant term must be absent from all constraint operands.
	let builder = CircuitBuilder::new();
	let a = builder.add_inout();
	let b = builder.add_inout();
	let zero = builder.add_constant(Word::ZERO);
	let (sum, _cout) = builder.iadd_cin_cout(a, b, zero);
	let expected = builder.add_inout();
	builder.assert_false("check", builder.bxor(sum, expected));
	let circuit = builder.build();

	let cs = circuit.constraint_system();
	let constants = &cs.constants;

	let zero_const_indices: HashSet<usize> = constants
		.iter()
		.enumerate()
		.filter(|&(_, v)| *v == Word::ZERO)
		.map(|(i, _)| i)
		.collect();

	// Only a constant-segment index can name a constant, so the private and inout words are left
	// alone however their indices happen to number.
	let assert_no_zero_constants = |operands: &[Operand], kind: &str| {
		for operand in operands {
			for term in operand {
				let index = term.value_index;
				assert!(
					index.segment() != ValueSegment::Constant
						|| !zero_const_indices.contains(&(index.index() as usize)),
					"zero constant at {index:?} found in {kind} operand",
				);
			}
		}
	};

	for constraint in &cs.and_constraints {
		assert_no_zero_constants(&constraint.0, "AND");
	}
	for constraint in &cs.imul_constraints {
		assert_no_zero_constants(&constraint.0, "IMUL");
	}
	for constraint in &cs.bmul_constraints {
		assert_no_zero_constants(&constraint.0, "BMUL");
	}
}

// Promoting a gate output to a public output costs neither a second committed word nor a
// constraint, unlike declaring an inout wire and asserting the gate output against it.
//
// The three circuits below compute the same conjunction and differ only in how its result is
// exposed. `a & b` is an AND-gate output, so it occupies a committed word in every one of them.
#[test]
fn mark_inout_promotes_without_duplicating() {
	// The inout count, the private count, and the constraint count of one circuit. An equality
	// assertion is a ZERO constraint and the conjunction is an AND constraint, so the count spans
	// both sets to stay independent of which one each lands in.
	let shape = |build: &dyn Fn(&CircuitBuilder)| {
		let builder = CircuitBuilder::new();
		build(&builder);
		let circuit = builder.build();
		let layout = circuit.value_vec_layout();
		let cs = circuit.constraint_system();
		(layout.n_inout, layout.n_private(), cs.and_constraints.len() + cs.zero_constraints.len())
	};

	// The result kept alive by pinning: private, and not part of the public interface.
	let (pinned_inout, pinned_private, pinned_constraints) = shape(&|builder| {
		let a = builder.add_inout();
		let b = builder.add_inout();
		builder.force_commit(builder.band(a, b));
	});

	// The result asserted against a separately declared public output.
	let (asserted_inout, asserted_private, asserted_constraints) = shape(&|builder| {
		let a = builder.add_inout();
		let b = builder.add_inout();
		let out = builder.add_inout();
		builder.assert_eq("out", out, builder.band(a, b));
	});

	// The result promoted in place.
	let (promoted_inout, promoted_private, promoted_constraints) = shape(&|builder| {
		let a = builder.add_inout();
		let b = builder.add_inout();
		builder.mark_inout(builder.band(a, b));
	});

	// Asserting duplicates the conjunction: its own committed word, plus the declared output's,
	// plus the constraint tying them together.
	assert_eq!(asserted_inout + asserted_private, pinned_inout + pinned_private + 1);
	assert_eq!(asserted_constraints, pinned_constraints + 1);

	// Promoting costs nothing over pinning — it is the same word, relabelled public.
	assert_eq!(promoted_inout + promoted_private, pinned_inout + pinned_private);
	assert_eq!(promoted_constraints, pinned_constraints);
	assert_eq!(promoted_inout, pinned_inout + 1);
	assert_eq!(promoted_private, pinned_private - 1);
}

// A promoted wire is derived by the gate that produces it, so a filler leaves it alone and reads
// the computed value back — unlike a declared inout wire, which the filler must assign.
//
// The XOR is the case promotion has to pin: gate fusion inlines a linear definition into its
// consumers, which would leave the public word with nothing defining it.
#[test]
fn mark_inout_wire_is_derived_by_population() {
	let builder = CircuitBuilder::new();
	let a = builder.add_inout();
	let b = builder.add_inout();
	let c = builder.add_inout();
	// The XOR feeds the AND, so gate fusion has somewhere to inline it — the case promotion has
	// to pin against.
	let xor = builder.bxor(a, b);
	let and = builder.band(xor, c);
	builder.mark_inout(xor);
	builder.mark_inout(and);
	let circuit = builder.build();

	// Both promoted wires are public.
	assert_eq!(circuit.value_vec_layout().n_inout, 5);
	for wire in [xor, and] {
		assert_eq!(circuit.witness_index(wire).segment(), ValueSegment::InOut);
	}

	// Only the two inputs are assigned; the derived words are left unset.
	let mut filler = circuit.new_witness_filler();
	filler[a] = Word(0xF0F0_F0F0_F0F0_F0F0);
	filler[b] = Word(0xFF00_FF00_FF00_FF00);
	filler[c] = Word(0xFFFF_0000_FFFF_0000);
	circuit.populate_wire_witness(&mut filler).unwrap();

	assert_eq!(filler[xor], Word(0x0FF0_0FF0_0FF0_0FF0));
	assert_eq!(filler[and], Word(0x0FF0_0000_0FF0_0000));

	// Every promoted word is defined by a constraint. This is what pinning buys for the XOR: left
	// unpinned, gate fusion would inline the linear definition and leave the public word free for
	// a prover to choose.
	let cs = circuit.constraint_system();
	let operands = cs
		.zero_constraints
		.iter()
		.flat_map(|c| c.0.iter())
		.chain(cs.and_constraints.iter().flat_map(|c| c.0.iter()));
	let constrained: HashSet<ValueIndex> =
		operands.flatten().map(|term| term.value_index).collect();
	for wire in [xor, and] {
		assert!(
			constrained.contains(&circuit.witness_index(wire)),
			"a promoted word must be defined by a constraint"
		);
	}
}

// `inout()` is the reverse of `witness_index` over the inout segment, so it must agree with it
// position by position. The segment is ordered by wire creation, which for a promoted wire is when
// its gate ran rather than when it was promoted: `and` is created before `xor` is promoted, yet
// follows it here.
#[test]
fn inout_lists_the_public_wires_in_segment_order() {
	let builder = CircuitBuilder::new();
	let a = builder.add_inout();
	let b = builder.add_inout();
	let xor = builder.bxor(a, b);
	let and = builder.band(xor, a);
	builder.mark_inout(and);
	builder.mark_inout(xor);
	let circuit = builder.build();

	assert_eq!(circuit.inout(), [a, b, xor, and]);
	for (index, &wire) in circuit.inout().iter().enumerate() {
		assert_eq!(circuit.witness_index(wire), ValueIndex::inout(index as u32));
	}
}

/// A chip taking no inout words, as a whole system of its own.
fn empty_chip() -> CircuitM4 {
	CircuitM4::from(CircuitBuilder::new().build())
}

/// A one-chip system, as a leaf chip a main circuit calls once.
fn leaf_caller() -> CircuitM4 {
	let builder = CircuitBuilder::new();
	let chip = builder.add_chip(empty_chip());
	builder.call_chip(chip, &[]);
	builder.build_m4()
}

// Registering a system splices its chips in behind its main, so the IDs inside it have to move
// with them. Nesting two levels puts a call at every depth: main's calls to the two systems, each
// outer main's call to its own leaf.
//
// Calling one system twice and the other once also separates the counts the registered systems
// declared, where each leaf served one call, from the counts here.
#[test]
fn add_chip_remaps_the_ids_of_a_nested_system() {
	let builder = CircuitBuilder::new();
	let twice = builder.add_chip(leaf_caller());
	let once = builder.add_chip(leaf_caller());
	for chip in [twice, twice, once] {
		builder.call_chip(chip, &[]);
	}
	let circuit = builder.build_m4();

	// Each system takes two slots, its main then its leaf, and its call names the leaf's new slot.
	assert_eq!((twice.chip_id(), once.chip_id()), (0, 2));
	let callees = |(chip, _): &(EmbeddedCircuit, usize)| {
		chip.chip_calls
			.iter()
			.map(|call| call.chip_id)
			.collect::<Vec<_>>()
	};
	assert_eq!(
		circuit.chips.iter().map(callees).collect::<Vec<_>>(),
		[vec![1], vec![], vec![3], vec![]]
	);

	// The twice-called system's leaf serves both of its main's instances, not the one call it
	// declared before being registered.
	assert_eq!(circuit.chips.iter().map(|&(_, n)| n).collect::<Vec<_>>(), [2, 2, 1, 1]);
	circuit.validate().unwrap();
}

// `build` returns a plain circuit, which has nowhere to carry the chips its calls would name.
#[test]
#[should_panic(expected = "builds with CircuitBuilder::build_m4")]
fn build_rejects_a_builder_carrying_chips() {
	let builder = CircuitBuilder::new();
	builder.add_chip(empty_chip());
	builder.build();
}

// A call passes its words positionally against the callee's inout segment, so a call of the wrong
// length would silently shift every word past the gap.
#[test]
#[should_panic(expected = "chip #0 takes 2 inout words")]
fn call_chip_rejects_the_wrong_number_of_words() {
	let chip = CircuitBuilder::new();
	chip.assert_eq("eq", chip.add_inout(), chip.add_inout());

	let builder = CircuitBuilder::new();
	let chip = builder.add_chip(CircuitM4::from(chip.build()));
	builder.call_chip(chip, &[builder.add_witness()]);
}

// A call reads its words out of the value vector, so the wires it names have to survive the build
// as committed words. Nothing else here reads the conjunction, so the call is what keeps it.
#[test]
fn call_chip_resolves_and_pins_the_wires_it_passes() {
	let chip = CircuitBuilder::new();
	chip.assert_eq("eq", chip.add_inout(), chip.add_inout());

	let builder = CircuitBuilder::new();
	let chip_ref = builder.add_chip(CircuitM4::from(chip.build()));
	let a = builder.add_inout();
	let b = builder.add_inout();
	let and = builder.band(a, b);
	builder.call_chip(chip_ref, &[and, a]);
	let circuit = builder.build_m4();

	// The call names one word per wire, at the index the build gave that wire.
	let main = &circuit.main.circuit;
	let operands = circuit.main.chip_calls[0]
		.inout
		.iter()
		.map(|operand| operand.as_slice())
		.collect::<Vec<_>>();
	assert_eq!(
		operands,
		[
			[ShiftedValueIndex::plain(main.witness_index(and))].as_slice(),
			[ShiftedValueIndex::plain(main.witness_index(a))].as_slice(),
		]
	);

	// The conjunction is a committed private word rather than an uncommitted temporary, and the
	// AND constraint defining it survives. Nothing but the call reads it, so the call is what
	// keeps that definition — a committed word without one would be a prover's to choose.
	assert_eq!(main.witness_index(and).segment(), ValueSegment::Private);
	let cs = main.constraint_system();
	assert_eq!(cs.and_constraints.len(), 1);
	assert!(
		cs.and_constraints[0]
			.0
			.iter()
			.flatten()
			.any(|term| term.value_index == main.witness_index(and))
	);

	circuit.validate().unwrap();
}

// What a linear argument costs. `a ^ b` reaching a constraint fuses into its operand and occupies
// no word, but a chip call reads its words out of the value vector, so passing the same expression
// commits it and keeps the equation defining it.
#[test]
fn call_chip_commits_a_linear_argument() {
	// The XOR reaching a constraint fuses into its operand, so it holds no word of the value
	// vector: it stays an uncommitted temporary of the circuit.
	let builder = CircuitBuilder::new();
	let (a, b) = (builder.add_inout(), builder.add_inout());
	let xor = builder.bxor(a, b);
	builder.mark_inout(builder.band(xor, b));
	let fused = builder.build();
	assert_eq!(fused.witness_index(xor).segment(), ValueSegment::Scratch);

	// The same XOR passed to a chip call instead.
	let chip = CircuitBuilder::new();
	chip.assert_eq("eq", chip.add_inout(), chip.add_inout());

	let builder = CircuitBuilder::new();
	let chip_ref = builder.add_chip(CircuitM4::from(chip.build()));
	let (a, b) = (builder.add_inout(), builder.add_inout());
	let xor = builder.bxor(a, b);
	builder.call_chip(chip_ref, &[xor, a]);
	let called = builder.build_m4().main.circuit;

	assert_eq!(called.witness_index(xor).segment(), ValueSegment::Private);
	assert_eq!(called.value_vec_layout().n_private(), 1);
	assert_eq!(called.constraint_system().n_zero_constraints(), 1);
}

/// A gadget returning its outputs in the reverse of the order their gates ran.
///
/// The inout segment is ordered by wire creation, so a chip built from this holds the exclusive-or
/// before the conjunction while the gadget's interface holds them the other way round. A call site
/// that passed its words in interface order would name them crosswise.
struct AndThenXor;

impl Hint for AndThenXor {
	const NAME: &'static str = "test.and_then_xor";

	fn shape(&self, _dimensions: &[usize]) -> (usize, usize) {
		(2, 2)
	}

	fn execute(&self, _dimensions: &[usize], inputs: &[Word], outputs: &mut [Word]) {
		outputs[0] = Word(inputs[0].as_u64() & inputs[1].as_u64());
		outputs[1] = Word(inputs[0].as_u64() ^ inputs[1].as_u64());
	}
}

impl ChipGadget for AndThenXor {
	fn build(&self, builder: &CircuitBuilder, _dimensions: &[usize], inputs: &[Wire]) -> Vec<Wire> {
		let xor = builder.bxor(inputs[0], inputs[1]);
		let and = builder.band(inputs[0], inputs[1]);
		vec![and, xor]
	}
}

/// A gadget built out of [`AndThenXor`], which is what reaches a gadget from inside a chip body.
struct XorOfAndThenXor;

impl Hint for XorOfAndThenXor {
	const NAME: &'static str = "test.xor_of_and_then_xor";

	fn shape(&self, _dimensions: &[usize]) -> (usize, usize) {
		(2, 1)
	}

	fn execute(&self, _dimensions: &[usize], inputs: &[Word], outputs: &mut [Word]) {
		let (a, b) = (inputs[0].as_u64(), inputs[1].as_u64());
		outputs[0] = Word((a & b) ^ (a ^ b));
	}
}

impl ChipGadget for XorOfAndThenXor {
	fn build(&self, builder: &CircuitBuilder, dimensions: &[usize], inputs: &[Wire]) -> Vec<Wire> {
		let inner = builder.build_gadget(AndThenXor, dimensions, inputs);
		vec![builder.bxor(inner[0], inner[1])]
	}
}

/// The words `AndThenXor` relates, as the circuit that registers no chip computes them.
const AND_THEN_XOR_CASE: [u64; 4] = [0b1100, 0b1010, 0b1000, 0b0110];

// A builder that no chip serves builds the gadget's gates, so a circuit that never registers one
// is the circuit it was before the gadget became registrable.
#[test]
fn build_gadget_emits_gates_where_no_chip_serves_the_gadget() {
	let builder = CircuitBuilder::new();
	let (a, b) = (builder.add_inout(), builder.add_inout());
	let out = builder.build_gadget(AndThenXor, &[], &[a, b]);

	// `build` accepts this builder, which is what says no chip was registered.
	let circuit = builder.build();
	let mut w = circuit.new_witness_filler();
	w[a] = Word(AND_THEN_XOR_CASE[0]);
	w[b] = Word(AND_THEN_XOR_CASE[1]);
	circuit.populate_wire_witness(&mut w).unwrap();

	assert_eq!(w[out[0]], Word(AND_THEN_XOR_CASE[2]));
	assert_eq!(w[out[1]], Word(AND_THEN_XOR_CASE[3]));
}

// The same gadget, on a builder holding its chip. Main computes nothing: the hint hands it the
// outputs and the call is the only thing relating them to the inputs.
#[test]
fn build_gadget_calls_the_chip_serving_the_gadget() {
	let builder = CircuitBuilder::new();
	builder.register_chip(AndThenXor, &[]);
	let (a, b) = (builder.add_inout(), builder.add_inout());
	let out = builder.build_gadget(AndThenXor, &[], &[a, b]);
	let circuit = builder.build_m4();

	circuit.validate().unwrap();
	let cs = circuit.to_constraint_system();
	cs.validate().unwrap();

	// The chip holds the exclusive-or before the conjunction, so the call passes its last two
	// words the other way round from the order the gadget returns them.
	let main = &circuit.main.circuit;
	let operands = circuit.main.chip_calls[0]
		.inout
		.iter()
		.map(|operand| operand.as_slice())
		.collect::<Vec<_>>();
	assert_eq!(
		operands,
		[
			[ShiftedValueIndex::plain(main.witness_index(a))].as_slice(),
			[ShiftedValueIndex::plain(main.witness_index(b))].as_slice(),
			[ShiftedValueIndex::plain(main.witness_index(out[1]))].as_slice(),
			[ShiftedValueIndex::plain(main.witness_index(out[0]))].as_slice(),
		]
	);

	let witness = circuit
		.generate_witness(|filler| {
			filler[a] = Word(AND_THEN_XOR_CASE[0]);
			filler[b] = Word(AND_THEN_XOR_CASE[1]);
		})
		.unwrap();

	// The gadget's caller reads the same wires it would have read from the gates.
	assert_eq!(witness.main[main.witness_index(out[0])], Word(AND_THEN_XOR_CASE[2]));
	assert_eq!(witness.main[main.witness_index(out[1])], Word(AND_THEN_XOR_CASE[3]));

	// The chip instance recomputes both outputs over the words the call named, so this is what
	// holds the hint to the gates.
	witness.verify(&cs).unwrap();
}

// Registering reaches the subcircuits of the builder that registered, since they build the same
// circuit. A gadget deep in a hierarchy needs nothing threaded down to it.
#[test]
fn a_subcircuit_reaches_a_chip_the_root_builder_registered() {
	let builder = CircuitBuilder::new();
	builder.register_chip(AndThenXor, &[]);
	let (a, b) = (builder.add_inout(), builder.add_inout());
	builder
		.subcircuit("nested")
		.build_gadget(AndThenXor, &[], &[a, b]);

	let circuit = builder.build_m4();
	assert_eq!(circuit.main.chip_calls.len(), 1);
	circuit.validate().unwrap();
}

// A chip's own gates are emitted on a builder registering nothing, so a gadget inside a chip body
// lands as gates however the outer circuit builds the same gadget.
#[test]
fn a_chip_body_emits_the_gadgets_it_uses_as_gates() {
	let builder = CircuitBuilder::new();
	builder.register_chip(AndThenXor, &[]);
	builder.register_chip(XorOfAndThenXor, &[]);
	let (a, b) = (builder.add_inout(), builder.add_inout());
	builder.build_gadget(AndThenXor, &[], &[a, b]);
	builder.build_gadget(XorOfAndThenXor, &[], &[a, b]);

	let circuit = builder.build_m4();
	assert_eq!(circuit.main.chip_calls.len(), 2);
	assert!(
		circuit
			.chips
			.iter()
			.all(|(chip, _)| chip.chip_calls.is_empty())
	);
	circuit.validate().unwrap();

	let cs = circuit.to_constraint_system();
	let witness = circuit
		.generate_witness(|filler| {
			filler[a] = Word(AND_THEN_XOR_CASE[0]);
			filler[b] = Word(AND_THEN_XOR_CASE[1]);
		})
		.unwrap();
	witness.verify(&cs).unwrap();
}

// Two chips for one gadget shape would each hold a table, serving calls that could have shared one.
#[test]
#[should_panic(expected = "already a chip for dimensions []")]
fn register_chip_rejects_a_gadget_it_already_serves() {
	let builder = CircuitBuilder::new();
	builder.register_chip(AndThenXor, &[]);
	builder.register_chip(AndThenXor, &[]);
}

// A gadget's shape is what its interface is built from, so gates disagreeing with it would leave
// the chip and its call sites different sizes.
#[test]
#[should_panic(expected = "built 1 outputs, its shape declares 2")]
fn register_chip_rejects_a_gadget_disagreeing_with_its_shape() {
	struct Miscounted;

	impl Hint for Miscounted {
		const NAME: &'static str = "test.miscounted";

		fn shape(&self, _dimensions: &[usize]) -> (usize, usize) {
			(1, 2)
		}

		fn execute(&self, _dimensions: &[usize], _inputs: &[Word], outputs: &mut [Word]) {
			outputs.fill(Word::ZERO);
		}
	}

	impl ChipGadget for Miscounted {
		fn build(
			&self,
			builder: &CircuitBuilder,
			_dimensions: &[usize],
			inputs: &[Wire],
		) -> Vec<Wire> {
			vec![builder.shl(inputs[0], 1)]
		}
	}

	CircuitBuilder::new().register_chip(Miscounted, &[]);
}

// The inputs are passed positionally, so the wrong number of them would shift every word past the
// gap, whether they reach the gates or a call.
#[test]
#[should_panic(expected = "takes 2 inputs, given 1")]
fn build_gadget_rejects_the_wrong_number_of_inputs() {
	let builder = CircuitBuilder::new();
	builder.build_gadget(AndThenXor, &[], &[builder.add_inout()]);
}
