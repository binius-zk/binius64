// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers
use std::collections::HashSet;

use binius_core::{
	constraint_system::{Operand, ShiftedValueIndex, WitnessIndex, WitnessSegment},
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
		w[a] = Word(rng.random::<u64>());
		w[b] = Word(rng.random::<u64>());
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
		w[a] = Word(rng.random::<u64>());
		w[b] = Word(rng.random::<u64>());
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
/// inlined into the assertion's operand; the assertion itself emits an AND constraint either way.
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

	// The `bxor` linear constraint moves to the ZERO set, leaving only the assertion's AND.
	assert_eq!(cs.n_zero_constraints(), 1);
	assert_eq!(cs.n_and_constraints(), 1);

	// The Zero constraint XORs the linear constraint's terms with its destination, and names no
	// all-ones constant.
	let all_one = WitnessIndex::constant(0);
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

/// Builds `((x << 32) >> 32) & y == z` with gate fusion left on.
///
/// A left-then-right shift pair is not expressible as one shifted operand, so fusion cannot
/// inline the intermediate into the `band` and has to commit it. That committed definition lowers
/// to a ZERO constraint like any other linear constraint.
fn build_committed_lin_def_circuit() -> (Circuit, Wire, Wire, Wire) {
	let builder = CircuitBuilder::new();
	let x = builder.add_inout();
	let y = builder.add_inout();
	let z = builder.add_inout();
	let low = builder.shr(builder.shl(x, 32), 32);
	builder.assert_eq("and", builder.band(low, y), z);
	(builder.build(), x, y, z)
}

#[test]
fn test_zero_constraints_reach_a_fused_committed_lin_def() {
	let (zero_circuit, x, y, z) = build_committed_lin_def_circuit();
	let zero_cs = zero_circuit.constraint_system();

	// The committed shift pair is a ZERO constraint, and it names no all-ones constant — the AND
	// set holds only the `band` and the assertion.
	assert_eq!(zero_cs.n_zero_constraints(), 1);
	assert_eq!(zero_cs.n_and_constraints(), 2);
	assert!(
		zero_cs.zero_constraints[0]
			.val()
			.iter()
			.all(|svi| svi.value_index != WitnessIndex::constant(0))
	);

	let mut filler = zero_circuit.new_witness_filler();
	filler[x] = Word(0x1234_5678_9abc_def0);
	filler[y] = Word(0x0fed_cba9_8765_4321);
	filler[z] = Word(0x9abc_def0 & 0x0fed_cba9_8765_4321);
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
	use crate::compiler::hints::Hint;

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

	let pooled = CircuitBuilder::new();
	pooled.enable_scratch_pooling();
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
	assert_eq!(unpooled_layout.offset_inout, pooled_layout.offset_inout);
	assert_eq!(unpooled_layout.offset_witness, pooled_layout.offset_witness);
	assert_eq!(unpooled_layout.n_hidden_words, pooled_layout.n_hidden_words);
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
	let builder = CircuitBuilder::new();
	builder.enable_scratch_pooling();
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
	let builder = CircuitBuilder::new();
	builder.enable_scratch_pooling();
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
					index.segment() != WitnessSegment::Constant
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
