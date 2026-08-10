// Copyright 2025 Irreducible Inc.
use crate::compiler::{
	Wire,
	constraint_builder::{ConstraintBuilder, expr},
	gate_fusion::{
		Stat,
		commit_set::{self, MAX_DEPTH},
		legraph::LeGraph,
	},
};

/// Test helper to create a Wire with a given ID
fn w(id: u32) -> Wire {
	Wire::from_u32(id)
}

/// Test helper to build a simple test circuit and verify the commit set
fn test_commit_set(
	build_constraints: impl FnOnce(&mut ConstraintBuilder),
	expected_committed: &[Wire],
	expected_not_committed: &[Wire],
) {
	let mut cb = ConstraintBuilder::new();
	build_constraints(&mut cb);

	let mut stat = Stat::default();
	let mut leg = LeGraph::new(&cb);
	commit_set::run_decide_commit_set(&mut leg, &mut stat);
	let commit_set = leg.commit_set();

	// Verify expected wires are committed
	for &wire in expected_committed {
		assert!(
			commit_set.contains(wire),
			"Wire {:?} should be committed but wasn't. Commit set: {:?}",
			wire,
			commit_set
		);
	}

	// Verify expected wires are NOT committed (i.e., can be inlined)
	for &wire in expected_not_committed {
		assert!(
			!commit_set.contains(wire),
			"Wire {:?} should NOT be committed but was. Commit set: {:?}",
			wire,
			commit_set
		);
	}
}

#[test]
fn test_simple_xor_inlining() {
	// Test: y = x ^ a, z = y ^ b, then z is used in AND
	// Both y and z should be inlinable into the AND constraint
	test_commit_set(
		|cb| {
			// y = x ^ a
			cb.linear(expr::xor2(w(0), w(1)), w(2));
			// z = y ^ b
			cb.linear(expr::xor2(w(2), w(3)), w(4));
			// Use z in AND constraint (creates the root)
			cb.and(w(4), w(5), w(6));
		},
		&[],           // Nothing should be committed
		&[w(2), w(4)], // Both linear defs can be inlined
	);
}

#[test]
fn test_xor_used_in_and_constraint() {
	// Test: y = x ^ a, and(y, b, c)
	// y should be inlinable into the AND constraint
	test_commit_set(
		|cb| {
			// y = x ^ a
			cb.linear(expr::xor2(w(0), w(1)), w(2));
			// and(y, b, c)
			cb.and(w(2), w(3), w(4));
		},
		&[],     // y can be inlined
		&[w(2)], // y should not be committed
	);
}

#[test]
fn test_shift_composition_same_type() {
	// Test: y = x << 10, z = y << 20
	// Shifts should compose: z = x << 30
	test_commit_set(
		|cb| {
			// y = x << 10
			cb.linear(expr::sll(w(0), 10), w(1));
			// z = y << 20
			cb.linear(expr::sll(w(1), 20), w(2));
			// Use z in an AND constraint so it becomes a root
			cb.and(w(2), w(3), w(4));
		},
		&[], // Both shifts can be composed and inlined
		&[w(1), w(2)],
	);
}

#[test]
fn test_srl_composition() {
	// Test: y = x >> 10, z = y >> 20
	// Shifts should compose: z = x >> 30
	test_commit_set(
		|cb| {
			// y = x >> 10
			cb.linear(expr::srl(w(0), 10), w(1));
			// z = y >> 20
			cb.linear(expr::srl(w(1), 20), w(2));
			// Use z in an AND constraint so it becomes a root
			cb.and(w(2), w(3), w(4));
		},
		&[], // Both shifts can be composed and inlined
		&[w(1), w(2)],
	);
}

#[test]
fn test_sar_composition() {
	// y = sar(x, 31), z = sar(y, 1) -> compose to sar(x, 32), within range
	test_commit_set(
		|cb| {
			cb.linear(expr::sar(w(0), 31), w(1));
			cb.linear(expr::sar(w(1), 1), w(2));
			cb.and(w(2), w(3), w(4));
		},
		&[], // inlinable
		&[w(1), w(2)],
	);
}

#[test]
fn test_sar_incompatible_with_srl() {
	// y = sar(x, 7), z = srl(y, 1) -> incompatible; y must be committed
	test_commit_set(
		|cb| {
			cb.linear(expr::sar(w(0), 7), w(1));
			cb.linear(expr::srl(w(1), 1), w(2));
			cb.and(w(2), w(3), w(4));
		},
		&[w(1)], // commit producer
		&[w(2)], // z can inline its (now committed) input
	);
}

#[test]
fn test_sar_incompatible_with_sll() {
	// y = sar(x, 5), z = sll(y, 2) -> incompatible; y must be committed
	test_commit_set(
		|cb| {
			cb.linear(expr::sar(w(0), 5), w(1));
			cb.linear(expr::sll(w(1), 2), w(2));
			cb.and(w(2), w(3), w(4));
		},
		&[w(1)], // commit producer
		&[w(2)],
	);
}

#[test]
fn test_sar_incompatible_with_rotr() {
	// y = sar(x, 5), z = rotr(y, 10) -> incompatible types; y must be committed
	test_commit_set(
		|cb| {
			cb.linear(expr::sar(w(0), 5), w(1));
			cb.linear(expr::rotr(w(1), 10), w(2));
			cb.and(w(2), w(3), w(4));
		},
		&[w(1)],
		&[w(2)],
	);
}

#[test]
fn test_all_or_nothing_across_and_and_mul() {
	// x = sll(a, 20)
	// y = x ^ c (OK to inline)
	// z = srl(x, 5) (incompatible with sll)
	// Use y in AND and z in IMUL -> x must be committed due to mixed uses
	test_commit_set(
		|cb| {
			cb.linear(expr::sll(w(0), 20), w(1)); // x
			cb.linear(expr::xor2(w(1), w(2)), w(3)); // y = x ^ c
			cb.linear(expr::srl(w(1), 5), w(4)); // z = x >> 5
			cb.and(w(3), w(5), w(6));
			cb.imul(w(4), w(7), w(8), w(9));
		},
		&[w(1)],       // x must be committed
		&[w(3), w(4)], // y and z can inline their inputs (subject to x being committed)
	);
}

#[test]
fn test_sll_boundary_63_vs_64() {
	// Compose to 63 -> OK; compose to 64 -> commit
	// Case 1: 32 + 31 = 63
	test_commit_set(
		|cb| {
			cb.linear(expr::sll(w(0), 32), w(1));
			cb.linear(expr::sll(w(1), 31), w(2));
			cb.and(w(2), w(3), w(4));
		},
		&[],
		&[w(1), w(2)],
	);

	// Case 2: 32 + 32 = 64 -> commit first
	test_commit_set(
		|cb| {
			cb.linear(expr::sll(w(0), 32), w(5));
			cb.linear(expr::sll(w(5), 32), w(6));
			cb.and(w(6), w(7), w(8));
		},
		&[w(5)],
		&[w(6)],
	);
}

#[test]
fn test_srl_boundary_63_vs_64() {
	// Case 1: 16 + 47 = 63 -> OK
	test_commit_set(
		|cb| {
			cb.linear(expr::srl(w(0), 16), w(1));
			cb.linear(expr::srl(w(1), 47), w(2));
			cb.and(w(2), w(3), w(4));
		},
		&[],
		&[w(1), w(2)],
	);

	// Case 2: 48 + 16 = 64 -> commit first
	test_commit_set(
		|cb| {
			cb.linear(expr::srl(w(0), 48), w(5));
			cb.linear(expr::srl(w(5), 16), w(6));
			cb.and(w(6), w(7), w(8));
		},
		&[w(5)],
		&[w(6)],
	);
}

#[test]
fn test_sar_boundary_63_vs_64() {
	// Case 1: 40 + 23 = 63 -> OK
	test_commit_set(
		|cb| {
			cb.linear(expr::sar(w(0), 40), w(1));
			cb.linear(expr::sar(w(1), 23), w(2));
			cb.and(w(2), w(3), w(4));
		},
		&[],
		&[w(1), w(2)],
	);

	// Case 2: 32 + 32 = 64 -> commit first
	test_commit_set(
		|cb| {
			cb.linear(expr::sar(w(0), 32), w(5));
			cb.linear(expr::sar(w(5), 32), w(6));
			cb.and(w(6), w(7), w(8));
		},
		&[w(5)],
		&[w(6)],
	);
}

#[test]
fn test_zero_shift_composition() {
	// A shift of no distance is the identity, whichever variant spells it, so two of them
	// compose however their variants differ. Nothing here needs committing.
	test_commit_set(
		|cb| {
			// y = sll(x, 0)
			cb.linear(expr::sll(w(0), 0), w(1));
			// z = srl(y, 0)
			cb.linear(expr::srl(w(1), 0), w(2));
			cb.and(w(2), w(3), w(4));
		},
		&[],
		&[w(1), w(2)],
	);

	// Same-type zero shifts compose trivially
	test_commit_set(
		|cb| {
			// y = srl(x, 0)
			cb.linear(expr::srl(w(0), 0), w(1));
			// z = srl(y, 0)
			cb.linear(expr::srl(w(1), 0), w(2));
			cb.and(w(2), w(3), w(4));
		},
		&[],
		&[w(1), w(2)],
	);
}

#[test]
fn test_rotr_zero_inlining() {
	// y = a ^ b; z = rotr(y, 0); use z in AND. Both y and z should be inlinable.
	test_commit_set(
		|cb| {
			cb.linear(expr::xor2(w(0), w(1)), w(2)); // y
			cb.linear(expr::rotr(w(2), 0), w(3)); // z
			cb.and(w(3), w(4), w(5));
		},
		&[],
		&[w(2), w(3)],
	);
}

#[test]
fn test_diamond_fanout_inlining() {
	// P = x ^ y
	// Q = P ^ c
	// R = P ^ d
	// S = Q ^ R
	// Use S in AND -> Expect P,Q,R,S all inlinable (no shifts)
	test_commit_set(
		|cb| {
			cb.linear(expr::xor2(w(0), w(1)), w(2)); // P
			cb.linear(expr::xor2(w(2), w(3)), w(4)); // Q
			cb.linear(expr::xor2(w(2), w(5)), w(6)); // R
			cb.linear(expr::xor2(w(4), w(6)), w(7)); // S
			cb.and(w(7), w(8), w(9));
		},
		&[],
		&[w(2), w(4), w(6), w(7)],
	);
}

#[test]
fn test_shift_composition_different_types() {
	// Test: y = x << 10, z = y >> 20
	// Different shift types cannot compose, y must be committed
	test_commit_set(
		|cb| {
			// y = x << 10
			cb.linear(expr::sll(w(0), 10), w(1));
			// z = y >> 20
			cb.linear(expr::srl(w(1), 20), w(2));
			// Use z in an AND constraint
			cb.and(w(2), w(3), w(4));
		},
		&[w(1)], // y must be committed (incompatible shifts)
		&[w(2)], // z can still be inlined
	);
}

#[test]
fn test_rotr_distributes_over_xor() {
	// Test: y = a ^ b, z = rotr(y, 5)
	// Should distribute: z = rotr(a, 5) ^ rotr(b, 5)
	test_commit_set(
		|cb| {
			// y = a ^ b
			cb.linear(expr::xor2(w(0), w(1)), w(2));
			// z = rotr(y, 5)
			cb.linear(expr::rotr(w(2), 5), w(3));
			// Use z in an AND constraint
			cb.and(w(3), w(4), w(5));
		},
		&[], // Both can be inlined (rotr distributes over xor)
		&[w(2), w(3)],
	);
}

#[test]
fn test_rotr_distributes_over_multi_xor() {
	// Test: y = a ^ b ^ c, z = rotr(y, 7)
	// Should distribute: z = rotr(a, 7) ^ rotr(b, 7) ^ rotr(c, 7)
	test_commit_set(
		|cb| {
			// y = a ^ b ^ c
			cb.linear(expr::xor3(w(0), w(1), w(2)), w(3));
			// z = rotr(y, 7)
			cb.linear(expr::rotr(w(3), 7), w(4));
			// Use z in an AND constraint
			cb.and(w(4), w(5), w(6));
		},
		&[], // Both can be inlined (rotr distributes over xor)
		&[w(3), w(4)],
	);
}

#[test]
fn test_incompatible_shift_sequence() {
	// Test: y = a >> 10, z = y << 5
	// Different shift types in sequence cannot compose (srl then sll)
	test_commit_set(
		|cb| {
			// y = a >> 10
			cb.linear(expr::srl(w(0), 10), w(2));
			// z = y << 5
			cb.linear(expr::sll(w(2), 5), w(3));
			// Use z in an AND constraint
			cb.and(w(3), w(4), w(5));
		},
		&[w(2)], // y must be committed (incompatible shift sequence)
		&[w(3)], // z can still be inlined
	);
}

#[test]
fn test_multiple_uses_all_or_nothing() {
	// Test: x = a << 20, y = x ^ c, z = x >> 5
	// x is used in both y and z. Since we have x shifted left,
	// and z tries to shift it right, these are incompatible shift types.
	// Therefore x must be committed (all-or-nothing principle)
	test_commit_set(
		|cb| {
			// x = a << 20
			cb.linear(expr::sll(w(0), 20), w(2));
			// y = x ^ c (composable - shift can distribute over XOR)
			cb.linear(expr::xor2(w(2), w(3)), w(4));
			// z = x >> 5 (incompatible - can't compose sll with srl)
			cb.linear(expr::srl(w(2), 5), w(5));
			// Use y and z in AND constraints
			cb.and(w(4), w(6), w(7));
			cb.and(w(5), w(8), w(9));
		},
		&[w(2)],       // x must be committed (incompatible shift types)
		&[w(4), w(5)], // y and z can be inlined
	);
}

#[test]
fn test_fixed_point_iteration() {
	// Test: a = input1 ^ input2
	//       b = a >> 10 (srl shift)
	//       c = b ^ input4
	//       d = b << 5 (sll shift - incompatible with srl)
	// b has incompatible uses (used with both XOR and incompatible shift)
	test_commit_set(
		|cb| {
			// a = input1 ^ input2
			cb.linear(expr::xor2(w(0), w(1)), w(2));
			// b = a >> 10
			cb.linear(expr::srl(w(2), 10), w(4));
			// c = b ^ input4
			cb.linear(expr::xor2(w(4), w(5)), w(6));
			// d = b << 5 (incompatible - can't compose srl with sll)
			cb.linear(expr::sll(w(4), 5), w(7));
			// Use c and d in AND constraints
			cb.and(w(6), w(8), w(9));
			cb.and(w(7), w(10), w(11));
		},
		&[w(4)],             // b must be committed (incompatible shift types)
		&[w(2), w(6), w(7)], // a, c, and d can be inlined
	);
}

#[test]
fn test_rotr_composition() {
	// Test: y = rotr(x, 10), z = rotr(y, 15)
	// Should compose to: z = rotr(x, 25)
	test_commit_set(
		|cb| {
			// y = rotr(x, 10)
			cb.linear(expr::rotr(w(0), 10), w(1));
			// z = rotr(y, 15)
			cb.linear(expr::rotr(w(1), 15), w(2));
			// Use z in AND constraint
			cb.and(w(2), w(3), w(4));
		},
		&[], // Both rotations can compose
		&[w(1), w(2)],
	);
}

#[test]
fn test_complex_xor_chain() {
	// Test: y = a ^ b ^ c, z = y ^ d ^ e
	// Both should be inlinable
	test_commit_set(
		|cb| {
			// y = a ^ b ^ c
			cb.linear(expr::xor3(w(0), w(1), w(2)), w(3));
			// z = y ^ d ^ e
			cb.linear(expr::xor3(w(3), w(4), w(5)), w(6));
			// Use z in AND constraint
			cb.and(w(6), w(7), w(8));
		},
		&[], // All can be inlined
		&[w(3), w(6)],
	);
}

#[test]
fn test_wire_used_in_imul_constraint() {
	// Test: y = x ^ a, mul(y, b) = hi:lo
	// y should be inlinable into the IMUL constraint
	test_commit_set(
		|cb| {
			// y = x ^ a
			cb.linear(expr::xor2(w(0), w(1)), w(2));
			// mul(y, b) = hi:lo
			cb.imul(w(2), w(3), w(4), w(5));
		},
		&[],     // y can be inlined
		&[w(2)], // y should not be committed
	);
}

#[test]
fn test_shifted_wire_in_non_linear_use() {
	// Test: y = x >> 5, and(y, a, b)
	// Since y is already shifted and used in non-linear constraint,
	// we need to be careful about inlining
	test_commit_set(
		|cb| {
			// y = x >> 5
			cb.linear(expr::srl(w(0), 5), w(1));
			// and(y, a, b)
			cb.and(w(1), w(2), w(3));
		},
		&[], // Simple shift can be inlined
		&[w(1)],
	);
}

#[test]
fn test_multiple_non_linear_uses() {
	// Test: y = x ^ a, and(y, b, c), and(y, d, e)
	// y used in multiple AND constraints - should still be inlinable
	test_commit_set(
		|cb| {
			// y = x ^ a
			cb.linear(expr::xor2(w(0), w(1)), w(2));
			// and(y, b, c)
			cb.and(w(2), w(3), w(4));
			// and(y, d, e)
			cb.and(w(2), w(5), w(6));
		},
		&[], // y can be inlined into both AND constraints
		&[w(2)],
	);
}

#[test]
fn test_deep_xor_tree() {
	// Test a deeper tree of XOR operations
	// a = x ^ y
	// b = z ^ w
	// c = a ^ b
	// All should be inlinable
	test_commit_set(
		|cb| {
			// a = x ^ y
			cb.linear(expr::xor2(w(0), w(1)), w(2));
			// b = z ^ w
			cb.linear(expr::xor2(w(3), w(4)), w(5));
			// c = a ^ b
			cb.linear(expr::xor2(w(2), w(5)), w(6));
			// Use c in AND constraint
			cb.and(w(6), w(7), w(8));
		},
		&[], // All can be inlined
		&[w(2), w(5), w(6)],
	);
}

#[test]
fn test_shift_overflow_prevention() {
	// Test: y = x << 40, z = y << 30
	// Combined shift would be 70, which exceeds 64 bits
	// y must be committed
	test_commit_set(
		|cb| {
			// y = x << 40
			cb.linear(expr::sll(w(0), 40), w(1));
			// z = y << 30
			cb.linear(expr::sll(w(1), 30), w(2));
			// Use z in AND constraint
			cb.and(w(2), w(3), w(4));
		},
		&[w(1)], // y must be committed (shift overflow)
		&[w(2)], // z can still be inlined
	);
}

#[test]
fn test_rotr_wraps_correctly() {
	// Test: y = rotr(x, 50), z = rotr(y, 30)
	// Combined rotation should be (50 + 30) % 64 = 16
	// Both should be inlinable
	test_commit_set(
		|cb| {
			// y = rotr(x, 50)
			cb.linear(expr::rotr(w(0), 50), w(1));
			// z = rotr(y, 30)
			cb.linear(expr::rotr(w(1), 30), w(2));
			// Use z in AND constraint
			cb.and(w(2), w(3), w(4));
		},
		&[], // Both can be composed (rotation wraps)
		&[w(1), w(2)],
	);
}

#[test]
fn test_rotr_large_composition() {
	// Test: y = rotr(x, 63), z = rotr(y, 63)
	// Combined rotation should be (63 + 63) % 64 = 62
	// Both should be inlinable
	test_commit_set(
		|cb| {
			// y = rotr(x, 63)
			cb.linear(expr::rotr(w(0), 63), w(1));
			// z = rotr(y, 63)
			cb.linear(expr::rotr(w(1), 63), w(2));
			// Use z in AND constraint
			cb.and(w(2), w(3), w(4));
		},
		&[], // Both can be composed (rotation wraps at 64)
		&[w(1), w(2)],
	);
}

#[test]
fn test_no_linear_defs() {
	// Test with only AND constraints, no linear constraints
	test_commit_set(
		|cb| {
			// Just AND constraints, no linear defs
			cb.and(w(0), w(1), w(2));
			cb.and(w(3), w(4), w(5));
		},
		&[], // Nothing to commit
		&[], // No linear defs to inline
	);
}

#[test]
fn test_linear_def_no_uses() {
	// Test: y = x ^ a, but y is never used
	// Unused linear defs don't need to be committed
	test_commit_set(
		|cb| {
			// y = x ^ a (unused)
			cb.linear(expr::xor2(w(0), w(1)), w(2));
			// Some other AND constraint
			cb.and(w(3), w(4), w(5));
		},
		&[],     // Unused def doesn't need committing
		&[w(2)], // Not committed
	);
}

#[test]
fn test_mixed_shift_in_xor() {
	// Test: y = (x << 5) ^ (z >> 3), used in AND
	// The operand has mixed shifts, but they're at the term level
	test_commit_set(
		|cb| {
			// y = (x << 5) ^ (z >> 3)
			cb.linear(expr::xor2(expr::sll(w(0), 5), expr::srl(w(1), 3)), w(2));
			// and(y, a, b)
			cb.and(w(2), w(3), w(4));
		},
		&[], // Can be inlined (shifts are on individual terms)
		&[w(2)],
	);
}

#[test]
fn test_recursive_commit_propagation() {
	// Test: a = input >> 15 (srl)
	//       b = a ^ input3
	//       c = a << 10 (sll - incompatible with srl)
	// a has incompatible uses (XOR in b, incompatible shift in c)
	test_commit_set(
		|cb| {
			// a = input >> 15
			cb.linear(expr::srl(w(0), 15), w(2));
			// b = a ^ input3
			cb.linear(expr::xor2(w(2), w(3)), w(4));
			// c = a << 10 (incompatible - can't compose srl with sll)
			cb.linear(expr::sll(w(2), 10), w(5));
			// d = b ^ input4
			cb.linear(expr::xor2(w(4), w(6)), w(7));
			// Use c and d in AND constraints
			cb.and(w(5), w(8), w(9));
			cb.and(w(7), w(10), w(11));
		},
		&[w(2)],             // a must be committed (incompatible uses)
		&[w(4), w(5), w(7)], // b, c, and d can be inlined
	);
}

#[test]
fn test_rotr_with_unshifted_xor_terms() {
	// Test the specific bug we fixed: rotr(a ^ b, n) where a and b are unshifted
	// This tests that Rotr(n) composes correctly with None (unshifted terms)
	test_commit_set(
		|cb| {
			// y = a ^ b (both unshifted)
			cb.linear(expr::xor2(w(0), w(1)), w(2));
			// z = rotr(y, 63)
			cb.linear(expr::rotr(w(2), 63), w(3));
			// Use z in an AND constraint
			cb.and(w(3), w(4), w(5));
		},
		&[], // Everything should be inlinable - rotr distributes over unshifted XOR
		&[w(2), w(3)],
	);
}

#[test]
fn test_single_term_chain_past_the_depth_cap_stays_inlined() {
	// A definition of one term substitutes one term for one term into each consumer.
	// So a run of them cannot grow any operand, however long the run gets.
	// The depth cap exists to stop operands growing, so it must not fire on such a run.
	//
	// Fixture state: 9 chained rotations, against a cap of 6.
	//
	//   w(0) ──rotr─> w(1) ──rotr─> ... ──rotr─> w(9) ──> AND
	//   depth:            8              7  ...      0
	//
	// Wires w(1) and w(2) sit at depth 8 and 7, both past the cap, yet neither may commit.
	let n = MAX_DEPTH as u32 + 3;

	test_commit_set(
		|cb| {
			for i in 0..n {
				cb.linear(expr::rotr(w(i), 1), w(i + 1));
			}
			cb.and(w(n), w(n + 1), w(n + 2));
		},
		// Nothing is committed: every rotation composes and none of them can grow an operand.
		&[],
		&(1..=n).map(w).collect::<Vec<_>>(),
	);
}

#[test]
fn test_depth_still_counts_through_exempt_definitions() {
	// Exempting a definition from the cap must not reset how deep its producers sit.
	// Otherwise a run of one-term definitions would hide an arbitrarily deep cone above it.
	//
	// Fixture state: one two-term definition, then 8 chained rotations, against a cap of 6.
	//
	//   w(0) ─┐
	//         ├─> w(2) ──rotr─> w(3) ──rotr─> ... ──rotr─> w(10) ──> AND
	//   w(1) ─┘
	//   depth:      8              7             ...          0
	//
	// The rotations are exempt at depth 7 and below, but each one still adds a level.
	// So the two-term definition lands at depth 8 and commits there.
	let n = MAX_DEPTH as u32 + 2;

	test_commit_set(
		|cb| {
			cb.linear(expr::xor2(w(0), w(1)), w(2));
			for i in 2..n + 2 {
				cb.linear(expr::rotr(w(i), 1), w(i + 1));
			}
			cb.and(w(n + 2), w(n + 3), w(n + 4));
		},
		&[w(2)],
		&(3..=n + 2).map(w).collect::<Vec<_>>(),
	);
}

#[test]
fn test_exempt_definition_with_a_non_composing_shift_is_still_committed() {
	// The depth exemption reasons about operand size only.
	// It never licenses an inline that the shifts themselves forbid.
	//
	// Fixture state: a rotation at the top of a run of 8 left shifts, all one-term.
	//
	//   w(0) ──rotr─> w(1) ──sll─> w(2) ──sll─> ... ──sll─> w(9) ──> AND
	//
	// A rotation and a left shift are different kinds, so they never merge into one shift.
	// Inlining w(1) would ask for exactly that merge, so w(1) has to be committed.
	let n = MAX_DEPTH as u32 + 3;

	test_commit_set(
		|cb| {
			cb.linear(expr::rotr(w(0), 1), w(1));
			for i in 1..n {
				cb.linear(expr::sll(w(i), 1), w(i + 1));
			}
			cb.and(w(n), w(n + 1), w(n + 2));
		},
		&[w(1)],
		&(2..=n).map(w).collect::<Vec<_>>(),
	);
}

#[test]
fn test_rotr_with_mixed_shift_xor() {
	// Test: y = a ^ (b << 5), z = rotr(y, 10)
	// When we try to inline with rotr(10), the Sll(5) is incompatible
	test_commit_set(
		|cb| {
			// b_shifted = b << 5
			cb.linear(expr::sll(w(1), 5), w(6));
			// y = a ^ b_shifted (a is unshifted, b_shifted has Sll)
			cb.linear(expr::xor2(w(0), w(6)), w(2));
			// z = rotr(y, 10)
			cb.linear(expr::rotr(w(2), 10), w(3));
			// Use z in an AND constraint
			cb.and(w(3), w(4), w(5));
		},
		&[w(6)],       // b_shifted must be committed (can't compose Rotr with Sll)
		&[w(2), w(3)], // y and z can still be inlined
	);
}

#[test]
fn test_chained_shifts_commit_before_the_accumulated_distance_leaves_the_width() {
	// Inlining a run of same-kind shifts merges them into one shift of the summed distance.
	// A left shift drops the bits it pushes out.
	// So a summed distance of 64 or more has no single-shift form at all.
	// Whether a run may be inlined therefore turns on the sum of its distances.
	// It does not turn on any one pair of neighbours, which is why no pair here objects.
	//
	// Fixture state: 4 chained left shifts of 20, at depth 3, well inside the cap of 6.
	//
	//   w(0) ──sll20─> w(1) ──sll20─> w(2) ──sll20─> w(3) ──sll20─> w(4) ──> AND
	//   running sum:        80             60             40             20
	test_commit_set(
		|cb| {
			cb.linear(expr::sll(w(0), 20), w(1));
			cb.linear(expr::sll(w(1), 20), w(2));
			cb.linear(expr::sll(w(2), 20), w(3));
			cb.linear(expr::sll(w(3), 20), w(4));
			cb.and(w(4), w(5), w(6));
		},
		&[w(1)],             // reaching w(0) needs a shift of 80, past the 64-bit width
		&[w(2), w(3), w(4)], // a shift of 60 still has a single-shift form
	);
}

#[test]
fn test_a_long_single_term_shift_run_commits_as_often_as_the_width_requires() {
	// The cap exempts one-term definitions, so the width is all that can break a run of them.
	// A run this long must therefore be broken exactly as often as the width demands.
	//
	// Fixture state: 65 chained left shifts of 1, every one of them exempt from the cap.
	//
	//   w(0) ──sll1─> w(1) ──sll1─> ... ──sll1─> w(65) ──> AND
	//
	// An inlined stretch may sum to at most 63, so it may span at most 63 links.
	// Walking up from the AND, links w(65) down to w(3) fill that budget exactly.
	// Reaching w(2) would make 64, so w(2) commits and the count restarts above it.
	let n = 65;

	let mut cb = ConstraintBuilder::new();
	for i in 0..n {
		cb.linear(expr::sll(w(i), 1), w(i + 1));
	}
	cb.and(w(n), w(n + 1), w(n + 2));

	let mut stat = Stat::default();
	let mut leg = LeGraph::new(&cb);
	commit_set::run_decide_commit_set(&mut leg, &mut stat);

	// One break covers all 65 links, and it lands on the one the width forces.
	let committed: Vec<u32> = (1..=n)
		.filter(|&i| leg.commit_set().contains(w(i)))
		.collect();
	assert_eq!(committed, vec![2]);
}
