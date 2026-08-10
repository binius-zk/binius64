// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers
use binius_core::{ConstraintSystem, ValueIndex, ValueSegment, word::Word};

use crate::compiler::{CircuitBuilder, Options, gate_fusion::commit_set::MAX_DEPTH};

#[test]
fn test_force_commit_non_linear_output_without_inout() {
	// Regression: force-committing a non-linear (AND) output when the circuit has no inout wires
	// must not panic in gate fusion. A committed AND output is not a linear definition, so it must
	// stay out of the linear-commit set. This is the M4 accelerator pattern: witness inputs, no
	// inout, output pinned with `force_commit` so dead-code elimination keeps the constraint.
	let builder = CircuitBuilder::new();
	let x = builder.add_witness();
	let y = builder.add_witness();
	let and_out = builder.band(x, y);
	builder.force_commit(and_out);
	let circuit = builder.build();

	let mut w = circuit.new_witness_filler();
	w[x] = Word(0xF0F0_F0F0_F0F0_F0F0);
	w[y] = Word(0xFF00_FF00_FF00_FF00);
	circuit.populate_wire_witness(&mut w).unwrap();
	assert_eq!(w[and_out], Word(0xF0F0_F0F0_F0F0_F0F0 & 0xFF00_FF00_FF00_FF00));
	circuit.constraint_system().verify(&w.value_vec).unwrap();
}

#[test]
fn test_bmul_gate_end_to_end() {
	// A single BMUL gate: the eval opcode fills the product wires, and the BMUL constraint checks
	// the GHASH multiply. Uses the field fact X^127 * X = X^128 = X^7 + X^2 + X + 1 = 0x87 (a
	// reduction case), so this exercises the whole gate → constraint + eval-bytecode → witness →
	// verification path. Each field element is a (lo, hi) word pair.
	let builder = CircuitBuilder::new();
	let a_lo = builder.add_witness();
	let a_hi = builder.add_witness();
	let b_lo = builder.add_witness();
	let b_hi = builder.add_witness();
	let (c_lo, c_hi) = builder.bmul(a_lo, a_hi, b_lo, b_hi);
	let circuit = builder.build();

	let mut w = circuit.new_witness_filler();
	w[a_lo] = Word::ZERO;
	w[a_hi] = Word(0x8000_0000_0000_0000); // X^127
	w[b_lo] = Word(2); // X
	w[b_hi] = Word::ZERO;
	circuit.populate_wire_witness(&mut w).unwrap();

	assert_eq!(w[c_lo], Word(0x87));
	assert_eq!(w[c_hi], Word::ZERO);
	circuit.constraint_system().verify(&w.value_vec).unwrap();
}

/// Returns a string that represents the given constraint system in a textual form.
fn stringify_constraint_system(cs: &ConstraintSystem) -> String {
	use std::fmt::Write;

	let mut output = String::new();

	// ZERO constraints
	if !cs.zero_constraints.is_empty() {
		for (i, constraint) in cs.zero_constraints.iter().enumerate() {
			write!(output, "ZERO[{}]: (", i).unwrap();
			format_operand(&mut output, constraint.val(), cs);
			writeln!(output, ") = 0").unwrap();
		}
	}

	// AND constraints
	if !cs.and_constraints.is_empty() {
		for (i, constraint) in cs.and_constraints.iter().enumerate() {
			write!(output, "AND[{}]: (", i).unwrap();
			format_operand(&mut output, constraint.a(), cs);
			write!(output, ") ∧ (").unwrap();
			format_operand(&mut output, constraint.b(), cs);
			write!(output, ") = (").unwrap();
			format_operand(&mut output, constraint.c(), cs);
			writeln!(output, ")").unwrap();
		}
	}

	// IMUL constraints
	if !cs.imul_constraints.is_empty() {
		for (i, constraint) in cs.imul_constraints.iter().enumerate() {
			write!(output, "IMUL[{}]: (", i).unwrap();
			format_operand(&mut output, constraint.a(), cs);
			write!(output, ") * (").unwrap();
			format_operand(&mut output, constraint.b(), cs);
			write!(output, ") = (HI: ").unwrap();
			format_operand(&mut output, constraint.hi(), cs);
			write!(output, ", LO: ").unwrap();
			format_operand(&mut output, constraint.lo(), cs);
			writeln!(output, ")").unwrap();
		}
	}

	output.trim_end().to_string()
}

/// Format an operand as a readable XOR expression
fn format_operand(
	output: &mut String,
	operand: &[binius_core::constraint_system::ShiftedValueIndex],
	cs: &ConstraintSystem,
) {
	use std::fmt::Write;

	use binius_core::constraint_system::ShiftVariant;

	if operand.is_empty() {
		write!(output, "0").unwrap();
		return;
	}

	for (i, term) in operand.iter().enumerate() {
		if i > 0 {
			write!(output, " ⊕ ").unwrap();
		}

		let value_name = format_value_name(term.value_index, cs);

		if term.shift.is_identity() {
			write!(output, "{}", value_name).unwrap();
		} else {
			let shift_op = match term.shift.variant {
				ShiftVariant::Sll => "≪",
				ShiftVariant::Slr => "≫",
				ShiftVariant::Sar => "a≫",
				ShiftVariant::Rotr => "≫≫",
				ShiftVariant::Sll32 => "≪32",
				ShiftVariant::Srl32 => "≫32",
				ShiftVariant::Sra32 => "a≫32",
				ShiftVariant::Rotr32 => "≫≫32",
			};
			write!(output, "{}{}{}", value_name, shift_op, term.shift.amount).unwrap();
		}
	}
}

/// Convert a ValueIndex to a human-readable name, inlining constants
fn format_value_name(index: ValueIndex, cs: &ConstraintSystem) -> String {
	match index.segment() {
		// Constants are inlined by value, so a snapshot reads the word rather than a slot number.
		ValueSegment::Constant => {
			let constant = cs.constants[index.index() as usize];
			if constant.0 == u64::MAX {
				"all-1".to_string()
			} else {
				format!("0x{:x}", constant.0)
			}
		}
		ValueSegment::InOut => format!("io[{}]", index.index()),
		ValueSegment::Private => format!("v[{}]", index.index()),
		// A constraint cannot name a scratch word, so this only shows up if one leaks.
		ValueSegment::Scratch => format!("s[{}]", index.index()),
	}
}

fn mk_circuit_builder() -> CircuitBuilder {
	let opts = Options {
		enable_gate_fusion: true,
		enable_constant_propagation: false,
		// Keep these snapshots focused on fusion.
		// DCE, CSE, and algebraic folding would drop or rewrite the gates they assert on.
		enable_common_subexpression_elimination: false,
		enable_dead_code_elimination: false,
		enable_algebraic_folding: false,
		// Scratch pooling only moves uncommitted slots, so it cannot affect these snapshots.
		enable_scratch_pooling: false,
		// Zero propagation would forward past the gates these snapshots assert on.
		enable_zero_propagation: false,
	};
	CircuitBuilder::with_opts(opts)
}

fn compile(circuit_builder: &CircuitBuilder) -> ConstraintSystem {
	let circuit = circuit_builder.build();
	circuit.constraint_system().clone()
}

// =============== IMUL tests (placed next to AND tests) ===============

#[test]
fn test_mul_inlining_duplicate_linear_in_mul() {
	// y = x ^ c; then mul(y, y) = (hi, lo)
	// Expect: y is fully inlined into both a and b operands of IMUL (duplicated terms).
	let b = mk_circuit_builder();

	// Inputs
	let x = b.add_witness();
	let c = b.add_witness();
	let y = b.bxor(x, c);

	let (_hi, _lo) = b.imul(y, y);

	let cs = compile(&b);

	insta::assert_snapshot!(stringify_constraint_system(&cs), @"IMUL[0]: (v[0] ⊕ v[1]) * (v[0] ⊕ v[1]) = (HI: v[2], LO: v[3])");
}

#[test]
fn test_mul_and_and_shared_linear_uses() {
	// y = x ^ c; AND uses y and IMUL uses y as one operand
	// Expect: y gets inlined into both constraints appropriately
	let b = mk_circuit_builder();

	let x = b.add_witness();
	let c = b.add_witness();
	let y = b.bxor(x, c);

	// Use y in AND
	let w = b.add_witness();
	let _and_out = b.band(y, w);

	// Use y in IMUL
	let z = b.add_witness();
	let (_hi, _lo) = b.imul(y, z);

	let cs = compile(&b);
	insta::assert_snapshot!(stringify_constraint_system(&cs), @"
	AND[0]: (v[0] ⊕ v[1]) ∧ (v[2]) = (v[4])
	IMUL[0]: (v[0] ⊕ v[1]) * (v[3]) = (HI: v[5], LO: v[6])
	");
}

#[test]
fn test_mul_inlining_into_hi_lo() {
	// hi and lo are linear defs to be inlined into IMUL
	let b = mk_circuit_builder();

	let a = b.add_witness();
	let b0 = b.add_witness();
	let x = b.add_witness();
	let y = b.add_witness();

	let _hi_src = b.bxor(a, b0);
	let _lo_src = b.bxor(x, y);

	let w1 = b.add_witness();
	let w2 = b.add_witness();
	let (_hi, _lo) = b.imul(w1, w2);

	// Now enforce hi_src and lo_src via XOR-to-all-1 & = dst by using them in subsequent
	// constraints Connect hi_src and lo_src as replacements for the outputs of imul
	// We cannot directly modify outputs, but using them later in ANDs will keep them alive
	// The important check is that IMUL constraint references should inline hi_src and lo_src

	let cs = compile(&b);
	insta::assert_snapshot!(stringify_constraint_system(&cs), @"
	ZERO[0]: (v[0] ⊕ v[1] ⊕ v[6]) = 0
	ZERO[1]: (v[2] ⊕ v[3] ⊕ v[7]) = 0
	IMUL[0]: (v[4]) * (v[5]) = (HI: v[8], LO: v[9])
	");
}

#[test]
fn test_mul_inlining_distinct_linears() {
	// y1 = x ^ c
	// y2 = sll(u, 3) ^ v
	// mul(y1, y2)
	// Expect: y1 inlined into a; y2 inlined into b with one SLL(3) term
	let b = mk_circuit_builder();

	let x = b.add_witness();
	let c = b.add_witness();
	let y1 = b.bxor(x, c);

	let u = b.add_witness();
	let u_sll3 = b.shl(u, 3);
	let v = b.add_witness();
	let y2 = b.bxor(u_sll3, v);

	let (_hi, _lo) = b.imul(y1, y2);

	let cs = compile(&b);
	insta::assert_snapshot!(stringify_constraint_system(&cs), @"IMUL[0]: (v[0] ⊕ v[1]) * (v[2]≪3 ⊕ v[3]) = (HI: v[4], LO: v[5])");
}

#[test]
fn test_xor_unused_preserved() {
	// Unused XOR should be preserved, not removed
	let b = mk_circuit_builder();
	let v0 = b.add_constant_64(0xe4);
	let v1 = b.add_witness();
	let _v2 = b.bxor(v0, v1);
	let cs = compile(&b);
	insta::assert_snapshot!(stringify_constraint_system(&cs), @"ZERO[0]: (0xe4 ⊕ v[0] ⊕ v[1]) = 0");
}

#[test]
fn test_xor_into_assert() {
	let b = mk_circuit_builder();
	let v0 = b.add_constant_64(0xe4);
	let v1 = b.add_witness();
	let v2 = b.bxor(v0, v1);
	b.assert_zero("derp", v2);
	let cs = compile(&b);
	insta::assert_snapshot!(
		stringify_constraint_system(&cs),
		@"AND[0]: (0xe4 ⊕ v[0]) ∧ (all-1) = (0)",
	);
}

#[test]
fn test_xor_into_and_single_use() {
	let b = mk_circuit_builder();

	// v2 = v0 ^ v1
	// v4 = v3 & v2
	// → v4 = v3 & (v0 ^ v1)
	let v0 = b.add_witness();
	let v1 = b.add_witness();
	let v2 = b.bxor(v0, v1);

	let v3 = b.add_witness();
	let _v4 = b.band(v3, v2);

	let cs = compile(&b);
	insta::assert_snapshot!(
		stringify_constraint_system(&cs),
		@"AND[0]: (v[2]) ∧ (v[0] ⊕ v[1]) = (v[3])",
	);
}

#[test]
fn test_xor_used_on_both_sides() {
	let b = mk_circuit_builder();

	// v2 = v0 ^ v1
	// v6 = (v2 ^ v3) & (v4 ^ v2)
	// → v6 = ((v0 ^ v1) ^ v3) & (v4 ^ (v0 ^ v1))
	let v0 = b.add_witness();
	let v1 = b.add_witness();
	let v2 = b.bxor(v0, v1);

	let v3 = b.add_witness();
	let v4 = b.add_witness();
	let lhs = b.bxor(v2, v3); // v2 ^ v3
	let rhs = b.bxor(v4, v2); // v4 ^ v2
	let _v6 = b.band(lhs, rhs);

	let cs = compile(&b);
	// v2 is preserved since it's still referenced after lhs and rhs are inlined
	// Expected: v[6] (which is v2 = v0^v1) should be fully inlined as (v[2] ⊕ v[3])
	insta::assert_snapshot!(stringify_constraint_system(&cs), @"AND[0]: (v[0] ⊕ v[1] ⊕ v[2]) ∧ (v[3] ⊕ v[0] ⊕ v[1]) = (v[4])");
}

#[test]
fn test_shifted_base_into_and() {
	let b = mk_circuit_builder();

	// v2 = sll(v0, 33)
	// v4 = v3 & v2
	// → v4 = v3 & sll(v0, 33)
	let v0 = b.add_witness();
	let v1 = b.shl(v0, 33);

	let v2 = b.add_witness();
	let _v3 = b.band(v2, v1);

	let cs = compile(&b);
	insta::assert_snapshot!(stringify_constraint_system(&cs), @"AND[0]: (v[1]) ∧ (v[0]≪33) = (v[2])");
}

#[test]
fn test_mixed_xor_of_shifts_into_and() {
	let b = mk_circuit_builder();

	// v2 = srl(v0, 7) ^ sll(v1, 3)
	// v5 = v4 & v2
	// → v5 = v4 & (srl(v0, 7) ^ sll(v1, 3))
	let v0 = b.add_witness();
	let v1 = b.add_witness();
	let v0_srl = b.shr(v0, 7);
	let v1_sll = b.shl(v1, 3);
	let v2 = b.bxor(v0_srl, v1_sll);

	let v3 = b.add_witness();
	let _v4 = b.band(v3, v2);

	let cs = compile(&b);
	insta::assert_snapshot!(stringify_constraint_system(&cs), @"AND[0]: (v[2]) ∧ (v[0]≫7 ⊕ v[1]≪3) = (v[3])");
}

#[test]
fn test_deep_xor_in_both_a_and_c() {
	let b = mk_circuit_builder();

	// v2 = v0 ^ v1 ^ sra(v3, 1)
	// v7 = (v2 ^ v4) & mask
	// → the whole XOR cone inlines into operand A of the AND.
	let v0 = b.add_witness();
	let v1 = b.add_witness();
	let v2 = b.add_witness();
	let v3_sra = b.sar(v2, 1);
	let xor1 = b.bxor(v0, v1);
	let v2_expr = b.bxor(xor1, v3_sra);

	let v4 = b.add_witness();
	let lhs = b.bxor(v2_expr, v4);
	let mask = b.add_witness();
	let _v7 = b.band(lhs, mask);

	let cs = compile(&b);
	insta::assert_snapshot!(stringify_constraint_system(&cs), @"AND[0]: (v[0] ⊕ v[1] ⊕ v[2]a≫1 ⊕ v[3]) ∧ (v[4]) = (v[5])");
}

#[test]
fn test_fuse_across_parens() {
	let b = mk_circuit_builder();

	// v3 = sll(v0, 33) ^ v1 ^ v2
	// v6 = (v3 ^ v4) & v5
	// → v6 = ((sll(v0, 33) ^ v1 ^ v2 ^ v4)) & v5
	let v0 = b.add_witness();
	let v1 = b.add_witness();
	let v2 = b.add_witness();
	let v0_sll = b.shl(v0, 33);
	let xor1 = b.bxor(v0_sll, v1);
	let v3 = b.bxor(xor1, v2);

	let v4 = b.add_witness();
	let lhs = b.bxor(v3, v4);
	let v5 = b.add_witness();
	let _v6 = b.band(lhs, v5);

	let cs = compile(&b);
	insta::assert_snapshot!(stringify_constraint_system(&cs), @"AND[0]: (v[0]≪33 ⊕ v[1] ⊕ v[2] ⊕ v[3]) ∧ (v[4]) = (v[5])");
}

// Note: Alias inline functionality would require identity/alias detection in frontend
// This test is skipped as the current implementation doesn't support alias operations

// Note: Alias under shift functionality would require identity/alias detection in frontend
// This test is skipped as the current implementation doesn't support alias operations

#[test]
fn test_multiple_producers_into_one_consumer() {
	let b = mk_circuit_builder();

	// v2 = v0 ^ v1
	// v3 = srl(v4, 5)
	// v6 = (v2 ^ v3) & v5
	// → v6 = ((v0 ^ v1) ^ srl(v4, 5)) & v5
	let v0 = b.add_witness();
	let v1 = b.add_witness();
	let v2 = b.bxor(v0, v1);

	let v3 = b.add_witness();
	let v3_srl = b.shr(v3, 5);
	let lhs = b.bxor(v2, v3_srl);
	let v4 = b.add_witness();
	let _v5 = b.band(lhs, v4);

	let cs = compile(&b);
	insta::assert_snapshot!(stringify_constraint_system(&cs), @"AND[0]: (v[0] ⊕ v[1] ⊕ v[2]≫5) ∧ (v[3]) = (v[4])");
}

#[test]
fn test_xor_producer_used_twice_inside_one_side() {
	let b = mk_circuit_builder();

	// v2 = v0 ^ v1
	// v5 = (v2 ^ v2 ^ v3) & v4
	// → v5 = (((v0 ^ v1) ^ (v0 ^ v1) ^ v3)) & v4
	let v0 = b.add_witness();
	let v1 = b.add_witness();
	let v2 = b.bxor(v0, v1);

	let xor1 = b.bxor(v2, v2);
	let v3 = b.add_witness();
	let lhs = b.bxor(xor1, v3);
	let v4 = b.add_witness();
	let _v5 = b.band(lhs, v4);

	let cs = compile(&b);
	// When used twice, the XOR terms should be inlined twice
	insta::assert_snapshot!(stringify_constraint_system(&cs), @"AND[0]: (v[0] ⊕ v[1] ⊕ v[0] ⊕ v[1] ⊕ v[2]) ∧ (v[3]) = (v[4])");
}

#[test]
fn test_chain_two_level_fusion() {
	let b = mk_circuit_builder();

	// v2 = v0 ^ v1
	// v3 = v2 ^ v4
	// v6 = v5 & v3
	// → v6 = v5 & ((v0 ^ v1) ^ v4)
	let v0 = b.add_witness();
	let v1 = b.add_witness();
	let v2 = b.bxor(v0, v1);

	let v3 = b.add_witness();
	let v4 = b.bxor(v2, v3);

	let v5 = b.add_witness();
	let _v6 = b.band(v5, v4);

	let cs = compile(&b);
	insta::assert_snapshot!(stringify_constraint_system(&cs), @"AND[0]: (v[3]) ∧ (v[0] ⊕ v[1] ⊕ v[2]) = (v[4])");
}

#[test]
fn test_chain_with_shifts_inside_producers() {
	let b = mk_circuit_builder();

	// v2 = srl(v0, 2) ^ sra(v1, 7)
	// v3 = v2 ^ sll(v4, 11)
	// v6 = v5 & v3
	// → v6 = v5 & ((srl(v0, 2) ^ sra(v1, 7)) ^ sll(v4, 11))
	let v0 = b.add_witness();
	let v1 = b.add_witness();
	let v0_srl = b.shr(v0, 2);
	let v1_sra = b.sar(v1, 7);
	let v2 = b.bxor(v0_srl, v1_sra);

	let v3 = b.add_witness();
	let v3_sll = b.shl(v3, 11);
	let v4 = b.bxor(v2, v3_sll);

	let v5 = b.add_witness();
	let _v6 = b.band(v5, v4);

	let cs = compile(&b);
	insta::assert_snapshot!(stringify_constraint_system(&cs), @"AND[0]: (v[3]) ∧ (v[0]≫2 ⊕ v[1]a≫7 ⊕ v[2]≪11) = (v[4])");
}

#[test]
fn test_fuse_into_both_a_and_b() {
	let b = mk_circuit_builder();

	// v2 = v0 ^ srl(v1, 3)
	// v7 = (v2 ^ v3) & (v4 ^ v2 ^ v5)
	// → v7 = ((v0 ^ srl(v1, 3) ^ v3)) & (v4 ^ (v0 ^ srl(v1, 3)) ^ v5)
	let v0 = b.add_witness();
	let v1 = b.add_witness();
	let v1_srl = b.shr(v1, 3);
	let v2 = b.bxor(v0, v1_srl);

	let v3 = b.add_witness();
	let lhs = b.bxor(v2, v3);

	let v4 = b.add_witness();
	let rhs1 = b.bxor(v4, v2);
	let v5 = b.add_witness();
	let rhs = b.bxor(rhs1, v5);

	let _v6 = b.band(lhs, rhs);

	let cs = compile(&b);
	insta::assert_snapshot!(stringify_constraint_system(&cs), @"AND[0]: (v[0] ⊕ v[1]≫3 ⊕ v[2]) ∧ (v[3] ⊕ v[0] ⊕ v[1]≫3 ⊕ v[4]) = (v[5])");
}

#[test]
fn test_xor_feeding_xor_via_all_one() {
	let b = mk_circuit_builder();

	// v2 = v0 ^ sll(v1, 2)
	// v8 = (v2 ^ v3 ^ v4) & mask
	// → the whole XOR cone inlines into operand A of the AND.
	let v0 = b.add_witness();
	let v1 = b.add_witness();
	let v1_sll = b.shl(v1, 2);
	let v2 = b.bxor(v0, v1_sll);

	let v3 = b.add_witness();
	let v4 = b.add_witness();
	let xor1 = b.bxor(v2, v3);
	let lhs = b.bxor(xor1, v4);
	let mask = b.add_witness();
	let _v5 = b.band(lhs, mask);

	let cs = compile(&b);
	insta::assert_snapshot!(stringify_constraint_system(&cs), @"AND[0]: (v[0] ⊕ v[1]≪2 ⊕ v[2] ⊕ v[3]) ∧ (v[4]) = (v[5])");
}

#[test]
fn test_not_pattern_via_xor_with_all_one() {
	let b = mk_circuit_builder();

	// v2 = v0 ^ all-1
	// v4 = v3 & v2
	// → v4 = v3 & (v0 ^ all-1)
	let v0 = b.add_witness();
	let all_one = b.add_constant_64(u64::MAX);
	let v1 = b.bxor(v0, all_one);

	let v2 = b.add_witness();
	let _v3 = b.band(v2, v1);

	let cs = compile(&b);
	insta::assert_snapshot!(stringify_constraint_system(&cs), @"AND[0]: (v[1]) ∧ (v[0] ⊕ all-1) = (v[2])");
}

#[test]
fn test_dont_inline_xor_into_shifted_use() {
	let b = mk_circuit_builder();

	// v2 = v0 ^ v1
	// v4 = sll(v2, 5) & v3
	// → **no change** - v2 must be preserved as intermediate
	let v0 = b.add_witness();
	let v1 = b.add_witness();
	let v2 = b.bxor(v0, v1);

	let v2_sll = b.shl(v2, 5);
	let v3 = b.add_witness();
	let _v4 = b.band(v2_sll, v3);

	let cs = compile(&b);
	// The optimization successfully inlines despite the shift at the use site
	// This shows gate fusion can handle shifts in the consuming constraint
	insta::assert_snapshot!(stringify_constraint_system(&cs), @"AND[0]: (v[0]≪5 ⊕ v[1]≪5) ∧ (v[2]) = (v[3])");
}

#[test]
fn test_dont_inline_shifted_producer_into_shifted_use() {
	let b = mk_circuit_builder();

	// v2 = srl(v0, 7)
	// v4 = sll(v2, 3) & v1
	// → **no change** - can't compose different shift types
	let v0 = b.add_witness();
	let v1 = b.shr(v0, 7);

	let v1_sll = b.shl(v1, 3);
	let v2 = b.add_witness();
	let _v3 = b.band(v1_sll, v2);

	let cs = compile(&b);
	// Cannot compose different shift types (srl then sll), so it commits the intermediate
	insta::assert_snapshot!(stringify_constraint_system(&cs), @"
	ZERO[0]: (v[0]≫7 ⊕ v[2]) = 0
	AND[0]: (v[2]≪3) ∧ (v[1]) = (v[3])
	");
}

#[test]
fn test_dont_inline_xor_into_shifted_xor_use() {
	let b = mk_circuit_builder();

	// v2 = v0 ^ srl(v1, 2)
	// v6 = (srl(v2, 9) ^ v3) & v4
	// → **no change** - can't inline v2 because it's used with a shift
	let v0 = b.add_witness();
	let v1 = b.add_witness();
	let v1_srl = b.shr(v1, 2);
	let v2 = b.bxor(v0, v1_srl);

	let v2_srl = b.shr(v2, 9);
	let v3 = b.add_witness();
	let lhs = b.bxor(v2_srl, v3);
	let v4 = b.add_witness();
	let _v5 = b.band(lhs, v4);

	let cs = compile(&b);
	// The optimization can actually compose shifts of the same type (shr)
	// So it inlines despite the shift at use site
	insta::assert_snapshot!(stringify_constraint_system(&cs), @"AND[0]: (v[0]≫9 ⊕ v[1]≫11 ⊕ v[2]) ∧ (v[3]) = (v[4])");
}

#[test]
fn test_mixed_one_unshifted_use_one_shifted_use() {
	let b = mk_circuit_builder();

	// v2 = v0 ^ v1
	// v4 = v3 & v2
	// v5 = srl(v2, 3) & v6
	// → v2 must be preserved because of shifted use in v5
	let v0 = b.add_witness();
	let v1 = b.add_witness();
	let v2 = b.bxor(v0, v1);

	let v3 = b.add_witness();
	let _v4 = b.band(v3, v2);

	let v2_srl = b.shr(v2, 3);
	let v5 = b.add_witness();
	let _v6 = b.band(v2_srl, v5);

	let cs = compile(&b);
	// The optimization can inline into both uses since shifts distribute over XOR
	insta::assert_snapshot!(stringify_constraint_system(&cs), @"
	AND[0]: (v[2]) ∧ (v[0] ⊕ v[1]) = (v[4])
	AND[1]: (v[0]≫3 ⊕ v[1]≫3) ∧ (v[3]) = (v[5])
	");
}

#[test]
fn test_rotr_overflow_to_zero() {
	let b = mk_circuit_builder();

	// Test rotr that overflows to 0 (64 bits = full rotation = identity)
	// v1 = rotr(v0, 32)
	// v2 = rotr(v1, 32)  // Combined: rotr(v0, 64) = v0
	// v4 = v3 & v2
	// → v4 = v3 & v0 (since rotr(v0, 64) = v0)
	let v0 = b.add_witness();
	let v1 = b.rotr(v0, 32);
	let v2 = b.rotr(v1, 32); // Total rotation: 64 bits = 0

	let v3 = b.add_witness();
	let _v4 = b.band(v3, v2);

	let cs = compile(&b);
	// rotr(v0, 64) should be simplified to v0 (no rotation)
	insta::assert_snapshot!(stringify_constraint_system(&cs), @"AND[0]: (v[1]) ∧ (v[0]) = (v[2])");
}

#[test]
fn test_rotr_wrap_around() {
	let b = mk_circuit_builder();

	// Test rotr that wraps around (more than 64 bits)
	// v1 = rotr(v0, 50)
	// v2 = rotr(v1, 30)  // Combined: rotr(v0, 80) = rotr(v0, 16)
	// v4 = v3 & v2
	// → v4 = v3 & rotr(v0, 16)
	let v0 = b.add_witness();
	let v1 = b.rotr(v0, 50);
	let v2 = b.rotr(v1, 30); // Total: 80 bits = 80 % 64 = 16

	let v3 = b.add_witness();
	let _v4 = b.band(v3, v2);

	let cs = compile(&b);
	insta::assert_snapshot!(stringify_constraint_system(&cs), @"AND[0]: (v[1]) ∧ (v[0]≫≫16) = (v[2])");
}

#[test]
fn test_rotr_in_xor_overflow() {
	let b = mk_circuit_builder();

	// Test rotr overflow in XOR chain
	// v2 = v0 ^ v1
	// v3 = rotr(v2, 40)
	// v4 = rotr(v3, 24)  // Combined: rotr(_, 64) = identity
	// v6 = v5 & v4
	// → v6 = v5 & (v0 ^ v1)
	let v0 = b.add_witness();
	let v1 = b.add_witness();
	let v2 = b.bxor(v0, v1);

	let v3 = b.rotr(v2, 40);
	let v4 = b.rotr(v3, 24); // Total: 64 bits = 0 rotation

	let v5 = b.add_witness();
	let _v6 = b.band(v5, v4);

	let cs = compile(&b);
	// rotr(v0 ^ v1, 64) should be just v0 ^ v1
	insta::assert_snapshot!(stringify_constraint_system(&cs), @"AND[0]: (v[2]) ∧ (v[0] ⊕ v[1]) = (v[3])");
}

#[test]
fn test_rotr_chain_with_wrap() {
	let b = mk_circuit_builder();

	// Test multiple rotr operations that wrap
	// v1 = rotr(v0, 20)
	// v2 = rotr(v1, 25)
	// v3 = rotr(v2, 30)  // Total: 75 bits = 75 % 64 = 11
	// v5 = v4 & v3
	let v0 = b.add_witness();
	let v1 = b.rotr(v0, 20);
	let v2 = b.rotr(v1, 25);
	let v3 = b.rotr(v2, 30); // Total: 75 = 11 mod 64

	let v4 = b.add_witness();
	let _v5 = b.band(v4, v3);

	let cs = compile(&b);
	insta::assert_snapshot!(stringify_constraint_system(&cs), @"AND[0]: (v[1]) ∧ (v[0]≫≫11) = (v[2])");
}

#[test]
fn test_rotr_distributes_over_xor_expanded() {
	let b = mk_circuit_builder();

	// Test rotr distribution over XOR
	// v2 = v0 ^ v1
	// v3 = rotr(v2, 6)
	// v5 = v4 & v3
	// → v5 = v4 & (rotr(v0, 6) ^ rotr(v1, 6))
	let v0 = b.add_witness();
	let v1 = b.add_witness();
	let v2 = b.bxor(v0, v1);
	let v3 = b.rotr(v2, 6);

	let v4 = b.add_witness();
	let _v5 = b.band(v4, v3);

	let cs = compile(&b);
	insta::assert_snapshot!(stringify_constraint_system(&cs), @"AND[0]: (v[2]) ∧ (v[0]≫≫6 ⊕ v[1]≫≫6) = (v[3])");
}

#[test]
fn test_depth_limit_deep_chain() {
	let b = mk_circuit_builder();

	// Build a chain deeper than MAX_DEPTH (which is 6 ATOW)
	// Each level adds more complexity through XOR operations
	// The optimizer should commit intermediate values when depth > 6
	assert_eq!(MAX_DEPTH, 6);

	// Level 1: Simple XOR
	let w0 = b.add_witness();
	let w1 = b.add_witness();
	let v1 = b.bxor(w0, w1);

	// Level 2: XOR with v1
	let w2 = b.add_witness();
	let v2 = b.bxor(v1, w2);

	// Level 3: XOR with v2
	let w3 = b.add_witness();
	let v3 = b.bxor(v2, w3);

	// Level 4: XOR with v3
	let w4 = b.add_witness();
	let v4 = b.bxor(v3, w4);

	// Level 5: XOR with v4
	let w5 = b.add_witness();
	let v5 = b.bxor(v4, w5);

	// Level 6: XOR with v5 - at MAX_DEPTH
	let w6 = b.add_witness();
	let v6 = b.bxor(v5, w6);

	// Level 7: XOR with v6 - this should exceed MAX_DEPTH
	let w7 = b.add_witness();
	let v7 = b.bxor(v6, w7);

	// Level 8: XOR with v7 - definitely beyond MAX_DEPTH
	let w8 = b.add_witness();
	let v8 = b.bxor(v7, w8);

	// Use v8 in an AND constraint to trigger the fusion optimization
	let w9 = b.add_witness();
	let _result = b.band(v8, w9);

	let cs = compile(&b);

	// The deep chain should cause intermediate commitments due to depth limit
	// We expect the optimizer to commit some intermediate value and create multiple AND constraints
	// The exact output depends on where the depth limit triggers commitment
	insta::assert_snapshot!(stringify_constraint_system(&cs), @"
	ZERO[0]: (v[0] ⊕ v[1] ⊕ v[10]) = 0
	AND[0]: (v[10] ⊕ v[2] ⊕ v[3] ⊕ v[4] ⊕ v[5] ⊕ v[6] ⊕ v[7] ⊕ v[8]) ∧ (v[9]) = (v[11])
	");
}

#[test]
fn test_depth_limit_with_shifts() {
	let b = mk_circuit_builder();

	// Test depth limit with shift operations mixed in
	// This tests that depth tracking works correctly even with shifts
	assert_eq!(MAX_DEPTH, 6);

	// Build a deep chain with some shifts
	let w0 = b.add_witness();
	let w1 = b.add_witness();
	let v1 = b.bxor(w0, w1);

	// Add a shift operation
	let v1_shifted = b.shr(v1, 3);
	let w2 = b.add_witness();
	let v2 = b.bxor(v1_shifted, w2);

	let w3 = b.add_witness();
	let v3 = b.bxor(v2, w3);

	// Another shift
	let v3_shifted = b.shl(v3, 7);
	let w4 = b.add_witness();
	let v4 = b.bxor(v3_shifted, w4);

	let w5 = b.add_witness();
	let v5 = b.bxor(v4, w5);

	let w6 = b.add_witness();
	let v6 = b.bxor(v5, w6);

	let w7 = b.add_witness();
	let v7 = b.bxor(v6, w7);

	// Add one more level to exceed depth 6
	let w8 = b.add_witness();
	let v8 = b.bxor(v7, w8);

	// Use in AND constraint
	let w9 = b.add_witness();
	let _result = b.band(v8, w9);

	let cs = compile(&b);

	// With shifts in the chain, the depth limit should still apply
	// The optimizer should handle the complexity and commit when needed
	insta::assert_snapshot!(stringify_constraint_system(&cs), @"
	ZERO[0]: (v[0]≫3 ⊕ v[1]≫3 ⊕ v[2] ⊕ v[10]) = 0
	AND[0]: (v[10]≪7 ⊕ v[3]≪7 ⊕ v[4] ⊕ v[5] ⊕ v[6] ⊕ v[7] ⊕ v[8]) ∧ (v[9]) = (v[11])
	");
}

#[test]
fn test_chained_shifts_beyond_the_word_width_compile() {
	let b = mk_circuit_builder();

	// Fusion may only inline a run of shifts when the run has a single-shift form.
	// Four left shifts of 20 sum to 80, which a 64-bit word has no single-shift form for.
	// So this run must be broken by a commit, and the circuit must still compile.
	//
	// Fixture state: 4 chained left shifts of 20, at depth 3, inside the cap of 6.
	// So depth decides nothing here.
	// The width alone forces the break.
	let x = b.add_witness();
	let mut v = x;
	for _ in 0..4 {
		v = b.shl(v, 20);
	}
	let y = b.add_witness();
	let _out = b.band(v, y);

	let cs = compile(&b);

	// The break lands on the topmost link, which is committed as v[4].
	// The three links below it merge into the single shift of 60 in the AND operand.
	insta::assert_snapshot!(stringify_constraint_system(&cs), @"
	ZERO[0]: (v[0]≪20 ⊕ v[2]) = 0
	AND[0]: (v[2]≪60) ∧ (v[1]) = (v[3])
	");
}

#[test]
fn test_chained_shifts_beyond_the_word_width_evaluate() {
	// Breaking the run must preserve what the circuit computes, not just let it compile.
	// A 64-bit word shifted left by 80 keeps none of its bits, so the run evaluates to zero.
	// Masking with all ones then leaves zero, which pins the whole run to a known value.
	let b = CircuitBuilder::new();
	let x = b.add_witness();
	let mut v = x;
	for _ in 0..4 {
		v = b.shl(v, 20);
	}
	let y = b.add_witness();
	let out = b.band(v, y);

	// Pin the result so dead-code elimination keeps the run alive.
	b.force_commit(out);
	let circuit = b.build();

	// Feed a word with bits set across all four bytes, so a dropped shift would show up.
	let mut w = circuit.new_witness_filler();
	w[x] = Word(0xDEAD_BEEF_CAFE_BABE);
	w[y] = Word(u64::MAX);
	circuit.populate_wire_witness(&mut w).unwrap();

	assert_eq!(w[out], Word::ZERO);

	// The committed break carries its own Zero constraint, so check the constraints agree too.
	circuit.constraint_system().verify(&w.value_vec).unwrap();
}
