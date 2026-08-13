// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers
//! Select operation.
//!
//! Returns `out = MSB(cond) ? t : f`.
//!
//! # Algorithm
//!
//! The gate inspects the MSB (Most Significant Bit) of the condition value to select between
//! two inputs. Logically shifting the condition right by 63 leaves that bit alone in the word,
//! so the MSB-bool becomes the GHASH-field scalar `0` or `1` and the selection is the product
//! `out ⊕ f = (cond >> 63) · (t ⊕ f)`.
//!
//! Scaling by `1` is the field identity and scaling by `0` annihilates, so the product needs no
//! reduction: it stays within the low 64 coefficients its factor occupies. Both factors and the
//! product therefore have a zero high word.
//!
//! # Constraints
//!
//! The gate generates 1 BMUL constraint:
//! - `(cond >> 63) · (t ⊕ f) = out ⊕ f`

use crate::{
	eval_form::BytecodeBuilder,
	gates::{EmitCtx, GateKind, OpcodeShape},
	ir::GateParam,
	lower::{ConstraintBuilder, expr},
};

/// One of two words, chosen by the most significant bit of a third.
pub struct Select;

impl GateKind for Select {
	const SHAPE: OpcodeShape = OpcodeShape::new(3, 1);

	fn constrain(gate: GateParam<'_>, cb: &mut ConstraintBuilder) {
		let [cond, t, f] = gate.in_wires();
		let [out] = gate.out_wires();

		// (cond >> 63) · (t ⊕ f) = out ⊕ f, over the GHASH field.
		// Every factor fits in the low word, so each high word is the empty operand.
		cb.bmul(
			expr::srl(cond, 63),
			expr::empty(),
			expr::xor2(t, f),
			expr::empty(),
			expr::xor2(out, f),
			expr::empty(),
		);
	}

	fn emit(gate: GateParam<'_>, ctx: EmitCtx<'_>, bc: &mut BytecodeBuilder) {
		let [cond, t, f] = gate.in_wires();
		let [out] = gate.out_wires();

		bc.emit_select(ctx.reg(out), ctx.reg(cond), ctx.reg(t), ctx.reg(f));
	}
}

#[cfg(test)]
mod tests {
	use binius_core::word::Word;
	use rand::prelude::*;

	use crate::builder::CircuitBuilder;

	#[test]
	fn test_select_basic() {
		// Build a circuit with Select gate
		let builder = CircuitBuilder::new();
		let a = builder.add_inout();
		let b = builder.add_inout();
		let cond = builder.add_inout();
		let actual = builder.select(cond, b, a);
		let expected = builder.add_inout();
		builder.assert_eq("select", actual, expected);
		let circuit = builder.build();

		// Test specific cases
		let test_cases = [
			// (a, b, cond, expected)
			(
				0x1234567890ABCDEF_u64,
				0xFEDCBA0987654321_u64,
				0x7FFFFFFFFFFFFFFF_u64,
				0x1234567890ABCDEF_u64,
			), // MSB=0, select f (a)
			(
				0x1234567890ABCDEF_u64,
				0xFEDCBA0987654321_u64,
				0x8000000000000000_u64,
				0xFEDCBA0987654321_u64,
			), // MSB=1, select t (b)
			(
				0x0000000000000000_u64,
				0xFFFFFFFFFFFFFFFF_u64,
				0xFFFFFFFFFFFFFFFF_u64,
				0xFFFFFFFFFFFFFFFF_u64,
			), // All ones cond, select t (b)
			(
				0xAAAAAAAAAAAAAAAA_u64,
				0x5555555555555555_u64,
				0x0000000000000000_u64,
				0xAAAAAAAAAAAAAAAA_u64,
			), // Zero cond, select f (a)
		];

		for (a_val, b_val, cond_val, expected_val) in test_cases {
			let mut w = circuit.new_witness_filler();
			w[a] = Word(a_val);
			w[b] = Word(b_val);
			w[cond] = Word(cond_val);
			w[expected] = Word(expected_val);
			w.circuit.populate_wire_witness(&mut w).unwrap();

			// Verify constraints
			let cs = circuit.constraint_system();
			cs.verify(&w.into_value_vec()).unwrap();
		}
	}

	#[test]
	fn test_select_random() {
		// Build a circuit with Select gate
		let builder = CircuitBuilder::new();
		let a = builder.add_inout();
		let b = builder.add_inout();
		let cond = builder.add_inout();
		let actual = builder.select(cond, b, a);
		let expected = builder.add_inout();
		builder.assert_eq("select", actual, expected);
		let circuit = builder.build();

		// Test with random values
		let mut rng = StdRng::seed_from_u64(42);
		for _ in 0..1000 {
			let mut w = circuit.new_witness_filler();
			let a_val = rng.next_u64();
			let b_val = rng.next_u64();
			let cond_val = rng.next_u64();

			// Expected value based on MSB of condition
			let expected_val = if (cond_val as i64) < 0 { b_val } else { a_val };

			w[a] = Word(a_val);
			w[b] = Word(b_val);
			w[cond] = Word(cond_val);
			w[expected] = Word(expected_val);
			w.circuit.populate_wire_witness(&mut w).unwrap();

			// Verify constraints
			let cs = circuit.constraint_system();
			cs.verify(&w.into_value_vec()).unwrap();
		}
	}
}
