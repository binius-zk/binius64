// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers
//! N-way bitwise XOR operation.
//!
//! Returns `z = x0 ^ x1 ^ ... ^ xn`.
//!
//! # Constraints
//!
//! The gate generates 1 linear constraint:
//! - `x0 ⊕ x1 ⊕ ... ⊕ xn = z`

use crate::{
	eval_form::BytecodeBuilder,
	gates::{EmitCtx, GateKind, OpcodeShape},
	ir::GateParam,
	lower::{ConstraintBuilder, WireExprTerm, expr},
};

/// The exclusive-or of any number of words.
///
/// This is the one kind whose arity is not fixed: its single dimension is the input count.
pub struct BxorMulti;

impl GateKind for BxorMulti {
	/// The fixed part of the shape; the input count comes from the dimension.
	const SHAPE: OpcodeShape = OpcodeShape::new(0, 1);

	fn shape(dimensions: &[usize]) -> OpcodeShape {
		let [n_inputs] = <[usize; 1]>::try_from(dimensions)
			.expect("a multi-way exclusive-or carries its input count as its one dimension");
		OpcodeShape::new(n_inputs, 1)
	}

	fn constrain(gate: GateParam<'_>, cb: &mut ConstraintBuilder) {
		let [z] = gate.out_wires();

		// (x0 ⊕ x1 ⊕ ... ⊕ xn) = z
		// The terms are handed over as an iterator, since the operand collects them itself.
		let terms = gate.inputs.iter().map(|&wire| WireExprTerm::from(wire));
		cb.linear(expr::xor_multi(terms), z);
	}

	fn emit(gate: GateParam<'_>, ctx: EmitCtx<'_>, bc: &mut BytecodeBuilder) {
		let [z] = gate.out_wires();

		let input_regs: Vec<u32> = gate.inputs.iter().map(|&wire| ctx.reg(wire)).collect();
		bc.emit_bxor_multi(ctx.reg(z), &input_regs);
	}
}

#[cfg(test)]
mod tests {
	use binius_core::word::Word;
	use rand::prelude::*;

	use crate::builder::CircuitBuilder;

	#[test]
	fn test_bxor_multi() {
		// Test the n-way XOR gate with different input sizes
		let builder = CircuitBuilder::new();

		// Test with 3 inputs
		let a = builder.add_inout();
		let b = builder.add_inout();
		let c = builder.add_inout();
		let result_3 = builder.bxor_multi(&[a, b, c]);
		let expected_3 = builder.add_inout();
		builder.assert_eq("xor3", result_3, expected_3);

		// Test with 4 inputs
		let d = builder.add_inout();
		let result_4 = builder.bxor_multi(&[a, b, c, d]);
		let expected_4 = builder.add_inout();
		builder.assert_eq("xor4", result_4, expected_4);

		// Test with 5 inputs
		let e = builder.add_inout();
		let result_5 = builder.bxor_multi(&[a, b, c, d, e]);
		let expected_5 = builder.add_inout();
		builder.assert_eq("xor5", result_5, expected_5);

		let circuit = builder.build();

		// Test with random values
		let mut rng = StdRng::seed_from_u64(123);
		for _ in 0..1000 {
			let mut w = circuit.new_witness_filler();
			w[a] = Word(rng.random());
			w[b] = Word(rng.random());
			w[c] = Word(rng.random());
			w[d] = Word(rng.random());
			w[e] = Word(rng.random());

			// Expected results
			w[expected_3] = Word(w[a].0 ^ w[b].0 ^ w[c].0);
			w[expected_4] = Word(w[a].0 ^ w[b].0 ^ w[c].0 ^ w[d].0);
			w[expected_5] = Word(w[a].0 ^ w[b].0 ^ w[c].0 ^ w[d].0 ^ w[e].0);

			w.circuit.populate_wire_witness(&mut w).unwrap();
		}
	}

	#[test]
	fn test_bxor_multi_edge_cases() {
		let builder = CircuitBuilder::new();

		// Test with single input (should return the input itself)
		let single = builder.add_inout();
		let result_single = builder.bxor_multi(&[single]);
		assert_eq!(result_single, single, "Single input should return itself");

		// Test with two inputs (should use regular bxor)
		let a = builder.add_inout();
		let b = builder.add_inout();
		let result_2 = builder.bxor_multi(&[a, b]);
		let expected_2 = builder.add_inout();
		builder.assert_eq("xor2", result_2, expected_2);

		let circuit = builder.build();

		// Verify two-input case works correctly
		let mut rng = StdRng::seed_from_u64(456);
		for _ in 0..100 {
			let mut w = circuit.new_witness_filler();
			w[a] = Word(rng.random());
			w[b] = Word(rng.random());
			w[expected_2] = Word(w[a].0 ^ w[b].0);
			w[single] = Word(rng.random());

			w.circuit.populate_wire_witness(&mut w).unwrap();
		}
	}

	#[test]
	#[should_panic(expected = "bxor_multi requires at least one input")]
	fn test_bxor_multi_empty_panic() {
		let builder = CircuitBuilder::new();
		builder.bxor_multi(&[]);
	}
}
