// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers
//! Constant evaluation support for gates.

use binius_core::{Word, constraint_system::ShiftVariant};

use super::exec::ghash_mul;
use crate::compiler::{
	gate::Opcode,
	gate_graph::{Gate, GateGraph},
	hints::HintRegistry,
};

/// Evaluates a gate whose inputs are all constant, returning the values of its output wires.
///
/// `constants` holds the values of the gate's input wires, in order. `hint_registry` must contain
/// any hint referenced by the gate. For [`Opcode::Hint`] gates this is the registry populated by
/// [`CircuitBuilder::call_hint`](crate::compiler::CircuitBuilder::call_hint); for other gates an
/// empty registry is fine.
///
/// Assertion gates have no outputs. They return an empty vector when the constant inputs satisfy
/// the assertion, and the violation message otherwise.
pub fn evaluate_gate_constants(
	graph: &GateGraph,
	gate: Gate,
	constants: &[Word],
	hint_registry: &HintRegistry,
) -> Result<Vec<Word>, String> {
	let data = &graph.gates[gate];
	match data.opcode {
		Opcode::Band => {
			let [x, y] = constants else { unreachable!() };
			Ok(vec![*x & *y])
		}
		Opcode::Bor => {
			let [x, y] = constants else { unreachable!() };
			Ok(vec![*x | *y])
		}
		Opcode::Bxor => {
			let [x, y] = constants else { unreachable!() };
			Ok(vec![*x ^ *y])
		}
		Opcode::BxorMulti => Ok(vec![constants.iter().fold(Word::ZERO, |acc, &x| acc ^ x)]),
		Opcode::Fax => {
			let [x, y, w] = constants else { unreachable!() };
			Ok(vec![(*x & *y) ^ *w])
		}
		Opcode::Select => {
			let [cond, t, f] = constants else {
				unreachable!()
			};
			Ok(vec![if cond.is_msb_true() { *t } else { *f }])
		}
		Opcode::Shift => {
			let [x] = constants else { unreachable!() };
			let [variant, n] = data.immediates[..] else {
				unreachable!()
			};
			let variant = ShiftVariant::from_u8(variant as u8)
				.expect("shift gate carries a valid ShiftVariant discriminant");
			Ok(vec![match variant {
				ShiftVariant::Sll => *x << n,
				ShiftVariant::Slr => *x >> n,
				ShiftVariant::Sar => x.sar(n),
				ShiftVariant::Rotr => x.rotr(n),
				ShiftVariant::Sll32 => x.sll32(n),
				ShiftVariant::Srl32 => x.srl32(n),
				ShiftVariant::Sra32 => x.sra32(n),
				ShiftVariant::Rotr32 => x.rotr32(n),
			}])
		}
		Opcode::IaddCinCout => {
			// The carry in is carried in the MSB.
			let [a, b, cin] = constants else {
				unreachable!()
			};
			let (sum, cout) = a.iadd_cin_cout(*b, *cin >> 63);
			Ok(vec![sum, cout])
		}
		Opcode::Iadd32 => {
			let [a, b] = constants else { unreachable!() };
			let (sum, cout) = a.iadd_cout_32(*b);
			Ok(vec![sum, cout])
		}
		Opcode::Iadd32CinCout => {
			let [a, b, cin] = constants else {
				unreachable!()
			};
			let (sum, cout) = a.iadd32_cin_cout(*b, *cin);
			Ok(vec![sum, cout])
		}
		Opcode::IsubBinBout => {
			// The borrow in is carried in the MSB.
			let [a, b, bin] = constants else {
				unreachable!()
			};
			let (diff, bout) = a.isub_bin_bout(*b, *bin >> 63);
			Ok(vec![diff, bout])
		}
		Opcode::Imul => {
			let [x, y] = constants else { unreachable!() };
			let (hi, lo) = x.imul(*y);
			Ok(vec![hi, lo])
		}
		Opcode::Bmul => {
			let [a_lo, a_hi, b_lo, b_hi] = constants else {
				unreachable!()
			};
			let (c_lo, c_hi) = ghash_mul(*a_lo, *a_hi, *b_lo, *b_hi);
			Ok(vec![c_lo, c_hi])
		}
		Opcode::IcmpUlt => {
			// The borrow word of `¬x + y`: its MSB is set exactly when `x < y`.
			let [x, y] = constants else { unreachable!() };
			let (_sum, bout) = (*x ^ Word::ALL_ONE).iadd_cin_cout(*y, Word::ZERO);
			Ok(vec![bout])
		}
		Opcode::IcmpEq => {
			// `ALL_ONE + (x ^ y)` carries out of the top bit exactly when the operands differ, so
			// inverting the MSB of the carry word yields the equality flag.
			let [x, y] = constants else { unreachable!() };
			let (_sum, cout) = Word::ALL_ONE.iadd_cin_cout(*x ^ *y, Word::ZERO);
			Ok(vec![cout ^ Word::MSB_ONE])
		}
		Opcode::AssertEq => {
			let [x, y] = constants else { unreachable!() };
			if x != y {
				return Err(format!("{x:?} != {y:?}"));
			}
			Ok(Vec::new())
		}
		Opcode::AssertEqCond => {
			let [x, y, cond] = constants else {
				unreachable!()
			};
			if cond.is_msb_true() && x != y {
				return Err(format!("conditional assert: {x:?} != {y:?}"));
			}
			Ok(Vec::new())
		}
		Opcode::AssertZero => {
			let [x] = constants else { unreachable!() };
			if *x != Word::ZERO {
				return Err(format!("{x:?} != 0"));
			}
			Ok(Vec::new())
		}
		Opcode::AssertNonZero => {
			let [x] = constants else { unreachable!() };
			if *x == Word::ZERO {
				return Err(format!("{x:?} == 0"));
			}
			Ok(Vec::new())
		}
		Opcode::AssertFalse => {
			let [x] = constants else { unreachable!() };
			if x.is_msb_true() {
				return Err(format!("{x:?} MSB is true"));
			}
			Ok(Vec::new())
		}
		Opcode::AssertTrue => {
			let [x] = constants else { unreachable!() };
			if x.is_msb_false() {
				return Err(format!("{x:?} MSB is false"));
			}
			Ok(Vec::new())
		}
		Opcode::Hint => {
			let hint_id = data.immediates[0];
			let (_n_in, n_out) = hint_registry.shape(hint_id, &data.dimensions);
			let mut outputs = vec![Word::ZERO; n_out];
			hint_registry.execute(hint_id, &data.dimensions, constants, &mut outputs);
			Ok(outputs)
		}
	}
}

#[cfg(test)]
mod tests {
	use binius_core::Word;

	use super::*;

	/// Helper to create a gate with constant inputs for testing
	fn create_test_gate(opcode: Opcode, input_values: &[Word]) -> (GateGraph, Gate, Vec<Word>) {
		let mut graph = GateGraph::new();
		let root = graph.path_spec_tree.root();

		// Create constant wires for inputs using the proper helper function
		let input_wires: Vec<_> = input_values
			.iter()
			.map(|&val| graph.add_constant(val))
			.collect();

		// Create output wires using the proper helper function
		let outputs: Vec<_> = (0..opcode.shape(&[]).n_out)
			.map(|_| graph.add_witness())
			.collect();

		let gate = graph.emit_gate(root, opcode, input_wires, outputs);

		(graph, gate, input_values.to_vec())
	}

	fn eval(opcode: Opcode, input_values: &[Word]) -> Result<Vec<Word>, String> {
		let (graph, gate, constants) = create_test_gate(opcode, input_values);
		evaluate_gate_constants(&graph, gate, &constants, &HintRegistry::new())
	}

	/// Evaluates a single gate by compiling it to bytecode and running the interpreter.
	///
	/// This is the oracle for the differential tests below: [`evaluate_gate_constants`] and the
	/// bytecode emitters are independent statements of the same gate semantics, so the two must
	/// agree on every input.
	fn eval_via_interpreter(
		graph: &GateGraph,
		gate: Gate,
		constants: &[Word],
		hints: &HintRegistry,
	) -> Result<Vec<Word>, String> {
		use binius_core::{ValueIndex, ValueVec, ValueVecLayout};

		use crate::compiler::{
			eval_form::{BytecodeBuilder, exec::Executor, scalar::ExecutionContext},
			gate,
			gate_graph::Wire,
		};

		let data = &graph.gates[gate];
		let shape = data.shape(hints);
		let wire_count = data.wires.len();
		let layout = ValueVecLayout {
			n_const: shape.const_in.len(),
			n_inout: 0,
			n_witness: 0,
			n_internal: wire_count,
			offset_inout: shape.const_in.len(),
			offset_witness: shape.const_in.len(),
			n_hidden_words: wire_count - shape.const_in.len(),
			n_scratch: 0,
		};
		let mut value_vec = ValueVec::new(&layout);
		for (i, &v) in shape.const_in.iter().chain(constants.iter()).enumerate() {
			value_vec[ValueIndex(i as u32)] = v;
		}
		let wire_to_reg =
			|wire: Wire| -> u32 { data.wires.iter().position(|&w| w == wire).unwrap() as u32 };
		let mut builder = BytecodeBuilder::new();
		gate::emit_gate_bytecode(gate, graph, &mut builder, wire_to_reg, hints);
		let (bytecode, _) = builder.finalize();
		let mut ctx = ExecutionContext::new(&mut value_vec);
		Executor::new(&bytecode, hints).run(&mut ctx);
		ctx.check_assertions(None).map_err(|e| format!("{e:?}"))?;
		let start = shape.const_in.len() + shape.n_in;
		Ok((start..start + shape.n_out)
			.map(|i| value_vec[ValueIndex(i as u32)])
			.collect())
	}

	#[test]
	fn differential_against_interpreter() {
		let vals = [
			Word::ZERO,
			Word::ALL_ONE,
			Word::MSB_ONE,
			Word::from_u64(1),
			Word::from_u64(0x8000000000000000),
			Word::from_u64(0x123456789ABCDEF0),
			Word::from_u64(0xFFFFFFFF),
			Word::from_u64(0x5),
			Word::from_u64(0x9),
		];
		let opcodes = [
			(Opcode::Band, 2),
			(Opcode::Bor, 2),
			(Opcode::Bxor, 2),
			(Opcode::Fax, 3),
			(Opcode::Select, 3),
			(Opcode::IaddCinCout, 3),
			(Opcode::Iadd32, 2),
			(Opcode::Iadd32CinCout, 3),
			(Opcode::IsubBinBout, 3),
			(Opcode::Imul, 2),
			(Opcode::Bmul, 4),
			(Opcode::IcmpUlt, 2),
			(Opcode::IcmpEq, 2),
			(Opcode::AssertEq, 2),
			(Opcode::AssertEqCond, 3),
			(Opcode::AssertZero, 1),
			(Opcode::AssertNonZero, 1),
			(Opcode::AssertFalse, 1),
			(Opcode::AssertTrue, 1),
		];
		let mut checked = 0;
		for (opcode, n_in) in opcodes {
			// Enumerate every combination of `vals` of length `n_in`.
			let total = vals.len().pow(n_in as u32);
			for mut k in 0..total {
				let inputs: Vec<Word> = (0..n_in)
					.map(|_| {
						let v = vals[k % vals.len()];
						k /= vals.len();
						v
					})
					.collect();
				let (graph, gate, constants) = create_test_gate(opcode, &inputs);
				let hints = HintRegistry::new();
				let want = eval_via_interpreter(&graph, gate, &constants, &hints);
				let got = evaluate_gate_constants(&graph, gate, &constants, &hints);
				assert_eq!(
					want.is_ok(),
					got.is_ok(),
					"{opcode:?} {inputs:?}: ok mismatch, interpreter={want:?} match={got:?}"
				);
				if let (Ok(want), Ok(got)) = (want, got) {
					assert_eq!(want, got, "{opcode:?} {inputs:?}");
				}
				checked += 1;
			}
		}
		println!("differential: {checked} cases checked");
	}

	#[test]
	fn differential_shift_and_bxor_multi() {
		let vals = [
			Word::ZERO,
			Word::ALL_ONE,
			Word::MSB_ONE,
			Word::from_u64(1),
			Word::from_u64(0x123456789ABCDEF0),
			Word::from_u64(0xFFFFFFFF),
		];
		let hints = HintRegistry::new();
		let mut checked = 0;

		for variant in 0..8u32 {
			for n in [0u32, 1, 7, 31, 32, 63] {
				for &x in &vals {
					let mut graph = GateGraph::new();
					let root = graph.path_spec_tree.root();
					let input = graph.add_constant(x);
					let out = graph.add_witness();
					let gate = graph.emit_gate_generic(
						root,
						Opcode::Shift,
						[input],
						[out],
						&[],
						&[variant, n],
					);
					let want = eval_via_interpreter(&graph, gate, &[x], &hints);
					let got = evaluate_gate_constants(&graph, gate, &[x], &hints);
					assert_eq!(want, got, "shift variant={variant} n={n} x={x:?}");
					checked += 1;
				}
			}
		}

		for n_in in 1..=5usize {
			for mut k in 0..vals.len().pow(n_in as u32) {
				let inputs: Vec<Word> = (0..n_in)
					.map(|_| {
						let v = vals[k % vals.len()];
						k /= vals.len();
						v
					})
					.collect();
				let mut graph = GateGraph::new();
				let root = graph.path_spec_tree.root();
				let input_wires: Vec<_> = inputs.iter().map(|&v| graph.add_constant(v)).collect();
				let out = graph.add_witness();
				let gate = graph.emit_gate_generic(
					root,
					Opcode::BxorMulti,
					input_wires,
					[out],
					&[n_in],
					&[],
				);
				let want = eval_via_interpreter(&graph, gate, &inputs, &hints);
				let got = evaluate_gate_constants(&graph, gate, &inputs, &hints);
				assert_eq!(want, got, "bxor_multi {inputs:?}");
				checked += 1;
			}
		}
		println!("differential shift/bxor_multi: {checked} cases checked");
	}

	#[test]
	fn test_band_constant_eval() {
		let result =
			eval(Opcode::Band, &[Word::from_u64(0xFF00FF00), Word::from_u64(0x0F0F0F0F)]).unwrap();
		assert_eq!(result[0], Word::from_u64(0x0F000F00));
	}

	#[test]
	fn test_bxor_constant_eval() {
		let result =
			eval(Opcode::Bxor, &[Word::from_u64(0xFF00FF00), Word::from_u64(0x0F0F0F0F)]).unwrap();
		assert_eq!(result[0], Word::from_u64(0xF00FF00F));
	}

	#[test]
	fn test_bor_constant_eval() {
		let result =
			eval(Opcode::Bor, &[Word::from_u64(0xFF00FF00), Word::from_u64(0x0F0F0F0F)]).unwrap();
		assert_eq!(result[0], Word::from_u64(0xFF0FFF0F));
	}

	#[test]
	fn test_imul_constant_eval() {
		// Test IMUL (has 2 outputs: hi, lo)
		let result =
			eval(Opcode::Imul, &[Word::from_u64(0x123456789ABCDEF0), Word::from_u64(0x10)])
				.unwrap();
		assert_eq!(result[1], Word::from_u64(0x23456789ABCDEF00)); // lo
		assert_eq!(result[0], Word::from_u64(0x1)); // hi
	}

	#[test]
	fn test_bmul_constant_eval() {
		// Test BMUL (4 inputs a_lo, a_hi, b_lo, b_hi; 2 outputs c_lo, c_hi). X^127 * X = X^128,
		// which reduces (X^128 + X^7 + X^2 + X + 1 = 0) to X^7 + X^2 + X + 1 = 0x87.
		let result = eval(
			Opcode::Bmul,
			&[
				Word::ZERO,                         // a_lo
				Word::from_u64(0x8000000000000000), // a_hi = X^127
				Word::from_u64(2),                  // b_lo = X
				Word::ZERO,                         // b_hi
			],
		)
		.unwrap();
		assert_eq!(result[0], Word::from_u64(0x87)); // c_lo
		assert_eq!(result[1], Word::ZERO); // c_hi
	}

	#[test]
	fn test_iadd_cin_cout_constant_eval() {
		// Test with carry in (MSB = 1 means carry bit is 1)
		let result = eval(
			Opcode::IaddCinCout,
			&[
				Word::from_u64(0xFFFFFFFFFFFFFFFF),
				Word::from_u64(0x1),
				Word::from_u64(0x8000000000000000), // carry in (MSB = 1)
			],
		)
		.unwrap();
		assert_eq!(result[0], Word::from_u64(0x1)); // sum: 0xFF...FF + 1 + 1 = 1 (with overflow)
		// Carry out shows carries at all bit positions
		assert_eq!(result[1], Word::from_u64(0xFFFFFFFFFFFFFFFF));
	}

	#[test]
	fn test_isub_bin_bout_constant_eval() {
		// Test subtraction: 0x10 - 0x5 = 0xB
		let result = eval(
			Opcode::IsubBinBout,
			&[
				Word::from_u64(0x10),
				Word::from_u64(0x5),
				Word::from_u64(0x0), // no borrow in
			],
		)
		.unwrap();
		assert_eq!(result[0], Word::from_u64(0xB)); // diff: 0x10 - 0x5 = 0xB
		// Borrow out shows borrows at bit positions - for 0x10 - 0x5, borrows occur at bits 0-3
		assert_eq!(result[1], Word::from_u64(0xF)); // borrow out at bits 0-3
	}

	#[test]
	fn test_icmp_eq_constant_eval() {
		let equal = eval(Opcode::IcmpEq, &[Word::from_u64(0x7), Word::from_u64(0x7)]).unwrap();
		assert!(equal[0].is_msb_true());

		let differ = eval(Opcode::IcmpEq, &[Word::from_u64(0x7), Word::from_u64(0x8)]).unwrap();
		assert!(differ[0].is_msb_false());
	}

	#[test]
	fn test_icmp_ult_constant_eval() {
		let less = eval(Opcode::IcmpUlt, &[Word::from_u64(0x5), Word::from_u64(0x9)]).unwrap();
		assert!(less[0].is_msb_true());

		let greater = eval(Opcode::IcmpUlt, &[Word::from_u64(0x9), Word::from_u64(0x5)]).unwrap();
		assert!(greater[0].is_msb_false());

		let equal = eval(Opcode::IcmpUlt, &[Word::from_u64(0x9), Word::from_u64(0x9)]).unwrap();
		assert!(equal[0].is_msb_false());
	}

	#[test]
	fn test_select_constant_eval() {
		let taken =
			eval(Opcode::Select, &[Word::MSB_ONE, Word::from_u64(0xAA), Word::from_u64(0xBB)])
				.unwrap();
		assert_eq!(taken[0], Word::from_u64(0xAA));

		let not_taken =
			eval(Opcode::Select, &[Word::ZERO, Word::from_u64(0xAA), Word::from_u64(0xBB)])
				.unwrap();
		assert_eq!(not_taken[0], Word::from_u64(0xBB));
	}

	#[test]
	fn test_assert_gates_constant_eval() {
		assert!(eval(Opcode::AssertEq, &[Word::from_u64(7), Word::from_u64(7)]).is_ok());
		assert!(eval(Opcode::AssertEq, &[Word::from_u64(7), Word::from_u64(8)]).is_err());

		assert!(eval(Opcode::AssertZero, &[Word::ZERO]).is_ok());
		assert!(eval(Opcode::AssertZero, &[Word::from_u64(1)]).is_err());

		assert!(eval(Opcode::AssertNonZero, &[Word::from_u64(1)]).is_ok());
		assert!(eval(Opcode::AssertNonZero, &[Word::ZERO]).is_err());

		assert!(eval(Opcode::AssertTrue, &[Word::MSB_ONE]).is_ok());
		assert!(eval(Opcode::AssertTrue, &[Word::ZERO]).is_err());

		assert!(eval(Opcode::AssertFalse, &[Word::ZERO]).is_ok());
		assert!(eval(Opcode::AssertFalse, &[Word::MSB_ONE]).is_err());

		// The condition is the third input; a false condition suppresses the check.
		assert!(
			eval(Opcode::AssertEqCond, &[Word::from_u64(7), Word::from_u64(8), Word::ZERO]).is_ok()
		);
		assert!(
			eval(Opcode::AssertEqCond, &[Word::from_u64(7), Word::from_u64(8), Word::MSB_ONE])
				.is_err()
		);
	}
}
