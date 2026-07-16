// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use binius_core::{ValueIndex, ValueVec, ValueVecLayout, word::Word};

use crate::compiler::{
	circuit::PopulateError,
	eval_form::{
		BytecodeBuilder,
		interpreter::{ExecutionContext, Interpreter},
	},
	hints::HintRegistry,
};

/// Test harness for interpreter tests that makes them much more concise
struct InterpreterTest {
	builder: BytecodeBuilder,
	values: Vec<Word>,
}

impl InterpreterTest {
	fn new() -> Self {
		Self {
			builder: BytecodeBuilder::new(),
			values: Vec::new(),
		}
	}

	/// Set the witness values that will be used in the test
	fn with_values(mut self, values: Vec<Word>) -> Self {
		self.values = values;
		self
	}

	/// Emit an assert_eq_cond instruction
	fn assert_eq_cond(mut self, cond: u32, x: u32, y: u32) -> Self {
		self.builder.emit_assert_eq_cond(cond, x, y, 1);
		self
	}

	/// Emit a select instruction
	fn select(mut self, dst: u32, a: u32, b: u32, cond: u32) -> Self {
		self.builder.emit_select(dst, a, b, cond);
		self
	}

	/// Emit a bmul (GHASH-field multiply) instruction
	fn bmul(
		mut self,
		dst_lo: u32,
		dst_hi: u32,
		a_lo: u32,
		a_hi: u32,
		b_lo: u32,
		b_hi: u32,
	) -> Self {
		self.builder
			.emit_bmul(dst_lo, dst_hi, a_lo, a_hi, b_lo, b_hi);
		self
	}

	/// Run the test and expect success (no assertion failures)
	fn expect_success(self) {
		let (result, ctx) = self.execute();
		assert!(result.is_ok(), "Interpreter should execute successfully");
		assert!(ctx.check_assertions(None).is_ok(), "Should have no assertion failures");
	}

	/// Run the test and expect assertion failure
	fn expect_assertion_failure(self) {
		let (result, ctx) = self.execute();
		assert!(result.is_ok(), "Interpreter should execute successfully");
		assert!(ctx.check_assertions(None).is_err(), "Should have assertion failures");
	}

	/// Run the test and check that specific values match expectations
	fn expect_values(self, expected: Vec<(u32, Word)>) {
		let (bytecode, _) = self.builder.finalize();

		// Create value vec with the right size
		let n_witness = self.values.len();
		let mut value_vec = ValueVec::new(ValueVecLayout {
			n_const: 0,
			n_inout: 0,
			n_witness,
			n_internal: 0,
			offset_inout: 0,
			offset_witness: 0,
			n_hidden_words: n_witness,
			n_scratch: 0,
		});

		// Set the values
		for (i, value) in self.values.into_iter().enumerate() {
			value_vec[ValueIndex(i as u32)] = value;
		}

		let hint_registry = HintRegistry::new();
		let mut interpreter = Interpreter::new(&bytecode, &hint_registry);
		let mut ctx = ExecutionContext::new(&mut value_vec);

		let result = interpreter.run(&mut ctx);
		assert!(result.is_ok(), "Interpreter should execute successfully");

		// Check the expected values
		for (idx, expected_value) in expected {
			let actual = value_vec[ValueIndex(idx)];
			assert_eq!(
				actual, expected_value,
				"Wire {} should have value {:?}, got {:?}",
				idx, expected_value, actual
			);
		}
	}

	/// Execute the bytecode and return the result and context
	fn execute(self) -> (Result<(), PopulateError>, ExecutionContext<'static>) {
		let (bytecode, _) = self.builder.finalize();

		// Create value vec with the right size
		let n_witness = self.values.len();
		let mut value_vec = ValueVec::new(ValueVecLayout {
			n_const: 0,
			n_inout: 0,
			n_witness,
			n_internal: 0,
			offset_inout: 0,
			offset_witness: 0,
			n_hidden_words: n_witness,
			n_scratch: 0,
		});

		// Set the values
		for (i, value) in self.values.into_iter().enumerate() {
			value_vec[ValueIndex(i as u32)] = value;
		}

		let hint_registry = HintRegistry::new();
		let mut interpreter = Interpreter::new(&bytecode, &hint_registry);

		// Leak the value_vec to get 'static lifetime - this is ok in tests
		let value_vec = Box::leak(Box::new(value_vec));
		let mut ctx = ExecutionContext::new(value_vec);

		let result = interpreter.run(&mut ctx);
		(result, ctx)
	}
}

/// Helper to create MSB-true value (MSB set to 1)
fn msb_true(lower_bits: u64) -> Word {
	Word(0x8000000000000000 | lower_bits)
}

/// Helper to create MSB-false value (MSB set to 0)
fn msb_false(lower_bits: u64) -> Word {
	Word(0x7FFFFFFFFFFFFFFF & lower_bits)
}

#[test]
fn test_assert_eq_cond() {
	// MSB=0, values different - should NOT trigger assertion
	InterpreterTest::new()
		.with_values(vec![
			msb_false(0x7FFFFFFFFFFFFFFF), // cond: all bits except MSB
			Word(42),                      // x
			Word(99),                      // y (different)
		])
		.assert_eq_cond(0, 1, 2)
		.expect_success();

	// MSB=1, values equal - should succeed
	InterpreterTest::new()
		.with_values(vec![
			msb_true(0), // cond: only MSB set
			Word(100),   // x
			Word(100),   // y (same)
		])
		.assert_eq_cond(0, 1, 2)
		.expect_success();

	// MSB=1, values different - should FAIL
	InterpreterTest::new()
		.with_values(vec![
			msb_true(0x7FFFFFFFFFFFFFFF), // cond: all bits set
			Word(42),                     // x
			Word(99),                     // y (different)
		])
		.assert_eq_cond(0, 1, 2)
		.expect_assertion_failure();

	// Only MSB matters, not other bits (MSB=0 with other bits set)
	InterpreterTest::new()
		.with_values(vec![
			msb_false(0xFF), // cond: low byte set but MSB=0
			Word(1000),      // x
			Word(2000),      // y (different)
		])
		.assert_eq_cond(0, 1, 2)
		.expect_success();

	// Edge case: MSB=1 with only one other bit
	InterpreterTest::new()
		.with_values(vec![
			msb_true(1), // cond: MSB and LSB set
			Word(5),     // x
			Word(10),    // y (different)
		])
		.assert_eq_cond(0, 1, 2)
		.expect_assertion_failure();
}

#[test]
fn test_select_msb_behavior() {
	// Test that select uses MSB to choose between values
	// select(dst, cond, t, f) writes f to dst if MSB(cond)=0, t if MSB(cond)=1

	// MSB=0 should select 'f' (wire 1)
	InterpreterTest::new()
		.with_values(vec![
			Word(42),        // wire 0: t
			Word(99),        // wire 1: f
			msb_false(0xFF), // wire 2: cond with MSB=0
			Word::ZERO,      // wire 3: dst (will be overwritten)
		])
		.select(3, 2, 0, 1) // dst=3, cond=2, t=0, f=1
		.expect_values(vec![(3, Word(99))]); // dst should have value of f

	// MSB=1 should select 't' (wire 0)
	InterpreterTest::new()
		.with_values(vec![
			Word(42),    // wire 0: t
			Word(99),    // wire 1: f
			msb_true(0), // wire 2: cond with MSB=1
			Word::ZERO,  // wire 3: dst
		])
		.select(3, 2, 0, 1)
		.expect_values(vec![(3, Word(42))]); // dst should have value of t

	// Test with all bits except MSB set (should still select 'f')
	InterpreterTest::new()
		.with_values(vec![
			Word(1),                  // wire 0: t
			Word(2),                  // wire 1: f
			Word(0x7FFFFFFFFFFFFFFF), // wire 2: cond (all bits except MSB)
			Word::ZERO,               // wire 3: dst
		])
		.select(3, 2, 0, 1)
		.expect_values(vec![(3, Word(2))]);

	// Test with all bits set (should select 't')
	InterpreterTest::new()
		.with_values(vec![
			Word(100),                // wire 0: t
			Word(200),                // wire 1: f
			Word(0xFFFFFFFFFFFFFFFF), // wire 2: cond (all bits set)
			Word::ZERO,               // wire 3: dst
		])
		.select(3, 2, 0, 1)
		.expect_values(vec![(3, Word(100))]);
}

// BMUL (GHASH-field multiply) tests. Each field element is a `(lo, hi)` word pair with `lo`
// carrying the coefficients of 1..X^63 and `hi` those of X^64..X^127. Wires 0-3 hold the inputs
// (a_lo, a_hi, b_lo, b_hi) and wires 4-5 receive the product (c_lo, c_hi). The expected values are
// field-theory facts, independent of the implementation.

#[test]
fn test_bmul_identity() {
	// a * 1 = a, where the field element 1 is (lo=1, hi=0).
	InterpreterTest::new()
		.with_values(vec![
			Word(0x0123456789ABCDEF), // a_lo
			Word(0xFEDCBA9876543210), // a_hi
			Word(1),                  // b_lo = 1
			Word::ZERO,               // b_hi
			Word::ZERO,               // c_lo (dst)
			Word::ZERO,               // c_hi (dst)
		])
		.bmul(4, 5, 0, 1, 2, 3)
		.expect_values(vec![(4, Word(0x0123456789ABCDEF)), (5, Word(0xFEDCBA9876543210))]);
}

#[test]
fn test_bmul_zero() {
	// a * 0 = 0.
	InterpreterTest::new()
		.with_values(vec![
			Word(0x0123456789ABCDEF), // a_lo
			Word(0xFEDCBA9876543210), // a_hi
			Word::ZERO,               // b_lo = 0
			Word::ZERO,               // b_hi = 0
			Word(0xdead),             // c_lo (dst, overwritten)
			Word(0xbeef),             // c_hi (dst, overwritten)
		])
		.bmul(4, 5, 0, 1, 2, 3)
		.expect_values(vec![(4, Word::ZERO), (5, Word::ZERO)]);
}

#[test]
fn test_bmul_x_times_x() {
	// X * X = X^2 (no reduction needed): X is (lo=2, hi=0), X^2 is (lo=4, hi=0).
	InterpreterTest::new()
		.with_values(vec![
			Word(2),    // a_lo = X
			Word::ZERO, // a_hi
			Word(2),    // b_lo = X
			Word::ZERO, // b_hi
			Word::ZERO, // c_lo (dst)
			Word::ZERO, // c_hi (dst)
		])
		.bmul(4, 5, 0, 1, 2, 3)
		.expect_values(vec![(4, Word(4)), (5, Word::ZERO)]);
}

#[test]
fn test_bmul_reduction() {
	// X^127 * X = X^128, which reduces via X^128 + X^7 + X^2 + X + 1 = 0 to
	// X^7 + X^2 + X + 1 = 0x87 (lo), 0 (hi). X^127 is bit 63 of the hi word.
	InterpreterTest::new()
		.with_values(vec![
			Word::ZERO,               // a_lo
			Word(0x8000000000000000), // a_hi = X^127
			Word(2),                  // b_lo = X
			Word::ZERO,               // b_hi
			Word::ZERO,               // c_lo (dst)
			Word::ZERO,               // c_hi (dst)
		])
		.bmul(4, 5, 0, 1, 2, 3)
		.expect_values(vec![(4, Word(0x87)), (5, Word::ZERO)]);
}
