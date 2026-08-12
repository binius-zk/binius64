// Copyright 2026 The Binius Developers

//! Values a circuit cannot compute, supplied at witness time.
//!
//! A circuit has no division, so an inverse arrives as a hint.
//! The evaluator derives it from wires the circuit already holds, which keeps it out of the
//! circuit's inputs and out of the replay's way.
//!
//! A hint on its own constrains nothing, so whoever calls one owes the constraints that pin its
//! result. The inverse gadget does that where it calls this.

use binius_core::word::Word;
use binius_field::{BinaryField128bGhash as B128, arithmetic_traits::InvertOrZero};
use binius_frontend::Hint;

/// Reads a `(lo, hi)` wire pair as a field element.
fn elem_of(words: &[Word]) -> B128 {
	B128::new(((words[1].as_u64() as u128) << 64) | words[0].as_u64() as u128)
}

/// Writes a field element into a `(lo, hi)` wire pair.
fn write_elem(value: B128, words: &mut [Word]) {
	let value = u128::from(value);
	words[0] = Word::from_u64(value as u64);
	words[1] = Word::from_u64((value >> 64) as u64);
}

/// The multiplicative inverse, or zero.
///
/// A real implementation keeps this hint and adds the check that the product is one, which is what
/// makes the inverse binding. Without it a prover may supply anything.
pub struct InvertOrZeroHint;

impl Hint for InvertOrZeroHint {
	const NAME: &'static str = "binius_recursion::invert_or_zero";

	fn shape(&self, _dimensions: &[usize]) -> (usize, usize) {
		(2, 2)
	}

	fn execute(&self, _dimensions: &[usize], inputs: &[Word], outputs: &mut [Word]) {
		write_elem(elem_of(inputs).invert_or_zero(), outputs);
	}
}
