// Copyright 2026 The Binius Developers

//! Stand-ins for the field gadgets the skeleton does not build.
//!
//! Each computes the right value at witness time and constrains nothing. A hint is the right shape
//! for the stand-in because the evaluator derives it from wires the circuit already holds, so the
//! result is not a circuit input and the replay does not have to know it exists. Replacing one
//! with a real gadget is a local change: keep the hint, add the constraints that pin it.

use binius_core::word::Word;
use binius_field::{
	BinaryField1b as B1, BinaryField128bGhash as B128, ExtensionField,
	arithmetic_traits::InvertOrZero,
};
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

/// The `B1` subfield-coefficient transpose ring switching performs.
///
/// A hint cannot be generic over the subfield: it is registered under a name that has to be one
/// constant, and it receives words rather than typed elements, so it cannot recover the subfield
/// its caller had. This one is therefore `B1` only, which is what ring switching asks for, and
/// `SymbolicElem::square_transpose` checks that before reaching it rather than letting another
/// subfield arrive here and be silently transposed as `B1`.
///
/// A real implementation would do this with a bit-matrix transpose over the wire pairs, which is
/// cheap; this stands in until then.
pub struct SquareTransposeB1Hint;

impl SquareTransposeB1Hint {
	/// The extension degree this hint transposes over.
	pub const DEGREE: usize = <B128 as ExtensionField<B1>>::DEGREE;
}

impl Hint for SquareTransposeB1Hint {
	const NAME: &'static str = "binius_recursion::square_transpose_b1";

	fn shape(&self, _dimensions: &[usize]) -> (usize, usize) {
		(2 * Self::DEGREE, 2 * Self::DEGREE)
	}

	fn execute(&self, _dimensions: &[usize], inputs: &[Word], outputs: &mut [Word]) {
		let mut elems = (0..Self::DEGREE)
			.map(|i| elem_of(&inputs[2 * i..]))
			.collect::<Vec<_>>();
		<B128 as ExtensionField<B1>>::square_transpose(&mut elems);
		for (i, elem) in elems.into_iter().enumerate() {
			write_elem(elem, &mut outputs[2 * i..]);
		}
	}
}
