// Copyright 2026 The Binius Developers

//! Stand-ins for the field gadgets the skeleton does not build.
//!
//! Each computes the right value at witness time and constrains nothing. A hint is the right shape
//! for the stand-in because the evaluator derives it from wires the circuit already holds, so the
//! result is not a circuit input and the replay does not have to know it exists. Replacing one
//! with a real gadget is a local change: keep the hint, add the constraints that pin it.

use binius_core::word::Word;
use binius_field::{BinaryField128bGhash as B128, ExtensionField, arithmetic_traits::InvertOrZero};
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

/// The subfield-coefficient transpose ring switching performs.
///
/// `dimensions[0]` is the extension degree, so the hint takes and returns that many elements at
/// two words each. A real implementation would do this with a bit-matrix transpose over the wire
/// pairs, which is cheap; this stands in until then.
pub struct SquareTransposeHint;

impl Hint for SquareTransposeHint {
	const NAME: &'static str = "binius_recursion::square_transpose";

	fn shape(&self, dimensions: &[usize]) -> (usize, usize) {
		let degree = dimensions[0];
		(2 * degree, 2 * degree)
	}

	fn execute(&self, dimensions: &[usize], inputs: &[Word], outputs: &mut [Word]) {
		let degree = dimensions[0];
		let mut elems = (0..degree)
			.map(|i| elem_of(&inputs[2 * i..]))
			.collect::<Vec<_>>();

		// The transpose is over the subfield the degree names. Ring switching only ever asks for
		// `B1`, which is the case the degree of 128 selects.
		match degree {
			d if d == <B128 as ExtensionField<binius_field::BinaryField1b>>::DEGREE => {
				<B128 as ExtensionField<binius_field::BinaryField1b>>::square_transpose(&mut elems);
			}
			1 => {}
			d => panic!("no transpose for extension degree {d}"),
		}

		for (i, elem) in elems.into_iter().enumerate() {
			write_elem(elem, &mut outputs[2 * i..]);
		}
	}
}
