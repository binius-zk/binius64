// Copyright 2025 Irreducible Inc.
use binius_core::word::Word;
use binius_frontend::{CircuitBuilder, Wire, hints::ByteVecConcatHint};

use crate::{
	fixed_byte_vec::ByteVec,
	slice::{assert_slice_eq, slice},
};

/// Computes the concatenation of a list of [`ByteVec`]s as a new [`ByteVec`].
///
/// The returned vector has:
/// - capacity (number of data wires) equal to the sum of the inputs' capacities, and
/// - runtime length equal to the sum of the inputs' `len_bytes` values.
///
/// The output data wires are populated by [`ByteVecConcatHint`]; soundness is enforced by
/// extracting each input's range from the output via [`slice()`] and asserting equality with the
/// input data using [`assert_slice_eq`]. Bytes of `output.data` beyond `output.len_bytes` are
/// unconstrained.
pub fn concat(b: &CircuitBuilder, inputs: &[ByteVec]) -> ByteVec {
	let dimensions: Vec<usize> = inputs.iter().map(|v| v.data.len()).collect();
	let mut hint_inputs: Vec<Wire> = inputs.iter().flat_map(|v| v.data.iter().copied()).collect();
	hint_inputs.extend(inputs.iter().map(|v| v.len_bytes));

	let output_data = b.call_hint(ByteVecConcatHint::new(), &dimensions, &hint_inputs);

	let mut inputs_iter = inputs.iter();
	let Some(first_input) = inputs_iter.next() else {
		let zero = b.add_constant(Word::ZERO);
		return ByteVec::new(output_data, zero);
	};

	// The first input is special because it's aligned. For this one, we don't need to slice the
	// output.
	let mut words_upper_bound = first_input.data.len();
	assert_slice_eq(
		b,
		"subslice eq[0]",
		first_input.len_bytes,
		&output_data[..words_upper_bound],
		&first_input.data,
	);

	let mut offset = first_input.len_bytes;
	for (i, input) in inputs_iter.enumerate() {
		words_upper_bound += input.data.len();
		let (next_offset, _) = b.iadd(offset, input.len_bytes);

		let sb = b.subcircuit(format!("concat_term[{}]", i + 1));
		let extracted = slice(
			&sb,
			next_offset,
			input.len_bytes,
			&output_data[..words_upper_bound],
			offset,
			input.data.len(),
		);

		assert_slice_eq(
			b,
			format!("subslice eq[{}]", i + 1),
			input.len_bytes,
			&extracted,
			&input.data,
		);

		offset = next_offset;
	}

	let output_len_bytes = offset;
	ByteVec::new(output_data, output_len_bytes)
}

#[cfg(test)]
mod tests {
	use anyhow::{Result, anyhow};
	use binius_core::verify::verify_constraints;

	use super::*;

	/// Build a circuit calling [`concat`] with `inputs` of the given capacities (in wires),
	/// populate it from the provided bytes, and return the actual concatenated bytes plus the
	/// circuit-verification result. Inputs use inout wires so the test driver can populate them.
	fn run_concat(input_max_lens: &[usize], input_data: &[&[u8]]) -> Result<Vec<u8>> {
		assert_eq!(input_max_lens.len(), input_data.len());

		let b = CircuitBuilder::new();
		let inputs: Vec<ByteVec> = input_max_lens
			.iter()
			.map(|&n| ByteVec::new_inout(&b, n))
			.collect();
		let output = concat(&b, &inputs);

		let circuit = b.build();
		let mut filler = circuit.new_witness_filler();
		for (input, &data) in inputs.iter().zip(input_data) {
			input.populate_len_bytes(&mut filler, data.len());
			input.populate_data(&mut filler, data);
		}

		circuit
			.populate_wire_witness(&mut filler)
			.map_err(|e| anyhow!("populate_wire_witness: {e}"))?;

		let total_len = input_data.iter().map(|d| d.len()).sum::<usize>();
		let mut bytes = Vec::with_capacity(total_len);
		for &w in &output.data {
			let word = filler[w].as_u64();
			for j in 0..8 {
				bytes.push(((word >> (j * 8)) & 0xff) as u8);
			}
		}
		bytes.truncate(total_len);

		let cs = circuit.constraint_system();
		verify_constraints(cs, &filler.into_value_vec())
			.map_err(|msg| anyhow!("verify_constraints: {msg}"))?;

		Ok(bytes)
	}

	fn assert_concat_eq(input_max_lens: &[usize], input_data: &[&[u8]], expected: &[u8]) {
		let bytes = run_concat(input_max_lens, input_data).unwrap();
		assert_eq!(bytes, expected);
	}

	#[test]
	fn two_terms() {
		assert_concat_eq(&[1, 1], &[b"hello", b"world"], b"helloworld");
	}

	#[test]
	fn three_terms() {
		assert_concat_eq(&[1, 1, 1], &[b"foo", b"bar", b"baz"], b"foobarbaz");
	}

	#[test]
	fn single_term() {
		assert_concat_eq(&[1], &[b"hello"], b"hello");
	}

	#[test]
	fn empty_middle_term() {
		assert_concat_eq(&[1, 1, 1], &[b"hello", b"", b"world"], b"helloworld");
	}

	#[test]
	fn all_terms_empty() {
		assert_concat_eq(&[1, 1], &[b"", b""], b"");
	}

	#[test]
	fn no_inputs() {
		assert_concat_eq(&[], &[], b"");
	}

	#[test]
	fn unaligned_terms() {
		assert_concat_eq(&[1, 2], &[b"hello12", b"world456"], b"hello12world456");
	}

	#[test]
	fn single_byte_terms() {
		assert_concat_eq(&[1, 1, 1, 1, 1], &[b"a", b"b", b"c", b"d", b"e"], b"abcde");
	}

	#[test]
	fn domain_concat() {
		assert_concat_eq(
			&[1, 1, 1, 1, 1],
			&[b"api", b".", b"example", b".", b"com"],
			b"api.example.com",
		);
	}

	#[test]
	fn different_term_max_lens() {
		assert_concat_eq(&[1, 3], &[b"short", b"a very long string"], b"shorta very long string");
	}

	#[test]
	fn mixed_term_sizes() {
		assert_concat_eq(
			&[1, 1, 4, 1, 2],
			&[b"hi", b".", b"this is a much longer term", b".", b"bye"],
			b"hi.this is a much longer term.bye",
		);
	}

	#[test]
	fn many_terms() {
		// 50 two-byte terms.
		let input_max_lens = vec![1usize; 50];
		let data: Vec<Vec<u8>> = (0..50u8).map(|i| vec![i, i]).collect();
		let data_refs: Vec<&[u8]> = data.iter().map(|v| v.as_slice()).collect();
		let expected: Vec<u8> = data.iter().flatten().copied().collect();
		assert_concat_eq(&input_max_lens, &data_refs, &expected);
	}

	#[test]
	fn full_word_terms() {
		// Terms with lengths that are exact multiples of 8.
		assert_concat_eq(&[1, 2], &[b"01234567", b"abcdefgh01234567"], b"01234567abcdefgh01234567");
	}

	#[test]
	fn mutated_output_fails_constraints() {
		// Build and populate a valid concatenation, then mutate one of the hint-produced output
		// data wires. `verify_constraints` should reject because the slice extraction no longer
		// matches the inputs.
		let b = CircuitBuilder::new();
		let inputs = vec![ByteVec::new_inout(&b, 1), ByteVec::new_inout(&b, 1)];
		let output = concat(&b, &inputs);

		let circuit = b.build();
		let mut filler = circuit.new_witness_filler();
		inputs[0].populate_len_bytes(&mut filler, 5);
		inputs[0].populate_data(&mut filler, b"hello");
		inputs[1].populate_len_bytes(&mut filler, 5);
		inputs[1].populate_data(&mut filler, b"world");

		circuit.populate_wire_witness(&mut filler).unwrap();
		// Corrupt one byte of the hint output.
		filler[output.data[0]] = Word(filler[output.data[0]].as_u64() ^ 1);

		let cs = circuit.constraint_system();
		assert!(verify_constraints(cs, &filler.into_value_vec()).is_err());
	}

	#[cfg(test)]
	mod proptests {
		use proptest::prelude::*;
		use rand::{Rng, SeedableRng, rngs::StdRng};

		use super::*;

		fn random_bytes(len: usize, seed: u64) -> Vec<u8> {
			let mut rng = StdRng::seed_from_u64(seed);
			let mut data = vec![0u8; len];
			rng.fill_bytes(&mut data);
			data
		}

		fn term_strategy() -> impl Strategy<Value = (Vec<u8>, usize)> {
			(0..=24usize, any::<u64>()).prop_map(|(len, seed)| {
				let max_len = (len.div_ceil(8)).max(1);
				(random_bytes(len, seed), max_len)
			})
		}

		fn terms_strategy() -> impl Strategy<Value = Vec<(Vec<u8>, usize)>> {
			prop::collection::vec(term_strategy(), 1..=4)
		}

		proptest! {
			#[test]
			fn correct_concatenation(terms in terms_strategy()) {
				let input_max_lens: Vec<usize> = terms.iter().map(|(_, n)| *n).collect();
				let data: Vec<&[u8]> = terms.iter().map(|(d, _)| d.as_slice()).collect();
				let expected: Vec<u8> = data.iter().flat_map(|d| d.iter().copied()).collect();
				let bytes = run_concat(&input_max_lens, &data).unwrap();
				prop_assert_eq!(bytes, expected);
			}
		}
	}
}
