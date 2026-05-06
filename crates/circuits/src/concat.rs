// Copyright 2025 Irreducible Inc.
use binius_core::word::Word;
use binius_frontend::{CircuitBuilder, Wire};

use crate::{fixed_byte_vec::ByteVec, multiplexer::single_wire_multiplex, slice::create_byte_mask};

/// Concatenates the given `terms` into a single [`ByteVec`].
///
/// Returns a `ByteVec` whose `data` holds the byte-level concatenation of each term's bytes
/// (packed in little-endian 8-bytes-per-word) and whose `len_bytes` equals the sum of the
/// terms' runtime lengths. Bytes past the runtime length, and any trailing words past
/// `ceil(len_bytes / 8)`, are zero.
///
/// # Arguments
/// * `b` - Circuit builder
/// * `terms` - Slice of terms to concatenate, in order
/// * `max_words` - Number of output wires; if `None`, defaults to the sum of `term.data.len()`
///   across all terms (the smallest size that always fits the concatenation).
///
/// # Constraints
/// * Each term's runtime length must fit in its capacity (`term.len_bytes <= term.data.len() * 8`).
/// * The sum of term lengths must fit in `max_words * 8` bytes.
pub fn concat(b: &CircuitBuilder, terms: &[ByteVec], max_words: Option<usize>) -> ByteVec {
	let max_words = max_words.unwrap_or_else(|| terms.iter().map(|t| t.data.len()).sum());
	let max_bytes = max_words << 3;

	let zero = b.add_constant(Word::ZERO);

	// For each term word we produce two contributions:
	//   - a "low" word that lands at output[word_offset + word_idx]
	//   - a "high" word that lands at output[word_offset + word_idx + 1] (only non-zero when
	//     byte_offset != 0)
	// We accumulate contributions into `output` by XOR; contributions at the same byte position
	// from different term words can never overlap.
	let mut output = vec![zero; max_words];
	let mut offset = zero;
	for (i, term) in terms.iter().enumerate() {
		let b = b.subcircuit(format!("term[{i}]"));

		// Verify the term's runtime length fits in its capacity.
		let too_long = b.icmp_ugt(term.len_bytes, b.add_constant_64((term.data.len() << 3) as u64));
		b.assert_false("term_length_check", too_long);

		let word_offset = b.shr(offset, 3); // offset / 8
		let byte_offset = b.band(offset, b.add_constant(Word(7))); // offset % 8

		for (word_idx, &term_word) in term.data.iter().enumerate() {
			let b = b.subcircuit(format!("word[{word_idx}]"));
			let word_byte_offset = word_idx << 3;

			// word_partially_valid = (word_byte_offset < term.len_bytes)
			let word_partially_valid =
				b.icmp_ult(b.add_constant(Word(word_byte_offset as u64)), term.len_bytes);

			// bytes_remaining = term.len_bytes - word_byte_offset (mask clamps to 8).
			let neg_start = b.add_constant(Word((-(word_byte_offset as i64)) as u64));
			let (bytes_remaining, _) = b.iadd(term.len_bytes, neg_start);
			let mask = create_byte_mask(&b, bytes_remaining);

			// Mask invalid bytes within the term word, and zero out completely if past term
			// boundary (so this word contributes nothing).
			let masked = b.band(term_word, mask);
			let masked = b.select(word_partially_valid, masked, zero);

			// Compute shifted versions for each possible byte_offset (0..7).
			let shifted_low_candidates: Vec<Wire> =
				(0..8u32).map(|i| b.shl(masked, i * 8)).collect();
			let shifted_high_candidates: Vec<Wire> = (0..8u32)
				.map(|i| {
					if i == 0 {
						zero
					} else {
						b.shr(masked, (8 - i) * 8)
					}
				})
				.collect();
			let shifted_low = single_wire_multiplex(&b, &shifted_low_candidates, byte_offset);
			let shifted_high = single_wire_multiplex(&b, &shifted_high_candidates, byte_offset);

			// dest_low = word_offset + word_idx is the output position receiving the low bytes.
			// The high bytes (if any) land at dest_low + 1.
			let (dest_low, _) = b.iadd(word_offset, b.add_constant(Word(word_idx as u64)));

			// Conditionally add contributions to each output position.
			for (p, output_p) in output.iter_mut().enumerate() {
				let p_wire = b.add_constant(Word(p as u64));
				let is_low_dest = b.icmp_eq(p_wire, dest_low);
				let low_contrib = b.select(is_low_dest, shifted_low, zero);
				*output_p = b.bxor(*output_p, low_contrib);

				if p > 0 {
					let p_minus_1 = b.add_constant(Word((p - 1) as u64));
					let is_high_dest = b.icmp_eq(p_minus_1, dest_low);
					let high_contrib = b.select(is_high_dest, shifted_high, zero);
					*output_p = b.bxor(*output_p, high_contrib);
				}
			}
		}

		(offset, _) = b.iadd(offset, term.len_bytes);
	}

	// `offset` now equals the total length. Verify it fits in max_words.
	let total_len = offset;
	let too_long = b.icmp_ugt(total_len, b.add_constant_64(max_bytes as u64));
	b.assert_false("concat_length_check", too_long);

	ByteVec::new(output, total_len)
}

#[cfg(test)]
mod tests {
	use anyhow::{Result, anyhow};
	use binius_core::verify::verify_constraints;
	use binius_frontend::util::pack_bytes_into_wires_le;
	use rand::prelude::*;

	use super::*;

	// Test utilities

	struct ConcatTestSetup {
		builder: CircuitBuilder,
		len_joined: Wire,
		joined: Vec<Wire>,
		terms: Vec<ByteVec>,
	}

	/// Helper to create a concat circuit with given parameters.
	///
	/// Allocates inout wires for the expected joined buffer and each term, builds the concat
	/// gadget, and asserts the returned `ByteVec` equals the expected wires. Returns a
	/// `ConcatTestSetup` so the test can populate the term and expected-joined values.
	fn create_concat_circuit(max_n_joined: usize, term_max_lens: Vec<usize>) -> ConcatTestSetup {
		let b = CircuitBuilder::new();

		let len_joined = b.add_inout();
		let joined: Vec<Wire> = (0..max_n_joined).map(|_| b.add_inout()).collect();

		let terms: Vec<ByteVec> = term_max_lens
			.into_iter()
			.map(|max_len| ByteVec {
				len_bytes: b.add_inout(),
				data: (0..max_len).map(|_| b.add_inout()).collect(),
			})
			.collect();

		// Compute concat and assert its outputs equal the (caller-supplied) expected wires.
		let computed = concat(&b, &terms, Some(max_n_joined));
		b.assert_eq("concat_len", computed.len_bytes, len_joined);
		for (i, (&a, &e)) in computed.data.iter().zip(&joined).enumerate() {
			b.assert_eq(format!("concat_data[{i}]"), a, e);
		}

		ConcatTestSetup {
			builder: b,
			len_joined,
			joined,
			terms,
		}
	}

	/// Helper to test a concatenation scenario.
	///
	/// Sets up a circuit with the given parameters and verifies that the concatenation of
	/// `term_data` equals `expected_joined`.
	fn test_concat(
		max_n_joined: usize,
		term_max_lens: Vec<usize>,
		expected_joined: &[u8],
		term_data: &[&[u8]],
	) -> Result<()> {
		let setup = create_concat_circuit(max_n_joined, term_max_lens);
		let circuit = setup.builder.build();
		let mut filler = circuit.new_witness_filler();

		// Set the expected joined length and bytes.
		filler[setup.len_joined] = Word(expected_joined.len() as u64);
		pack_bytes_into_wires_le(&mut filler, &setup.joined, expected_joined);

		// Set up each term.
		for (i, data_bytes) in term_data.iter().enumerate() {
			setup.terms[i].populate_len_bytes(&mut filler, data_bytes.len());
			setup.terms[i].populate_data(&mut filler, data_bytes);
		}

		circuit.populate_wire_witness(&mut filler)?;

		// Verify constraints.
		let cs = circuit.constraint_system();
		verify_constraints(cs, &filler.into_value_vec())
			.map_err(|msg| anyhow!("verify_constraints: {}", msg))?;

		Ok(())
	}

	// Basic concatenation tests

	#[test]
	fn test_two_terms_concat() {
		// Verify basic two-term concatenation works correctly
		test_concat(2, vec![1, 1], b"helloworld", &[b"hello", b"world"]).unwrap();
	}

	#[test]
	fn test_three_terms_concat() {
		// Verify three-term concatenation maintains correct order
		test_concat(3, vec![1, 1, 1], b"foobarbaz", &[b"foo", b"bar", b"baz"]).unwrap();
	}

	#[test]
	fn test_single_term() {
		// Edge case: single term should equal the joined result
		test_concat(1, vec![1], b"hello", &[b"hello"]).unwrap();
	}

	// Empty term handling tests

	#[test]
	fn test_empty_term() {
		// Verify empty terms are handled correctly in the middle
		test_concat(2, vec![1, 1, 1], b"helloworld", &[b"hello", b"", b"world"]).unwrap();
	}

	#[test]
	fn test_all_terms_empty() {
		// Edge case: all empty terms should produce empty result
		test_concat(1, vec![1, 1], b"", &[b"", b""]).unwrap();
	}

	// Word alignment tests

	#[test]
	fn test_unaligned_terms() {
		// Test terms that don't align to word boundaries
		// This exercises the unaligned word extraction logic
		test_concat(3, vec![1, 2], b"hello12world456", &[b"hello12", b"world456"]).unwrap();
	}

	#[test]
	fn test_single_byte_terms() {
		// Test many small terms to verify offset tracking
		test_concat(1, vec![1, 1, 1, 1, 1], b"abcde", &[b"a", b"b", b"c", b"d", b"e"]).unwrap();
	}

	// Real-world use case tests

	#[test]
	fn test_domain_concat() {
		// Realistic example: concatenating domain name parts
		test_concat(
			4,
			vec![1, 1, 1, 1, 1],
			b"api.example.com",
			&[b"api", b".", b"example", b".", b"com"],
		)
		.unwrap();
	}

	// Error detection tests

	#[test]
	fn test_length_mismatch() {
		// Verify the circuit rejects incorrect length claims
		// Test where claimed length doesn't match actual concatenation
		let setup = create_concat_circuit(2, vec![1, 1]);
		let circuit = setup.builder.build();
		let mut filler = circuit.new_witness_filler();

		// Claim joined is 8 bytes but terms sum to 10
		filler[setup.len_joined] = Word(8);
		pack_bytes_into_wires_le(&mut filler, &setup.joined, b"helloworld");

		setup.terms[0].populate_len_bytes(&mut filler, 5);
		setup.terms[0].populate_data(&mut filler, b"hello");
		setup.terms[1].populate_len_bytes(&mut filler, 5);
		setup.terms[1].populate_data(&mut filler, b"world");

		let result = circuit.populate_wire_witness(&mut filler);
		assert!(result.is_err());
	}

	#[test]
	fn test_full_last_word_rejects_wrong_data() {
		// Verify the circuit correctly rejects wrong data when the last word has 8 bytes

		// Setup: term with 16 bytes (2 full words)
		let correct_data = b"0123456789ABCDEF";
		let wrong_data = b"0123456789ABCDXX"; // Last 2 bytes are wrong
		assert_eq!(correct_data.len(), 16);
		assert_eq!(wrong_data.len(), 16);

		let setup = create_concat_circuit(2, vec![2]);
		let circuit = setup.builder.build();
		let mut filler = circuit.new_witness_filler();

		// Populate with WRONG data in joined array
		filler[setup.len_joined] = Word(16);
		pack_bytes_into_wires_le(&mut filler, &setup.joined, wrong_data);

		// But claim it matches the CORRECT data in the term
		setup.terms[0].populate_len_bytes(&mut filler, 16);
		setup.terms[0].populate_data(&mut filler, correct_data);

		// This should fail since the data doesn't match
		let result = circuit.populate_wire_witness(&mut filler);
		assert!(result.is_err(), "Circuit should reject wrong data");
	}

	#[test]
	fn test_multiple_full_words_rejects_wrong_data() {
		// Test with 32 bytes - verify rejection works for multiple full words
		let correct_data = b"0123456789ABCDEF0123456789ABCDEF";
		let wrong_data = b"0123456789ABCDEF0123456789ABCDXX"; // Last word wrong

		let setup = create_concat_circuit(4, vec![4]);
		let circuit = setup.builder.build();
		let mut filler = circuit.new_witness_filler();

		filler[setup.len_joined] = Word(32);
		pack_bytes_into_wires_le(&mut filler, &setup.joined, wrong_data);
		setup.terms[0].populate_len_bytes(&mut filler, 32);
		setup.terms[0].populate_data(&mut filler, correct_data);

		let result = circuit.populate_wire_witness(&mut filler);

		// Should reject wrong data
		assert!(result.is_err(), "Circuit should reject wrong data");
	}

	// Variable term size tests

	#[test]
	fn test_different_term_max_lens() {
		// Terms can have different maximum sizes
		// This allows efficient circuits when term sizes vary significantly
		test_concat(4, vec![1, 3], b"shorta very long string", &[b"short", b"a very long string"])
			.unwrap();
	}

	#[test]
	fn test_mixed_term_sizes() {
		// Complex example with varied term sizes matching real-world usage
		test_concat(
			6,
			vec![1, 1, 4, 1, 2],
			b"hi.this is a much longer term.bye",
			&[b"hi", b".", b"this is a much longer term", b".", b"bye"],
		)
		.unwrap();
	}

	/// Helper to run a concat test with given data.
	///
	/// - `term_specs`: Vector of (data, max_len) pairs for each term
	/// - `joined_override`: If Some, use this as joined data instead of concatenating terms
	/// - `should_succeed`: Whether we expect the circuit to accept or reject
	fn run_concat_test(
		term_specs: Vec<(Vec<u8>, usize)>,
		joined_override: Option<Vec<u8>>,
		should_succeed: bool,
	) {
		let expected_joined_bytes: Vec<u8> = if joined_override.is_none() {
			term_specs
				.iter()
				.flat_map(|(data_bytes, _)| data_bytes.clone())
				.collect()
		} else {
			joined_override.clone().unwrap()
		};

		let max_n_joined = expected_joined_bytes.len().div_ceil(8);
		let term_max_lens: Vec<usize> = term_specs.iter().map(|(_, max_len)| *max_len).collect();

		let setup = create_concat_circuit(max_n_joined, term_max_lens);
		let circuit = setup.builder.build();
		let mut filler = circuit.new_witness_filler();

		filler[setup.len_joined] = Word(expected_joined_bytes.len() as u64);
		pack_bytes_into_wires_le(&mut filler, &setup.joined, &expected_joined_bytes);

		for (i, (data_bytes, _)) in term_specs.iter().enumerate() {
			setup.terms[i].populate_len_bytes(&mut filler, data_bytes.len());
			setup.terms[i].populate_data(&mut filler, data_bytes);
		}

		let result = circuit.populate_wire_witness(&mut filler);
		if should_succeed {
			assert!(result.is_ok(), "Expected success but got: {result:?}");
		} else {
			assert!(result.is_err(), "Expected failure but succeeded");
		}
	}

	fn random_byte_string(len: usize) -> Vec<u8> {
		let mut rng = StdRng::seed_from_u64(len as u64);
		let mut data = vec![0u8; len];
		rng.fill_bytes(&mut data);
		data
	}

	#[test]
	fn test_extra_data_rejected() {
		let term_specs = vec![(random_byte_string(5), 2), (random_byte_string(5), 2)];

		let mut joined_with_extra: Vec<u8> = term_specs
			.iter()
			.flat_map(|(data_bytes, _)| data_bytes.clone())
			.collect();
		joined_with_extra.push(42); // Add extra byte

		run_concat_test(term_specs, Some(joined_with_extra), false);
	}

	// Property-based tests
	//
	// These tests use proptest to verify the circuit behaves correctly
	// across a wide range of randomly generated inputs.

	#[cfg(test)]
	mod proptest_tests {
		use proptest::prelude::*;
		use rstest::rstest;

		use super::*;

		/// Strategy for generating random byte arrays for term data.
		fn term_data_strategy() -> impl Strategy<Value = Vec<u8>> {
			(0..=24usize).prop_map(random_byte_string)
		}

		/// Strategy for generating term specifications with proper word alignment.
		///
		/// Each term gets a max_len that is:
		/// - At least as large as the actual data
		/// - Rounded up to the nearest multiple of 8
		fn term_specs_strategy() -> impl Strategy<Value = Vec<(Vec<u8>, usize)>> {
			prop::collection::vec(
				term_data_strategy().prop_map(|data| {
					let max_len = (data.len().div_ceil(8) * 8).max(8);
					(data, max_len)
				}),
				1..=3,
			)
		}

		#[rstest]
		#[case(0, 1)]
		#[case(2, 1)]
		#[case(2, 2)]
		#[case(10, 2)]
		#[case(10, 3)]
		#[case(18, 3)]
		fn test_single_term_concatenation(#[case] len: usize, #[case] max_words: usize) {
			// Special case: single term should equal joined
			let data_bytes = random_byte_string(len);
			let term_specs = vec![(data_bytes, max_words)];
			run_concat_test(term_specs, None, true);
		}

		proptest! {
			#[test]
			fn test_correct_concatenation(term_specs in term_specs_strategy()) {
				// Verify correct concatenations are accepted
				run_concat_test(term_specs, None, true);
			}

			#[test]
			fn test_empty_terms_allowed(n_terms in 1usize..=5) {
				// Verify empty terms are handled correctly
				let mut term_specs = vec![];
				for i in 0..n_terms {
					if i % 2 == 0 {
						term_specs.push((vec![], 8));
					} else {
						term_specs.push((vec![b'x'; i], (i.div_ceil(8) * 8).max(8)));
					}
				}
				run_concat_test(term_specs, None, true);
			}

			#[test]
			fn test_wrong_joined_data(term_specs in term_specs_strategy()) {
				// Verify incorrect joined data is rejected
				prop_assume!(!term_specs.is_empty());

				let correct_joined: Vec<u8> = term_specs.iter()
					.flat_map(|(data, _)| data.clone())
					.collect();

				prop_assume!(!correct_joined.is_empty());

				let mut wrong_joined = correct_joined.clone();
				wrong_joined[0] ^= 1; // Flip one bit

				run_concat_test(term_specs, Some(wrong_joined), false);
			}

			#[test]
			fn test_wrong_last_byte(term_specs in term_specs_strategy()) {
				// Test modification of the LAST byte (would catch the bug)
				prop_assume!(!term_specs.is_empty());

				let correct_joined: Vec<u8> = term_specs.iter()
					.flat_map(|(data_bytes, _)| data_bytes.clone())
					.collect();

				prop_assume!(!correct_joined.is_empty());

				let mut wrong_joined = correct_joined.clone();
				let last_idx = wrong_joined.len() - 1;
				wrong_joined[last_idx] ^= 1; // Flip one bit in LAST byte

				run_concat_test(term_specs, Some(wrong_joined), false);
			}


			#[test]
			fn test_wrong_length_rejected(term_specs in term_specs_strategy()) {
				// Test that mismatched lengths are rejected
				prop_assume!(term_specs.len() >= 2);

				let correct_joined: Vec<u8> = term_specs.iter()
					.flat_map(|(data_bytes, _)| data_bytes.clone())
					.collect();

				prop_assume!(!correct_joined.is_empty());

				// Create joined data that's too short
				let short_joined = correct_joined[..correct_joined.len() - 1].to_vec();

				run_concat_test(term_specs, Some(short_joined), false);
			}

			#[test]
			fn test_swapped_terms_rejected(a in term_data_strategy(), b in term_data_strategy()) {
				// Test that swapping terms is detected
				prop_assume!(a != b && !a.is_empty() && !b.is_empty());

				let max_len_a = a.len().div_ceil(8);
				let max_len_b = b.len().div_ceil(8);

				let term_specs = vec![(a.clone(), max_len_a), (b.clone(), max_len_b)];
				let mut swapped_joined = b.clone();
				swapped_joined.extend(&a);

				run_concat_test(term_specs, Some(swapped_joined), false);
			}

			#[test]
			fn test_large_terms(n_terms in 1usize..=3, base_size in 50usize..=200) {
				// Test with larger data sizes
				let mut term_specs = vec![];
				for i in 0..n_terms {
					let size = base_size + i * 10;
					let data = vec![i as u8; size];
					let max_len = size.div_ceil(8);
					term_specs.push((data, max_len));
				}
				run_concat_test(term_specs, None, true);
			}

			#[test]
			fn test_word_boundary_terms(offset in 0usize..8) {
				// Test terms that specifically align/misalign with word boundaries
				let term1 = vec![1u8; offset];
				let term2 = vec![2u8; 8 - offset];
				let term3 = vec![3u8; 16];

				let term_specs = vec![
					(term1, 1),
					(term2, 1),
					(term3, 2),
				];

				run_concat_test(term_specs, None, true);
			}

			#[test]
			fn test_partial_term_data_rejected(term_specs in term_specs_strategy()) {
				// Test that providing partial term data is rejected
				prop_assume!(term_specs.len() >= 2);
				prop_assume!(term_specs[0].0.len() > 1);

				// Build correct joined
				let correct_joined: Vec<u8> = term_specs.iter()
					.flat_map(|(data_bytes, _)| data_bytes.clone())
					.collect();

				// But claim first term is shorter than it actually is
				let mut modified_specs = term_specs.clone();
				let shortened_len_bytes = modified_specs[0].0.len() - 1;
				modified_specs[0].0.truncate(shortened_len_bytes);

				// This should fail because total length won't match
				run_concat_test(modified_specs, Some(correct_joined), false);
			}
		}

		#[test]
		fn test_full_word_terms() {
			// Test terms with lengths that are multiples of 8
			let lengths = vec![1, 2, 3, 4, 5, 6];

			for len in lengths {
				let data_bytes = vec![0x55u8; len << 3]; // Repeated pattern
				let mut wrong_data = data_bytes.clone();
				wrong_data[(len << 3) - 1] = 0xAA; // Change last byte

				let term_specs = vec![(data_bytes.clone(), len)];

				// Should reject wrong data
				run_concat_test(term_specs.clone(), Some(wrong_data.clone()), false);
			}
		}

		// Additional deterministic edge case tests
		#[test]
		fn test_maximum_terms() {
			// Test with many terms to ensure no stack overflow or performance issues
			let term_specs: Vec<(Vec<u8>, usize)> =
				(0..50).map(|i| (vec![i as u8; 2], 1)).collect();
			run_concat_test(term_specs, None, true);
		}

		#[test]
		fn test_all_empty_terms() {
			// Test edge case of all empty terms
			let term_specs = vec![(vec![], 1), (vec![], 1), (vec![], 1)];
			run_concat_test(term_specs, None, true);
		}

		#[test]
		fn test_zero_length_joined_mismatch() {
			// Test when joined is empty but terms aren't
			let term_specs = vec![(vec![1, 2, 3], 1)];
			run_concat_test(term_specs, Some(vec![]), false);
		}
	}
}
