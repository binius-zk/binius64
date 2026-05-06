// Copyright 2025 Irreducible Inc.
use binius_core::word::Word;
use binius_frontend::{CircuitBuilder, Wire};

use crate::multiplexer::single_wire_multiplex;

/// Extracts a slice from an input byte array and returns it as a vector of packed 64-bit words.
///
/// Returns the bytes from `input` starting at `offset` for `len_slice` bytes, packed into
/// `max_n_words` little-endian 64-bit words. Bytes past `len_slice` in the active words and any
/// trailing words past `ceil(len_slice / 8)` are zero.
///
/// # Limitations
/// All size and offset values must fit within 32 bits. Specifically:
/// - `len_input` must be < 2^32
/// - `len_slice` must be < 2^32
/// - `offset` must be < 2^32
/// - `offset + len_slice` must be < 2^32
///
/// These limitations are enforced by the circuit constraints.
///
/// # Arguments
/// * `b` - Circuit builder
/// * `len_input` - Actual input size in bytes
/// * `len_slice` - Actual slice size in bytes
/// * `input` - Input array packed as words (8 bytes per word)
/// * `offset` - Byte offset where slice starts
/// * `max_n_words` - Number of output wires; the maximum slice length in bytes is `max_n_words * 8`
///
/// # Returns
/// A `Vec<Wire>` of length `max_n_words` containing the extracted slice bytes packed in
/// little-endian order. Bytes past `len_slice` are zero.
///
/// # Panics
/// * If `input.len() * 8 > u32::MAX`
/// * If `max_n_words * 8 > u32::MAX`
pub fn slice(
	b: &CircuitBuilder,
	len_input: Wire,
	len_slice: Wire,
	input: &[Wire],
	offset: Wire,
	max_n_words: usize,
) -> Vec<Wire> {
	// Static assertions to ensure maximum sizes fit within 32 bits
	let max_len_input = input.len() << 3;
	let max_len_slice = max_n_words << 3;

	assert!(max_len_input <= u32::MAX as usize, "max_n_input must be < 2^32");
	assert!(max_len_slice <= u32::MAX as usize, "max_n_slice must be < 2^32");

	// Ensure all values fit in 32 bits to prevent overflow in iadd_32
	b.assert_zero("offset_32bit", b.shr(offset, 32));
	b.assert_zero("len_slice_32bit", b.shr(len_slice, 32));
	b.assert_zero("len_input_32bit", b.shr(len_input, 32));

	// Verify bounds: offset + len_slice <= len_input
	let (offset_plus_len_slice, _) = b.iadd(offset, len_slice);
	let in_bounds = b.icmp_ule(offset_plus_len_slice, len_input);
	b.assert_true("bounds_check", in_bounds);

	// Decompose offset = word_offset * 8 + byte_offset
	let word_offset = b.shr(offset, 3); // offset / 8
	let byte_offset = b.band(offset, b.add_constant(Word(7))); // offset % 8

	// For each output word, compute the corresponding bytes of the slice. Trailing positions past
	// `len_slice` are zeroed via the byte mask and the `word_partially_valid` guard.
	let zero = b.add_constant(Word::ZERO);
	(0..max_n_words)
		.map(|slice_idx| {
			let b = b.subcircuit(format!("slice_word[{slice_idx}]"));

			// Check if this word is within the actual slice
			let word_start_bytes = slice_idx << 3;
			let word_partially_valid =
				b.icmp_ult(b.add_constant(Word(word_start_bytes as u64)), len_slice);

			// Calculate which input word(s) we need
			let (input_word_idx, _) = b.iadd(word_offset, b.add_constant(Word(slice_idx as u64)));

			let extracted_word = extract_word(&b, input, input_word_idx, byte_offset);

			// For every word, calculate how many bytes are valid and apply appropriate mask:
			//     bytes_remaining = len_slice - word_start_bytes (mask clamps internally to 8).
			let neg_start = b.add_constant(Word((-(word_start_bytes as i64)) as u64));
			let (bytes_remaining, _) = b.iadd(len_slice, neg_start);
			let mask = create_byte_mask(&b, bytes_remaining);

			let masked_extracted = b.band(extracted_word, mask);
			b.select(word_partially_valid, masked_extracted, zero)
		})
		.collect()
}

/// Extracts a word from the input array at the specified word index and byte offset.
///
/// This function handles both aligned and unaligned word extraction:
/// - **Aligned** (byte_offset = 0): Directly selects the word at `word_idx`
/// - **Unaligned** (byte_offset = 1-7): Combines bytes from two adjacent words
///
/// # Arguments
/// * `b` - Circuit builder
/// * `input` - Array of input words to extract from
/// * `word_idx` - Index of the word to extract
/// * `byte_offset` - Byte offset within the word (0-7)
///
/// # Returns
/// A wire containing the extracted 8-byte word
pub fn extract_word(b: &CircuitBuilder, input: &[Wire], word_idx: Wire, byte_offset: Wire) -> Wire {
	let (next_word_idx, _) = b.iadd(word_idx, b.add_constant(Word(1)));
	// Aligned case: directly select the word
	let aligned_word = single_wire_multiplex(b, input, word_idx);
	let next_word = single_wire_multiplex(b, input, next_word_idx);
	let zero = b.add_constant(Word::ZERO);

	let candidates: Vec<Wire> = (0..8)
		.map(|i| {
			let shifted_aligned = b.shr(aligned_word, i << 3);
			let shifted_next = if i == 0 {
				zero
			} else {
				b.shl(next_word, (8 - i) << 3)
			};
			b.bor(shifted_aligned, shifted_next)
		})
		.collect();
	single_wire_multiplex(b, &candidates, byte_offset)
}

/// Creates a byte mask with the first `n_bytes` bytes set to 0xFF and remaining bytes to 0x00.
///
/// This function generates masks for partial word validation:
/// - n_bytes = 0: 0x0000000000000000
/// - n_bytes = 1: 0x00000000000000FF
/// - n_bytes = 2: 0x000000000000FFFF
/// - ...
/// - n_bytes = 7: 0x00FFFFFFFFFFFFFF
/// - n_bytes ≥ 8: 0xFFFFFFFFFFFFFFFF
///
/// # Arguments
/// * `b` - Circuit builder
/// * `n_bytes` - Number of bytes to include in the mask (0-8 or more)
///
/// # Returns
/// A wire containing the byte mask
pub fn create_byte_mask(b: &CircuitBuilder, n_bytes: Wire) -> Wire {
	// Handle values ≥ 8 by treating them as 8
	let eight = b.add_constant(Word(8));
	let is_lt_eight = b.icmp_ult(n_bytes, eight);
	let all_one = b.add_constant(Word::ALL_ONE);

	let masks: Vec<Wire> = (0..8)
		.map(|i| b.add_constant_64(0x00FFFFFFFFFFFFFF >> ((7 - i) << 3)))
		.collect();
	let in_range_mask = single_wire_multiplex(b, &masks, n_bytes);
	b.select(is_lt_eight, in_range_mask, all_one)
}

#[cfg(test)]
mod tests {
	use binius_core::verify::verify_constraints;
	use binius_frontend::util::pack_bytes_into_wires_le;

	use super::{CircuitBuilder, Wire, Word, slice};

	/// Build a test circuit that takes input + offset wires, calls `slice`, and asserts the
	/// returned bytes equal a separately allocated `expected` byte vector. Returns the wires the
	/// caller needs to populate.
	struct SliceTestSetup {
		builder: CircuitBuilder,
		len_input: Wire,
		len_slice: Wire,
		offset: Wire,
		input: Vec<Wire>,
		expected: Vec<Wire>,
	}

	fn build_slice_check(n_input_words: usize, n_slice_words: usize) -> SliceTestSetup {
		let builder = CircuitBuilder::new();
		let len_input = builder.add_inout();
		let len_slice = builder.add_inout();
		let offset = builder.add_inout();
		let input: Vec<Wire> = (0..n_input_words).map(|_| builder.add_inout()).collect();
		let expected: Vec<Wire> = (0..n_slice_words).map(|_| builder.add_inout()).collect();
		let actual = slice(&builder, len_input, len_slice, &input, offset, n_slice_words);
		for (i, (&a, &e)) in actual.iter().zip(&expected).enumerate() {
			builder.assert_eq(format!("slice_eq[{i}]"), a, e);
		}
		SliceTestSetup {
			builder,
			len_input,
			len_slice,
			offset,
			input,
			expected,
		}
	}

	/// Run a success-case test: pack `input_data` and `expected_slice_data` into the test setup,
	/// run the circuit, and verify constraints.
	fn run_slice_success(
		setup: SliceTestSetup,
		len_input_val: u64,
		len_slice_val: u64,
		offset_val: u64,
		input_data: &[u8],
		expected_slice_data: &[u8],
	) {
		let circuit = setup.builder.build();
		let mut filler = circuit.new_witness_filler();
		filler[setup.len_input] = Word(len_input_val);
		filler[setup.len_slice] = Word(len_slice_val);
		filler[setup.offset] = Word(offset_val);
		pack_bytes_into_wires_le(&mut filler, &setup.input, input_data);
		pack_bytes_into_wires_le(&mut filler, &setup.expected, expected_slice_data);

		circuit.populate_wire_witness(&mut filler).unwrap();
		let cs = circuit.constraint_system();
		verify_constraints(cs, &filler.into_value_vec()).unwrap();
	}

	/// Run a failure-case test: expect `populate_wire_witness` to error.
	fn run_slice_failure(
		setup: SliceTestSetup,
		len_input_val: u64,
		len_slice_val: u64,
		offset_val: u64,
		input_data: &[u8],
		expected_slice_data: &[u8],
	) {
		let circuit = setup.builder.build();
		let mut filler = circuit.new_witness_filler();
		filler[setup.len_input] = Word(len_input_val);
		filler[setup.len_slice] = Word(len_slice_val);
		filler[setup.offset] = Word(offset_val);
		pack_bytes_into_wires_le(&mut filler, &setup.input, input_data);
		pack_bytes_into_wires_le(&mut filler, &setup.expected, expected_slice_data);
		assert!(circuit.populate_wire_witness(&mut filler).is_err());
	}

	#[test]
	fn test_aligned_slice() {
		// 16-byte input, 8-byte slice at offset 0
		let setup = build_slice_check(2, 1);
		let input_data = [
			0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d,
			0x0e, 0x0f,
		];
		let slice_data = [0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07];
		run_slice_success(setup, 16, 8, 0, &input_data, &slice_data);
	}

	#[test]
	fn test_unaligned_slice() {
		// 16-byte input, 8-byte slice at offset 3
		let setup = build_slice_check(2, 1);
		let input_data = [
			0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d,
			0x0e, 0x0f,
		];
		let slice_data = [0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a];
		run_slice_success(setup, 16, 8, 3, &input_data, &slice_data);
	}

	#[test]
	fn test_bounds_check() {
		// offset(5) + len_slice(8) > len_input(10) → bounds check fails.
		let setup = build_slice_check(2, 1);
		let dummy_input = vec![0u8; 10];
		let dummy_slice = vec![0u8; 8];
		run_slice_failure(setup, 10, 8, 5, &dummy_input, &dummy_slice);
	}

	#[test]
	fn test_bounds_check_edge_case() {
		// Exact boundary: offset(5) + len_slice(5) == len_input(10).
		let setup = build_slice_check(2, 1);
		let input_data = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
		let slice_data = vec![5, 6, 7, 8, 9];
		run_slice_success(setup, 10, 5, 5, &input_data, &slice_data);
	}

	#[test]
	fn test_empty_slice() {
		// len_slice = 0 — output is all zeros.
		let setup = build_slice_check(2, 1);
		let input_data = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
		run_slice_success(setup, 10, 0, 5, &input_data, &[]);
	}

	#[test]
	fn test_mismatched_slice_content() {
		// Caller's expected slice differs from the actual extracted bytes — the external
		// `assert_eq` in `build_slice_check` should fail.
		let setup = build_slice_check(2, 1);
		let input_data = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
		// Actual extracted slice at offset 2 with len 5 is [2,3,4,5,6]; we claim [0,1,2,3,4].
		let wrong_slice_data = vec![0, 1, 2, 3, 4];
		run_slice_failure(setup, 10, 5, 2, &input_data, &wrong_slice_data);
	}

	#[test]
	fn test_offset_at_end() {
		// Empty slice at offset 10, where len_input = 10.
		let setup = build_slice_check(2, 1);
		let input_data = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
		run_slice_success(setup, 10, 0, 10, &input_data, &[]);
	}

	#[test]
	fn test_multiple_byte_extraction_paths() {
		// Verify byte extraction works for a range of offsets in a fresh circuit each time.
		// (The expected slice differs per case so we can't reuse one circuit.)
		for word_idx in 0..3 {
			for byte_offset in 0..8 {
				let offset_val = word_idx * 8 + byte_offset;
				if offset_val + 8 > 24 {
					continue;
				}
				let setup = build_slice_check(3, 1);
				let input_data: Vec<u8> = (0..24).map(|i| i as u8).collect();
				let slice_data: Vec<u8> = input_data[offset_val..offset_val + 8].to_vec();
				run_slice_success(setup, 24, 8, offset_val as u64, &input_data, &slice_data);
			}
		}
	}

	#[test]
	fn test_partial_word_zero_padding() {
		// `slice` returns words zero-padded past `len_slice`. For len_slice=12 (1.5 words), the
		// upper 4 bytes of word 1 must be zero — and the assert_eq will catch any drift.
		let setup = build_slice_check(3, 2);
		let input_data = vec![
			0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, // word 0
			0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f, // word 1
			0x10, 0x11, 0x12, 0x13, // partial word 2
		];
		// Expected slice: 12 valid bytes, word-1 high half zero-padded.
		let correct_slice = [
			0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, // word 0
			0x08, 0x09, 0x0a, 0x0b, 0x00, 0x00, 0x00, 0x00, // word 1 padded to 16 bytes
		];
		run_slice_success(setup, 20, 12, 0, &input_data, &correct_slice);
	}

	#[test]
	fn test_partial_word_rejects_garbage_padding() {
		// If the caller's expected slice has nonzero bytes past len_slice, the assert_eq
		// against the (zero-padded) returned wires must fail.
		let setup = build_slice_check(3, 2);
		let input_data = vec![
			0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d,
			0x0e, 0x0f, 0x10, 0x11, 0x12, 0x13,
		];
		let circuit = setup.builder.build();
		let mut filler = circuit.new_witness_filler();
		filler[setup.len_input] = Word(20);
		filler[setup.len_slice] = Word(12);
		filler[setup.offset] = Word(0);
		pack_bytes_into_wires_le(&mut filler, &setup.input, &input_data);
		// Word 0 correct, word 1 has garbage 0xffffffff in the upper half (past len_slice).
		filler[setup.expected[0]] = Word(0x0706050403020100);
		filler[setup.expected[1]] = Word(0xffffffff0b0a0908);
		assert!(circuit.populate_wire_witness(&mut filler).is_err());
	}

	#[test]
	fn test_large_offset_overflow() {
		// offset has bit 32 set → 32-bit assertion fails.
		let setup = build_slice_check(2, 1);
		let circuit = setup.builder.build();
		let mut filler = circuit.new_witness_filler();
		filler[setup.len_input] = Word(10);
		filler[setup.len_slice] = Word(5);
		filler[setup.offset] = Word(1u64 << 32);
		pack_bytes_into_wires_le(&mut filler, &setup.input, &[0u8; 10]);
		pack_bytes_into_wires_le(&mut filler, &setup.expected, &[0u8; 5]);
		assert!(circuit.populate_wire_witness(&mut filler).is_err());
	}

	#[test]
	fn test_32bit_validation() {
		// offset with bit 33 set
		let setup = build_slice_check(2, 1);
		let circuit = setup.builder.build();
		let mut filler = circuit.new_witness_filler();
		filler[setup.len_input] = Word(10);
		filler[setup.len_slice] = Word(5);
		filler[setup.offset] = Word(1u64 << 33);
		pack_bytes_into_wires_le(&mut filler, &setup.input, &[0u8; 10]);
		pack_bytes_into_wires_le(&mut filler, &setup.expected, &[0u8; 5]);
		assert!(circuit.populate_wire_witness(&mut filler).is_err());

		// len_input with upper 32 bits set
		let setup = build_slice_check(2, 1);
		let circuit = setup.builder.build();
		let mut filler = circuit.new_witness_filler();
		filler[setup.len_input] = Word(0xffffffff00000010);
		filler[setup.len_slice] = Word(5);
		filler[setup.offset] = Word(0);
		pack_bytes_into_wires_le(&mut filler, &setup.input, &[0u8; 10]);
		pack_bytes_into_wires_le(&mut filler, &setup.expected, &[0u8; 5]);
		assert!(circuit.populate_wire_witness(&mut filler).is_err());

		// len_slice with bit 32 set
		let setup = build_slice_check(2, 1);
		let circuit = setup.builder.build();
		let mut filler = circuit.new_witness_filler();
		filler[setup.len_input] = Word(10);
		filler[setup.len_slice] = Word(0x100000005);
		filler[setup.offset] = Word(0);
		pack_bytes_into_wires_le(&mut filler, &setup.input, &[0u8; 10]);
		pack_bytes_into_wires_le(&mut filler, &setup.expected, &[0u8; 5]);
		assert!(circuit.populate_wire_witness(&mut filler).is_err());
	}

	#[test]
	fn test_edge_case_len_input_zero() {
		// Empty input + empty slice at offset 0.
		let setup = build_slice_check(2, 1);
		run_slice_success(setup, 0, 0, 0, &[], &[]);
	}

	#[test]
	fn test_edge_case_len_input_zero_with_nonzero_slice() {
		// Empty input + non-empty slice → bounds check fails.
		let setup = build_slice_check(2, 1);
		run_slice_failure(setup, 0, 5, 0, &[], &[1, 2, 3, 4, 5]);
	}

	#[test]
	fn test_padding_beyond_actual_data() {
		// 12 bytes of input data padded into 3 input words; 8-byte slice at offset 2.
		let setup = build_slice_check(3, 2);
		let input_data = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
		// Slice is 8 bytes, so word 1 is fully zero-padded.
		let slice_data = vec![2, 3, 4, 5, 6, 7, 8, 9];
		run_slice_success(setup, 12, 8, 2, &input_data, &slice_data);
	}

	#[test]
	fn test_direct_masking_logic() {
		// Test the masking logic directly (Rust-side, no circuit).
		let slice_word = Word(0xffffffff_0b0a0908);
		let extracted_word = Word(0x00000000_0b0a0908);
		let mask = Word(0x00000000_ffffffff);

		let masked_slice = slice_word & mask;
		let masked_extracted = extracted_word & mask;

		assert_eq!(masked_slice, masked_extracted);
		assert_eq!(masked_slice ^ masked_extracted, Word::ZERO);
	}
}
