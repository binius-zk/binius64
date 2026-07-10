// Copyright 2025 Irreducible Inc.
use binius_core::{consts::WORD_SIZE_BYTES, word::Word};
use binius_frontend::{CircuitBuilder, Wire, hints::Hint};

use crate::{
	fixed_byte_vec::{ByteVec, extract_const_range},
	slice::{assert_slice_eq, slice},
};

/// Hint computing the concatenation of a list of fixed-capacity byte vectors.
///
/// Each input is described by a wire-count dimension `d_i` and contributes `d_i` data wires
/// (8 bytes each, little-endian) plus one `len_bytes` wire. The hint reads the actual byte
/// prefix of each input (per its `len_bytes`) and packs the concatenated bytes back into
/// `sum(d_i)` little-endian output words, zero-padding any trailing space.
struct ByteVecConcatHint;

impl ByteVecConcatHint {
	const fn new() -> Self {
		Self
	}
}

impl Hint for ByteVecConcatHint {
	const NAME: &'static str = "binius.byte_vec_concat";

	fn shape(&self, dimensions: &[usize]) -> (usize, usize) {
		let total_data: usize = dimensions.iter().sum();
		(total_data + dimensions.len(), total_data)
	}

	fn execute(&self, dimensions: &[usize], inputs: &[Word], outputs: &mut [Word]) {
		let total_data: usize = dimensions.iter().sum();
		let (data_wires, len_wires) = inputs.split_at(total_data);

		let mut bytes = Vec::with_capacity(total_data * 8);
		let mut cursor = 0;
		for (&d, &len_word) in dimensions.iter().zip(len_wires) {
			let words = &data_wires[cursor..cursor + d];
			cursor += d;
			let len = (len_word.as_u64() as usize).min(d * 8);
			bytes.extend(
				words
					.iter()
					.flat_map(|w| w.as_u64().to_le_bytes())
					.take(len),
			);
		}

		for (out, chunk) in outputs.iter_mut().zip(bytes.chunks(8)) {
			let mut buf = [0u8; 8];
			buf[..chunk.len()].copy_from_slice(chunk);
			*out = Word(u64::from_le_bytes(buf));
		}
		for out in &mut outputs[bytes.len().div_ceil(8)..] {
			*out = Word::ZERO;
		}
	}
}

/// Computes the concatenation of a list of [`ByteVec`]s as a new [`ByteVec`].
///
/// The returned vector has:
/// - capacity (number of data wires) equal to the sum of the inputs' capacities,
/// - runtime length equal to the sum of the inputs' `len_bytes` values, and
/// - `len_range` equal to the sum of the inputs' ranges (`sum(start)..sum(end)`).
///
/// The output data wires are populated by a prover-side concatenation hint; soundness is enforced
/// by constraining each input's region of the output to equal that input's data. How that
/// constraint is emitted depends on what is known at circuit-build time:
///
/// - When a term's byte offset into the output is a compile-time constant (i.e. every preceding
///   term has a constant length), its region is extracted with constant shifts and masks (no
///   multiplexer or dynamic-shift machinery).
/// - When that term *also* has a constant length, the comparison degenerates further to a plain
///   per-word `assert_eq` (with a constant mask on the final partial word), dropping the
///   saturating-diff / variable-shift / select machinery of [`assert_slice_eq`] entirely.
/// - Otherwise (a dynamic offset, once some preceding term has a dynamic length) the term falls
///   back to the fully-dynamic [`slice()`] + [`assert_slice_eq`] path.
///
/// Bytes of `output.data` beyond `output.len_bytes` are unconstrained.
pub fn concat(b: &CircuitBuilder, inputs: &[ByteVec]) -> ByteVec {
	let dimensions: Vec<usize> = inputs.iter().map(|v| v.data.len()).collect();
	let mut hint_inputs: Vec<Wire> = inputs.iter().flat_map(|v| v.data.iter().copied()).collect();
	hint_inputs.extend(inputs.iter().map(|v| v.len_bytes));

	let output_data = b.call_hint(ByteVecConcatHint::new(), &dimensions, &hint_inputs);

	// Running byte offset of the current term into the output, tracked both as a wire (`offset`)
	// and as a compile-time range (`offset_range`). The offset is a compile-time constant exactly
	// when `offset_range` is a point (all preceding terms have constant lengths), in which case the
	// constant value is `offset_range.start`.
	let mut offset = b.add_constant(Word::ZERO);
	let mut offset_range = 0usize..0usize;
	// Cumulative number of output words spanned up to and including the current term; bounds the
	// `input` slice handed to the dynamic `slice` extraction.
	let mut words_upper_bound = 0usize;

	for (i, input) in inputs.iter().enumerate() {
		words_upper_bound += input.data.len();
		let name = format!("subslice eq[{i}]");

		let offset_is_const = offset_range.start == offset_range.end;
		let len_is_const = input.len_range.start() == input.len_range.end();

		// Advance the running offset wire, emitting as few adder gates as possible:
		// - while both offset and length stay constant, keep it a folded constant (no `iadd`);
		// - when the offset is a constant zero (the leading term), `0 + len_bytes = len_bytes`;
		// - otherwise add the runtime length.
		let next_offset = if offset_is_const && len_is_const {
			b.add_constant_64((offset_range.start + input.len_range.start()) as u64)
		} else if offset_is_const && offset_range.start == 0 {
			input.len_bytes
		} else {
			b.iadd(offset, input.len_bytes).0
		};

		if offset_is_const {
			let off = offset_range.start;
			if len_is_const {
				// Constant offset and constant length: a fully static per-word comparison.
				assert_const_slice_eq(
					b,
					&name,
					&output_data,
					off,
					&input.data,
					*input.len_range.start(),
				);
			} else {
				// Constant offset, dynamic length: extract the term's (capacity-sized) region with
				// constant shifts, then mask the comparison to the runtime length.
				let region = extract_const_range(
					b,
					&output_data,
					off..off + input.data.len() * WORD_SIZE_BYTES,
				);
				assert_slice_eq(b, &name, input.len_bytes, &region, &input.data);
			}
		} else {
			// Dynamic offset: fall back to the fully-dynamic slice extraction.
			let sb = b.subcircuit(format!("concat_term[{i}]"));
			let extracted = slice(
				&sb,
				next_offset,
				input.len_bytes,
				&output_data[..words_upper_bound],
				offset,
				input.data.len(),
			);
			assert_slice_eq(b, &name, input.len_bytes, &extracted, &input.data);
		}

		offset = next_offset;
		offset_range = (offset_range.start + input.len_range.start())
			..(offset_range.end + input.len_range.end());
	}

	ByteVec::new_with_len_range(output_data, offset, offset_range.start..=offset_range.end)
}

/// Asserts that `output[offset..offset + len]` equals the first `len` bytes of `input`, where
/// `offset` and `len` are compile-time constants. Lowers to constant-shift extraction plus a plain
/// per-word `assert_eq`, masking the final partial word to `len % 8` bytes (bytes of `input` beyond
/// `len` are ignored, matching [`assert_slice_eq`]'s semantics).
fn assert_const_slice_eq(
	b: &CircuitBuilder,
	name: &str,
	output: &[Wire],
	offset: usize,
	input: &[Wire],
	len: usize,
) {
	// Neither side's final word is zeroed past `len`: `extract_const_range` leaves the extracted
	// word's high bytes as the next term's data, and `input`'s bytes past its length are
	// unconstrained. So on the partial boundary word we mask both sides to the valid `len % 8`
	// bytes before comparing.
	let extracted = extract_const_range(b, output, offset..offset + len);
	let n_words = extracted.len();
	let final_bytes = len % WORD_SIZE_BYTES;
	for (k, &a) in extracted.iter().enumerate() {
		let e = input[k];
		if k + 1 == n_words && final_bytes != 0 {
			let mask = b.add_constant_64((1u64 << (final_bytes * 8)) - 1);
			b.assert_eq(format!("{name}[{k}]"), b.band(a, mask), b.band(e, mask));
		} else {
			b.assert_eq(format!("{name}[{k}]"), a, e);
		}
	}
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

	/// A concat input for [`run_concat_mixed`]: either a constant-length term (built with
	/// `new_const_len`, exercising the static comparison paths) or a dynamic-length term (built
	/// with `new_inout`, exercising the dynamic `slice` path).
	enum Term<'a> {
		Const(&'a [u8]),
		Dyn { bytes: &'a [u8], cap_words: usize },
	}

	impl Term<'_> {
		fn bytes(&self) -> &[u8] {
			match self {
				Term::Const(d) => d,
				Term::Dyn { bytes, .. } => bytes,
			}
		}
	}

	/// Build a concat over a mix of constant- and dynamic-length terms, populate, verify
	/// constraints, and return the concatenated bytes. Lets tests drive the constant-offset /
	/// constant-length branches that the `new_inout`-based `run_concat` never reaches.
	fn run_concat_mixed(terms: &[Term]) -> Result<Vec<u8>> {
		let b = CircuitBuilder::new();
		let inputs: Vec<ByteVec> = terms
			.iter()
			.map(|t| match t {
				Term::Const(d) => {
					let n_words = d.len().div_ceil(8);
					let data: Vec<Wire> = (0..n_words).map(|_| b.add_witness()).collect();
					ByteVec::new_const_len(&b, data, d.len())
				}
				Term::Dyn { cap_words, .. } => ByteVec::new_inout(&b, *cap_words),
			})
			.collect();

		let output = concat(&b, &inputs);

		let circuit = b.build();
		let mut filler = circuit.new_witness_filler();
		for (input, t) in inputs.iter().zip(terms) {
			match t {
				Term::Const(d) => input.populate_data(&mut filler, d),
				Term::Dyn { bytes, .. } => {
					input.populate_len_bytes(&mut filler, bytes.len());
					input.populate_data(&mut filler, bytes);
				}
			}
		}

		circuit
			.populate_wire_witness(&mut filler)
			.map_err(|e| anyhow!("populate_wire_witness: {e}"))?;

		let total_len: usize = terms.iter().map(|t| t.bytes().len()).sum();
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

	#[test]
	fn const_terms_aligned() {
		// All lengths are multiples of 8: constant offsets, no boundary masking.
		let bytes = run_concat_mixed(&[Term::Const(b"01234567"), Term::Const(b"abcdefgh01234567")])
			.unwrap();
		assert_eq!(bytes, b"01234567abcdefgh01234567");
	}

	#[test]
	fn const_terms_unaligned() {
		// Non-8-multiple lengths exercise unaligned constant-offset extraction and the masked
		// final-word comparison.
		let bytes = run_concat_mixed(&[
			Term::Const(b"hello"),
			Term::Const(b"world!"),
			Term::Const(b"abc"),
		])
		.unwrap();
		assert_eq!(bytes, b"helloworld!abc");
	}

	#[test]
	fn const_single_term() {
		let bytes = run_concat_mixed(&[Term::Const(b"solo")]).unwrap();
		assert_eq!(bytes, b"solo");
	}

	#[test]
	fn const_empty_middle_term() {
		let bytes =
			run_concat_mixed(&[Term::Const(b"ab"), Term::Const(b""), Term::Const(b"cd")]).unwrap();
		assert_eq!(bytes, b"abcd");
	}

	#[test]
	fn const_then_dynamic() {
		// First term constant ⇒ second term has a constant offset but dynamic length: the
		// "constant offset, dynamic length" branch.
		let bytes = run_concat_mixed(&[
			Term::Const(b"hdr-"),
			Term::Dyn {
				bytes: b"payload",
				cap_words: 2,
			},
		])
		.unwrap();
		assert_eq!(bytes, b"hdr-payload");
	}

	#[test]
	fn dynamic_then_const() {
		// First term dynamic ⇒ offset is dynamic for the second (constant-length) term, which then
		// takes the fully-dynamic slice path.
		let bytes = run_concat_mixed(&[
			Term::Dyn {
				bytes: b"payload",
				cap_words: 2,
			},
			Term::Const(b"-end"),
		])
		.unwrap();
		assert_eq!(bytes, b"payload-end");
	}

	#[test]
	fn const_mutated_output_fails_constraints() {
		// The constant-length comparison path must still bind the hint output to the inputs:
		// corrupting a constrained output byte should be rejected.
		let b = CircuitBuilder::new();
		let inputs = vec![
			ByteVec::new_const_len(&b, vec![b.add_witness()], 5),
			ByteVec::new_const_len(&b, vec![b.add_witness()], 5),
		];
		let output = concat(&b, &inputs);

		let circuit = b.build();
		let mut filler = circuit.new_witness_filler();
		inputs[0].populate_data(&mut filler, b"hello");
		inputs[1].populate_data(&mut filler, b"world");

		circuit.populate_wire_witness(&mut filler).unwrap();
		// Corrupt a byte within the constrained region of the hint output.
		filler[output.data[0]] = Word(filler[output.data[0]].as_u64() ^ 1);

		let cs = circuit.constraint_system();
		assert!(verify_constraints(cs, &filler.into_value_vec()).is_err());
	}

	#[test]
	fn const_unaligned_then_dynamic() {
		// Unaligned constant offset (5) feeding a dynamic-length term.
		let bytes = run_concat_mixed(&[
			Term::Const(b"hello"),
			Term::Dyn {
				bytes: b" world",
				cap_words: 1,
			},
		])
		.unwrap();
		assert_eq!(bytes, b"hello world");
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

			/// Exercise the constant-length path (`new_const_len` terms) across many length
			/// combinations, especially non-8-multiples that hit the masked boundary-word logic and
			/// unaligned constant-offset extraction.
			#[test]
			fn correct_const_concatenation(
				lens in prop::collection::vec(0..=20usize, 1..=5),
				seed in any::<u64>(),
			) {
				let term_bytes: Vec<Vec<u8>> = lens
					.iter()
					.enumerate()
					.map(|(i, &n)| random_bytes(n, seed.wrapping_add(i as u64)))
					.collect();
				let terms: Vec<Term> = term_bytes.iter().map(|d| Term::Const(d.as_slice())).collect();
				let expected: Vec<u8> = term_bytes.iter().flatten().copied().collect();
				let bytes = run_concat_mixed(&terms).unwrap();
				prop_assert_eq!(bytes, expected);
			}
		}
	}
}
