// Copyright 2026 Irreducible Inc.
//! Hint computing the concatenation of a list of fixed-capacity byte vectors.
//!
//! Each input is described by a wire-count dimension `d_i` and contributes `d_i` data wires
//! (8 bytes each, little-endian) plus one `len_bytes` wire. The hint reads the actual byte
//! prefix of each input (per its `len_bytes`) and packs the concatenated bytes back into
//! `sum(d_i)` little-endian output words, zero-padding any trailing space.

use binius_core::Word;

use super::Hint;

pub struct ByteVecConcatHint;

impl ByteVecConcatHint {
	pub const fn new() -> Self {
		Self
	}
}

impl Default for ByteVecConcatHint {
	fn default() -> Self {
		Self::new()
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
