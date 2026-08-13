// Copyright 2026 The Binius Developers
//! Small shared circuit gadgets.

use binius_core::Word;
use binius_frontend::{CircuitBuilder, Wire};

/// Zero the high `n` bits of a 64-bit word, keeping the low `64 - n` bits in place.
///
/// Lowers to a left-then-right shift pair. The two shifts do not compose into one shifted
/// operand, so gate fusion commits the intermediate and spends one constraint on it — the same
/// count masking with a `band` against a constant costs today. The pair is the cheaper form only
/// once a committed linear definition lowers to a Zero constraint rather than an AND constraint.
///
/// # Example
///
/// ```
/// use binius_circuits::util::clear_high_bits;
/// use binius_frontend::CircuitBuilder;
///
/// let builder = CircuitBuilder::new();
/// let word = builder.add_witness();
/// // Keep the low 32 bits, zeroing the high 32.
/// let low_half = clear_high_bits(&builder, word, 32);
/// ```
pub fn clear_high_bits(builder: &CircuitBuilder, w: Wire, n: u32) -> Wire {
	builder.shr(builder.shl(w, n), n)
}

/// Splits each 64-bit wire into two little-endian 32-bit word wires (low half, then high half).
///
/// Returns exactly `num_words` words, zero-padding when `data` runs out.
pub(crate) fn split_u32_words(
	builder: &CircuitBuilder,
	data: &[Wire],
	num_words: usize,
) -> Vec<Wire> {
	let mut words = Vec::with_capacity(num_words);
	for &w in data {
		if words.len() >= num_words {
			break;
		}
		words.push(clear_high_bits(builder, w, 32));
		if words.len() >= num_words {
			break;
		}
		words.push(builder.shr(w, 32));
	}
	while words.len() < num_words {
		words.push(builder.add_constant_64(0));
	}
	words
}

/// Splits a byte vector into `num_words` 32-bit words, forcing every byte at index `>= valid_bytes`
/// to zero.
///
/// The zeroing closes a malleability gap:
/// - a byte vector leaves bytes past its length unconstrained,
/// - a hash compression mixes the whole block, including those bytes,
/// - pinning them stops a prover from choosing the padding to alter the digest.
///
/// Each word is handled by its position relative to the content:
/// - fully inside: passed through,
/// - fully past: replaced by the zero constant,
/// - straddling the boundary: masked to its low valid bytes.
pub(crate) fn zeroed_u32_words(
	builder: &CircuitBuilder,
	data: &[Wire],
	valid_bytes: usize,
	num_words: usize,
) -> Vec<Wire> {
	let raw = split_u32_words(builder, data, num_words);
	let zero = builder.add_constant_64(0);
	(0..num_words)
		.map(|i| {
			let word_start = 4 * i;
			if word_start + 4 <= valid_bytes {
				raw[i]
			} else if word_start >= valid_bytes {
				zero
			} else {
				let keep = valid_bytes - word_start;
				let mask = builder.add_constant_64((1u64 << (keep * 8)) - 1);
				builder.band(raw[i], mask)
			}
		})
		.collect()
}

/// Returns a wire that is all-ones exactly when every wire in `booleans` is all-ones.
///
/// The fold starts from the all-ones constant, so an empty iterator yields all-ones.
pub(crate) fn all_true(builder: &CircuitBuilder, booleans: impl IntoIterator<Item = Wire>) -> Wire {
	booleans
		.into_iter()
		.fold(builder.add_constant(Word::ALL_ONE), |lhs, rhs| builder.band(lhs, rhs))
}
