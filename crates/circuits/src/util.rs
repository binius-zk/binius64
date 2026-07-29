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

/// Returns a wire that is all-ones exactly when every wire in `booleans` is all-ones.
///
/// The fold starts from the all-ones constant, so an empty iterator yields all-ones.
pub(crate) fn all_true(builder: &CircuitBuilder, booleans: impl IntoIterator<Item = Wire>) -> Wire {
	booleans
		.into_iter()
		.fold(builder.add_constant(Word::ALL_ONE), |lhs, rhs| builder.band(lhs, rhs))
}
