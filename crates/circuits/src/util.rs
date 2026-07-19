// Copyright 2026 The Binius Developers
//! Small shared circuit gadgets.

use binius_core::Word;
use binius_frontend::{CircuitBuilder, Wire};

/// Zero the high `n` bits of a 64-bit word, keeping the low `64 - n` bits in place.
///
/// Lowers to a left-then-right shift pair, which is cheaper than masking with a `band`
/// against a constant.
pub(crate) fn clear_high_bits(builder: &CircuitBuilder, w: Wire, n: u32) -> Wire {
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
