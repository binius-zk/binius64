// Copyright 2026 The Binius Developers
//! Small shared circuit gadgets.

use binius_frontend::{CircuitBuilder, Wire};

/// Zero the high `n` bits of a 64-bit word, keeping the low `64 - n` bits in place.
///
/// Lowers to a left-then-right shift pair, which is cheaper than masking with a `band`
/// against a constant.
pub(crate) fn clear_high_bits(builder: &CircuitBuilder, w: Wire, n: u32) -> Wire {
	builder.shr(builder.shl(w, n), n)
}
