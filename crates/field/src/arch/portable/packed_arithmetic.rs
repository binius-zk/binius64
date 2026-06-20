// Copyright 2024-2025 Irreducible Inc.

use crate::underlier::UnderlierType;

/// Interleave using the provided even mask slice.
///
/// See [Hacker's Delight](https://dl.acm.org/doi/10.5555/2462741), Section 7-3.
pub fn interleave_with_mask<U: UnderlierType>(
	a: U,
	b: U,
	log_block_len: usize,
	even_mask: &[U],
) -> (U, U) {
	assert!(log_block_len < even_mask.len());

	let block_len = 1 << log_block_len;
	let t = ((a >> block_len) ^ b) & even_mask[log_block_len];
	let c = a ^ t << block_len;
	let d = b ^ t;

	(c, d)
}

/// Generate the mask with ones in the odd packed element positions and zeros in even
macro_rules! interleave_mask_even {
	($underlier:ty, $tower_level:literal) => {{
		let scalar_bits = 1 << $tower_level;

		let mut mask: $underlier = (1 << scalar_bits) - 1;
		let log_width = <$underlier>::LOG_BITS - $tower_level;
		let mut i = 1;
		while i < log_width {
			mask |= mask << (scalar_bits << i);
			i += 1;
		}

		mask
	}};
}

pub(crate) use interleave_mask_even;
