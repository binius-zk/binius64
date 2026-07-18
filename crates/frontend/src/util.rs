// Copyright 2025 Irreducible Inc.

//! Various utilities for circuit building.

use std::iter;

use binius_core::Word;

use crate::compiler::{CircuitBuilder, Wire, circuit::WitnessFiller};

/// Populate the given wires from bytes using little-endian packed 64-bit words.
///
/// If `bytes` is not a multiple of 8, the last word is zero-padded.
///
/// If there are more wires than needed to hold all bytes, the remaining wires
/// are filled with `Word::ZERO`.
///
/// # Panics
/// * If bytes.len() exceeds wires.len() * 8
pub fn pack_bytes_into_wires_le(w: &mut WitnessFiller, wires: &[Wire], bytes: &[u8]) {
	let max_value_size = wires.len() * 8;
	assert!(
		bytes.len() <= max_value_size,
		"bytes length {} exceeds maximum {}",
		bytes.len(),
		max_value_size
	);

	// Pack bytes into words
	for (&wire, chunk) in iter::zip(wires, bytes.chunks(8)) {
		let mut chunk_arr = [0u8; 8];
		chunk_arr[..chunk.len()].copy_from_slice(chunk);
		w[wire] = Word(u64::from_le_bytes(chunk_arr));
	}

	// Zero out remaining words
	for &wire in &wires[bytes.len().div_ceil(8)..] {
		w[wire] = Word::ZERO;
	}
}

/// Returns a BigUint from u64 limbs with little-endian ordering
pub fn num_biguint_from_u64_limbs<I>(limbs: I) -> num_bigint::BigUint
where
	I: IntoIterator,
	I::Item: std::borrow::Borrow<u64>,
	I::IntoIter: ExactSizeIterator,
{
	use std::borrow::Borrow;

	use num_bigint::BigUint;

	let iter = limbs.into_iter();
	// Each u64 becomes two u32s (low word first for little-endian)
	let mut digits = Vec::with_capacity(iter.len() * 2);
	for item in iter {
		let double_digit = *item.borrow();
		// push:
		// - low 32 bits
		// - high 32 bits
		digits.push(double_digit as u32);
		digits.push((double_digit >> 32) as u32);
	}
	BigUint::new(digits)
}

/// Check that all boolean wires in an iterable are simultaneously true.
pub fn all_true(b: &CircuitBuilder, booleans: impl IntoIterator<Item = Wire>) -> Wire {
	booleans
		.into_iter()
		.fold(b.add_constant(Word::ALL_ONE), |lhs, rhs| b.band(lhs, rhs))
}

/// Swap the byte order of the word.
///
/// Breaks the word down to bytes and reassembles in reversed order.
pub fn byteswap(b: &CircuitBuilder, word: Wire) -> Wire {
	let bytes = (0..8).map(|j| {
		let byte = b.extract_byte(word, j as u32);
		b.shl(byte, (56 - 8 * j) as u32)
	});
	bytes
		.reduce(|lhs, rhs| b.bxor(lhs, rhs))
		.expect("Word::BITS > 0")
}
