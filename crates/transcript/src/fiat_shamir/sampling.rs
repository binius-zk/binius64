// Copyright 2024-2025 Irreducible Inc.
// Copyright (c) 2024 The Plonky3 authors

//! Traits used to sample random values in a public-coin interactive protocol.
//!
//! These interfaces are taken from [p3_challenger](https://github.com/Plonky3/Plonky3/blob/main/challenger/src/lib.rs) in [Plonky3].
//!
//! Plonky3 is dual-licensed under MIT OR Apache 2.0. We use it under Apache 2.0.
//!
//! [Plonky3]: <https://github.com/plonky3/plonky3>

use std::array;

use bytes::Buf;

#[auto_impl::auto_impl(&mut)]
pub trait CanSample<T> {
	fn sample(&mut self) -> T;

	fn sample_array<const N: usize>(&mut self) -> [T; N] {
		array::from_fn(|_| self.sample())
	}

	fn sample_vec(&mut self, n: usize) -> Vec<T> {
		(0..n).map(|_| self.sample()).collect()
	}
}

#[auto_impl::auto_impl(&mut)]
pub trait CanSampleBits<T> {
	fn sample_bits(&mut self, bits: usize) -> T;
}

/// The widest index a single [`CanSampleBits`] draw can address.
///
/// A draw returns a `u32`, so it addresses at most `2^32` positions.
/// A wider request is clamped to this rather than refused, which is a deliberate contract.
///
/// The clamp is fine for a caller that just wants some bits.
/// It is not fine for one addressing a domain: the clamped draw never reaches the range's tail.
/// Such a caller must hold its own log-size to this ceiling before it draws.
pub const MAX_SAMPLE_BITS: usize = u32::BITS as usize;

pub fn sample_bits_reader<Reader: Buf>(mut reader: Reader, bits: usize) -> u32 {
	let bits = bits.min(MAX_SAMPLE_BITS);

	let bytes_to_sample = size_of::<u32>();

	let mut bytes = [0u8; size_of::<u32>()];

	reader.copy_to_slice(&mut bytes[..bytes_to_sample]);

	let unmasked = u32::from_le_bytes(bytes);
	let mask = 1u32.checked_shl(bits as u32).map_or(u32::MAX, |x| x - 1);
	mask & unmasked
}
