use std::iter::{self, zip};

use binius_field::{Field, PackedField};
use binius_math::{FieldBuffer, FieldSlice};

// Generates a pushforward given the eq kernel for the evaluation point of the lookup value multilinear and the slice of indices.
pub fn generate_pushforward<P, F>(
	indices: &[usize],
	eq_kernel: &FieldBuffer<P>,
	table_len: usize,
) -> FieldBuffer<P>
where
	P: PackedField<Scalar = F>,
	F: Field,
{
	assert_eq!(indices.len(), eq_kernel.len());
	let mut pushforward = vec![F::ZERO; table_len];
	for (&idx, eq_i) in zip(indices.iter(), eq_kernel.iter_scalars()) {
		assert!(idx < table_len);
		pushforward[idx] += eq_i
	}

	FieldBuffer::from_values(&pushforward)
}

pub fn generate_lookup_values<P, F>(indices: &[usize], table: &FieldBuffer<P>) -> FieldBuffer<P>
where
	P: PackedField<Scalar = F>,
	F: Field,
{
	let mut lookup_values = vec![F::ZERO; indices.len()];
	for (&idx, lookup) in zip(indices.iter(), lookup_values.iter_mut()) {
		*lookup = table.get(idx);
	}

	FieldBuffer::from_values(&lookup_values)
}

/// Computes per-index fingerprints for multiple lookup index arrays.
///
/// Each index is mapped to a field element by summing `fingerprint_scalar^bit`
/// over all set bits in the index. This is effectively a linear hash of the
/// binary representation of the index with base `fingerprint_scalar`.
///
/// Inputs:
/// - `indices`: `N_LOOKUPS` slices of indices to fingerprint.
/// - `fingerprint_scalar`: base used for per-bit powers.
/// - `max_table_log_len`: maximum number of bits to consider (usually a table
///   log-length); the effective length is clamped to `usize::BITS`.
///
/// Algorithm:
/// - Precompute the powers `fingerprint_scalar^bit` for each bit position.
/// - Build per-byte lookup tables (256 entries) that sum the contributions of
///   the bits in a byte.
/// - For each index, scan bytes from least significant to most significant and
///   accumulate the precomputed contributions, stopping once the remaining
///   bits are zero.
///
/// Notes:
/// - The fingerprint only depends on the lowest `effective_log_len` bits.
/// - The per-byte tables avoid per-bit branching in the hot loop.
pub fn generate_index_fingerprints<P: PackedField<Scalar = F>, F: Field, const N_LOOKUPS: usize>(
	indices: [&[usize]; N_LOOKUPS],
	fingerprint_scalar: F,
	max_table_log_len: usize,
) -> [FieldBuffer<P>; N_LOOKUPS] {
	// Indices are usize, so only the lowest usize::BITS can ever contribute.
	let effective_log_len = max_table_log_len.min(usize::BITS as usize);
	// Precompute fingerprint_scalar^bit for each possible bit position.
	let powers = iter::successors(Some(F::ONE), |&prev| Some(prev * fingerprint_scalar))
		.take(effective_log_len)
		.collect::<Vec<_>>();

	const CHUNK_BITS: usize = 8;
	const CHUNK_SIZE: usize = 1 << CHUNK_BITS;
	const CHUNK_MASK: usize = CHUNK_SIZE - 1;

	// Build per-byte lookup tables so each index is reduced a byte at a time.
	let chunk_tables = powers
		.chunks(CHUNK_BITS)
		.map(|chunk| {
			let mut bit_powers = [F::ZERO; CHUNK_BITS];
			for (bit, &power) in chunk.iter().enumerate() {
				bit_powers[bit] = power;
			}

			// table[value] = sum of bit_powers for all set bits in value.
			let mut table = vec![F::ZERO; CHUNK_SIZE];
			for value in 1usize..CHUNK_SIZE {
				let bit = value.trailing_zeros() as usize;
				table[value] = table[value & (value - 1)] + bit_powers[bit];
			}
			table
		})
		.collect::<Vec<_>>();

	std::array::from_fn(|lookup_idx| {
		let values = indices[lookup_idx]
			.iter()
			.map(|&index| {
				// Sum per-byte contributions of the index's set bits.
				let mut acc = F::ZERO;
				let mut remaining = index;
				for table in chunk_tables.iter() {
					let byte = remaining & CHUNK_MASK;
					acc += table[byte];
					remaining >>= CHUNK_BITS;
					if remaining == 0 {
						break;
					}
				}
				acc
			})
			.collect::<Vec<_>>();
		FieldBuffer::from_values(&values)
	})
}
pub struct Index<'a, P: PackedField, const N_BITS: usize> {
	pub bit_wise_mles: [FieldSlice<'a, P>; N_BITS],
}
