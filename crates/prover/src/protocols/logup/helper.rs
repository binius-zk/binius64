use std::iter::{self, zip};

use binius_field::{Field, PackedField};
use binius_math::{FieldBuffer, FieldSlice};

/// Builds a pushforward table by accumulating `eq_kernel` values at lookup indices.
///
/// The output has length `table_len` and is zero everywhere except at indices
/// referenced by `indices`, where the corresponding `eq_kernel` values are added.
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

/// Collects lookup values from a table at the specified indices.
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
/// Each index is mapped to:
/// `shift_scalar + sum_{bit set in index} fingerprint_scalar^bit`
/// over bit positions `0..effective_log_len`, where
/// `effective_log_len = min(max_table_log_len, usize::BITS)`.
///
/// This yields a linear hash of the binary representation of the index with
/// base `fingerprint_scalar`, plus an additive offset `shift_scalar`.
///
/// Algorithm:
/// - Precompute per-bit powers of `fingerprint_scalar`.
/// - Group powers into byte-sized chunks and build 256-entry lookup tables.
/// - For each index, scan bytes from least significant to most significant and
///   accumulate the table contributions (plus the shift), stopping once the
///   remaining bits are zero.
///
/// Notes:
/// - Only the lowest `effective_log_len` bits contribute to the fingerprint.
/// - The per-byte tables avoid per-bit branching in the hot loop.
pub fn generate_index_fingerprints<P: PackedField<Scalar = F>, F: Field, const N_LOOKUPS: usize>(
	indices: [&[usize]; N_LOOKUPS],
	fingerprint_scalar: F,
	shift_scalar: F,
	max_table_log_len: usize,
) -> [FieldBuffer<P>; N_LOOKUPS] {
	// Indices are usize, so only the lowest usize::BITS can contribute.
	let effective_log_len = max_table_log_len.min(usize::BITS as usize);
	let bit_powers = iter::successors(Some(F::ONE), |&prev| Some(prev * fingerprint_scalar))
		.take(effective_log_len)
		.collect::<Vec<_>>();
	let byte_tables = build_byte_tables(&bit_powers);

	std::array::from_fn(|lookup_idx| {
		let values = indices[lookup_idx]
			.iter()
			.map(|&index| fingerprint_index(index, shift_scalar, &byte_tables))
			.collect::<Vec<_>>();
		FieldBuffer::from_values(&values)
	})
}

const BYTE_BITS: usize = 8;
const BYTE_SIZE: usize = 1 << BYTE_BITS;
const BYTE_MASK: usize = BYTE_SIZE - 1;

fn build_byte_tables<F: Field>(bit_powers: &[F]) -> Vec<[F; BYTE_SIZE]> {
	bit_powers
		.chunks(BYTE_BITS)
		.map(|chunk| {
			let mut chunk_powers = [F::ZERO; BYTE_BITS];
			for (bit, &power) in chunk.iter().enumerate() {
				chunk_powers[bit] = power;
			}

			let mut table = [F::ZERO; BYTE_SIZE];
			for value in 1usize..BYTE_SIZE {
				let bit = value.trailing_zeros() as usize;
				table[value] = table[value & (value - 1)] + chunk_powers[bit];
			}
			table
		})
		.collect()
}

fn fingerprint_index<F: Field>(index: usize, shift_scalar: F, byte_tables: &[[F; BYTE_SIZE]]) -> F {
	let mut acc = shift_scalar;
	let mut remaining = index;
	for table in byte_tables.iter() {
		let byte = remaining & BYTE_MASK;
		acc += table[byte];
		remaining >>= BYTE_BITS;
		if remaining == 0 {
			break;
		}
	}
	acc
}

/// Holds bit-level MLEs for a lookup index representation.
///
/// Each entry corresponds to a multilinear polynomial for one bit position.
pub struct Index<'a, P: PackedField, const N_BITS: usize> {
	pub bit_wise_mles: [FieldSlice<'a, P>; N_BITS],
}
