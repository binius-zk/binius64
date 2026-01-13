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
	let mut pushforward = vec![F::ZERO; eq_kernel.len()];
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

pub fn generate_index_fingerprints<P: PackedField<Scalar = F>, F: Field, const N_LOOKUPS: usize>(
	indices: [&[usize]; N_LOOKUPS],
	fingerprint_scalar: F,
	max_table_log_len: usize,
) -> [FieldBuffer<P>; N_LOOKUPS] {
	let effective_log_len = max_table_log_len.min(usize::BITS as usize);
	let powers = iter::successors(Some(F::ONE), |&prev| Some(prev * fingerprint_scalar))
		.take(effective_log_len)
		.collect::<Vec<_>>();

	const CHUNK_BITS: usize = 8;
	let num_chunks = (effective_log_len + CHUNK_BITS - 1) / CHUNK_BITS;
	let chunk_tables = (0..num_chunks)
		.map(|chunk_idx| {
			let mut bit_powers = [F::ZERO; CHUNK_BITS];
			let chunk_offset = chunk_idx * CHUNK_BITS;
			for bit in 0..CHUNK_BITS {
				let global_bit = chunk_offset + bit;
				if global_bit < effective_log_len {
					bit_powers[bit] = powers[global_bit];
				}
			}

			let mut table = vec![F::ZERO; 1 << CHUNK_BITS];
			for value in 1usize..(1 << CHUNK_BITS) {
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
				let mut acc = F::ZERO;
				let mut remaining = index;
				for chunk_idx in 0..num_chunks {
					let byte = (remaining & 0xFF) as usize;
					acc += chunk_tables[chunk_idx][byte];
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
