// Copyright 2025-2026 The Binius Developers
use std::{
	array,
	iter::{self, zip},
	ops::Deref,
};

use binius_field::{Field, PackedField};
use binius_ip_prover::channel::IPProverChannel;
use binius_math::{
	FieldBuffer, FieldSlice, inner_product::inner_product, multilinear::eq::eq_ind_partial_eval,
};
use binius_utils::{
	checked_arithmetics::log2_ceil_usize,
	rayon::iter::{IntoParallelIterator, ParallelIterator},
};
use itertools::{Itertools, concat};

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
	assert_eq!(indices.len().next_power_of_two(), eq_kernel.len());
	let mut pushforward = vec![F::ZERO; table_len];
	for (&idx, eq_i) in zip(indices.iter(), eq_kernel.iter_scalars()) {
		assert!(idx < table_len);
		pushforward[idx] += eq_i
	}

	FieldBuffer::from_values(&pushforward)
}

pub fn batch_pushforwards<P, F, const N_TABLES: usize>(
	push_forwards: &[FieldBuffer<P>; N_TABLES],
) -> FieldBuffer<P>
where
	P: PackedField<Scalar = F>,
	F: Field,
{
	// We assume that all pushforwards are of the same length.
	let pushforward_log_len = push_forwards[0].log_len();
	let batch_next_pow_2 = 1 << (log2_ceil_usize(N_TABLES) + pushforward_log_len);
	let mut batch_pushforward: Vec<F> = push_forwards
		.iter()
		.flat_map(|push_forward| push_forward.iter_scalars())
		.collect();

	batch_pushforward.resize_with(batch_next_pow_2, || F::ZERO);

	FieldBuffer::from_values(&batch_pushforward)
}

/// Collects lookup values from a table at the specified indices.
pub fn generate_lookup_values<P, F>(indices: &[usize], table: &FieldBuffer<P>) -> FieldBuffer<P>
where
	P: PackedField<Scalar = F>,
	F: Field,
{
	// This mirrors how the lookup MLE is formed in the lookup argument.
	let mut lookup_values = vec![F::ZERO; indices.len()];
	for (&idx, lookup) in zip(indices.iter(), lookup_values.iter_mut()) {
		*lookup = table.get(idx);
	}

	FieldBuffer::from_values(&lookup_values)
}

/// Computes the shifted Reed Solomon fingerprint of the enumeration MLE, which is the MLE whose
/// dense representation is simply the natural lexicographic ordering of field elements in the
pub fn generate_enumeration_fingerprint<P: PackedField<Scalar = F>, F: Field>(
	log_len: usize,
	fingerprint_scalar: F,
	shift_scalar: F,
) -> FieldBuffer<P> {
	// Indices are usize, so only the lowest usize::BITS can contribute.
	let bit_powers = iter::successors(Some(F::ONE), |&prev| Some(prev * fingerprint_scalar))
		.take(log_len)
		.collect::<Vec<_>>();
	let byte_tables = build_byte_tables(&bit_powers);
	let enum_fingerprint = (0..1 << log_len)
		.into_par_iter()
		.map(|index| fingerprint_index(index, shift_scalar, &byte_tables))
		.collect::<Vec<_>>();
	FieldBuffer::from_values(&enum_fingerprint)
}

///Concatenates index arrays by table id and fingerprints them. Padding to next power of 2.
pub fn concatenate_and_fingerprint_indexes<
	P: PackedField<Scalar = F>,
	F: Field,
	Index: Deref<Target = [usize]>,
	const N_TABLES: usize,
>(
	indices: &[Index],
	table_ids: &[usize],
	fingerprint_scalar: F,
	shift_scalar: F,
	max_table_log_len: usize,
) -> [FieldBuffer<P>; N_TABLES] {
	// Indices are usize, so only the lowest usize::BITS can contribute.
	let effective_log_len = max_table_log_len.min(usize::BITS as usize);
	let concatenated_indices: [Vec<usize>; N_TABLES] = concatenate_indices(&indices, table_ids);
	let bit_powers = iter::successors(Some(F::ONE), |&prev| Some(prev * fingerprint_scalar))
		.take(effective_log_len)
		.collect::<Vec<_>>();
	let byte_tables = build_byte_tables(&bit_powers);

	std::array::from_fn(|table_id| {
		let mut values = concatenated_indices[table_id]
			.iter()
			.map(|&index| fingerprint_index(index, shift_scalar, &byte_tables))
			.collect::<Vec<_>>();
		values.resize_with(values.len().next_power_of_two(), || F::ZERO);
		FieldBuffer::from_values(&values)
	})
}

/// Concatenates indices by table id. Assumes equal number of lookups per table.
pub fn concatenate_indices<Index: Deref<Target = [usize]>, const N_TABLES: usize>(
	indices: &[Index],
	table_ids: &[usize],
) -> [Vec<usize>; N_TABLES] {
	array::from_fn(|i| {
		table_ids
			.iter()
			.copied()
			.filter(|&j| i == j)
			.flat_map(|j| indices[j].to_vec())
			.collect()
	})
}

pub fn batch_lookup_evals<F: Field, const N_TABLES: usize>(
	lookup_evals: &[F],
	eval_point: &[F],
	table_ids: &[usize],
	channel: &mut impl IPProverChannel<F>,
) -> ([F; N_TABLES], Vec<F>) {
	let grouped_evals = zip(lookup_evals.into_iter().copied(), table_ids.into_iter().copied())
		.into_group_map_by(|&(_, id)| id);

	assert!(
		grouped_evals.iter().map(|(_, vals)| vals.len()).all_equal(),
		"There must be an equal number of lookups into each table"
	);
	// We assume each table has an equal number of lookups. This mainly serves to simplify the
	// structure of the various sumchecks in the protocol. A possible future todo would be to remove
	// this assumption.

	let batch_log_len = log2_ceil_usize(grouped_evals[&0].len().next_power_of_two());
	let batching_prefix = channel.sample_many(batch_log_len);
	let batch_weights = eq_ind_partial_eval::<F>(&batching_prefix);

	let mut batched_evals = [F::ZERO; N_TABLES];

	// Iterate over the lookup evals per table and batch them using the batch_weights,
	for (table_id, vals) in grouped_evals {
		let (evals, _): (Vec<_>, Vec<_>) = vals.into_iter().unzip();
		batched_evals[table_id] = zip(evals, batch_weights.iter_scalars())
			.map(|(eval, weight)| eval * weight)
			.sum();
	}

	let extended_eval_point = [batching_prefix, eval_point.to_vec()].concat();

	(batched_evals, extended_eval_point)
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
