use std::iter::zip;

use binius_field::{Field, PackedBinaryField64x1b, PackedField, arch::OptimalPackedB128};
use binius_math::FieldBuffer;

// Generates a pushforward given a table and slice of indices.
fn generate_pushforward<P, F>(
	indices: &[u32],
	eq_kernel: FieldBuffer<P>,
	table: &FieldBuffer<P>,
) -> FieldBuffer<P>
where
	P: PackedField<Scalar = F>,
	F: Field,
{
	let mut pushforward = vec![F::ZERO; table.len()];
	for (&idx, eq_i) in zip(indices.iter(), eq_kernel.iter_scalars()) {
		pushforward[idx as usize] += eq_i
	}

	FieldBuffer::from_values(&pushforward)
}

fn generate_lookup_values<P, F>(indices: &[u32], table: &FieldBuffer<P>) -> FieldBuffer<P>
where
	P: PackedField<Scalar = F>,
	F: Field,
{
	let mut lookup_values = vec![F::ZERO; indices.len()];
	for (&idx, lookup) in zip(indices.iter(), lookup_values.iter_mut()) {
		*lookup = table.get(idx as usize);
	}

	FieldBuffer::from_values(&lookup_values)
}
