// Copyright 2026 The Binius Developers

//! One operand column with its bit and constraint axes collapsed, leaving the instance axis.

use binius_compute::{Allocator, VecLike};
use binius_core::word::Word;
use binius_field::{Field, PackedField};
use binius_math::{FieldBuffer, FieldVec};
use binius_prover::fold_word::fold_words;
use binius_utils::{checked_arithmetics::checked_log_2, rayon::prelude::*};
use binius_verifier::config::B128;

/// Builds the instance-axis multilinear of one operand column, folded over its bit and constraint
/// axes.
///
/// The operand column is constraint-major: `row = local_constraint * n_instances + instance`.
/// Folding collapses the two other axes and leaves one field element per instance:
///
/// ```text
/// M[rho] = sum_{local, j} lagrange[j] * r_x_tensor[local] * bit_j(column[local * K + rho])
/// ```
///
/// - The Lagrange weights fold each 64-bit word over its bit axis at the shared univariate
///   challenge.
/// - The constraint tensor `r_x_tensor` folds the constraint axis; the caller expands it once per
///   operation and shares it across the operands.
///
/// Its evaluation at the operation's instance point equals that operation's oblong operand claim.
/// So the re-randomization sumcheck can transport that claim to a shared instance point.
pub fn operand_rho_multilinear<A, P>(
	alloc: &A,
	column: &[Word],
	lagrange: &[B128],
	r_x_tensor: &[B128],
) -> FieldVec<P, A>
where
	A: Allocator,
	P: PackedField<Scalar = B128>,
{
	// Fold each word's bits at the univariate challenge: one scalar per row, laid out
	// constraint-major.
	// Folding into scalars keeps the row indexing flat for the constraint fold.
	let folded_rows = fold_words::<B128, B128, _>(alloc, column, lagrange);
	let folded_rows = folded_rows.as_ref();

	// Produce the packed instance-axis multilinear directly into the allocator's buffer, one packed
	// element per parallel task.
	// Each element's lanes are the constraint folds of consecutive instances.
	// Lanes past the instance count are the multilinear's zero padding.
	//
	// The constraint axis is the high, strided axis: constraint `local` of instance `rho` sits at
	// row `local * n_instances + rho`.
	let n_constraints = r_x_tensor.len();
	let n_instances = folded_rows.len() / n_constraints;
	let log_instances = checked_log_2(n_instances);
	let log_packed = log_instances.saturating_sub(P::LOG_WIDTH);
	let packed_len = 1usize << log_packed;
	let mut packed = alloc.alloc::<P>(packed_len);
	packed
		.spare_capacity_mut()
		.par_iter_mut()
		.enumerate()
		.for_each(|(packed_index, slot)| {
			slot.write(P::from_scalars((0..P::WIDTH).map(|lane| {
				let instance = (packed_index << P::LOG_WIDTH) | lane;
				if instance < n_instances {
					r_x_tensor
						.iter()
						.enumerate()
						.map(|(local, &weight)| {
							weight * folded_rows[local * n_instances + instance]
						})
						.sum()
				} else {
					B128::ZERO
				}
			})));
		});
	// Safety: every packed slot is written exactly once by the parallel loop above.
	unsafe { packed.set_len(packed_len) };

	FieldBuffer::new(log_instances, packed)
}
