// Copyright 2026 The Binius Developers

//! Witness construction for the logUp* prover.
//!
//! These helpers build the multilinears that the two fractional-addition circuits run over:
//!
//! - the looker numerator `eq_r`, the equality indicator at the evaluation point,
//! - the looker denominator `c - I`, with `I` the embedded index column,
//! - the table denominator `c - J`, with `J` the embedded table positions,
//! - the pushforward `Y = I_* eq_r`, the looker numerator scattered onto table positions.

use binius_field::{BinaryField1b, ExtensionField, Field, PackedField};
use binius_math::{FieldBuffer, multilinear::eq::eq_ind_partial_eval};
use binius_utils::rayon::prelude::*;

/// Embed a table position `j` into the field through the `GF(2)`-linear basis.
///
/// ```text
///     iota(j) = sum_{t : bit t of j is set} basis(t)
/// ```
///
/// This is the same embedding the verifier uses for the table-side denominator `J`.
/// It makes a position and an index value that point to it embed to the same field element.
pub fn embed_position<F>(j: usize) -> F
where
	F: Field + ExtensionField<BinaryField1b>,
{
	// usize::BITS bounds the loop; positions with a set bit at t contribute basis(t).
	(0..usize::BITS as usize)
		.filter(|&t| (j >> t) & 1 == 1)
		.map(<F as ExtensionField<BinaryField1b>>::basis)
		.fold(F::ZERO, |acc, b| acc + b)
}

/// Build the looker numerator `eq_r`, the equality indicator at the evaluation point.
///
/// `eq_r[i] = eq(eval_point, i)`, the multilinear `X = eq_r` of [Soukhanov25, Section 4].
///
/// [Soukhanov25]: <https://eprint.iacr.org/2025/946>
pub fn equality_indicator<F, P>(eval_point: &[F]) -> FieldBuffer<P>
where
	F: Field,
	P: PackedField<Scalar = F>,
{
	// The equality indicator's hypercube values are the Lagrange weights at the point.
	eq_ind_partial_eval(eval_point)
}

/// Build the looker denominator `c - I` over the `n`-variable looker cube.
///
/// Entry `i` is `c - iota(index[i])`, the logUp denominator for looker row `i`.
pub fn looker_denominator<F, P>(c: F, index: &[usize]) -> FieldBuffer<P>
where
	F: Field + ExtensionField<BinaryField1b>,
	P: PackedField<Scalar = F>,
{
	// One denominator per looker row: shift the challenge by the row's embedded index value.
	let values = index
		.iter()
		.map(|&i| c - embed_position::<F>(i))
		.collect::<Vec<_>>();
	FieldBuffer::from_values(&values)
}

/// Build the table denominator `c - J` over the `m`-variable table cube.
///
/// Entry `j` is `c - iota(j)`, the logUp denominator for table position `j`.
pub fn table_denominator<F, P>(c: F, table_n_vars: usize) -> FieldBuffer<P>
where
	F: Field + ExtensionField<BinaryField1b>,
	P: PackedField<Scalar = F>,
{
	// One denominator per table position: shift the challenge by the position's embedding.
	let values = (0..1usize << table_n_vars)
		.map(|j| c - embed_position::<F>(j))
		.collect::<Vec<_>>();
	FieldBuffer::from_values(&values)
}

/// Build the pushforward `Y = I_* eq_r` over the `m`-variable table cube.
///
/// ```text
///     Y[j] = sum_{i : index[i] = j} eq_r[i]
/// ```
///
/// `Y` is the dual of the pullback under the inner product, so `<T, Y> = (I^* T)(eval_point)`.
/// It has only `2^m` entries, which is the cost saving over committing the `2^n`-entry pullback.
///
/// # Preconditions
///
/// * every `index[i]` is less than `2^table_n_vars`.
pub fn pushforward<F, P>(
	eq_r: &FieldBuffer<P>,
	index: &[usize],
	table_n_vars: usize,
) -> FieldBuffer<P>
where
	F: Field,
	P: PackedField<Scalar = F>,
{
	// Row count at or above which parallel scatter beats the single-threaded scan.
	//
	// Obtained by experimentation, can be tuned in the future.
	const PARALLEL_THRESHOLD: usize = 1 << 18;

	let table_size = 1usize << table_n_vars;

	let values = if index.len() < PARALLEL_THRESHOLD {
		// One thread scatters every row into a single bucket array.
		let mut buckets = vec![F::ZERO; table_size];
		for (eq_i, &j) in eq_r.iter_scalars().zip(index) {
			buckets[j] += eq_i;
		}
		buckets
	} else {
		// Each job folds a contiguous run of rows into its own bucket array, reading eq_r in order.
		//
		// The per-job arrays are then summed position by position.
		index
			.par_iter()
			.enumerate()
			.fold(
				|| vec![F::ZERO; table_size],
				|mut buckets, (i, &j)| {
					buckets[j] += eq_r.get(i);
					buckets
				},
			)
			.reduce(
				|| vec![F::ZERO; table_size],
				|mut acc, partial| {
					for (slot, add) in acc.iter_mut().zip(partial) {
						*slot += add;
					}
					acc
				},
			)
	};
	FieldBuffer::from_values(&values)
}

#[cfg(test)]
mod tests {
	use binius_field::{
		Field,
		arch::{OptimalB128, OptimalPackedB128},
	};
	use binius_math::{FieldBuffer, test_utils::random_field_buffer};
	use proptest::prelude::*;
	use rand::prelude::*;

	use super::pushforward;

	type F = OptimalB128;
	type P = OptimalPackedB128;

	// An independent single-threaded scatter, the reference the dispatched result must match.
	fn reference(eq_r: &FieldBuffer<P>, index: &[usize], m: usize) -> Vec<F> {
		let mut values = vec![F::ZERO; 1usize << m];
		for (i, &j) in index.iter().enumerate() {
			values[j] += eq_r.get(i);
		}
		values
	}

	// Assert pushforward equals the reference on a random instance of shape (n, m).
	fn check(n: usize, m: usize, seed: u64) {
		let mut rng = StdRng::seed_from_u64(seed);
		let eq_r = random_field_buffer::<P>(&mut rng, n);
		let index = (0..(1usize << n))
			.map(|_| rng.random_range(0..(1usize << m)))
			.collect::<Vec<_>>();

		let got = pushforward::<F, P>(&eq_r, &index, m)
			.iter_scalars()
			.collect::<Vec<_>>();
		assert_eq!(got, reference(&eq_r, &index, m));
	}

	proptest! {
		#![proptest_config(ProptestConfig::with_cases(8))]

		// 2^18 rows crosses the threshold, so this fuzzes the parallel scatter.
		// Small m forces heavy collisions into few buckets.
		#[test]
		fn parallel_scatter_matches_reference(seed in any::<u64>(), m in 1usize..=6) {
			check(18, m, seed);
		}
	}

	#[test]
	fn sequential_scatter_matches_reference() {
		// Below the threshold the single-threaded path runs.
		// n = 0 is the one-row edge; (10, 1) packs 2^10 rows into 2 buckets.
		check(0, 3, 7);
		check(10, 1, 42);
	}
}
