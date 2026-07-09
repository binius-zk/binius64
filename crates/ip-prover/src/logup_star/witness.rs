// Copyright 2026 The Binius Developers

//! Witness construction for the logUp* prover.
//!
//! These helpers build the multilinears that the two fractional-addition circuits run over:
//!
//! - the looker numerator `eq_r`, the equality indicator at the evaluation point,
//! - the looker denominator `c - I`, with `I` the embedded index column,
//! - the table denominator `c - J`, with `J` the embedded table positions,
//! - the pushforward `Y = I_* eq_r`, the looker numerator scattered onto table positions.

use std::iter;

use binius_field::{BinaryField, Divisible, Field, PackedField, util::powers};
use binius_math::{FieldBuffer, multilinear::eq::scaled_eq_ind_partial_eval};
use binius_utils::rayon::prelude::*;
use itertools::izip;

use super::prove::Looker;

/// Build every looker's gamma-scaled numerator and the combined pushforward `Y`.
///
/// Looker `j`'s numerator is `gamma^j * eq_{r_j}`, the scaled equality indicator its
/// fractional-addition circuit runs over, so the fractional sum of the per-looker circuits is the
/// gamma-combination of the looker sums. The combined pushforward is the scatter of the same
/// numerators:
///
/// ```text
///     Y = sum_j gamma^j * (I_j)_* eq_{r_j}
/// ```
///
/// # Preconditions
///
/// * `lookers` is non-empty, every looker has the same evaluation point length `n`, every index
///   column has `2^n` entries, and every index entry is less than `2^table_n_vars`.
#[tracing::instrument(
	skip_all,
	level = "debug",
	name = "Build logup* witnesses",
	fields(n_lookers = lookers.len(), table_n_vars)
)]
pub fn combined_lookers<F, P>(
	lookers: &[Looker<'_, F>],
	gamma: F,
	table_n_vars: usize,
) -> (Vec<FieldBuffer<P>>, FieldBuffer<P>)
where
	F: Field,
	P: PackedField<Scalar = F>,
{
	assert!(!lookers.is_empty(), "at least one looker is required");
	let n = lookers[0].eval_point.len();

	let numerators = izip!(lookers, powers(gamma))
		.map(|(looker, power)| {
			assert_eq!(
				looker.eval_point.len(),
				n,
				"every looker evaluation point must have the same length"
			);
			assert_eq!(
				looker.index.len(),
				1 << n,
				"index column has {} entries but {} were expected for {n} variables",
				looker.index.len(),
				1usize << n,
			);
			// Looker j's numerator is gamma^j * eq_{r_j}.
			// Seeding the expansion with gamma^j folds the scale into the tensor product.
			// That keeps it to one pass over one 2^n buffer.
			scaled_eq_ind_partial_eval::<P>(looker.eval_point, power)
		})
		.collect::<Vec<_>>();

	// Scatter every looker's numerator onto the shared table cube, summed into one buffer.
	let combined = combined_pushforward::<F, P>(&numerators, lookers, table_n_vars);

	(numerators, combined)
}

/// Scatter every looker's numerator onto the shared `m`-variable table cube and sum.
///
/// ```text
///     Y[v] = sum_j sum_{i : index_j[i] = v} numerator_j[i]
/// ```
///
/// The per-looker `gamma^j` scale already lives in each numerator.
/// So the plain sum of the scatters is the gamma-combined pushforward.
/// The sum is over a field, so the accumulation order does not matter.
///
/// # Performance
///
/// The scatter over every looker row is the dominant `n`-axis cost.
/// Two choices keep it lean:
///
/// - Each looker is read sequentially in row order, so no row pays an indexed lane-extract.
/// - The work parallelizes across lookers, not rows.
/// - So each task fills a single `2^m` accumulator for its run of lookers.
/// - The per-task accumulators merge in a single pass.
///
/// With few lookers this leaves cores idle on the `n`-axis.
/// The target regime has one looker per column, so the tasks stay busy.
///
/// # Preconditions
///
/// * `numerators` and `lookers` have equal length.
/// * Each numerator has one entry per row of its looker's index column.
/// * Every index entry is less than `2^table_n_vars`.
fn combined_pushforward<F, P>(
	numerators: &[FieldBuffer<P>],
	lookers: &[Looker<'_, F>],
	table_n_vars: usize,
) -> FieldBuffer<P>
where
	F: Field,
	P: PackedField<Scalar = F>,
{
	// One accumulator slot per table position.
	let table_size = 1usize << table_n_vars;

	let buckets = (0..numerators.len())
		.into_par_iter()
		// Each task scatters a contiguous run of lookers into its own accumulator.
		.fold(
			|| vec![F::ZERO; table_size],
			|mut acc, j| {
				scatter_add(&mut acc, &numerators[j], lookers[j].index);
				acc
			},
		)
		// Merge the per-task accumulators position by position into one.
		.reduce(
			|| vec![F::ZERO; table_size],
			|mut acc, partial| {
				for (slot, add) in iter::zip(acc.iter_mut(), partial) {
					*slot += add;
				}
				acc
			},
		);

	// Repack the merged scalar accumulator into the packed table buffer.
	FieldBuffer::from_values(&buckets)
}

/// Scatter-add one looker's numerator onto the table accumulator in row order.
///
/// ```text
///     acc[index[i]] += numerator[i]
/// ```
///
/// The numerator is read sequentially, so each row is a lane read, not an indexed lookup.
#[inline]
fn scatter_add<F, P>(acc: &mut [F], numerator: &FieldBuffer<P>, index: &[usize])
where
	F: Field,
	P: PackedField<Scalar = F>,
{
	// Row i's numerator value lands in the table position that row indexes into.
	for (value, &target) in numerator.iter_scalars().zip(index) {
		acc[target] += value;
	}
}

/// Embed a table position `j` into the field through the `GF(2)`-linear basis.
///
/// ```text
///     iota(j) = sum_{t : bit t of j is set} basis(t)
/// ```
///
/// This is the same embedding the verifier uses for the table-side denominator `J`.
/// It makes a position and an index value that point to it embed to the same field element.
///
/// The `GF(2)`-linear basis of a binary tower field is its underlier's bit basis: basis element
/// `t` is the field element whose underlier has only bit `t` set. So `iota(j)` is just the field
/// element whose underlier is `j`, which we build directly instead of summing basis elements.
#[inline]
pub fn embed_position<F>(j: usize) -> F
where
	F: BinaryField<Underlier: Divisible<u64>>,
{
	F::from_underlier(F::Underlier::from_iter(iter::once(j as u64)))
}

/// Build the looker denominator `c - I` over the `n`-variable looker cube.
///
/// Entry `i` is `c - iota(index[i])`, the logUp denominator for looker row `i`.
///
/// # Preconditions
///
/// * `index.len()` is a power of two.
pub fn looker_denominator<F, P>(c: F, index: &[usize]) -> FieldBuffer<P>
where
	F: BinaryField<Underlier: Divisible<u64>>,
	P: PackedField<Scalar = F>,
{
	// n, the number of looker variables, from the 2^n rows.
	let log_len = index.len().ilog2() as usize;

	// One denominator per row: c minus the row's embedded index value.
	// Subtract a full word at a time: one packed subtraction per word, built in parallel.
	let c_packed = P::broadcast(c);
	let packed = index
		.par_chunks(P::WIDTH)
		.map(|chunk| c_packed - P::from_scalars(chunk.iter().copied().map(embed_position::<F>)))
		.collect::<Vec<_>>();

	FieldBuffer::new(log_len, packed.into_boxed_slice())
}

/// Build the table denominator `c - J` over the `m`-variable table cube.
///
/// Entry `j` is `c - iota(j)`, the logUp denominator for table position `j`.
pub fn table_denominator<F, P>(c: F, table_n_vars: usize) -> FieldBuffer<P>
where
	F: BinaryField<Underlier: Divisible<u64>>,
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
/// This is the single-looker scatter.
/// The prover combines many lookers by summing their scatters onto the same cube.
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
	// One accumulator slot per table position, all starting empty.
	let mut buckets = vec![F::ZERO; 1usize << table_n_vars];
	// Add each row's numerator value into the position it indexes into.
	scatter_add(&mut buckets, eq_r, index);
	// Repack the scalar accumulator into the packed table buffer.
	FieldBuffer::from_values(&buckets)
}

#[cfg(test)]
mod tests {
	use binius_field::{
		Field,
		arch::{OptimalB128, OptimalPackedB128},
	};
	use binius_math::{
		FieldBuffer,
		test_utils::{random_field_buffer, random_scalars},
	};
	use proptest::prelude::*;
	use rand::prelude::*;

	use super::{Looker, combined_pushforward, embed_position, looker_denominator, pushforward};

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

	#[test]
	fn pushforward_matches_reference() {
		// n = 0: the single-row edge.
		check(0, 3, 7);
		// 2^10 rows collapsed into 2 buckets: every position takes heavy collisions.
		check(10, 1, 42);
		// A wider 16-bucket cube with sparser collisions.
		check(12, 4, 1);
	}

	// Reference scatter: the gamma-combined pushforward, single-threaded, one pass per looker.
	// The fused parallel build must reproduce this exactly.
	fn combined_reference(
		numerators: &[FieldBuffer<P>],
		indices: &[Vec<usize>],
		m: usize,
	) -> Vec<F> {
		let mut acc = vec![F::ZERO; 1usize << m];
		for (numerator, index) in numerators.iter().zip(indices) {
			for (value, &target) in numerator.iter_scalars().zip(index) {
				acc[target] += value;
			}
		}
		acc
	}

	// Assert the fused scatter equals the reference on a random multi-looker instance.
	fn check_combined(n: usize, m: usize, n_lookers: usize, seed: u64) {
		let mut rng = StdRng::seed_from_u64(seed);

		// Each looker gets its own numerator buffer and its own index column.
		let numerators = (0..n_lookers)
			.map(|_| random_field_buffer::<P>(&mut rng, n))
			.collect::<Vec<_>>();
		let indices = (0..n_lookers)
			.map(|_| {
				(0..(1usize << n))
					.map(|_| rng.random_range(0..(1usize << m)))
					.collect::<Vec<_>>()
			})
			.collect::<Vec<_>>();

		// The scatter reads only the index column.
		// The evaluation point and claim are unused here, so leave them empty.
		let eval_points = vec![Vec::<F>::new(); n_lookers];
		let lookers = indices
			.iter()
			.zip(&eval_points)
			.map(|(index, eval_point)| Looker {
				index,
				eval_point,
				eval_claim: F::ZERO,
			})
			.collect::<Vec<_>>();

		let got = combined_pushforward::<F, P>(&numerators, &lookers, m)
			.iter_scalars()
			.collect::<Vec<_>>();
		assert_eq!(got, combined_reference(&numerators, &indices, m));
	}

	#[test]
	fn combined_pushforward_small_cases() {
		// One looker: the combined scatter degenerates to a single pushforward.
		check_combined(4, 3, 1, 5);
		// n = 0: each looker contributes a single row.
		check_combined(0, 3, 3, 6);
	}

	proptest! {
		#![proptest_config(ProptestConfig::with_cases(8))]

		// Fuzz the fused scatter across shapes.
		// Small m forces heavy collisions into few buckets.
		// Several lookers exercise the parallel fold and the merging reduce.
		#[test]
		fn combined_pushforward_matches_reference(
			seed in any::<u64>(),
			n in 0usize..=10,
			m in 1usize..=6,
			n_lookers in 1usize..=5,
		) {
			check_combined(n, m, n_lookers, seed);
		}
	}

	// The scalar reference for the looker denominator: c - iota(index[i]) per row.
	fn denominator_reference(c: F, index: &[usize]) -> Vec<F> {
		index.iter().map(|&i| c - embed_position::<F>(i)).collect()
	}

	#[test]
	fn looker_denominator_small_cases() {
		let c = F::new(7);

		// n = 0: a single row, so the packed word carries one meaningful lane.
		let one_row = looker_denominator::<F, P>(c, &[3])
			.iter_scalars()
			.collect::<Vec<_>>();
		assert_eq!(one_row, denominator_reference(c, &[3]));

		// n = 2: four rows with distinct embedded positions.
		let index = [0usize, 1, 2, 5];
		let four_rows = looker_denominator::<F, P>(c, &index)
			.iter_scalars()
			.collect::<Vec<_>>();
		assert_eq!(four_rows, denominator_reference(c, &index));
	}

	proptest! {
		#![proptest_config(ProptestConfig::with_cases(16))]

		// The direct packed build must equal the scalar reference, value by value.
		// n spans below, at, and above the packing width; index values exercise multi-bit embeddings.
		#[test]
		fn looker_denominator_matches_reference(seed in any::<u64>(), n in 0usize..=8) {
			let mut rng = StdRng::seed_from_u64(seed);
			let c = random_scalars::<F>(&mut rng, 1)[0];
			let index = (0..(1usize << n))
				.map(|_| rng.random_range(0..(1usize << 12)))
				.collect::<Vec<_>>();

			let got = looker_denominator::<F, P>(c, &index).iter_scalars().collect::<Vec<_>>();
			prop_assert_eq!(got, denominator_reference(c, &index));
		}
	}
}
