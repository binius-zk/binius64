// Copyright 2026 The Binius Developers

//! Witness construction for the logUp* prover.
//!
//! These helpers build the multilinears that the two fractional-addition circuits run over:
//!
//! - the looker numerator `eq_r`, the equality indicator at the evaluation point,
//! - the looker denominator `c - I`, with `I` the embedded index column,
//! - the negated table denominator `J - c`, with `J` the embedded table positions,
//! - the pushforward `Y = I_* eq_r`, the looker numerator scattered onto table positions.

use std::iter;

use binius_compute::{Allocator, VecLike};
use binius_field::{BinaryField, Divisible, Field, PackedField};
use binius_math::{
	FieldBuffer, FieldSlice, FieldVec, multilinear::eq::scaled_eq_ind_partial_eval_into,
};
use binius_utils::rayon::{current_num_threads, prelude::*};

use super::prove::TableLookup;

/// The witnesses [`combined_lookers`] builds: the numerators grouped per table, and one
/// pushforward per table.
type LogupWitnesses<P, A> = (Vec<Vec<FieldVec<P, A>>>, Vec<FieldVec<P, A>>);

/// Build each table's gamma-scaled looker numerators and its combined pushforward `Y`.
///
/// Within a table, looker `i`'s numerator is `gamma^i * eq_{r_i}`, the scaled equality indicator
/// its fractional-addition circuit runs over, so the fractional sum of that table's looker circuits
/// is the gamma-combination of their sums. The table's pushforward is the scatter of those same
/// numerators:
///
/// ```text
///     Y = sum_i gamma^i * (I_i)_* eq_{r_i}
/// ```
///
/// Each table uses its own `gamma`, so the tables share nothing here.
///
/// Both the numerators and the pushforwards are drawn from `alloc`: the numerators become the leaf
/// layers of the per-looker fractional-addition circuits, and a committing caller hands the
/// pushforwards to the channel, which owns them until the openings run.
///
/// # Preconditions
///
/// * `tables` is non-empty, every table has at least one looker, every looker's index column has
///   `2^n` entries for its own evaluation point length `n`, and every index entry is less than its
///   table's size.
#[tracing::instrument(
	skip_all,
	level = "debug",
	name = "Build logup* witnesses",
	fields(n_tables = tables.len())
)]
pub fn combined_lookers<A, F, P>(
	alloc: &A,
	gamma: F,
	tables: &[TableLookup<'_, P>],
) -> LogupWitnesses<P, A>
where
	A: Allocator,
	F: Field,
	P: PackedField<Scalar = F>,
{
	assert!(!tables.is_empty(), "at least one table is required");
	assert!(
		tables.iter().all(|table| !table.lookers.is_empty()),
		"every table must have at least one looker"
	);
	// Every index must address a real position in its table, for the embedding and pushforward to
	// be valid. This is a precondition: the O(n) scan is compiled out of release builds. It is
	// checked up front because an out-of-range index would otherwise surface as an opaque
	// out-of-bounds panic inside the scatter-add.
	debug_assert!(
		tables.iter().all(|table| {
			let table_size = 1usize << table.table.log_len();
			table
				.lookers
				.iter()
				.all(|looker| looker.index.iter().all(|&j| j < table_size))
		}),
		"every index entry must be less than the size of the table its looker reads"
	);

	// Build one numerator per looker, fanned out across all of them at once.
	// Why fan out: the per-looker expansion is itself parallel.
	//   But it under-saturates the machine at moderate n.
	//   Spreading the lookers over the cores fills them.
	// The 2^n backing buffers are drawn from `alloc` up front on this thread, so the parallel
	// region only fills them — no allocator traffic inside the rayon closures. Lookers may differ
	// in length, so each buffer is sized to its own looker.
	//
	// Within a table, looker `i` is scaled by gamma^i; the same series serves every table, since
	// the per-table denominator challenges separate them. The powers chain is sequential, so the
	// scales are materialized once here, ahead of the parallel region.
	// Invariant: the fill writes results back in the flattened order (the zip is index-aligned).
	let max_table_lookers = tables
		.iter()
		.map(|table| table.lookers.len())
		.max()
		.expect("tables is non-empty");
	let scales = gamma.powers().take(max_table_lookers).collect::<Vec<_>>();
	let flat = tables
		.iter()
		.flat_map(|table| iter::zip(&table.lookers, &scales))
		.collect::<Vec<_>>();
	let buffers = flat
		.iter()
		.map(|(looker, _)| {
			let packed_len = 1 << looker.eval_point.len().saturating_sub(P::LOG_WIDTH);
			alloc.alloc::<P>(packed_len)
		})
		.collect::<Vec<_>>();
	let flat_numerators = (buffers, flat.as_slice())
		.into_par_iter()
		.map(|(buffer, &(looker, &scale))| {
			let n = looker.eval_point.len();
			assert_eq!(
				looker.index.len(),
				1 << n,
				"index column has {} entries but {} were expected for {n} variables",
				looker.index.len(),
				1usize << n,
			);
			// Seeding the expansion with the scale folds it into the tensor product.
			// That keeps it to one pass over one 2^n buffer.
			scaled_eq_ind_partial_eval_into(looker.eval_point, scale, buffer)
		})
		.collect::<Vec<_>>();

	// Scatter each table's numerators onto its own cube, summed into one buffer. The scatter reads
	// the numerators from rayon tasks, so it borrows them as slices: `Allocator::Vec` is declared
	// only `Send`, so a numerator cannot be shared across tasks by reference.
	//
	// The tables are walked one at a time rather than in parallel: each scatter is already parallel
	// over the looker rows that dominate its cost, and drawing a buffer from `alloc` inside a rayon
	// task is not available here.
	let mut remaining = flat_numerators.as_slice();
	let mut grouped_slices = Vec::with_capacity(tables.len());
	for table in tables {
		let (mine, rest) = remaining.split_at(table.lookers.len());
		remaining = rest;
		grouped_slices.push(mine.iter().map(FieldBuffer::to_ref).collect::<Vec<_>>());
	}
	let pushforwards = iter::zip(tables, &grouped_slices)
		.map(|(table, numerators)| {
			let indexes = table
				.lookers
				.iter()
				.map(|looker| looker.index)
				.collect::<Vec<_>>();
			combined_pushforward::<A, F, P>(alloc, numerators, &indexes, table.table.log_len())
		})
		.collect::<Vec<_>>();

	// Regroup the numerators themselves to match, now that the borrows above are done with.
	let mut flat_iter = flat_numerators.into_iter();
	let numerators = tables
		.iter()
		.map(|table| {
			flat_iter
				.by_ref()
				.take(table.lookers.len())
				.collect::<Vec<_>>()
		})
		.collect::<Vec<_>>();

	(numerators, pushforwards)
}

/// Scatter one table's lookers' numerators onto its `m`-variable cube and sum.
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
/// * `numerators` and `indexes` have equal length.
/// * Each numerator has one entry per row of its looker's index column.
/// * Every index entry is less than `2^table_n_vars`.
fn combined_pushforward<A, F, P>(
	alloc: &A,
	numerators: &[FieldSlice<'_, P>],
	indexes: &[&[usize]],
	table_n_vars: usize,
) -> FieldVec<P, A>
where
	A: Allocator,
	F: Field,
	P: PackedField<Scalar = F>,
{
	// One accumulator slot per table position.
	let table_size = 1usize << table_n_vars;
	let n_lookers = numerators.len();

	// One accumulator per worker, allocated up front on this thread rather than inside the
	// parallel region. Each worker scatters a contiguous chunk of lookers into its own
	// accumulator; the chunk count is capped at the number of workers (and at the looker count),
	// so at most one accumulator is allocated per busy core.
	// A table no looker reads takes no chunk at all; it still needs the one zero accumulator, which
	// is its honest all-zero pushforward.
	let n_workers = current_num_threads().clamp(1, n_lookers.max(1));
	let chunk_size = n_lookers.div_ceil(n_workers).max(1);
	let mut accumulators = iter::repeat_with(|| vec![F::ZERO; table_size])
		.take(n_lookers.div_ceil(chunk_size).max(1))
		.collect::<Vec<_>>();

	(
		accumulators.par_iter_mut(),
		numerators.par_chunks(chunk_size),
		indexes.par_chunks(chunk_size),
	)
		.into_par_iter()
		.for_each(|(acc, numerator_chunk, index_chunk)| {
			for (numerator, index) in iter::zip(numerator_chunk, index_chunk) {
				scatter_add(acc, numerator, index);
			}
		});

	// Merge the per-worker accumulators position by position into the first. The merge is a single
	// pass of `n_workers` sums per slot, negligible against the scatter over every looker row.
	let mut buckets = accumulators.pop().expect("at least one accumulator");
	for partial in &accumulators {
		for (slot, add) in iter::zip(buckets.iter_mut(), partial) {
			*slot += *add;
		}
	}

	// Repack the merged scalar accumulator into the packed table buffer.
	FieldBuffer::from_values_in(alloc, &buckets)
}

/// Scatter-add one looker's numerator onto the table accumulator in row order.
///
/// ```text
///     acc[index[i]] += numerator[i]
/// ```
///
/// The numerator is read sequentially, so each row is a lane read, not an indexed lookup.
#[inline]
fn scatter_add<F, P>(acc: &mut [F], numerator: &FieldSlice<'_, P>, index: &[usize])
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
pub fn looker_denominator<A, F, P>(alloc: &A, c: F, index: &[usize]) -> FieldVec<P, A>
where
	A: Allocator,
	F: BinaryField<Underlier: Divisible<u64>>,
	P: PackedField<Scalar = F>,
{
	// n, the number of looker variables, from the 2^n rows.
	let log_len = index.len().ilog2() as usize;

	// One denominator per row: c minus the row's embedded index value.
	// Subtract a full word at a time: one packed subtraction per word, built in parallel straight
	// into the allocator's buffer.
	let packed_len = 1 << log_len.saturating_sub(P::LOG_WIDTH);
	let c_packed = P::broadcast(c);
	let mut packed = alloc.alloc::<P>(packed_len);
	packed
		.spare_capacity_mut()
		.par_iter_mut()
		.zip(index.par_chunks(P::WIDTH))
		.for_each(|(slot, chunk)| {
			slot.write(c_packed - P::from_scalars(chunk.iter().copied().map(embed_position::<F>)));
		});
	// Safety: every packed slot is written exactly once by the parallel loop above.
	unsafe { packed.set_len(packed_len) };

	FieldBuffer::new(log_len, packed)
}

/// Build the negated table denominator `J - c` over the `m`-variable table cube.
///
/// Entry `j` is `iota(j) - c`. The logUp denominator for table position `j` is `c - iota(j)`; the
/// table's fraction enters the sum of every instance negated, and carrying that negation on the
/// denominator rather than the numerator costs nothing here, where the entries are built anyway.
pub fn table_denominator<A, F, P>(alloc: &A, c: F, table_n_vars: usize) -> FieldVec<P, A>
where
	A: Allocator,
	F: BinaryField<Underlier: Divisible<u64>>,
	P: PackedField<Scalar = F>,
{
	// One denominator per table position: shift the position's embedding by the challenge.
	let values = (0..1usize << table_n_vars)
		.map(|j| embed_position::<F>(j) - c)
		.collect::<Vec<_>>();
	FieldBuffer::from_values_in(alloc, &values)
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
	scatter_add(&mut buckets, &eq_r.to_ref(), index);
	// Repack the scalar accumulator into the packed table buffer.
	FieldBuffer::from_values(&buckets)
}

#[cfg(test)]
mod tests {
	use binius_compute::GlobalAllocator;
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

	use super::{combined_pushforward, embed_position, looker_denominator, pushforward};

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

		// The scatter reads only the index columns.
		let index_slices = indices.iter().map(Vec::as_slice).collect::<Vec<_>>();
		let numerator_slices = numerators
			.iter()
			.map(FieldBuffer::to_ref)
			.collect::<Vec<_>>();
		let got =
			combined_pushforward::<_, F, P>(&GlobalAllocator, &numerator_slices, &index_slices, m)
				.iter_scalars()
				.collect::<Vec<_>>();
		assert_eq!(got, combined_reference(&numerators, &indices, m));
	}

	#[test]
	fn combined_pushforward_of_no_lookers_is_zero() {
		// A table no looker reads still needs a pushforward buffer; its honest value is all zeros.
		let got = combined_pushforward::<_, F, P>(&GlobalAllocator, &[], &[], 3)
			.iter_scalars()
			.collect::<Vec<_>>();
		assert_eq!(got, vec![F::ZERO; 8]);
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
		let one_row = looker_denominator::<_, F, P>(&GlobalAllocator, c, &[3])
			.iter_scalars()
			.collect::<Vec<_>>();
		assert_eq!(one_row, denominator_reference(c, &[3]));

		// n = 2: four rows with distinct embedded positions.
		let index = [0usize, 1, 2, 5];
		let four_rows = looker_denominator::<_, F, P>(&GlobalAllocator, c, &index)
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

			let got = looker_denominator::<_, F, P>(&GlobalAllocator, c, &index)
				.iter_scalars()
				.collect::<Vec<_>>();
			prop_assert_eq!(got, denominator_reference(c, &index));
		}
	}
}
