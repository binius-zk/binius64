// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

//! Folding a matrix of single-bit rows against one weight per row.

use std::{array, iter};

use binius_field::{
	BinaryField, Divisible, PackedField, WithUnderlier, transpose::transpose_square_blocks_array,
	util::expand_subset_sums_array,
};
use binius_verifier::config::B1;

/// Weights one subset-sum table covers.
///
/// Eight is the widest group whose lookup index still fits one byte.
/// One table load then replaces eight conditional additions.
pub const WEIGHTS_PER_TABLE: usize = 1 << LOG_WEIGHTS_PER_TABLE;

/// Base-2 log of the weights one subset-sum table covers.
pub const LOG_WEIGHTS_PER_TABLE: usize = 3;

/// The rows one table covers, one bit per scalar.
pub type RowGroup<PB> = [PB; WEIGHTS_PER_TABLE];

/// Subset-sum tables for folding a matrix of single-bit rows against one weight per row.
///
/// The fold contracts the row axis, leaving one field element per column:
///
/// ```text
///     out[b] = sum_r weight[r] * bit_b(row[r])
/// ```
///
/// Taking eight rows at a time turns that inner sum into a single table lookup.
/// Table `g` covers rows `8g` through `8g + 7` and holds every subset sum of their weights.
#[derive(Debug, Clone)]
pub struct RowFoldTables<F, const N_TABLES: usize> {
	tables: [[F; 1 << WEIGHTS_PER_TABLE]; N_TABLES],
}

impl<F: BinaryField, const N_TABLES: usize> RowFoldTables<F, N_TABLES> {
	/// Builds the tables from one weight per row, from the first row onwards.
	///
	/// Weights past the end of the slice read as zero.
	/// Those weight rows past the end of the matrix, which read as zero as well.
	/// So one table layout serves a chunk the row list does not fill.
	pub fn new(weights: &[F]) -> Self {
		let tables = array::from_fn(|group| {
			// Weights of the eight rows this table covers, zero where the slice has run out.
			// A group beyond the end of the slice starts at its end, so it copies nothing.
			let mut group_weights = [F::ZERO; WEIGHTS_PER_TABLE];
			let start = (group * WEIGHTS_PER_TABLE).min(weights.len());
			let available = (weights.len() - start).min(WEIGHTS_PER_TABLE);
			group_weights[..available].copy_from_slice(&weights[start..start + available]);

			// Enumerate all 256 subset sums, so any byte of set bits indexes its sum in one load.
			expand_subset_sums_array(group_weights)
		});

		Self { tables }
	}

	/// Folds each group of eight rows into the column sums.
	///
	/// Groups pair with tables in order, so the `g`th group is weighted by rows `8g` onwards.
	/// An iterator yielding fewer groups leaves the remaining rows out, which reads them as zero.
	///
	/// # Preconditions
	///
	/// * A row must be one byte wide per table, so the groups cover every column exactly once.
	#[inline]
	pub fn fold_into<PB>(
		&self,
		groups: impl IntoIterator<Item = RowGroup<PB>>,
		sums: &mut ColumnSums<F, N_TABLES>,
	) where
		PB: PackedField<Scalar = B1> + WithUnderlier,
		PB::Underlier: Divisible<u8>,
	{
		// One byte of a row per table is what makes the accumulator line up with the columns.
		const {
			assert!(
				PB::WIDTH == WEIGHTS_PER_TABLE * N_TABLES,
				"the row width must be one byte per table"
			);
		}

		// Pairing here rather than at the call site is what keeps a group with its own weights.
		for (group, table) in iter::zip(groups, &self.tables) {
			fold_group(group, table, &mut sums.groups);
		}
	}
}

/// One field element per column of the matrix, summed across row groups.
///
/// Reading the groups end to end walks the columns in order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ColumnSums<F, const N_TABLES: usize> {
	// Invariant: entry `j` of group `i` is column `8i + j`, which is flat index `8i + j`.
	//
	// So the nesting is the identity permutation, and exists only because an array length of
	// `WEIGHTS_PER_TABLE * N_TABLES` is not expressible in a const generic on stable.
	groups: [[F; WEIGHTS_PER_TABLE]; N_TABLES],
}

impl<F: BinaryField, const N_TABLES: usize> ColumnSums<F, N_TABLES> {
	/// The sums before any group has been folded in.
	pub const fn zero() -> Self {
		Self {
			groups: [[F::ZERO; WEIGHTS_PER_TABLE]; N_TABLES],
		}
	}

	/// The sums, in column order.
	#[inline]
	pub const fn as_slice(&self) -> &[F] {
		self.groups.as_flattened()
	}

	/// Adds every column's sum into `out`, scaled by one weight.
	///
	/// A caller folding a matrix in chunks scales each chunk by its own weight on the way out.
	///
	/// # Preconditions
	///
	/// * `out` must hold one element per column.
	#[inline]
	pub fn add_scaled_to(&self, weight: F, out: &mut [F]) {
		debug_assert_eq!(out.len(), self.as_slice().len()); // precondition

		for (out_i, &sum) in iter::zip(out, self.as_slice()) {
			*out_i += sum * weight;
		}
	}
}

impl<F: BinaryField, const N_TABLES: usize> Default for ColumnSums<F, N_TABLES> {
	fn default() -> Self {
		Self::zero()
	}
}

/// Folds one group of eight rows into the column sums.
///
/// Rows arrive one bit per scalar, so a row is one packed element and a column is a scalar index.
/// Transposing the group exchanges its row axis with the low three bits of the column index:
///
/// ```text
///     before:  element r, bit 8i + j  =  row r, column 8i + j
///     after:   element j, bit 8i + t  =  row t, column 8i + j
/// ```
///
/// So byte `i` of element `j` carries the eight rows' bits at column `8i + j`.
/// One lookup of that byte yields those rows' whole contribution to that column.
#[inline]
fn fold_group<F, PB, const N_TABLES: usize>(
	mut group: RowGroup<PB>,
	table: &[F; 1 << WEIGHTS_PER_TABLE],
	sums: &mut [[F; WEIGHTS_PER_TABLE]; N_TABLES],
) where
	F: BinaryField,
	PB: PackedField<Scalar = B1> + WithUnderlier,
	PB::Underlier: Divisible<u8>,
{
	// The transpose rewrites the group in place, and the caller handed over its copy.
	transpose_square_blocks_array::<PB, LOG_WEIGHTS_PER_TABLE, WEIGHTS_PER_TABLE>(&mut group);

	for (j, row) in group.iter().enumerate() {
		// Byte `i` holds this group's bits at column `8i + j`, so it indexes that column's sum.
		for (i, byte) in Divisible::<u8>::value_iter(row.to_underlier()).enumerate() {
			sums[i][j] += table[byte as usize];
		}
	}
}

#[cfg(test)]
mod tests {
	use binius_field::{Field, PackedBinaryField64x1b, PackedBinaryField128x1b, Random};
	use binius_math::test_utils::random_scalars;
	use binius_verifier::config::B128;
	use rand::prelude::*;

	use super::*;

	// The fold, written straight from its definition: every set bit adds its row's weight to that
	// bit's column.
	//
	//     out[b] = sum_r weight[r] * bit_b(row[r])
	fn naive_fold<F: BinaryField, PB: PackedField<Scalar = B1>>(
		rows: &[PB],
		weights: &[F],
	) -> Vec<F> {
		let mut out = vec![F::ZERO; PB::WIDTH];
		for (row, &weight) in iter::zip(rows, weights) {
			for (column, bit) in row.iter().enumerate() {
				if bit == B1::ONE {
					out[column] += weight;
				}
			}
		}
		out
	}

	// One row per lane of the packed type, grouped eight at a time.
	fn random_groups<PB: PackedField + Random, const N_TABLES: usize>(
		rng: &mut StdRng,
	) -> Vec<RowGroup<PB>> {
		(0..N_TABLES)
			.map(|_| array::from_fn(|_| PB::random(&mut *rng)))
			.collect()
	}

	fn check_matches_naive<PB, const N_TABLES: usize>(seed: u64, n_weights: usize)
	where
		PB: PackedField<Scalar = B1> + WithUnderlier + Random,
		PB::Underlier: Divisible<u8>,
	{
		let mut rng = StdRng::seed_from_u64(seed);

		// One row per weight the fold covers, and one weight per row the tables cover.
		let groups = random_groups::<PB, N_TABLES>(&mut rng);
		let weights = random_scalars::<B128>(&mut rng, n_weights);

		let tables = RowFoldTables::<B128, N_TABLES>::new(&weights);
		let mut sums = ColumnSums::zero();
		tables.fold_into(groups.iter().copied(), &mut sums);

		// The naive side needs the rows laid out flat, in the order the tables weight them.
		let rows = groups.concat();
		let mut padded = weights;
		padded.resize(rows.len(), B128::ZERO);

		assert_eq!(
			sums.as_slice(),
			naive_fold(&rows, &padded),
			"fold differs at seed {seed}, {n_weights} weights"
		);
	}

	#[test]
	fn fold_matches_the_definition() {
		// The two row widths the callers run at: 64-bit words and 128-bit field elements.
		//
		//     64 columns  -> 8 tables of 8 rows
		//     128 columns -> 16 tables of 8 rows
		check_matches_naive::<PackedBinaryField64x1b, 8>(0, 64);
		check_matches_naive::<PackedBinaryField128x1b, 16>(1, 128);
	}

	#[test]
	fn weights_past_the_end_read_as_zero() {
		// A row list that does not fill the tables must fold as the same list zero-padded up to
		// them. This is what lets one table layout serve a partial chunk.
		//
		//     weights: [w_0 .. w_20]           rows 21..63 weigh nothing
		//     padded : [w_0 .. w_20, 0 .. 0]
		check_matches_naive::<PackedBinaryField64x1b, 8>(2, 21);
		check_matches_naive::<PackedBinaryField64x1b, 8>(3, 0);
		check_matches_naive::<PackedBinaryField128x1b, 16>(4, 100);
	}

	#[test]
	fn sums_read_out_in_column_order() {
		let mut rng = StdRng::seed_from_u64(5);

		// Weight row 0 alone, so every column's sum is that weight exactly where row 0 has a set
		// bit, and zero elsewhere. That pins the read-out order against the row's own bits.
		let weights = random_scalars::<B128>(&mut rng, 1);
		let row = PackedBinaryField64x1b::random(&mut rng);
		let mut groups = [[PackedBinaryField64x1b::default(); WEIGHTS_PER_TABLE]; 8];
		groups[0][0] = row;

		let tables = RowFoldTables::<B128, 8>::new(&weights);
		let mut sums = ColumnSums::zero();
		tables.fold_into(groups.iter().copied(), &mut sums);

		for (column, bit) in row.iter().enumerate() {
			let expected = if bit == B1::ONE {
				weights[0]
			} else {
				B128::ZERO
			};
			assert_eq!(sums.as_slice()[column], expected, "column {column}");
		}
	}

	#[test]
	fn scaling_out_multiplies_every_column() {
		let mut rng = StdRng::seed_from_u64(6);

		// Folding a chunk and scaling it must equal scaling each column sum by hand, which is what
		// lets a caller pay one multiply per column instead of one per row.
		let groups = random_groups::<PackedBinaryField64x1b, 8>(&mut rng);
		let weights = random_scalars::<B128>(&mut rng, 64);
		let scale = random_scalars::<B128>(&mut rng, 1)[0];

		let tables = RowFoldTables::<B128, 8>::new(&weights);
		let mut sums = ColumnSums::zero();
		tables.fold_into(groups.iter().copied(), &mut sums);

		// Start from a non-zero accumulator, so the addition is exercised and not just the scale.
		let mut out = random_scalars::<B128>(&mut rng, 64);
		let before = out.clone();
		sums.add_scaled_to(scale, &mut out);

		for (column, ((&got, &was), &sum)) in
			iter::zip(iter::zip(&out, &before), sums.as_slice()).enumerate()
		{
			assert_eq!(got, was + sum * scale, "column {column}");
		}
	}

	#[test]
	fn folding_no_groups_leaves_the_sums_at_zero() {
		// An empty matrix contributes nothing, so the sums stay where they started.
		let mut rng = StdRng::seed_from_u64(7);
		let weights = random_scalars::<B128>(&mut rng, 64);

		let tables = RowFoldTables::<B128, 8>::new(&weights);
		let mut sums = ColumnSums::zero();
		tables.fold_into(iter::empty::<RowGroup<PackedBinaryField64x1b>>(), &mut sums);

		assert_eq!(sums, ColumnSums::zero());
	}
}
