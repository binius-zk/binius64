// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

//! Folding a matrix of single-bit rows against one weight per row.

use std::array;

use binius_field::{
	BinaryField, Divisible, PackedField, WithUnderlier, transpose::transpose_square_blocks_array,
	util::expand_subset_sums_array,
};
use binius_verifier::config::B1;

/// Weights one subset-sum table covers.
///
/// Eight is the widest group whose lookup index still fits one byte.
/// One table load then replaces eight conditional additions.
///
/// The row fold below weights rows, so its tables cover eight rows.
/// The bit-axis word fold weights bit positions and reuses the same geometry, so this counts
/// weights rather than naming either axis.
pub const WEIGHTS_PER_TABLE: usize = 1 << LOG_WEIGHTS_PER_TABLE;

/// Base-2 log of the weights one subset-sum table covers.
pub const LOG_WEIGHTS_PER_TABLE: usize = 3;

/// Builds the subset-sum tables of a bitwise row fold.
///
/// # Overview
///
/// A row fold contracts a matrix over GF(2) against one weight per row:
///
/// ```text
///     out[b] = sum_r weight[r] * bit_b(row[r])
/// ```
///
/// Taking eight rows at a time turns that inner sum into a single table lookup.
/// Table `g` covers rows `8g .. 8g+8` and holds every subset sum of their eight weights.
/// A byte carrying those eight rows' bits at one column then indexes their contribution directly.
///
/// # Arguments
///
/// * `weights` - one weight per row, from the first row onwards
///
/// # Why short input is allowed
///
/// Weights past the end of the slice are read as zero.
/// They would weight rows past the end of a chunk, which are themselves read as zero.
/// So a zero weight and its absent row contribute nothing either way.
/// This is what lets one table layout serve a chunk that the row list does not fill.
pub fn row_fold_tables<F: BinaryField, const N_TABLES: usize>(
	weights: &[F],
) -> [[F; 1 << WEIGHTS_PER_TABLE]; N_TABLES] {
	array::from_fn(|group| {
		// Weights of the eight rows this table covers, zero where the slice has run out.
		// A group beyond the end of the slice starts at its end, so it copies nothing.
		let mut group_weights = [F::ZERO; WEIGHTS_PER_TABLE];
		let start = (group * WEIGHTS_PER_TABLE).min(weights.len());
		let available = (weights.len() - start).min(WEIGHTS_PER_TABLE);
		group_weights[..available].copy_from_slice(&weights[start..start + available]);

		// Enumerate all 256 subset sums, so any byte of set bits indexes its sum in one load.
		expand_subset_sums_array(group_weights)
	})
}

/// Folds one group of eight rows into a column accumulator.
///
/// # Overview
///
/// Rows arrive one bit per scalar, so a row is one packed element and a column is a scalar index.
/// One table covers one group, holding every subset sum of that group's eight row weights.
///
/// # Algorithm
///
/// Transposing the group exchanges its row axis with the low three bits of the column index:
///
/// ```text
///     before:  element r, bit 8i + j  =  row r, column 8i + j
///     after:   element j, bit 8i + t  =  row t, column 8i + j
/// ```
///
/// So byte `i` of element `j` then carries the eight rows' bits at column `8i + j`.
/// One lookup of that byte yields those rows' whole contribution to that column.
///
/// The accumulator is nested to match, as `[byte of column index][low three bits]`.
/// Reading that nesting in order walks the columns in order.
///
/// # Preconditions
///
/// * The row width in bits must equal eight times the accumulator's outer length.
/// * Rows past the end of the matrix must be passed as zero, which contributes nothing.
#[inline]
pub fn fold_row_group<F, PB, const N_TABLES: usize>(
	rows: &[PB; WEIGHTS_PER_TABLE],
	table: &[F; 1 << WEIGHTS_PER_TABLE],
	acc: &mut [[F; WEIGHTS_PER_TABLE]; N_TABLES],
) where
	F: BinaryField,
	PB: PackedField<Scalar = B1> + WithUnderlier,
	PB::Underlier: Divisible<u8>,
{
	// One byte of a row per table is what makes the nesting below line up with the columns.
	const {
		assert!(
			PB::WIDTH == WEIGHTS_PER_TABLE * N_TABLES,
			"the row width must be one byte per table"
		);
	}

	// The transpose consumes its input, so work on a copy and leave the caller's rows intact.
	let mut group = *rows;
	transpose_square_blocks_array::<PB, LOG_WEIGHTS_PER_TABLE, WEIGHTS_PER_TABLE>(&mut group);

	for (j, row) in group.iter().enumerate() {
		// Byte `i` holds this group's bits at column `8i + j`, so it indexes that column's sum.
		for (i, byte) in Divisible::<u8>::value_iter(row.to_underlier()).enumerate() {
			acc[i][j] += table[byte as usize];
		}
	}
}
