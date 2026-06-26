// Copyright 2025 Irreducible Inc.
//! Equality-indicator convert table for the AND-reduction univariate round message.
//!
//! The round message weights every hypercube vertex by a large-field equality factor.
//!
//! - The innermost equality coordinates are tabulated into a convert table.
//! - Their per-vertex large-field multiplies become table lookups and XORs.
//! - This is the convert-table idea of Flock Section 4.3.
//! - The output equals the unoptimized weighting; only the arithmetic changes.

use binius_core::word::Word;
use binius_field::{AESTowerField8b, BinaryField, PackedField};
use binius_math::multilinear::eq::eq_ind_partial_eval;
use binius_utils::rayon::prelude::*;
use binius_verifier::protocols::bitand::ROWS_PER_HYPERCUBE_VERTEX;
use itertools::izip;

use super::{
	ntt_lookup::NTTLookup, sumcheck_round_messages::univariate_round_message_extension_domain,
};

/// Number of innermost equality coordinates folded into the convert table.
///
/// - The table holds one row per value of these coordinates: `2^4 = 16` rows.
/// - Each row has `256` field elements, so the table is `64` KiB for a 128-bit field.
/// - That size stays resident in cache during the round message.
pub const INNER_EQ_VARS: usize = 4;

/// Row count of the convert table: one weight per value of the folded coordinates (`2^4 = 16`).
pub const INNER_EQ_SIZE: usize = 1 << INNER_EQ_VARS;

/// Words spanned by one super-chunk.
///
/// - A super-chunk fixes the outer coordinates.
/// - It sweeps the eight small-coordinate words across all 16 inner values.
const WORDS_PER_SUPERCHUNK: usize = 8 * INNER_EQ_SIZE;

/// Convert table: an inner-coordinate index and an `F_{2^8}` byte map to a weighted field value.
///
/// Entry `(b, v)` holds `eq_inner[b] * phi(v)`:
/// - `eq_inner[b]` is the equality weight of inner index `b`.
/// - `phi(v)` embeds the byte `v` into the large field.
///
/// This turns a per-vertex large-field multiply into one table lookup.
struct EqConvertTable<F> {
	/// Row `b` holds the `256` products `eq_inner[b] * phi(v)` for byte values `v`.
	rows: Box<[[F; 256]; INNER_EQ_SIZE]>,
}

impl<F: BinaryField + From<AESTowerField8b>> EqConvertTable<F> {
	/// Builds the table from the inner equality weights.
	///
	/// # Arguments
	///
	/// - `eq_inner`: the 16 equality weights over the inner coordinates.
	fn new(eq_inner: &[F]) -> Self {
		// One weight per inner index; reject a mismatched expansion up front.
		assert_eq!(eq_inner.len(), INNER_EQ_SIZE);

		// Zero-initialize the 16 x 256 table of weighted embeddings.
		let mut rows = Box::new([[F::ZERO; 256]; INNER_EQ_SIZE]);

		// Fill each row b with the products eq_inner[b] * phi(v) over all byte values v.
		for (b, row) in rows.iter_mut().enumerate() {
			let weight = eq_inner[b];
			for (v, cell) in row.iter_mut().enumerate() {
				// phi(v): the byte denotes a subfield element, embedded into the large field.
				let phi = F::from(AESTowerField8b::new(v as u8));
				*cell = weight * phi;
			}
		}

		Self { rows }
	}

	/// Returns `eq_inner[b] * phi(v)` for inner index `b` and byte value `v`.
	#[inline]
	fn get(&self, b: usize, v: u8) -> F {
		self.rows[b][v as usize]
	}
}

/// Computes the AND-reduction univariate round message via the equality-indicator convert table.
///
/// The result equals the unoptimized round message; only the equality weighting is reorganized.
/// The large-field equality weighting is factored into an inner block and an outer block:
/// - the inner block is absorbed into a convert table and applied with lookups and XORs,
/// - the outer block is applied with one large-field multiply per output row per super-chunk.
///
/// With fewer than four large-field coordinates there is no inner block, so it delegates to the
/// unoptimized path.
///
/// # Arguments
///
/// - `first_col`, `second_col`, `third_col`: the one-bit oblong multilinears `a`, `b`, `c`.
/// - `big_field_zerocheck_challenges`: the large-field zerocheck coordinates, innermost first.
/// - `ntt_lookup`: the lookup-based low-degree-extension table for the skipped bit-index variables.
/// - `small_field_zerocheck_challenges`: the small-field coordinates handled in `F_{2^8}`.
///
/// # Returns
///
/// The round polynomial evaluations on the extension domain.
///
/// # Algorithm
///
/// Split the large-field coordinates into an inner block (the innermost four) and the rest.
///
/// ```text
///   for each outer super-chunk s (one large-field weight eq_outer[s]):
///       row_acc[r] = 0
///       for b in 0..INNER_EQ_SIZE:                 # the inner coordinates
///           t = low-degree extension of word-chunk (s, b), small-eq weighted   # in F_{2^8}
///           for r in 0..ROWS:
///               row_acc[r] += convert[b][byte(t, r)]      # lookup + XOR, no large-field multiply
///       for r in 0..ROWS:
///           out[r] += eq_outer[s] * row_acc[r]            # one large-field multiply per row
/// ```
///
/// # Performance
///
/// - Large-field multiplies in the weighting drop by a factor of 16 (the inner-block size).
/// - Each eliminated multiply becomes one table lookup and one XOR.
pub fn univariate_round_message<F, P>(
	first_col: &[Word],
	second_col: &[Word],
	third_col: &[Word],
	big_field_zerocheck_challenges: &[F],
	ntt_lookup: &NTTLookup<P>,
	small_field_zerocheck_challenges: &[AESTowerField8b],
) -> [F; ROWS_PER_HYPERCUBE_VERTEX]
where
	F: BinaryField + From<AESTowerField8b>,
	P: PackedField<Scalar = AESTowerField8b>,
{
	// One packed element carries all 64 output rows of a vertex.
	assert_eq!(P::WIDTH, ROWS_PER_HYPERCUBE_VERTEX);

	// With fewer than INNER_EQ_VARS large-field coordinates there is no inner block to tabulate.
	// Fall back to the unoptimized weighting over the full equality buffer.
	if big_field_zerocheck_challenges.len() < INNER_EQ_VARS {
		let eq_big = eq_ind_partial_eval::<F>(big_field_zerocheck_challenges);
		return univariate_round_message_extension_domain(
			first_col,
			second_col,
			third_col,
			&eq_big,
			ntt_lookup,
			small_field_zerocheck_challenges,
		);
	}

	// Innermost INNER_EQ_VARS coordinates form the tabulated inner block; the rest are outer.
	// The innermost coordinates occupy the low bits of the chunk index, matching the eq-buffer
	// order.
	let (inner, outer) = big_field_zerocheck_challenges.split_at(INNER_EQ_VARS);

	// Expand the inner coordinates into their 16 equality weights and bake them into the table.
	let eq_inner = eq_ind_partial_eval::<F>(inner);
	let convert = EqConvertTable::<F>::new(eq_inner.as_ref());

	// Expand the outer coordinates into one large-field weight per super-chunk.
	let eq_outer = eq_ind_partial_eval::<F>(outer);
	let eq_outer = eq_outer.as_ref();

	// Broadcast the small-field equality weights to packed form for the inner kernel.
	let eq_ind_small = eq_ind_partial_eval::<AESTowerField8b>(small_field_zerocheck_challenges)
		.iter_scalars()
		.map(P::broadcast)
		.collect::<Vec<_>>();

	// Process each outer super-chunk independently, then XOR the per-super-chunk contributions.
	(
		first_col.par_chunks(WORDS_PER_SUPERCHUNK),
		second_col.par_chunks(WORDS_PER_SUPERCHUNK),
		third_col.par_chunks(WORDS_PER_SUPERCHUNK),
		eq_outer.par_iter(),
	)
		.into_par_iter()
		.map(|(a_super, b_super, c_super, &eq_outer_weight)| {
			// Accumulator over the inner block for this super-chunk, one entry per output row.
			// Built from convert-table lookups and XORs, with no large-field multiply.
			let mut row_acc = [F::ZERO; ROWS_PER_HYPERCUBE_VERTEX];

			// Range the inner coordinates: inner index b in [0, INNER_EQ_SIZE).
			for b in 0..INNER_EQ_SIZE {
				// The eight small-coordinate words for inner step b within this super-chunk.
				let lo = b * 8;
				let summed_ntt = summed_ntt_for_chunk(
					&a_super[lo..lo + 8],
					&b_super[lo..lo + 8],
					&c_super[lo..lo + 8],
					&eq_ind_small,
					ntt_lookup,
				);

				// Fold each row's byte into the accumulator through the convert table.
				// The table already holds the weighted embedding, so no large-field multiply runs
				// here.
				for (acc_i, scalar) in row_acc.iter_mut().zip(summed_ntt.into_iter()) {
					*acc_i += convert.get(b, u8::from(scalar));
				}
			}

			// Apply the single outer large-field weight to every row.
			let mut partial = [F::ZERO; ROWS_PER_HYPERCUBE_VERTEX];
			for (out_i, &acc_i) in partial.iter_mut().zip(row_acc.iter()) {
				*out_i = eq_outer_weight * acc_i;
			}
			partial
		})
		.reduce(
			|| [F::ZERO; ROWS_PER_HYPERCUBE_VERTEX],
			|mut lhs, rhs| {
				// Combine two super-chunk contributions by XOR, the additive group of the field.
				for (lhs_i, rhs_i) in lhs.iter_mut().zip(rhs) {
					*lhs_i += rhs_i;
				}
				lhs
			},
		)
}

/// Low-degree-extends one eight-word chunk and weights it by the small-field equality indicator.
///
/// The chunk spans the three small coordinates of one hypercube vertex.
///
/// - Each of the three one-bit oblong columns is mapped through the lookup-based NTT.
/// - The columns are combined as `a * b - c` and summed over the small coordinates.
///
/// # Returns
///
/// The 64 output-row evaluations as one packed `F_{2^8}` element.
fn summed_ntt_for_chunk<P>(
	a_chunk: &[Word],
	b_chunk: &[Word],
	c_chunk: &[Word],
	eq_ind_small: &[P],
	ntt_lookup: &NTTLookup<P>,
) -> P
where
	P: PackedField<Scalar = AESTowerField8b>,
{
	// Running sum over the eight small-coordinate vertices, in packed F_{2^8}.
	let mut summed_ntt = P::zero();

	// Each iteration handles one vertex of the small sub-hypercube and its equality weight.
	for (a_i, b_i, c_i, &weight) in izip!(a_chunk, b_chunk, c_chunk, eq_ind_small) {
		// Low-degree-extend the three columns from their 64-bit words via the lookup table.
		let [first_col_ntt, second_col_ntt, third_col_ntt] =
			ntt_lookup.multi_ntt_array([a_i.0, b_i.0, c_i.0]);

		// Form the AND residual a * b - c and weight it by this vertex's small equality value.
		summed_ntt += (first_col_ntt * second_col_ntt - third_col_ntt) * weight;
	}

	summed_ntt
}

#[cfg(test)]
mod tests {
	use std::iter::repeat_with;

	use binius_field::{PackedAESBinaryField64x8b, Random, WithUnderlier};
	use binius_math::BinarySubspace;
	use binius_verifier::{config::B128, protocols::bitand::SKIPPED_VARS};
	use proptest::prelude::*;
	use rand::{SeedableRng, rngs::StdRng};

	use super::*;

	// The three small-field coordinates handled in F_{2^8}, matching the live protocol parameters.
	const SMALL_CHALLENGES: [AESTowerField8b; 3] = [
		AESTowerField8b::new(0x2),
		AESTowerField8b::new(0x4),
		AESTowerField8b::new(0x10),
	];

	// A random 64-bit word drawn from the low half of a random large-field element.
	// This routes randomness through the field sampler, sidestepping rand prelude method clashes.
	fn random_words(n: usize, rng: &mut StdRng) -> Vec<Word> {
		repeat_with(|| Word(u128::from(B128::random(&mut *rng).to_underlier()) as u64))
			.take(n)
			.collect()
	}

	// Runs the optimized and the unoptimized round message on one witness and asserts they agree.
	fn assert_optimized_matches_reference(log_num_rows: usize, seed: u64) {
		let mut rng = StdRng::seed_from_u64(seed);

		// One word per skipped-variable block: 2^(log_num_rows - SKIPPED_VARS) words per column.
		let n_words = 1usize << (log_num_rows - SKIPPED_VARS);

		// Random a, b columns and c = a & b, a satisfying AND-constraint witness.
		let first = random_words(n_words, &mut rng);
		let second = random_words(n_words, &mut rng);
		let third: Vec<Word> = first.iter().zip(&second).map(|(&a, &b)| a & b).collect();

		// Random large-field coordinates, innermost first.
		let n_big = log_num_rows - SKIPPED_VARS - SMALL_CHALLENGES.len();
		let big: Vec<B128> = repeat_with(|| B128::random(&mut rng)).take(n_big).collect();

		// The lookup-based low-degree-extension table for the skipped bit-index variables.
		let prover_message_domain = BinarySubspace::<AESTowerField8b>::with_dim(SKIPPED_VARS + 1);
		let ntt_lookup = NTTLookup::<PackedAESBinaryField64x8b>::new(&prover_message_domain);

		// Reference: the unoptimized weighting over the full equality buffer.
		let eq_big = eq_ind_partial_eval::<B128>(&big);
		let reference: [B128; ROWS_PER_HYPERCUBE_VERTEX] =
			univariate_round_message_extension_domain(
				&first,
				&second,
				&third,
				&eq_big,
				&ntt_lookup,
				&SMALL_CHALLENGES,
			);

		// Optimized: the convert-table weighting over the raw coordinates.
		let optimized = univariate_round_message::<B128, PackedAESBinaryField64x8b>(
			&first,
			&second,
			&third,
			&big,
			&ntt_lookup,
			&SMALL_CHALLENGES,
		);

		// Both compute the same polynomial, so every evaluation must match exactly.
		assert_eq!(optimized, reference);
	}

	#[test]
	fn optimized_matches_reference_with_inner_block() {
		// log_num_rows = 14: SKIPPED_VARS(6) + small(3) + big(5), so the inner block of 4 applies.
		assert_optimized_matches_reference(14, 0);
	}

	#[test]
	fn optimized_matches_reference_fallback_path() {
		// log_num_rows = 11: skipped(6) + small(3) + big(2), so big is below the inner-block size.
		//
		//     big = 2 < 4 → no inner block → delegates to the unoptimized path
		assert_optimized_matches_reference(11, 1);
	}

	proptest! {
		// Across random witnesses and sizes, the optimized path reproduces the reference exactly.
		#[test]
		fn optimized_matches_reference_proptest(
			extra_big_vars in 0usize..4,
			seed in any::<u64>(),
		) {
			// log_num_rows = 13 gives big = 4 (the smallest size with a full inner block).
			assert_optimized_matches_reference(13 + extra_big_vars, seed);
		}
	}
}
