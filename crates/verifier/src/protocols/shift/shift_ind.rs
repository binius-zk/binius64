// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

//! Evaluation of the shift indicator multilinear extensions.

use std::iter;

use binius_core::word::Word;
use binius_field::FieldOps;
use binius_math::{line::extrapolate_line, multilinear::eq::eq_one_var};

use super::SHIFT_VARIANT_COUNT;

/// The number of bit variables a half-word (`*32`) shift variant acts over.
const HALF_WORD_LOG_BITS: usize = Word::LOG_BITS - 1;

/// Evaluates the eight shift indicator multilinear extensions at one point.
///
/// A shift indicator is the 3·[`Word::LOG_BITS`]-variate multilinear extension of the predicate
/// relating an output bit index `i`, an input bit index `j` and a shift amount `s`:
///
/// ```text
/// sll   i == j + s            sll32   i / 32 == j / 32 and i % 32 == j % 32 + s % 32
/// srl   j == i + s            srl32   likewise, over each half independently
/// sra   j == min(i + s, 63)   sra32
/// rotr  j == (i + s) % 64     rotr32
/// ```
///
/// The results are ordered by [`ShiftVariant`](binius_core::ShiftVariant), so the array indexes
/// straight by variant.
///
/// ## Preconditions
///
/// * `r_i`, `r_j` and `r_s` each have exactly [`Word::LOG_BITS`] entries
pub fn evaluate_shift_inds<E: FieldOps>(
	r_i: &[E],
	r_j: &[E],
	r_s: &[E],
) -> [E; SHIFT_VARIANT_COUNT] {
	assert_eq!(r_i.len(), Word::LOG_BITS); // precondition
	assert_eq!(r_j.len(), Word::LOG_BITS); // precondition
	assert_eq!(r_s.len(), Word::LOG_BITS); // precondition

	let [sll, srl, sra, rotr] = evaluate_variants(r_i, r_j, r_s);

	// A half-word variant applies the same four rules to each 32-bit half, reading only the low
	// bits of the shift amount. The two halves never mix, so the indicator also carries the
	// equality of the halves the two bit indices fall in.
	let [sll32, srl32, sra32, rotr32] = evaluate_variants(
		&r_i[..HALF_WORD_LOG_BITS],
		&r_j[..HALF_WORD_LOG_BITS],
		&r_s[..HALF_WORD_LOG_BITS],
	);
	let same_half = eq_one_var(r_i[HALF_WORD_LOG_BITS].clone(), r_j[HALF_WORD_LOG_BITS].clone());

	[
		sll,
		srl,
		sra,
		rotr,
		same_half.clone() * sll32,
		same_half.clone() * srl32,
		same_half.clone() * sra32,
		same_half * rotr32,
	]
}

/// Evaluates the four shift indicators over `r_i.len()` bit variables, in variant order.
///
/// All four are built from the indicators of the addition `i + s`: whether it equals `j`, and
/// whether it equals `j` only after wrapping past the top bit.
fn evaluate_variants<E: FieldOps>(r_i: &[E], r_j: &[E], r_s: &[E]) -> [E; 4] {
	let (srl, wrapped) = evaluate_sum_inds(r_i, r_j, r_s);

	// A left shift is a right shift with the two bit indices exchanged: `i == j + s`.
	let (sll, _) = evaluate_sum_inds(r_j, r_i, r_s);

	// An arithmetic right shift agrees with the logical one until the sum overflows, past which
	// every output bit reads the sign bit. The product of the `r_j` coordinates is the equality
	// indicator selecting that top position.
	let sign_position: E = r_j.iter().cloned().product();
	let sra = srl.clone() + sign_position * evaluate_overflow_ind(r_i, r_s);

	// A rotation keeps the bits the sum carries past the top, wrapping them around instead.
	let rotr = srl.clone() + wrapped;

	[sll, srl, sra, rotr]
}

/// Evaluates the multilinear extensions of the two indicators of the addition `i + s`:
/// `i + s == j`, and `i + s == j + 2^n` — the sum wrapped around the top bit.
///
/// The addition is checked one bit position at a time, carrying between positions, so the state
/// is the pair of indicators that the positions below sum correctly with a carry out of 0 and of
/// 1 respectively. Both start certain: the empty sum is correct and carries nothing.
fn evaluate_sum_inds<E: FieldOps>(r_i: &[E], r_j: &[E], r_s: &[E]) -> (E, E) {
	iter::zip(iter::zip(r_i, r_j), r_s).fold(
		(E::one(), E::zero()),
		|(no_carry, carry), ((i, j), s)| {
			// A position's transition reads `j` and `s` through three weights only: the two agree,
			// only `j` is set, or only `s` is set.
			let both = j.clone() * s;
			let j_only = j.clone() - &both;
			let s_only = s.clone() - &both;
			let agree = eq_one_var(j.clone(), s.clone());

			// The transitions at `i = 0` and at `i = 1`, interpolated at the coordinate. With
			// `i = 0` the position sums correctly either when `j` agrees with `s` and nothing
			// carries in, or when only `j` is set and the carry coming in fills it; it carries out
			// only when `s` absorbs a carry in. With `i = 1` the position carries out whenever `s`
			// is set or a carry comes in, and sums correctly against the `j` those leave.
			(
				extrapolate_line(
					agree.clone() * &no_carry + j_only.clone() * &carry,
					j_only * &no_carry,
					i.clone(),
				),
				extrapolate_line(
					s_only.clone() * &carry,
					s_only * &no_carry + agree * &carry,
					i.clone(),
				),
			)
		},
	)
}

/// Evaluates the multilinear extension of the indicator that the addition `i + s` overflows past
/// the top bit, `i + s >= 2^n`.
///
/// This is [`evaluate_sum_inds`]'s carry state with the input bit index summed out: exactly one
/// `j` completes the addition for each `(i, s)`, so summing over `j` leaves the carry alone.
fn evaluate_overflow_ind<E: FieldOps>(r_i: &[E], r_s: &[E]) -> E {
	iter::zip(r_i, r_s).fold(E::zero(), |carry, (i, s)| {
		// With `i = 0` only a carry in that `s` propagates survives; with `i = 1` the position
		// carries out whenever `s` is set or a carry comes in.
		extrapolate_line(s.clone() * &carry, s.clone() + (E::one() - s) * &carry, i.clone())
	})
}

#[cfg(test)]
mod tests {
	use std::array;

	use binius_field::{Field, Ghash128b as B128};
	use binius_math::test_utils::random_scalars;
	use rand::{RngExt, SeedableRng, rngs::StdRng};

	use super::*;

	/// The bit-index hypercube vertex at `index`.
	fn bit_index_vertex(index: usize) -> [B128; Word::LOG_BITS] {
		array::from_fn(|bit| {
			if (index >> bit) & 1 == 1 {
				B128::ONE
			} else {
				B128::ZERO
			}
		})
	}

	/// Over the hypercube the eight evaluations must be the indicators of the integer relations
	/// the shift variants are defined by.
	#[test]
	fn hypercube_vertices_match_the_shift_relations() {
		let mut rng = StdRng::seed_from_u64(0);

		for _trial in 0..1024 {
			let i = rng.random_range(0..Word::BITS);
			let j = rng.random_range(0..Word::BITS);
			let s = rng.random_range(0..Word::BITS);

			let evals = evaluate_shift_inds(
				&bit_index_vertex(i),
				&bit_index_vertex(j),
				&bit_index_vertex(s),
			);

			let (i_lo, j_lo, s_lo) = (i % 32, j % 32, s % 32);
			let same_half = i / 32 == j / 32;
			let expected = [
				i == j + s,
				j == i + s,
				j == (i + s).min(Word::BITS - 1),
				j == (i + s) % Word::BITS,
				same_half && i_lo == j_lo + s_lo,
				same_half && j_lo == i_lo + s_lo,
				same_half && j_lo == (i_lo + s_lo).min(31),
				same_half && j_lo == (i_lo + s_lo) % 32,
			];

			for (variant, (eval, expected)) in iter::zip(evals, expected).enumerate() {
				let expected = if expected { B128::ONE } else { B128::ZERO };
				assert_eq!(eval, expected, "variant {variant} at i={i}, j={j}, s={s}");
			}
		}
	}

	/// The evaluations must be multilinear in every coordinate: their hypercube values pin the
	/// polynomials down only together with this.
	#[test]
	fn evaluations_are_multilinear_in_every_coordinate() {
		let mut rng = StdRng::seed_from_u64(0);
		let point = [
			random_scalars::<B128>(&mut rng, Word::LOG_BITS),
			random_scalars::<B128>(&mut rng, Word::LOG_BITS),
			random_scalars::<B128>(&mut rng, Word::LOG_BITS),
		]
		.concat();

		let evaluate = |point: &[B128]| {
			let (r_i, rest) = point.split_at(Word::LOG_BITS);
			let (r_j, r_s) = rest.split_at(Word::LOG_BITS);
			evaluate_shift_inds(r_i, r_j, r_s)
		};

		// Every coordinate of all three points, restricted to 0 and to 1.
		for coordinate in 0..point.len() {
			let restrict = |value| {
				let mut restricted = point.clone();
				restricted[coordinate] = value;
				evaluate(&restricted)
			};

			for (variant, ((eval, at_zero), at_one)) in
				iter::zip(iter::zip(evaluate(&point), restrict(B128::ZERO)), restrict(B128::ONE))
					.enumerate()
			{
				assert_eq!(
					eval,
					extrapolate_line(at_zero, at_one, point[coordinate]),
					"variant {variant} is not linear in coordinate {coordinate}"
				);
			}
		}
	}
}
