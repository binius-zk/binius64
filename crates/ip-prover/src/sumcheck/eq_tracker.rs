// Copyright 2023-2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

//! Equality-indicator state carried across the rounds of an MLE-check.
//!
//! An MLE-check weights its hypercube sum by the equality indicator at a fixed point.
//! Carrying that indicator whole would cost one buffer entry per hypercube vertex.
//!
//! [Gruen24] section 3.2 splits it into three factors, per round:
//!
//! ```text
//!     scalar      equality terms of the coordinates already bound
//!     linear      the term in the variable this round binds
//!     expansion   the indicator over the coordinates still untouched
//! ```
//!
//! Only the expansion needs a buffer.
//! It holds one variable fewer than the columns, since the bound one is not expanded.
//!
//! The other two factors are the scalar state every tracker here carries.
//!
//! [Gruen24]: <https://eprint.iacr.org/2024/108>

use std::cmp::min;

use binius_field::{Field, PackedField};
use binius_ip::sumcheck::RoundCoeffs;
use binius_math::{
	field_buffer::FieldBuffer,
	multilinear::eq::{eq_ind_partial_eval, eq_ind_truncate_low_inplace, eq_one_var},
};

use super::round_evals::{RoundEvals2, round_coeffs_by_eq};

/// Where the rounds have reached in the point, and the product they accrued.
#[derive(Debug, Clone)]
struct EqPrefix<F: Field> {
	/// The whole point, including coordinates already bound.
	eval_point: Vec<F>,
	/// How many coordinates are still unbound, counting the one bound next.
	n_vars_remaining: usize,
	/// The product of equality terms over the coordinates already bound.
	eq_prefix_eval: F,
}

impl<F: Field> EqPrefix<F> {
	/// Starts at the full point, with nothing bound and an empty product.
	fn new(eval_point: &[F]) -> Self {
		Self {
			eval_point: eval_point.to_vec(),
			n_vars_remaining: eval_point.len(),
			// An empty product is one, so the first round scales by nothing.
			eq_prefix_eval: F::ONE,
		}
	}

	/// Returns the coordinate of the variable the next round binds.
	///
	/// Rounds bind from the highest coordinate down.
	/// So this walks the point backwards.
	fn next_coordinate(&self) -> F {
		self.eval_point[self.n_vars_remaining - 1]
	}

	/// Binds the current variable to `challenge`, accruing its equality term.
	fn advance(&mut self, challenge: F) {
		// precondition
		assert!(self.n_vars_remaining > 0);

		// The bound variable's linear term becomes a constant in the product.
		self.eq_prefix_eval *= eq_one_var(challenge, self.next_coordinate());
		self.n_vars_remaining -= 1;
	}
}

/// Equality-indicator state for one point, holding the expansion as one buffer.
///
/// A shared column store keeps this shape.
/// Its expansion is read in the same chunks as the columns beside it.
#[derive(Debug, Clone)]
pub struct EqTracker<P: PackedField> {
	/// Where the rounds have reached, and the product they accrued.
	prefix: EqPrefix<P::Scalar>,
	/// The indicator over the coordinates no round has reached.
	expansion: FieldBuffer<P>,
}

impl<F: Field, P: PackedField<Scalar = F>> EqTracker<P> {
	/// Expands the indicator for every coordinate of `eval_point` bar the highest.
	pub fn new(eval_point: &[F]) -> Self {
		// The round about to run keeps the highest coordinate as a linear term.
		// It is therefore never expanded.
		let expanded = &eval_point[..eval_point.len().saturating_sub(1)];
		Self {
			prefix: EqPrefix::new(eval_point),
			expansion: eq_ind_partial_eval(expanded),
		}
	}

	/// Returns the whole point, including coordinates already bound.
	pub fn eval_point(&self) -> &[F] {
		&self.prefix.eval_point
	}

	/// Returns the coordinate of the variable the next round binds.
	pub fn next_coordinate(&self) -> F {
		self.prefix.next_coordinate()
	}

	/// Returns the product of equality terms over the coordinates already bound.
	pub const fn eq_prefix_eval(&self) -> F {
		self.prefix.eq_prefix_eval
	}

	/// Returns the expansion over the coordinates no round has reached.
	pub const fn expansion(&self) -> &FieldBuffer<P> {
		&self.expansion
	}

	/// Returns the expansion for a caller that contracts the values itself.
	pub const fn expansion_mut(&mut self) -> &mut FieldBuffer<P> {
		&mut self.expansion
	}

	/// Advances one round, shrinking the expansion and accruing the equality term.
	///
	/// # Arguments
	///
	/// * `challenge` - The value this round binds the current variable to.
	/// * `shrink` - How the expansion drops its highest variable.
	fn advance(&mut self, challenge: F, shrink: impl FnOnce(&mut FieldBuffer<P>, usize)) {
		// The expansion trails the columns by the coordinate this round binds.
		debug_assert_eq!(self.expansion.log_len(), self.prefix.n_vars_remaining - 1);

		// The last round finds a lone scalar, with nothing left to drop.
		if self.expansion.log_len() > 0 {
			let shrunk = self.expansion.log_len() - 1;
			shrink(&mut self.expansion, shrunk);
		}

		self.prefix.advance(challenge);
	}

	/// Advances one round, contracting the expansion onto the coordinates that remain.
	pub fn fold(&mut self, challenge: F) {
		// Summing the two halves marginalises out the highest variable.
		self.advance(challenge, |expansion, shrunk| eq_ind_truncate_low_inplace(expansion, shrunk));
	}

	/// Advances one round over values the caller has already contracted.
	///
	/// A fused read pass sums the halves during its own traversal.
	/// Only the bookkeeping is then owed, so the stale tail is dropped.
	pub fn truncate_one_var(&mut self, challenge: F) {
		self.advance(challenge, |expansion, shrunk| expansion.truncate(shrunk));
	}
}

/// Equality-indicator state for one point, holding the expansion as an outer product.
///
/// A prover walking the hypercube in chunks reads the two factors at different rates:
///
/// ```text
///     expansion[s * chunk_len + c]  =  chunk[c] * suffix[s]
///
///     chunk    read per element, so it should stay in cache
///     suffix   read once per chunk, as a single scalar
/// ```
///
/// Keeping the per-element factor small is what lets it stay resident.
#[derive(Debug, Clone)]
pub struct ChunkedEqTracker<P: PackedField> {
	/// Where the rounds have reached, and the product they accrued.
	prefix: EqPrefix<P::Scalar>,
	/// The inner factor, over the lowest coordinates, indexed within one chunk.
	chunk: FieldBuffer<P>,
	/// The outer factor, over the coordinates above, indexed by chunk.
	suffix: FieldBuffer<P>,
}

impl<F: Field, P: PackedField<Scalar = F>> ChunkedEqTracker<P> {
	/// Expands the indicator as two factors, for every coordinate bar the highest.
	///
	/// # Arguments
	///
	/// * `max_chunk_vars` - Ceiling on the inner factor's variable count.
	/// * `eval_point` - The point the claim is weighted at.
	pub fn new(max_chunk_vars: usize, eval_point: &[F]) -> Self {
		// The round about to run keeps the highest coordinate as a linear term.
		let expanded = &eval_point[..eval_point.len().saturating_sub(1)];

		// A point below the ceiling puts everything in the inner factor.
		let chunk_vars = min(max_chunk_vars, expanded.len());

		// The inner factor takes the low coordinates, which vary within a chunk.
		let (chunk_point, suffix_point) = expanded.split_at(chunk_vars);
		Self {
			prefix: EqPrefix::new(eval_point),
			chunk: eq_ind_partial_eval(chunk_point),
			suffix: eq_ind_partial_eval(suffix_point),
		}
	}

	/// Returns the coordinate of the variable the next round binds.
	pub fn next_coordinate(&self) -> F {
		self.prefix.next_coordinate()
	}

	/// Returns the inner factor, indexed within one chunk.
	pub const fn chunk(&self) -> &FieldBuffer<P> {
		&self.chunk
	}

	/// Returns the outer factor, indexed by chunk.
	pub const fn suffix(&self) -> &FieldBuffer<P> {
		&self.suffix
	}

	/// Interpolates a degree-2 round polynomial from its sampled evaluations.
	///
	/// # Arguments
	///
	/// * `sum` - The claim this round's polynomial must satisfy.
	/// * `prime_evals` - Sampled evaluations, with the equality factors removed.
	///
	/// # Returns
	///
	/// * The polynomial without its equality factors, which the next claim reduces.
	/// * The same polynomial carrying both factors, which goes on the wire.
	pub fn interpolate2(
		&self,
		sum: F,
		prime_evals: RoundEvals2<F>,
	) -> (RoundCoeffs<F>, RoundCoeffs<F>) {
		let alpha = self.next_coordinate();

		// Dropping the equality factor lowers the degree by one.
		// So one evaluation fewer pins the polynomial down.
		let prime_coeffs = prime_evals.interpolate_eq(sum, alpha);

		// Multiplying the linear term back in restores the second factor.
		// Scaling by the accrued product restores the first.
		let round_coeffs = round_coeffs_by_eq(&prime_coeffs, alpha) * self.prefix.eq_prefix_eval;

		(prime_coeffs, round_coeffs)
	}

	/// Advances one round, contracting whichever factor holds the bound coordinate.
	pub fn fold(&mut self, challenge: F) {
		// Together the factors trail the columns by the coordinate this round binds.
		debug_assert_eq!(
			self.chunk.log_len() + self.suffix.log_len(),
			self.prefix.n_vars_remaining - 1
		);

		// Rounds bind downwards, and the outer factor holds the higher coordinates.
		// So it empties before the inner factor is touched at all.
		//
		//     suffix non-empty  ->  shrink the outer factor
		//     suffix empty      ->  shrink the inner factor
		//     both empty        ->  last round, nothing to drop
		let factor = if self.suffix.log_len() > 0 {
			Some(&mut self.suffix)
		} else if self.chunk.log_len() > 0 {
			Some(&mut self.chunk)
		} else {
			None
		};

		// Summing the two halves marginalises out that factor's highest variable.
		if let Some(factor) = factor {
			eq_ind_truncate_low_inplace(factor, factor.log_len() - 1);
		}

		self.prefix.advance(challenge);
	}
}

#[cfg(test)]
mod tests {
	use binius_field::FieldOps;
	use binius_math::test_utils::{Packed128b, random_scalars};
	use rand::{SeedableRng, rngs::StdRng};

	use super::*;

	type P = Packed128b;
	type F = <P as FieldOps>::Scalar;

	#[test]
	fn fold_contracts_to_the_unbound_prefix() {
		// Invariant: the expansion trails the columns by one variable.
		//
		// The round's own coordinate lives in the linear term, not in the sum.
		// So contracting once must rebuild a fresh expansion of the shorter prefix.
		//
		// Fixture state: 6 coordinates, bound from the highest down.
		//
		//     round 0:  5 expanded,  binds z_5
		//     round 1:  4 expanded,  binds z_4
		//     round 5:  0 expanded,  binds z_0
		let mut rng = StdRng::seed_from_u64(0);
		let n_vars = 6;
		let point = random_scalars::<F>(&mut rng, n_vars);
		let challenges = random_scalars::<F>(&mut rng, n_vars);

		let mut tracker = EqTracker::<P>::new(&point);
		// The reference product, accrued independently of the tracker.
		let mut expected_prefix = F::ONE;

		for (round, &challenge) in challenges.iter().enumerate() {
			let unbound = n_vars - round;

			// The coordinate on offer is the highest one still unbound.
			assert_eq!(tracker.next_coordinate(), point[unbound - 1]);
			// Only the already-bound coordinates have entered the product.
			assert_eq!(tracker.eq_prefix_eval(), expected_prefix);

			// The expansion matches one built from scratch over the lower coordinates.
			let expected = eq_ind_partial_eval::<P>(&point[..unbound - 1]);
			assert_eq!(tracker.expansion().log_len(), expected.log_len());
			for i in 0..expected.len() {
				assert_eq!(tracker.expansion().get(i), expected.get(i), "round {round}, slot {i}");
			}

			tracker.fold(challenge);
			expected_prefix *= eq_one_var(challenge, point[unbound - 1]);
		}

		// Every coordinate is bound, so the product covers the whole point.
		assert_eq!(tracker.eq_prefix_eval(), expected_prefix);
	}

	#[test]
	fn truncate_matches_fold_except_in_the_values() {
		// Invariant: a fused read pass sums the halves during its own traversal.
		//
		// It then owes the tracker only the bookkeeping.
		// So the two entries agree on everything but the buffer contents.
		//
		//     fold      sums the halves,  so the values contract
		//     truncate  drops the tail,   so the values stay the original front
		//     both      same length, same accrued product
		let mut rng = StdRng::seed_from_u64(1);
		let n_vars = 5;
		let point = random_scalars::<F>(&mut rng, n_vars);
		let challenges = random_scalars::<F>(&mut rng, n_vars);

		// Two trackers over one point, advanced through opposite entries.
		let mut folded = EqTracker::<P>::new(&point);
		let mut truncated = EqTracker::<P>::new(&point);
		// The values the truncating path never rewrites.
		let original = eq_ind_partial_eval::<P>(&point[..n_vars - 1]);

		for (round, &challenge) in challenges.iter().enumerate() {
			folded.fold(challenge);
			truncated.truncate_one_var(challenge);

			// Both entries run the same bookkeeping, so it cannot diverge.
			assert_eq!(folded.eq_prefix_eval(), truncated.eq_prefix_eval(), "round {round}");
			assert_eq!(
				folded.expansion().log_len(),
				truncated.expansion().log_len(),
				"round {round}"
			);

			// Truncation leaves the front of the original values untouched.
			for i in 0..truncated.expansion().len() {
				assert_eq!(
					truncated.expansion().get(i),
					original.get(i),
					"round {round}, slot {i}"
				);
			}
		}
	}

	#[test]
	fn chunk_and_suffix_factor_the_expansion() {
		// Invariant: the chunked mode never materialises the full expansion.
		//
		// Its consumer reads one inner value and weights it by one outer scalar.
		// So the outer product must rebuild the full expansion after every fold.
		//
		// Fixture state: 7 coordinates, inner factor capped at 3.
		//
		//     round 0:  chunk 3 vars,  suffix 3 vars
		//     round 3:  chunk 3 vars,  suffix 0 vars
		//     round 6:  both a lone scalar
		//
		//     expansion[s * chunk_len + c] == chunk[c] * suffix[s]
		let mut rng = StdRng::seed_from_u64(2);
		let n_vars = 7;
		let max_chunk_vars = 3;
		let point = random_scalars::<F>(&mut rng, n_vars);
		let challenges = random_scalars::<F>(&mut rng, n_vars);

		let mut tracker = ChunkedEqTracker::<P>::new(max_chunk_vars, &point);

		for (round, &challenge) in challenges.iter().enumerate() {
			let unbound = n_vars - round;
			// The reference the two factors must rebuild between them.
			let full = eq_ind_partial_eval::<P>(&point[..unbound - 1]);
			let chunk = tracker.chunk();
			let suffix = tracker.suffix();

			// Between them the factors cover exactly the expanded coordinates.
			assert_eq!(chunk.log_len() + suffix.log_len(), unbound - 1, "round {round}");

			// The outer factor indexes chunks, the inner one indexes within a chunk.
			for s in 0..suffix.len() {
				for c in 0..chunk.len() {
					assert_eq!(
						full.get(s * chunk.len() + c),
						chunk.get(c) * suffix.get(s),
						"round {round}, suffix {s}, chunk {c}"
					);
				}
			}

			tracker.fold(challenge);
		}
	}
}
