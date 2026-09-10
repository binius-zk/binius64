// Copyright 2026 The Binius Developers

//! The Zero reduction.
//!
//! A ZERO constraint asserts that one constraint array vanishes. Because the constraint is linear,
//! the polynomial whose vanishing is at stake is the array's oblong-multilinearization `d_hat`
//! itself, rather than a product of witness multilinears. A multilinear that vanishes on the cube
//! is the zero polynomial, so a single evaluation at a point the prover could not predict certifies
//! the constraints outright.
//!
//! The reduction therefore carries no prover message and no sumcheck round. Its whole output is the
//! claim
//!
//! ```text
//! d_hat(r_zhat_prime, rho) = 0
//! ```
//!
//! which joins the BitAnd, IntMul and BinMul claims in the batch the shift reduction consumes. The
//! shift reduction, running against the committed witness, is what certifies it: a violated ZERO
//! constraint makes the true batched evaluation differ from the claimed one, and its final check
//! fails.

/// Builds the constraint point the Zero reduction closes at.
///
/// The reduction runs directly after the BitAnd sumcheck and evaluates at the constraint half of
/// the point that sumcheck has just produced, `r_x_star`: `rho` is its
/// length-`log_zero_constraints` prefix. The sumcheck runs over the longest of the AND, IMUL and
/// BMUL arrays, so fresh challenges from `sample` extend `r_x_star` only when the ZERO array is
/// longer still. The prover and verifier derive it identically, so `sample` draws the same
/// challenges at the same point in both transcripts.
///
/// Asked for `max(r_x_star.len(), log_zero_constraints)` coordinates instead, it returns the
/// unified constraint point every operation is claimed at a prefix of, `rho` included. It draws the
/// same challenges either way.
///
/// `sample` stands in for the channel: the prover and verifier channel traits are unrelated, so one
/// function can only reach both through a closure.
///
/// Every coordinate of `rho` is uniform and independent — those inherited from the BitAnd sumcheck
/// are its per-round verifier challenges — so the soundness bound is plain Schwartz-Zippel on a
/// polynomial the witness commitment pinned down before the first challenge was drawn. The
/// reduction's *input* zerocheck challenges would serve too, but they open with the deterministic
/// Rijndael coordinates, which carry no randomness and would need absorbing by a separate
/// F_2-independence argument first.
///
/// Sharing the point with the BitAnd reduction is sound despite the resulting correlation: the Zero
/// bound appeals only to the marginal distribution of the challenges it consumes, which reuse
/// leaves untouched. What it does require is that every challenge be drawn after the witness is
/// committed, which the phase ordering guarantees.
pub fn reduction_point<F: Clone>(
	r_x_star: &[F],
	log_zero_constraints: usize,
	mut sample: impl FnMut() -> F,
) -> Vec<F> {
	// Each set is padded to a power-of-two row count independently, so a ZERO set with more rows
	// than every AND, IMUL and BMUL set runs past `r_x_star` and samples the rest.
	(0..log_zero_constraints)
		.map(|i| r_x_star.get(i).map_or_else(&mut sample, F::clone))
		.collect()
}

#[cfg(test)]
mod tests {
	use super::*;

	/// Panics if called: a point that fits inside the BitAnd point must sample nothing.
	fn unused_sample() -> u32 {
		panic!("no challenge should be sampled");
	}

	#[test]
	fn test_reduction_point_truncates_when_zero_set_is_smaller() {
		let bitand = [1u32, 2, 3, 4, 5];
		assert_eq!(reduction_point(&bitand, 3, unused_sample), vec![1, 2, 3]);
	}

	#[test]
	fn test_reduction_point_extends_when_zero_set_is_larger() {
		let bitand = [1u32, 2, 3];
		let mut extra = [7u32, 8].into_iter();
		assert_eq!(
			reduction_point(&bitand, 5, || extra.next().expect("two extra challenges")),
			vec![1, 2, 3, 7, 8]
		);
	}

	#[test]
	fn test_reduction_point_of_equal_sets_is_the_bitand_point() {
		let bitand = [1u32, 2, 3, 4];
		assert_eq!(reduction_point(&bitand, 4, unused_sample), bitand.to_vec());
	}

	#[test]
	fn test_reduction_point_of_an_empty_zero_set_is_empty() {
		let bitand = [1u32, 2, 3, 4];
		assert!(reduction_point(&bitand, 0, unused_sample).is_empty());
	}
}
