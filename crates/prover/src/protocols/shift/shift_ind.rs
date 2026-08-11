// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

//! The sumcheck binding the bit index the h polynomials sum over, and the shift indicator
//! partial evaluations it runs over.

use binius_compute::Allocator;
use binius_core::word::Word;
use binius_field::{BinaryField, FieldOps, PackedField};
use binius_ip_prover::{
	channel::IPProverChannel,
	sumcheck::{bivariate_product_prover, prove_single},
};
use binius_math::{
	BinarySubspace, FieldBuffer, FieldVec, inner_product::inner_product,
	multilinear::eq::eq_ind_partial_eval, univariate::lagrange_evals,
};

/// The number of bit variables a half-word (`*32`) shift variant acts over.
const HALF_WORD_LOG_BITS: usize = Word::LOG_BITS - 1;

/// The shift reduction's last sumcheck, over the bit index the h polynomials sum over.
///
/// The scalar phase 2 weights every shift key by is that sum, at the phase-1 challenges:
///
/// $$
/// h(r_j, r_s, r_v) = \sum_i \widetilde{L}(i) \cdot
///     \sum_{\text{op}} \widetilde{eq}(r_v, \text{op}) \cdot \text{ind}_{\text{op}}(i, r_j, r_s)
/// $$
///
/// Proving it as a sumcheck over the two multilinears in `i` leaves the verifier one evaluation
/// of each at a random point, which it computes directly with
/// [`evaluate_shift_inds`](binius_verifier::protocols::shift::evaluate_shift_inds) — no
/// summation over the bit index, and no partial evaluation.
pub struct ShiftIndSumcheck<P: PackedField, A: Allocator> {
	/// The Lagrange evaluations at the univariate challenge, over the bit index.
	l_tilde: FieldVec<P, A>,
	/// The shift indicators interpolated over the shift variant, over the bit index.
	shift_ind: FieldVec<P, A>,
	/// The claimed sum of their product: the h evaluation.
	h_eval: P::Scalar,
}

impl<F: BinaryField, P: PackedField<Scalar = F>, A: Allocator> ShiftIndSumcheck<P, A> {
	/// Builds the two multilinears the sumcheck runs over, and their claimed sum.
	///
	/// # Arguments
	///
	/// - `domain_subspace`: the univariate evaluation domain.
	/// - `r_zhat_prime`: the univariate challenge, shared by every operation.
	/// - `r_j`, `r_s`, `r_v`: phase 1's challenges — the input bit position, the shift amount and
	///   the shift variant.
	pub fn new(
		alloc: &A,
		domain_subspace: &BinarySubspace<F>,
		r_zhat_prime: F,
		r_j: &[F],
		r_s: &[F],
		r_v: &[F],
	) -> Self {
		let l_tilde = lagrange_evals(domain_subspace, r_zhat_prime);
		let l_tilde = l_tilde.as_ref();

		let shift_ind = build_shift_ind(r_j, r_s, r_v);
		let h_eval = inner_product(l_tilde.iter().copied(), shift_ind.iter().copied());

		Self {
			l_tilde: FieldBuffer::from_values_in(alloc, l_tilde),
			shift_ind: FieldBuffer::from_values_in(alloc, &shift_ind),
			h_eval,
		}
	}

	/// The h evaluation, which phase 2 weights every shift key by.
	pub const fn h_eval(&self) -> F {
		self.h_eval
	}

	/// Sends the h evaluation and proves it, in the [`Word::LOG_BITS`] rounds that bind the bit
	/// index.
	///
	/// The reduced evaluations the sumcheck ends at are dropped: the verifier evaluates both
	/// multilinears at the challenge point itself.
	pub fn prove(self, channel: &mut impl IPProverChannel<F>, alloc: &A) {
		channel.send_one(self.h_eval);
		let prover = bivariate_product_prover(alloc, [self.l_tilde, self.shift_ind], self.h_eval);
		prove_single(prover, channel);
	}
}

/// Builds the shift indicators over the bit index, interpolated over the shift variant, at the
/// phase-1 challenges: one scalar per bit index.
///
/// Phase 1 folded the variant axis into its own sumcheck, so the eight indicators contribute a
/// single multilinear here rather than one each.
fn build_shift_ind<F: BinaryField>(r_j: &[F], r_s: &[F], r_v: &[F]) -> Vec<F> {
	let (sigma, sigma_prime) = partial_eval_sigmas(r_j, r_s);
	let sigma_transpose = partial_eval_sigmas_transpose(r_j, r_s);
	let phi = partial_eval_phi(r_s);
	// The equality indicator selecting the sign position, the top input bit.
	let sign_position: F = r_j.iter().copied().product();

	// A half-word variant applies the same four rules to each 32-bit half, reading only the low
	// bits of the shift amount. The two halves never mix, so each indicator also carries the
	// equality of the halves the two bit indices fall in.
	let (sigma32, sigma32_prime) =
		partial_eval_sigmas(&r_j[..HALF_WORD_LOG_BITS], &r_s[..HALF_WORD_LOG_BITS]);
	let sigma32_transpose =
		partial_eval_sigmas_transpose(&r_j[..HALF_WORD_LOG_BITS], &r_s[..HALF_WORD_LOG_BITS]);
	let phi32 = partial_eval_phi(&r_s[..HALF_WORD_LOG_BITS]);
	let sign_position32: F = r_j[..HALF_WORD_LOG_BITS].iter().copied().product();
	let same_half = eq_ind_partial_eval::<F>(&r_j[HALF_WORD_LOG_BITS..]);

	let r_v_tensor = eq_ind_partial_eval::<F>(r_v);
	(0..Word::BITS)
		.map(|index| {
			let (half, low) = (index >> HALF_WORD_LOG_BITS, index % (1 << HALF_WORD_LOG_BITS));
			let same_half = same_half.as_ref()[half];
			// The eight indicators at this bit index, in `ShiftVariant` order.
			let shift_inds = [
				sigma_transpose[index],
				sigma[index],
				sigma[index] + sign_position * phi[index],
				sigma[index] + sigma_prime[index],
				same_half * sigma32_transpose[low],
				same_half * sigma32[low],
				same_half * (sigma32[low] + sign_position32 * phi32[low]),
				same_half * (sigma32[low] + sigma32_prime[low]),
			];
			inner_product(shift_inds, r_v_tensor.as_ref().iter().copied())
		})
		.collect()
}

/// Partial evaluation of the shift indicator helper polynomials $\sigma, \sigma'$ over all i on the
/// hypercube.
///
/// Given fixed j and s, computes sigma and sigma_prime for all possible i values.
/// Returns (sigma, sigma_prime) as Vecs of length `1 << r_j.len()`.
fn partial_eval_sigmas<E: FieldOps>(r_j: &[E], r_s: &[E]) -> (Vec<E>, Vec<E>) {
	assert_eq!(r_j.len(), r_s.len(), "r_j and r_s must have the same length");

	let n = r_j.len();
	let mut sigma = vec![E::zero(); 1 << n];
	let mut sigma_prime = vec![E::zero(); 1 << n];
	sigma[0] = E::one();

	// Process each bit position
	for k in 0..n {
		let j_k = r_j[k].clone();
		let s_k = r_s[k].clone();

		// Precompute boolean combinations for this bit
		let both = j_k.clone() * &s_k;
		let j_one_s = j_k.clone() - &both; // j_k * (1 - s_k)
		let one_j_s = s_k.clone() - &both; // (1 - j_k) * s_k
		let xor = j_k + s_k;
		let eq = E::one() + &xor;

		// Update arrays for this bit position
		for i in 0..(1 << k) {
			// Update upper halves first (i_k = 1)
			sigma[(1 << k) | i] = j_one_s.clone() * &sigma[i];
			sigma_prime[(1 << k) | i] = one_j_s.clone() * &sigma[i] + eq.clone() * &sigma_prime[i];

			// Update lower halves (i_k = 0)
			let sigma_i = sigma[i].clone();
			let sigma_prime_i = sigma_prime[i].clone();
			sigma[i] = eq.clone() * &sigma_i + j_one_s.clone() * &sigma_prime_i;
			sigma_prime[i] = sigma_prime_i * &one_j_s;
		}
	}

	(sigma, sigma_prime)
}

/// Partial evaluation of the shift indicator helper polynomial $\phi$ over all i on the hypercube.
///
/// Given fixed s, computes phi for all possible i values.
fn partial_eval_phi<E: FieldOps>(r_s: &[E]) -> Vec<E> {
	let n = r_s.len();
	let mut phi = vec![E::zero(); 1 << n];

	// Process each bit position
	for k in 0..n {
		let s_k = r_s[k].clone();

		// Update arrays for this bit position
		for i in 0..(1 << k) {
			// Update for i_k = 1
			phi[(1 << k) | i] = s_k.clone() + (E::one() + &s_k) * &phi[i];
			let temp = phi[(1 << k) | i].clone() - &s_k;
			phi[i] += &temp;
		}
	}

	phi
}

/// Partial evaluation of transposed sigma for SLL.
///
/// Since sll_ind(i, j, s) = srl_ind(j, i, s), this computes sigma with i and j swapped.
fn partial_eval_sigmas_transpose<E: FieldOps>(r_j: &[E], r_s: &[E]) -> Vec<E> {
	assert_eq!(r_j.len(), r_s.len(), "r_j and r_s must have the same length");

	let n = r_j.len();
	let mut sigma_transpose = vec![E::zero(); 1 << n];
	let mut sigma_transpose_prime = vec![E::zero(); 1 << n];
	sigma_transpose[0] = E::one();

	// Process each bit position
	for k in 0..n {
		let j_k = r_j[k].clone();
		let s_k = r_s[k].clone();

		// Precompute boolean combinations for this bit (with i and j swapped)
		let both = j_k.clone() * &s_k;
		let xor = j_k + s_k;
		let eq = E::one() + &xor;
		let zero = eq.clone() + &both;

		// Update arrays for this bit position
		for i in 0..(1 << k) {
			// Update for i_k = 1
			sigma_transpose[(1 << k) | i] =
				xor.clone() * &sigma_transpose[i] + zero.clone() * &sigma_transpose_prime[i];
			sigma_transpose_prime[(1 << k) | i] = both.clone() * &sigma_transpose_prime[i];

			// Update for i_k = 0
			let sigma_t = sigma_transpose[i].clone();
			sigma_transpose_prime[i] =
				both.clone() * &sigma_t + xor.clone() * &sigma_transpose_prime[i];
			sigma_transpose[i] = zero.clone() * &sigma_t;
		}
	}

	sigma_transpose
}

#[cfg(test)]
mod tests {
	use std::array;

	use binius_field::{BinaryField128bGhash as B128, Field};
	use binius_math::{
		multilinear::eq::eq_ind_partial_eval_scalars, test_utils::random_scalars,
		univariate::lagrange_evals_scalars,
	};
	use binius_verifier::protocols::shift::{LOG_SHIFT_VARIANT_COUNT, evaluate_shift_inds};
	use rand::{SeedableRng, rngs::StdRng};

	use super::*;

	// Ground truth for a shift-indicator MLE, independent of the recurrence under test.
	//
	// Fix j, s to the challenges r_j, r_s.
	//
	// Over the hypercube in i, the indicator's MLE expands over the (j, s) cube as:
	//     mle[i] = sum_{j, s in {0,1}^n : cond(i, j, s)} eq(r_j, j) * eq(r_s, s)
	fn reference_indicator(
		r_j: &[B128],
		r_s: &[B128],
		cond: impl Fn(usize, usize, usize) -> bool,
	) -> Vec<B128> {
		let n = r_j.len();
		// eq_j[j] = eq(r_j, j), eq_s[s] = eq(r_s, s).
		//
		// Both index little-endian, matching the recurrence's bit order.
		let eq_j = eq_ind_partial_eval_scalars(r_j);
		let eq_s = eq_ind_partial_eval_scalars(r_s);

		(0..1 << n)
			.map(|i| {
				let mut acc = B128::ZERO;
				for j in 0..1 << n {
					for s in 0..1 << n {
						if cond(i, j, s) {
							acc += eq_j[j] * eq_s[s];
						}
					}
				}
				acc
			})
			.collect()
	}

	// Draw a pseudo-random challenge (r_j, r_s).
	// The fixed seed keeps failures reproducible.
	fn challenges(n: usize) -> (Vec<B128>, Vec<B128>) {
		let mut rng = StdRng::seed_from_u64(0);
		(random_scalars(&mut rng, n), random_scalars(&mut rng, n))
	}

	#[test]
	fn srl_matches_reference() {
		// srl: output bit i reads input bit j = i + s.
		// Bits shifted past the top vanish, since no such j is in range.
		let (r_j, r_s) = challenges(6);
		let (sigma, _) = partial_eval_sigmas(&r_j, &r_s);
		assert_eq!(sigma, reference_indicator(&r_j, &r_s, |i, j, s| j == i + s));
	}

	#[test]
	fn sll_matches_reference() {
		// sll is the transpose of srl.
		// Output bit i = j + s reads input bit j.
		let (r_j, r_s) = challenges(6);
		let sigma_transpose = partial_eval_sigmas_transpose(&r_j, &r_s);
		assert_eq!(sigma_transpose, reference_indicator(&r_j, &r_s, |i, j, s| i == j + s));
	}

	#[test]
	fn sra_matches_reference() {
		// sra behaves like srl within range.
		// Past the shift, the sign bit j = 2^n - 1 fills every position.
		let (r_j, r_s) = challenges(6);
		let n = r_j.len();
		let (sigma, _) = partial_eval_sigmas(&r_j, &r_s);
		let phi = partial_eval_phi(&r_s);
		// prod(r_j) is the eq-indicator selecting the all-ones sign position j = 2^n - 1.
		let j_product: B128 = r_j.iter().copied().product();
		let sra: Vec<_> = (0..1 << n).map(|i| sigma[i] + j_product * phi[i]).collect();
		assert_eq!(sra, reference_indicator(&r_j, &r_s, |i, j, s| j == (i + s).min((1 << n) - 1)));
	}

	#[test]
	fn rotr_matches_reference() {
		// rotr wraps bits leaving the bottom back to the top.
		// So j = (i + s) mod 2^n.
		let (r_j, r_s) = challenges(6);
		let n = r_j.len();
		let (sigma, sigma_prime) = partial_eval_sigmas(&r_j, &r_s);
		let rotr: Vec<_> = (0..1 << n).map(|i| sigma[i] + sigma_prime[i]).collect();
		assert_eq!(rotr, reference_indicator(&r_j, &r_s, |i, j, s| j == (i + s) % (1 << n)));
	}

	/// The multilinear the prover sums over and the point evaluation the verifier checks it
	/// against must be the same polynomial: over the bit-index hypercube, the two agree.
	#[test]
	fn build_matches_the_verifier_point_evaluation() {
		let mut rng = StdRng::seed_from_u64(0);
		let r_j = random_scalars::<B128>(&mut rng, Word::LOG_BITS);
		let r_s = random_scalars::<B128>(&mut rng, Word::LOG_BITS);
		let r_v = random_scalars::<B128>(&mut rng, LOG_SHIFT_VARIANT_COUNT);

		let shift_ind = build_shift_ind(&r_j, &r_s, &r_v);
		let r_v_tensor = eq_ind_partial_eval_scalars(&r_v);

		for (index, &value) in shift_ind.iter().enumerate() {
			// The bit-index hypercube vertex at `index`.
			let r_i: [B128; Word::LOG_BITS] = array::from_fn(|bit| {
				if (index >> bit) & 1 == 1 {
					B128::ONE
				} else {
					B128::ZERO
				}
			});
			let expected =
				inner_product(evaluate_shift_inds(&r_i, &r_j, &r_s), r_v_tensor.iter().copied());
			assert_eq!(value, expected, "bit index {index}");
		}
	}

	/// The claimed sum is the h evaluation: the Lagrange weights contracted against the
	/// indicator multilinear.
	#[test]
	fn claimed_sum_is_the_weighted_indicator_sum() {
		use binius_compute::GlobalAllocator;
		use binius_field::{AESTowerField8b, PackedBinaryGhash2x128b, Random};

		type P = PackedBinaryGhash2x128b;

		let mut rng = StdRng::seed_from_u64(1);
		let r_zhat_prime = B128::random(&mut rng);
		let r_j = random_scalars::<B128>(&mut rng, Word::LOG_BITS);
		let r_s = random_scalars::<B128>(&mut rng, Word::LOG_BITS);
		let r_v = random_scalars::<B128>(&mut rng, LOG_SHIFT_VARIANT_COUNT);

		let subspace = BinarySubspace::<AESTowerField8b>::with_dim(Word::LOG_BITS).isomorphic();
		let sumcheck = ShiftIndSumcheck::<P, _>::new(
			&GlobalAllocator,
			&subspace,
			r_zhat_prime,
			&r_j,
			&r_s,
			&r_v,
		);

		let l_tilde = lagrange_evals_scalars(&subspace, &r_zhat_prime);
		let expected = inner_product(l_tilde, build_shift_ind(&r_j, &r_s, &r_v));
		assert_eq!(sumcheck.h_eval(), expected);
	}
}
