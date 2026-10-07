// Copyright 2023-2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use std::iter;

use binius_compute::Allocator;
use binius_core::word::Word;
use binius_field::{BinaryField, Field, PackedField, WideMul};
use binius_ip::sumcheck::RoundCoeffs;
use binius_ip_prover::sumcheck::{
	common::SumcheckProver, eq_tracker::ChunkedEqTracker, round_evals::RoundEvals,
	round_state::RoundState,
};
use binius_math::{FieldBuffer, FieldVec, multilinear::fold::fold_highest_var_inplace};
use binius_utils::rayon::prelude::*;
use itertools::izip;

use super::switchover::BinarySwitchover;

pub struct Claim<F: Field> {
	pub point: Vec<F>,
	pub value: F,
}

/// A [`SumcheckProver`] implementation that proves an mlecheck over many compositions of the
/// form `selected * selector + (1 - selector)`, where `selected` is the shared large field
/// multilinear and `selector` is one of the `Word::BITS` one-bit multilinears of a word list.
/// Unlike other multi mlecheck provers however the evaluation point is _not_ shared but is
/// specified per selector.
///
/// Selector `b` is the multilinear whose `i`-th value is bit `b` of word `i`. The bits stay packed
/// in words until `BinarySwitchover` folds them, which keeps the selectors' memory at the size of
/// `selected`. The need to expand the equality indicator for each selector still results in some
/// blowup.
pub struct SelectorMlecheckProver<'alloc, P: PackedField, A: Allocator> {
	last_coeffs_or_sums: RoundState<Vec<RoundCoeffs<P::Scalar>>, Vec<P::Scalar>>,
	selected: FieldVec<P, A>,
	eq_trackers: Vec<ChunkedEqTracker<P>>,
	weights: Vec<P::Scalar>,
	switchover: BinarySwitchover<'alloc, P, A>,
}

impl<'alloc, F: BinaryField, P: PackedField<Scalar = F>, A: Allocator>
	SelectorMlecheckProver<'alloc, P, A>
{
	/// Constructs a prover, given `selected` being the shared large field multilinear, individual
	/// `claims` per selector, `words` whose bits are the selectors (missing words read as zero),
	/// and `weights` to combine the per-selector round polynomials into one (one weight per claim).
	///
	/// The prover exposes a single claim — the `weights`-combination `Σ_i weights[i] · C_i` of the
	/// per-selector claims. Supplying the equality-indicator tensor `eq_k(γ, ·)` as the weights
	/// batches the claims with `eq_k(γ, i)`.
	pub fn new(
		alloc: &'alloc A,
		selected: FieldVec<P, A>,
		claims: Vec<Claim<F>>,
		words: &[Word],
		weights: Vec<F>,
	) -> Self {
		let n_vars = selected.log_len();

		assert_eq!(claims.len(), Word::BITS, "one claim per selector");

		assert!(
			claims.iter().all(|claim| claim.point.len() == n_vars),
			"multilinears must have equal number of variables"
		);

		assert_eq!(
			weights.len(),
			claims.len(),
			"number of weights must match the number of claims"
		);

		assert!(
			words.len() <= selected.len(),
			"words must not outnumber the selected multilinear's values"
		);

		const MAX_CHUNK_VARS: usize = 8;
		let (eq_trackers, sums) = claims
			.into_par_iter()
			.map(|Claim { point, value }| (ChunkedEqTracker::new(MAX_CHUNK_VARS, &point), value))
			.collect::<(Vec<_>, Vec<_>)>();

		let switchover = BinarySwitchover::new(alloc, words, n_vars);
		let last_coeffs_or_sums = RoundState::Claim(sums);

		Self {
			last_coeffs_or_sums,
			selected,
			eq_trackers,
			weights,
			switchover,
		}
	}
}

impl<F, P, A> SumcheckProver<F> for SelectorMlecheckProver<'_, P, A>
where
	F: BinaryField,
	P: PackedField<Scalar = F>,
	A: Allocator,
{
	fn n_vars(&self) -> usize {
		self.selected.log_len()
	}

	fn execute(&mut self) -> Vec<RoundCoeffs<F>> {
		let sums = self.last_coeffs_or_sums.claim();

		assert!(self.n_vars() > 0);

		// Perform chunked summation: for every row, evaluate all compositions and add up
		// results to an array of round evals accumulators. Alternative would be to sum each
		// composition on its own pass, but that would require reading the entirety of eq field
		// buffer on each pass, which will evict the latter from the cache. By doing chunked
		// compute, we reasonably hope that eq chunk always stays in L1 cache. We can also
		// leverage the outer product representation of the eq indicator.
		//
		// We also do switchover there, which by definition requires small scratchpads to hold
		// large field partial evaluations of the transparent multilinears.
		let chunk_vars = self
			.eq_trackers
			.first()
			.map(|eq_tracker| eq_tracker.chunk().log_len())
			.unwrap_or_default();
		let chunk_count = 1 << (self.n_vars() - 1 - chunk_vars);

		// The fold below reads both halves concurrently from many rayon tasks.
		// Borrowed halves cross that boundary for any backing store the buffer is built on.
		let (selected_0, selected_1) = self.selected.split_half();
		let selectors = self.switchover.selectors();
		let is_binary = selectors.is_binary();

		let packed_prime_evals = (0..chunk_count)
			.into_par_iter()
			.fold(
				|| {
					let binary_chunk = FieldBuffer::<P>::zeros(chunk_vars);
					let masks = [(); 2].map(|_| {
						(0..binary_chunk.as_ref().len())
							.map(|_| P::make_mask(iter::empty()))
							.collect::<Vec<_>>()
					});
					(
						vec![RoundEvals::<P, 2>::default(); sums.len()],
						binary_chunk.clone(),
						binary_chunk,
						masks,
					)
				},
				|(mut packed_prime_evals, mut binary_chunk_0, mut binary_chunk_1, mut masks),
				 chunk_index| {
					let selected_0_chunk = selected_0.chunk(chunk_vars, chunk_index);
					let selected_1_chunk = selected_1.chunk(chunk_vars, chunk_index);

					for (selector, (round_evals, eq_tracker)) in
						izip!(&mut packed_prime_evals, &self.eq_trackers).enumerate()
					{
						let eq_chunk = eq_tracker.chunk();
						let eq_suffix_eval = eq_tracker.suffix().get(chunk_index);

						// Accumulate `eq_i * composition` in unreduced (wide) form and reduce once
						// at the end of the chunk. Only the final multiply by `eq_i` is widened;
						// the `composition` product is reduced as usual because it feeds into that
						// widening multiply.
						let mut wide_y_1 = <P as WideMul>::Output::default();
						let mut wide_y_inf = <P as WideMul>::Output::default();
						let mut accumulate = |eq_i, y_1_prod, y_inf_prod| {
							wide_y_1 += P::wide_mul(eq_i, y_1_prod);
							wide_y_inf += P::wide_mul(eq_i, y_inf_prod);
						};
						if is_binary {
							// Every selector value is a bit, so the composition products are
							// lane selects.
							let [masks_1, masks_inf] = &mut masks;
							selectors.fill_bit_masks(
								selector,
								chunk_vars,
								chunk_index,
								[masks_1, masks_inf],
							);
							for (&eq_i, &selected_0_i, &selected_1_i, mask_1, mask_inf) in izip!(
								eq_chunk.as_ref(),
								selected_0_chunk.as_ref(),
								selected_1_chunk.as_ref(),
								&masks[0],
								&masks[1],
							) {
								// @one: selected if the selector is set, else 1
								// @inf: selected_inf if the selector halves differ, else 0
								let y_1_prod = (selected_1_i + P::one()).select(mask_1) + P::one();
								let y_inf_prod = (selected_0_i + selected_1_i).select(mask_inf);
								accumulate(eq_i, y_1_prod, y_inf_prod);
							}
						} else {
							selectors.fill_halves(
								selector,
								chunk_vars,
								chunk_index,
								[binary_chunk_0.as_mut_view(), binary_chunk_1.as_mut_view()],
							);
							for (
								&eq_i,
								&selected_0_i,
								&selected_1_i,
								&selector_0_i,
								&selector_1_i,
							) in izip!(
								eq_chunk.as_ref(),
								selected_0_chunk.as_ref(),
								selected_1_chunk.as_ref(),
								binary_chunk_0.as_ref(),
								binary_chunk_1.as_ref(),
							) {
								let selected_inf_i = selected_0_i + selected_1_i;
								let selector_inf_i = selector_0_i + selector_1_i;

								// selected * selector + (1 - selector)
								// @one: selector * (selected - 1) + 1
								// @inf: selector * selected (note that lower degree terms are
								// dropped)
								let y_1_prod = selector_1_i * (selected_1_i - P::one()) + P::one();
								let y_inf_prod = selector_inf_i * selected_inf_i;
								accumulate(eq_i, y_1_prod, y_inf_prod);
							}
						}
						let chunk_round_evals = RoundEvals([wide_y_1, wide_y_inf]).reduce::<P>();

						// Apply the common factor from the outer product representation of the eq
						// ind
						*round_evals += &(chunk_round_evals * eq_suffix_eval);
					}

					(packed_prime_evals, binary_chunk_0, binary_chunk_1, masks)
				},
			)
			.map(|(evals, ..)| evals)
			// A merge seeded with a partial that already exists never touches a buffer of zeros.
			// An identity would allocate and zero one accumulator per merge, then add all of it.
			.reduce_with(|lhs, rhs| izip!(lhs, rhs).map(|(l, r)| l + &r).collect())
			// An empty hypercube yields no partials at all, and its round evals are zero.
			.unwrap_or_else(|| vec![RoundEvals::<P, 2>::default(); sums.len()]);

		// This prover has multiple evaluation points and cannot implement MleCheckProver.
		let (prime_coeffs, round_coeffs) = izip!(&self.eq_trackers, sums, packed_prime_evals)
			.map(|(eq_tracker, &sum, packed_prime_evals)| {
				eq_tracker.interpolate2(sum, packed_prime_evals.sum_scalars(self.n_vars() - 1))
			})
			.unzip::<_, _, Vec<_>, Vec<_>>();

		self.last_coeffs_or_sums = RoundState::Coeffs(prime_coeffs);

		// Combine the per-claim round polynomials into the single weighted round polynomial
		// `Σ_i weights[i] · R_i`.
		let combined = izip!(round_coeffs, &self.weights)
			.map(|(coeffs, &w)| coeffs * w)
			.sum();
		vec![combined]
	}

	fn fold(&mut self, challenge: F) {
		let prime_coeffs = self.last_coeffs_or_sums.coeffs();

		assert!(self.n_vars() > 0);

		let sums = prime_coeffs
			.iter()
			.map(|coeffs| coeffs.evaluate(&challenge))
			.collect();

		self.eq_trackers
			.par_iter_mut()
			.for_each(|eq_tracker| eq_tracker.fold(challenge));

		self.switchover.fold(challenge);
		fold_highest_var_inplace(&mut self.selected, challenge);

		self.last_coeffs_or_sums = RoundState::Claim(sums);
	}

	fn finish(self) -> Vec<F> {
		assert_eq!(self.n_vars(), 0, "finish called out of order; sumcheck rounds remain");

		let mut multilinear_evals = self.switchover.finish().to_vec();
		multilinear_evals.push(self.selected.get(0));
		multilinear_evals
	}
}

#[cfg(test)]
mod tests {
	use std::iter::repeat_with;

	use binius_compute::GlobalAllocator;
	use binius_field::FieldOps;
	use binius_ip::sumcheck::verify;
	use binius_ip_prover::sumcheck::prove_single;
	use binius_math::{
		multilinear::{eq::eq_ind, evaluate::evaluate},
		test_utils::{Packed128b, random_scalars},
	};
	use binius_transcript::{ProverTranscript, fiat_shamir::HasherChallenger};
	use itertools::Itertools;
	use rand::prelude::*;
	use rstest::rstest;

	use super::*;

	type P = Packed128b;
	type F = <P as FieldOps>::Scalar;
	type StdChallenger = HasherChallenger<sha2::Sha256>;

	// Prove/verify roundtrip: drive the prover through a transcript, verify with the generic
	// sumcheck verifier, and reconstruct the reduced claim from the returned multilinear
	// evaluations. This mirrors the verifier's selector-sumcheck check (`verify_phase_3` in
	// `binius-verifier`'s intmul protocol), which recombines per-selector terms
	// `(selector·(selected − 1) + 1)·eq(point_i, r)` weighted by an equality tensor.
	//
	// The word list stops short of `2^n_vars`, so the missing words read as zero.
	#[rstest]
	#[case::padded_single_block(3)]
	#[case::exactly_one_block(6)]
	#[case::several_blocks(9)]
	fn test_selector_mlecheck_prove_verify(#[case] n_vars: usize) {
		let mut rng = StdRng::seed_from_u64(0);

		let selector_count = Word::BITS;

		let words = repeat_with(|| Word::from_u64(rng.random()))
			.take((1 << n_vars) - 3)
			.collect_vec();

		let selected_scalars = random_scalars::<F>(&mut rng, 1 << n_vars);
		let selected = FieldBuffer::<P>::from_values(&selected_scalars);

		// The 1-bit selector columns, extracted from the words.
		let selector_columns = (0..selector_count)
			.map(|b| {
				(0..1 << n_vars)
					.map(|i| {
						let bit = words.get(i).is_some_and(|word| word.extract_bit(b));
						if bit { F::ONE } else { F::ZERO }
					})
					.collect_vec()
			})
			.collect_vec();

		// One claim per selector: the composition `selected * selector + (1 - selector)` evaluated
		// at an independent random point.
		let points = repeat_with(|| random_scalars::<F>(&mut rng, n_vars))
			.take(selector_count)
			.collect_vec();
		let claims = izip!(&selector_columns, &points)
			.map(|(selector_scalars, point)| {
				let masked = izip!(&selected_scalars, selector_scalars)
					.map(|(&selected, &selector)| selected * selector + (F::ONE - selector))
					.collect_vec();
				let value = evaluate(&FieldBuffer::<P>::from_values(&masked), point);
				Claim {
					point: point.clone(),
					value,
				}
			})
			.collect_vec();

		let weights = random_scalars::<F>(&mut rng, selector_count);

		// The prover reduces the per-selector claims to a single weighted sumcheck claim.
		let claim: F = izip!(&claims, &weights).map(|(c, &w)| c.value * w).sum();

		let prover = SelectorMlecheckProver::new(
			&GlobalAllocator,
			FieldBuffer::from_view_in(&GlobalAllocator, selected.as_view()),
			claims,
			&words,
			weights.clone(),
		);

		// Run the prover through the transcript and append the final multilinear evaluations.
		let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
		let output = prove_single(prover, &mut prover_transcript);
		prover_transcript
			.message()
			.write_slice(&output.multilinear_evals);

		// Verify against the generic sumcheck verifier. The composition has degree 3: a degree-2
		// product (`selected * selector`) times the equality indicator.
		let mut verifier_transcript = prover_transcript.into_verifier();
		let sumcheck_output = verify(n_vars, 3, claim, &mut verifier_transcript).unwrap();

		assert_eq!(
			output.challenges, sumcheck_output.challenges,
			"prover and verifier challenges must match"
		);

		// The prover binds variables high-to-low; `evaluate` and `eq_ind` expect low-to-high.
		let mut reduced_point = sumcheck_output.challenges.clone();
		reduced_point.reverse();

		// `finish()` returns `[selector_0(r), .., selector_{k-1}(r), selected(r)]`.
		let multilinear_evals: Vec<F> = verifier_transcript
			.message()
			.read_vec(selector_count + 1)
			.unwrap();
		let (selector_evals, selected_eval) = multilinear_evals.split_at(selector_count);
		let selected_eval = selected_eval[0];

		// The claimed evaluations must match direct evaluation of the multilinears at the challenge
		// point.
		assert_eq!(selected_eval, evaluate(&selected, &reduced_point), "selected evaluation");
		for (i, (&selector_eval, selector_scalars)) in
			izip!(selector_evals, &selector_columns).enumerate()
		{
			assert_eq!(
				selector_eval,
				evaluate(&FieldBuffer::<P>::from_values(selector_scalars), &reduced_point),
				"selector {i} evaluation"
			);
		}

		// Reconstruct the reduced sumcheck claim from the multilinear evaluations:
		// `Σ_i weights[i] · (selected(r)·selector_i(r) + (1 − selector_i(r))) · eq(point_i, r)`.
		let expected_eval: F = izip!(selector_evals, &points, &weights)
			.map(|(&selector_eval, point, &weight)| {
				let composition = selected_eval * selector_eval + (F::ONE - selector_eval);
				weight * composition * eq_ind(point, &reduced_point)
			})
			.sum();
		assert_eq!(
			expected_eval, sumcheck_output.eval,
			"reduced sumcheck claim must match the composition evaluated at the challenge point"
		);
	}
}
