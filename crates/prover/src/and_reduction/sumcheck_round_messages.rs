// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use std::{borrow::Cow, iter};

use binius_core::word::Word;
use binius_field::{AESTowerField8b, BinaryField, PackedField};
use binius_math::{FieldBuffer, multilinear::eq::eq_ind_partial_eval};
use binius_utils::rayon::prelude::*;
use binius_verifier::{
	config::PROVER_SMALL_FIELD_ZEROCHECK_CHALLENGES, protocols::bitand::ROWS_PER_HYPERCUBE_VERTEX,
};
use itertools::izip;

use super::ntt_lookup::NTTLookup;

/// Generates a univariate polynomial for the sumcheck protocol in AND constraint reduction.
///
/// Let our oblong polynomials be A(Z, X₀, ...), B(Z, X₀, ...), and C(Z, X₀, ...)
///
/// Let our zerocheck challenges be (r₀, ...)
///
/// It turns out that the first k zerocheck challenges can actually be deterministic, since our
/// polynomials have 1-bit coefficients as long as their tensor product expansion is an
/// F2-linearly-independent set.
///
/// Note: Deterministic here means that the first k zerocheck challenges are a compile-time
/// agreed-upon parameter to the proof, and not sampled randomly by the verifier
///
///
/// We choose k=3 because we want them to be in a field isomorphic to the 8-bit NTT domain field
///
/// Computes a univariate polynomial:
/// R₀(Z) = ∑_{X₀,...,Xₙ₋₁ ∈ {0,1}} (A·B - C)·eq(X₀,...,Xₙ₋₁; r₀,...,rₙ₋₁)
///
/// This is zero at every point on the hypercube IFF A*B-C evaluates to zero at (r₀,...,rₙ₋₁)
/// for every Z on the univariate domain. Since R₀(Z) is 0 on the univariate domain, the prover
/// sends only enough values such that the verifier learns a domain of evaluations of size >
/// deg(R₀(Z))
///
/// # Arguments
///
/// * `log_words` - Base-2 logarithm of the number of words in each column
/// * `a_words` - First multiplicand (a) as a one-bit oblong multilinear polynomial
/// * `b_words` - Second multiplicand (b) as a one-bit oblong multilinear polynomial
/// * `c_words` - Product constraint (c) as a one-bit oblong multilinear polynomial
/// * `eq_ind_big_field_challenges` - Partial equality indicator evaluations for big field variables
/// * `ntt_lookup` - NTT lookup tables for efficient low-degree extension of each column
///
/// # Returns
///
/// The evaluations of R₀(Z), a univariate polynomial of degree at most 2*(|D| - 1) where |D| is the
/// domain size, on another, disjoint |D|-sized domain. This allows the verifier to construct R₀(Z),
/// since it must equal zero on D.
///
/// # Type Parameters
///
/// * `F` - The challenge field type (must be a binary field)
/// * `PNTTDomain` - The packed extension field type for NTT operations (width must equal
///   `ROWS_PER_HYPERCUBE_VERTEX`)
pub fn univariate_round_message_extension_domain<F, PNTTDomain>(
	log_words: usize,
	a_words: &[Word],
	b_words: &[Word],
	c_words: &[Word],
	eq_ind_big_field_challenges: &FieldBuffer<F>,
	ntt_lookup: &NTTLookup<PNTTDomain>,
) -> [F; ROWS_PER_HYPERCUBE_VERTEX]
where
	F: BinaryField + From<AESTowerField8b>,
	PNTTDomain: PackedField<Scalar = AESTowerField8b>,
{
	// This assertion is used as a workaround for Rust's limited support for const-generics,
	// ideally we would just use PNTTDomain::WIDTH everywhere instead, but since this function only
	// supports 128b underliers, this is set to a constant. We would need to rethink for support of
	// multiple underlier sizes
	assert_eq!(PNTTDomain::WIDTH, ROWS_PER_HYPERCUBE_VERTEX);

	assert_eq!(
		eq_ind_big_field_challenges.log_len(),
		log_words.saturating_sub(PROVER_SMALL_FIELD_ZEROCHECK_CHALLENGES.len())
	);
	for col in [a_words, b_words, c_words] {
		assert_eq!(col.len(), 1 << log_words);
	}

	let eq_ind_small: [_; 8] =
		eq_ind_partial_eval::<AESTowerField8b>(&PROVER_SMALL_FIELD_ZEROCHECK_CHALLENGES)
			.iter_scalars()
			.map(PNTTDomain::broadcast)
			.collect::<Vec<_>>()
			.try_into()
			.expect("PROVER_SMALL_FIELD_ZEROCHECK_CHALLENGES.len() == 3; 2^3 == 8");

	// Process columns in fixed-length chunks of 8 to assist compiler in loop unrolling.
	let a_col_chunks = duplicate_to_fixed_chunks::<8>(a_words);
	let b_col_chunks = duplicate_to_fixed_chunks::<8>(b_words);
	let c_col_chunks = duplicate_to_fixed_chunks::<8>(c_words);

	// Accumulate resulting polynomial evals by iterating over each hypercube vertex.
	(a_col_chunks.as_ref(), b_col_chunks.as_ref(), c_col_chunks.as_ref())
		.into_par_iter()
		.map(|(a_chunk, b_chunk, c_chunk)| {
			let mut summed_ntt = PNTTDomain::zero();

			for (a_i, b_i, c_i, &weight) in
				izip!(a_chunk, b_chunk, c_chunk, eq_ind_small.each_ref())
			{
				// Compute the low-degree extension of each column via the lookup table.
				let [first_col_ntt, second_col_ntt, third_col_ntt] =
					ntt_lookup.multi_ntt_array([a_i.0, b_i.0, c_i.0]);

				// Compute the weighted composition of the LDE values.
				summed_ntt += (first_col_ntt * second_col_ntt - third_col_ntt) * weight;
			}

			summed_ntt
		})
		.zip(eq_ind_big_field_challenges.as_ref())
		.fold_with([F::ZERO; ROWS_PER_HYPERCUBE_VERTEX], |mut acc, (summed_ntt, &eq_weight)| {
			for (acc_i, summed_ntt_i) in iter::zip(&mut acc, summed_ntt.into_iter()) {
				*acc_i += eq_weight * F::from(summed_ntt_i);
			}
			acc
		})
		.reduce(
			|| [F::ZERO; ROWS_PER_HYPERCUBE_VERTEX],
			|mut lhs, rhs| {
				for (lhs_i, rhs_i) in iter::zip(&mut lhs, rhs) {
					*lhs_i += rhs_i;
				}
				lhs
			},
		)
}

/// View the words as a slice of fixed-length arrays.
///
/// If the number of words is less than N, then repeat it into an N-length array. Repeating
/// corresponds to variable padding over the boolean hypercube.
///
/// ## Preconditions
///
/// * `words` must be a power of two
/// * `N` must be a power of two
fn duplicate_to_fixed_chunks<const N: usize>(words: &[Word]) -> Cow<'_, [[Word; N]]> {
	assert!(words.len().is_power_of_two());
	assert!(N.is_power_of_two());

	let (chunks, leftover) = words.as_chunks::<N>();

	assert!(
		chunks.is_empty() || leftover.is_empty(),
		"words.len() and N are both powers of two; either words.len() is divisible by N or less than it"
	);

	if chunks.is_empty() {
		let mut repeated = [Word::ZERO; N];
		for chunk in repeated.chunks_mut(words.len()) {
			chunk.copy_from_slice(words);
		}
		Cow::Owned(vec![repeated])
	} else {
		Cow::Borrowed(chunks)
	}
}

#[cfg(test)]
mod test {
	use std::{iter, iter::repeat_with};

	use binius_core::word::Word;
	use binius_field::{
		AESTowerField8b, Field, PackedAESBinaryField64x8b, Random,
		linear_transformation::{
			BytewiseLookupTransformationFactory, LinearTransformationFactory,
			OutputWrappingTransformationFactory,
		},
	};
	use binius_math::{
		BinarySubspace, FieldBuffer,
		multilinear::eq::eq_ind_partial_eval,
		univariate::{extrapolate_over_subspace, lagrange_evals_scalars},
	};
	use binius_verifier::{
		config::B128,
		protocols::bitand::{ROWS_PER_HYPERCUBE_VERTEX, SKIPPED_VARS},
	};
	use itertools::izip;
	use rand::prelude::*;

	use super::univariate_round_message_extension_domain;
	use crate::{and_reduction::ntt_lookup::NTTLookup, fold_word::fold_words_with_transform};

	fn random_words(log_num_words: usize, mut rng: impl Rng) -> Vec<Word> {
		repeat_with(|| Word(rng.random()))
			.take(1 << log_num_words)
			.collect()
	}

	// Sends the sum claim from first multilinear round (second overall round)
	pub fn sum_claim<BF: Field + From<B128>>(
		first_col: &FieldBuffer<BF>,
		second_col: &FieldBuffer<BF>,
		third_col: &FieldBuffer<BF>,
		eq_ind: &FieldBuffer<BF>,
	) -> BF {
		izip!(first_col.as_ref(), second_col.as_ref(), third_col.as_ref(), eq_ind.as_ref())
			.map(|(a, b, c, eq)| (*a * *b - *c) * *eq)
			.sum()
	}

	#[test]
	fn test_first_round_message_matches_next_round_sum_claim() {
		// Setup
		let log_num_rows = 10;
		let mut rng = StdRng::from_seed([0; 32]);

		let small_field_zerocheck_challenges = [
			AESTowerField8b::new(2),
			AESTowerField8b::new(4),
			AESTowerField8b::new(16),
		];

		let big_field_zerocheck_challenges =
			vec![
				B128::random(&mut rng);
				log_num_rows - SKIPPED_VARS - small_field_zerocheck_challenges.len()
			];

		let log_num_words = log_num_rows - SKIPPED_VARS;
		let mlv_1 = random_words(log_num_words, &mut rng);
		let mlv_2 = random_words(log_num_words, &mut rng);
		let mlv_3: Vec<Word> = iter::zip(&mlv_1, &mlv_2).map(|(&a, &b)| a & b).collect();

		let eq_ind_only_big = eq_ind_partial_eval(&big_field_zerocheck_challenges);

		// Agreed-upon proof parameter

		let prover_message_domain = BinarySubspace::with_dim(SKIPPED_VARS + 1);
		let ntt_lookup = NTTLookup::<PackedAESBinaryField64x8b>::new(&prover_message_domain);

		let verifier_message_domain = prover_message_domain.isomorphic::<B128>();

		// Prover generates first round message
		let first_round_message_on_ext_domain = univariate_round_message_extension_domain::<B128, _>(
			log_num_words,
			&mlv_1,
			&mlv_2,
			&mlv_3,
			&eq_ind_only_big,
			&ntt_lookup,
		);

		let mut first_round_message_coeffs = vec![B128::ZERO; 2 * ROWS_PER_HYPERCUBE_VERTEX];

		first_round_message_coeffs[ROWS_PER_HYPERCUBE_VERTEX..2 * ROWS_PER_HYPERCUBE_VERTEX]
			.copy_from_slice(&first_round_message_on_ext_domain);

		// Verifier checks the accuracy of the message by challenging the prover and folding
		// polynomials transparently

		let verifier_input_domain: BinarySubspace<B128> =
			verifier_message_domain.reduce_dim(verifier_message_domain.dim() - 1);

		let first_sumcheck_challenge = B128::random(&mut rng);
		let expected_next_round_sum = extrapolate_over_subspace(
			&verifier_message_domain,
			&first_round_message_coeffs,
			first_sumcheck_challenge,
		);

		let lagrange_evals =
			lagrange_evals_scalars(&verifier_input_domain, first_sumcheck_challenge);
		let transform =
			OutputWrappingTransformationFactory::new(BytewiseLookupTransformationFactory)
				.create(&lagrange_evals);

		let folded_first_mle: FieldBuffer<B128> = fold_words_with_transform(&transform, &mlv_1);
		let folded_second_mle: FieldBuffer<B128> = fold_words_with_transform(&transform, &mlv_2);
		let folded_third_mle: FieldBuffer<B128> = fold_words_with_transform(&transform, &mlv_3);

		let upcasted_small_field_challenges: Vec<_> = small_field_zerocheck_challenges
			.into_iter()
			.map(B128::from)
			.collect();

		let verifier_field_zerocheck_challenges: Vec<_> = upcasted_small_field_challenges
			.iter()
			.chain(big_field_zerocheck_challenges.iter())
			.copied()
			.collect();

		let verifier_field_eq = eq_ind_partial_eval(&verifier_field_zerocheck_challenges);
		let actual_next_round_sum =
			sum_claim(&folded_first_mle, &folded_second_mle, &folded_third_mle, &verifier_field_eq);

		assert_eq!(expected_next_round_sum, actual_next_round_sum);
	}
}
