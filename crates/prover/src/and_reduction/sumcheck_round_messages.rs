// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use std::{array, borrow::Cow, iter};

use binius_core::word::Word;
use binius_field::{
	AESTowerField8b as B8, BinaryField, BinaryField1b as B1, ExtensionField, PackedField, WideMul,
	util::expand_subset_sums_array,
};
use binius_math::multilinear::eq::eq_ind_partial_eval;
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
	big_field_challenges: &[F],
	ntt_lookup: &NTTLookup<PNTTDomain>,
) -> [F; ROWS_PER_HYPERCUBE_VERTEX]
where
	F: BinaryField + From<B8>,
	PNTTDomain: PackedField<Scalar = B8>,
{
	// This assertion is used as a workaround for Rust's limited support for const-generics,
	// ideally we would just use PNTTDomain::WIDTH everywhere instead, but since this function only
	// supports 128b underliers, this is set to a constant. We would need to rethink for support of
	// multiple underlier sizes
	assert_eq!(PNTTDomain::WIDTH, ROWS_PER_HYPERCUBE_VERTEX);

	const N_FIXED_SMALL_CHALLENGES: usize = PROVER_SMALL_FIELD_ZEROCHECK_CHALLENGES.len();
	const N_FIXED_LARGE_CHALLENGES: usize = 4;

	const LOG_CHUNK_SIZE: usize = N_FIXED_SMALL_CHALLENGES + N_FIXED_LARGE_CHALLENGES;

	assert_eq!(big_field_challenges.len(), log_words.saturating_sub(N_FIXED_SMALL_CHALLENGES));
	for col in [a_words, b_words, c_words] {
		assert_eq!(col.len(), 1 << log_words);
	}

	let eq_ind_small: [_; 1 << N_FIXED_SMALL_CHALLENGES] =
		eq_ind_partial_eval::<B8>(&PROVER_SMALL_FIELD_ZEROCHECK_CHALLENGES)
			.iter_scalars()
			.map(PNTTDomain::broadcast)
			.collect::<Vec<_>>()
			.try_into()
			.expect("PROVER_SMALL_FIELD_ZEROCHECK_CHALLENGES.len() == N_FIXED_SMALL_CHALLENGES");

	// We don't actually use fixed large challenges yet, so just take a prefix of the big field
	// challenges passed in.
	//
	// TODO: Use some fixed challenges instead throughout the protocol.
	let (fixed_large_challenges, extra_challenges) = if big_field_challenges.len()
		< N_FIXED_LARGE_CHALLENGES
	{
		let mut fixed_large_challenges = [F::ZERO; N_FIXED_LARGE_CHALLENGES];
		fixed_large_challenges[..big_field_challenges.len()].copy_from_slice(big_field_challenges);
		(fixed_large_challenges, &[][..])
	} else {
		let (fixed_large_challenges, extra_challenges) =
			big_field_challenges.split_at(N_FIXED_LARGE_CHALLENGES);
		let fixed_large_challenges: [_; N_FIXED_LARGE_CHALLENGES] = fixed_large_challenges
			.try_into()
			.expect("big_field_challenges.len() >= N_FIXED_LARGE_CHALLENGES");
		(fixed_large_challenges, extra_challenges)
	};

	let eq_ind_fixed_large: [_; 1 << N_FIXED_LARGE_CHALLENGES] =
		eq_ind_partial_eval::<F>(&fixed_large_challenges)
			.as_ref()
			.try_into()
			.expect("fixed_large_challenges.len() == N_FIXED_LARGE_CHALLENGES");

	let outer_weight_mul_maps = eq_ind_fixed_large.map(B8ToExtMulMap::new);

	let eq_ind_extra = eq_ind_partial_eval::<F>(extra_challenges);

	// Process columns in fixed-length chunks of 8 to assist compiler in loop unrolling.
	let a_col_chunks = duplicate_to_fixed_chunks::<{ 1 << LOG_CHUNK_SIZE }>(a_words);
	let b_col_chunks = duplicate_to_fixed_chunks::<{ 1 << LOG_CHUNK_SIZE }>(b_words);
	let c_col_chunks = duplicate_to_fixed_chunks::<{ 1 << LOG_CHUNK_SIZE }>(c_words);

	// Accumulate resulting polynomial evals by iterating over each hypercube vertex.
	(a_col_chunks.as_ref(), b_col_chunks.as_ref(), c_col_chunks.as_ref())
		.into_par_iter()
		.map(|(a_chunk, b_chunk, c_chunk)| {
			// Reshape the chunk arrays into arrays of arrays
			let [a_subchunks, b_subchunks, c_subchunks] =
				[a_chunk, b_chunk, c_chunk].map(|chunk| {
					bytemuck::must_cast_ref::<
						[Word; 1 << LOG_CHUNK_SIZE],
						[[Word; 1 << N_FIXED_SMALL_CHALLENGES]; 1 << N_FIXED_LARGE_CHALLENGES],
					>(chunk)
				});

			let mut acc = [F::ZERO; ROWS_PER_HYPERCUBE_VERTEX];
			for (a_subchunk, b_subchunk, c_subchunk, outer_weight) in
				izip!(a_subchunks, b_subchunks, c_subchunks, &outer_weight_mul_maps)
			{
				let mut summed_ntt = <PNTTDomain as WideMul>::Output::default();
				for (a_i, b_i, c_i, inner_weight) in
					izip!(a_subchunk, b_subchunk, c_subchunk, &eq_ind_small)
				{
					// Compute the low-degree extension of each column via the lookup table.
					let [first_col_ntt, second_col_ntt, third_col_ntt] =
						ntt_lookup.multi_ntt_array([a_i.0, b_i.0, c_i.0]);

					// Compute the weighted composition of the LDE values.
					summed_ntt += PNTTDomain::wide_mul(
						first_col_ntt * second_col_ntt - third_col_ntt,
						*inner_weight,
					);
				}

				let summed_ntt_reduced = PNTTDomain::reduce(summed_ntt);
				for (acc_i, summed_ntt_i) in iter::zip(&mut acc, summed_ntt_reduced.into_iter()) {
					*acc_i += outer_weight.call(summed_ntt_i);
				}
			}
			acc
		})
		.zip(eq_ind_extra.as_ref())
		.map(|(mut acc, eq_weight)| {
			for acc_i in &mut acc {
				*acc_i *= eq_weight;
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

/// Represents a precomputed multiplication map by an extension field constant for
/// [`B8`].`
///
/// Multiplication by a constant for a binary field is an $\mathbb{F}_2$-linear transform. For small
/// inputs, such as $\mathbb{F}_{2^8}$ elements, this can be represented by a small lookup table.
struct B8ToExtMulMap<F> {
	lookup: [F; 256],
}

impl<F: BinaryField + From<B8>> B8ToExtMulMap<F> {
	fn new(val: F) -> Self {
		let basis_images: [F; 8] = array::from_fn(|i| {
			let basis = <B8 as ExtensionField<B1>>::basis(i);
			F::from(basis) * val
		});
		Self {
			lookup: expand_subset_sums_array(basis_images),
		}
	}

	#[inline]
	fn call(&self, input: B8) -> F {
		self.lookup[input.val() as usize]
	}
}

#[cfg(test)]
mod test {
	use std::iter::repeat_with;

	use binius_field::{
		BinaryField128bGhash as B128, Field, PackedAESBinaryField64x8b, Random,
		linear_transformation::{
			BytewiseLookupTransformationFactory, LinearTransformationFactory,
			OutputWrappingTransformationFactory,
		},
	};
	use binius_math::{
		BinarySubspace, FieldBuffer,
		univariate::{extrapolate_over_subspace, lagrange_evals_scalars},
	};
	use binius_verifier::protocols::bitand::SKIPPED_VARS;
	use rand::prelude::*;

	use super::*;
	use crate::fold_word::fold_words_with_transform;

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

		let small_field_zerocheck_challenges = PROVER_SMALL_FIELD_ZEROCHECK_CHALLENGES;

		let big_field_zerocheck_challenges =
			vec![
				B128::random(&mut rng);
				log_num_rows - SKIPPED_VARS - small_field_zerocheck_challenges.len()
			];

		let log_num_words = log_num_rows - SKIPPED_VARS;
		let mlv_1 = random_words(log_num_words, &mut rng);
		let mlv_2 = random_words(log_num_words, &mut rng);
		let mlv_3: Vec<Word> = iter::zip(&mlv_1, &mlv_2).map(|(&a, &b)| a & b).collect();

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
			&big_field_zerocheck_challenges,
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
