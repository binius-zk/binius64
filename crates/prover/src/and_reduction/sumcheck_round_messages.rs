// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use std::{array, borrow::Cow, iter};

use binius_core::word::Word;
use binius_field::{
	AESTowerField8b as B8, BinaryField, BinaryField1b as B1, ExtensionField, Field,
	PackedAESBinaryField64x8b as Packed64xB8, PackedField, WideMul, util::expand_subset_sums_array,
};
use binius_math::{BinarySubspace, multilinear::eq::eq_ind_partial_eval};
use binius_utils::rayon::{self, iter::Either, prelude::*};
use binius_verifier::{
	config::PROVER_SMALL_FIELD_ZEROCHECK_CHALLENGES, protocols::bitand::ROWS_PER_HYPERCUBE_VERTEX,
};
use itertools::izip;

use super::ntt_lookup::NTTLookup;

/// Number of big field zerocheck challenges whose equality indicator is expanded per window.
const N_FIXED_LARGE_CHALLENGES: usize = 4;

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
/// The product constraint column C is not an input.
/// Each C word is derived in registers as the AND of the matching A and B words.
/// A satisfying witness makes that derivation exact on every row.
/// So no third column is ever built or streamed.
///
/// # Arguments
///
/// * `log_words` - Base-2 logarithm of the constraint axis's length
/// * `a_words` - First multiplicand (a) as a one-bit oblong multilinear polynomial
/// * `b_words` - Second multiplicand (b) as a one-bit oblong multilinear polynomial
/// * `eq_ind_big_field_challenges` - Partial equality indicator evaluations for big field variables
/// * `prover_message_domain` - The NTT domain subspace (dimension `SKIPPED_VARS + 1`) from which
///   the low-degree-extension lookup table is built internally
///
/// # Preconditions
///
/// * The two columns have equal length, at most `1 << log_words`. They need not fill the constraint
///   axis: a shorter column has its remaining rows read as zero. Such a row forces the derived `C =
///   A & B` to zero as well, so `A * B - C` vanishes on it at every point of the univariate domain
///   and it adds nothing to the message.
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
pub fn univariate_round_message_extension_domain<F>(
	log_words: usize,
	a_words: &[Word],
	b_words: &[Word],
	big_field_challenges: &[F],
	prover_message_domain: &BinarySubspace<B8>,
) -> [F; ROWS_PER_HYPERCUBE_VERTEX]
where
	F: BinaryField + From<B8>,
{
	const N_FIXED_SMALL_CHALLENGES: usize = PROVER_SMALL_FIELD_ZEROCHECK_CHALLENGES.len();

	const LOG_CHUNK_SIZE: usize = N_FIXED_SMALL_CHALLENGES + N_FIXED_LARGE_CHALLENGES;

	assert_eq!(big_field_challenges.len(), log_words.saturating_sub(N_FIXED_SMALL_CHALLENGES));
	assert_eq!(a_words.len(), b_words.len());
	assert!(a_words.len() <= 1 << log_words);

	let ntt_lookup = tracing::debug_span!("Compute univariate LDE table")
		.in_scope(|| NTTLookup::new(prover_message_domain));

	let eq_ind_small: [_; 1 << N_FIXED_SMALL_CHALLENGES] =
		eq_ind_partial_eval::<B8>(&PROVER_SMALL_FIELD_ZEROCHECK_CHALLENGES)
			.iter_scalars()
			.map(Packed64xB8::broadcast)
			.collect::<Vec<_>>()
			.try_into()
			.expect("PROVER_SMALL_FIELD_ZEROCHECK_CHALLENGES.len() == N_FIXED_SMALL_CHALLENGES");

	let (eq_ind_fixed_large, extra_challenges) = eq_ind_fixed_large(big_field_challenges);
	let outer_weight_mul_maps = eq_ind_fixed_large.map(B8ToExtMulMap::new);
	let eq_ind_extra = eq_ind_partial_eval::<F>(extra_challenges);

	const CHUNK_SIZE: usize = 1 << LOG_CHUNK_SIZE;

	let a_chunks_iter = padded_chunks::<CHUNK_SIZE>(a_words);
	let b_chunks_iter = padded_chunks::<CHUNK_SIZE>(b_words);

	(a_chunks_iter, b_chunks_iter)
		.into_par_iter()
		.map(|(a_chunk, b_chunk)| {
			// Reshape the chunk arrays into arrays of arrays
			let [a_subchunks, b_subchunks] = [&a_chunk, &b_chunk].map(|chunk| {
				bytemuck::must_cast_ref::<
					[Word; CHUNK_SIZE],
					[[Word; 1 << N_FIXED_SMALL_CHALLENGES]; 1 << N_FIXED_LARGE_CHALLENGES],
				>(chunk)
			});

			let mut acc = [F::ZERO; ROWS_PER_HYPERCUBE_VERTEX];
			for (a_subchunk, b_subchunk, outer_weight) in
				izip!(a_subchunks, b_subchunks, &outer_weight_mul_maps)
			{
				let mut summed_ntt = <Packed64xB8 as WideMul>::Output::default();
				for (&a_i, &b_i, inner_weight) in izip!(a_subchunk, b_subchunk, &eq_ind_small) {
					let c_i = a_i & b_i;

					// Compute the low-degree extension of each column via the lookup table.
					let a_lde = ntt_lookup.ntt(a_i);
					let b_lde = ntt_lookup.ntt(b_i);
					let c_lde = ntt_lookup.ntt(c_i);

					// Compute the weighted composition of the LDE values.
					summed_ntt += Packed64xB8::wide_mul(a_lde * b_lde - c_lde, *inner_weight);
				}

				let summed_ntt_reduced = Packed64xB8::reduce(summed_ntt);
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

/// The equality indicator expansion of the first `N_FIXED_LARGE_CHALLENGES` big field zerocheck
/// challenges, along with the challenges past them.
///
/// The expansion weights the windows' subchunks, while the extra challenges weight the windows
/// themselves. A challenge vector shorter than `N_FIXED_LARGE_CHALLENGES` is zero-extended to that
/// length, which leaves no extra challenges: a column that short occupies a single window, whose
/// unused index bits are the ones the zero challenges cover.
fn eq_ind_fixed_large<F: Field>(
	big_field_challenges: &[F],
) -> ([F; 1 << N_FIXED_LARGE_CHALLENGES], &[F]) {
	if big_field_challenges.len() < N_FIXED_LARGE_CHALLENGES {
		let eq_ind_fixed_large = eq_ind_partial_eval::<F>(big_field_challenges);
		let mut eq_ind_fixed_large_padded = [F::ZERO; 1 << N_FIXED_LARGE_CHALLENGES];
		eq_ind_fixed_large_padded[..eq_ind_fixed_large.len()]
			.copy_from_slice(eq_ind_fixed_large.as_ref());

		(eq_ind_fixed_large_padded, &[][..])
	} else {
		let (fixed_large_challenges, extra_challenges) =
			big_field_challenges.split_at(N_FIXED_LARGE_CHALLENGES);
		let fixed_large_challenges: [_; N_FIXED_LARGE_CHALLENGES] = fixed_large_challenges
			.try_into()
			.expect("big_field_challenges.len() >= N_FIXED_LARGE_CHALLENGES");

		let eq_ind_fixed_large: [_; 1 << N_FIXED_LARGE_CHALLENGES] =
			eq_ind_partial_eval::<F>(&fixed_large_challenges)
				.as_ref()
				.try_into()
				.expect("fixed_large_challenges.len() == N_FIXED_LARGE_CHALLENGES");

		(eq_ind_fixed_large, extra_challenges)
	}
}

/// The words as chunks of `CHUNK_SIZE`, the last one padded out if the words run out partway
/// through it.
///
/// A chunk the words fill is borrowed in place; only the partial one is copied. The copy zero-
/// extends the words to the constraint axis's length, then repeats that axis across the rest of the
/// chunk. A zero row forces the derived `C = A & B` to zero as well, so `A * B - C` vanishes on it
/// at every point of the univariate domain and it adds nothing to the round message.
///
/// Repetition is a no-op unless the whole axis is shorter than one chunk. In that case the chunk's
/// index bits past the axis carry the fixed small zerocheck challenges, which are non-zero: summing
/// a non-zero eq challenge over a duplicated coordinate gives back exactly one copy, whereas
/// leaving those slots zero would scale the axis by `1 + r`. Repetition is what keeps such a
/// column's round message equal to the message the verifier reconstructs over the axis's own
/// variables.
///
/// # Preconditions
///
/// * `CHUNK_SIZE` is a power of two
/// * The constraint axis is `words.len()` rounded up to a power of two
fn padded_chunks<const CHUNK_SIZE: usize>(
	words: &[Word],
) -> impl IndexedParallelIterator<Item = Cow<'_, [Word; CHUNK_SIZE]>> {
	let chunks_iter = words.par_chunks_exact(CHUNK_SIZE);
	let tail = chunks_iter.remainder();

	let chunks_iter = chunks_iter.map(|chunk| {
		Cow::Borrowed(
			<&[Word; CHUNK_SIZE]>::try_from(chunk)
				.expect("chunks_exact produces slices with len CHUNK_SIZE"),
		)
	});

	if tail.is_empty() {
		Either::Right(chunks_iter)
	} else {
		// The axis's rows within one chunk. Both are powers of two, so this divides `CHUNK_SIZE`.
		let axis_rows = words.len().next_power_of_two().min(CHUNK_SIZE);

		let mut tail_padded = [Word::ZERO; CHUNK_SIZE];
		tail_padded[..tail.len()].copy_from_slice(tail);

		let (axis, rest) = tail_padded.split_at_mut(axis_rows);
		for copy in rest.chunks_exact_mut(axis_rows) {
			copy.copy_from_slice(axis);
		}

		Either::Left(chunks_iter.chain(rayon::iter::once(Cow::Owned(tail_padded))))
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
	const fn call(&self, input: B8) -> F {
		self.lookup[input.val() as usize]
	}
}

#[cfg(test)]
mod test {
	use std::iter::repeat_with;

	use binius_compute::GlobalAllocator;
	use binius_field::{BinaryField128bGhash as B128, Field, Random};
	use binius_math::{
		BinarySubspace, FieldBuffer,
		univariate::{extrapolate_over_subspace, lagrange_evals_scalars},
	};
	use binius_utils::checked_arithmetics::log2_ceil_usize;
	use binius_verifier::protocols::bitand::SKIPPED_VARS;
	use rand::prelude::*;

	use super::*;
	use crate::fold_word::BitAxisFolder;

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
		// Fixed seed keeps the random witness reproducible across runs.
		let mut rng = StdRng::from_seed([0; 32]);

		// 2^10 rows total; each 64-bit word packs 2^SKIPPED_VARS rows, leaving this many words.
		let log_num_words = 10 - SKIPPED_VARS;

		// Every word is random and non-zero, so no window is skipped.
		// This pins the dense path, where the skip must leave the result unchanged.
		let mlv_1 = random_words(log_num_words, &mut rng);
		let mlv_2 = random_words(log_num_words, &mut rng);

		assert_round_message_consistent(&mlv_1, &mlv_2, &mut rng);
	}

	/// The width in words of one round-message window: `2^(3 + 4) = 128`.
	const WINDOW: usize = 1 << (PROVER_SMALL_FIELD_ZEROCHECK_CHALLENGES.len() + 4);

	/// The column lengths covering every windowing regime.
	fn windowing_shapes() -> [usize; 6] {
		[
			// A single-row axis: one window, filled by repetition.
			1,
			// A sub-window axis with a real zero tail inside it, then repeated.
			3,
			// Exactly one window, no padding at all.
			WINDOW,
			// One whole window plus a straddling one, padded up to two.
			WINDOW + 1,
			// A whole number of windows, padded up to four.
			3 * WINDOW,
			// Whole windows plus a straddling one, padded up to four.
			3 * WINDOW + 17,
		]
	}

	// An unpadded column's round message agrees with the verifier's own fold of that column.
	//
	// The padded-vs-unpadded equality above would also hold if both were wrong in the same way.
	// This pins the message to the independently folded claim at every windowing shape.
	#[test]
	fn test_first_round_message_with_unpadded_columns() {
		let mut rng = StdRng::from_seed([3; 32]);

		for n_words in windowing_shapes() {
			let [a, b] = array::from_fn(|_| {
				repeat_with(|| Word(rng.random::<u64>()))
					.take(n_words)
					.collect::<Vec<_>>()
			});
			assert_round_message_consistent(&a, &b, &mut rng);
		}
	}

	/// Asserts the first-round univariate message agrees with the next-round sum claim.
	///
	/// The check mirrors the verifier at one random challenge:
	/// - Extrapolate the round message at the challenge to get the expected next-round sum.
	/// - Fold A, B, and C = A & B at the same challenge, then form the sum claim directly.
	/// - The two values must be equal.
	///
	/// The columns need not have a power-of-two length; the constraint axis is then the next power
	/// of two, and both sides read the rows past the columns' end as zero.
	fn assert_round_message_consistent(mlv_1: &[Word], mlv_2: &[Word], mut rng: impl Rng) {
		assert_eq!(mlv_1.len(), mlv_2.len());
		let log_num_words = log2_ceil_usize(mlv_1.len());

		// The prover pins only as many small-field challenges as the axis has coordinates, so an
		// axis shorter than the fixed set uses a prefix of it.
		let small_field_zerocheck_challenges = &PROVER_SMALL_FIELD_ZEROCHECK_CHALLENGES
			[..log_num_words.min(PROVER_SMALL_FIELD_ZEROCHECK_CHALLENGES.len())];

		let big_field_zerocheck_challenges =
			vec![
				B128::random(&mut rng);
				log_num_words.saturating_sub(PROVER_SMALL_FIELD_ZEROCHECK_CHALLENGES.len())
			];

		// The round message derives C = A & B internally.
		// This materialized copy feeds only the verifier-side transparent fold below.
		let mlv_3: Vec<Word> = iter::zip(mlv_1, mlv_2).map(|(&a, &b)| a & b).collect();

		// Agreed-upon proof parameter

		let prover_message_domain = BinarySubspace::with_dim(SKIPPED_VARS + 1);

		let verifier_message_domain = prover_message_domain.isomorphic::<B128>();

		// Prover generates first round message
		let first_round_message_on_ext_domain = univariate_round_message_extension_domain::<B128>(
			log_num_words,
			mlv_1,
			mlv_2,
			&big_field_zerocheck_challenges,
			&prover_message_domain,
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
			&first_sumcheck_challenge,
		);

		let lagrange_evals =
			lagrange_evals_scalars(&verifier_input_domain, &first_sumcheck_challenge);
		let folder = BitAxisFolder::new(&lagrange_evals);

		let folded_first_mle: FieldBuffer<B128> = folder.fold(&GlobalAllocator, mlv_1);
		let folded_second_mle: FieldBuffer<B128> = folder.fold(&GlobalAllocator, mlv_2);
		let folded_third_mle: FieldBuffer<B128> = folder.fold(&GlobalAllocator, &mlv_3);

		let upcasted_small_field_challenges: Vec<_> = small_field_zerocheck_challenges
			.iter()
			.copied()
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
