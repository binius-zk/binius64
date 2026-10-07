// Copyright 2026 The Binius Developers

use std::{array, iter};

use binius_compute::Allocator;
use binius_field::{BinaryField, PackedField, WideMul};
use binius_ip::sumcheck::RoundCoeffs;
use binius_ip_prover::sumcheck::{
	common::MleCheckProver, eq_tracker::ChunkedEqTracker, round_evals::RoundEvals,
	round_state::RoundState,
};
use binius_math::FieldBuffer;
use binius_utils::rayon::prelude::*;
use itertools::izip;

use super::switchover::BinarySwitchover;

/// An [`MleCheckProver`] for the composition `a · b − c` of three one-bit multilinears.
///
/// Each multilinear is one bit column of a word list, held by a [`BinarySwitchover`], so the
/// columns are never embedded as field buffers over the whole hypercube.
pub struct BitColumnMlecheckProver<'alloc, P: PackedField, A: Allocator> {
	n_vars: usize,
	eval_point: Vec<P::Scalar>,
	last_coeffs_or_sum: RoundState<RoundCoeffs<P::Scalar>, P::Scalar>,
	eq_tracker: ChunkedEqTracker<P>,
	columns: [BinarySwitchover<'alloc, P, A>; 3],
}

impl<'alloc, F: BinaryField, P: PackedField<Scalar = F>, A: Allocator>
	BitColumnMlecheckProver<'alloc, P, A>
{
	/// Constructs a prover of the claim that `a · b − c` evaluates to `eval_claim` at
	/// `eval_point`, with `columns` being `[a, b, c]`.
	///
	/// ## Preconditions
	///
	/// * every column is a single selector over `eval_point.len()` variables
	pub fn new(
		columns: [BinarySwitchover<'alloc, P, A>; 3],
		eval_point: Vec<F>,
		eval_claim: F,
	) -> Self {
		const MAX_CHUNK_VARS: usize = 8;
		Self {
			n_vars: eval_point.len(),
			eq_tracker: ChunkedEqTracker::new(MAX_CHUNK_VARS, &eval_point),
			eval_point,
			last_coeffs_or_sum: RoundState::Claim(eval_claim),
			columns,
		}
	}
}

impl<F, P, A> MleCheckProver<F> for BitColumnMlecheckProver<'_, P, A>
where
	F: BinaryField,
	P: PackedField<Scalar = F>,
	A: Allocator,
{
	fn n_vars(&self) -> usize {
		self.n_vars
	}

	fn execute(&mut self) -> Vec<RoundCoeffs<F>> {
		let &sum = self.last_coeffs_or_sum.claim();

		assert!(self.n_vars > 0);

		// Chunked like `SelectorMlecheckProver`: each chunk fills small scratch buffers with the
		// columns' halves and reads the eq chunk while it stays in cache.
		let chunk_vars = self.eq_tracker.chunk().log_len();
		let chunk_count = 1 << (self.n_vars - 1 - chunk_vars);
		let selectors = self.columns.each_ref().map(|column| column.selectors());
		let is_binary = selectors[0].is_binary();

		let packed_prime_evals = (0..chunk_count)
			.into_par_iter()
			.fold(
				|| {
					let scratch =
						array::from_fn::<_, 6, _>(|_| FieldBuffer::<P>::zeros(chunk_vars));
					let masks = array::from_fn::<_, 6, _>(|_| {
						(0..scratch[0].as_ref().len())
							.map(|_| P::make_mask(iter::empty()))
							.collect::<Vec<_>>()
					});
					(RoundEvals::<P, 2>::default(), scratch, masks)
				},
				|(mut round_evals, mut scratch, mut masks), chunk_index| {
					let eq_chunk = self.eq_tracker.chunk().as_ref();
					// a * b - c
					// @one: a_1 * b_1 - c_1
					// @inf: (a_0 + a_1) * (b_0 + b_1) (lower degree terms are dropped)
					let chunk_round_evals = if is_binary {
						// Every value is a bit, so weighting by eq is a lane select and the round
						// needs no multiplication at all.
						let [a_1, a_inf, b_1, b_inf, c_1, c_inf] = &mut masks;
						for (selectors, mask_1, mask_inf) in
							izip!(&selectors, [a_1, b_1, c_1], [a_inf, b_inf, c_inf])
						{
							selectors.fill_bit_masks(
								0,
								chunk_vars,
								chunk_index,
								[mask_1, mask_inf],
							);
						}
						let [a_1, a_inf, b_1, b_inf, c_1, _] = &masks;
						let (y_1, y_inf) = izip!(eq_chunk, a_1, a_inf, b_1, b_inf, c_1).fold(
							(P::zero(), P::zero()),
							|(y_1, y_inf), (eq_i, a_1_i, a_inf_i, b_1_i, b_inf_i, c_1_i)| {
								(
									y_1 + eq_i.select(a_1_i).select(b_1_i) + eq_i.select(c_1_i),
									y_inf + eq_i.select(a_inf_i).select(b_inf_i),
								)
							},
						);
						RoundEvals([y_1, y_inf])
					} else {
						let [a_0, a_1, b_0, b_1, c_0, c_1] = &mut scratch;
						for (selectors, half_0, half_1) in
							izip!(&selectors, [a_0, b_0, c_0], [a_1, b_1, c_1])
						{
							selectors.fill_halves(
								0,
								chunk_vars,
								chunk_index,
								[half_0.as_mut_view(), half_1.as_mut_view()],
							);
						}
						let [a_0, a_1, b_0, b_1, _, c_1] = &scratch;
						let mut wide_y_1 = <P as WideMul>::Output::default();
						let mut wide_y_inf = <P as WideMul>::Output::default();
						for (&eq_i, &a_0_i, &a_1_i, &b_0_i, &b_1_i, &c_1_i) in izip!(
							eq_chunk,
							a_0.as_ref(),
							a_1.as_ref(),
							b_0.as_ref(),
							b_1.as_ref(),
							c_1.as_ref(),
						) {
							wide_y_1 += P::wide_mul(eq_i, a_1_i * b_1_i - c_1_i);
							wide_y_inf += P::wide_mul(eq_i, (a_0_i + a_1_i) * (b_0_i + b_1_i));
						}
						RoundEvals([wide_y_1, wide_y_inf]).reduce::<P>()
					};
					round_evals += &(chunk_round_evals * self.eq_tracker.suffix().get(chunk_index));

					(round_evals, scratch, masks)
				},
			)
			.map(|(round_evals, ..)| round_evals)
			.reduce(RoundEvals::default, |lhs, rhs| lhs + &rhs);

		let (prime_coeffs, _) = self
			.eq_tracker
			.interpolate2(sum, packed_prime_evals.sum_scalars(self.n_vars - 1));
		self.last_coeffs_or_sum = RoundState::Coeffs(prime_coeffs.clone());
		vec![prime_coeffs]
	}

	fn fold(&mut self, challenge: F) {
		let sum = self.last_coeffs_or_sum.coeffs().evaluate(&challenge);

		assert!(self.n_vars > 0);

		self.eq_tracker.fold(challenge);
		for column in &mut self.columns {
			column.fold(challenge);
		}
		self.n_vars -= 1;

		self.last_coeffs_or_sum = RoundState::Claim(sum);
	}

	fn finish(self) -> Vec<F> {
		assert_eq!(self.n_vars, 0, "finish called out of order; sumcheck rounds remain");

		self.columns
			.into_iter()
			.flat_map(|column| column.finish())
			.collect()
	}

	fn eval_point(&self) -> &[F] {
		&self.eval_point
	}
}

#[cfg(test)]
mod tests {
	use binius_compute::GlobalAllocator;
	use binius_core::word::Word;
	use binius_field::{Field, FieldOps};
	use binius_ip_prover::sumcheck::{prove_single_mlecheck, quadratic_mlecheck_prover};
	use binius_math::test_utils::{Packed128b, random_scalars};
	use binius_transcript::{ProverTranscript, fiat_shamir::HasherChallenger};
	use itertools::Itertools;
	use rand::prelude::*;
	use rstest::rstest;

	use super::*;

	type P = Packed128b;
	type F = <P as FieldOps>::Scalar;
	type StdChallenger = HasherChallenger<sha2::Sha256>;

	// The prover must write the same transcript as the quadratic mlecheck prover over the embedded
	// bit columns, and reduce to the same evaluations. The word lists stop short of `2^n_vars`, so
	// the missing words read as zero.
	#[rstest]
	#[case::one_var(1)]
	#[case::below_one_block(4)]
	#[case::exactly_one_block(6)]
	#[case::several_blocks(11)]
	fn test_matches_embedded_columns(#[case] n_vars: usize) {
		let mut rng = StdRng::seed_from_u64(0);
		let n_words = (1 << n_vars) - 1;
		let word_lists = array::from_fn::<_, 3, _>(|_| {
			(0..n_words)
				.map(|_| Word::from_u64(rng.random()))
				.collect_vec()
		});
		let eval_point = random_scalars::<F>(&mut rng, n_vars);
		let eval_claim = random_scalars::<F>(&mut rng, 1)[0];

		let embedded = word_lists.each_ref().map(|words| {
			(0..1 << n_vars)
				.map(|i| {
					let bit = words.get(i).is_some_and(|word| word.extract_bit(0));
					if bit { F::ONE } else { F::ZERO }
				})
				.collect::<FieldBuffer<P>>()
		});
		let expected_prover = quadratic_mlecheck_prover(
			&GlobalAllocator,
			embedded,
			|[a, b, c]| a * b - c,
			|[a, b, _c]| a * b,
			eval_point.clone(),
			eval_claim,
		);
		let mut expected_transcript = ProverTranscript::new(StdChallenger::default());
		let expected_output = prove_single_mlecheck(expected_prover, &mut expected_transcript);

		let columns = word_lists
			.each_ref()
			.map(|words| BinarySwitchover::new_bit(&GlobalAllocator, words, 0, n_vars));
		let prover = BitColumnMlecheckProver::<P, _>::new(columns, eval_point, eval_claim);
		let mut transcript = ProverTranscript::new(StdChallenger::default());
		let output = prove_single_mlecheck(prover, &mut transcript);

		assert_eq!(output.multilinear_evals, expected_output.multilinear_evals);
		assert_eq!(transcript.finalize(), expected_transcript.finalize());
	}
}
