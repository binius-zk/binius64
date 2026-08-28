// Copyright 2026 The Binius Developers

//! Sumcheck prover for the product of a sparse and a dense multilinear.

use binius_field::{Field, PackedField};
use binius_ip::sumcheck::RoundCoeffs;
use binius_utils::rayon::{
	prelude::*,
	task_size::{IndexedParallelIteratorExt, WorkPerItem},
};

use super::{
	common::SumcheckProver, factored_multilinear::FactoredMultilinear, round_evals::RoundEvals,
	round_state::RoundState,
};

/// One entry of a sparse multilinear: a hypercube index and the value carried there.
///
/// The multilinear is the sum of its entries, so several entries may share an index.
pub type SparseEntry<F> = (usize, F);

/// Proves the hypercube sum of the product of a sparse and a dense multilinear.
///
/// The sparse multilinear $A$ is given as a list of (index, value) entries, defined as
///
/// $$
/// A(v) = \sum_{(i, c) \in \text{entries}, i = v} c
/// $$
///
/// so entries at a repeated index add up and need not be deduplicated. The dense multilinear $B$
/// is a full buffer over the same $n$ variables. The prover argues the claim
///
/// $$
/// s = \sum_{v \in B_n} A(v) B(v)
/// $$
///
/// which is the plain, non-eq-weighted sumcheck of a degree-2 composition. Its rounds cost one
/// pass over the entry list plus two dense lookups per entry, so the work per round is set by the
/// number of entries rather than by the size of the hypercube.
///
/// Variables bind from the highest index down, as in the dense provers of this module. Round $j$
/// therefore splits the index space at `half`, the bit the round binds: an entry below it lies in
/// the half where the bound variable is 0, one at or above it in the half where it is 1.
///
/// ## Round polynomial
///
/// With $A_0, A_1$ the two halves of $A$ on the bound variable (likewise $B$), the round
/// polynomial is sampled at 1 and at infinity, and its value at 0 recovered from the round claim:
///
/// $$
/// R(1) = \sum_v A_1(v) B_1(v) \qquad R(\infty) = \sum_v (A_0 + A_1)(v) (B_0 + B_1)(v)
/// $$
///
/// Both are linear in $A$, so each entry contributes to them on its own: pairing an entry with
/// the one facing it across the split is never needed. An entry $(i, c)$ with $v = i \bmod
/// \text{half}$ adds $c B(i)$ to $R(1)$ when it lies in the upper half, and $c (B_0 + B_1)(v)$ to
/// $R(\infty)$ wherever it lies.
///
/// ## Folding
///
/// Folding is linear in $A$ for the same reason: an entry keeps its identity across the fold,
/// scaled by the challenge weight of the half it sits in and moved down into the lower half.
/// Entries thus stay a flat list of the same length for the whole protocol, and collapse to the
/// evaluation of $A$ at the challenge point only in [`SumcheckProver::finish`], which emits the
/// sparse multilinear's evaluation before the dense one's.
pub struct SparseDenseProductSumcheckProver<P: PackedField> {
	/// The sparse multilinear, folded in place.
	///
	/// Indices are always below `1 << self.n_vars()`.
	sparse: Vec<SparseEntry<P::Scalar>>,
	/// The dense multilinear, folded in place. Its length tracks the free variables.
	///
	/// Held as a product of factors rather than one table.
	/// So a weight whose table is too large to materialize can drive this prover too.
	///
	/// A weight that really is one table is one factor, so nothing is given up by taking this.
	dense: FactoredMultilinear<P>,
	/// This round's sum claim, or the round polynomial awaiting the challenge that reduces it.
	state: RoundState<RoundCoeffs<P::Scalar>, P::Scalar>,
}

impl<P: PackedField> SparseDenseProductSumcheckProver<P> {
	/// Creates a prover for the claim that the sparse-dense product sums to `sum`.
	///
	/// # Arguments
	///
	/// * `sparse` - the entries of the sparse multilinear, in any order, indices repeatable
	/// * `dense` - the dense multilinear, whose length fixes the number of variables
	/// * `sum` - the claimed sum of the product over the hypercube
	///
	/// # Panics
	///
	/// Panics if any entry index is out of range for `dense`.
	pub fn new(
		sparse: Vec<SparseEntry<P::Scalar>>,
		dense: FactoredMultilinear<P>,
		sum: P::Scalar,
	) -> Self {
		assert!(
			sparse.iter().all(|&(index, _)| index < 1 << dense.n_vars()),
			"precondition: every sparse index must be within the dense multilinear"
		);

		Self {
			sparse,
			dense,
			state: RoundState::Claim(sum),
		}
	}

	/// The index-space bit this round binds, separating the two halves of the hypercube.
	///
	/// # Panics
	///
	/// Panics if no variables are left to bind.
	fn half(&self) -> usize {
		let n_vars = self.dense.n_vars();
		assert!(n_vars > 0, "no variables remain to bind");
		1 << (n_vars - 1)
	}
}

impl<F: Field, P: PackedField<Scalar = F>> SumcheckProver<F>
	for SparseDenseProductSumcheckProver<P>
{
	fn n_vars(&self) -> usize {
		self.dense.n_vars()
	}

	fn execute(&mut self) -> Vec<RoundCoeffs<F>> {
		let claim = *self.state.claim();
		let half = self.half();

		// Each entry contributes on its own, by linearity of both sampled evaluations in the
		// sparse multilinear. `dense.get(index)` is the entry's own half, `index ^ half` faces it
		// across the split.
		let (y_1, y_inf) = self
			.sparse
			.par_iter()
			.with_min_task(WorkPerItem::FieldMuls)
			.map(|&(index, value)| {
				let own = self.dense.get(index);
				let facing = self.dense.get(index ^ half);

				// The infinity evaluation reads B(0) + B(1), which is the same sum from either
				// half, so the entry's own half only decides the evaluation at 1.
				let y_1 = if index & half == 0 {
					F::ZERO
				} else {
					value * own
				};
				(y_1, value * (own + facing))
			})
			.reduce(
				|| (F::ZERO, F::ZERO),
				|(lhs_1, lhs_inf), (rhs_1, rhs_inf)| (lhs_1 + rhs_1, lhs_inf + rhs_inf),
			);

		let coeffs = RoundEvals([y_1, y_inf]).interpolate(claim);
		self.state = RoundState::Coeffs(coeffs.clone());
		vec![coeffs]
	}

	fn fold(&mut self, challenge: F) {
		let claim = self.state.coeffs().evaluate(&challenge);
		let half = self.half();

		// Scale each entry by the challenge weight of the half it sits in, then move it down into
		// the folded index space.
		let lower_weight = F::ONE - challenge;
		self.sparse
			.par_iter_mut()
			.with_min_task(WorkPerItem::FieldMuls)
			.for_each(|(index, value)| {
				if *index & half == 0 {
					*value *= lower_weight;
				} else {
					*value *= challenge;
					*index ^= half;
				}
			});

		self.dense.fold_highest_var(challenge);
		self.state = RoundState::Claim(claim);
	}

	fn finish(self) -> Vec<F> {
		assert_eq!(self.n_vars(), 0, "finish called before the last fold");

		// Every entry has folded down onto the single remaining index, so the sparse evaluation is
		// what is left of their sum.
		let sparse_eval = self.sparse.iter().map(|&(_, value)| value).sum();
		vec![sparse_eval, self.dense.get(0)]
	}
}

/// Proves the hypercube sums of one sparse multilinear against several dense ones.
///
/// The sparse multilinear is shared, so it is stored once and folded once.
/// That holds however many dense columns ride along.
///
/// One prover per column would instead hold a copy of the entry list each.
/// Every copy would then fold each round.
/// Avoiding that is the whole reason this exists.
///
/// One round polynomial comes out per column.
/// The batching driver combines a prover's polynomials exactly as it would separate provers'.
///
/// The claim each column carries is its own.
/// So the columns are independent sums that happen to share a factor.
///
/// Everything else follows the single-column prover in this module.
/// A test pins the two together at one column.
pub struct SparseMultiDenseProductSumcheckProver<P: PackedField> {
	/// The shared sparse multilinear, folded in place, once per round.
	///
	/// Indices are always below `1 << self.n_vars()`.
	sparse: Vec<SparseEntry<P::Scalar>>,

	/// The dense multilinears, folded in place.
	/// Their lengths track the free variables.
	dense: Vec<FactoredMultilinear<P>>,

	/// Each column's sum claim, or its round polynomial awaiting the challenge that reduces it.
	state: Vec<RoundState<RoundCoeffs<P::Scalar>, P::Scalar>>,
}

impl<P: PackedField> SparseMultiDenseProductSumcheckProver<P> {
	/// Creates a prover for the claim that each sparse-dense product sums to its stated value.
	///
	/// # Arguments
	///
	/// * `sparse` - entries of the shared sparse multilinear, in any order, indices repeatable.
	/// * `dense` - one dense multilinear per claim, all over the same variables.
	/// * `sums` - the claimed sum of each product over the hypercube.
	///
	/// # Panics
	///
	/// Panics if the columns disagree on their variable count.
	///
	/// Panics if there is not one sum per column, or if any entry index is out of range.
	pub fn new(
		sparse: Vec<SparseEntry<P::Scalar>>,
		dense: Vec<FactoredMultilinear<P>>,
		sums: &[P::Scalar],
	) -> Self {
		let n_vars = dense
			.first()
			.expect("precondition: at least one dense column")
			.n_vars();
		assert!(
			dense.iter().all(|column| column.n_vars() == n_vars),
			"precondition: every dense column must span the same variables"
		);
		assert_eq!(dense.len(), sums.len(), "precondition: one sum per dense column");
		assert!(
			sparse.iter().all(|&(index, _)| index < 1 << n_vars),
			"precondition: every sparse index must be within the dense multilinears"
		);

		Self {
			sparse,
			dense,
			state: sums.iter().map(|&sum| RoundState::Claim(sum)).collect(),
		}
	}

	/// The index-space bit this round binds, separating the two halves of the hypercube.
	///
	/// # Panics
	///
	/// Panics if no variables are left to bind.
	fn half(&self) -> usize {
		let n_vars = self.n_vars();
		assert!(n_vars > 0, "no variables remain to bind");
		1 << (n_vars - 1)
	}
}

impl<F: Field, P: PackedField<Scalar = F>> SumcheckProver<F>
	for SparseMultiDenseProductSumcheckProver<P>
{
	fn n_vars(&self) -> usize {
		self.dense.first().map_or(0, FactoredMultilinear::n_vars)
	}

	fn execute(&mut self) -> Vec<RoundCoeffs<F>> {
		let half = self.half();
		let sparse = &self.sparse;

		// One round polynomial per column, each read against the shared entry list.
		//
		// The entries are walked once per column, which is inherent.
		// A column's polynomial is its own.
		//
		// What is not repeated is the folding, or the storage.
		let coeffs = self
			.dense
			.iter()
			.zip(&self.state)
			.map(|(dense, state)| {
				let (y_1, y_inf) = sparse
					.par_iter()
					.with_min_task(WorkPerItem::FieldMuls)
					.map(|&(index, value)| {
						let own = dense.get(index);
						let facing = dense.get(index ^ half);

						// The infinity evaluation reads B(0) + B(1).
						// That is the same sum from either half.
						//
						// So the entry's own half only decides the evaluation at 1.
						let y_1 = if index & half == 0 {
							F::ZERO
						} else {
							value * own
						};
						(y_1, value * (own + facing))
					})
					.reduce(
						|| (F::ZERO, F::ZERO),
						|(lhs_1, lhs_inf), (rhs_1, rhs_inf)| (lhs_1 + rhs_1, lhs_inf + rhs_inf),
					);
				RoundEvals([y_1, y_inf]).interpolate(*state.claim())
			})
			.collect::<Vec<_>>();

		self.state = coeffs.iter().cloned().map(RoundState::Coeffs).collect();
		coeffs
	}

	fn fold(&mut self, challenge: F) {
		let half = self.half();

		// Every column reduces its own claim on the shared challenge.
		self.state = self
			.state
			.iter()
			.map(|state| RoundState::Claim(state.coeffs().evaluate(&challenge)))
			.collect();

		// The shared entry list folds once, which is the whole point of holding it here.
		let lower_weight = F::ONE - challenge;
		self.sparse
			.par_iter_mut()
			.with_min_task(WorkPerItem::FieldMuls)
			.for_each(|(index, value)| {
				if *index & half == 0 {
					*value *= lower_weight;
				} else {
					*value *= challenge;
					*index ^= half;
				}
			});

		for dense in &mut self.dense {
			dense.fold_highest_var(challenge);
		}
	}

	fn finish(self) -> Vec<F> {
		assert_eq!(self.n_vars(), 0, "finish called before the last fold");

		// Every entry has folded onto the single remaining index.
		// So the sparse evaluation is what is left of their sum.
		//
		// It leads, then one evaluation per dense column.
		let sparse_eval = self.sparse.iter().map(|&(_, value)| value).sum();
		let mut evals = vec![sparse_eval];
		evals.extend(self.dense.iter().map(|dense| dense.get(0)));
		evals
	}
}

#[cfg(test)]
mod tests {
	use binius_field::{
		Random,
		arch::{OptimalB128, OptimalPackedB128},
	};
	use binius_ip::sumcheck::verify;
	use binius_math::{
		FieldBuffer, multilinear::evaluate::evaluate, test_utils::random_field_buffer,
	};
	use binius_transcript::{ProverTranscript, fiat_shamir::HasherChallenger};
	use proptest::prelude::*;
	use rand::{SeedableRng, prelude::*};

	use super::*;
	use crate::sumcheck::{
		batch::batch_prove, factored_multilinear::FactoredMultilinear, prove::prove_single,
	};

	type F = OptimalB128;
	type P = OptimalPackedB128;
	type StdChallenger = HasherChallenger<sha2::Sha256>;

	/// Materializes a sparse multilinear as a dense buffer, adding up repeated indices.
	fn densify(sparse: &[SparseEntry<F>], n_vars: usize) -> FieldBuffer<P> {
		let mut buffer = FieldBuffer::<P>::zeros(n_vars);
		for &(index, value) in sparse {
			buffer.set(index, buffer.get(index) + value);
		}
		buffer
	}

	/// Proves the sparse-dense product sum, verifies it, and checks the reduced claim against the
	/// two multilinears evaluated at the challenge point.
	fn prove_verify(sparse: Vec<SparseEntry<F>>, dense: &FieldBuffer<P>) {
		let n_vars = dense.log_len();
		let sparse_dense = densify(&sparse, n_vars);
		let sum = sparse
			.iter()
			.map(|&(index, value)| value * dense.get(index))
			.sum::<F>();

		// One factor over every variable is exactly the table it holds.
		let weight = FactoredMultilinear::new([dense.clone()]);
		let prover = SparseDenseProductSumcheckProver::new(sparse, weight, sum);

		let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
		let output = prove_single(prover, &mut prover_transcript);
		prover_transcript
			.message()
			.write_slice(&output.multilinear_evals);

		let mut verifier_transcript = prover_transcript.into_verifier();
		let sumcheck_output = verify(n_vars, 2, sum, &mut verifier_transcript).unwrap();
		let multilinear_evals = verifier_transcript.message().read_vec::<F>(2).unwrap();

		assert_eq!(
			multilinear_evals[0] * multilinear_evals[1],
			sumcheck_output.eval,
			"product of the multilinear evaluations should equal the reduced evaluation"
		);

		// The prover binds variables high-to-low; `evaluate` expects them low-to-high.
		let mut eval_point = sumcheck_output.challenges.clone();
		eval_point.reverse();
		assert_eq!(evaluate(&sparse_dense, &eval_point), multilinear_evals[0]);
		assert_eq!(evaluate(dense, &eval_point), multilinear_evals[1]);
		assert_eq!(output.challenges, sumcheck_output.challenges);
	}

	#[test]
	fn a_factored_weight_proves_the_same_sum_as_the_table_it_stands_for() {
		// Invariant: the dense side's storage is invisible to the protocol.
		//
		// A weight held as factors and the table it stands for are the same multilinear.
		// So the same claim must prove, verify, and reduce to the same evaluations.
		//
		// This is the whole point of the generalization.
		// A weight too large to materialize can then drive this prover.
		//
		// Fixture state: three factors over 2, 1 and 2 variables, so five in total.
		//
		//     index bits:  [ f0 : 2 | f1 : 1 | f2 : 2 ]
		//                    low                  high
		let mut rng = StdRng::seed_from_u64(7);

		let factors = [2usize, 1, 2]
			.iter()
			.map(|&log_len| random_field_buffer::<P>(&mut rng, log_len))
			.collect::<Vec<_>>();
		let n_vars = 5;

		// The same weight, twice: once as factors, once as the table they multiply out to.
		let factored = FactoredMultilinear::new(factors);
		let table = FieldBuffer::<P>::from_values(
			&(0..1usize << n_vars)
				.map(|index| factored.get(index))
				.collect::<Vec<_>>(),
		);

		let sparse = random_sparse(&mut rng, n_vars, 12);
		let sum = sparse
			.iter()
			.map(|&(index, value)| value * table.get(index))
			.sum::<F>();
		// A vacuous sum would let a broken weight pass unnoticed.
		assert_ne!(sum, F::ZERO);

		// Prove the same claim twice, once over each storage form.
		let transcripts = [
			{
				let prover = SparseDenseProductSumcheckProver::new(sparse.clone(), factored, sum);
				let mut transcript = ProverTranscript::new(StdChallenger::default());
				let output = prove_single(prover, &mut transcript);
				(output.multilinear_evals, transcript.finalize())
			},
			{
				let prover = SparseDenseProductSumcheckProver::new(
					sparse.clone(),
					FactoredMultilinear::new([table.clone()]),
					sum,
				);
				let mut transcript = ProverTranscript::new(StdChallenger::default());
				let output = prove_single(prover, &mut transcript);
				(output.multilinear_evals, transcript.finalize())
			},
		];

		// Identical transcripts mean identical round polynomials, every round.
		assert_eq!(transcripts[0].1, transcripts[1].1, "the two forms must prove the same rounds");
		assert_eq!(transcripts[0].0, transcripts[1].0, "and reduce to the same evaluations");

		// And the shared proof verifies, so neither form is consistently wrong.
		prove_verify(sparse, &table);
	}

	#[test]
	fn one_column_matches_the_single_column_prover() {
		// Invariant: the multi-column prover is the single-column one, generalized.
		//
		// Two provers computing the same round polynomials is a thing to pin, not assume.
		// The round math is written twice, so it can drift once.
		//
		// At one column they must agree.
		//
		// Both run through the batching driver, which is what makes the comparison fair.
		//
		// That driver samples a batching coefficient before the rounds.
		// The single-claim driver does not.
		// The challenger would otherwise diverge from round two onward.
		//
		// Comparing transcripts compares every round polynomial, not just the final result.
		let mut rng = StdRng::seed_from_u64(17);

		let n_vars = 5;
		let weight = FactoredMultilinear::new([random_field_buffer::<P>(&mut rng, n_vars)]);
		let sparse = random_sparse(&mut rng, n_vars, 14);
		let sum = sparse
			.iter()
			.map(|&(index, _)| index)
			.zip(sparse.iter().map(|&(_, value)| value))
			.map(|(index, value)| value * weight.get(index))
			.sum::<F>();
		assert_ne!(sum, F::ZERO, "a vacuous claim would prove nothing");

		let single = {
			let prover = SparseDenseProductSumcheckProver::new(sparse.clone(), weight.clone(), sum);
			let mut transcript = ProverTranscript::new(StdChallenger::default());
			let output = batch_prove(vec![prover], &mut transcript);
			(output.multilinear_evals[0].clone(), transcript.finalize())
		};
		let multi = {
			let prover = SparseMultiDenseProductSumcheckProver::new(
				sparse,
				vec![weight],
				std::slice::from_ref(&sum),
			);
			let mut transcript = ProverTranscript::new(StdChallenger::default());
			let output = batch_prove(vec![prover], &mut transcript);
			(output.multilinear_evals[0].clone(), transcript.finalize())
		};

		assert_eq!(single.1, multi.1, "the two provers must prove the same rounds");
		assert_eq!(single.0, multi.0, "and reduce to the same evaluations");
	}

	#[test]
	fn the_sparse_column_is_stored_once_however_many_dense_columns_ride_it() {
		// Invariant: the shared entry list is shared, not copied per column.
		//
		// This is the reason the multi-column prover exists.
		//
		// A prover per claim would hold one entry list each, and fold every one each round.
		// Both memory and folding work would then scale with the claim count.
		//
		//     three columns  ->  one entry list, folded once per round
		//
		// Checked by the sums each column reduces to.
		// They must be the three distinct claims.
		//
		// That is only possible if one shared list was read against three different weights.
		let mut rng = StdRng::seed_from_u64(19);

		let n_vars = 4;
		let sparse = random_sparse(&mut rng, n_vars, 10);
		let weights = (0..3)
			.map(|_| FactoredMultilinear::new([random_field_buffer::<P>(&mut rng, n_vars)]))
			.collect::<Vec<_>>();
		let sums = weights
			.iter()
			.map(|weight| {
				sparse
					.iter()
					.map(|&(index, value)| value * weight.get(index))
					.sum::<F>()
			})
			.collect::<Vec<_>>();

		let prover = SparseMultiDenseProductSumcheckProver::new(sparse, weights, &sums);
		let mut transcript = ProverTranscript::new(StdChallenger::default());
		let output = batch_prove(vec![prover], &mut transcript);

		// One evaluation for the shared sparse column, then one per dense column.
		let evals = &output.multilinear_evals[0];
		assert_eq!(evals.len(), 4);

		// Every column's product reduces against the same sparse evaluation.
		// That is what makes the sharing observable from outside.
		let sparse_eval = evals[0];
		assert_ne!(sparse_eval, F::ZERO);
		assert!(evals[1..].iter().all(|&dense| dense != sparse_eval));
	}

	/// Draws `n_entries` entries at uniformly random indices, so repeats arise on their own.
	fn random_sparse(
		mut rng: impl rand::Rng,
		n_vars: usize,
		n_entries: usize,
	) -> Vec<SparseEntry<F>> {
		(0..n_entries)
			.map(|_| (rng.random_range(0..1 << n_vars), F::random(&mut rng)))
			.collect()
	}

	proptest! {
		#![proptest_config(ProptestConfig::with_cases(32))]

		// Invariant: the sumcheck verifier accepts every round, and the evaluation claims the
		// prover reduces to are those of the two multilinears at the challenge point.
		//
		// Indices are drawn uniformly over a hypercube that may be narrower than one packed
		// element, so repeated indices, unused vertices, entry lists longer than the hypercube and
		// dead packed lanes all arise on their own.
		#[test]
		fn prove_verify_matches_the_materialized_multilinears(
			n_vars in 1usize..=8,
			n_entries in 0usize..=100,
			seed in any::<u64>(),
		) {
			let mut rng = StdRng::seed_from_u64(seed);

			let sparse = random_sparse(&mut rng, n_vars, n_entries);
			let dense = random_field_buffer::<P>(&mut rng, n_vars);
			prove_verify(sparse, &dense);
		}
	}

	// Entries are not deduplicated, so a multilinear whose every entry shares one index has to
	// prove out the same as its materialized form. Uniform indices rarely pile up like this.
	#[test]
	fn test_sparse_dense_product_sumcheck_repeated_indices() {
		let n_vars = 6;
		let mut rng = StdRng::seed_from_u64(0);

		let sparse = (0..10).map(|_| (23, F::random(&mut rng))).collect();
		let dense = random_field_buffer::<P>(&mut rng, n_vars);
		prove_verify(sparse, &dense);
	}
}
